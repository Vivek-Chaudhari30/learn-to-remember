"""Measure peak GPU memory and time of one training-style log-prob pass
(forward + backward through LoRA, gradient checkpointing on) on a long row.
This checks that the M3 logit chunking resolves the open item in DECISIONS.md
(the full [B, L, V] logits at 4k context are too big for a T4).

    python scripts/logprob_memory.py --model Qwen/Qwen2.5-1.5B-Instruct --dtype fp16 \
        --lengths 2048,4096 --gen-tokens 600 --out results/memory

Modes:
  full     the M0 path: full [B, L, V] logits, then gather/logsumexp
  chunked  sequence_logprobs over all positions, 1024 per slice
  masked   sequence_logprobs only at loss-mask positions (what training uses)
A mode that runs out of memory is recorded as "OOM", not skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from learn_to_remember.grpo.core import position_ids_from_mask, sequence_logprobs  # noqa: E402
from learn_to_remember.model_utils import load_config, load_policy_for_training  # noqa: E402


def full_logprobs(model, ids, attn):
    logits = model(input_ids=ids, attention_mask=attn,
                   position_ids=position_ids_from_mask(attn)).logits[:, :-1].float()
    sel = logits.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return F.pad(sel - torch.logsumexp(logits, -1), (1, 0), value=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--lengths", default="2048,4096")
    ap.add_argument("--gen-tokens", type=int, default=600,
                    help="loss-mask tokens per row (spread over 6 turns)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--modes", default="masked,chunked,full")
    ap.add_argument("--out", default="results/memory")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "GPU only"
    cfg = load_config(os.path.join(ROOT, "configs", "base.yaml"))
    model, tok, dev = load_policy_for_training(a.model, lora_cfg=cfg["train"]["lora"],
                                               dtype_name=a.dtype, grad_checkpoint=True)
    model.train()
    out = []
    for L in [int(x) for x in a.lengths.split(",")]:
        g = torch.Generator().manual_seed(0)
        ids = torch.randint(0, len(tok), (a.batch, L), generator=g).to(dev)
        attn = torch.ones_like(ids)
        mask = torch.zeros(a.batch, L, device=dev)
        per = a.gen_tokens // 6
        for k in range(6):                                  # 6 turns spread over the row
            s = int(L * (0.2 + 0.13 * k))
            mask[:, s:s + per] = 1
        for mode in a.modes.split(","):
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            t0 = time.perf_counter()
            try:
                if mode == "full":
                    lp = full_logprobs(model, ids, attn)
                else:
                    lp = sequence_logprobs(model, ids, attn,
                                           logit_mask=mask if mode == "masked" else None)
                loss = -(lp * mask).sum() / mask.sum()
                loss.backward()
                torch.cuda.synchronize()
                rec = {"length": L, "mode": mode, "status": "ok",
                       "seconds": round(time.perf_counter() - t0, 3),
                       "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                       "peak_over_weights_gb": round(
                           (torch.cuda.max_memory_allocated() - base) / 2**30, 3)}
            except torch.cuda.OutOfMemoryError:
                rec = {"length": L, "mode": mode, "status": "OOM"}
            finally:
                lp = loss = None
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            rec.update(batch=a.batch, gen_tokens=a.gen_tokens, model=a.model, dtype=a.dtype,
                       gpu=torch.cuda.get_device_name(dev))
            print(rec, flush=True)
            out.append(rec)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "logprob_memory.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
