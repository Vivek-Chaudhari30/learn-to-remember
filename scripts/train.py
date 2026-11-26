"""Train a LoRA policy with the hand-written multi-turn GRPO loop (M4).

    # overfit sanity check (8 fixed questions)
    python scripts/train.py --config configs/overfit.yaml --model Qwen/Qwen2.5-1.5B-Instruct \
        --dtype fp16 --out results/train/overfit --save-dir checkpoints/overfit

    # measure seconds per step on the main config (no checkpoint)
    python scripts/train.py --config configs/train_notes.yaml --model ... --steps 3 \
        --timing --out results/train/timing

    # main run (experiment 2), resumable
    python scripts/train.py --config configs/train_notes.yaml --model ... --dtype fp16 \
        --out results/train/notes_l0.2 --save-dir checkpoints/notes_l0.2 --resume

Writes <out>/train_log.json (one entry per step), <out>/run.json (config,
environment, timing), and <out>/rollouts/train_samples.jsonl (best and worst
trajectory per step; gitignored).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_notes.yaml")
    ap.add_argument("--model", default=None, help="overrides model.id")
    ap.add_argument("--dtype", default=None, help="overrides model.dtype (fp16 on a T4)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--gen-batch", type=int, default=None)
    ap.add_argument("--prompts-per-step", type=int, default=None)
    ap.add_argument("--group-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-gen-tokens", type=int, default=None, help="plumbing smoke only")
    ap.add_argument("--timing", action="store_true",
                    help="measure s/step only: no checkpoint, writes <out>/timing.json")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--save-dir", default="checkpoints/run")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch

    from learn_to_remember.env.episode import EnvConfig
    from learn_to_remember.env.tokens import HFTokenCounter
    from learn_to_remember.eval import make_episodes
    from learn_to_remember.eval_runner import env_meta
    from learn_to_remember.grpo.rollout import HFGenerator, chat_format_for
    from learn_to_remember.grpo.train import QuestionSampler, make_scaler, train_loop
    from learn_to_remember.model_utils import (
        load_config, load_policy_for_training, maybe_resume, set_seed,
    )
    from learn_to_remember.workspace.generate import generate_company

    cfg = load_config(a.config if os.path.isabs(a.config) else os.path.join(ROOT, a.config))
    t, r = cfg["train"], cfg["rollout"]
    if a.model:
        cfg["model"]["id"] = a.model
    if a.dtype:
        cfg["model"]["dtype"] = a.dtype
    for k, v in (("micro_batch", a.micro_batch), ("gen_batch", a.gen_batch),
                 ("prompts_per_step", a.prompts_per_step), ("lr", a.lr), ("steps", a.steps)):
        if v is not None:
            t[k] = v
    if a.group_size:
        r["group_size"] = a.group_size
    if a.max_gen_tokens:
        cfg["env"]["max_gen_tokens_per_turn"] = a.max_gen_tokens
    if a.timing:
        t["checkpoint_every"] = 0
    model_id = cfg["model"]["id"]
    env_cfg = EnvConfig.from_dict(cfg["env"], cfg["reward"])

    set_seed(t["seed"])
    policy, tok, device = load_policy_for_training(
        model_id, lora_cfg=t["lora"], dtype_name=cfg["model"]["dtype"],
        grad_checkpoint=t.get("grad_checkpoint", False))
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in policy.parameters())
    lora_dtypes = sorted({str(p.dtype) for p in policy.parameters() if p.requires_grad})
    print(f"model {model_id} on {device} | LoRA {n_train:,}/{n_all:,} params, {lora_dtypes}")

    fmt = chat_format_for(model_id, tok)
    gen = HFGenerator(policy, tok, stop_ids=fmt.stop_ids, max_new_tokens=env_cfg.max_gen_tokens,
                      greedy=False, temperature=r["temperature"], top_p=r["top_p"],
                      batch_size=t["gen_batch"])
    counter = HFTokenCounter(tok)
    ws = generate_company(cfg["workspace"]["seed_a"], "company_a", with_train=True)
    train_qs = ws.splits["train"]
    cond = t["condition"]
    version = "v3" if cond == "no_notes" else cond
    sampler = QuestionSampler(train_qs, per_step=t["prompts_per_step"], version=version,
                              mix=t["question_mix"], seed=t["seed"], overfit_n=t["overfit_n"])

    def episodes(qs):
        return make_episodes(ws, qs, cond, env_cfg, counter)

    optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad),
                                  lr=t["lr"])
    start, history = (maybe_resume(a.save_dir, policy, optimizer, device)
                      if a.resume else (0, []))
    scaler = make_scaler(policy)
    os.makedirs(a.out, exist_ok=True)
    run = {"config": cfg, "args": vars(a), "meta": env_meta(policy, tok, model_id, None,
                                                            cfg["model"]["dtype"]),
           "n_train_questions": len(train_qs), "lora_trainable": n_train,
           "lora_dtypes": lora_dtypes, "grad_scaler": scaler is not None,
           "overfit_ids": [q["id"] for q in sampler.questions] if t["overfit_n"] else None}
    with open(os.path.join(a.out, "run.json"), "w") as f:
        json.dump(run, f, indent=1, default=str)
    print(f"condition {cond} | mix {sampler.mix} | P={sampler.per_step} G={r['group_size']} "
          f"| lr {t['lr']} | kl {t['kl_coef']} | adv std-norm {t.get('adv_std_norm', True)} "
          f"| lambda {env_cfg.cost_lambda} "
          f"| steps {start}..{t['steps'] - 1} | scaler {scaler is not None}")

    # --timing: the loop's end-of-run checkpoint goes to a throwaway dir, deleted below,
    # so no weights land in results/.
    tmp_ckpt = tempfile.mkdtemp(prefix="ltr_timing_") if a.timing else None
    t0 = time.perf_counter()
    history = train_loop(policy=policy, tok=tok, fmt=fmt, generate=gen,
                         make_episodes=episodes, sampler=sampler, optimizer=optimizer, tcfg=t,
                         group_size=r["group_size"], steps=t["steps"],
                         save_dir=a.save_dir if not a.timing else tmp_ckpt,
                         log_path=os.path.join(a.out, "train_log.json"), start_step=start,
                         history=history, sample_dir=os.path.join(a.out, "rollouts"),
                         scaler=scaler, log=lambda s: print(s, flush=True))
    if tmp_ckpt:
        shutil.rmtree(tmp_ckpt, ignore_errors=True)
    secs = [h["sec_step"] for h in history]
    timing = {"steps_measured": len(secs), "sec_per_step": secs,
              "mean_sec_per_step": sum(secs) / len(secs) if secs else None,
              "mean_sec_rollout": sum(h["sec_rollout"] for h in history) / max(1, len(history)),
              "mean_sec_update": sum(h["sec_ref"] + h["sec_update"] for h in history)
              / max(1, len(history)),
              "wall_seconds": time.perf_counter() - t0,
              "P": sampler.per_step, "G": r["group_size"], "micro_batch": t["micro_batch"],
              "gen_batch": t["gen_batch"], "gpu": run["meta"].get("gpu"),
              "dtype": run["meta"]["dtype"], "model": model_id}
    if torch.cuda.is_available():
        timing["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    with open(os.path.join(a.out, "timing.json"), "w") as f:
        json.dump(timing, f, indent=1)
    print(json.dumps({k: v for k, v in timing.items() if k != "sec_per_step"}))


if __name__ == "__main__":
    main()
