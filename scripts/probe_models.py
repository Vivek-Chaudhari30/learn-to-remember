"""Model probe (spec §3.5): each candidate on the 50 probe questions (company A,
notes v3, greedy). Applies the selection rule: valid tool calls >= 80% and
accuracy in [20%, 60%].

    python scripts/probe_models.py --dtype fp16 \
        --models Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen3-1.7B --out results/probe

Writes results/probe/<model>/company_a__v3.json (+ samples) and
results/probe/probe_summary.json. It reports the verdicts; the final choice is
recorded by hand in docs/DECISIONS.md.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from learn_to_remember.eval import probe_verdict, qtype_table, summary_table  # noqa: E402
from learn_to_remember.eval_runner import run_jobs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen3-1.7B")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="plumbing only")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="results/probe")
    a = ap.parse_args()

    out = []
    for m in a.models.split(","):
        slug = m.split("/")[-1].lower()
        (rec,) = run_jobs(model_id=m, companies=["company_a"], conditions=["v3"],
                          out_dir=os.path.join(a.out, slug), subset="probe", limit=a.limit,
                          dtype=a.dtype, gen_batch=a.gen_batch, n_boot=a.n_boot)
        v = probe_verdict(rec["summary"])
        out.append({"model": m, **v, "seconds_per_question": rec["meta"]["seconds_per_question"],
                    "gpu": rec["meta"].get("gpu"), "dtype": rec["meta"]["dtype"],
                    "summary_overall": rec["summary"]["overall"], "label": slug,
                    "summary": rec["summary"]})
        print(f"{m}: valid calls {v['valid_call_rate']:.3f}, accuracy {v['accuracy']:.3f}, "
              f"passes={v['passes']}")
        try:                                   # free GPU memory before the next model
            import torch
            del rec
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    pairs = [(r["label"], r["summary"]) for r in out]
    print(summary_table(pairs) + "\n\n" + qtype_table(pairs))
    with open(os.path.join(a.out, "probe_summary.json"), "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "summary"} for r in out], f, indent=1)
    passing = [r["model"] for r in out if r["passes"]]
    print("passing the §3.5 rule:", passing or "none")


if __name__ == "__main__":
    main()
