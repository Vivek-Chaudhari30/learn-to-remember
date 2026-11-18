"""Evaluate a model (base or base + LoRA adapter) on the eval sets, greedy.

Baselines (M3), on a GPU:
    python scripts/run_eval.py --model Qwen/Qwen2.5-1.5B-Instruct --dtype fp16 \
        --companies company_a,company_b --conditions v3,no_notes --out results/baselines/qwen2.5-1.5b

Timing first (never estimate s/question):
    python scripts/run_eval.py ... --limit 16 --out results/timing/...

Mac plumbing smoke (0.5B, no conclusions):
    .venv/bin/python scripts/run_eval.py --model Qwen/Qwen2.5-0.5B-Instruct --limit 4 \
        --conditions v3 --companies company_a --out results/smoke --n-boot 100

Writes one JSON per (company, condition) (see eval_runner.py), and rebuilds
tables.md / merges timing.json over everything in --out, so several calls can
share one folder.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from learn_to_remember.eval import qtype_table, summary_table  # noqa: E402
from learn_to_remember.eval_runner import run_jobs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--companies", default="company_a,company_b")
    ap.add_argument("--conditions", default="v3,no_notes",
                    help="comma list of v0,v1,v2,v3,no_notes")
    ap.add_argument("--subset", default="all", choices=["all", "probe"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--episodes-per-rollout", type=int, default=64)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    records = run_jobs(model_id=a.model, adapter=a.adapter, companies=a.companies.split(","),
                       conditions=a.conditions.split(","), out_dir=a.out, subset=a.subset,
                       limit=a.limit, dtype=a.dtype, device=a.device, gen_batch=a.gen_batch,
                       episodes_per_rollout=a.episodes_per_rollout, config=a.config,
                       n_boot=a.n_boot)
    # Several calls may share one --out folder (e.g. company A then company B), so
    # tables.md is rebuilt from every result file there and timing.json is merged.
    import glob
    recs = []
    for p in sorted(glob.glob(os.path.join(a.out, "*__*.json"))):
        if not p.endswith("_samples.json"):
            with open(p) as f:
                recs.append(json.load(f))
    pairs = [(f"{r['meta']['company']} {r['meta']['condition']}", r["summary"]) for r in recs]
    tables = summary_table(pairs) + "\n\nAccuracy by qtype:\n\n" + qtype_table(pairs) + "\n"
    models = sorted({r["meta"]["model_id"] + (f" + {r['meta']['adapter']}" if r["meta"].get("adapter")
                                               else "") for r in recs})
    with open(os.path.join(a.out, "tables.md"), "w") as f:
        f.write("model: " + "; ".join(models) + "\n\n" + tables)
    print(tables)
    tpath = os.path.join(a.out, "timing.json")
    timing = []
    if os.path.exists(tpath):
        with open(tpath) as f:
            timing = json.load(f)
    new = [{k: r["meta"][k] for k in ("company", "condition", "n_questions", "wall_seconds",
                                      "seconds_per_question", "gen_batch", "device")}
           | {"gpu": r["meta"].get("gpu")} for r in records]
    done = {(t["company"], t["condition"]) for t in new}
    timing = [t for t in timing if (t["company"], t["condition"]) not in done] + new
    with open(tpath, "w") as f:
        json.dump(timing, f, indent=1)


if __name__ == "__main__":
    main()
