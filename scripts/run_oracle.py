"""Run the scripted oracle on the eval sets: companies A and B x notes v0-v3 x no-notes.

    .venv/bin/python scripts/run_oracle.py

Counts tokens with the Qwen2.5 tokenizer, so caps and the cost term match real
runs. Writes results/oracle/oracle_results.json and prints a markdown table.
Accuracy below ~100% means some questions cannot be solved within the env's
limits; every failure carries a reason (see src/learn_to_remember/oracle.py).
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from statistics import mean

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from learn_to_remember.env.episode import EnvConfig, behavior  # noqa: E402
from learn_to_remember.env.tokens import load_qwen_counter  # noqa: E402
from learn_to_remember.oracle import Oracle, make_env  # noqa: E402
from learn_to_remember.reward import build_catalog  # noqa: E402
from learn_to_remember.workspace.generate import generate_company  # noqa: E402
from learn_to_remember.workspace.notes import VERSIONS  # noqa: E402
from learn_to_remember.workspace.questions import QTYPES  # noqa: E402


def main():
    counter = load_qwen_counter()
    cfg = EnvConfig()
    out = []
    for company, seed, with_train in (("company_a", 0, True), ("company_b", 1, False)):
        ws = generate_company(seed, company, with_train=with_train)
        cat = build_catalog(ws.world)
        for cond in list(VERSIONS) + ["no_notes"]:
            version = None if cond == "no_notes" else cond
            env = make_env(ws, version, counter, cfg)
            oracle = Oracle(ws, version or "v3", notes=version is not None)
            rows = []
            for q in ws.splits["eval"]:
                res, why = oracle.run(env, q, cat, cfg)
                b = behavior(res)
                rows.append({"id": q["id"], "qtype": res.qtype, "correct": res.correct,
                             "reason": why, "tokens_read": res.tokens_read,
                             "turns": b["n_turns"], "reward": res.reward})
            by_q = {t: [r for r in rows if r["qtype"] == t] for t in QTYPES}
            out.append({
                "company": company, "condition": cond, "n": len(rows),
                "correct": sum(r["correct"] for r in rows),
                "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 4),
                "by_qtype": {t: f"{sum(r['correct'] for r in v)}/{len(v)}"
                             for t, v in by_q.items() if v},
                "failures": dict(Counter(r["reason"] for r in rows if not r["correct"])),
                "failed_ids": [r["id"] for r in rows if not r["correct"]],
                "mean_tokens_read": round(mean(r["tokens_read"] for r in rows), 1),
                "mean_turns": round(mean(r["turns"] for r in rows), 2),
                "max_turns_used": max(r["turns"] for r in rows),
                "fallbacks": oracle.fallbacks,
            })
            print(f"{company} {cond}: {out[-1]['correct']}/{len(rows)}", file=sys.stderr)

    os.makedirs(os.path.join(ROOT, "results", "oracle"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "oracle", "oracle_results.json"), "w") as f:
        f.write(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("| company | condition | accuracy | lookup/multihop/aggregate/gap/stale | "
          "failures | mean tokens read | mean turns |")
    print("|---|---|---|---|---|---|---|")
    for r in out:
        qt = " / ".join(r["by_qtype"].get(t, "-") for t in QTYPES)
        fails = ", ".join(f"{k} {v}" for k, v in r["failures"].items()) or "none"
        print(f"| {r['company']} | {r['condition']} | {r['correct']}/{r['n']} "
              f"({100 * r['accuracy']:.1f}%) | {qt} | {fails} | {r['mean_tokens_read']} | "
              f"{r['mean_turns']} |")


if __name__ == "__main__":
    main()
