"""Generate company A (seed 0, train + eval) and company B (seed 1, eval only).

    .venv/bin/python scripts/generate_workspace.py            # with Qwen token counts
    .venv/bin/python scripts/generate_workspace.py --no-tokenizer

Writes:
  data/workspaces/<company>/   full workspace (JSONL + browsable .md), gitignored
  data/eval/<company>_eval.jsonl  committed eval sets
  data/eval/manifest.json         sha256 of every artifact, to verify regeneration
  results/workspace_stats.json    the numbers behind the stats table in docs/RESULTS.md
and prints that table as markdown.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from learn_to_remember.workspace.generate import (  # noqa: E402
    generate_company, serialize, stats, write_workspace,
)
from learn_to_remember.workspace.notes import VERSIONS  # noqa: E402
from learn_to_remember.workspace.questions import QTYPES  # noqa: E402

COMPANIES = [("company_a", 0, True), ("company_b", 1, False)]


def table(all_stats: list[dict]) -> str:
    a = {s["company"]: s for s in all_stats}
    names = [s["company"] for s in all_stats]
    u = all_stats[0]["unit"]
    rows = [("Seed", lambda s: s["seed"]),
            ("Entities (office/team/person/project/client/contract)",
             lambda s: f'{s["n_entities"]} ({"/".join(str(s["entities"][k]) for k in ["office", "team", "person", "project", "client", "contract"])})'),
            ("Change events", lambda s: s["n_events"]),
            ("Raw documents", lambda s: s["docs"]["n"]),
            (f"Raw {u} total", lambda s: f'{s["docs"][f"total_{u}"]:,}'),
            (f"Raw {u} per doc (min / median / max)",
             lambda s: f'{s["docs"][f"min_{u}"]} / {s["docs"][f"median_{u}"]:.0f} / {s["docs"][f"max_{u}"]}')]
    for v in VERSIONS:
        rows.append((f"Notes {v}: pages (coverage, truncated, {u})",
                     lambda s, v=v: f'{s["notes"][v]["pages"]} ({s["notes"][v]["coverage_actual"]:.2f}, '
                                    f'{s["notes"][v]["truncated_pages"]}, {s["notes"][v][f"total_{u}"]:,})'))
    rows += [("Held-out entities", lambda s: s["n_heldout_entities"]),
             ("Train questions", lambda s: s["questions"]["train"]["n"]),
             ("Eval questions (held-out-entity Qs)",
              lambda s: f'{s["questions"]["eval"]["n"]} ({s["questions"]["eval"]["heldout_questions"]})')]
    for split in ("train", "eval"):
        rows.append((f"{split} qtype @v3 ({'/'.join(QTYPES)})",
                     lambda s, sp=split: "/".join(str(s["questions"][sp]["qtype_v3"][t]) for t in QTYPES)))
    for v in VERSIONS:
        rows.append((f"eval qtype @{v}",
                     lambda s, v=v: "/".join(str(s["questions"]["eval"]["qtype_by_version"][v][t])
                                             for t in QTYPES)))
    for v in VERSIONS:
        rows.append((f"eval answerable from notes alone @{v}",
                     lambda s, v=v: f'{s["questions"]["eval"]["notes_ok_by_version"][v]}'
                                    f'/{s["questions"]["eval"]["n"]}'))
    out = ["| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for label, f in rows:
        out.append(f"| {label} | " + " | ".join(str(f(a[n])) for n in names) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "workspaces"))
    ap.add_argument("--no-tokenizer", action="store_true")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="Qwen2.5 0.5B and 1.5B share one tokenizer")
    args = ap.parse_args()

    count = None
    if not args.no_tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        count = lambda t: len(tok(t, add_special_tokens=False)["input_ids"])  # noqa: E731

    eval_dir = os.path.join(ROOT, "data", "eval")
    os.makedirs(eval_dir, exist_ok=True)
    manifests, all_stats = {}, []
    for company, seed, with_train in COMPANIES:
        ws = generate_company(seed, company, with_train=with_train)
        manifests[company] = write_workspace(ws, os.path.join(args.out, company))
        with open(os.path.join(eval_dir, f"{company}_eval.jsonl"), "w") as f:
            f.write(serialize(ws)["questions_eval.jsonl"])
        all_stats.append(stats(ws, count))
        print(f"{company}: {len(ws.docs)} docs, {len(ws.splits['train'])} train, "
              f"{len(ws.splits['eval'])} eval", file=sys.stderr)

    with open(os.path.join(eval_dir, "manifest.json"), "w") as f:
        f.write(json.dumps(manifests, indent=2, sort_keys=True) + "\n")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "workspace_stats.json"), "w") as f:
        f.write(json.dumps(all_stats, indent=2, sort_keys=True) + "\n")
    print(table(all_stats))


if __name__ == "__main__":
    main()
