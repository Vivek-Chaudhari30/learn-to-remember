"""Build the RQ1-RQ5 result tables from eval result files (M5).

Every number comes from a result file written by eval_runner; a cell whose file
does not exist says "not run". Differences between runs are paired (same
questions): bootstrap 95% CI on the difference and an exact McNemar p for
accuracy. Nothing here decides a hypothesis is "confirmed"; the tables show
the estimate and its CI, and the reader (and README) state what they support.
"""

from __future__ import annotations

import json
import os

from .compare import mcnemar_exact, paired_diff, relative_change
from .eval import summary_table
from .workspace.questions import QTYPES

NOT_RUN = "not run"


class Runs:
    """Look up <dir>/<company>__<condition>.json for a run label."""

    def __init__(self, spec: dict, root: str):
        self.root = root
        self.slug = spec.get("slug", "")
        self.dirs = {k: [d.format(slug=self.slug) for d in v] for k, v in spec["runs"].items()}
        self._cache: dict = {}

    def get(self, run: str, company: str, condition: str) -> dict | None:
        key = (run, company, condition)
        if key not in self._cache:
            self._cache[key] = None
            for d in self.dirs.get(run, []):
                p = os.path.join(self.root, d, f"{company}__{condition}.json")
                if os.path.exists(p):
                    with open(p) as f:
                        self._cache[key] = json.load(f)
                    break
        return self._cache[key]


def _pct(m):
    if m is None:
        return NOT_RUN
    v, (lo, hi) = m["value"], m["ci95"]
    return f"{100 * v:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"


def _num(m, nd=0):
    if m is None:
        return NOT_RUN
    v, (lo, hi) = m["value"], m["ci95"]
    return f"{v:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def _acc(rec, qtype=None):
    if rec is None:
        return NOT_RUN
    s = rec["summary"]
    blk = s["overall"] if qtype is None else s["by_qtype"].get(qtype)
    return _pct(blk["accuracy"]) if blk else "-"


def _diff(a, b, metric="accuracy", qtype=None, pct=True):
    """'+x.x [lo, hi], McNemar p' for b - a, or 'not run'."""
    if a is None or b is None:
        return NOT_RUN
    d = paired_diff(a["rows"], b["rows"], metric, qtype=qtype)
    if d is None:
        return "-"
    k = 100 if pct else 1
    s = f"{k * d['diff']:+.1f} [{k * d['ci95'][0]:+.1f}, {k * d['ci95'][1]:+.1f}]"
    if metric == "accuracy":
        s += f", p={mcnemar_exact(a['rows'], b['rows'], qtype=qtype)['p']:.3g}"
    return s


def _rel(a, b, metric="tokens_read"):
    if a is None or b is None:
        return NOT_RUN
    r = relative_change(a["rows"], b["rows"], metric)
    if r is None:
        return "-"
    return f"{100 * r['change']:+.1f}% [{100 * r['ci95'][0]:+.1f}, {100 * r['ci95'][1]:+.1f}]"


def _model_line(runs: Runs) -> str:
    for run in runs.dirs:
        for company in ("company_a", "company_b"):
            for cond in ("v3", "no_notes"):
                rec = runs.get(run, company, cond)
                if rec:
                    m = rec["meta"]
                    return f"Model: {m['model_id']} ({m['dtype']}, {m.get('gpu') or m['device']})."
    return "Model: not run."


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def rq1(runs: Runs) -> str:
    A = "company_a"
    base_nn, base_n = runs.get("base", A, "no_notes"), runs.get("base", A, "v3")
    tr_nn, tr_n = runs.get("nonotes_l0.2", A, "no_notes"), runs.get("notes_l0.2", A, "v3")
    out = ["### RQ1: notes x training (company A eval, accuracy % [95% CI])", "",
           "| | no notes | notes (v3) |", "|---|---|---|",
           f"| base model | {_acc(base_nn)} | {_acc(base_n)} |",
           f"| trained (λ=0.2) | {_acc(tr_nn)} | {_acc(tr_n)} |", "",
           "Paired differences (points, [95% CI], exact McNemar p):", "",
           "| comparison | Δ accuracy |", "|---|---|",
           f"| notes only − base | {_diff(base_nn, base_n)} |",
           f"| training only − base | {_diff(base_nn, tr_nn)} |",
           f"| notes + training − notes only | {_diff(base_n, tr_n)} |",
           f"| notes + training − training only | {_diff(tr_nn, tr_n)} |", ""]
    return "\n".join(out)


def rq2(runs: Runs) -> str:
    A = "company_a"
    rows = [("base", runs.get("base", A, "v3"))] + [
        (f"trained λ={lam}", runs.get(f"notes_l{lam}", A, "v3")) for lam in ("0", "0.2", "0.5")]
    ref = rows[1][1]                     # λ = 0
    out = ["### RQ2: cost penalty (company A, notes v3)", "",
           "| run | accuracy % | tokens read / q | zero-lookup % | Δ accuracy vs λ=0 | "
           "tokens vs λ=0 |", "|---|---|---|---|---|---|"]
    for label, rec in rows:
        o = rec["summary"]["overall"] if rec else None
        out.append(f"| {label} | {_acc(rec)} | {_num(o['mean_tokens_read']) if o else NOT_RUN} | "
                   f"{_pct(o['zero_lookup_rate']) if o else NOT_RUN} | "
                   f"{'-' if label == 'trained λ=0' else _diff(ref, rec)} | "
                   f"{'-' if label == 'trained λ=0' else _rel(ref, rec)} |")
    out += ["", "Hypothesis (spec): a moderate λ cuts tokens read by ≥30% for ≤2 points of "
            "accuracy. Read it off the λ=0.2 row: both CIs, not the point estimates.", ""]
    return "\n".join(out)


def rq3(runs: Runs) -> str:
    A, vs = "company_a", ("v0", "v1", "v2", "v3")
    out = ["### RQ3: notes drift (company A, accuracy % by eval notes version)", "",
           "| run | " + " | ".join(vs) + " |", "|---|" + "---|" * len(vs)]
    for label, run in (("base", "base"), ("trained on v0", "v0_l0.2"),
                       ("trained on v3", "notes_l0.2")):
        out.append(f"| {label} | " + " | ".join(_acc(runs.get(run, A, v)) for v in vs) + " |")
    out += ["", "| paired Δ accuracy | " + " | ".join(vs) + " |", "|---|" + "---|" * len(vs)]
    for label, a, b in (("trained on v0 − base", "base", "v0_l0.2"),
                        ("trained on v3 − trained on v0", "v0_l0.2", "notes_l0.2")):
        out.append(f"| {label} | " + " | ".join(
            _diff(runs.get(a, A, v), runs.get(b, A, v)) for v in vs) + " |")
    out.append("")
    return "\n".join(out)


def rq4(runs: Runs) -> str:
    B = "company_b"
    base = runs.get("base", B, "v3")
    out = ["### RQ4: transfer to unseen company B (notes v3, no training on B)", "",
           "| run | accuracy % | Δ vs base |", "|---|---|---|",
           f"| base | {_acc(base)} | - |"]
    for label, run in (("trained on A, v3 notes", "notes_l0.2"), ("trained on A, v0 notes", "v0_l0.2")):
        rec = runs.get(run, B, "v3")
        out.append(f"| {label} | {_acc(rec)} | {_diff(base, rec)} |")
    out.append("")
    return "\n".join(out)


def rq5(runs: Runs) -> str:
    A = "company_a"
    base, tr = runs.get("base", A, "v3"), runs.get("notes_l0.2", A, "v3")
    out = ["### RQ5: behavior (company A, notes v3), base vs trained (λ=0.2)", ""]
    pairs = [(lab, r["summary"]) for lab, r in (("base", base), ("trained", tr)) if r]
    out.append(summary_table(pairs) if pairs else NOT_RUN)
    out += ["", "| paired Δ (trained − base) | value [95% CI] |", "|---|---|",
            f"| file-tool use on gap questions | {_diff(base, tr, 'used_files', 'gap')} |",
            f"| file-tool use on stale questions | {_diff(base, tr, 'used_files', 'stale')} |",
            f"| file-tool use on lookup questions (false alarms) | "
            f"{_diff(base, tr, 'used_files', 'lookup')} |",
            f"| answers with zero lookups | {_diff(base, tr, 'zero_lookup')} |",
            f"| lookups per question | {_diff(base, tr, 'n_lookups', pct=False)} |",
            f"| tokens read (relative) | {_rel(base, tr)} |", "",
            "Accuracy by qtype:", "",
            "| run | " + " | ".join(QTYPES) + " |", "|---|" + "---|" * len(QTYPES),
            "| base | " + " | ".join(_acc(base, t) for t in QTYPES) + " |",
            "| trained | " + " | ".join(_acc(tr, t) for t in QTYPES) + " |",
            "| Δ (paired) | " + " | ".join(_diff(base, tr, qtype=t) for t in QTYPES) + " |", ""]
    return "\n".join(out)


def build_report(spec: dict, root: str) -> str:
    runs = Runs(spec, root)
    parts = ["# Results report (generated by scripts/make_report.py)", "",
             _model_line(runs), "",
             "All accuracy numbers are greedy, on the full eval sets, with percentile "
             "bootstrap 95% CIs over questions. Δ rows are paired on the same questions.", "",
             rq1(runs), rq2(runs), rq3(runs), rq4(runs), rq5(runs)]
    return "\n".join(parts)
