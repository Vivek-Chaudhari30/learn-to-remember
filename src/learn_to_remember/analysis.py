"""Figure data for the M6 write-up, pulled out of the result files.

Pure functions (no plotting, no model): figures.py and notebooks/analysis.ipynb
both call these, so the notebook and scripts/make_figures.py cannot disagree.
Every value comes from a file written by eval_runner (eval results) or
train_loop (train_log.json). A run whose file does not exist is listed under
"missing" and drawn as nothing; nothing is filled in or interpolated.

A point is (value, lo, hi): the estimate and its 95% CI. For summary metrics the
CI is the one eval.summarize stored; for per-qtype rates computed here it is a
percentile bootstrap over questions with the same defaults (2000, seed 0).
"""

from __future__ import annotations

import json
import os

import numpy as np

from .report import Runs
from .workspace.questions import QTYPES

A, B = "company_a", "company_b"
VERSIONS = ("v0", "v1", "v2", "v3")
LAMBDAS = ("0", "0.2", "0.5")

# train_log.json keys worth a panel, with readable labels.
CURVE_KEYS = (
    ("reward_mean", "reward (mean)"),
    ("frac_correct", "fraction correct"),
    ("tokens_read_mean", "tokens read / trajectory"),
    ("valid_call_rate", "valid tool-call rate"),
    ("zero_lookup_rate", "zero-lookup answers"),
    ("frac_traj_with_repeats", "trajectories with repeated calls"),
    ("kl", "KL to reference"),
    ("entropy", "entropy (sampled tokens)"),
    ("frac_zero_var_groups", "zero-variance groups"),
)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def point(m) -> tuple | None:
    """A summary metric block {'value', 'ci95'} -> (value, lo, hi)."""
    if not m:
        return None
    lo, hi = m["ci95"] if m.get("ci95") else (m["value"], m["value"])
    return (m["value"], lo, hi)


def metric_point(rec: dict | None, metric: str, qtype: str | None = None):
    if rec is None:
        return None
    s = rec["summary"]
    blk = s["overall"] if qtype is None else s["by_qtype"].get(qtype)
    return point(blk.get(metric)) if blk else None


def mean_ci(values, *, n_boot: int = 2000, seed: int = 0) -> tuple | None:
    """Mean with a percentile bootstrap 95% CI; None for no values."""
    x = np.asarray([float(v) for v in values])
    if x.size == 0:
        return None
    rng = np.random.default_rng(seed)
    boots = x[rng.integers(0, x.size, (n_boot, x.size))].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return (float(x.mean()), float(lo), float(hi))


def rolling(xs: list, w: int) -> list:
    """Trailing mean over the last w finite values (None stays out of the mean)."""
    out, buf = [], []
    for v in xs:
        if v is not None and np.isfinite(v):
            buf.append(float(v))
        buf = buf[-w:]
        out.append(float(np.mean(buf)) if buf else None)
    return out


# --------------------------------------------------------------------------- #
# Training logs
# --------------------------------------------------------------------------- #
def load_train_log(root: str, d: str) -> list | None:
    p = os.path.join(root, d, "train_log.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def train_curves(history: list, keys=CURVE_KEYS, window: int = 10) -> dict:
    """{'step': [...], key: {'label', 'raw', 'smooth'}} for keys present in the log."""
    steps = [h["step"] for h in history]
    out = {"step": steps, "window": window, "series": {}}
    for k, label in keys:
        raw = [h.get(k) for h in history]
        if all(v is None for v in raw):
            continue
        out["series"][k] = {"label": label, "raw": raw, "smooth": rolling(raw, window)}
    return out


def train_runs(spec: dict, root: str, window: int = 10) -> dict:
    """{run: curves or None} for every run listed under spec['train_logs']."""
    return {run: (lambda h: train_curves(h, window=window) if h else None)(
                load_train_log(root, d))
            for run, d in spec.get("train_logs", {}).items()}


# --------------------------------------------------------------------------- #
# RQ figures
# --------------------------------------------------------------------------- #
def _missing(labels_recs) -> list[str]:
    return [lab for lab, rec in labels_recs if rec is None]


def rq1_data(runs: Runs) -> dict:
    cells = {("base", "no notes"): runs.get("base", A, "no_notes"),
             ("base", "notes v3"): runs.get("base", A, "v3"),
             ("trained", "no notes"): runs.get("nonotes_l0.2", A, "no_notes"),
             ("trained", "notes v3"): runs.get("notes_l0.2", A, "v3")}
    groups = ["no notes", "notes v3"]
    series = {s: [metric_point(cells[(s, g)], "accuracy") for g in groups]
              for s in ("base", "trained")}
    return {"groups": groups, "series": series,
            "missing": [f"{s}, {g}" for (s, g), r in cells.items() if r is None]}


def rq2_data(runs: Runs) -> dict:
    labelled = [("base", runs.get("base", A, "v3"))] + [
        (f"λ={lam}", runs.get(f"notes_l{lam}", A, "v3")) for lam in LAMBDAS]
    pts = [{"label": lab, "accuracy": metric_point(r, "accuracy"),
            "tokens": metric_point(r, "mean_tokens_read"),
            "zero_lookup": metric_point(r, "zero_lookup_rate")}
           for lab, r in labelled if r is not None]
    return {"points": pts, "missing": _missing(labelled)}


def rq3_data(runs: Runs) -> dict:
    series, missing = {}, []
    for label, run in (("base", "base"), ("trained on v0", "v0_l0.2"),
                       ("trained on v3", "notes_l0.2")):
        recs = [runs.get(run, A, v) for v in VERSIONS]
        missing += [f"{label}, {v}" for v, r in zip(VERSIONS, recs) if r is None]
        if any(recs):
            series[label] = [metric_point(r, "accuracy") for r in recs]
    return {"versions": list(VERSIONS), "series": series, "missing": missing}


def rq4_data(runs: Runs) -> dict:
    labelled = [("base", runs.get("base", B, "v3")),
                ("trained on v3", runs.get("notes_l0.2", B, "v3")),
                ("trained on v0", runs.get("v0_l0.2", B, "v3"))]
    return {"bars": [(lab, metric_point(r, "accuracy")) for lab, r in labelled if r],
            "missing": _missing(labelled)}


def _by_qtype_rate(rec, field: str) -> list:
    if rec is None:
        return [None] * len(QTYPES)
    return [mean_ci([r[field] for r in rec["rows"] if r["qtype"] == t]) for t in QTYPES]


def behavior_by_qtype(runs: Runs) -> dict:
    """Accuracy and file-tool use per v3 qtype, base vs trained (λ=0.2), company A.
    File use on gap/stale is the wanted fallback; on lookup it is a false alarm."""
    base, tr = runs.get("base", A, "v3"), runs.get("notes_l0.2", A, "v3")
    present = {lab: r for lab, r in (("base", base), ("trained", tr)) if r}
    return {"qtypes": list(QTYPES),
            "accuracy": {lab: _by_qtype_rate(r, "correct") for lab, r in present.items()},
            "used_files": {lab: _by_qtype_rate(r, "used_files") for lab, r in present.items()},
            "missing": _missing((("base", base), ("trained", tr)))}


def lookup_histogram(runs: Runs, cap: int = 4) -> dict:
    """Share of questions answered after 0, 1, ..., cap+ lookups."""
    labels = [str(i) for i in range(cap)] + [f"{cap}+"]
    out, labelled = {}, (("base", runs.get("base", A, "v3")),
                         ("trained", runs.get("notes_l0.2", A, "v3")))
    for lab, rec in labelled:
        if rec is None:
            continue
        n = np.array([min(r["n_lookups"], cap) for r in rec["rows"]])
        out[lab] = [float(np.mean(n == i)) for i in range(cap + 1)]
    return {"bins": labels, "series": out, "missing": _missing(labelled)}


# --------------------------------------------------------------------------- #
# Paired trajectory examples (README / notebook)
# --------------------------------------------------------------------------- #
def _trajectories(root: str, d: str, company: str, condition: str) -> dict:
    """id -> readable trajectory: every rollout if kept, else the saved samples."""
    tag = f"{company}__{condition}"
    full = os.path.join(root, d, "rollouts", f"{tag}.jsonl")
    if os.path.exists(full):
        with open(full) as f:
            recs = [json.loads(line) for line in f if line.strip()]
        return {r["id"]: r for r in recs}
    samp = os.path.join(root, d, f"{tag}_samples.json")
    if os.path.exists(samp):
        with open(samp) as f:
            return {r["id"]: r for r in json.load(f)}
    return {}


def paired_examples(spec: dict, root: str, run_a: str = "base", run_b: str = "notes_l0.2",
                    company: str = A, condition: str = "v3", k: int = 3) -> dict:
    """Questions run_b got right and run_a got wrong ('wins'), and the reverse
    ('losses'), with both trajectories when they were saved. Picks the first k by
    id within each qtype order, so the choice is fixed, not cherry-picked."""
    runs = Runs(spec, root)
    ra, rb = runs.get(run_a, company, condition), runs.get(run_b, company, condition)
    if ra is None or rb is None:
        return {"wins": [], "losses": [], "n_wins": None, "n_losses": None,
                "missing": [r for r, x in ((run_a, ra), (run_b, rb)) if x is None]}
    ia = {r["id"]: r for r in ra["rows"]}
    ib = {r["id"]: r for r in rb["rows"]}
    shared = sorted(set(ia) & set(ib), key=lambda i: (QTYPES.index(ia[i]["qtype"]), i))
    wins = [i for i in shared if ib[i]["correct"] and not ia[i]["correct"]]
    losses = [i for i in shared if ia[i]["correct"] and not ib[i]["correct"]]

    def trajs(run):
        for d in runs.dirs.get(run, []):
            t = _trajectories(root, d, company, condition)
            if t:
                return t
        return {}

    ta, tb = trajs(run_a), trajs(run_b)

    def pick(ids):
        # Prefer questions whose two trajectories were both saved.
        have = [i for i in ids if i in ta and i in tb] + [i for i in ids
                                                          if not (i in ta and i in tb)]
        return [{"id": i, "qtype": ia[i]["qtype"], run_a: ta.get(i), run_b: tb.get(i),
                 "row_a": ia[i], "row_b": ib[i]} for i in have[:k]]

    return {"wins": pick(wins), "losses": pick(losses), "n_wins": len(wins),
            "n_losses": len(losses), "missing": []}


def format_trajectory(t: dict | None, max_obs: int = 300) -> str:
    """Plain-text view of one trajectory record (eval.trajectory_record format)."""
    if t is None:
        return "(trajectory not saved)"
    lines = [f"Q: {t['question']}", f"gold: {t['gold']} | answer: {t['answer']} | "
             f"correct: {t['correct']} | tokens read: {t['tokens_read']} | ended: {t['ended']}"]
    for n, turn in enumerate(t["turns"], 1):
        lines.append(f"--- turn {n} (assistant)\n{turn['assistant'].strip()}")
        if turn.get("observation"):
            obs = turn["observation"]
            obs = obs if len(obs) <= max_obs else obs[:max_obs] + f" ... [{len(obs)} chars]"
            lines.append(f"--- tool ({turn['tool']}, {turn['tokens']} tokens)\n{obs}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# What exists
# --------------------------------------------------------------------------- #
def availability(spec: dict, root: str) -> list[tuple[str, str, str, bool]]:
    """(run, company, condition, exists) for every eval file the report reads."""
    runs = Runs(spec, root)
    want = [("base", A, c) for c in ("no_notes", "v3", "v0", "v1", "v2")] + \
           [("base", B, "v3"), ("nonotes_l0.2", A, "no_notes")] + \
           [("notes_l0.2", A, v) for v in VERSIONS] + [("notes_l0.2", B, "v3")] + \
           [(f"notes_l{lam}", A, "v3") for lam in ("0", "0.5")] + \
           [("v0_l0.2", A, v) for v in VERSIONS] + [("v0_l0.2", B, "v3")]
    return [(r, c, k, runs.get(r, c, k) is not None) for r, c, k in want]
