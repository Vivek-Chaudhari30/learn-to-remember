"""Paired comparisons between two eval runs on the same questions (M5).

Two runs evaluated on one question set are paired: question i was answered by
both. Pairing removes question difficulty from the noise, so these tests are
tighter (and more honest) than comparing two independent CIs.

  paired_diff(a, b, "accuracy")   b - a, with a paired bootstrap 95% CI
  relative_change(a, b, "tokens_read")  mean(b) / mean(a) - 1, paired bootstrap CI
  mcnemar_exact(a, b)             exact two-sided McNemar test on correctness

A difference is only called a difference when its CI excludes 0 (spec §8).
"""

from __future__ import annotations

from math import comb

import numpy as np

_FIELDS = {"accuracy": "correct", "tokens_read": "tokens_read", "n_lookups": "n_lookups",
           "reward": "reward", "zero_lookup": "zero_lookup_answer", "used_files": "used_files",
           "n_turns": "n_turns"}


def align(rows_a: list[dict], rows_b: list[dict]) -> tuple[list[dict], list[dict]]:
    """Rows of both runs on the questions they share, in one order. Refuses to
    pair runs whose question sets differ (that would be a silent mismatch)."""
    ia = {r["id"]: r for r in rows_a}
    ib = {r["id"]: r for r in rows_b}
    if set(ia) != set(ib):
        raise ValueError(f"runs cover different questions ({len(set(ia) ^ set(ib))} differ)")
    ids = sorted(ia)
    return [ia[i] for i in ids], [ib[i] for i in ids]


def _vals(rows, metric):
    return np.array([float(r[_FIELDS[metric]]) for r in rows])


def _subset(a, b, qtype):
    if qtype is None:
        return a, b
    keep = [i for i, r in enumerate(a) if r["qtype"] == qtype]
    return [a[i] for i in keep], [b[i] for i in keep]


def paired_diff(rows_a, rows_b, metric: str = "accuracy", *, qtype: str | None = None,
                n_boot: int = 2000, seed: int = 0) -> dict:
    """mean(b) - mean(a) over shared questions, percentile paired-bootstrap CI."""
    a, b = _subset(*align(rows_a, rows_b), qtype)
    if not a:
        return None
    d = _vals(b, metric) - _vals(a, metric)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"diff": float(d.mean()), "ci95": [float(lo), float(hi)], "n": len(d),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def relative_change(rows_a, rows_b, metric: str = "tokens_read", *, qtype: str | None = None,
                    n_boot: int = 2000, seed: int = 0) -> dict | None:
    """mean(b) / mean(a) - 1 (e.g. -0.3 = 30% fewer tokens), paired bootstrap CI."""
    a, b = _subset(*align(rows_a, rows_b), qtype)
    if not a:
        return None
    va, vb = _vals(a, metric), _vals(b, metric)
    if va.mean() == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(va), (n_boot, len(va)))
    ma, mb = va[idx].mean(axis=1), vb[idx].mean(axis=1)
    ok = ma > 0
    lo, hi = np.quantile(mb[ok] / ma[ok] - 1, [0.025, 0.975])
    return {"change": float(vb.mean() / va.mean() - 1), "ci95": [float(lo), float(hi)],
            "n": len(va), "excludes_zero": bool(lo > 0 or hi < 0)}


def mcnemar_exact(rows_a, rows_b, *, qtype: str | None = None) -> dict:
    """Exact two-sided McNemar test on per-question correctness.
    n01 = wrong in a, right in b; n10 = right in a, wrong in b."""
    a, b = _subset(*align(rows_a, rows_b), qtype)
    n01 = sum(1 for x, y in zip(a, b) if not x["correct"] and y["correct"])
    n10 = sum(1 for x, y in zip(a, b) if x["correct"] and not y["correct"])
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "p": 1.0}
    k = min(n01, n10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"n01": n01, "n10": n10, "p": p}
