"""Paired comparisons (compare.py) and the RQ report (report.py), on synthetic
result files written in the eval_runner format."""

import json
import os

import pytest

from learn_to_remember.compare import align, mcnemar_exact, paired_diff, relative_change
from learn_to_remember.eval import summarize
from learn_to_remember.report import NOT_RUN, build_report
from learn_to_remember.workspace.questions import QTYPES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rows(correct, tokens=None, condition="v3", files=None):
    n = len(correct)
    tokens = tokens or [100] * n
    files = files or [False] * n
    return [{"id": f"q{i:03d}", "qtype": QTYPES[i % 5], "condition": condition,
             "correct": bool(c), "tokens_read": t, "n_lookups": 1, "reward": float(c),
             "tokens_per_lookup": [t], "used_files": f, "used_notes": True,
             "valid_calls": 2, "n_calls_total": 2, "repeated_calls": 0,
             "zero_lookup_answer": False, "n_turns": 2, "ended": "answer",
             "gen_lens": [20, 10], "hit_gen_cap": [False, False], "traj_tokens": 900,
             "seconds": 1.0}
            for i, (c, t, f) in enumerate(zip(correct, tokens, files))]


def test_paired_diff_and_mcnemar_exact_values():
    a = rows([0] * 20)
    b = rows([1] * 5 + [0] * 15)
    d = paired_diff(a, b)
    assert d["diff"] == pytest.approx(0.25) and d["n"] == 20
    assert d["ci95"][0] > 0 and d["excludes_zero"]
    m = mcnemar_exact(a, b)
    assert (m["n01"], m["n10"]) == (5, 0) and m["p"] == pytest.approx(2 / 32)
    assert mcnemar_exact(a, a)["p"] == 1.0
    # 3 vs 3 discordant -> p = 1
    c = rows([1, 1, 1, 0, 0, 0] + [0] * 14)
    e = rows([0, 0, 0, 1, 1, 1] + [0] * 14)
    assert mcnemar_exact(c, e)["p"] == pytest.approx(1.0)
    assert not paired_diff(c, e)["excludes_zero"]


def test_mcnemar_matches_binomial_tail():
    from math import comb
    a = rows([1] * 2 + [0] * 10 + [1] * 8)
    b = rows([0] * 2 + [1] * 10 + [1] * 8)      # n10=2, n01=10
    p = mcnemar_exact(a, b)["p"]
    assert p == pytest.approx(2 * sum(comb(12, i) for i in range(3)) / 2 ** 12)


def test_relative_change_and_alignment_and_qtype_subset():
    a = rows([1] * 10, tokens=[200] * 10)
    b = list(reversed(rows([1] * 10, tokens=[100] * 10)))   # order must not matter
    r = relative_change(a, b)
    assert r["change"] == pytest.approx(-0.5) and r["ci95"] == pytest.approx([-0.5, -0.5])
    with pytest.raises(ValueError):
        align(a, rows([1] * 9))
    g = paired_diff(rows([0] * 10), rows([1] * 10), qtype="gap")
    assert g["n"] == 2 and g["diff"] == 1.0


def _write(root, d, company, cond, rs, model="Qwen/Qwen2.5-1.5B-Instruct"):
    os.makedirs(os.path.join(root, d), exist_ok=True)
    rec = {"meta": {"model_id": model, "dtype": "torch.float16", "gpu": "Tesla T4",
                    "device": "cuda"},
           "summary": summarize(rs, n_boot=200), "rows": rs}
    with open(os.path.join(root, d, f"{company}__{cond}.json"), "w") as f:
        json.dump(rec, f)


SPEC = {"slug": "m", "runs": {"base": ["b/{slug}/a", "b/{slug}/b"],
                              "notes_l0.2": ["t/main"], "nonotes_l0.2": ["t/nn"]}}


def test_report_fills_what_exists_and_says_not_run_otherwise(tmp_path):
    root = str(tmp_path)
    _write(root, "b/m/a", "company_a", "v3", rows([1] * 8 + [0] * 12, tokens=[400] * 20))
    _write(root, "b/m/a", "company_a", "no_notes",
           rows([1] * 4 + [0] * 16, condition="no_notes"))
    _write(root, "t/main", "company_a", "v3", rows([1] * 14 + [0] * 6, tokens=[200] * 20))
    text = build_report(SPEC, root)
    assert "Model: Qwen/Qwen2.5-1.5B-Instruct" in text
    assert "| base model | 20.0 [" in text and "40.0 [" in text and "70.0 [" in text
    assert "| notes + training − notes only | +30.0 [" in text     # 70 - 40 paired
    assert "| training only − base | not run |" in text            # nonotes run missing
    assert "| tokens read (relative) | -50.0% [-50.0, -50.0] |" in text
    assert NOT_RUN in text.split("### RQ3")[1]                     # no drift runs


def test_report_on_repo_with_no_results_is_all_not_run():
    from learn_to_remember.model_utils import load_config
    spec = load_config(os.path.join(ROOT, "configs", "report.yaml"))
    text = build_report(spec, ROOT)
    assert "Model: not run." in text
    assert text.count(NOT_RUN) > 20
