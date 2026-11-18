"""M6 figure data (analysis.py), the figures (figures.py) and the analysis
notebook, on synthetic result files written in the eval_runner / train_loop
formats. Synthetic numbers here test plumbing only; they are never results."""

import json
import os
import sys
import types

import numpy as np
import pytest

from learn_to_remember import analysis as an
from learn_to_remember.eval import summarize
from learn_to_remember.figures import make_all
from learn_to_remember.report import Runs
from learn_to_remember.workspace.questions import QTYPES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPEC = {"slug": "m",
        "runs": {"base": ["b/a", "b/b", "b/drift"], "notes_l0.2": ["t/main"],
                 "nonotes_l0.2": ["t/nn"], "notes_l0": ["t/l0"], "notes_l0.5": ["t/l05"],
                 "v0_l0.2": ["t/v0"]},
        "train_logs": {"notes_l0.2": "tr/main", "notes_l0": "tr/l0", "notes_l0.5": "tr/l05",
                       "overfit": "tr/of"}}


def make_rows(n, p_correct, condition, seed, lookups=(1, 2), p_files=0.3, tokens=300):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        nl = int(rng.choice(lookups))
        out.append({"id": f"q{i:03d}", "qtype": QTYPES[i % 5], "condition": condition,
                    "correct": bool(rng.random() < p_correct), "tokens_read": tokens * nl,
                    "n_lookups": nl, "reward": 0.0, "tokens_per_lookup": [tokens] * nl,
                    "used_files": bool(rng.random() < p_files), "used_notes": True,
                    "valid_calls": nl + 1, "n_calls_total": nl + 1, "repeated_calls": 0,
                    "zero_lookup_answer": nl == 0, "n_turns": nl + 1, "ended": "answer",
                    "gen_lens": [20] * (nl + 1), "hit_gen_cap": [False] * (nl + 1),
                    "traj_tokens": 900, "seconds": 1.0})
    return out


def write_eval(root, d, company, cond, rows):
    os.makedirs(os.path.join(root, d), exist_ok=True)
    rec = {"meta": {"model_id": "Qwen/Qwen2.5-1.5B-Instruct", "dtype": "torch.float16",
                    "gpu": "Tesla T4", "device": "cuda"},
           "summary": summarize(rows, n_boot=100), "rows": rows}
    with open(os.path.join(root, d, f"{company}__{cond}.json"), "w") as f:
        json.dump(rec, f)


def traj(row):
    return {"id": row["id"], "question": f"question {row['id']}?", "gold": "x",
            "qtype": row["qtype"], "answer": "x" if row["correct"] else "y",
            "correct": row["correct"], "reward": 0.0, "ended": "answer",
            "tokens_read": row["tokens_read"],
            "turns": [{"assistant": "<tool_call>...</tool_call>", "tool": "read_note",
                       "args": {}, "error": None, "observation": "o" * 500, "tokens": 120},
                      {"assistant": "<answer>x</answer>", "tool": None, "args": None,
                       "error": None, "observation": None, "tokens": 0}],
            "gen_lens": [20, 10]}


def write_rollouts(root, d, company, cond, rows):
    os.makedirs(os.path.join(root, d, "rollouts"), exist_ok=True)
    with open(os.path.join(root, d, "rollouts", f"{company}__{cond}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(traj(r)) + "\n")


def write_train(root, d, steps, seed):
    rng = np.random.default_rng(seed)
    hist = [{"step": s, "reward_mean": 0.3 + s / steps * 0.3 + 0.05 * rng.standard_normal(),
             "frac_correct": 0.4 + s / steps * 0.2, "tokens_read_mean": 900 - 3 * s,
             "valid_call_rate": 0.9, "zero_lookup_rate": 0.02, "frac_traj_with_repeats": 0.1,
             "kl": 0.001 * s, "entropy": 0.5, "frac_zero_var_groups": 0.2,
             "question_ids": ["q000"]} for s in range(steps)]
    os.makedirs(os.path.join(root, d), exist_ok=True)
    with open(os.path.join(root, d, "train_log.json"), "w") as f:
        json.dump(hist, f)


@pytest.fixture(scope="module")
def full_root(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("results"))
    A, B = "company_a", "company_b"
    base_v3 = make_rows(50, 0.4, "v3", 1)
    write_eval(root, "b/a", A, "v3", base_v3)
    write_rollouts(root, "b/a", A, "v3", base_v3)
    write_eval(root, "b/a", A, "no_notes", make_rows(50, 0.2, "no_notes", 2, lookups=(0, 1, 2)))
    write_eval(root, "b/b", B, "v3", make_rows(50, 0.35, "v3", 3))
    for k, v in enumerate(("v0", "v1", "v2")):
        write_eval(root, "b/drift", A, v, make_rows(50, 0.2 + 0.05 * k, v, 10 + k))
    main_v3 = make_rows(50, 0.6, "v3", 4, lookups=(0, 1, 1, 5), p_files=0.5)
    write_eval(root, "t/main", A, "v3", main_v3)
    write_rollouts(root, "t/main", A, "v3", main_v3)
    for k, v in enumerate(("v0", "v1", "v2")):
        write_eval(root, "t/main", A, v, make_rows(50, 0.3 + 0.05 * k, v, 20 + k))
    write_eval(root, "t/main", B, "v3", make_rows(50, 0.45, "v3", 5))
    write_eval(root, "t/nn", A, "no_notes", make_rows(50, 0.3, "no_notes", 6))
    write_eval(root, "t/l0", A, "v3", make_rows(50, 0.62, "v3", 7, tokens=500))
    write_eval(root, "t/l05", A, "v3", make_rows(50, 0.5, "v3", 8, tokens=150))
    for v in an.VERSIONS:
        write_eval(root, "t/v0", A, v, make_rows(50, 0.45, v, 30 + an.VERSIONS.index(v)))
    write_eval(root, "t/v0", B, "v3", make_rows(50, 0.4, "v3", 9))
    write_train(root, "tr/main", 60, 0)
    write_train(root, "tr/l0", 60, 1)
    write_train(root, "tr/l05", 60, 2)
    write_train(root, "tr/of", 20, 3)
    return root


# --------------------------------------------------------------------------- #
# analysis.py
# --------------------------------------------------------------------------- #
def test_rolling_and_mean_ci():
    assert an.rolling([1, 2, 3, 4], 2) == [1.0, 1.5, 2.5, 3.5]
    assert an.rolling([None, 2, None, 4], 2) == [None, 2.0, 2.0, 3.0]
    assert an.mean_ci([]) is None
    assert an.mean_ci([1, 1, 1]) == (1.0, 1.0, 1.0)
    v, lo, hi = an.mean_ci([0, 1] * 50)
    assert v == 0.5 and lo < 0.5 < hi


def test_rq_data_matches_the_stored_summaries(full_root):
    runs = Runs(SPEC, full_root)
    d = an.rq1_data(runs)
    base_nn = runs.get("base", "company_a", "no_notes")["summary"]["overall"]["accuracy"]
    assert d["series"]["base"][0] == (base_nn["value"], *base_nn["ci95"])
    assert d["missing"] == []
    pts = {p["label"]: p for p in an.rq2_data(runs)["points"]}
    assert set(pts) == {"base", "λ=0", "λ=0.2", "λ=0.5"}
    assert pts["λ=0.5"]["tokens"][0] < pts["λ=0"]["tokens"][0]
    r3 = an.rq3_data(runs)
    assert list(r3["series"]) == ["base", "trained on v0", "trained on v3"]
    assert all(p is not None for s in r3["series"].values() for p in s)
    assert [b[0] for b in an.rq4_data(runs)["bars"]] == ["base", "trained on v3", "trained on v0"]


def test_behavior_rates_and_histogram_are_exact(full_root):
    runs = Runs(SPEC, full_root)
    rows = runs.get("notes_l0.2", "company_a", "v3")["rows"]
    beh = an.behavior_by_qtype(runs)
    for i, t in enumerate(QTYPES):
        sel = [r for r in rows if r["qtype"] == t]
        assert beh["used_files"]["trained"][i][0] == pytest.approx(
            np.mean([r["used_files"] for r in sel]))
        assert beh["accuracy"]["trained"][i][0] == pytest.approx(
            np.mean([r["correct"] for r in sel]))
    h = an.lookup_histogram(runs)
    assert h["bins"] == ["0", "1", "2", "3", "4+"]
    tr = h["series"]["trained"]
    assert sum(tr) == pytest.approx(1.0)
    assert tr[4] == pytest.approx(np.mean([r["n_lookups"] >= 4 for r in rows]))
    assert tr[0] == pytest.approx(np.mean([r["n_lookups"] == 0 for r in rows]))


def test_missing_runs_are_listed_not_filled(tmp_path):
    root = str(tmp_path)
    write_eval(root, "b/a", "company_a", "v3", make_rows(20, 0.4, "v3", 1))
    runs = Runs(SPEC, root)
    d = an.rq1_data(runs)
    assert d["series"]["base"][0] is None and d["series"]["base"][1] is not None
    assert d["series"]["trained"] == [None, None]
    assert set(d["missing"]) == {"base, no notes", "trained, no notes", "trained, notes v3"}
    assert [p["label"] for p in an.rq2_data(runs)["points"]] == ["base"]
    assert an.behavior_by_qtype(runs)["missing"] == ["trained"]
    avail = an.availability(SPEC, root)
    assert sum(e for *_, e in avail) == 1


def test_train_curves(full_root):
    t = an.train_runs(SPEC, full_root, window=5)
    c = t["notes_l0.2"]
    assert c["step"] == list(range(60))
    assert set(c["series"]) == {k for k, _ in an.CURVE_KEYS}
    s = c["series"]["tokens_read_mean"]
    assert s["smooth"][10] == pytest.approx(np.mean(s["raw"][6:11]))
    assert an.train_runs({"train_logs": {"x": "nope"}}, full_root) == {"x": None}


def test_paired_examples_are_fixed_and_use_saved_trajectories(full_root):
    ex = an.paired_examples(SPEC, full_root, k=2)
    runs = Runs(SPEC, full_root)
    a = {r["id"]: r for r in runs.get("base", "company_a", "v3")["rows"]}
    b = {r["id"]: r for r in runs.get("notes_l0.2", "company_a", "v3")["rows"]}
    assert ex["n_wins"] == sum(b[i]["correct"] and not a[i]["correct"] for i in a)
    assert ex["n_losses"] == sum(a[i]["correct"] and not b[i]["correct"] for i in a)
    for w in ex["wins"]:
        assert w["row_b"]["correct"] and not w["row_a"]["correct"]
        assert w["base"]["id"] == w["id"] == w["notes_l0.2"]["id"]
    assert an.paired_examples(SPEC, full_root, k=2) == ex          # deterministic
    text = an.format_trajectory(ex["wins"][0]["base"], max_obs=50)
    assert "Q: question" in text and "[500 chars]" in text
    assert an.format_trajectory(None) == "(trajectory not saved)"
    none = an.paired_examples(SPEC, full_root, run_b="nonotes_l0.2")
    assert none["wins"] == [] and none["missing"] == ["nonotes_l0.2"]


# --------------------------------------------------------------------------- #
# figures.py
# --------------------------------------------------------------------------- #
EXPECTED = {"train_notes_l0.2", "train_notes_l0", "train_notes_l0.5", "train_overfit",
            "train_lambda_sweep", "rq1_notes_x_training", "rq2_cost_tradeoff", "rq3_drift",
            "rq4_transfer", "rq5_behavior_by_qtype", "rq5_lookups"}


def test_make_all_draws_every_figure_with_full_results(full_root, tmp_path):
    res = make_all(SPEC, full_root, str(tmp_path / "fig"), dpi=60)
    assert set(res["written"]) == EXPECTED and res["not_run"] == []
    for p in res["written"].values():
        assert os.path.getsize(p) > 5000
    idx = (tmp_path / "fig" / "INDEX.md").read_text()
    assert "[rq3_drift](rq3_drift.png)" in idx and "Not run" not in idx


def test_make_all_with_no_results_draws_nothing(tmp_path):
    res = make_all(SPEC, str(tmp_path), str(tmp_path / "fig"))
    assert res["written"] == {} and set(res["not_run"]) == EXPECTED
    idx = (tmp_path / "fig" / "INDEX.md").read_text()
    assert "No figure has data yet." in idx and "`rq1_notes_x_training`" in idx
    assert not [f for f in os.listdir(tmp_path / "fig") if f.endswith(".png")]


def test_make_all_with_partial_results(tmp_path):
    root = str(tmp_path)
    write_eval(root, "b/a", "company_a", "v3", make_rows(20, 0.4, "v3", 1))
    res = make_all(SPEC, root, str(tmp_path / "fig"), dpi=60)
    assert set(res["written"]) == {"rq1_notes_x_training", "rq2_cost_tradeoff", "rq3_drift",
                                   "rq5_behavior_by_qtype", "rq5_lookups"}
    assert "rq4_transfer" in res["not_run"] and "train_lambda_sweep" in res["not_run"]


def test_repo_spec_lists_every_training_run():
    from learn_to_remember.model_utils import load_config
    spec = load_config(os.path.join(ROOT, "configs", "report.yaml"))
    assert set(spec["train_logs"]) == {"overfit", "notes_l0.2", "nonotes_l0.2", "notes_l0",
                                       "notes_l0.5", "v0_l0.2"}
    assert set(spec["train_logs"]) - {"overfit"} <= set(spec["runs"])


# --------------------------------------------------------------------------- #
# notebooks/analysis.ipynb
# --------------------------------------------------------------------------- #
def test_analysis_notebook_runs_on_synthetic_results(full_root, tmp_path, monkeypatch):
    with open(os.path.join(ROOT, "notebooks", "analysis.ipynb")) as f:
        nb = json.load(f)
    code = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]
    assert code and all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(SPEC))
    monkeypatch.setenv("LTR_RESULTS_ROOT", full_root)
    monkeypatch.setenv("LTR_SPEC", str(spec_path))
    monkeypatch.setenv("LTR_FIG_DIR", str(tmp_path / "fig"))
    shown = []
    disp = types.ModuleType("IPython.display")
    disp.display = lambda *a, **k: shown.append(a)
    disp.Image = lambda filename=None, **k: ("img", filename)
    disp.Markdown = lambda s: ("md", s)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.display", disp)
    ns = {"__name__": "__main__"}
    for src in code:
        exec(compile(src, "analysis.ipynb", "exec"), ns)
    imgs = [a[0][1] for a in shown if isinstance(a[0], tuple) and a[0][0] == "img"]
    assert {os.path.basename(p)[:-4] for p in imgs} == EXPECTED
    assert any(isinstance(a[0], tuple) and a[0][0] == "md" and "RQ1" in a[0][1] for a in shown)
