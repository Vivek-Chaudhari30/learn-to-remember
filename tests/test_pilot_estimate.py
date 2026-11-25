"""scripts/pilot_estimate.py: the full-run counts and the scaling from pilot timings."""

import importlib.util
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "pilot_estimate", os.path.join(ROOT, "scripts", "pilot_estimate.py"))
pe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pe)


def test_question_counts_match_eval_files():
    for name, n in (("company_a_eval.jsonl", pe.N_A), ("company_b_eval.jsonl", pe.N_B)):
        with open(os.path.join(ROOT, "data", "eval", name)) as f:
            assert sum(1 for line in f if line.strip()) == n


def test_full_run_counts():
    assert pe.TRAIN_STEPS == 1063
    assert pe.EVAL_QUESTIONS == 32 + 16 * 293 + 4 * 300


def test_estimate_scales_linearly():
    e = pe.estimate_hours(sec_per_step=100.0, sec_per_question=2.0)
    assert abs(e["train_h"] - 1063 * 100 / 3600) < 1e-9
    assert abs(e["eval_h"] - pe.EVAL_QUESTIONS * 2 / 3600) < 1e-9
    assert abs(e["total_h"] - e["train_h"] - e["eval_h"]) < 1e-9


def test_step_times_drop_warmup():
    log = [{"sec_step": 50.0}, {"sec_step": 10.0}, {"sec_step": 12.0}]
    assert pe.step_times(log) == [10.0, 12.0]
    assert pe.step_times(log[:1]) == [50.0]


def test_main_flags_nonfinite_loss(tmp_path, monkeypatch):
    train = tmp_path / "results" / "train" / "main"
    train.mkdir(parents=True)
    (train / "train_log.json").write_text(json.dumps(
        [{"step": 0, "sec_step": 5.0}, {"step": 1, "sec_step": 4.0, "nonfinite_loss": True}]))
    ev = tmp_path / "results" / "baselines" / "m" / "company_a"
    ev.mkdir(parents=True)
    (ev / "timing.json").write_text(json.dumps([{"seconds_per_question": 1.5}]))
    monkeypatch.setattr(pe, "ROOT", str(tmp_path))
    assert pe.main() == 1
