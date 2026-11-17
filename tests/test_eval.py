"""Eval harness metrics, bootstrap CIs, probe rule and probe subset."""

import json
import os
from types import SimpleNamespace

import pytest

from learn_to_remember.env.episode import EnvConfig, run_text_episode
from learn_to_remember.env.tokens import WhitespaceCounter
from learn_to_remember.env.tools import ToolEnv
from learn_to_remember.eval import (
    bootstrap_ci, episode_row, probe_question_ids, probe_verdict, summarize,
    summary_table, qtype_table,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = [{"path": "files/a.md", "date": "2025-01-01", "title": "Status report: Project Heron",
         "text": "# Status report: Project Heron\nOwner: Maria Okafor\n"}]
NOTES = [{"note_id": "notes/projects/heron", "title": "Project Heron", "as_of": "2025-09-30",
          "text": "# Project Heron\nowner: Maria Okafor\n"}]
CATALOG = {"person": ["Maria Okafor", "Ana Silva"]}


def call(name, **args):
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"


def _q(i, qtype):
    return {"id": f"q{i}", "question": "Who owns Project Heron?", "answer_type": "person",
            "gold": "Maria Okafor", "qtype_by_version": {"v3": qtype}}


def _row(i, qtype, turns, condition="v3"):
    env = ToolEnv(DOCS, None if condition == "no_notes" else NOTES, WhitespaceCounter())
    it = iter(turns)
    res = run_text_episode(lambda m: next(it), env, _q(i, qtype), CATALOG, EnvConfig(),
                           WhitespaceCounter())
    traj = SimpleNamespace(result=res, gen_lens=[10] * len(res.turns),
                           hit_gen_cap=[False] * len(res.turns), ids=[0] * 50)
    return episode_row(traj, _q(i, qtype), condition, seconds=1.0)


NOTE = call("read_note", note_id="notes/projects/heron")
FILE = call("read_file", path="files/a.md")
RIGHT, WRONG = call("answer", text="Maria Okafor"), call("answer", text="Ana Silva")


@pytest.fixture
def rows():
    return [
        _row(0, "lookup", [NOTE, RIGHT]),                 # notes only, correct
        _row(1, "lookup", [NOTE, FILE, RIGHT]),           # false alarm
        _row(2, "gap", [FILE, RIGHT]),                    # fallback
        _row(3, "stale", [NOTE, WRONG]),                  # no fallback, wrong
        _row(4, "multihop", ["garbage", RIGHT]),          # zero-lookup, 1 invalid call
    ]


def test_point_metrics(rows):
    o = summarize(rows, n_boot=200)["overall"]
    assert o["n"] == 5
    assert o["accuracy"]["value"] == pytest.approx(4 / 5)
    assert o["fallback_rate"]["value"] == pytest.approx(1 / 2)     # gap yes, stale no
    assert o["false_alarm_rate"]["value"] == pytest.approx(1 / 2)  # 1 of 2 lookup Qs
    assert o["valid_call_rate"]["value"] == pytest.approx(10 / 11)  # pooled over turns
    assert o["zero_lookup_rate"]["value"] == pytest.approx(1 / 5)
    assert o["lookups_per_q"]["value"] == pytest.approx((1 + 2 + 1 + 1 + 0) / 5)
    assert o["mean_turns"]["value"] == pytest.approx((2 + 3 + 2 + 2 + 2) / 5)
    n_tok = len(NOTES[0]["text"].split())
    f_tok = len(DOCS[0]["text"].split())
    assert o["median_tokens_per_lookup"]["value"] == pytest.approx(
        sorted([n_tok, n_tok, f_tok, f_tok, n_tok])[2])
    assert o["ended"] == {"answer": 5}
    assert o["gen_tokens_per_turn"]["mean"] == 10 and o["turns_hit_gen_cap"] == 0.0


def test_per_qtype_blocks_and_undefined_metrics(rows):
    s = summarize(rows, n_boot=200)
    assert set(s["by_qtype"]) == {"lookup", "multihop", "gap", "stale"}
    assert s["by_qtype"]["lookup"]["fallback_rate"] is None        # only gap/stale
    assert s["by_qtype"]["gap"]["false_alarm_rate"] is None        # only lookup
    assert s["by_qtype"]["multihop"]["median_tokens_per_lookup"] is None  # no lookups
    assert s["by_qtype"]["stale"]["accuracy"]["value"] == 0.0


def test_false_alarm_undefined_without_notes():
    rows = [_row(0, "lookup", [FILE, RIGHT], "no_notes"), _row(1, "gap", [FILE, RIGHT], "no_notes")]
    o = summarize(rows, n_boot=100)["overall"]
    assert o["false_alarm_rate"] is None and o["fallback_rate"]["value"] == 1.0


def test_bootstrap_is_seeded_and_brackets_the_estimate(rows):
    a = bootstrap_ci(rows, "accuracy", "v3", n_boot=500, seed=3)
    b = bootstrap_ci(rows, "accuracy", "v3", n_boot=500, seed=3)
    assert a == b and a[0] <= 0.8 <= a[1] and a[0] < a[1]
    same = [_row(i, "lookup", [NOTE, RIGHT]) for i in range(4)]
    assert bootstrap_ci(same, "accuracy", "v3", n_boot=200) == [1.0, 1.0]


def test_summarize_refuses_mixed_conditions(rows):
    with pytest.raises(AssertionError):
        summarize(rows + [_row(9, "gap", [FILE, RIGHT], "no_notes")])


def test_probe_rule_edges():
    def s(valid, acc):
        return {"overall": {"valid_call_rate": {"value": valid}, "accuracy": {"value": acc}}}
    assert probe_verdict(s(0.80, 0.20))["passes"]
    assert probe_verdict(s(0.95, 0.60))["passes"]
    assert not probe_verdict(s(0.79, 0.40))["passes"]
    v = probe_verdict(s(0.99, 0.61))
    assert not v["passes"] and v["valid_ok"] and not v["accuracy_ok"]
    assert not probe_verdict(s(0.9, 0.19))["passes"]


def test_tables_render(rows):
    s = summarize(rows, n_boot=100)
    t = summary_table([("demo", s)])
    assert "| demo | 5 | 80.0 [" in t
    assert "lookup" in qtype_table([("demo", s)])


def test_probe_subset_is_stratified_deterministic_and_committed():
    qs = [json.loads(l) for l in open(os.path.join(ROOT, "data/eval/company_a_eval.jsonl"))]
    ids = probe_question_ids(qs)
    assert ids == probe_question_ids(qs) and len(ids) == len(set(ids)) == 50
    by = {q["id"]: q["qtype_by_version"]["v3"] for q in qs}
    counts = {}
    for i in ids:
        counts[by[i]] = counts.get(by[i], 0) + 1
    assert set(counts.values()) == {10}
    with open(os.path.join(ROOT, "data/eval/probe_company_a.json")) as f:
        assert json.load(f)["ids"] == ids
