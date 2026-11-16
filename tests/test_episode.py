"""Episode runner: turn limit, token accounting, error handling, reward, behavior."""

import json

import pytest

from learn_to_remember.env.episode import Episode, EnvConfig, behavior, run_text_episode
from learn_to_remember.env.tokens import WhitespaceCounter
from learn_to_remember.env.tools import ToolEnv

DOCS = [{"path": "files/a.md", "date": "2025-01-01", "title": "Status report: Project Heron",
         "text": "# Status report: Project Heron\n\nOwner: Maria Okafor\nBudget: $1,000,000\n"}]
NOTES = [{"note_id": "notes/projects/heron", "title": "Project Heron", "as_of": "2025-09-30",
          "text": "# Project Heron\nowner: Maria Okafor\n"}]
CATALOG = {"person": ["Maria Okafor", "Ana Silva"]}
Q = {"id": "q1", "question": "Who owns Project Heron?", "answer_type": "person",
     "gold": "Maria Okafor", "qtype_by_version": {"v3": "lookup"}}


def call(name, **args):
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"


@pytest.fixture
def env():
    return ToolEnv(DOCS, NOTES, WhitespaceCounter(), max_output_tokens=400)


def scripted(turns):
    it = iter(turns)
    return lambda messages: next(it)


def run(env, turns, **cfg):
    return run_text_episode(scripted(turns), env, Q, CATALOG, EnvConfig(**cfg),
                            WhitespaceCounter())


def test_correct_answer_after_one_lookup(env):
    r = run(env, [call("read_note", note_id="notes/projects/heron"),
                  call("answer", text="Maria Okafor")])
    assert r.correct and r.valid_answer and r.ended == "answer"
    assert r.tokens_read == WhitespaceCounter().count(NOTES[0]["text"])
    assert r.reward == pytest.approx(1.1 - 0.2 * r.tokens_read / 4000)


def test_wrong_answer_still_gets_format_bonus(env):
    r = run(env, [call("answer", text="Ana Silva")])
    assert not r.correct and r.valid_answer and r.reward == pytest.approx(0.1)


def test_turn_limit_counts_answer_and_skips_last_non_answer(env):
    q = call("search_files", query="Heron")
    r = run(env, [q] * 6, max_turns=6)
    assert r.ended == "turn_limit" and r.answer is None and not r.valid_answer
    assert len(r.turns) == 6
    assert [t["executed"] for t in r.turns] == [True] * 5 + [False]
    assert r.tokens_read == sum(t["tokens"] for t in r.turns[:5])
    assert r.turns[5]["tokens"] == 0
    assert r.reward == pytest.approx(-0.2 * r.tokens_read / 4000)


def test_answer_on_last_turn_counts(env):
    q = call("search_files", query="Heron")
    r = run(env, [q] * 5 + [call("answer", text="Maria Okafor")], max_turns=6)
    assert r.correct and r.ended == "answer"


def test_malformed_call_uses_a_turn_and_costs_tokens(env):
    r = run(env, ["I think it is Maria Okafor.", call("answer", text="Maria Okafor")])
    assert r.turns[0]["parse_error"] == "no_tool_call"
    assert r.turns[0]["tokens"] > 0 and r.tokens_read == r.turns[0]["tokens"]
    assert r.correct and len(r.turns) == 2


def test_invalid_answer_does_not_end_episode(env):
    r = run(env, [call("answer"), call("answer", text="Maria Okafor")])
    assert r.turns[0]["tool_error"] == "bad_args" and r.correct


def test_no_notes_env_rejects_notes_tools():
    env = ToolEnv(DOCS, None, WhitespaceCounter())
    r = run(env, [call("read_note", note_id="notes/projects/heron"),
                  call("answer", text="Maria Okafor")])
    assert r.turns[0]["tool_error"] == "unknown_tool"


def test_step_after_done_raises(env):
    ep = Episode(env, Q, CATALOG, EnvConfig())
    ep.step(call("answer", text="x"))
    with pytest.raises(RuntimeError):
        ep.step(call("answer", text="y"))


def test_context_limit_ends_episode(env):
    r = run(env, [call("search_files", query="Heron")] * 6, max_context_tokens=300,
            max_gen_tokens=256)
    assert r.ended == "context_limit" and len(r.turns) == 1


def test_behavior_metrics(env):
    s = call("search_files", query="Heron")
    r = run(env, [s, s, "garbage", call("answer", text="Maria Okafor")])
    b = behavior(r)
    assert b["n_lookups"] == 2 and b["used_files"] and not b["used_notes"]
    assert b["repeated_calls"] == 1 and b["valid_calls"] == 3 and b["n_calls_total"] == 4
    assert not b["zero_lookup_answer"]
    z = behavior(run(env, [call("answer", text="Maria Okafor")]))
    assert z["zero_lookup_answer"] and z["n_lookups"] == 0
