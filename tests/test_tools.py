"""BM25 and the five tools, on a tiny hand-made workspace."""

import math

import pytest

from learn_to_remember.env.bm25 import BM25, tokenize
from learn_to_remember.env.tokens import TRUNC_MARKER, WhitespaceCounter
from learn_to_remember.env.tools import ToolEnv

DOCS = [
    {"path": "files/2024/01/a.md", "date": "2024-01-05", "title": "Status report: Project Heron",
     "text": "# Status report: Project Heron\nDate: 2024-01-05\n\nBudget: $1,000,000\nOwner: Ana Silva\n"},
    {"path": "files/2025/11/b.md", "date": "2025-11-02", "title": "Update: Project Heron",
     "text": "# Update: Project Heron\nDate: 2025-11-02\n\nThe budget for Project Heron has been "
             "revised from $1,000,000 to $1.2M.\n"},
    {"path": "files/2025/03/c.md", "date": "2025-03-01", "title": "Contract summary: C-1589",
     "text": "# Contract summary: C-1589\n\nContract C-1589 with Bluefin Health has been signed.\n"},
    {"path": "files/2025/04/d.md", "date": "2025-04-01", "title": "Long",
     "text": "# Long\n\n" + " ".join(f"word{i}" for i in range(1000)) + "\n"},
]
NOTES = [
    {"note_id": "notes/projects/heron", "title": "Project Heron", "as_of": "2025-09-30",
     "text": "# Project Heron\nnote_id: notes/projects/heron\nas_of: 2025-09-30\nbudget: $1,000,000\n"},
    {"note_id": "notes/people/ana-silva", "title": "Ana Silva", "as_of": "2025-09-30",
     "text": "# Ana Silva\nnote_id: notes/people/ana-silva\nas_of: 2025-09-30\nrole: Designer\n"},
]


@pytest.fixture
def env():
    return ToolEnv(DOCS, NOTES, WhitespaceCounter(), max_output_tokens=50)


@pytest.fixture
def env_nonotes():
    return ToolEnv(DOCS, None, WhitespaceCounter(), max_output_tokens=50)


# --------------------------------------------------------------------------- #
# BM25
# --------------------------------------------------------------------------- #
def test_tokenize_keeps_ids_and_drops_stopwords():
    assert tokenize("What is the value of C-1589?") == ["value", "c-1589"]


def test_bm25_matches_formula_single_term():
    idx = BM25(["apple banana", "banana banana cherry", "cherry"], k1=1.5, b=0.75)
    n, N = 2, 3
    idf = math.log(1 + (N - n + 0.5) / (n + 0.5))
    avgdl = (2 + 3 + 1) / 3
    f, dl = 2, 3
    expected = idf * f * 2.5 / (f + 1.5 * (1 - 0.75 + 0.75 * dl / avgdl))
    assert idx.scores("banana")[1] == pytest.approx(expected)
    assert idx.search("banana", 1)[0][0] == 1


def test_bm25_ties_break_by_index_and_filter_applies():
    idx = BM25(["x y", "x y", "x y"])
    assert [i for i, _ in idx.search("x", 3)] == [0, 1, 2]
    assert [i for i, _ in idx.search("x", 3, allowed=lambda i: i != 0)] == [1, 2]
    assert idx.search("nothing", 3) == []


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
def test_tool_lists_per_condition(env, env_nonotes):
    assert env.tool_names == ["search_notes", "read_note", "search_files", "read_file", "answer"]
    assert env_nonotes.tool_names == ["search_files", "read_file", "answer"]
    r = env_nonotes.call("search_notes", {"query": "Heron"})
    assert not r.ok and r.error == "unknown_tool"


def test_search_notes_format_and_ranking(env):
    r = env.call("search_notes", {"query": "Project Heron budget"})
    lines = r.text.splitlines()
    assert lines[1].startswith("1. notes/projects/heron | Project Heron | as_of 2025-09-30 | ")
    assert "budget: $1,000,000" in lines[1]
    assert r.meta["ids"][0] == "notes/projects/heron"


def test_search_files_shows_dates_and_since_filters(env):
    r = env.call("search_files", {"query": "Project Heron budget", "k": 5})
    assert {"files/2024/01/a.md", "files/2025/11/b.md"} <= set(r.meta["ids"])
    assert "| 2025-11-02 |" in r.text
    r2 = env.call("search_files", {"query": "Project Heron budget", "since": "2025-10-01"})
    assert r2.meta["ids"] == ["files/2025/11/b.md"]
    assert "since 2025-10-01" in r2.text
    bad = env.call("search_files", {"query": "Heron", "since": "last week"})
    assert not bad.ok and bad.error == "bad_args"


def test_contract_ids_are_searchable(env):
    assert env.call("search_files", {"query": "C-1589"}).meta["ids"][0] == "files/2025/03/c.md"


@pytest.mark.parametrize("k,n", [(1, 1), ("2", 2), (0, 1), (99, 4)])
def test_k_is_clamped(env, k, n):
    r = env.call("search_files", {"query": "Heron Contract Long word1 budget", "k": k})
    assert len(r.meta["ids"]) == n


@pytest.mark.parametrize("args", [{}, {"query": ""}, {"query": 5}, {"query": "x", "k": "many"}])
def test_search_bad_args(env, args):
    r = env.call("search_files", args)
    assert not r.ok and r.error == "bad_args" and r.text.startswith("Error:")


def test_no_results_message(env):
    r = env.call("search_notes", {"query": "zzzz"})
    assert r.ok and "No notes match" in r.text and r.meta["ids"] == []


def test_read_note_variants(env):
    assert env.call("read_note", {"note_id": "notes/projects/heron"}).text == NOTES[0]["text"]
    assert env.call("read_note", {"note_id": "/notes/projects/heron.md"}).ok
    r = env.call("read_note", {"note_id": "notes/projects/nope"})
    assert not r.ok and r.error == "not_found" and "search_notes" in r.text


def test_read_file_variants(env):
    assert env.call("read_file", {"path": "files/2025/03/c.md"}).text == DOCS[2]["text"]
    assert env.call("read_file", {"path": "2025/03/c.md"}).ok          # missing files/ prefix
    r = env.call("read_file", {"path": "files/missing.md"})
    assert not r.ok and r.error == "not_found"
    assert env.call("read_file", {}).error == "bad_args"


def test_outputs_are_capped_and_counted(env):
    r = env.call("read_file", {"path": "files/2025/04/d.md"})
    assert r.truncated and r.text.endswith(TRUNC_MARKER)
    assert r.tokens <= 50 and r.tokens == WhitespaceCounter().count(r.text)
    small = env.call("read_note", {"note_id": "notes/people/ana-silva"})
    assert not small.truncated and small.tokens == WhitespaceCounter().count(small.text)


def test_errors_are_counted_too(env):
    r = env.call("read_file", {"path": "nope"})
    assert r.tokens == WhitespaceCounter().count(r.text) > 0


def test_answer_tool(env):
    r = env.call("answer", {"text": "  Ana Silva "})
    assert r.ok and r.meta["answer"] == "Ana Silva" and r.tokens == 0
    for bad in ({}, {"text": ""}, {"text": 3}):
        assert env.call("answer", bad).error == "bad_args"
