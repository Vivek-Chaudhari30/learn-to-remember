"""Invariants of the synthetic workspace (PROJECT.md §3.1), checked before RL
touches anything:
  - the same seed gives byte-identical output, across processes too;
  - every gold answer can be derived from the raw documents;
  - every gap entity has no note page (or the page omits the field);
  - every stale note disagrees with the latest raw value;
  - train and eval questions don't overlap;
  - held-out entities never appear in training questions.
"""

import os
import re
import subprocess
import sys
from datetime import date

import pytest

from learn_to_remember.workspace.entities import END, MIN_GAP_DAYS, START
from learn_to_remember.workspace.formats import (
    N_DATE_STYLES, N_MONEY_STYLES, parse_date, parse_money, render_date, render_money,
)
from learn_to_remember.workspace.generate import generate_company, manifest, serialize
from learn_to_remember.workspace.notes import VERSIONS, build_notes
from learn_to_remember.workspace.questions import QTYPES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def ws():
    return generate_company(0, "company_a")


@pytest.fixture(scope="module")
def ws_b():
    return generate_company(1, "company_b", with_train=False)


@pytest.fixture(scope="module")
def raw_index(ws):
    """(eid, attr) -> [(date, value, path)] from document assertions only."""
    idx = {}
    for d in ws.docs:
        for eid, attr, value, _ in d.asserts:
            idx.setdefault((eid, attr), []).append((d.date, value, d.path))
    return idx


def latest_raw(raw_index, eid, attr):
    entries = raw_index[(eid, attr)]
    last = max(e[0] for e in entries)
    vals = {repr(e[1]) for e in entries if e[0] == last}
    assert len(vals) == 1, f"conflicting raw values for {eid}.{attr} on {last}"
    return next(e[1] for e in entries if e[0] == last)


def all_questions(ws):
    return ws.splits["eval"] + ws.splits["train"]


# --------------------------------------------------------------------------- #
# Formats
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("v", [100_000, 350_000, 1_000_000, 1_250_000, 4_160_000, 12_340_000])
def test_money_render_parse_round_trip(v):
    for s in range(N_MONEY_STYLES):
        assert parse_money(render_money(v, s)) == v, render_money(v, s)


@pytest.mark.parametrize("d", [date(2024, 1, 1), date(2025, 3, 14), date(2025, 12, 31)])
def test_date_render_parse_round_trip(d):
    for s in range(N_DATE_STYLES):
        assert parse_date(render_date(d, s)) == d, render_date(d, s)


def test_ambiguous_numeric_dates_rejected():
    assert parse_date("03/04/2025") is None


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_same_seed_byte_identical_in_process(ws):
    again = generate_company(0, "company_a")
    assert serialize(again) == serialize(ws)


def test_same_seed_byte_identical_across_processes(ws):
    """Different PYTHONHASHSEEDs would expose any reliance on hash() or on
    unsorted set iteration."""
    expected = manifest(ws)["sha256"]
    code = ("import json; from learn_to_remember.workspace.generate import "
            "generate_company, manifest; "
            "print(json.dumps(manifest(generate_company(0, 'company_a'))['sha256']))")
    for hs in ("0", "4242"):
        env = dict(os.environ, PYTHONHASHSEED=hs,
                   PYTHONPATH=os.path.join(ROOT, "src"))
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                             text=True, check=True).stdout.strip().splitlines()[-1]
        import json
        assert json.loads(out) == expected, f"PYTHONHASHSEED={hs}"


def test_different_seed_gives_different_company(ws, ws_b):
    names_a = {e.name for e in ws.world.entities.values() if e.kind == "person"}
    names_b = {e.name for e in ws_b.world.entities.values() if e.kind == "person"}
    assert names_a != names_b


# --------------------------------------------------------------------------- #
# Size (spec §3.1)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fixture", ["ws", "ws_b"])
def test_sizes_within_spec(fixture, request):
    w = request.getfixturevalue(fixture)
    assert 150 <= len(w.world.entities) <= 250
    assert 400 <= len(w.docs) <= 800
    for d in w.docs:
        n = len(d.text.split())
        assert 100 <= n <= 330, (d.path, n)


# --------------------------------------------------------------------------- #
# Raw documents are the source of truth
# --------------------------------------------------------------------------- #
def test_every_assertion_matches_timeline_and_text(ws):
    w = ws.world
    for d in ws.docs:
        for eid, attr, value, rendered in d.asserts:
            assert w.value_at(eid, attr, d.date) == value, (d.path, eid, attr)
            assert rendered in d.text, (d.path, rendered)
            # the entity itself is named in the doc (contracts by id)
            assert w.entities[eid].name in d.text, (d.path, eid)


def test_chatter_asserts_nothing(ws):
    for d in ws.docs:
        if d.dtype == "chatter":
            assert not d.asserts and not d.lists


def test_final_value_of_every_fact_is_asserted_after_its_last_change(ws, raw_index):
    w = ws.world
    for q in all_questions(ws):
        if q["structure"] == "aggregate":
            continue
        for eid, attr in q["facts"]:
            entries = raw_index.get((eid, attr))
            assert entries, f"{eid}.{attr} never asserted"
            assert max(e[0] for e in entries) >= w.last_change(eid, attr)
            assert latest_raw(raw_index, eid, attr) == w.latest(eid, attr)


def test_every_gold_answer_derivable_from_raw_docs(ws, raw_index):
    """Re-derive each gold answer using ONLY document assertions: the latest
    doc per fact for single/multihop, and the latest year-end list for
    aggregates (with each item's latest raw status)."""
    lists = {}
    for d in ws.docs:
        for anchor, name, ids in d.lists:
            lists[(anchor, name)] = (d.date, ids)     # docs are date-sorted
    from learn_to_remember.workspace.questions import AGGREGATE

    for q in all_questions(ws):
        if q["structure"] == "aggregate":
            kind, lname, status, _ = AGGREGATE[q["family"]]
            if lname == "teams":                      # teams never move offices
                n = sum(1 for (t, a) in raw_index if a == "office" and t.startswith("T")
                        and latest_raw(raw_index, t, "office") == q["anchor"])
            else:
                _, ids = lists[(q["anchor"], lname)]
                if status:
                    ids = [x for x in ids if latest_raw(raw_index, x, "status") == status]
                n = len(ids)
            assert n == q["gold_value"], q["id"]
        else:
            cur = None
            for eid, attr in q["facts"]:
                eid = eid if cur is None else cur
                cur = latest_raw(raw_index, eid, attr)
            assert cur == q["gold_value"], q["id"]


def test_no_fact_changes_twice_within_min_gap(ws):
    for (eid, attr), tl in ws.world.timeline.items():
        for (d1, _), (d2, _) in zip(tl, tl[1:]):
            assert (d2 - d1).days >= MIN_GAP_DAYS, (eid, attr, d1, d2)


def test_all_dates_inside_timeline(ws):
    for d in ws.docs:
        assert START <= d.date <= END


# --------------------------------------------------------------------------- #
# Notes
# --------------------------------------------------------------------------- #
def _pages(ws, v):
    return {p["entity"]: p for p in ws.notes[v] if p["entity"]}


def test_gap_questions_really_lack_the_note(ws):
    n_checked = 0
    for q in all_questions(ws):
        if q["structure"] != "single":
            continue
        eid, attr = q["facts"][0]
        for v in VERSIONS:
            pages = _pages(ws, v)
            kind = q["gap_kind_by_version"][v]
            if q["qtype_by_version"][v] != "gap":
                assert kind is None
                continue
            n_checked += 1
            if kind == "missing_page":
                assert eid not in pages, (q["id"], v)
                assert not any(p["entity"] == eid for p in ws.notes[v]), (q["id"], v)
            else:
                assert kind == "missing_field"
                assert attr not in pages[eid]["fields"], (q["id"], v)
                assert f"\n{attr}:" not in pages[eid]["text"], (q["id"], v)
    assert n_checked > 0


def test_stale_notes_disagree_with_latest_raw_value(ws, raw_index):
    n_checked = 0
    for q in all_questions(ws):
        if q["structure"] != "single":
            continue
        eid, attr = q["facts"][0]
        for v, ver in VERSIONS.items():
            page = _pages(ws, v).get(eid)
            label = q["qtype_by_version"][v]
            if label == "stale":
                n_checked += 1
                assert page["fields"][attr] != latest_raw(raw_index, eid, attr), (q["id"], v)
                assert page["fields"][attr] == ws.world.value_at(eid, attr, ver.snapshot)
            elif label == "lookup":
                assert page["fields"][attr] == latest_raw(raw_index, eid, attr), (q["id"], v)
    assert n_checked > 0


def test_coverage_is_nested_and_grows(ws):
    prev = None
    for v in VERSIONS:
        cur = set(_pages(ws, v))
        if prev is not None:
            assert prev <= cur, v
            assert len(cur) > len(prev)
        prev = cur


def test_notes_as_of_and_sources_respect_snapshot(ws):
    paths = {d.path: d.date for d in ws.docs}
    for v, ver in VERSIONS.items():
        for p in ws.notes[v]:
            assert p["as_of"] == ver.snapshot.isoformat()
            assert f"as_of: {ver.snapshot.isoformat()}" in p["text"]
            for s in p["sources"]:
                assert paths[s] <= ver.snapshot
            if p["entity"]:
                assert ws.world.exists_at(p["entity"], ver.snapshot)


def test_v3_is_imperfect_on_purpose(ws):
    """Approved deviation: the main run trains on v3, so v3 must still have
    gap and stale questions (docs/DECISIONS.md)."""
    labels = [q["qtype_by_version"]["v3"] for q in ws.splits["train"]]
    assert labels.count("gap") > 0 and labels.count("stale") > 0


def test_prose_recipe_builds_with_same_fields(ws):
    prose = build_notes(ws.world, ws.docs, VERSIONS["v2"], recipe="prose")
    struct = ws.notes["v2"]
    assert [p["note_id"] for p in prose] == [p["note_id"] for p in struct]
    assert [p["fields"] for p in prose] == [p["fields"] for p in struct]
    assert all("\nowner:" not in p["text"] for p in prose)


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def test_train_and_eval_do_not_overlap(ws):
    tr, ev = ws.splits["train"], ws.splits["eval"]
    assert not {q["question"] for q in tr} & {q["question"] for q in ev}
    key = lambda q: (q["family"], q["anchor"])  # noqa: E731
    assert not {key(q) for q in tr} & {key(q) for q in ev}
    facts = lambda qs: {tuple(f) for q in qs for f in q["facts"]}  # noqa: E731
    assert not facts(tr) & facts(ev)


def test_heldout_entities_never_in_training_questions(ws):
    w = ws.world
    held = set(ws.splits["heldout"])
    assert held
    pats = [re.compile(r"(?<![\w-])" + re.escape(w.display(e) if w.entities[e].kind == "project"
                                                  else w.entities[e].name) + r"(?![\w-])")
            for e in sorted(held)]
    for q in ws.splits["train"]:
        assert not held & set(q["entities"]), q["id"]
        assert not q["heldout"]
        for p in pats:
            assert not p.search(q["question"]), (q["id"], p.pattern)
            assert not p.search(q["gold"]), (q["id"], p.pattern)
    assert any(q["heldout"] for q in ws.splits["eval"])


def test_eval_balance(ws, ws_b):
    for split, floor in ((ws.splits["eval"], 40), (ws_b.splits["eval"], 60)):
        counts = {t: sum(q["qtype"] == t for q in split) for t in QTYPES}
        assert all(c >= floor for c in counts.values()), counts
        assert all(c <= 60 for c in counts.values()), counts
    assert ws_b.splits["train"] == []


def test_question_ids_unique_and_fields_present(ws):
    ids = [q["id"] for q in all_questions(ws)]
    assert len(ids) == len(set(ids))
    for q in all_questions(ws):
        assert set(q["qtype_by_version"]) == set(VERSIONS)
        assert q["qtype"] == q["qtype_by_version"]["v3"]
        assert q["gold"]


# --------------------------------------------------------------------------- #
# Committed artifacts match regeneration
# --------------------------------------------------------------------------- #
def test_committed_eval_sets_match_regeneration(ws, ws_b):
    import hashlib
    import json
    path = os.path.join(ROOT, "data", "eval", "manifest.json")
    if not os.path.exists(path):
        pytest.skip("run scripts/generate_workspace.py first")
    committed = json.load(open(path))
    for w in (ws, ws_b):
        assert committed[w.company]["sha256"] == manifest(w)["sha256"], w.company
        text = open(os.path.join(ROOT, "data", "eval", f"{w.company}_eval.jsonl")).read()
        assert (hashlib.sha256(text.encode()).hexdigest()
                == committed[w.company]["sha256"]["questions_eval.jsonl"])
