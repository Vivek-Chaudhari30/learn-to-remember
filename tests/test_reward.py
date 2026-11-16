"""Answer normalizer + reward. Forgiving about surface form, strict about content."""

import pytest

from learn_to_remember.reward import compute_reward, is_correct

CATALOG = {
    "person": ["Maria Okafor", "Maria Silva", "Ana Silva"],
    "team": ["Data Platform", "Data Products", "Payments"],
    "office": ["Lisbon", "Berlin"],
    "client": ["Bluefin Health", "Bluefin Capital"],
    "project": ["Heron", "Night Heron", "Kestrel"],
    "city": ["Lisbon", "Berlin", "Toronto"],
    "role": ["Software Engineer", "Senior Software Engineer", "Product Manager"],
}


def Q(atype, gold):
    return {"answer_type": atype, "gold": gold}


@pytest.mark.parametrize("ans", ["Maria Okafor", "maria okafor", "Maria Okafor.", "It's Maria Okafor",
                                 "  MARIA   OKAFOR "])
def test_person_surface_forms(ans):
    assert is_correct(ans, Q("person", "Maria Okafor"), CATALOG)


@pytest.mark.parametrize("ans", ["Maria", "Okafor", "Maria Okafor or Ana Silva",
                                 "Maria Silva", "Ana Silva, Maria Okafor", ""])
def test_person_rejects_partial_and_hedged(ans):
    assert not is_correct(ans, Q("person", "Maria Okafor"), CATALOG)


def test_sibling_codenames():
    assert is_correct("Project Heron", Q("project", "Heron"), CATALOG)
    assert not is_correct("Project Night Heron", Q("project", "Heron"), CATALOG)
    assert is_correct("Night Heron", Q("project", "Night Heron"), CATALOG)
    assert not is_correct("Heron", Q("project", "Night Heron"), CATALOG)


def test_nested_roles_and_aliases():
    assert is_correct("Software Engineer", Q("role", "Software Engineer"), CATALOG)
    assert not is_correct("Senior Software Engineer", Q("role", "Software Engineer"), CATALOG)
    assert is_correct("the Lisbon office", Q("office", "Lisbon"), CATALOG)
    assert is_correct("Data Platform team", Q("team", "Data Platform"), CATALOG)
    assert not is_correct("Data Products", Q("team", "Data Platform"), CATALOG)
    assert not is_correct("Bluefin", Q("client", "Bluefin Health"), CATALOG)


@pytest.mark.parametrize("ans", ["$1,250,000", "$1.25M", "1,250,000 USD", "$1.25 million",
                                 "1.25 million", "The budget is $1,250,000.", "1250000", "$1,250,000.00"])
def test_money_forms(ans):
    assert is_correct(ans, Q("money", "$1,250,000"), CATALOG)


@pytest.mark.parametrize("ans", ["$1.2M", "$1,250,000 or $1.3M", "about a million", "1.25"])
def test_money_rejects(ans):
    assert not is_correct(ans, Q("money", "$1,250,000"), CATALOG)


@pytest.mark.parametrize("ans", ["2025-12-09", "December 9, 2025", "9 Dec 2025", "Dec 9, 2025",
                                 "It ends on December 9th, 2025."])
def test_date_forms(ans):
    assert is_correct(ans, Q("date", "2025-12-09"), CATALOG)


@pytest.mark.parametrize("ans", ["2025-12-10", "December 2025", "12/09/2025",
                                 "2025-12-09 or 2026-01-15"])
def test_date_rejects(ans):
    assert not is_correct(ans, Q("date", "2025-12-09"), CATALOG)


@pytest.mark.parametrize("ans,ok", [("7", True), ("seven", True), ("7 projects", True),
                                    ("There are 7.", True), ("6", False), ("6 or 7", False),
                                    ("none", False)])
def test_counts(ans, ok):
    assert is_correct(ans, Q("count", "7"), CATALOG) is ok


@pytest.mark.parametrize("ans,gold,ok", [
    ("on hold", "on hold", True), ("On-hold", "on hold", True), ("paused", "on hold", True),
    ("It is currently active", "active", True), ("in progress", "active", True),
    ("canceled", "cancelled", True), ("complete", "completed", True),
    ("in planning", "planning", True), ("active or on hold", "active", False),
    ("inactive", "active", False),
])
def test_status(ans, gold, ok):
    assert is_correct(ans, Q("status", gold), CATALOG) is ok


def test_city():
    assert is_correct("Toronto", Q("city", "Toronto"), CATALOG)
    assert not is_correct("Toronto or Berlin", Q("city", "Toronto"), CATALOG)


# --------------------------------------------------------------------------- #
# Reward formula (spec §3.3)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("correct,valid,tokens,lam,expected", [
    (True, True, 0, 0.2, 1.1),
    (True, True, 2000, 0.2, 1.1 - 0.2 * 0.5),
    (False, True, 1000, 0.5, 0.1 - 0.5 * 0.25),
    (False, False, 800, 0.2, -0.2 * 0.2),       # no answer: no format bonus
    (True, True, 10_000, 0.5, 1.1 - 0.5),       # cost is capped at lambda
    (True, True, 3000, 0.0, 1.1),
])
def test_reward_formula(correct, valid, tokens, lam, expected):
    r = compute_reward(correct=correct, valid_answer=valid, tokens_read=tokens,
                       cost_lambda=lam, budget=4000)
    assert r == pytest.approx(expected)
