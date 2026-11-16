"""Answer normalizer and the deterministic reward (spec §3.3).

    r = 1[correct] + 0.1 * 1[episode ended with a valid answer() call]
        - lambda * min(1, tokens_read / B)

Correctness is checked per answer type and is deliberately hard to game:
  - names (person, team, office, client, project, city, role): find every
    catalog name of that type in the answer and keep only the maximal matches
    (so "Night Heron" hides the "Heron" inside it, and "Senior Software
    Engineer" hides "Software Engineer"). Correct iff exactly the gold remains.
    Listing several candidates fails, and so does a first name alone.
  - money / date / count: parse every value of that type in the answer.
    Correct iff exactly one distinct value is present and it equals the gold.
  - status: map synonyms to the five statuses. Exactly one must be mentioned.
The idea mirrors GRPO-Zero's verifier: forgiving about surface form, strict
about content.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date

from .workspace.entities import ROLE_LADDERS, CEO_ROLE, STATUSES
from .workspace.formats import parse_date, parse_money

CORRECT_REWARD = 1.0
FORMAT_REWARD = 0.1

STATUS_SYNONYMS = {
    "planning": ["planning", "in planning", "planned", "planning stage", "not started"],
    "active": ["active", "in progress", "ongoing", "underway"],
    "on hold": ["on hold", "on-hold", "onhold", "paused"],
    "completed": ["completed", "complete", "done", "finished"],
    "cancelled": ["cancelled", "canceled", "cancel"],
}
NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty".split())}

_MONEY_SPAN = re.compile(
    r"(?:US)?\$\s?\d[\d,]*(?:\.\d+)?\s*(?:million|thousand|mm|mn|m|k)?\b(?:\s*usd)?"
    r"|\b\d[\d,]*(?:\.\d+)?\s*(?:million|thousand|mm|mn|m|k)?\s*(?:usd|dollars)\b"
    r"|\b\d[\d,]*(?:\.\d+)?\s*(?:million|thousand|m|k)\b",
    re.IGNORECASE)
_DATE_SPANS = [
    re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b"),
    re.compile(r"\b[A-Za-z]{3,9}\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b"),
    re.compile(r"\b\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9}\.?,?\s+\d{4}\b"),
]


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = s.replace("’", "'").replace("‘", "'")
    s = re.sub(r"[^\w\s\-']", " ", s)
    s = s.replace("-", " ")
    return re.sub(r"\s+", " ", s).strip()


def _maximal_matches(answer: str, names: list[str]) -> set[str]:
    """Catalog names occurring in `answer` (word-bounded), dropping any match
    whose span lies inside a longer match."""
    a = f" {normalize(answer)} "
    spans = []
    for n in set(normalize(x) for x in names):
        if not n:
            continue
        for m in re.finditer(r"(?<=\s)" + re.escape(n) + r"(?=\s)", a):
            spans.append((m.start(), m.end(), n))
    keep = set()
    for s, e, n in spans:
        if not any(s2 <= s and e <= e2 and (e2 - s2) > (e - s) for s2, e2, _ in spans):
            keep.add(n)
    return keep


def _money_values(text: str) -> set[int]:
    vals = set()
    for m in _MONEY_SPAN.finditer(text):
        v = parse_money(m.group(0))
        if v is not None:
            vals.add(v)
    return vals


def _date_values(text: str) -> set[date]:
    vals, taken = set(), []
    for rx in _DATE_SPANS:
        for m in rx.finditer(text):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            d = parse_date(m.group(0))
            if d is not None:
                vals.add(d)
                taken.append((m.start(), m.end()))
    return vals


_NUM_TOKEN = re.compile(r"\d[\d,.]*\d|\d")


def _plain_ints(text: str) -> set[int]:
    """Integers written as digits ("7", "1,250,000"). Decimals such as "1.25"
    are not integers and are skipped, not split into 1 and 25."""
    out = set()
    for tok in _NUM_TOKEN.findall(text):
        tok = tok.rstrip(".,")
        if "." in tok:
            continue
        out.add(int(tok.replace(",", "")))
    return out


def _count_values(text: str) -> set[int]:
    vals = _plain_ints(text)
    vals |= {NUMBER_WORDS[w] for w in normalize(text).split() if w in NUMBER_WORDS}
    return vals


def _statuses(text: str) -> set[str]:
    found = set()
    for status, syns in STATUS_SYNONYMS.items():
        if _maximal_matches(text, syns):
            found.add(status)
    return found


ALL_ROLES = sorted({r for rs in ROLE_LADDERS.values() for r in rs} | {CEO_ROLE})


def is_correct(answer: str, question: dict, catalog: dict[str, list[str]]) -> bool:
    """`catalog[answer_type]` lists every valid name of that type in the
    workspace (from `build_catalog`)."""
    if not answer or not answer.strip():
        return False
    atype, gold = question["answer_type"], question["gold"]
    if atype == "money":
        vals = _money_values(answer)
        if not vals:   # a bare amount like "1250000" with no $/USD/M marker
            vals = {v for v in _plain_ints(answer) if v >= 1000}
        return vals == {parse_money(gold)}
    if atype == "date":
        return _date_values(answer) == {date.fromisoformat(gold)}
    if atype == "count":
        return _count_values(answer) == {int(gold)}
    if atype == "status":
        return _statuses(answer) == {gold}
    names = list(catalog.get(atype, [])) + [gold]
    return _maximal_matches(answer, names) == {normalize(gold)}


def build_catalog(world) -> dict[str, list[str]]:
    """Every name per answer type, used to reject hedged answers."""
    from .workspace.entities import CITIES
    by_kind = {}
    for e in world.entities.values():
        by_kind.setdefault(e.kind, []).append(e.name)
    return {"person": by_kind["person"], "team": by_kind["team"], "office": by_kind["office"],
            "client": by_kind["client"], "project": by_kind["project"],
            "contract": by_kind["contract"], "city": list(CITIES), "role": ALL_ROLES,
            "status": list(STATUSES)}


def compute_reward(*, correct: bool, valid_answer: bool, tokens_read: int,
                   cost_lambda: float, budget: int) -> float:
    """Spec §3.3. No answer() within K turns -> correct=False and no format bonus."""
    r = (CORRECT_REWARD if correct else 0.0) + (FORMAT_REWARD if valid_answer else 0.0)
    return r - cost_lambda * min(1.0, tokens_read / budget)
