"""Surface forms for money and dates, plus a deterministic sub-RNG helper.

Raw documents render the same value in several ways ("$1,250,000", "$1.25M",
"1,250,000 USD"), the way real company documents do. Notes use one canonical
form. The parsers here invert every renderer exactly. The reward's answer
normalizer (M2) builds on them, and the tests check `parse(render(x)) == x` for
every style.
"""

from __future__ import annotations

import hashlib
import random
import re
from datetime import date

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
MONTHS_SHORT = [m[:3] for m in MONTHS]
_MONTH_LOOKUP = {m.lower(): i + 1 for i, m in enumerate(MONTHS)}
_MONTH_LOOKUP.update({m.lower(): i + 1 for i, m in enumerate(MONTHS_SHORT)})
_MONTH_LOOKUP["sept"] = 9

N_MONEY_STYLES = 4
N_DATE_STYLES = 4


def sub_rng(seed: int, *keys) -> random.Random:
    """An independent RNG for (seed, keys...). Uses sha256, never Python's
    `hash()`, which is salted per process for strings and would break
    byte-identical regeneration."""
    h = hashlib.sha256(("|".join([str(seed), *map(str, keys)])).encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


# --------------------------------------------------------------------------- #
# Money (integer dollars; budgets and contract values are multiples of 10,000)
# --------------------------------------------------------------------------- #
def _trim(x: float) -> str:
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return s


def render_money(v: int, style: int = 0) -> str:
    """0: $1,250,000   1: $1.25M / $350K   2: 1,250,000 USD   3: $1.25 million / $350 thousand"""
    if style == 0:
        return f"${v:,}"
    if style == 1:
        return f"${_trim(v / 1e6)}M" if v >= 1_000_000 else f"${_trim(v / 1e3)}K"
    if style == 2:
        return f"{v:,} USD"
    if style == 3:
        return (f"${_trim(v / 1e6)} million" if v >= 1_000_000
                else f"${_trim(v / 1e3)} thousand")
    raise ValueError(style)


_MONEY_RE = re.compile(
    r"^\$?\s*([\d,]*\.?\d+)\s*(m|mm|mn|million|k|thousand|usd|dollars)?\s*(usd|dollars)?$",
    re.IGNORECASE)


def parse_money(s: str) -> int | None:
    """Inverse of every `render_money` style, plus a few lenient variants
    ("1.25 million", "$1250000"). Returns integer dollars or None."""
    t = s.strip().rstrip(".").replace("US$", "$").strip()
    m = _MONEY_RE.match(t)
    if not m:
        return None
    num = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or "").lower()
    mult = {"m": 1e6, "mm": 1e6, "mn": 1e6, "million": 1e6,
            "k": 1e3, "thousand": 1e3}.get(unit, 1.0)
    return int(round(num * mult))


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #
def render_date(d: date, style: int = 0) -> str:
    """0: 2025-03-14   1: March 14, 2025   2: 14 Mar 2025   3: Mar 14, 2025"""
    if style == 0:
        return d.isoformat()
    if style == 1:
        return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"
    if style == 2:
        return f"{d.day} {MONTHS_SHORT[d.month - 1]} {d.year}"
    if style == 3:
        return f"{MONTHS_SHORT[d.month - 1]} {d.day}, {d.year}"
    raise ValueError(style)


_ISO_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
_MDY_RE = re.compile(r"^([A-Za-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})$")
_DMY_RE = re.compile(r"^(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\.?,?\s+(\d{4})$")


def parse_date(s: str) -> date | None:
    """Inverse of every `render_date` style. Numeric d/m/y forms are rejected
    on purpose because they are ambiguous between US and EU order."""
    t = s.strip().rstrip(".")
    try:
        if m := _ISO_RE.match(t):
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if m := _MDY_RE.match(t):
            mon = _MONTH_LOOKUP.get(m.group(1).lower())
            return date(int(m.group(3)), mon, int(m.group(2))) if mon else None
        if m := _DMY_RE.match(t):
            mon = _MONTH_LOOKUP.get(m.group(2).lower())
            return date(int(m.group(3)), mon, int(m.group(1))) if mon else None
    except ValueError:
        return None
    return None
