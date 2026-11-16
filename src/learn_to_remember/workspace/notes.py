"""Notes: the text memory. An imperfect, versioned index over the raw workspace.

One page per covered entity, plus index pages. A version is defined by:
  - coverage:   the fraction of entities that get a page. Nested: a page in
                v_k also exists in v_{k+1}, unless the entity did not exist yet.
  - snapshot:   values are as of this date. Anything that changed later is
                stale, and entities created later have no page at all.
  - truncation: the probability that a page leaves out 1-2 of its fields.
  - recipe:     "structured" (key: value + links) or "prose".

Each page carries `as_of:` and links to the raw files it was built from, so a
model can check a note against its sources. The raw workspace is identical
across versions; only the notes change (the drift experiment).

v3 is deliberately NOT perfect (approved deviation, see docs/DECISIONS.md):
the main run trains on v3 and must still see gap and stale cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .entities import ATTRS, DATE_ATTRS, MONEY_ATTRS, World
from .formats import render_money, sub_rng


@dataclass(frozen=True)
class NoteVersion:
    name: str
    coverage: float
    snapshot: date
    truncation: float


VERSIONS = {
    "v0": NoteVersion("v0", 0.40, date(2024, 9, 30), 0.30),
    "v1": NoteVersion("v1", 0.60, date(2025, 2, 28), 0.20),
    "v2": NoteVersion("v2", 0.80, date(2025, 6, 30), 0.15),
    "v3": NoteVersion("v3", 0.90, date(2025, 9, 30), 0.10),
}

KIND_DIR = {"office": "offices", "team": "teams", "person": "people", "project": "projects",
            "client": "clients", "contract": "contracts"}
LIST_FIELDS = {
    "office": ["teams", "projects", "people"],
    "team": ["members", "projects"],
    "client": ["contracts", "projects"],
}
MAX_SOURCES = 3


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in name.lower()).strip("-")


def note_id(world: World, eid: str) -> str:
    e = world.entities[eid]
    return f"notes/{KIND_DIR[e.kind]}/{slug(e.name)}"


def notable_entities(world: World) -> list[str]:
    """Entities that can have a page: everything except the CEO and the
    Leadership pseudo-team (both are excluded from questions too)."""
    return [e for e in world.entities if e not in (world.ceo, world.leadership)]


def list_value(world: World, eid: str, name: str, d: date) -> list[str]:
    kind = world.entities[eid].kind
    if kind == "office":
        if name == "teams":
            return [t for t in world.office_teams_at(eid, d) if t != world.leadership]
        if name == "projects":
            return world.office_projects_at(eid, d)
        if name == "people":
            return [p for p in world.office_people_at(eid, d) if p != world.ceo]
    if kind == "team":
        if name == "members":
            return world.team_members_at(eid, d)
        if name == "projects":
            return world.team_projects_at(eid, d)
    if kind == "client" and name == "contracts":
        return world.client_contracts_at(eid, d)
    if kind == "client" and name == "projects":
        return [p for p in world.of_kind("project", d) if world.value_at(p, "client", d) == eid]
    raise ValueError((kind, name))


def _fmt(world: World, attr: str, value) -> str:
    if value is None:
        return "none (internal)" if attr == "client" else "none"
    if attr in MONEY_ATTRS:
        return render_money(value, 0)
    if attr in DATE_ATTRS:
        return value                       # canonical ISO
    if attr == "project":
        return world.display(value)
    if attr in ("head", "lead", "office", "team", "manager", "owner", "client",
                "account_manager"):
        return world.name(value)
    return str(value)


def _list_line(world: World, name: str, x: str, d: date) -> str:
    if name == "projects":
        return f"- {world.display(x)} ({world.value_at(x, 'status', d)})"
    return f"- {world.name(x)}"


def build_notes(world: World, docs: list, version: NoteVersion,
                recipe: str = "structured") -> list[dict]:
    """All note pages for one version, index pages first."""
    if recipe not in ("structured", "prose"):
        raise ValueError(recipe)
    snap = version.snapshot
    ents = notable_entities(world)
    # nested coverage: a fixed per-company priority order over entities
    order = sorted(ents, key=lambda e: sub_rng(world.seed, "note-priority", e).random())
    n_cov = int(round(version.coverage * len(ents)))
    covered = [e for e in order[:n_cov] if world.exists_at(e, snap)]
    covered.sort(key=lambda e: (world.entities[e].kind, note_id(world, e)))

    # sources: the latest docs (<= snapshot) that assert something about the entity
    src: dict[str, list[str]] = {}
    for doc in docs:
        if doc.date > snap:
            break
        for eid in sorted({a[0] for a in doc.asserts}):
            src.setdefault(eid, []).append(doc.path)

    pages = []
    for eid in covered:
        e = world.entities[eid]
        rng = sub_rng(world.seed, "note-trunc", version.name, eid)
        fields = {a: world.value_at(eid, a, snap) for a in ATTRS[e.kind]}
        lists = {n: list_value(world, eid, n, snap) for n in LIST_FIELDS.get(e.kind, [])}
        if e.kind == "person" and fields.get("manager") is None:
            fields.pop("manager")
        omitted = []
        if rng.random() < version.truncation:
            keys = list(fields) + list(lists)
            omitted = sorted(rng.sample(keys, min(len(keys), rng.choice([1, 1, 2]))))
        for k in omitted:
            fields.pop(k, None)
            lists.pop(k, None)
        sources = src.get(eid, [])[-MAX_SOURCES:]
        nid = note_id(world, eid)
        title = world.display(eid) if e.kind in ("project",) else e.name
        if e.kind == "office":
            title = f"{e.name} office"
        if e.kind == "team":
            title = f"{e.name} team"

        if recipe == "structured":
            lines = [f"# {title}", f"note_id: {nid}", f"type: {e.kind}", f"as_of: {snap}"]
            lines += [f"{a}: {_fmt(world, a, v)}" for a, v in fields.items()]
            for n, ids in lists.items():
                lines.append(f"{n} ({len(ids)}):")
                lines += [_list_line(world, n, x, snap) for x in ids]
            lines.append("sources:")
            lines += [f"- {s}" for s in sources] or ["- none"]
        else:
            facts = "; ".join(f"its {a.replace('_', ' ')} is {_fmt(world, a, v)}"
                              for a, v in fields.items())
            lines = [f"# {title}", f"note_id: {nid}", f"as_of: {snap}", "",
                     f"{title} ({e.kind}): {facts}." if facts else f"{title} ({e.kind})."]
            for n, ids in lists.items():
                items = ", ".join(_list_line(world, n, x, snap)[2:] for x in ids)
                lines.append(f"Its {n} ({len(ids)}): {items or 'none'}.")
            lines.append("Sources: " + (", ".join(sources) or "none") + ".")
        pages.append({
            "note_id": nid, "entity": eid, "kind": e.kind, "title": title,
            "version": version.name, "as_of": snap.isoformat(), "recipe": recipe,
            "fields": fields, "lists": lists, "omitted": omitted, "sources": sources,
            "text": "\n".join(lines) + "\n",
        })

    # index pages: one per kind plus a root index
    index_pages = []
    root = [f"# Notes index", f"note_id: notes/index", f"as_of: {snap}", "",
            "Browse by type:"]
    for kind, d in KIND_DIR.items():
        ps = [p for p in pages if p["kind"] == kind]
        root.append(f"- notes/index/{d} ({len(ps)} pages)")
        lines = [f"# Index: {d}", f"note_id: notes/index/{d}", f"as_of: {snap}", ""]
        lines += [f"- {p['title']}: {p['note_id']}" for p in ps]
        index_pages.append({
            "note_id": f"notes/index/{d}", "entity": None, "kind": "index",
            "title": f"Index: {d}", "version": version.name, "as_of": snap.isoformat(),
            "recipe": recipe, "fields": {}, "lists": {}, "omitted": [], "sources": [],
            "text": "\n".join(lines) + "\n"})
    root_page = {"note_id": "notes/index", "entity": None, "kind": "index",
                 "title": "Notes index", "version": version.name, "as_of": snap.isoformat(),
                 "recipe": recipe, "fields": {}, "lists": {}, "omitted": [], "sources": [],
                 "text": "\n".join(root) + "\n"}
    return [root_page] + index_pages + pages
