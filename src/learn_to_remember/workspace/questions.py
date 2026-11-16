"""Questions with programmatic gold answers, labeled per note version.

Three question structures:
  - single:    one fact ("What is Project Heron's current budget?")
  - multihop:  a chain of facts ("Who manages the owner of Project Heron?")
  - aggregate: a count over a list ("How many active projects does the Lisbon
               office have?")

Gold is always the value at END (2025-12-31) in the world timeline, which the
raw documents assert (see documents.py). The spec's qtype depends on the notes,
so it is computed **per note version**. A single-fact question is:
  - gap    if the entity has no page (gap_kind = missing_page) or the page
           omits that field (gap_kind = missing_field);
  - stale  if the page's value differs from the latest raw value;
  - lookup otherwise (the notes are correct).
Multihop and aggregate questions keep their structural qtype and record
`notes_ok_by_version` (could the notes alone give the right answer?).
The primary `qtype` is the v3 label, and eval is balanced on it.

Splits (per company):
  - held-out entities: ~10% of people, projects, clients and contracts. They
    never appear in any training question (anchor, hop or answer).
  - eval: balanced across the 5 v3 qtypes, round-robin over question families.
  - train: questions that share no fact (entity, attr) with any eval question
    and touch no held-out entity, with up to 2 paraphrases each.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .entities import END, World
from .formats import render_money, sub_rng
from .notes import VERSIONS, list_value

QTYPES = ("lookup", "multihop", "aggregate", "gap", "stale")

# (kind, attr) -> (answer_type, templates). {d} is the entity's display name.
SINGLE = {
    ("project", "owner"): ("person", ["Who is the current owner of {d}?",
                                      "Who owns {d} right now?",
                                      "Which person is responsible for {d} at the moment?"]),
    ("project", "budget"): ("money", ["What is the current budget of {d}?",
                                      "How much budget does {d} have right now?",
                                      "What budget is {d} currently working with?"]),
    ("project", "status"): ("status", ["What is the current status of {d}?",
                                       "Is {d} active, on hold, completed, cancelled or in planning?",
                                       "What state is {d} in right now?"]),
    ("project", "end_date"): ("date", ["What is the current target end date of {d}?",
                                       "When is {d} expected to finish?",
                                       "By what date is {d} scheduled to be done?"]),
    ("project", "start_date"): ("date", ["When did {d} start?",
                                         "What is the start date of {d}?"]),
    ("project", "team"): ("team", ["Which team runs {d}?",
                                   "Which team is responsible for {d}?"]),
    ("project", "client"): ("client", ["Who is the client for {d}?",
                                       "Which client is {d} for?"]),
    ("person", "team"): ("team", ["Which team is {d} on?",
                                  "What team does {d} currently belong to?"]),
    ("person", "role"): ("role", ["What is {d}'s current job title?",
                                  "What role does {d} hold right now?"]),
    ("person", "manager"): ("person", ["Who is {d}'s manager?",
                                       "Who does {d} currently report to?"]),
    ("person", "start_date"): ("date", ["When did {d} start at the company?",
                                        "What is {d}'s start date?"]),
    ("person", "office"): ("office", ["Which office is {d} based in?",
                                      "In which office does {d} work?"]),
    ("contract", "value"): ("money", ["What is the current value of contract {d}?",
                                      "How much is contract {d} worth now?"]),
    ("contract", "signed_date"): ("date", ["When was contract {d} signed?",
                                           "What is the signing date of contract {d}?"]),
    ("contract", "client"): ("client", ["Which client is contract {d} with?",
                                        "Who is the client on contract {d}?"]),
    ("contract", "project"): ("project", ["Which project does contract {d} cover?",
                                          "What project is contract {d} linked to?"]),
    ("client", "account_manager"): ("person", ["Who is the account manager for {d}?",
                                               "Who manages the {d} account?"]),
    # client.industry is deliberately NOT asked: it follows from the client's name
    # ("Redwood Insurance" -> insurance), so it could be answered without a lookup.
    ("client", "hq_city"): ("city", ["Where is {d} headquartered?",
                                     "In which city is {d}'s headquarters?"]),
    ("team", "lead"): ("person", ["Who leads the {d} team?",
                                  "Who is the current lead of the {d} team?"]),
    ("team", "office"): ("office", ["Which office is the {d} team based in?",
                                    "Where is the {d} team located?"]),
    ("office", "head"): ("person", ["Who heads the {d} office?",
                                    "Who is the current head of the {d} office?"]),
}
ATTR_ANSWER_TYPE = {(k, a): v[0] for (k, a), v in SINGLE.items()}
ATTR_ANSWER_TYPE[("person", "role")] = "role"

# (anchor kind, [attr chain]) -> templates
MULTIHOP = {
    ("project", ("owner", "manager")): ["Who is the manager of the person who owns {d}?",
                                        "Who does the owner of {d} report to?"],
    ("project", ("team", "lead")): ["Who leads the team responsible for {d}?",
                                    "Who is the lead of the team that runs {d}?"],
    ("project", ("team", "office")): ["Which office is the team behind {d} based in?",
                                      "In which office is the team that runs {d} located?"],
    ("project", ("client", "account_manager")): [
        "Who is the account manager for the client of {d}?",
        "Who manages the account of the client that {d} is for?"],
    ("contract", ("project", "budget")): [
        "What is the current budget of the project covered by contract {d}?",
        "How much budget does the project linked to contract {d} have?"],
    ("contract", ("project", "owner")): ["Who owns the project linked to contract {d}?",
                                         "Who is the owner of the project that contract {d} covers?"],
    ("person", ("manager", "role")): ["What is the job title of {d}'s manager?",
                                      "What role does the person {d} reports to hold?"],
}
# aggregate family -> (anchor kind, list name, status filter or None, templates)
AGGREGATE = {
    "office_projects_active": ("office", "projects", "active",
                               ["How many active projects do teams in the {d} office run?",
                                "How many projects are currently active in the {d} office?"]),
    "office_projects_on_hold": ("office", "projects", "on hold",
                                ["How many projects in the {d} office are on hold?",
                                 "How many on-hold projects do teams in the {d} office have?"]),
    "office_projects_completed": ("office", "projects", "completed",
                                  ["How many projects run by the {d} office have been completed?",
                                   "How many completed projects does the {d} office have?"]),
    "office_projects_planning": ("office", "projects", "planning",
                                 ["How many projects in the {d} office are still in planning?",
                                  "How many of the {d} office's projects are in the planning stage?"]),
    "office_teams": ("office", "teams", None,
                     ["How many teams are based in the {d} office?",
                      "How many teams does the {d} office host?"]),
    "team_projects_completed": ("team", "projects", "completed",
                                ["How many projects has the {d} team completed?",
                                 "How many of the {d} team's projects are completed?"]),
    "client_projects": ("client", "projects", None,
                        ["How many projects do we run for {d}?",
                         "For how many of our projects is {d} the client?"]),
    "team_projects_active": ("team", "projects", "active",
                             ["How many active projects does the {d} team run?",
                              "How many of the {d} team's projects are currently active?"]),
    "team_headcount": ("team", "members", None,
                       ["How many people are currently on the {d} team?",
                        "What is the current headcount of the {d} team?"]),
    "office_headcount": ("office", "people", None,
                         ["How many people are currently based in the {d} office?",
                          "What is the headcount of the {d} office?"]),
    "client_contracts": ("client", "contracts", None,
                         ["How many contracts do we have with {d}?",
                          "How many contracts has {d} signed with us?"]),
}


@dataclass
class Candidate:
    key: str
    family: str
    structure: str                   # single | multihop | aggregate
    anchor: str
    facts: list                      # [(eid, attr)] read to answer
    entities: list                   # every entity id the question touches
    gold_value: object
    answer_type: str
    templates: list
    labels: dict = field(default_factory=dict)   # version -> qtype
    gap_kind: dict = field(default_factory=dict)  # version -> missing_page/missing_field/None
    notes_ok: dict = field(default_factory=dict)  # version -> bool


def render_gold(world: World, answer_type: str, value) -> tuple[str, list[str]]:
    """Canonical gold string plus accepted aliases."""
    if answer_type in ("person", "team", "client"):
        name = world.name(value)
        return name, ([f"{name} team", f"the {name} team"] if answer_type == "team" else [])
    if answer_type == "office":
        name = world.name(value)
        return name, [f"{name} office", f"the {name} office"]
    if answer_type == "project":
        name = world.name(value)
        return name, [f"Project {name}"]
    if answer_type == "money":
        return render_money(value, 0), []
    if answer_type == "count":
        return str(value), []
    return str(value), []


def _display(world: World, eid: str) -> str:
    e = world.entities[eid]
    return world.display(eid) if e.kind == "project" else e.name


def _notes_lookup(pages_by_entity: dict, eid: str, attr: str):
    """(status, value) for one fact in one note version."""
    page = pages_by_entity.get(eid)
    if page is None:
        return "missing_page", None
    if attr in page["omitted"] or attr not in page["fields"]:
        return "missing_field", None
    return "ok", page["fields"][attr]


def build_candidates(world: World, notes: dict[str, list[dict]]) -> list[Candidate]:
    excluded = {world.ceo, world.leadership}
    by_v = {v: {p["entity"]: p for p in pages if p["entity"]} for v, pages in notes.items()}
    cands: list[Candidate] = []

    for (kind, attr), (atype, tmpls) in SINGLE.items():
        for eid in world.of_kind(kind):
            if eid in excluded:
                continue
            val = world.latest(eid, attr)
            if val is None or (atype in ("person", "team") and val in excluded):
                continue
            ents = [eid] + ([val] if atype in ("person", "team", "office", "client",
                                               "project") else [])
            c = Candidate(f"single:{kind}.{attr}:{eid}", f"{kind}.{attr}", "single", eid,
                          [(eid, attr)], ents, val, atype, tmpls)
            for v, pages in by_v.items():
                status, nv = _notes_lookup(pages, eid, attr)
                if status != "ok":
                    c.labels[v], c.gap_kind[v], c.notes_ok[v] = "gap", status, False
                elif nv != val:
                    c.labels[v], c.gap_kind[v], c.notes_ok[v] = "stale", None, False
                else:
                    c.labels[v], c.gap_kind[v], c.notes_ok[v] = "lookup", None, True
            cands.append(c)

    for (kind, chain), tmpls in MULTIHOP.items():
        for eid in world.of_kind(kind):
            if eid in excluded:
                continue
            cur, facts, ents, ok = eid, [], [eid], True
            for a in chain:
                facts.append((cur, a))
                nxt = world.latest(cur, a)
                if nxt is None or nxt in excluded:
                    ok = False
                    break
                cur = nxt
                if isinstance(cur, str) and cur in world.entities:
                    ents.append(cur)
            if not ok:
                continue
            last_attr = chain[-1]
            atype = {"manager": "person", "lead": "person", "office": "office",
                     "account_manager": "person", "budget": "money", "owner": "person",
                     "role": "role"}[last_attr]
            c = Candidate(f"multihop:{kind}.{'.'.join(chain)}:{eid}",
                          f"{kind}.{'.'.join(chain)}", "multihop", eid, facts, ents, cur,
                          atype, tmpls)
            for v, pages in by_v.items():
                # notes alone suffice iff every hop is present and fresh
                hop_ok = True
                for fe, fa in facts:
                    status, nv = _notes_lookup(pages, fe, fa)
                    if status != "ok" or nv != world.latest(fe, fa):
                        hop_ok = False
                        break
                c.labels[v], c.gap_kind[v], c.notes_ok[v] = "multihop", None, hop_ok
            cands.append(c)

    for fam, (kind, lname, status, tmpls) in AGGREGATE.items():
        for eid in world.of_kind(kind):
            if eid in excluded:
                continue
            items = list_value(world, eid, lname, END)
            if status:
                items = [x for x in items if world.latest(x, "status") == status]
            if not items:
                continue                     # skip zero counts: too easy to guess
            c = Candidate(f"aggregate:{fam}:{eid}", fam, "aggregate", eid,
                          [(eid, fam)], [eid], len(items), "count", tmpls)
            for v, pages in by_v.items():
                page = pages.get(eid)
                ok = page is not None and lname in page["lists"]
                if ok:
                    lst = page["lists"][lname]
                    if status:
                        # the notes show each project's status as of the snapshot
                        snap = VERSIONS[v].snapshot
                        lst = [x for x in lst if world.value_at(x, "status", snap) == status]
                    ok = len(lst) == len(items)
                c.labels[v], c.gap_kind[v], c.notes_ok[v] = "aggregate", None, ok
            cands.append(c)
    return cands


def _record(world, c: Candidate, company: str, split: str, qid: str, template: int,
            heldout: set) -> dict:
    gold, aliases = render_gold(world, c.answer_type, c.gold_value)
    return {
        "id": qid, "company": company, "split": split, "family": c.family,
        "structure": c.structure,
        "question": c.templates[template].format(d=_display(world, c.anchor)),
        "template": template, "gold": gold, "aliases": aliases,
        "gold_value": c.gold_value, "answer_type": c.answer_type,
        "anchor": c.anchor, "entities": c.entities, "facts": [list(f) for f in c.facts],
        "qtype": c.labels["v3"], "qtype_by_version": c.labels,
        "gap_kind_by_version": c.gap_kind, "notes_ok_by_version": c.notes_ok,
        "heldout": any(e in heldout for e in c.entities),
    }


def choose_heldout(world: World, frac: float = 0.10) -> set[str]:
    rng = sub_rng(world.seed, "heldout")
    out = set()
    for kind in ("person", "project", "client", "contract"):
        ids = [e for e in world.of_kind(kind) if e != world.ceo]
        out.update(rng.sample(ids, max(1, int(round(frac * len(ids))))))
    return out


def make_splits(world: World, notes: dict, company: str, *, eval_per_qtype: int = 60,
                eval_bucket_cap: float = 0.6, n_train: int = 2000,
                with_train: bool = True, paraphrases: int = 2) -> dict:
    """Return {"eval": [...], "train": [...], "heldout": [...], "candidates": n}."""
    cands = build_candidates(world, notes)
    heldout = choose_heldout(world) if with_train else set()
    rng = sub_rng(world.seed, "splits", company)

    # ---- eval: balanced on v3 qtype, round-robin over families ------------
    buckets: dict[str, dict[str, list[Candidate]]] = {q: defaultdict(list) for q in QTYPES}
    for c in cands:
        buckets[c.labels["v3"]][c.family].append(c)
    eval_sel: list[Candidate] = []
    for q in QTYPES:
        fams = buckets[q]
        n_avail = sum(len(v) for v in fams.values())
        cap = n_avail if not with_train else int(eval_bucket_cap * n_avail)
        target = min(eval_per_qtype, cap)
        pools = {f: rng.sample(v, len(v)) for f, v in sorted(fams.items())}
        chosen: list[Candidate] = []
        while len(chosen) < target:
            progressed = False
            for f in sorted(pools):
                if pools[f] and len(chosen) < target:
                    chosen.append(pools[f].pop())
                    progressed = True
            if not progressed:
                break
        eval_sel += chosen

    eval_records = []
    for i, c in enumerate(eval_sel):
        t = rng.randrange(len(c.templates))
        eval_records.append(_record(world, c, company, "eval", f"{company}-eval-{i:04d}", t,
                                    heldout))

    out = {"eval": eval_records, "train": [], "heldout": sorted(heldout),
           "n_candidates": len(cands)}
    if not with_train:
        return out

    # ---- train: no shared facts with eval, no held-out entities ----------
    eval_facts = {tuple(f) for c in eval_sel for f in c.facts}
    eval_keys = {c.key for c in eval_sel}
    pool = [c for c in cands
            if c.key not in eval_keys
            and not any(tuple(f) in eval_facts for f in c.facts)
            and not any(e in heldout for e in c.entities)]
    items = []
    for c in pool:
        ts = rng.sample(range(len(c.templates)), min(paraphrases, len(c.templates)))
        items += [(c, t) for t in ts]
    rng.shuffle(items)
    items = items[:n_train]
    out["train"] = [_record(world, c, company, "train", f"{company}-train-{i:04d}", t, heldout)
                    for i, (c, t) in enumerate(items)]
    return out
