"""The synthetic company: entities plus a dated timeline of every fact.

`build_world(seed)` creates a fictional company and evolves it over two years
(2024-01-01 .. 2025-12-31) through dated change events: budget revisions, owner
changes, status changes, promotions, transfers, re-orgs, hires, new projects and
contracts. Each fact (entity, attribute) has a timeline of (date, value)
entries. That timeline is the ground truth from which documents, notes and gold
answers are all derived, so they cannot disagree by accident.

Design choices that matter downstream:
  - No fact changes twice within MIN_GAP_DAYS. Documents that bundle a week's
    events (meeting notes) then never contain two values of one fact.
  - Q4 2025 has a planning-cycle wave of changes. Notes snapshotted before it
    (all versions, including v3) are therefore stale on real facts.
  - Distractors are built in: first names repeat across people, and some
    project codenames are near-duplicates ("Heron" / "Night Heron").

Everything is driven by `sub_rng(seed, ...)`. No sets of strings are iterated
unsorted and no `hash()` is used, so output is byte-identical across runs and
processes.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import date, timedelta

from .formats import sub_rng

START = date(2024, 1, 1)
END = date(2025, 12, 31)          # "today" for every question
EVENTS_FROM = date(2024, 1, 10)   # the first week is reserved for directories
EVENTS_END = date(2025, 12, 12)   # year-end reviews are written after this
MIN_GAP_DAYS = 14
Q4_WAVE = (date(2025, 10, 1), EVENTS_END)

STATUSES = ["planning", "active", "on hold", "completed", "cancelled"]
OPEN_STATUSES = ("planning", "active", "on hold")

ATTRS = {
    "office": ["head"],
    "team": ["office", "lead"],
    "person": ["team", "role", "manager", "start_date", "office"],
    "project": ["owner", "team", "budget", "status", "start_date", "end_date", "client"],
    "client": ["industry", "hq_city", "account_manager"],
    "contract": ["client", "value", "signed_date", "project"],
}
REF_ATTRS = {"head", "lead", "office", "team", "manager", "owner", "client",
             "account_manager", "project"}
MONEY_ATTRS = {"budget", "value"}
DATE_ATTRS = {"start_date", "end_date", "signed_date"}

ROLE_LADDERS = {
    "eng": ["Software Engineer", "Senior Software Engineer", "Staff Software Engineer",
            "Principal Engineer"],
    "data": ["Data Analyst", "Senior Data Analyst", "Lead Data Analyst"],
    "product": ["Product Manager", "Senior Product Manager", "Group Product Manager"],
    "design": ["Product Designer", "Senior Product Designer", "Design Lead"],
    "mgmt": ["Engineering Manager", "Senior Engineering Manager", "Director of Engineering"],
}
ROLE_TRACK = {r: t for t, rs in ROLE_LADDERS.items() for r in rs}
CEO_ROLE = "Chief Executive Officer"

# --------------------------------------------------------------------------- #
# Name pools (pure data, so generation needs no LLM and is free)
# --------------------------------------------------------------------------- #
FIRST_NAMES = [
    "Maria", "Ana", "James", "Priya", "Wei", "Fatima", "Lucas", "Sofia", "Omar", "Elena",
    "David", "Aisha", "Daniel", "Yuki", "Carlos", "Nadia", "Ethan", "Leila", "Mateo", "Hana",
    "Samuel", "Ines", "Noah", "Amara", "Jonas", "Chloe", "Ravi", "Zara", "Tomas", "Mei",
    "Kofi", "Laura", "Arjun", "Sara", "Felix", "Nora", "Diego", "Lina", "Victor", "Grace",
    "Hugo", "Iris", "Marco", "Asha", "Leo",
]
LAST_NAMES = [
    "Okafor", "Silva", "Costa", "Chen", "Patel", "Kowalski", "Nguyen", "Haddad", "Rossi",
    "Moreau", "Schmidt", "Tanaka", "Garcia", "Ivanova", "Mensah", "Larsen", "Novak", "Ali",
    "Fischer", "Kim", "Santos", "Dubois", "Murphy", "Sato", "Jansen", "Petrov", "Mwangi",
    "Lopez", "Weber", "Park", "Rahman", "Bianchi", "Nilsen", "Cohen", "Ortiz", "Yilmaz",
    "Andersen", "Kaur", "Romero", "Hoffmann", "Ito", "Popescu", "Adeyemi", "Martins", "Svoboda",
    "Barros", "Wright", "Lindqvist", "Ahmed", "Ferreira", "Gallo", "Horvat", "Iyer", "Juarez",
    "Keller", "Lambert", "Molina", "Nakamura", "Osei", "Pereira", "Quinn", "Reyes", "Suzuki",
    "Torres", "Ueda", "Vargas", "Wojcik", "Xu", "Yamada", "Zhou", "Abbasi", "Brennan",
    "Castillo", "Duarte", "Eriksen", "Fontaine", "Grant", "Hughes", "Ibrahim", "Jovanovic",
    "Kruger", "Lucero", "Marsh", "Nowak", "Olsen", "Pham", "Rocha", "Sharma", "Tran", "Vidal",
    "Walsh", "Young", "Zimmer", "Akhtar", "Bauer", "Carvalho", "Dalton", "Esposito", "Farouk",
    "Gupta", "Hale", "Ishikawa", "Jensen", "Kuznetsov", "Lund", "Mahmoud", "Nolan", "Ochoa",
    "Pinto", "Ramos", "Steiner", "Toivonen", "Usman", "Valdez", "Watts", "Yusuf", "Zeller",
]
CITIES = [
    "Lisbon", "Toronto", "Singapore", "Berlin", "Austin", "Nairobi", "Dublin", "Seoul",
    "Madrid", "Chicago", "Sydney", "Warsaw", "Denver", "Lagos", "Oslo", "Bangalore",
    "Montreal", "Prague", "Boston", "Lyon", "Osaka", "Helsinki", "Porto", "Vienna",
]
TEAM_NAMES = [
    "Data Platform", "Payments", "Mobile Apps", "Search", "Security", "Growth",
    "Infrastructure", "Customer Success", "Finance Systems", "Analytics", "Design Systems",
    "Billing", "Developer Tools", "Platform Reliability", "Marketing Tech", "Partnerships",
    "Data Products", "Identity", "Logistics", "Integrations",
]
CODENAMES = [
    "Heron", "Kestrel", "Osprey", "Falcon", "Wren", "Plover", "Ibis", "Tern", "Egret",
    "Merlin", "Swift", "Lark", "Finch", "Crane", "Raven", "Condor", "Puffin", "Avocet",
    "Basalt", "Cobalt", "Quartz", "Onyx", "Garnet", "Jasper", "Topaz", "Beryl", "Flint",
    "Cedar", "Juniper", "Willow", "Aspen", "Rowan", "Hazel", "Alder", "Sequoia", "Laurel",
    "Comet", "Nova", "Pulsar", "Quasar", "Nebula", "Zenith", "Aurora", "Meridian", "Vega",
    "Orion", "Lyra", "Atlas", "Tundra", "Mesa", "Delta", "Fjord", "Harbor", "Summit",
]
CODENAME_PREFIXES = ["Night", "Grey", "Little", "North", "Red", "Blue", "Silver", "Black"]
CLIENT_PREFIXES = [
    "Northwind", "Bluefin", "Granite", "Silverline", "Brightwater", "Ironwood", "Sunpeak",
    "Clearpath", "Redwood", "Harborview", "Evergreen", "Lakeshore", "Stonebridge", "Westgate",
    "Goldcrest", "Pinehill", "Riverbend", "Skyline", "Oakmont", "Fairfield", "Crescent",
    "Highland", "Summitview", "Kingsbay",
]
CLIENT_SUFFIXES = ["Logistics", "Health", "Capital", "Foods", "Energy", "Retail", "Media",
                   "Insurance", "Motors", "Telecom", "Pharma", "Analytics"]
INDUSTRIES = {
    "Logistics": "logistics", "Health": "healthcare", "Capital": "financial services",
    "Foods": "food and beverage", "Energy": "energy", "Retail": "retail", "Media": "media",
    "Insurance": "insurance", "Motors": "automotive", "Telecom": "telecommunications",
    "Pharma": "pharmaceuticals", "Analytics": "software",
}


@dataclass
class WorldConfig:
    n_offices: int = 6
    n_teams: int = 14
    n_people: int = 80          # initial headcount (plus the CEO); hires are added over time
    n_projects: int = 36        # initial projects; new ones are started over time
    n_clients: int = 22
    n_contracts: int = 22       # initial contracts; new ones are signed over time
    external_project_frac: float = 0.7
    sibling_codename_frac: float = 0.18
    # expected events per month (fractional parts resolved by the RNG)
    rates: dict = field(default_factory=lambda: {
        "budget_revision": 3.0, "owner_change": 1.2, "status_change": 2.2,
        "end_date_change": 1.0, "contract_amendment": 0.8, "account_manager_change": 0.5,
        "promotion": 1.6, "transfer": 1.2, "team_lead_change": 0.35,
        "office_head_change": 0.12, "hire": 1.2, "new_project": 0.6, "new_contract": 0.6,
    })
    q4_boost: dict = field(default_factory=lambda: {
        "budget_revision": 4.0, "status_change": 3.0, "owner_change": 3.0,
        "end_date_change": 3.5, "promotion": 4.0, "team_lead_change": 2.5, "hire": 1.5,
        "account_manager_change": 3.0, "contract_amendment": 3.0, "transfer": 2.0,
    })


@dataclass
class Entity:
    id: str
    kind: str
    name: str
    created: date
    aliases: list[str] = field(default_factory=list)


@dataclass
class Event:
    id: str
    date: date
    type: str
    subject: str                                   # main entity id
    changes: list[tuple[str, str, object, object]]  # (entity_id, attr, old, new)
    info: dict = field(default_factory=dict)


class World:
    """Entities + per-fact timelines + the events that produced them."""

    def __init__(self, seed: int):
        self.seed = seed
        self.entities: dict[str, Entity] = {}
        self.timeline: dict[tuple[str, str], list[tuple[date, object]]] = {}
        self.events: list[Event] = []
        self.ceo: str | None = None          # excluded from questions and notes
        self.leadership: str | None = None   # pseudo-team holding only the CEO

    # -- facts --------------------------------------------------------------
    def add(self, ent: Entity, attrs: dict) -> None:
        self.entities[ent.id] = ent
        for a in ATTRS[ent.kind]:
            self.timeline[(ent.id, a)] = [(ent.created, attrs.get(a))]

    def set(self, eid: str, attr: str, d: date, value) -> None:
        tl = self.timeline[(eid, attr)]
        assert d > tl[-1][0], f"non-monotonic change {eid}.{attr} at {d}"
        tl.append((d, value))

    def value_at(self, eid: str, attr: str, d: date):
        ent = self.entities[eid]
        if d < ent.created:
            return None
        val = None
        for when, v in self.timeline[(eid, attr)]:
            if when <= d:
                val = v
            else:
                break
        return val

    def latest(self, eid: str, attr: str):
        return self.value_at(eid, attr, END)

    def last_change(self, eid: str, attr: str) -> date:
        return self.timeline[(eid, attr)][-1][0]

    def exists_at(self, eid: str, d: date) -> bool:
        return self.entities[eid].created <= d

    def of_kind(self, kind: str, d: date | None = None) -> list[str]:
        return [e.id for e in self.entities.values()
                if e.kind == kind and (d is None or e.created <= d)]

    def name(self, eid: str | None) -> str | None:
        return None if eid is None else self.entities[eid].name

    def display(self, eid: str) -> str:
        """Name as written in prose: 'Project Heron', 'the Lisbon office', ..."""
        e = self.entities[eid]
        if e.kind == "project":
            return f"Project {e.name}"
        if e.kind == "office":
            return f"the {e.name} office"
        if e.kind == "team":
            return f"the {e.name} team"
        return e.name

    # -- derived lists (used by aggregate questions and roster documents) ---
    def team_members_at(self, team: str, d: date) -> list[str]:
        return [p for p in self.of_kind("person", d) if self.value_at(p, "team", d) == team]

    def office_people_at(self, office: str, d: date) -> list[str]:
        return [p for p in self.of_kind("person", d) if self.value_at(p, "office", d) == office]

    def office_teams_at(self, office: str, d: date) -> list[str]:
        return [t for t in self.of_kind("team", d) if self.value_at(t, "office", d) == office]

    def team_projects_at(self, team: str, d: date) -> list[str]:
        return [p for p in self.of_kind("project", d) if self.value_at(p, "team", d) == team]

    def office_projects_at(self, office: str, d: date) -> list[str]:
        return [p for p in self.of_kind("project", d)
                if self.value_at(self.value_at(p, "team", d), "office", d) == office]

    def client_contracts_at(self, client: str, d: date) -> list[str]:
        return [c for c in self.of_kind("contract", d) if self.value_at(c, "client", d) == client]


# --------------------------------------------------------------------------- #
# Building the initial company
# --------------------------------------------------------------------------- #
def _rand_date(rng, lo: date, hi: date) -> date:
    return lo + timedelta(days=rng.randint(0, (hi - lo).days))


def _round10k(x: float) -> int:
    return max(100_000, int(round(x / 10_000)) * 10_000)


class _Builder:
    def __init__(self, seed: int, cfg: WorldConfig):
        self.seed, self.cfg = seed, cfg
        self.w = World(seed)
        self.rng = sub_rng(seed, "world")
        self.used_person_names: set[str] = set()
        self.used_codenames: set[str] = set()
        self.used_contract_ids: set[str] = set()
        self.n_person = 0
        self.n_project = 0
        self.n_event = 0
        self.ceo = None
        self.leadership = None

    # -- name helpers --------------------------------------------------------
    def person_name(self) -> str:
        while True:
            n = f"{self.rng.choice(FIRST_NAMES)} {self.rng.choice(LAST_NAMES)}"
            if n not in self.used_person_names:
                self.used_person_names.add(n)
                return n

    def codename(self) -> str:
        existing = sorted(self.used_codenames)
        if existing and self.rng.random() < self.cfg.sibling_codename_frac:
            base = self.rng.choice([c for c in existing if " " not in c] or existing)
            for _ in range(20):
                cand = f"{self.rng.choice(CODENAME_PREFIXES)} {base.split()[-1]}"
                if cand not in self.used_codenames:
                    self.used_codenames.add(cand)
                    return cand
        free = [c for c in CODENAMES if c not in self.used_codenames]
        cand = self.rng.choice(free)
        self.used_codenames.add(cand)
        return cand

    def contract_id(self) -> str:
        while True:
            c = f"C-{self.rng.randint(1000, 9999)}"
            if c not in self.used_contract_ids:
                self.used_contract_ids.add(c)
                return c

    # -- entity constructors -------------------------------------------------
    def new_person(self, created: date, team: str, role: str, manager, start_date: date) -> str:
        pid = f"P{self.n_person:03d}"
        self.n_person += 1
        office = self.w.value_at(team, "office", created)
        self.w.add(Entity(pid, "person", self.person_name(), created),
                   {"team": team, "role": role, "manager": manager,
                    "start_date": start_date.isoformat(), "office": office})
        return pid

    def random_role(self, mgmt=False) -> str:
        if mgmt:
            return self.rng.choice(ROLE_LADDERS["mgmt"][:2])
        track = self.rng.choices(["eng", "data", "product", "design"], [50, 20, 15, 15])[0]
        ladder = ROLE_LADDERS[track]
        return ladder[self.rng.choices(range(len(ladder)), [45, 35, 15, 5][:len(ladder)])[0]]

    def new_project(self, created: date, team: str, status: str, start: date) -> str:
        pid = f"J{self.n_project:03d}"
        self.n_project += 1
        name = self.codename()
        members = [p for p in self.w.team_members_at(team, created) if p != self.ceo]
        owner = self.rng.choice(members)
        client = (self.rng.choice(self.w.of_kind("client", created))
                  if self.rng.random() < self.cfg.external_project_frac else None)
        end = start + timedelta(days=self.rng.randint(180, 540))
        self.w.add(Entity(pid, "project", name, created, aliases=[f"Project {name}"]),
                   {"owner": owner, "team": team,
                    "budget": _round10k(self.rng.uniform(300_000, 4_800_000)),
                    "status": status, "start_date": start.isoformat(),
                    "end_date": end.isoformat(), "client": client})
        return pid

    def new_contract(self, created: date, project: str, signed: date) -> str:
        cid = self.contract_id()
        self.w.add(Entity(cid, "contract", cid, created),
                   {"client": self.w.value_at(project, "client", created),
                    "value": _round10k(self.rng.uniform(100_000, 2_000_000)),
                    "signed_date": signed.isoformat(), "project": project})
        return cid

    def build_initial(self) -> None:
        cfg, rng, w = self.cfg, self.rng, self.w
        cities = rng.sample(CITIES, cfg.n_offices)
        offices = []
        for i, city in enumerate(cities):
            oid = f"O{i:02d}"
            w.add(Entity(oid, "office", city, START, aliases=[f"{city} office"]), {"head": None})
            offices.append(oid)

        # Leadership "team" holds only the CEO; it is excluded from questions.
        self.leadership = "T00"
        w.add(Entity("T00", "team", "Leadership", START), {"office": offices[0], "lead": None})
        team_names = rng.sample(TEAM_NAMES, cfg.n_teams)
        teams = []
        for i, tname in enumerate(team_names):
            tid = f"T{i + 1:02d}"
            office = offices[i] if i < len(offices) else rng.choice(offices)
            w.add(Entity(tid, "team", tname, START, aliases=[f"{tname} team"]),
                  {"office": office, "lead": None})
            teams.append(tid)

        def old_start():
            return _rand_date(rng, date(2014, 1, 1), date(2023, 11, 30))

        self.ceo = self.new_person(START, "T00", CEO_ROLE, None, old_start())
        w.timeline[("T00", "lead")] = [(START, self.ceo)]
        leads = {}
        for t in teams:
            lead = self.new_person(START, t, self.random_role(mgmt=True), None, old_start())
            leads[t] = lead
            w.timeline[(t, "lead")] = [(START, lead)]
        # office heads: one team lead per office reports to the CEO
        heads = {}
        for o in offices:
            cand = [leads[t] for t in teams if w.value_at(t, "office", START) == o]
            heads[o] = rng.choice(cand)
            w.timeline[(o, "head")] = [(START, heads[o])]
        for t in teams:
            o = w.value_at(t, "office", START)
            mgr = self.ceo if heads[o] == leads[t] else heads[o]
            w.timeline[(leads[t], "manager")] = [(START, mgr)]
        # members: every team gets at least 3, the rest are spread randomly
        n_members = cfg.n_people - cfg.n_teams
        alloc = teams * 3 + [rng.choice(teams) for _ in range(n_members - 3 * len(teams))]
        for t in alloc:
            self.new_person(START, t, self.random_role(), leads[t], old_start())

        regular = self.regular_people(START)
        for i in range(cfg.n_clients):
            while True:
                pre, suf = rng.choice(CLIENT_PREFIXES), rng.choice(CLIENT_SUFFIXES)
                name = f"{pre} {suf}"
                if name not in {e.name for e in w.entities.values()}:
                    break
            cid = f"K{i:02d}"
            w.add(Entity(cid, "client", name, START),
                  {"industry": INDUSTRIES[suf], "hq_city": rng.choice(CITIES),
                   "account_manager": rng.choice(regular)})

        for _ in range(cfg.n_projects):
            status = "active" if rng.random() < 0.7 else "planning"
            start = (_rand_date(rng, date(2023, 1, 1), date(2023, 12, 15)) if status == "active"
                     else START + timedelta(days=rng.randint(20, 90)))
            self.new_project(START, rng.choice(teams), status, start)

        ext = [p for p in w.of_kind("project") if w.value_at(p, "client", START)]
        for _ in range(cfg.n_contracts):
            self.new_contract(START, rng.choice(ext),
                              _rand_date(rng, date(2022, 6, 1), date(2023, 12, 20)))

    def regular_people(self, d: date) -> list[str]:
        """People who are not the CEO, a team lead or an office head at d."""
        leads = {self.w.value_at(t, "lead", d) for t in self.w.of_kind("team", d)}
        heads = {self.w.value_at(o, "head", d) for o in self.w.of_kind("office", d)}
        return [p for p in self.w.of_kind("person", d)
                if p != self.ceo and p not in leads and p not in heads]

    def regular_teams(self, d: date) -> list[str]:
        return [t for t in self.w.of_kind("team", d) if t != self.leadership]

    # -- events --------------------------------------------------------------
    def can_change(self, facts, d: date) -> bool:
        return all((d - self.w.last_change(e, a)).days >= MIN_GAP_DAYS for e, a in facts)

    def apply(self, d: date, etype: str, subject: str, changes, info=None) -> bool:
        """Apply a list of (eid, attr, new) changes atomically, or skip the
        event if any fact changed too recently or the change is a no-op."""
        changes = [(e, a, v) for e, a, v in changes if self.w.value_at(e, a, d) != v]
        if not changes or not self.can_change([(e, a) for e, a, _ in changes], d):
            return False
        if len({(e, a) for e, a, _ in changes}) != len(changes):
            return False
        recorded = []
        for e, a, v in changes:
            recorded.append((e, a, self.w.value_at(e, a, d), v))
            self.w.set(e, a, d, v)
        self.w.events.append(Event(f"E{self.n_event:04d}", d, etype, subject, recorded,
                                   info or {}))
        self.n_event += 1
        return True

    def try_event(self, etype: str, d: date) -> bool:
        w, rng = self.w, self.rng
        open_projects = [p for p in w.of_kind("project", d)
                         if w.value_at(p, "status", d) in OPEN_STATUSES]
        if etype == "budget_revision":
            if not open_projects:
                return False
            p = rng.choice(open_projects)
            old = w.value_at(p, "budget", d)
            u = rng.uniform(0.08, 0.4) * (1 if rng.random() < 0.65 else -1)
            return self.apply(d, etype, p, [(p, "budget", _round10k(old * (1 + u)))])
        if etype == "owner_change":
            if not open_projects:
                return False
            p = rng.choice(open_projects)
            team = w.value_at(p, "team", d)
            cand = [x for x in w.team_members_at(team, d)
                    if x not in (w.value_at(p, "owner", d), self.ceo)]
            return bool(cand) and self.apply(d, etype, p, [(p, "owner", rng.choice(cand))])
        if etype == "status_change":
            cands = []
            for p in open_projects:
                s = w.value_at(p, "status", d)
                if s == "planning" and d >= date.fromisoformat(w.value_at(p, "start_date", d)):
                    cands.append((p, "active", 1.0))
                elif s == "active":
                    late = d >= date.fromisoformat(w.value_at(p, "end_date", d)) - timedelta(90)
                    cands += [(p, "on hold", 0.25), (p, "completed", 1.2 if late else 0.15),
                              (p, "cancelled", 0.05)]
                elif s == "on hold":
                    cands += [(p, "active", 0.6), (p, "cancelled", 0.4)]
            if not cands:
                return False
            p, new, _ = rng.choices(cands, [c[2] for c in cands])[0]
            ch = [(p, "status", new)]
            if new == "completed":
                ch.append((p, "end_date", d.isoformat()))
            return self.apply(d, etype, p, ch)
        if etype == "end_date_change":
            cand = [p for p in open_projects if w.value_at(p, "status", d) != "on hold"]
            if not cand:
                return False
            p = rng.choice(cand)
            new = date.fromisoformat(w.value_at(p, "end_date", d)) + timedelta(
                days=rng.randint(30, 120))
            return self.apply(d, etype, p, [(p, "end_date", new.isoformat())])
        if etype == "contract_amendment":
            cs = w.of_kind("contract", d)
            c = rng.choice(cs)
            old = w.value_at(c, "value", d)
            u = rng.uniform(0.1, 0.3) * (1 if rng.random() < 0.7 else -1)
            return self.apply(d, etype, c, [(c, "value", _round10k(old * (1 + u)))])
        if etype == "account_manager_change":
            k = rng.choice(w.of_kind("client", d))
            cand = [x for x in self.regular_people(d) if x != w.value_at(k, "account_manager", d)]
            return self.apply(d, etype, k, [(k, "account_manager", rng.choice(cand))])
        if etype == "promotion":
            cand = []
            for x in w.of_kind("person", d):
                role = w.value_at(x, "role", d)
                if x == self.ceo:
                    continue
                ladder = ROLE_LADDERS[ROLE_TRACK[role]]
                if ladder.index(role) + 1 < len(ladder):
                    cand.append((x, ladder[ladder.index(role) + 1]))
            if not cand:
                return False
            x, new = rng.choice(cand)
            return self.apply(d, etype, x, [(x, "role", new)])
        if etype == "transfer":
            x = rng.choice(self.regular_people(d))
            old_team = w.value_at(x, "team", d)
            t = rng.choice([t for t in self.regular_teams(d) if t != old_team])
            return self.apply(d, etype, x, [
                (x, "team", t), (x, "office", w.value_at(t, "office", d)),
                (x, "manager", w.value_at(t, "lead", d))], {"from_team": old_team})
        if etype == "team_lead_change":
            heads = {w.value_at(o, "head", d) for o in w.of_kind("office", d)}
            teams = [t for t in self.regular_teams(d) if w.value_at(t, "lead", d) not in heads]
            if not teams:
                return False
            t = rng.choice(teams)
            old = w.value_at(t, "lead", d)
            members = [x for x in w.team_members_at(t, d) if x != old]
            if not members:
                return False
            new = rng.choice(members)
            office = w.value_at(t, "office", d)
            ch = [(t, "lead", new), (new, "manager", w.value_at(office, "head", d)),
                  (old, "manager", new)]
            if ROLE_TRACK[w.value_at(new, "role", d)] != "mgmt":
                ch.append((new, "role", "Engineering Manager"))
            ch += [(m, "manager", new) for m in members if m != new]
            return self.apply(d, etype, t, ch, {"old_lead": old})
        if etype == "office_head_change":
            o = rng.choice(w.of_kind("office", d))
            old = w.value_at(o, "head", d)
            leads = [w.value_at(t, "lead", d) for t in w.office_teams_at(o, d)
                     if t != self.leadership]
            cand = [x for x in leads if x != old]
            if not cand:
                return False
            new = rng.choice(cand)
            ch = [(o, "head", new), (new, "manager", self.ceo), (old, "manager", new)]
            ch += [(x, "manager", new) for x in leads if x not in (new, old)]
            return self.apply(d, etype, o, ch, {"old_head": old})
        if etype == "hire":
            t = rng.choice(self.regular_teams(d))
            # a hire is an entity creation, not a change, so it bypasses apply()
            pid = self.new_person(d, t, self.random_role(), w.value_at(t, "lead", d),
                                  d + timedelta(days=14))
            w.events.append(Event(f"E{self.n_event:04d}", d, etype, pid, [], {}))
            self.n_event += 1
            return True
        if etype == "new_project":
            t = rng.choice(self.regular_teams(d))
            pid = self.new_project(d, t, "planning", d + timedelta(days=rng.randint(20, 75)))
            w.events.append(Event(f"E{self.n_event:04d}", d, etype, pid, [], {}))
            self.n_event += 1
            return True
        if etype == "new_contract":
            ext = [p for p in open_projects if w.value_at(p, "client", d)]
            if not ext:
                return False
            cid = self.new_contract(d, rng.choice(ext), d)
            w.events.append(Event(f"E{self.n_event:04d}", d, etype, cid, [], {}))
            self.n_event += 1
            return True
        raise ValueError(etype)

    def run_events(self) -> None:
        cfg = self.cfg
        y, m = START.year, START.month
        while date(y, m, 1) <= EVENTS_END:
            first = max(date(y, m, 1), EVENTS_FROM)
            last = min(date(y, m, calendar.monthrange(y, m)[1]), EVENTS_END)
            in_q4 = first >= Q4_WAVE[0]
            mrng = sub_rng(self.seed, "month", y, m)
            planned = []
            for etype, rate in cfg.rates.items():
                r = rate * (cfg.q4_boost.get(etype, 1.0) if in_q4 else 1.0)
                n = int(r) + (1 if mrng.random() < r - int(r) else 0)
                for _ in range(n):
                    planned.append((_rand_date(mrng, first, last), etype))
            planned.sort()
            for d, etype in planned:
                for _ in range(10):          # retry with another target if blocked
                    if self.try_event(etype, d):
                        break
            m += 1
            if m == 13:
                y, m = y + 1, 1


def build_world(seed: int, cfg: WorldConfig | None = None) -> World:
    b = _Builder(seed, cfg or WorldConfig())
    b.build_initial()
    b.run_events()
    b.w.ceo = b.ceo
    b.w.leadership = b.leadership
    return b.w
