"""Raw documents: the workspace's source of truth, rendered from templates.

Every document records the facts it *asserts*, as (entity, attr, canonical
value, rendered text). Two invariants make "the gold answer is the latest value
in the raw files" checkable (see tests/test_workspace.py):
  - consistency: every assertion equals the timeline value at the doc's date;
  - coverage: every fact's final value is asserted by some doc dated on or
    after its last change.
Documents never restate an *old* value as current. Old values appear only in
explicit "from X to Y" phrasing, as realistic distractors.

Document kinds (dates in the header are always ISO; dates and money in the body
use mixed surface forms):
  - start-of-year directories: office directory, team rosters, client accounts
  - project status reports, every ~3-5 months per open project
  - one announcement per change event: email, weekly meeting notes (bundling a
    week's project changes per office), kickoff memo, contract summary,
    HR announcement
  - year-end reviews: office portfolio, office headcount, team roster, client
    account review. These make aggregate questions answerable from raw files
  - chatter: offsites, IT notices, kudos. They mention names and codenames but
    assert nothing (pure distractors)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from .entities import (
    DATE_ATTRS, END, EVENTS_END, MONEY_ATTRS, START, World,
)
from .formats import (
    N_DATE_STYLES, N_MONEY_STYLES, render_date, render_money, sub_rng,
)

MIN_WORDS = 110
MAX_WORDS = 330


@dataclass
class Doc:
    path: str
    date: date
    dtype: str
    title: str
    text: str = ""
    asserts: list = field(default_factory=list)   # (eid, attr, value, rendered)
    lists: list = field(default_factory=list)     # (anchor_eid, list_name, [ids])
    mentions: list = field(default_factory=list)  # entity ids named in the text

    def to_json(self) -> dict:
        return {"path": self.path, "date": self.date.isoformat(), "dtype": self.dtype,
                "title": self.title, "text": self.text,
                "asserts": [list(a) for a in self.asserts],
                "lists": [[a, n, list(ids)] for a, n, ids in self.lists],
                "mentions": self.mentions}


# --------------------------------------------------------------------------- #
# Filler: generic sentences that assert nothing
# --------------------------------------------------------------------------- #
FILLER = {
    "email": [
        "Let me know if you have any questions.",
        "Thanks to everyone who contributed to the review this week.",
        "I will share the updated plan in the usual channel.",
        "Please flag any concerns before the end of the week.",
        "The standard reporting templates still apply.",
        "Finance has been looped in and has the details.",
        "We will revisit this at the next quarterly planning session.",
        "Apologies for the late notice on this one.",
        "Happy to walk anyone through the reasoning in more detail.",
        "Please update any dashboards or trackers that reference this.",
        "No action is needed from most of you.",
        "The decision log has been updated accordingly.",
    ],
    "meeting": [
        "Action items from last week were reviewed and owners confirmed.",
        "The next sync is scheduled for the same time next week.",
        "Risks were discussed briefly; nothing new was escalated.",
        "The group agreed to keep the agenda short going forward.",
        "Notes are shared with everyone on the invite.",
        "Hiring updates were deferred to the staff meeting.",
        "A reminder was given to keep the tracker current.",
        "Questions on tooling were taken offline.",
        "The retro format will be tried again next month.",
        "Several attendees joined remotely.",
    ],
    "report": [
        "Overall delivery remains on the agreed scope.",
        "Dependencies with other teams are tracked in the shared board.",
        "Stakeholder communication continues on the usual cadence.",
        "No new security findings were raised this period.",
        "Testing coverage improved slightly over the last cycle.",
        "Documentation was refreshed for the current milestone.",
        "The team held its regular demo for interested stakeholders.",
        "Vendor coordination is proceeding as planned.",
        "Metrics are reviewed weekly by the project leads.",
        "Open questions are listed in the project wiki.",
    ],
    "hr": [
        "Please make them feel welcome.",
        "Managers have been briefed on the change.",
        "Reach out to People Operations with any questions.",
        "Updated org charts will follow in the internal directory.",
        "Thank you for your continued support during these changes.",
        "Onboarding materials are available on the intranet.",
        "Access requests are being handled by IT.",
        "This announcement is shared with the whole company.",
    ],
    "chatter": [
        "Snacks will be provided.",
        "Please RSVP by Thursday so we can plan for numbers.",
        "Remote colleagues can join through the usual video link.",
        "The kitchen on the third floor will be closed for cleaning.",
        "Remember to bring your badge.",
        "Photos from last time are on the shared drive.",
        "Volunteers are welcome to help with setup.",
        "Parking will be limited that day.",
        "Feedback on the format is always appreciated.",
        "The playlist is open for suggestions.",
    ],
}
# Shared fallback bank, used after the kind-specific bank runs out. Sentences
# are drawn without replacement, so a document never repeats one.
GENERAL = [
    "As always, the shared calendar has the latest meeting times.",
    "We are trying to keep these updates short and easy to scan.",
    "If anything here looks wrong, reply directly and we will correct it.",
    "The quarterly planning templates are available in the shared drive.",
    "A recording of the last review is available for anyone who missed it.",
    "Please keep customer details out of public channels.",
    "The internal wiki has background reading for new joiners.",
    "We will send a follow-up once the remaining questions are answered.",
    "Holiday schedules may affect response times over the coming weeks.",
    "Thanks for keeping the trackers and tickets up to date.",
    "The support rotation for next month has been posted.",
    "Any travel requests should go through the usual approval flow.",
    "Office hours for questions continue on Thursday afternoons.",
    "This summary will also be linked from the weekly newsletter.",
    "Comments on the draft process document are still welcome.",
    "A short survey about tooling will go out later this month.",
]
MAX_PASSING = 2
FILLER_KIND = {"status_report": "report", "email": "email", "meeting_notes": "meeting",
               "kickoff_memo": "report", "contract_summary": "report",
               "hr_announcement": "hr", "directory": "hr", "team_roster": "hr",
               "client_accounts": "report", "year_end_review": "report", "chatter": "chatter"}
PASSING = [
    "Separately, {p} shared a short demo of {j}.",
    "Thanks to {p} for help with the {j} review.",
    "{p} mentioned that {j} will present at the next all-hands.",
    "A reminder that {p} is collecting feedback on {j}.",
    "{p} and {q} are pairing on documentation this week.",
]


class _Writer:
    def __init__(self, world: World):
        self.w = world
        self.docs: list[Doc] = []
        self.counter = 0

    # -- rendering helpers ---------------------------------------------------
    def rv(self, attr: str, value, rng) -> str:
        """Render a canonical value as text, in a random surface form."""
        w = self.w
        if attr in MONEY_ATTRS:
            return render_money(value, rng.randrange(N_MONEY_STYLES))
        if attr in DATE_ATTRS:
            return render_date(date.fromisoformat(value), rng.randrange(N_DATE_STYLES))
        if attr == "project":
            return w.display(value)
        if attr in ("head", "lead", "office", "team", "manager", "owner", "client",
                    "account_manager"):
            return w.name(value)
        return str(value)

    def new_doc(self, d: date, dtype: str, title: str, slug: str) -> Doc:
        self.counter += 1
        path = f"files/{d:%Y}/{d:%m}/{d.isoformat()}-{dtype.replace('_', '-')}-{slug}-{self.counter:04d}.md"
        return Doc(path=path, date=d, dtype=dtype, title=title)

    def assert_fact(self, doc: Doc, eid: str, attr: str, rng) -> str:
        """Record that `doc` asserts eid.attr (its value at doc.date) and return
        the rendered text, which the caller must place in the body."""
        value = self.w.value_at(eid, attr, doc.date)
        assert value is not None, (eid, attr, doc.date)
        rendered = self.rv(attr, value, rng)
        doc.asserts.append((eid, attr, value, rendered))
        return rendered

    def finish(self, doc: Doc, body: list[str], rng, author: str | None = None) -> Doc:
        """Assemble header + body + filler up to a random target length."""
        w = self.w
        header = [f"# {doc.title}", f"Date: {doc.date.isoformat()}",
                  f"Type: {doc.dtype.replace('_', ' ')}"]
        if author:
            header.append(f"From: {w.name(author)}")
            doc.mentions.append(author)
        lines = list(body)
        target = rng.randint(MIN_WORDS + 10, 220)
        bank = FILLER[FILLER_KIND[doc.dtype]]
        pool = rng.sample(bank, len(bank)) + rng.sample(GENERAL, len(GENERAL))
        people = w.of_kind("person", doc.date)
        projects = w.of_kind("project", doc.date)

        def n_words():
            return sum(len(x.split()) for x in header + lines)

        n_passing = 0
        while n_words() < target and pool:
            if n_passing < MAX_PASSING and rng.random() < 0.2:
                n_passing += 1
                p, q = rng.sample(people, 2)
                j = rng.choice(projects)
                lines.append(rng.choice(PASSING).format(p=w.name(p), q=w.name(q),
                                                        j=w.display(j)))
                doc.mentions += [p, q, j]
            else:
                lines.append(pool.pop(0))      # kind-specific sentences first
        doc.text = "\n".join(header) + "\n\n" + "\n".join(lines) + "\n"
        doc.mentions = sorted(set(doc.mentions))
        self.docs.append(doc)
        return doc

    def anyone(self, d: date, rng) -> str:
        return rng.choice(self.w.of_kind("person", d))

    # -- start-of-year directories ------------------------------------------
    def directories(self) -> None:
        w = self.w
        for o in w.of_kind("office", START):
            rng = sub_rng(w.seed, "doc-office-dir", o)
            d = START + timedelta(days=1)
            doc = self.new_doc(d, "directory", f"Office directory: {w.name(o)}",
                               w.name(o).lower())
            body = [f"Office: {w.name(o)}",
                    f"Office head: {self.assert_fact(doc, o, 'head', rng)}", "",
                    f"Teams based in {w.name(o)}:"]
            for t in w.office_teams_at(o, d):
                if t == w.leadership:
                    continue
                self.assert_fact(doc, t, "office", rng)
                body.append(f"- {w.name(t)} (lead: {self.assert_fact(doc, t, 'lead', rng)})")
                doc.mentions.append(t)
            doc.mentions.append(o)
            self.finish(doc, body, rng)

        for t in w.of_kind("team", START):
            if t == w.leadership:
                continue
            rng = sub_rng(w.seed, "doc-roster-start", t)
            d = START + timedelta(days=2)
            doc = self.new_doc(d, "team_roster", f"Team roster: {w.name(t)}",
                               w.name(t).lower().replace(" ", "-"))
            body = [f"Team: {w.name(t)}", f"Lead: {w.name(w.value_at(t, 'lead', d))}", "",
                    "Members:"]
            for p in w.team_members_at(t, d):
                self.assert_fact(doc, p, "team", rng)
                office = self.assert_fact(doc, p, "office", rng)
                role = self.assert_fact(doc, p, "role", rng)
                start = self.assert_fact(doc, p, "start_date", rng)
                mgr = self.assert_fact(doc, p, "manager", rng)
                body.append(f"- {w.name(p)}: {role}, {office} office, started {start}, "
                            f"reports to {mgr}")
                doc.mentions.append(p)
            doc.mentions.append(t)
            self.finish(doc, body, rng)

        for k in w.of_kind("client", START):
            rng = sub_rng(w.seed, "doc-client-dir", k)
            d = START + timedelta(days=3)
            doc = self.new_doc(d, "client_accounts", f"Client account: {w.name(k)}",
                               w.name(k).lower().replace(" ", "-"))
            body = [f"Client: {w.name(k)}",
                    f"Industry: {self.assert_fact(doc, k, 'industry', rng)}",
                    f"Headquarters: {self.assert_fact(doc, k, 'hq_city', rng)}",
                    f"Account manager: {self.assert_fact(doc, k, 'account_manager', rng)}",
                    "", "Contracts on file:"]
            for c in w.client_contracts_at(k, d):
                self.assert_fact(doc, c, "client", rng)
                val = self.assert_fact(doc, c, "value", rng)
                signed = self.assert_fact(doc, c, "signed_date", rng)
                proj = self.assert_fact(doc, c, "project", rng)
                body.append(f"- {c}: {val}, signed {signed}, covering {proj}")
                doc.mentions += [c, w.latest(c, "project")]
            if len(body) == 6:
                body.append("- none yet")
            doc.mentions.append(k)
            self.finish(doc, body, rng)

    # -- status reports ------------------------------------------------------
    def status_report(self, p: str, d: date, rng) -> None:
        w = self.w
        doc = self.new_doc(d, "status_report", f"Status report: {w.display(p)}",
                           w.name(p).lower().replace(" ", "-"))
        owner = self.assert_fact(doc, p, "owner", rng)
        body = [f"Project: {w.display(p)}",
                f"Owner: {owner}",
                f"Team: {self.assert_fact(doc, p, 'team', rng)}",
                f"Status: {self.assert_fact(doc, p, 'status', rng)}",
                f"Budget: {self.assert_fact(doc, p, 'budget', rng)}",
                f"Start date: {self.assert_fact(doc, p, 'start_date', rng)}",
                f"Target end date: {self.assert_fact(doc, p, 'end_date', rng)}"]
        if w.value_at(p, "client", d):
            body.append(f"Client: {self.assert_fact(doc, p, 'client', rng)}")
        else:
            body.append("Client: internal project")
        body.append("")
        doc.mentions.append(p)
        self.finish(doc, body, rng, author=w.value_at(p, "owner", d))

    def status_reports(self) -> None:
        w = self.w
        for p in w.of_kind("project"):
            rng = sub_rng(w.seed, "doc-status", p)
            d = w.entities[p].created + timedelta(days=rng.randint(4, 25))
            while d <= EVENTS_END:
                if w.value_at(p, "status", d) not in ("planning", "active", "on hold"):
                    break
                self.status_report(p, d, rng)
                d += timedelta(days=rng.randint(90, 150))

    # -- event announcements -------------------------------------------------
    def project_change_line(self, doc: Doc, ev, rng) -> str:
        w = self.w
        p = ev.subject
        name = w.display(p)
        ch = {a: (old, new) for _, a, old, new in ev.changes}
        if ev.type == "budget_revision":
            old = self.rv("budget", ch["budget"][0], rng)
            new = self.assert_fact(doc, p, "budget", rng)
            return rng.choice([
                f"The budget for {name} has been revised from {old} to {new}.",
                f"{name}: budget changed from {old} to {new} after the finance review.",
                f"Finance approved a new budget of {new} for {name} (previously {old}).",
            ])
        if ev.type == "owner_change":
            old = w.name(ch["owner"][0])
            new = self.assert_fact(doc, p, "owner", rng)
            return rng.choice([
                f"Ownership of {name} moves from {old} to {new}.",
                f"{new} is taking over as owner of {name}, replacing {old}.",
            ])
        if ev.type == "status_change":
            new = self.assert_fact(doc, p, "status", rng)
            if new == "completed":
                end = self.assert_fact(doc, p, "end_date", rng)
                return f"{name} is now completed; it wrapped up on {end}."
            return rng.choice([f"{name} status changed to {new}.",
                               f"{name} is now {new}."])
        if ev.type == "end_date_change":
            old = self.rv("end_date", ch["end_date"][0], rng)
            new = self.assert_fact(doc, p, "end_date", rng)
            return rng.choice([
                f"The target end date for {name} has moved from {old} to {new}.",
                f"{name} will now finish by {new} instead of {old}.",
            ])
        raise ValueError(ev.type)

    def events(self) -> None:
        w = self.w
        project_types = ("budget_revision", "owner_change", "status_change", "end_date_change")
        meetings: dict[tuple, list] = {}
        for ev in w.events:
            rng = sub_rng(w.seed, "doc-event", ev.id)
            if ev.type in project_types:
                if rng.random() < 0.45:
                    office = w.value_at(w.value_at(ev.subject, "team", ev.date), "office",
                                        ev.date)
                    iso = ev.date.isocalendar()
                    meetings.setdefault((iso[0], iso[1], office), []).append(ev)
                    continue
                doc = self.new_doc(ev.date, "email", f"Update: {w.display(ev.subject)}",
                                   w.name(ev.subject).lower().replace(" ", "-"))
                line = self.project_change_line(doc, ev, rng)
                doc.mentions.append(ev.subject)
                author = w.value_at(ev.subject, "owner", ev.date)
                self.finish(doc, ["Hi all,", "", line, ""], rng, author=author)
            elif ev.type == "new_project":
                self.kickoff(ev, rng)
            elif ev.type in ("new_contract", "contract_amendment"):
                self.contract_doc(ev, rng)
            elif ev.type == "account_manager_change":
                self.handover(ev, rng)
            else:
                self.hr(ev, rng)

        for (y, wk, office), evs in sorted(meetings.items(), key=lambda kv: (
                kv[1][-1].date, kv[0][2])):
            d = max(e.date for e in evs)
            rng = sub_rng(w.seed, "doc-meeting", y, wk, office)
            doc = self.new_doc(d, "meeting_notes",
                               f"Weekly portfolio sync: {w.name(office)} office",
                               w.name(office).lower())
            attendees = rng.sample(w.office_people_at(office, d),
                                   min(4, len(w.office_people_at(office, d))))
            body = ["Attendees: " + ", ".join(w.name(a) for a in attendees), "",
                    "Decisions:"]
            for i, ev in enumerate(evs, 1):
                body.append(f"{i}. " + self.project_change_line(doc, ev, rng))
                doc.mentions.append(ev.subject)
            doc.mentions += attendees + [office]
            self.finish(doc, body + [""], rng)

    def kickoff(self, ev, rng) -> None:
        w = self.w
        p = ev.subject
        doc = self.new_doc(ev.date, "kickoff_memo", f"Kickoff: {w.display(p)}",
                           w.name(p).lower().replace(" ", "-"))
        body = [f"We are starting a new project, {w.display(p)}.",
                f"It will be run by the {self.assert_fact(doc, p, 'team', rng)} team, "
                f"with {self.assert_fact(doc, p, 'owner', rng)} as owner.",
                f"Approved budget: {self.assert_fact(doc, p, 'budget', rng)}.",
                f"Current status: {self.assert_fact(doc, p, 'status', rng)}.",
                f"Planned start: {self.assert_fact(doc, p, 'start_date', rng)}; "
                f"target end: {self.assert_fact(doc, p, 'end_date', rng)}."]
        if w.value_at(p, "client", ev.date):
            body.append(f"The client is {self.assert_fact(doc, p, 'client', rng)}.")
        else:
            body.append("This is an internal project with no external client.")
        doc.mentions.append(p)
        self.finish(doc, body + [""], rng, author=w.value_at(p, "owner", ev.date))

    def contract_doc(self, ev, rng) -> None:
        w = self.w
        c = ev.subject
        doc = self.new_doc(ev.date, "contract_summary", f"Contract summary: {c}", c.lower())
        client = self.assert_fact(doc, c, "client", rng)
        proj = self.assert_fact(doc, c, "project", rng)
        if ev.type == "new_contract":
            body = [f"Contract {c} with {client} has been signed.",
                    f"Signed on: {self.assert_fact(doc, c, 'signed_date', rng)}",
                    f"Total value: {self.assert_fact(doc, c, 'value', rng)}",
                    f"Linked project: {proj}"]
        else:
            old = self.rv("value", ev.changes[0][2], rng)
            new = self.assert_fact(doc, c, "value", rng)
            body = [f"Contract {c} with {client} has been amended.",
                    f"The contract value changes from {old} to {new}.",
                    f"Linked project: {proj}"]
        doc.mentions += [c, w.latest(c, "project"), w.latest(c, "client")]
        self.finish(doc, body + [""], rng,
                    author=w.value_at(w.value_at(c, "client", ev.date), "account_manager",
                                      ev.date))

    def handover(self, ev, rng) -> None:
        w = self.w
        k = ev.subject
        old = w.name(ev.changes[0][2])
        doc = self.new_doc(ev.date, "email", f"Account handover: {w.name(k)}",
                           w.name(k).lower().replace(" ", "-"))
        new = self.assert_fact(doc, k, "account_manager", rng)
        body = ["Hi all,", "",
                f"{new} is taking over as account manager for {w.name(k)} from {old}.",
                "Please route client requests accordingly.", ""]
        doc.mentions += [k, ev.changes[0][2], w.value_at(k, "account_manager", ev.date)]
        self.finish(doc, body, rng)

    def hr(self, ev, rng) -> None:
        w = self.w
        d = ev.date
        x = ev.subject
        doc = self.new_doc(d, "hr_announcement",
                           {"promotion": "Promotion", "transfer": "Team change",
                            "team_lead_change": "New team lead",
                            "office_head_change": "New office head",
                            "hire": "Welcome"}[ev.type] + f": {w.name(x)}",
                           w.name(x).lower().replace(" ", "-"))
        if ev.type == "promotion":
            body = [f"Please join us in congratulating {w.name(x)}, who has been promoted "
                    f"to {self.assert_fact(doc, x, 'role', rng)}."]
            doc.mentions.append(x)
        elif ev.type == "transfer":
            body = [f"{w.name(x)} is moving from the {w.name(ev.info['from_team'])} team to "
                    f"the {self.assert_fact(doc, x, 'team', rng)} team, based in the "
                    f"{self.assert_fact(doc, x, 'office', rng)} office, and will report to "
                    f"{self.assert_fact(doc, x, 'manager', rng)}."]
            doc.mentions += [x, ev.info["from_team"], w.value_at(x, "team", d)]
        elif ev.type == "hire":
            body = [f"Please welcome {w.name(x)}, who joins the "
                    f"{self.assert_fact(doc, x, 'team', rng)} team in "
                    f"{self.assert_fact(doc, x, 'office', rng)} as "
                    f"{self.assert_fact(doc, x, 'role', rng)}.",
                    f"Their first day is {self.assert_fact(doc, x, 'start_date', rng)}, and "
                    f"they will report to {self.assert_fact(doc, x, 'manager', rng)}."]
            doc.mentions += [x, w.value_at(x, "team", d)]
        elif ev.type in ("team_lead_change", "office_head_change"):
            attr = "lead" if ev.type == "team_lead_change" else "head"
            new = self.assert_fact(doc, x, attr, rng)
            new_id = w.value_at(x, attr, d)
            what = (f"lead of the {w.name(x)} team" if attr == "lead"
                    else f"head of the {w.name(x)} office")
            body = [f"Effective today, {new} takes over as {what}."]
            if (new_id, "role") in {(e, a) for e, a, _, _ in ev.changes}:
                body.append(f"{new} moves into the role of "
                            f"{self.assert_fact(doc, new_id, 'role', rng)}.")
            body.append(f"{new} will report to {self.assert_fact(doc, new_id, 'manager', rng)}.")
            others = [e for e, a, _, _ in ev.changes if a == "manager" and e != new_id]
            if others:
                body.append(f"The following people now report to {new}:")
                for e in others:
                    self.assert_fact(doc, e, "manager", rng)
                    body.append(f"- {w.name(e)}")
                    doc.mentions.append(e)
            doc.mentions += [x, new_id]
        else:
            raise ValueError(ev.type)
        self.finish(doc, body + [""], rng)

    # -- year-end reviews ----------------------------------------------------
    def year_end(self) -> None:
        w = self.w
        base = EVENTS_END + timedelta(days=3)
        for i, o in enumerate(w.of_kind("office")):
            rng = sub_rng(w.seed, "doc-ye-portfolio", o)
            d = base + timedelta(days=i % 5)
            doc = self.new_doc(d, "year_end_review", f"Year-end portfolio review: "
                               f"{w.name(o)} office", w.name(o).lower())
            projs = w.office_projects_at(o, d)
            body = [f"Projects run by teams in the {w.name(o)} office, with current status:"]
            for p in projs:
                st = self.assert_fact(doc, p, "status", rng)
                tm = self.assert_fact(doc, p, "team", rng)
                body.append(f"- {w.display(p)} ({tm}): {st}")
            doc.lists.append((o, "projects", projs))
            doc.mentions += projs + [o]
            self.finish(doc, body + [""], rng)

            rng = sub_rng(w.seed, "doc-ye-headcount", o)
            d = base + timedelta(days=5 + i % 5)
            doc = self.new_doc(d, "year_end_review", f"Year-end headcount: {w.name(o)} office",
                               w.name(o).lower())
            people = [p for p in w.office_people_at(o, d) if p != w.ceo]
            body = [f"Office head: {self.assert_fact(doc, o, 'head', rng)}",
                    f"Everyone based in the {w.name(o)} office at year end:"]
            for p in people:
                self.assert_fact(doc, p, "office", rng)
                body.append(f"- {w.name(p)} ({self.assert_fact(doc, p, 'team', rng)})")
            doc.lists.append((o, "people", people))
            doc.mentions += people + [o]
            self.finish(doc, body + [""], rng)

        for i, t in enumerate(w.of_kind("team")):
            if t == w.leadership:
                continue
            rng = sub_rng(w.seed, "doc-ye-roster", t)
            d = base + timedelta(days=2 + i % 7)
            doc = self.new_doc(d, "team_roster", f"Year-end roster: {w.name(t)}",
                               w.name(t).lower().replace(" ", "-"))
            members = w.team_members_at(t, d)
            body = [f"Team: {w.name(t)} ({self.assert_fact(doc, t, 'office', rng)} office)",
                    f"Lead: {self.assert_fact(doc, t, 'lead', rng)}", "", "Members:"]
            for p in members:
                self.assert_fact(doc, p, "team", rng)
                role = self.assert_fact(doc, p, "role", rng)
                mgr = self.assert_fact(doc, p, "manager", rng)
                body.append(f"- {w.name(p)}: {role}, reports to {mgr}")
            doc.lists.append((t, "members", members))
            projs = w.team_projects_at(t, d)
            body += ["", "Projects run by the team:"]
            for p in projs:
                self.assert_fact(doc, p, "team", rng)
                body.append(f"- {w.display(p)}: {self.assert_fact(doc, p, 'status', rng)}")
            if not projs:
                body.append("- none")
            doc.lists.append((t, "projects", projs))
            doc.mentions += members + projs + [t]
            self.finish(doc, body + [""], rng)

        for i, k in enumerate(w.of_kind("client")):
            rng = sub_rng(w.seed, "doc-ye-client", k)
            d = base + timedelta(days=4 + i % 9)
            doc = self.new_doc(d, "year_end_review", f"Year-end account review: {w.name(k)}",
                               w.name(k).lower().replace(" ", "-"))
            cs = w.client_contracts_at(k, d)
            body = [f"Client: {w.name(k)}",
                    f"Account manager: {self.assert_fact(doc, k, 'account_manager', rng)}",
                    "Contracts:"]
            for c in cs:
                self.assert_fact(doc, c, "client", rng)
                body.append(f"- {c}: {self.assert_fact(doc, c, 'value', rng)}")
            if not cs:
                body.append("- none")
            doc.lists.append((k, "contracts", cs))
            projs = [p for p in w.of_kind("project", d) if w.value_at(p, "client", d) == k]
            body.append("Projects delivered for this client:")
            for p in projs:
                self.assert_fact(doc, p, "client", rng)
                body.append(f"- {w.display(p)}: {self.assert_fact(doc, p, 'status', rng)}")
            if not projs:
                body.append("- none")
            doc.lists.append((k, "projects", projs))
            doc.mentions += cs + projs + [k]
            self.finish(doc, body + [""], rng)

    # -- chatter -------------------------------------------------------------
    def chatter(self, n: int) -> None:
        w = self.w
        topics = ["Team offsite", "All-hands reminder", "IT maintenance window",
                  "Kudos", "Lunch and learn", "Office move-in day", "Book club",
                  "Hackathon sign-ups"]
        for i in range(n):
            rng = sub_rng(w.seed, "doc-chatter", i)
            d = START + timedelta(days=rng.randint(7, (END - START).days - 20))
            topic = rng.choice(topics)
            doc = self.new_doc(d, "chatter", topic, topic.lower().replace(" ", "-"))
            people = w.of_kind("person", d)
            p, q = rng.sample(people, 2)
            j = rng.choice(w.of_kind("project", d))
            body = [f"Hi everyone, a quick note about the {topic.lower()}.",
                    rng.choice(PASSING).format(p=w.name(p), q=w.name(q), j=w.display(j))]
            doc.mentions += [p, q, j]
            self.finish(doc, body, rng)


def build_documents(world: World, n_chatter: int = 40) -> list[Doc]:
    wr = _Writer(world)
    wr.directories()
    wr.status_reports()
    wr.events()
    wr.year_end()
    wr.chatter(n_chatter)
    return sorted(wr.docs, key=lambda d: (d.date, d.path))
