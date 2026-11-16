"""A scripted oracle policy that proves every question is solvable in the env.

What the oracle is and isn't allowed to know (this is the point of the test):
  - PRIVILEGED about *what* to read. It knows whether a fact is fresh in the
    current notes version, and which raw documents assert the fact's final value.
  - NOT privileged about *getting* it. It must find the target with the same
    search tools and natural queries ("Project Heron budget"); the target must
    rank in the top k. It reads through the same 400-token cap, so the value
    must be visible in the text it got back. It has the same K-turn limit, and
    every call goes through the same <tool_call> text format and parser.
  - Its answer is rendered from what it read, then scored by the real reward.

So oracle accuracy < 100% means some question can't be answered within the
env's limits: a retrieval miss, truncation, or too few turns. It does not mean
the oracle is clever or dumb. Failures carry a reason.
"""

from __future__ import annotations

import json
import re

from .env.episode import Episode, EnvConfig
from .env.tools import ToolEnv
from .workspace.entities import MONEY_ATTRS, REF_ATTRS
from .workspace.notes import VERSIONS
from .workspace.questions import AGGREGATE, render_gold

ATTR_WORDS = {
    "budget": "budget", "owner": "owner", "status": "status", "end_date": "end date",
    "start_date": "start date", "team": "team", "client": "client", "role": "role",
    "manager": "manager reports to", "office": "office", "value": "contract value",
    "signed_date": "signed", "account_manager": "account manager", "hq_city": "headquarters",
    "lead": "lead", "head": "office head", "project": "project", "industry": "industry",
}
AGG_QUERY = {   # (anchor kind, list) -> query words for the year-end document
    ("office", "projects"): "year-end portfolio review projects",
    ("office", "people"): "year-end headcount",
    ("team", "members"): "year-end roster", ("team", "projects"): "year-end roster projects",
    ("client", "contracts"): "year-end account review contracts",
    ("client", "projects"): "year-end account review projects",
}
SEARCH_K = 10


class OracleFailure(Exception):
    pass


def call(name: str, **args) -> str:
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"


class Oracle:
    def __init__(self, ws, version: str, notes: bool):
        self.ws, self.w, self.version, self.notes = ws, ws.world, version, notes
        self.pages = {p["entity"]: p for p in ws.notes[version] if p["entity"]}
        self.fallbacks = {"since": 0, "roster": 0}   # how often each fallback query was needed
        self.by_fact = {}
        self.lists = {}
        for d in ws.docs:
            for eid, attr, value, rendered in d.asserts:
                self.by_fact.setdefault((eid, attr), []).append((d, value, rendered))
            for anchor, lname, ids in d.lists:
                self.lists.setdefault((anchor, lname), []).append((d, ids))

    # -- helpers --------------------------------------------------------------
    def query_name(self, eid: str) -> str:
        e = self.w.entities[eid]
        return {"project": f"Project {e.name}", "office": f"{e.name} office",
                "team": f"{e.name} team"}.get(e.kind, e.name)

    def note_fresh(self, eid: str, attr: str) -> bool:
        p = self.pages.get(eid)
        return (p is not None and attr in p["fields"]
                and p["fields"][attr] == self.w.latest(eid, attr))

    def _step(self, ep: Episode, text: str) -> str:
        obs, done = ep.step(text)
        if done:
            raise OracleFailure("out_of_turns")
        return obs

    def _find(self, ep, tool, query, target_ids, since=None) -> str:
        """Search; return the first target id that the results show."""
        args = {"query": query, "k": SEARCH_K}
        if since:
            args["since"] = since
        obs = self._step(ep, call(tool, **args))
        for line in obs.splitlines():
            for t in target_ids:
                if line.split(" | ")[0].endswith(" " + t):
                    return t
        return ""

    # -- one fact -------------------------------------------------------------
    def via_notes(self, ep, eid, attr):
        nid = self.pages[eid]["note_id"]
        if not self._find(ep, "search_notes", self.query_name(eid), [nid]):
            raise OracleFailure("note_not_retrieved")
        text = self._step(ep, call("read_note", note_id=nid))
        m = re.search(rf"^{attr}: (.+)$", text, re.MULTILINE)
        if not m:
            raise OracleFailure("note_field_not_visible")
        return self.parse_value(eid, attr, m.group(1).strip())

    def via_files(self, ep, eid, attr):
        final = self.w.latest(eid, attr)
        last = self.w.last_change(eid, attr)
        good = [(d, r) for d, v, r in self.by_fact.get((eid, attr), [])
                if v == final and d.date >= last]
        if not good:
            raise OracleFailure("no_raw_source")
        good.sort(key=lambda x: x[0].date, reverse=True)
        paths = [d.path for d, _ in good]
        # identifiers are searched alone ("C-4513"): attribute words like
        # "contract value" match every contract summary and outweigh the id
        q = (self.query_name(eid) if self.w.entities[eid].kind == "contract"
             else f"{self.query_name(eid)} {ATTR_WORDS[attr]}")
        hit = self._find(ep, "search_files", q, paths)
        if not hit:
            # fallback 1: the same query restricted to files since the last change
            self.fallbacks["since"] += 1
            hit = self._find(ep, "search_files", q, paths, since=last.isoformat())
        if not hit and self.w.entities[eid].kind in ("person", "team"):
            # fallback 2: person and team facts often live only in a team roster
            self.fallbacks["roster"] += 1
            name = self.w.entities[eid].name
            hit = self._find(ep, "search_files", f"{name} team roster", paths)
        if not hit:
            raise OracleFailure("file_not_retrieved")
        rendered = next(r for d, r in good if d.path == hit)
        text = self._step(ep, call("read_file", path=hit))
        if rendered not in text:
            raise OracleFailure("value_truncated")
        return final

    def parse_value(self, eid, attr, s):
        """Turn a structured note value back into a canonical value."""
        if attr in REF_ATTRS:
            kind = {"head": "person", "lead": "person", "manager": "person",
                    "owner": "person", "account_manager": "person", "office": "office",
                    "team": "team", "client": "client", "project": "project"}[attr]
            name = s[len("Project "):] if kind == "project" and s.startswith("Project ") else s
            for e in self.w.entities.values():
                if e.kind == kind and e.name == name:
                    return e.id
            raise OracleFailure("note_value_unparsed")
        if attr in MONEY_ATTRS:
            return int(s.replace("$", "").replace(",", ""))
        return s

    def fact(self, ep, eid, attr):
        if self.notes and self.note_fresh(eid, attr):
            return self.via_notes(ep, eid, attr)
        return self.via_files(ep, eid, attr)

    # -- aggregates -----------------------------------------------------------
    def aggregate(self, ep, q):
        kind, lname, status, _ = AGGREGATE[q["family"]]
        anchor = q["anchor"]
        if self.notes and q["notes_ok_by_version"][self.version]:
            nid = self.pages[anchor]["note_id"]
            if not self._find(ep, "search_notes", self.query_name(anchor), [nid]):
                raise OracleFailure("note_not_retrieved")
            text = self._step(ep, call("read_note", note_id=nid))
            m = re.search(rf"^{lname} \((\d+)\):\n((?:- .*\n?)*)", text, re.MULTILINE)
            if not m:
                raise OracleFailure("note_field_not_visible")
            if status is None:
                return int(m.group(1))
            items = [x for x in m.group(2).splitlines() if x.strip()]
            if len(items) != int(m.group(1)):
                raise OracleFailure("value_truncated")
            return sum(1 for x in items if x.endswith(f"({status})"))
        if lname == "teams":
            # count teams via the start-of-year office directory (teams never move)
            docs = [d for d in self.ws.docs if d.dtype == "directory"
                    and any(a[0] == anchor and a[1] == "head" for a in d.asserts)]
            target = docs[-1]
            hit = self._find(ep, "search_files", f"{self.query_name(anchor)} directory teams",
                             [target.path])
            if not hit:
                raise OracleFailure("file_not_retrieved")
            text = self._step(ep, call("read_file", path=hit))
            if "[... truncated]" in text:
                raise OracleFailure("value_truncated")
            return sum(1 for a in target.asserts if a[1] == "office")
        d, ids = self.lists[(anchor, lname)][-1]
        hit = self._find(ep, "search_files", f"{self.query_name(anchor)} {AGG_QUERY[(kind, lname)]}",
                         [d.path])
        if not hit:
            raise OracleFailure("file_not_retrieved")
        text = self._step(ep, call("read_file", path=hit))
        if "[... truncated]" in text:
            raise OracleFailure("value_truncated")
        if status:
            ids = [x for x in ids if d.date and self.w.value_at(x, "status", d.date) == status]
        return len(ids)

    # -- episode --------------------------------------------------------------
    def run(self, env: ToolEnv, q: dict, catalog: dict, cfg: EnvConfig):
        """Solve one question. Returns (EpisodeResult, failure reason or None)."""
        ep = Episode(env, q, catalog, cfg, self.version)
        reason = None
        try:
            if q["structure"] == "aggregate":
                value = self.aggregate(ep, q)
            else:
                cur = None
                for eid, attr in q["facts"]:
                    cur = self.fact(ep, eid if cur is None else cur, attr)
                value = cur
            text, _ = render_gold(self.w, q["answer_type"], value)
            ep.step(call("answer", text=text))
        except OracleFailure as e:
            reason = str(e)
            ep.force_end("oracle_failed")
        res = ep.result()
        if reason is None and not res.correct:
            reason = "wrong_value"
        return res, reason


def make_env(ws, version: str | None, counter, cfg: EnvConfig) -> ToolEnv:
    """ToolEnv for one workspace; version=None means the no-notes condition."""
    pages = ws.notes[version] if version else None
    return ToolEnv(ws.docs, pages, counter, max_output_tokens=cfg.max_output_tokens)


__all__ = ["Oracle", "OracleFailure", "make_env", "VERSIONS"]
