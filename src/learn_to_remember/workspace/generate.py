"""Build a whole company workspace from a seed, and serialize it reproducibly.

    ws = generate_company(seed=0, company="company_a")

returns the world, raw documents, notes for every version and the question
splits. `write_workspace` writes it as JSONL with sorted keys, so the same seed
gives byte-identical files on any machine. `manifest` hashes every artifact, so
a committed eval set can be checked against a regenerated workspace.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from statistics import median

from .documents import Doc, build_documents
from .entities import World, build_world
from .notes import VERSIONS, build_notes
from .questions import QTYPES, make_splits

GENERATOR_VERSION = "m1-2026-11-15"


@dataclass
class Workspace:
    company: str
    seed: int
    world: World
    docs: list[Doc]
    notes: dict[str, list[dict]]
    splits: dict


def generate_company(seed: int, company: str, *, with_train: bool = True,
                     recipe: str = "structured") -> Workspace:
    world = build_world(seed)
    docs = build_documents(world)
    notes = {v: build_notes(world, docs, ver, recipe) for v, ver in VERSIONS.items()}
    splits = make_splits(world, notes, company, with_train=with_train)
    return Workspace(company, seed, world, docs, notes, splits)


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #
def _dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def _jsonl(rows) -> str:
    return "".join(_dumps(r) + "\n" for r in rows)


def entity_rows(world: World) -> list[dict]:
    rows = []
    for e in world.entities.values():
        rows.append({"id": e.id, "kind": e.kind, "name": e.name, "aliases": e.aliases,
                     "created": e.created.isoformat(),
                     "timeline": {a: [[d.isoformat(), v] for d, v in tl]
                                  for (eid, a), tl in world.timeline.items() if eid == e.id}})
    return rows


def event_rows(world: World) -> list[dict]:
    return [{"id": ev.id, "date": ev.date.isoformat(), "type": ev.type, "subject": ev.subject,
             "changes": [list(c) for c in ev.changes], "info": ev.info} for ev in world.events]


def serialize(ws: Workspace) -> dict[str, str]:
    """Every artifact as {relative filename: text}."""
    files = {
        "entities.jsonl": _jsonl(entity_rows(ws.world)),
        "events.jsonl": _jsonl(event_rows(ws.world)),
        "raw_docs.jsonl": _jsonl(d.to_json() for d in ws.docs),
        "questions_eval.jsonl": _jsonl(ws.splits["eval"]),
        "questions_train.jsonl": _jsonl(ws.splits["train"]),
        "heldout.json": _dumps(ws.splits["heldout"]) + "\n",
    }
    for v, pages in ws.notes.items():
        files[f"notes_{v}.jsonl"] = _jsonl(pages)
    return files


def manifest(ws: Workspace, files: dict[str, str] | None = None) -> dict:
    files = files or serialize(ws)
    return {"company": ws.company, "seed": ws.seed, "generator": GENERATOR_VERSION,
            "sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in sorted(files.items())}}


def write_workspace(ws: Workspace, out_dir: str, *, write_markdown: bool = True) -> dict:
    """Write JSONL artifacts (and, optionally, browsable .md trees) to out_dir."""
    files = serialize(ws)
    os.makedirs(out_dir, exist_ok=True)
    for name, text in files.items():
        with open(os.path.join(out_dir, name), "w") as f:
            f.write(text)
    if write_markdown:
        for d in ws.docs:
            _write(os.path.join(out_dir, "raw", d.path), d.text)
        for v, pages in ws.notes.items():
            for p in pages:
                _write(os.path.join(out_dir, f"notes_{v}", p["note_id"] + ".md"), p["text"])
    m = manifest(ws, files)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        f.write(json.dumps(m, indent=2, sort_keys=True) + "\n")
    return m


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def stats(ws: Workspace, count_tokens=None) -> dict:
    """Counts for docs/RESULTS.md. `count_tokens(text) -> int` is optional
    (the Qwen tokenizer); without it, whitespace word counts are reported."""
    count = count_tokens or (lambda t: len(t.split()))
    unit = "tokens" if count_tokens else "words"
    w = ws.world
    kinds = {}
    for e in w.entities.values():
        kinds[e.kind] = kinds.get(e.kind, 0) + 1
    doc_lens = [count(d.text) for d in ws.docs]
    dtypes = {}
    for d in ws.docs:
        dtypes[d.dtype] = dtypes.get(d.dtype, 0) + 1
    notes = {}
    n_notable = len([e for e in w.entities if e not in (w.ceo, w.leadership)])
    for v, pages in ws.notes.items():
        ent_pages = [p for p in pages if p["entity"]]
        lens = [count(p["text"]) for p in pages]
        notes[v] = {"pages": len(ent_pages), "index_pages": len(pages) - len(ent_pages),
                    "coverage_actual": round(len(ent_pages) / n_notable, 3),
                    "truncated_pages": sum(1 for p in ent_pages if p["omitted"]),
                    f"total_{unit}": sum(lens), f"max_page_{unit}": max(lens),
                    "as_of": VERSIONS[v].snapshot.isoformat()}
    q = {}
    for split in ("train", "eval"):
        rows = ws.splits[split]
        q[split] = {"n": len(rows), "heldout_questions": sum(r["heldout"] for r in rows),
                    "qtype_v3": {t: sum(r["qtype"] == t for r in rows) for t in QTYPES},
                    "qtype_by_version": {v: {t: sum(r["qtype_by_version"][v] == t for r in rows)
                                             for t in QTYPES} for v in VERSIONS},
                    "notes_ok_by_version": {v: sum(r["notes_ok_by_version"][v] for r in rows)
                                            for v in VERSIONS}}
    return {"company": ws.company, "seed": ws.seed, "unit": unit,
            "entities": kinds, "n_entities": len(w.entities), "n_events": len(w.events),
            "docs": {"n": len(ws.docs), "by_type": dtypes, f"total_{unit}": sum(doc_lens),
                     f"min_{unit}": min(doc_lens), f"median_{unit}": median(doc_lens),
                     f"max_{unit}": max(doc_lens)},
            "notes": notes, "questions": q, "n_candidates": ws.splits["n_candidates"],
            "n_heldout_entities": len(ws.splits["heldout"])}
