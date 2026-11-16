"""The five tools: the harness, fixed and identical across all conditions.

    search_notes(query, k=5)  BM25 over note pages  -> id, title, as_of, best line
    read_note(note_id)        one note page, cut at the token cap
    search_files(query, k=5, since=None)
                              BM25 over raw docs    -> path, date, title, best line;
                              `since` (YYYY-MM-DD) keeps only files dated on/after it
    read_file(path)           one raw document, cut at the token cap
    answer(text)              ends the episode

The "no notes" condition builds the env with notes=None. The two notes tools
then don't exist: calling one is an unknown-tool error, and they are left out
of the schemas and the system prompt.

`since` on search_files is a deviation from the spec's tool table (see
DECISIONS.md, M2). BM25 has no notion of recency, and without it the newest
update to a fact was often outranked by older documents about the same entity.

Every tool output, errors included, goes through the same token cap and is
counted in `tokens`. That count is what the reward's cost term charges for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .bm25 import BM25, tokenize

NOTES_TOOLS = ("search_notes", "read_note")
FILE_TOOLS = ("search_files", "read_file")
SNIPPET_WORDS = 30
_SKIP_SNIPPET = ("note_id:", "type:", "as_of:", "date:", "sources:", "- files/")


@dataclass
class ToolResult:
    tool: str
    args: dict
    text: str
    ok: bool                       # False for any error result
    tokens: int = 0                # tokens of `text` returned to the model
    truncated: bool = False
    error: str | None = None       # short machine-readable error kind
    meta: dict = field(default_factory=dict)   # e.g. ids returned or read (for metrics)


def _get(x, key):
    return x[key] if isinstance(x, dict) else getattr(x, key)


def best_line(text: str, query: str) -> str:
    """The body line sharing the most query terms (ties: earliest), shortened."""
    q = set(tokenize(query))
    best, best_score = "", -1
    for line in text.splitlines()[1:]:          # skip the title line
        s = line.strip()
        if not s or s.lower().startswith(_SKIP_SNIPPET):
            continue
        score = len(q & set(tokenize(s)))
        if score > best_score:
            best, best_score = s, score
    words = best.split()
    return " ".join(words[:SNIPPET_WORDS]) + (" ..." if len(words) > SNIPPET_WORDS else "")


class ToolEnv:
    """Search/read over one workspace with one notes version (or none)."""

    def __init__(self, docs, notes_pages, counter, *, max_output_tokens: int = 400,
                 default_k: int = 5, max_k: int = 10):
        self.counter = counter
        self.cap = max_output_tokens
        self.default_k, self.max_k = default_k, max_k
        self.docs = [{"path": _get(d, "path"), "date": str(_get(d, "date")),
                      "title": _get(d, "title"), "text": _get(d, "text")} for d in docs]
        self.doc_by_path = {d["path"]: d for d in self.docs}
        self.doc_index = BM25([d["text"] for d in self.docs])
        self.has_notes = notes_pages is not None
        if self.has_notes:
            self.notes = list(notes_pages)
            self.note_by_id = {p["note_id"]: p for p in self.notes}
            # title field boost: without it, an office page loses to the dozens of
            # people/team pages that merely contain "office: <city>"
            self.note_index = BM25([f"{p['title']}\n{p['title']}\n{p['text']}"
                                    for p in self.notes])

    @property
    def tool_names(self) -> list[str]:
        return ([*NOTES_TOOLS] if self.has_notes else []) + [*FILE_TOOLS, "answer"]

    # -- dispatch -------------------------------------------------------------
    def call(self, name: str, args: dict) -> ToolResult:
        if name not in self.tool_names:
            return self._out(name, args, f"Error: unknown tool '{name}'. Available tools: "
                             + ", ".join(self.tool_names) + ".", error="unknown_tool")
        if name == "answer":
            text = args.get("text")
            if not isinstance(text, str) or not text.strip():
                return self._out(name, args, "Error: answer needs a non-empty 'text' "
                                 "argument.", error="bad_args")
            return ToolResult(name, args, "", ok=True, meta={"answer": text.strip()})
        if name in ("search_notes", "search_files"):
            q = args.get("query")
            if not isinstance(q, str) or not q.strip():
                return self._out(name, args, "Error: 'query' must be a non-empty string.",
                                 error="bad_args")
            k = args.get("k", self.default_k)
            try:
                k = int(k)
            except (TypeError, ValueError):
                return self._out(name, args, "Error: 'k' must be an integer.", error="bad_args")
            k = max(1, min(self.max_k, k))
            if name == "search_notes":
                return self._search_notes(q, k, args)
            since = args.get("since")
            if since is not None:
                from datetime import date
                try:
                    since = date.fromisoformat(str(since).strip()).isoformat()
                except ValueError:
                    return self._out(name, args, "Error: 'since' must be a date like "
                                     "2025-09-30.", error="bad_args")
            return self._search_files(q, k, args, since)
        key = "note_id" if name == "read_note" else "path"
        target = args.get(key)
        if not isinstance(target, str) or not target.strip():
            return self._out(name, args, f"Error: '{key}' must be a non-empty string.",
                             error="bad_args")
        return self._read_note(target, args) if name == "read_note" else self._read_file(
            target, args)

    def _out(self, tool, args, text, *, error=None, meta=None) -> ToolResult:
        out, n, trunc = self.counter.truncate(text, self.cap)
        return ToolResult(tool, args, out, ok=error is None, tokens=n, truncated=trunc,
                          error=error, meta=meta or {})

    # -- tools ----------------------------------------------------------------
    def _search_notes(self, q, k, args):
        hits = self.note_index.search(q, k)
        if not hits:
            return self._out("search_notes", args, f"No notes match '{q}'.",
                             meta={"ids": []})
        lines = [f"Top {len(hits)} notes for '{q}':"]
        ids = []
        for rank, (i, _) in enumerate(hits, 1):
            p = self.notes[i]
            ids.append(p["note_id"])
            lines.append(f"{rank}. {p['note_id']} | {p['title']} | as_of {p['as_of']} | "
                         f"{best_line(p['text'], q)}")
        return self._out("search_notes", args, "\n".join(lines), meta={"ids": ids})

    def _search_files(self, q, k, args, since=None):
        allowed = None if since is None else (lambda i: self.docs[i]["date"] >= since)
        hits = self.doc_index.search(q, k, allowed)
        where = f"'{q}'" + (f" since {since}" if since else "")
        if not hits:
            return self._out("search_files", args, f"No files match {where}.",
                             meta={"ids": []})
        lines = [f"Top {len(hits)} files for {where}:"]
        ids = []
        for rank, (i, _) in enumerate(hits, 1):
            d = self.docs[i]
            ids.append(d["path"])
            lines.append(f"{rank}. {d['path']} | {d['date']} | {d['title']} | "
                         f"{best_line(d['text'], q)}")
        return self._out("search_files", args, "\n".join(lines), meta={"ids": ids})

    @staticmethod
    def _clean_id(s: str) -> str:
        s = s.strip().strip("`'\"").lstrip("/")
        return s[:-3] if s.endswith(".md") and s.startswith("notes/") else s

    def _read_note(self, nid, args):
        key = self._clean_id(nid)
        page = self.note_by_id.get(key)
        if page is None:
            return self._out("read_note", args, f"Error: no note with id '{nid}'. Use "
                             "search_notes to find note ids.", error="not_found")
        return self._out("read_note", args, page["text"], meta={"ids": [key]})

    def _read_file(self, path, args):
        key = self._clean_id(path)
        if not key.startswith("files/") and f"files/{key}" in self.doc_by_path:
            key = f"files/{key}"
        doc = self.doc_by_path.get(key)
        if doc is None:
            return self._out("read_file", args, f"Error: no file at '{path}'. Use "
                             "search_files to find file paths.", error="not_found")
        return self._out("read_file", args, doc["text"], meta={"ids": [key]})
