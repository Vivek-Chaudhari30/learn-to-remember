"""The fixed system prompt and tool schemas (spec §3.2).

These are fixed for the whole project and logged verbatim in docs/DECISIONS.md.
Changing them invalidates every comparison, so a test pins their exact text.
Two variants differ only in what the notes condition adds: the "notes"
sentences and the two notes tools.
"""

from __future__ import annotations

TODAY = "2025-12-31"

_BASE = (
    "You answer questions about a company using its internal workspace. Today is {today}.\n"
    "{memory}"
    "Use the tools to look things up. Each turn, make exactly one tool call. You have at "
    "most {k} tool calls in total, and the final one must be answer. Reading costs: every "
    "token a tool returns counts against you, so prefer narrow searches and read only what "
    "you need.\n"
    "Give the final answer with the answer tool, as just the answer: a name, a number, a "
    "date or a short phrase, with no explanation."
)
_MEMORY_NOTES = (
    "There are two sources. The raw files are the source of truth. The notes are a curated "
    "summary of them; they are faster to use but may be incomplete or out of date. Each "
    "note has an as_of date and lists its source files. When a note is missing, lacks the "
    "detail you need, or may have changed after its as_of date, check the raw files.\n"
)
_MEMORY_FILES = "The raw files are the source of truth.\n"

TOOL_SCHEMAS = {
    "search_notes": {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "Keyword search over the notes. Returns note ids, titles, as_of "
                           "dates and the best-matching line.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "Search keywords."},
                "k": {"type": "integer", "description": "Number of results (1-10, default 5)."}},
                "required": ["query"]}}},
    "read_note": {
        "type": "function",
        "function": {
            "name": "read_note",
            "description": "Read one note page by its note id.",
            "parameters": {"type": "object", "properties": {
                "note_id": {"type": "string", "description": "A note id from search_notes."}},
                "required": ["note_id"]}}},
    "search_files": {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Keyword search over the raw files. Returns file paths, dates, "
                           "titles and the best-matching line.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "Search keywords."},
                "k": {"type": "integer", "description": "Number of results (1-10, default 5)."},
                "since": {"type": "string", "description": "Optional date YYYY-MM-DD: only "
                          "return files dated on or after it."}},
                "required": ["query"]}}},
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read one raw file by its path.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "A file path from search_files."}},
                "required": ["path"]}}},
    "answer": {
        "type": "function",
        "function": {
            "name": "answer",
            "description": "Give the final answer. This ends the episode.",
            "parameters": {"type": "object", "properties": {
                "text": {"type": "string", "description": "Just the answer."}},
                "required": ["text"]}}},
}


def system_prompt(notes: bool, max_turns: int = 6) -> str:
    return _BASE.format(today=TODAY, k=max_turns,
                        memory=_MEMORY_NOTES if notes else _MEMORY_FILES)


def tool_schemas(notes: bool) -> list[dict]:
    names = (["search_notes", "read_note"] if notes else []) + ["search_files", "read_file",
                                                                 "answer"]
    return [TOOL_SCHEMAS[n] for n in names]


def initial_messages(question: str, notes: bool, max_turns: int = 6) -> list[dict]:
    return [{"role": "system", "content": system_prompt(notes, max_turns)},
            {"role": "user", "content": question}]
