"""Parse a model turn in Qwen's native tool-call format.

Qwen2.5 emits:
    <tool_call>
    {"name": "search_notes", "arguments": {"query": "Project Heron"}}
    </tool_call>

Rules (each one is tested):
  - Only the FIRST <tool_call> block in a turn is executed. One call per turn
    keeps turn counting and the K limit unambiguous. Extra calls are reported
    in `n_calls` so the metric can count them.
  - A missing closing tag is tolerated (generation can stop on the turn's EOS).
  - ```json fences inside the block are stripped.
  - `arguments` may be a JSON object or a JSON-encoded string of one.
  - Anything else (no tag, bad JSON, no name, non-object arguments) is a
    ParseError. The env turns it into an error result, and the turn still counts.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

OPEN, CLOSE = "<tool_call>", "</tool_call>"
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")


@dataclass
class ToolCall:
    name: str
    arguments: dict
    n_calls: int = 1          # how many <tool_call> blocks the turn contained


@dataclass
class ParseError:
    kind: str                 # no_tool_call | bad_json | bad_structure
    message: str              # shown to the model as the tool result
    n_calls: int = 0


def parse_tool_call(text: str) -> ToolCall | ParseError:
    n_calls = text.count(OPEN)
    start = text.find(OPEN)
    if start == -1:
        return ParseError("no_tool_call", "Error: no tool call found. Call a tool using "
                          "<tool_call>{\"name\": ..., \"arguments\": {...}}</tool_call>, and "
                          "use the answer tool to give your final answer.")
    body = text[start + len(OPEN):]
    end = body.find(CLOSE)
    if end != -1:
        body = body[:end]
    nxt = body.find(OPEN)                     # unclosed first call followed by another
    if nxt != -1:
        body = body[:nxt]
    body = _FENCE_RE.sub("", body.strip()).strip()
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as e:
        return ParseError("bad_json", f"Error: the tool call is not valid JSON ({e.msg}).",
                          n_calls)
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str) or not obj["name"]:
        return ParseError("bad_structure", "Error: a tool call must be a JSON object with "
                          "a string 'name' and an 'arguments' object.", n_calls)
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            return ParseError("bad_structure", "Error: 'arguments' must be a JSON object.",
                              n_calls)
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return ParseError("bad_structure", "Error: 'arguments' must be a JSON object.", n_calls)
    return ToolCall(obj["name"].strip(), args, n_calls)
