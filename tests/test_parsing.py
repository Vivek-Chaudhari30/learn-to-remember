"""Edge cases of the Qwen <tool_call> parser. Malformed calls must become
error results, never crashes and never silently-accepted wrong calls."""

import pytest

from learn_to_remember.env.parsing import ParseError, ToolCall, parse_tool_call


def ok(text) -> ToolCall:
    r = parse_tool_call(text)
    assert isinstance(r, ToolCall), r
    return r


def err(text) -> ParseError:
    r = parse_tool_call(text)
    assert isinstance(r, ParseError), r
    return r


def test_standard_qwen_format():
    r = ok('<tool_call>\n{"name": "search_notes", "arguments": {"query": "Heron"}}\n</tool_call>')
    assert (r.name, r.arguments, r.n_calls) == ("search_notes", {"query": "Heron"}, 1)


def test_text_around_the_call_is_ignored():
    r = ok('Let me check.\n<tool_call>{"name": "read_file", "arguments": {"path": "files/x.md"}}'
           '</tool_call>\nthanks')
    assert r.name == "read_file" and r.arguments == {"path": "files/x.md"}


def test_missing_close_tag_is_tolerated():
    r = ok('<tool_call>\n{"name": "answer", "arguments": {"text": "42"}}')
    assert r.arguments == {"text": "42"}


def test_code_fence_inside_block():
    r = ok('<tool_call>\n```json\n{"name": "answer", "arguments": {"text": "x"}}\n```\n</tool_call>')
    assert r.name == "answer"


def test_arguments_as_json_string():
    r = ok('<tool_call>{"name": "search_files", "arguments": "{\\"query\\": \\"q\\"}"}</tool_call>')
    assert r.arguments == {"query": "q"}


def test_parameters_key_and_missing_arguments():
    assert ok('<tool_call>{"name": "answer", "parameters": {"text": "a"}}</tool_call>'
              ).arguments == {"text": "a"}
    assert ok('<tool_call>{"name": "answer"}</tool_call>').arguments == {}


def test_only_first_of_multiple_calls_runs():
    r = ok('<tool_call>{"name": "search_notes", "arguments": {"query": "a"}}</tool_call>'
           '<tool_call>{"name": "answer", "arguments": {"text": "b"}}</tool_call>')
    assert r.name == "search_notes" and r.n_calls == 2


def test_unclosed_first_call_followed_by_second():
    r = ok('<tool_call>{"name": "search_notes", "arguments": {"query": "a"}}\n'
           '<tool_call>{"name": "answer", "arguments": {"text": "b"}}</tool_call>')
    assert r.name == "search_notes" and r.n_calls == 2


@pytest.mark.parametrize("text,kind", [
    ("The answer is 42.", "no_tool_call"),
    ("", "no_tool_call"),
    ("<tool_call>{name: answer}</tool_call>", "bad_json"),
    ("<tool_call>{\"name\": \"answer\", \"arguments\": {\"text\": \"x\"}", "bad_json"),
    ("<tool_call>[1, 2]</tool_call>", "bad_structure"),
    ("<tool_call>{\"arguments\": {}}</tool_call>", "bad_structure"),
    ("<tool_call>{\"name\": \"\"}</tool_call>", "bad_structure"),
    ("<tool_call>{\"name\": \"answer\", \"arguments\": [\"x\"]}</tool_call>", "bad_structure"),
    ("<tool_call>{\"name\": \"answer\", \"arguments\": \"not json\"}</tool_call>", "bad_structure"),
])
def test_malformed_calls_are_errors(text, kind):
    e = err(text)
    assert e.kind == kind and e.message.startswith("Error:")
