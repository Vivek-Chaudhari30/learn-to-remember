"""The fixed prompt is pinned, and the oracle proves the eval sets are solvable.

If a prompt hash test fails, the system prompt or tool schemas changed. That
invalidates comparisons with earlier runs: update docs/DECISIONS.md (where the
exact text is logged) and these hashes together, on purpose.
"""

import hashlib
import json

import pytest

from learn_to_remember.env.episode import EnvConfig
from learn_to_remember.env.prompt import TOOL_SCHEMAS, system_prompt, tool_schemas
from learn_to_remember.oracle import Oracle, make_env
from learn_to_remember.reward import build_catalog
from learn_to_remember.workspace.generate import generate_company


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def test_system_prompts_are_pinned():
    assert _sha(system_prompt(True)) == (
        "b26e52d49dd806766824d6d4595eec68d4000aade357e4ff46e524c9b012db21")
    assert _sha(system_prompt(False)) == (
        "d5d6ff2f527df1146850de2e0d72fb093f8fb5563aa3dd6937876fc445cee355")
    assert _sha(json.dumps(TOOL_SCHEMAS, sort_keys=True)) == (
        "20f48205faf81cd5b2216764e021773a7f70abec6bb9dc9a57c3a83fccd857d0")


def test_no_notes_prompt_and_tools_omit_notes():
    assert "note" not in system_prompt(False).lower()
    assert [t["function"]["name"] for t in tool_schemas(False)] == [
        "search_files", "read_file", "answer"]
    assert [t["function"]["name"] for t in tool_schemas(True)][:2] == [
        "search_notes", "read_note"]


@pytest.fixture(scope="module")
def counter():
    try:
        from learn_to_remember.env.tokens import load_qwen_counter
        return load_qwen_counter(local_only=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Qwen tokenizer not cached locally ({e}); run scripts/run_oracle.py")


@pytest.mark.parametrize("seed,company", [(0, "company_a"), (1, "company_b")])
@pytest.mark.parametrize("version", ["v0", "v3", None])
def test_oracle_solves_eval(seed, company, version, counter):
    """Spec §3.1: a scripted oracle reaches ~100%. Threshold 98%. The measured
    residual failures are logged in docs/RESULTS.md."""
    ws = generate_company(seed, company, with_train=(seed == 0))
    cfg = EnvConfig()
    env = make_env(ws, version, counter, cfg)
    oracle = Oracle(ws, version or "v3", notes=version is not None)
    cat = build_catalog(ws.world)
    results = [oracle.run(env, q, cat, cfg) for q in ws.splits["eval"]]
    acc = sum(r.correct for r, _ in results) / len(results)
    reasons = {why for r, why in results if not r.correct}
    assert acc >= 0.98, (acc, reasons)
    assert reasons <= {"out_of_turns", "file_not_retrieved"}, reasons
    for r, _ in results:            # every tool output respected the cap
        assert all(t["tokens"] <= cfg.max_output_tokens for t in r.turns)
