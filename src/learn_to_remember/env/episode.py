"""One episode: a question, a sequence of model turns, a reward.

The env is step-based, `Episode.step(assistant_text) -> (observation, done)`, so
the same logic serves text policies (the oracle, API models, the eval harness)
and the token-level multi-turn rollout in M3, which keeps exact generated ids
and only needs the observation text.

Turn rules (logged in docs/DECISIONS.md):
  - K = max_turns model turns in total, and `answer` counts as one. Each turn
    executes at most one tool call.
  - A malformed call or a tool error returns an error observation and uses
    the turn.
  - If turn K is not a valid answer, the episode ends ("turn_limit"). A
    non-answer call on turn K is NOT executed and costs nothing: its output
    would never reach the model.
  - `tokens_read` = tokens of every observation returned (errors included).
  - The runner may end an episode early with `force_end("context_limit")`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..reward import compute_reward, is_correct
from .parsing import ParseError, parse_tool_call
from .prompt import initial_messages
from .tools import FILE_TOOLS, NOTES_TOOLS, ToolEnv


@dataclass
class EnvConfig:
    max_turns: int = 6
    max_output_tokens: int = 400
    max_gen_tokens: int = 256
    max_context_tokens: int = 4096
    cost_lambda: float = 0.2
    cost_budget: int = 4000

    @classmethod
    def from_dict(cls, env: dict, reward: dict) -> "EnvConfig":
        return cls(max_turns=env["max_tool_turns"], max_output_tokens=env["max_tool_output_tokens"],
                   max_gen_tokens=env["max_gen_tokens_per_turn"],
                   max_context_tokens=env["max_context_tokens"],
                   cost_lambda=reward["cost_lambda"], cost_budget=reward["cost_budget_tokens"])


@dataclass
class Turn:
    assistant: str
    tool: str | None = None
    args: dict | None = None
    parse_error: str | None = None
    tool_error: str | None = None
    observation: str = ""
    tokens: int = 0
    truncated: bool = False
    n_calls: int = 0
    executed: bool = True
    ids: list = field(default_factory=list)     # note ids / paths returned or read


@dataclass
class EpisodeResult:
    question_id: str
    qtype: str
    turns: list
    answer: str | None
    correct: bool
    valid_answer: bool
    tokens_read: int
    reward: float
    ended: str                                  # answer | turn_limit | context_limit

    def to_json(self) -> dict:
        return asdict(self)


class Episode:
    def __init__(self, env: ToolEnv, question: dict, catalog: dict, cfg: EnvConfig,
                 version: str = "v3"):
        self.env, self.q, self.catalog, self.cfg = env, question, catalog, cfg
        self.version = version
        self.turns: list[Turn] = []
        self.done = False
        self.answer: str | None = None
        self.ended: str | None = None
        self.tokens_read = 0

    def initial_messages(self) -> list[dict]:
        return initial_messages(self.q["question"], self.env.has_notes, self.cfg.max_turns)

    def step(self, assistant_text: str) -> tuple[str | None, bool]:
        """Apply one model turn. Returns (observation for the model or None, done)."""
        if self.done:
            raise RuntimeError("episode is over")
        last = len(self.turns) + 1 >= self.cfg.max_turns
        turn = Turn(assistant=assistant_text)
        self.turns.append(turn)
        call = parse_tool_call(assistant_text)
        if isinstance(call, ParseError):
            turn.parse_error, turn.n_calls = call.kind, call.n_calls
            res = self.env._out("parse_error", {}, call.message, error=call.kind)
        else:
            turn.tool, turn.args, turn.n_calls = call.name, call.arguments, call.n_calls
            if last and call.name != "answer":
                turn.executed = False
                return self._end("turn_limit")
            res = self.env.call(call.name, call.arguments)
            turn.tool_error = res.error
            if call.name == "answer" and res.ok:
                self.answer = res.meta["answer"]
                return self._end("answer")
        if last:                     # an error on turn K is never shown, so costs nothing
            return self._end("turn_limit")
        turn.observation, turn.tokens, turn.truncated = res.text, res.tokens, res.truncated
        turn.ids = res.meta.get("ids", [])
        self.tokens_read += res.tokens
        return res.text, False

    def force_end(self, reason: str) -> None:
        if not self.done:
            self._end(reason)

    def _end(self, reason: str):
        self.done, self.ended = True, reason
        return None, True

    def result(self) -> EpisodeResult:
        assert self.done, "episode not finished"
        valid = self.answer is not None
        correct = valid and is_correct(self.answer, self.q, self.catalog)
        reward = compute_reward(correct=correct, valid_answer=valid,
                                tokens_read=self.tokens_read,
                                cost_lambda=self.cfg.cost_lambda, budget=self.cfg.cost_budget)
        return EpisodeResult(self.q["id"], self.q["qtype_by_version"][self.version],
                             [asdict(t) for t in self.turns], self.answer, correct, valid,
                             self.tokens_read, reward, self.ended)


def run_text_episode(policy, env: ToolEnv, question: dict, catalog: dict, cfg: EnvConfig,
                     counter, version: str = "v3") -> EpisodeResult:
    """Drive an episode with a text policy: `policy(messages) -> assistant text`.

    Context is tracked approximately here (message contents only, no template
    overhead). The M3 rollout tracks exact token ids instead.
    """
    ep = Episode(env, question, catalog, cfg, version)
    messages = ep.initial_messages()
    used = sum(counter.count(m["content"]) for m in messages)
    while not ep.done:
        text = policy(messages)
        used += counter.count(text)
        obs, done = ep.step(text)
        messages.append({"role": "assistant", "content": text})
        if not done:
            messages.append({"role": "tool", "content": obs})
            used += counter.count(obs)
            if used + cfg.max_gen_tokens > cfg.max_context_tokens:
                ep.force_end("context_limit")
    return ep.result()


def behavior(result: EpisodeResult) -> dict:
    """Per-episode behavior metrics (spec §4); aggregated by the eval harness."""
    turns = result.turns
    executed = [t for t in turns if t["executed"] and t["tool"] != "answer"]
    lookups = [t for t in executed if t["tool"] in NOTES_TOOLS + FILE_TOOLS
               and not t["tool_error"] and not t["parse_error"]]
    calls = [(t["tool"], repr(sorted((t["args"] or {}).items()))) for t in turns if t["tool"]]
    return {
        "n_turns": len(turns),
        "n_lookups": len(lookups),
        "tokens_read": result.tokens_read,
        "tokens_per_lookup": [t["tokens"] for t in lookups],
        "used_files": any(t["tool"] in FILE_TOOLS for t in lookups),
        "used_notes": any(t["tool"] in NOTES_TOOLS for t in lookups),
        "valid_calls": sum(1 for t in turns if not t["parse_error"] and t["tool_error"]
                           not in ("unknown_tool", "bad_args")),
        "n_calls_total": len(turns),
        "repeated_calls": len(calls) - len(set(calls)),
        "zero_lookup_answer": result.valid_answer and not lookups,
    }
