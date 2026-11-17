"""Multi-turn rollout that keeps exact token ids, plus the loss mask (spec §3.4).

A trajectory is ONE token sequence:

    [prompt | model turn 1 | tool result 1 | model turn 2 | ... | model turn n]

and `mask` is 1 exactly on the tokens the model sampled (each turn's text and
its stop token), 0 on everything it was given: the prompt, tool results, and
the template glue between them.

The re-tokenization pitfall: re-rendering the whole conversation with the chat
template and tokenizing it again does NOT reproduce what the model saw and
sampled, for two reasons:
  1. Token boundaries. The model may sample a non-canonical split of a string
     (e.g. "ab" as "a" + "b"). Re-tokenizing merges it, so the log-probs would
     be scored on a sequence that was never sampled.
  2. Templates rewrite history. Qwen3's template drops the empty
     `<think></think>` block from earlier assistant turns, though the model was
     prompted with it at sampling time.
So the trajectory is built by appending: the sampled ids as they came out of
`generate`, then each tool result tokenized as a separate segment. A segment
starts right after the end-of-turn special token, and special tokens are hard
token boundaries, so tokenizing it alone is exact. `tests/test_rollout.py`
checks both claims.

`ChatFormat` derives the framing strings from the tokenizer's own chat template
(no hard-coded Qwen strings), so the same code serves Qwen2.5 and Qwen3.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from ..env.episode import Episode, EpisodeResult
from ..env.prompt import tool_schemas

_A = "QQASSISTANTQQ"      # sentinels used to cut the template's framing strings
_T = "QQTOOLRESULTQQ"


class ChatFormat:
    """Exact-id chat framing for one tokenizer + chat template.

    prompt_ids(messages, tools)   ids of the rendered prompt + generation prompt
    tool_segment_ids(observation) ids appended after a model turn's end token:
                                  glue + tool turn + the next generation prompt
    eot_id                        the end-of-turn token (Qwen: <|im_end|>)
    stop_ids                      tokens that end a turn when sampled
    """

    def __init__(self, tok, *, template_kwargs: dict | None = None,
                 extra_stop_ids: tuple[int, ...] = ()):
        self.tok = tok
        self.kw = dict(template_kwargs or {})
        sys_user = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
        no_gen = self.render(sys_user, add_generation_prompt=False)
        with_gen = self.render(sys_user, add_generation_prompt=True)
        assert with_gen.startswith(no_gen), "generation prompt is not a suffix"
        self.gen_prompt = with_gen[len(no_gen):]

        after_asst = self.render(sys_user + [{"role": "assistant", "content": _A}],
                                 add_generation_prompt=False).split(_A, 1)[1]
        eot_ids = self._ids(after_asst)
        self.eot_id = eot_ids[0]
        self.eot_str = tok.decode([self.eot_id])
        assert self.eot_id in tok.all_special_ids, f"turn end {self.eot_str!r} not special"
        assert after_asst.startswith(self.eot_str)

        convo = sys_user + [{"role": "assistant", "content": _A},
                            {"role": "tool", "content": _T}]
        tail = self.render(convo, add_generation_prompt=False).split(_A, 1)[1]
        assert tail.startswith(self.eot_str)
        pre, post = tail[len(self.eot_str):].split(_T, 1)
        self.tool_pre, self.tool_post = pre, post       # Qwen: "\n<|im_start|>user\n<tool_response>\n", ...
        stops = {self.eot_id, *extra_stop_ids}
        if tok.eos_token_id is not None:
            stops.add(tok.eos_token_id)
        self.stop_ids = tuple(sorted(stops))

    # -- rendering -------------------------------------------------------------
    def render(self, messages, *, tools=None, add_generation_prompt: bool) -> str:
        return self.tok.apply_chat_template(messages, tools=tools, tokenize=False,
                                            add_generation_prompt=add_generation_prompt,
                                            **self.kw)

    def _ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def prompt_ids(self, messages, tools) -> list[int]:
        return self._ids(self.render(messages, tools=tools, add_generation_prompt=True))

    def tool_segment_text(self, observation: str) -> str:
        return self.tool_pre + observation + self.tool_post + self.gen_prompt

    def tool_segment_ids(self, observation: str) -> list[int]:
        return self._ids(self.tool_segment_text(observation))

    def turn_text(self, new_ids: list[int]) -> str:
        """The assistant text of one sampled turn (stop token removed)."""
        if new_ids and new_ids[-1] in self.stop_ids:
            new_ids = new_ids[:-1]
        return self.tok.decode(new_ids, skip_special_tokens=False)


def chat_format_for(model_id: str, tok) -> ChatFormat:
    """Qwen3 runs in non-thinking mode (spec §3.5): its template's
    `enable_thinking=False` puts an empty think block in each generation prompt."""
    kw = {"enable_thinking": False} if "qwen3" in model_id.lower() else {}
    # Qwen's generation_config also stops on <|endoftext|>; the tokenizer's eos is
    # <|im_end|>. Stop on both (a turn ending in <|endoftext|> gets an
    # unmasked <|im_end|> appended, see rollout()).
    extra = tuple(i for i in [tok.convert_tokens_to_ids("<|endoftext|>")]
                  if isinstance(i, int) and i != tok.unk_token_id)
    return ChatFormat(tok, template_kwargs=kw, extra_stop_ids=extra)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
class HFGenerator:
    """Batched HF `generate` over a list of id sequences -> the new ids of each.

    Every sampling parameter is set explicitly. transformers fills any parameter
    left unset from the model's own generation_config, and Qwen2.5-Instruct's
    ships repetition_penalty 1.1, top_k 20, top_p 0.8, temperature 0.7. Left
    alone, that would bend "greedy" eval and make sampled tokens differ from
    the policy whose log-probs GRPO computes (see DECISIONS.md, M3).
    """

    def __init__(self, model, tok, *, stop_ids, max_new_tokens: int = 256,
                 greedy: bool = False, temperature: float = 1.0, top_p: float = 1.0,
                 batch_size: int = 8, record_logprobs: bool = False):
        from transformers import GenerationConfig

        self.model, self.tok = model, tok
        self.stop_ids = list(stop_ids)
        self.batch_size = batch_size
        self.record_logprobs = record_logprobs
        self.temperature = 1.0 if greedy else temperature
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else self.stop_ids[0]
        params = dict(max_new_tokens=max_new_tokens, do_sample=not greedy,
                      repetition_penalty=1.0, top_k=0, top_p=1.0 if greedy else top_p,
                      temperature=1.0 if greedy else temperature, min_p=None,
                      eos_token_id=self.stop_ids, pad_token_id=self.pad_id)
        self.gen_kwargs = params
        # set it on the inner HF model too: a PeftModel forwards generate() to it
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        for m in {id(model): model, id(base): base}.values():
            m.generation_config = GenerationConfig(**params)
        # one complete config per call (no extra kwargs, which transformers 5 deprecates)
        self.call_config = GenerationConfig(**params, return_dict_in_generate=True,
                                            output_logits=record_logprobs)

    @torch.no_grad()
    def __call__(self, seqs: list[list[int]]):
        """New ids per sequence, trimmed at the first stop token. With
        `record_logprobs`, (ids, log-probs) pairs instead: the sampling-time
        log pi(token) under temperature-scaled raw logits (exact for top_p=1),
        for checking teacher-forced log-probs against what was sampled."""
        device = next(self.model.parameters()).device
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))   # less padding
        out: list[list[int] | None] = [None] * len(seqs)
        was_training = self.model.training
        self.model.eval()
        for s in range(0, len(order), self.batch_size):
            idx = order[s:s + self.batch_size]
            L = max(len(seqs[i]) for i in idx)
            ids = torch.full((len(idx), L), self.pad_id, dtype=torch.long)
            attn = torch.zeros((len(idx), L), dtype=torch.long)
            for r, i in enumerate(idx):                                 # left padding
                ids[r, L - len(seqs[i]):] = torch.tensor(seqs[i])
                attn[r, L - len(seqs[i]):] = 1
            res = self.model.generate(input_ids=ids.to(device), attention_mask=attn.to(device),
                                      generation_config=self.call_config)
            gen = res.sequences
            for r, i in enumerate(idx):
                new = trim_at_stop(gen[r, L:].tolist(), self.stop_ids)
                if not self.record_logprobs:
                    out[i] = new
                    continue
                lps = [torch.log_softmax(res.logits[j][r].float() / self.temperature, -1)[t].item()
                       for j, t in enumerate(new)]
                out[i] = (new, lps)
        if was_training:
            self.model.train()
        return out


def trim_at_stop(new_ids: list[int], stop_ids) -> list[int]:
    """Keep up to and including the first stop token (drops padding after it)."""
    stops = set(stop_ids)
    for j, t in enumerate(new_ids):
        if t in stops:
            return new_ids[:j + 1]
    return new_ids


# --------------------------------------------------------------------------- #
# Rollout
# --------------------------------------------------------------------------- #
@dataclass
class Trajectory:
    ids: list[int]
    mask: list[int]                 # 1 = sampled by the model
    prompt_len: int
    result: EpisodeResult
    gen_lens: list[int] = field(default_factory=list)       # sampled tokens per turn
    hit_gen_cap: list[bool] = field(default_factory=list)   # turn cut at max_new_tokens
    seconds: float = 0.0                                     # wall time of the whole batch
    sample_logprobs: list[float] | None = None   # per mask-1 token, if the generator recorded them

    def segments(self) -> list[tuple[int, int, int]]:
        """(start, end, mask value) runs, for inspection and tests."""
        runs, start = [], 0
        for j in range(1, len(self.mask) + 1):
            if j == len(self.mask) or self.mask[j] != self.mask[start]:
                runs.append((start, j, self.mask[start]))
                start = j
        return runs


@dataclass
class _State:
    ep: Episode
    ids: list[int]
    mask: list[int]
    prompt_len: int
    gen_lens: list[int] = field(default_factory=list)
    hit_cap: list[bool] = field(default_factory=list)
    lps: list[float] | None = None


def rollout(generate, fmt: ChatFormat, episodes: list[Episode]) -> list[Trajectory]:
    """Run episodes to completion, batching every active episode per turn.

    `generate(list of id lists) -> list of new-id lists`, each trimmed at its
    first stop token (HFGenerator, or a scripted fake in tests).

    Per turn:
      - Context check first (same rule as run_text_episode): if the next turn's
        full generation budget does not fit in max_context_tokens, the episode
        ends with "context_limit". Every turn that runs can use all of it.
      - The sampled ids are appended as-is, mask 1. If the turn ended without
        the end-of-turn token (cut at max_new_tokens, or stopped on a different
        EOS), an end-of-turn token is appended with mask 0 so the next segment
        is framed correctly. The model did not sample it, so it isn't trained.
      - The env steps on the decoded text. If the episode continues, the tool
        segment is appended with mask 0.
    """
    t0 = time.perf_counter()
    states = []
    for ep in episodes:
        p = fmt.prompt_ids(ep.initial_messages(), tool_schemas(ep.env.has_notes))
        states.append(_State(ep, list(p), [0] * len(p), len(p)))
    while True:
        for s in states:
            cfg = s.ep.cfg
            if not s.ep.done and len(s.ids) + cfg.max_gen_tokens > cfg.max_context_tokens:
                s.ep.force_end("context_limit")
        active = [s for s in states if not s.ep.done]
        if not active:
            break
        outs = generate([s.ids for s in active])
        for s, new in zip(active, outs):
            if isinstance(new, tuple):                  # (ids, sampling log-probs)
                new, lp = new
                s.lps = (s.lps or []) + list(lp)
            new = list(new)
            s.ids += new
            s.mask += [1] * len(new)
            s.gen_lens.append(len(new))
            s.hit_cap.append(not new or new[-1] not in fmt.stop_ids)
            if not new or new[-1] != fmt.eot_id:
                s.ids.append(fmt.eot_id)
                s.mask.append(0)
            obs, done = s.ep.step(fmt.turn_text(new))
            if not done:
                seg = fmt.tool_segment_ids(obs)
                s.ids += seg
                s.mask += [0] * len(seg)
    dt = time.perf_counter() - t0
    return [Trajectory(s.ids, s.mask, s.prompt_len, s.ep.result(), s.gen_lens, s.hit_cap, dt,
                       s.lps) for s in states]


def collate(trajs: list[Trajectory], pad_id: int, device=None) -> dict:
    """Right-padded [B, L] tensors: input_ids, attention_mask, loss_mask."""
    L = max(len(t.ids) for t in trajs)
    ids = torch.full((len(trajs), L), pad_id, dtype=torch.long)
    attn = torch.zeros((len(trajs), L), dtype=torch.long)
    mask = torch.zeros((len(trajs), L), dtype=torch.float32)
    for r, t in enumerate(trajs):
        n = len(t.ids)
        ids[r, :n] = torch.tensor(t.ids)
        attn[r, :n] = 1
        mask[r, :n] = torch.tensor(t.mask, dtype=torch.float32)
    out = {"input_ids": ids, "attention_mask": attn, "loss_mask": mask}
    return {k: v.to(device) for k, v in out.items()} if device is not None else out
