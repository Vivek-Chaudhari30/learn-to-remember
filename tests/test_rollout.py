"""Multi-turn rollout: exact token ids and the loss mask (spec §3.4, M3).

Proves, with the real Qwen tokenizers (cached locally; skipped if absent):
  1. The mask is 1 exactly on sampled ids, 0 on the prompt, every tool segment
     and every bit of template glue.
  2. For Qwen2.5 with canonically tokenized turns, the exact-id trajectory is
     identical to rendering the whole conversation with the chat template and
     tokenizing it, i.e. building it by segments loses nothing.
  3. Where the two DIFFER (a non-canonical sampled split; Qwen3's template
     rewriting history), the trajectory keeps exactly what was sampled and
     what the model was prompted with.
  4. With a real (tiny, random) model, teacher-forced log-probs of the
     trajectory equal the sampling-time log-probs, token for token.
"""

import json

import pytest
import torch

from learn_to_remember.env.episode import Episode, EnvConfig, run_text_episode
from learn_to_remember.env.prompt import tool_schemas
from learn_to_remember.env.tokens import HFTokenCounter
from learn_to_remember.env.tools import ToolEnv
from learn_to_remember.grpo.core import sequence_logprobs
from learn_to_remember.grpo.rollout import (
    HFGenerator, chat_format_for, collate, rollout, trim_at_stop,
)

DOCS = [{"path": "files/a.md", "date": "2025-01-01", "title": "Status report: Project Heron",
         "text": "# Status report: Project Heron\n\nOwner: Maria Okafor\nBudget: $1,000,000\n"},
        {"path": "files/b.md", "date": "2025-03-01", "title": "Status report: Project Kite",
         "text": "# Status report: Project Kite\n\nOwner: Ana Silva\nBudget: $2,000,000\n"}]
NOTES = [{"note_id": "notes/projects/heron", "title": "Project Heron", "as_of": "2025-09-30",
          "text": "# Project Heron\nowner: Maria Okafor\n"}]
CATALOG = {"person": ["Maria Okafor", "Ana Silva"]}
Q1 = {"id": "q1", "question": "Who owns Project Heron?", "answer_type": "person",
      "gold": "Maria Okafor", "qtype_by_version": {"v3": "lookup"}}
Q2 = {"id": "q2", "question": "Who owns Project Kite?", "answer_type": "person",
      "gold": "Ana Silva", "qtype_by_version": {"v3": "gap"}}


def call(name, **args):
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"


def _tok(model_id):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except Exception:
        pytest.skip(f"{model_id} tokenizer not cached")


@pytest.fixture(scope="module")
def qwen25():
    return _tok("Qwen/Qwen2.5-0.5B-Instruct")


@pytest.fixture(scope="module")
def qwen3():
    return _tok("Qwen/Qwen3-1.7B")


class ScriptedGen:
    """Fake generator: per question, a list of turns. A turn is a text (encoded
    canonically + end-of-turn), or a list of raw ids (returned as-is)."""

    def __init__(self, tok, fmt, scripts: dict):
        self.tok, self.fmt = tok, fmt
        self.its = {q: iter(turns) for q, turns in scripts.items()}
        self.sampled = {q: [] for q in scripts}
        self.prompts_seen = {q: [] for q in scripts}
        self.calls = 0

    def __call__(self, seqs):
        self.calls += 1
        outs = []
        for seq in seqs:
            text = self.tok.decode(seq)
            q = next(q for q in self.its if q in text)
            turn = next(self.its[q])
            ids = list(turn) if isinstance(turn, list) else (
                self.tok(turn, add_special_tokens=False)["input_ids"] + [self.fmt.eot_id])
            self.sampled[q].append(ids)
            self.prompts_seen[q].append(list(seq))
            outs.append(ids)
        return outs


def _episodes(tok, questions, notes=True, **cfg):
    env = ToolEnv(DOCS, NOTES if notes else None, HFTokenCounter(tok), max_output_tokens=400)
    cfg = EnvConfig(**cfg)
    return [Episode(env, q, CATALOG, cfg) for q in questions], env, cfg


SCRIPT1 = [call("search_notes", query="Heron owner"),
           call("read_note", note_id="notes/projects/heron"),
           call("answer", text="Maria Okafor")]
SCRIPT2 = [call("search_files", query="Kite owner"), "I am not sure.",
           call("read_file", path="files/b.md"), call("answer", text="Ana Silva")]


def _messages(ep_question, notes, turns_text, observations):
    from learn_to_remember.env.prompt import initial_messages
    msgs = initial_messages(ep_question, notes)
    for i, t in enumerate(turns_text):
        msgs.append({"role": "assistant", "content": t})
        if i < len(observations):
            msgs.append({"role": "tool", "content": observations[i]})
    return msgs


# --------------------------------------------------------------------------- #
# 1. the mask
# --------------------------------------------------------------------------- #
def test_mask_is_one_exactly_on_sampled_ids(qwen25):
    fmt = chat_format_for("Qwen/Qwen2.5-0.5B-Instruct", qwen25)
    eps, env, _ = _episodes(qwen25, [Q1, Q2])
    gen = ScriptedGen(qwen25, fmt, {Q1["question"]: SCRIPT1, Q2["question"]: SCRIPT2})
    t1, t2 = rollout(gen, fmt, eps)
    for traj, key, n_turns in ((t1, Q1["question"], 3), (t2, Q2["question"], 4)):
        ids, mask = traj.ids, traj.mask
        assert len(ids) == len(mask)
        # prompt: identical to the template's prompt, all 0
        p = fmt.prompt_ids(eps[0 if key == Q1["question"] else 1].initial_messages(),
                           tool_schemas(True))
        assert ids[:traj.prompt_len] == p and not any(mask[:traj.prompt_len])
        # mask-1 tokens, in order, are exactly the sampled ids
        assert [i for i, m in zip(ids, mask) if m] == sum(gen.sampled[key], [])
        # runs alternate: prompt(0) gen(1) tool(0) gen(1) ... gen(1)
        runs = traj.segments()
        assert [m for _, _, m in runs] == [0] + [1, 0] * (n_turns - 1) + [1]
        # every 0-run after the prompt is exactly one tool segment
        obs = [t["observation"] for t in traj.result.to_json()["turns"][:-1]]
        zero_runs = [ids[a:b] for a, b, m in runs[1:] if m == 0]
        assert zero_runs == [fmt.tool_segment_ids(o) for o in obs]
        assert traj.gen_lens == [len(x) for x in gen.sampled[key]]
        assert not any(traj.hit_gen_cap)
    assert t1.result.correct and t2.result.correct and t2.result.ended == "answer"
    assert gen.calls == 4                      # batched: one call per turn while any is active


def test_each_turn_prompt_is_the_trajectory_prefix(qwen25):
    """What the generator saw at turn k is exactly the trajectory up to turn k."""
    fmt = chat_format_for("Qwen/Qwen2.5-0.5B-Instruct", qwen25)
    eps, _, _ = _episodes(qwen25, [Q2])
    gen = ScriptedGen(qwen25, fmt, {Q2["question"]: SCRIPT2})
    (traj,) = rollout(gen, fmt, eps)
    starts = [a for a, _, m in traj.segments() if m == 1]
    assert gen.prompts_seen[Q2["question"]] == [traj.ids[:s] for s in starts]


# --------------------------------------------------------------------------- #
# 2./3. re-tokenization
# --------------------------------------------------------------------------- #
def test_exact_ids_equal_full_render_for_canonical_turns_qwen25(qwen25):
    fmt = chat_format_for("Qwen/Qwen2.5-0.5B-Instruct", qwen25)
    eps, _, _ = _episodes(qwen25, [Q2])
    (traj,) = rollout(ScriptedGen(qwen25, fmt, {Q2["question"]: SCRIPT2}), fmt, eps)
    obs = [t["observation"] for t in traj.result.turns[:-1]]
    msgs = _messages(Q2["question"], True, SCRIPT2, obs)
    full = fmt.render(msgs, tools=tool_schemas(True), add_generation_prompt=False)
    # the template ends the last assistant turn with "<|im_end|>\n"; the
    # trajectory stops at the sampled <|im_end|>
    assert full.endswith(fmt.eot_str + "\n")
    full = full[:-1]
    assert qwen25.decode(traj.ids) == full
    assert traj.ids == qwen25(full, add_special_tokens=False)["input_ids"]


def test_non_canonical_sampled_split_is_kept(qwen25):
    """If the model samples a non-canonical split, re-tokenizing would score a
    different sequence. The trajectory keeps the sampled ids."""
    fmt = chat_format_for("Qwen/Qwen2.5-0.5B-Instruct", qwen25)
    text = call("answer", text="Maria Okafor")
    canon = qwen25(text, add_special_tokens=False)["input_ids"]
    # split "Maria" into single characters
    pre, post = text.split("Maria")
    ids = (qwen25(pre, add_special_tokens=False)["input_ids"]
           + [qwen25.convert_tokens_to_ids(c) for c in "Maria"]
           + qwen25(post, add_special_tokens=False)["input_ids"] + [fmt.eot_id])
    assert qwen25.decode(ids[:-1]) == text and ids[:-1] != canon
    eps, _, _ = _episodes(qwen25, [Q1])
    (traj,) = rollout(ScriptedGen(qwen25, fmt, {Q1["question"]: [ids]}), fmt, eps)
    assert traj.result.correct
    assert traj.ids[traj.prompt_len:] == ids                     # sampled ids, untouched
    retok = qwen25(qwen25.decode(traj.ids), add_special_tokens=False)["input_ids"]
    assert retok != traj.ids                                      # the pitfall is real


def test_qwen3_template_rewrites_history_but_trajectory_does_not(qwen3):
    fmt = chat_format_for("Qwen/Qwen3-1.7B", qwen3)
    assert "<think>" in fmt.gen_prompt                  # non-thinking: empty think block
    eps, _, _ = _episodes(qwen3, [Q2])
    gen = ScriptedGen(qwen3, fmt, {Q2["question"]: SCRIPT2})
    (traj,) = rollout(gen, fmt, eps)
    decoded = qwen3.decode(traj.ids)
    # the model was prompted with an empty think block before EVERY turn
    assert decoded.count("<think>\n\n</think>") == len(SCRIPT2)
    obs = [t["observation"] for t in traj.result.turns[:-1]]
    full = fmt.render(_messages(Q2["question"], True, SCRIPT2, obs),
                      tools=tool_schemas(True), add_generation_prompt=False)
    assert full.count("<think>") < len(SCRIPT2)          # the template dropped some
    assert decoded != full.rstrip("\n")
    # and the mask still covers exactly the sampled ids
    assert [i for i, m in zip(traj.ids, traj.mask) if m] == sum(gen.sampled[Q2["question"]], [])
    assert traj.result.correct


# --------------------------------------------------------------------------- #
# edge cases
# --------------------------------------------------------------------------- #
def test_turn_cut_at_gen_cap_gets_unmasked_end_token(qwen25):
    fmt = chat_format_for("Qwen/Qwen2.5-0.5B-Instruct", qwen25)
    cut = qwen25('<tool_call>\n{"name": "search_files", "argu',
                 add_special_tokens=False)["input_ids"]              # no stop token
    eps, _, _ = _episodes(qwen25, [Q1])
    gen = ScriptedGen(qwen25, fmt, {Q1["question"]: [cut, call("answer", text="Maria Okafor")]})
    (traj,) = rollout(gen, fmt, eps)
    j = traj.prompt_len + len(cut)
    assert traj.ids[traj.prompt_len:j] == cut and all(traj.mask[traj.prompt_len:j])
    assert traj.ids[j] == fmt.eot_id and traj.mask[j] == 0      # appended, not trained
    assert traj.hit_gen_cap == [True, False]
    assert traj.result.turns[0]["parse_error"] == "bad_json" and traj.result.correct


def test_other_eos_is_trained_then_framed(qwen25):
    fmt = chat_format_for("Qwen/Qwen2.5-0.5B-Instruct", qwen25)
    eot2 = qwen25.convert_tokens_to_ids("<|endoftext|>")
    assert eot2 in fmt.stop_ids and eot2 != fmt.eot_id
    first = qwen25(call("search_notes", query="Heron"), add_special_tokens=False)["input_ids"]
    eps, _, _ = _episodes(qwen25, [Q1])
    (traj,) = rollout(ScriptedGen(qwen25, fmt, {Q1["question"]: [first + [eot2],
                                                          call("answer", text="Maria Okafor")]}),
                      fmt, eps)
    j = traj.prompt_len + len(first)
    assert traj.ids[j] == eot2 and traj.mask[j] == 1             # sampled, so trained
    assert traj.ids[j + 1] == fmt.eot_id and traj.mask[j + 1] == 0
    assert traj.hit_gen_cap == [False, False] and traj.result.correct


def test_context_limit_is_checked_on_exact_ids(qwen25):
    fmt = chat_format_for("Qwen/Qwen2.5-0.5B-Instruct", qwen25)
    eps, _, cfg = _episodes(qwen25, [Q1], max_gen_tokens=256)
    p = len(fmt.prompt_ids(eps[0].initial_messages(), tool_schemas(True)))
    eps, _, cfg = _episodes(qwen25, [Q1], max_gen_tokens=256, max_context_tokens=p + 300)
    gen = ScriptedGen(qwen25, fmt, {Q1["question"]: [call("read_note", note_id="notes/projects/heron")]
                                    * 6})
    (traj,) = rollout(gen, fmt, eps)
    # turn 1 fits (p + 256 <= p + 300); afterwards the trajectory is > p + 44
    assert traj.result.ended == "context_limit" and len(traj.result.turns) == 1
    assert gen.calls == 1


def test_zero_length_and_trim():
    assert trim_at_stop([5, 6, 9, 7, 9], [9]) == [5, 6, 9]
    assert trim_at_stop([5, 6], [9]) == [5, 6]


def test_rollout_matches_text_runner_rewards(qwen25):
    """Same scripted turns through the token rollout and the M2 text runner give
    the same episode outcome and tokens read."""
    fmt = chat_format_for("Qwen/Qwen2.5-0.5B-Instruct", qwen25)
    eps, env, cfg = _episodes(qwen25, [Q1, Q2])
    t1, t2 = rollout(ScriptedGen(qwen25, fmt, {Q1["question"]: SCRIPT1, Q2["question"]: SCRIPT2}), fmt, eps)
    for traj, q, script in ((t1, Q1, SCRIPT1), (t2, Q2, SCRIPT2)):
        it = iter(script)
        ref = run_text_episode(lambda m: next(it), env, q, CATALOG, cfg, HFTokenCounter(qwen25))
        assert (traj.result.reward, traj.result.tokens_read, traj.result.correct,
                traj.result.ended) == (ref.reward, ref.tokens_read, ref.correct, ref.ended)


def test_collate_right_pads(qwen25):
    fmt = chat_format_for("Qwen/Qwen2.5-0.5B-Instruct", qwen25)
    eps, _, _ = _episodes(qwen25, [Q1, Q2])
    trajs = rollout(ScriptedGen(qwen25, fmt, {Q1["question"]: SCRIPT1, Q2["question"]: SCRIPT2}), fmt, eps)
    b = collate(trajs, pad_id=qwen25.pad_token_id)
    L = max(len(t.ids) for t in trajs)
    assert b["input_ids"].shape == b["loss_mask"].shape == (2, L)
    for r, t in enumerate(trajs):
        n = len(t.ids)
        assert b["input_ids"][r, :n].tolist() == t.ids
        assert b["loss_mask"][r, :n].tolist() == [float(m) for m in t.mask]
        assert b["attention_mask"][r].sum() == n and b["loss_mask"][r, n:].sum() == 0


# --------------------------------------------------------------------------- #
# 4. real model: teacher-forced log-probs == sampling log-probs
# --------------------------------------------------------------------------- #
def _tiny_qwen_with_vocab(vocab, seed=0):
    from transformers import AutoModelForCausalLM, Qwen2Config
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=vocab, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=4096, tie_word_embeddings=True)
    return AutoModelForCausalLM.from_config(cfg).eval()


def test_teacher_forced_logprobs_match_sampling_logprobs(qwen25):
    fmt = chat_format_for("Qwen/Qwen2.5-0.5B-Instruct", qwen25)
    model = _tiny_qwen_with_vocab(len(qwen25))
    gen = HFGenerator(model, qwen25, stop_ids=fmt.stop_ids, max_new_tokens=6,
                      temperature=1.0, batch_size=2, record_logprobs=True)
    torch.manual_seed(1)
    eps, _, _ = _episodes(qwen25, [Q1, Q2, Q1], max_gen_tokens=6)
    trajs = rollout(gen, fmt, eps)
    b = collate(trajs, pad_id=qwen25.pad_token_id)
    with torch.no_grad():
        lp = sequence_logprobs(model, b["input_ids"], b["attention_mask"],
                               logit_mask=b["loss_mask"], chunk_size=7)
    for r, t in enumerate(trajs):
        assert len(t.result.turns) == 6           # random model: never answers
        got = lp[r][b["loss_mask"][r] > 0].tolist()
        assert len(got) == len(t.sample_logprobs) == sum(t.mask)
        assert got == pytest.approx(t.sample_logprobs, abs=1e-4)


def test_greedy_generator_is_plain_argmax(qwen25):
    """Catches any logits processor sneaking in from the model's generation
    config (e.g. Qwen's repetition_penalty 1.1, top_k 20)."""
    from transformers import GenerationConfig
    model = _tiny_qwen_with_vocab(len(qwen25), seed=3)
    model.generation_config = GenerationConfig(repetition_penalty=1.5, top_k=5, top_p=0.5,
                                               temperature=0.3, do_sample=True)
    gen = HFGenerator(model, qwen25, stop_ids=[qwen25.convert_tokens_to_ids("<|im_end|>")],
                      max_new_tokens=12, greedy=True)
    seq = qwen25("hello hello hello", add_special_tokens=False)["input_ids"]
    (out,) = gen([seq])
    ids = list(seq)
    with torch.no_grad():
        for _ in range(len(out)):
            nxt = model(input_ids=torch.tensor([ids])).logits[0, -1].argmax().item()
            ids.append(nxt)
    assert ids[len(seq):] == out
    assert model.generation_config.repetition_penalty == 1.0
