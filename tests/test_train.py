"""The M4 GRPO step and loop, on a tiny random Qwen2 (real Qwen2.5 tokenizer).

Rollouts come from a scripted generator, so rewards are controlled and runs
are deterministic; everything after the rollout (advantages, reference
log-probs, masked loss, accumulation, optimizer, checkpoints) is the real code.
"""

import json
from collections import Counter

import pytest
import torch
import torch.nn.functional as F

from learn_to_remember.env.episode import Episode, EnvConfig
from learn_to_remember.env.tokens import HFTokenCounter
from learn_to_remember.env.tools import ToolEnv
from learn_to_remember.grpo.core import sequence_logprobs
from learn_to_remember.grpo.rollout import chat_format_for
from learn_to_remember.grpo.train import QuestionSampler, make_scaler, train_loop, train_step
from learn_to_remember.model_utils import apply_lora, load_config, maybe_resume
from learn_to_remember.workspace.questions import QTYPES

LORA = {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]}
DOCS = [{"path": "files/a.md", "date": "2025-01-01", "title": "Status report: Project Heron",
         "text": "# Status report: Project Heron\nOwner: Maria Okafor\n"}]
NOTES = [{"note_id": "notes/projects/heron", "title": "Project Heron", "as_of": "2025-09-30",
          "text": "# Project Heron\nowner: Maria Okafor\n"}]
CATALOG = {"person": ["Maria Okafor", "Ana Silva"]}


def call(name, **args):
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"


def _q(i):
    return {"id": f"q{i}", "question": f"Who owns Project Heron? (#{i})", "answer_type": "person",
            "gold": "Maria Okafor", "qtype_by_version": {"v3": QTYPES[i % 5]}}


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    except Exception:
        pytest.skip("Qwen2.5 tokenizer not cached")


def _policy(tok, seed=0):
    from transformers import AutoModelForCausalLM, Qwen2Config
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=16, intermediate_size=32,
                      num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                      max_position_embeddings=4096, tie_word_embeddings=True)
    p = apply_lora(AutoModelForCausalLM.from_config(cfg), LORA)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():                     # non-zero B so adapter != reference
        for n, prm in p.named_parameters():
            if "lora_B" in n:
                prm.copy_(torch.randn(prm.shape, generator=g) * 0.3)
    return p


class PositionalGen:
    """turn k, active row i -> scripts[i][k] (all rows stay active for 2 turns)."""

    def __init__(self, tok, fmt, scripts):
        self.tok, self.fmt, self.scripts, self.turn = tok, fmt, scripts, 0

    def __call__(self, seqs):
        out = [self.tok(self.scripts[i][self.turn], add_special_tokens=False)["input_ids"]
               + [self.fmt.eot_id] for i in range(len(seqs))]
        self.turn += 1
        return out


def _setup(tok, n_q=2, G=4, answers=None):
    fmt = chat_format_for("Qwen/Qwen2.5-0.5B-Instruct", tok)
    env = ToolEnv(DOCS, NOTES, HFTokenCounter(tok))
    cfg = EnvConfig()
    qs = [_q(i) for i in range(n_q)]
    answers = answers or ["Maria Okafor", "Ana Silva"]
    scripts = [[call("read_note", note_id="notes/projects/heron"),
                call("answer", text=answers[i % len(answers)])] for i in range(n_q * G)]

    def make_eps(group_qs):
        return [Episode(env, q, CATALOG, cfg) for q in group_qs]

    return fmt, qs, scripts, make_eps


TCFG = {"micro_batch": 2, "adv_eps": 1e-4, "ppo_epochs": 1, "clip_eps": 0.2, "kl_coef": 0.04,
        "grad_clip": 1e9, "loss_norm": "seq_mean"}


def _grads_after_step(tok, micro, **kw):
    policy = _policy(tok)
    fmt, qs, scripts, make_eps = _setup(tok, **kw)
    opt = torch.optim.SGD([p for p in policy.parameters() if p.requires_grad], lr=0.0)
    m, trajs = train_step(policy, tok, fmt, PositionalGen(tok, fmt, scripts), make_eps, qs, opt,
                          {**TCFG, "micro_batch": micro}, group_size=4)
    return {n: p.grad.clone() for n, p in policy.named_parameters() if p.requires_grad}, m, trajs


def test_accumulation_is_exact_across_micro_batch_sizes(tok):
    g1, m1, trajs = _grads_after_step(tok, 1)
    g3, m3, _ = _grads_after_step(tok, 3)          # 8 rows -> 3 + 3 + 2 (uneven)
    g8, m8, _ = _grads_after_step(tok, 8)
    assert len({len(t.ids) for t in trajs}) > 1    # rows really have different lengths
    for n in g1:
        assert torch.allclose(g1[n], g8[n], atol=1e-6), n
        assert torch.allclose(g3[n], g8[n], atol=1e-6), n
    assert m1["loss"] == pytest.approx(m8["loss"], abs=1e-6)
    assert m1["kl"] == pytest.approx(m8["kl"], abs=1e-6)
    assert m1["entropy"] == pytest.approx(m8["entropy"], rel=1e-5)


def test_step_metrics_and_on_policy_ratio(tok):
    _, m, trajs = _grads_after_step(tok, 2)
    assert m["n_trajectories"] == m["n_trained"] == 8
    assert m["frac_correct"] == 0.5 and m["frac_zero_var_groups"] == 0.0
    assert m["clip_frac"] == 0.0                   # ppo_epochs=1: old == new exactly
    assert m["kl"] > 0                             # adapter perturbed away from reference
    assert m["entropy"] > 0 and m["valid_call_rate"] == 1.0
    assert m["n_loss_tokens"] == sum(sum(t.mask) for t in trajs)
    assert m["ended"] == {"answer": 8}


def test_only_lora_weights_change(tok):
    policy = _policy(tok)
    fmt, qs, scripts, make_eps = _setup(tok)
    before = {n: p.detach().clone() for n, p in policy.named_parameters()}
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-2)
    train_step(policy, tok, fmt, PositionalGen(tok, fmt, scripts), make_eps, qs, opt, TCFG,
               group_size=4)
    changed = {n for n, p in policy.named_parameters() if not torch.equal(p, before[n])}
    assert changed and all("lora_" in n for n in changed)
    assert any("lora_A" in n for n in changed)


def test_zero_variance_groups_are_skipped_when_asked(tok):
    policy = _policy(tok)
    fmt, qs, scripts, make_eps = _setup(tok, answers=["Maria Okafor"])     # all correct
    opt = torch.optim.SGD([p for p in policy.parameters() if p.requires_grad], lr=0.1)
    m, _ = train_step(policy, tok, fmt, PositionalGen(tok, fmt, scripts), make_eps, qs, opt,
                      {**TCFG, "skip_zero_variance_groups": True}, group_size=4)
    assert m["frac_zero_var_groups"] == 1.0 and m["skipped_groups"] == 2
    assert m["n_trained"] == 0
    assert all(p.grad is None for p in policy.parameters() if p.requires_grad)
    # not skipped: the KL term still trains even with zero advantage
    policy = _policy(tok)
    fmt, qs, scripts, make_eps = _setup(tok, answers=["Maria Okafor"])
    opt = torch.optim.SGD([p for p in policy.parameters() if p.requires_grad], lr=0.1)
    m, _ = train_step(policy, tok, fmt, PositionalGen(tok, fmt, scripts), make_eps, qs, opt,
                      TCFG, group_size=4)
    assert m["n_trained"] == 8 and m["policy_loss"] == pytest.approx(0.0, abs=1e-6)


def test_resume_reproduces_an_uninterrupted_run(tok, tmp_path):
    def run(steps, start=0, save_dir=None, resume=False):
        torch.manual_seed(0)
        policy = _policy(tok)
        fmt, qs, scripts, make_eps = _setup(tok)
        opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-2)
        hist = []
        if resume:
            start, hist = maybe_resume(str(save_dir), policy, opt, "cpu")
        sampler = QuestionSampler(qs, per_step=2, mix="natural")

        class Fresh:                     # a fresh scripted generator per step
            def __call__(self, seqs):
                if not hasattr(self, "g") or self.g.turn == 2:
                    self.g = PositionalGen(tok, fmt, scripts)
                return self.g(seqs)
        hist = train_loop(policy=policy, tok=tok, fmt=fmt, generate=Fresh(),
                          make_episodes=make_eps, sampler=sampler, optimizer=opt,
                          tcfg={**TCFG, "checkpoint_every": 1}, group_size=4, steps=steps,
                          save_dir=str(save_dir or tmp_path / "x"), start_step=start,
                          history=hist, log_path=str(tmp_path / "log.json"), log=lambda s: None)
        return policy, hist

    full, h_full = run(3, save_dir=tmp_path / "full")
    run(2, save_dir=tmp_path / "part")
    resumed, h_res = run(3, save_dir=tmp_path / "part", resume=True)
    assert [h["step"] for h in h_res] == [0, 1, 2]
    for (n, a), (_, b) in zip(full.named_parameters(), resumed.named_parameters()):
        assert torch.allclose(a, b, atol=1e-6), n
    assert h_res[-1]["loss"] == pytest.approx(h_full[-1]["loss"], abs=1e-6)


# --------------------------------------------------------------------------- #
# sampler, config, scaler, entropy
# --------------------------------------------------------------------------- #
def _pool():
    # unbalanced like the real training mix
    sizes = {"lookup": 40, "multihop": 6, "aggregate": 4, "gap": 7, "stale": 3}
    qs, i = [], 0
    for t, n in sizes.items():
        for _ in range(n):
            qs.append({"id": f"t{i:03d}", "qtype_by_version": {"v3": t}})
            i += 1
    return qs


def test_stratified_sampler_is_balanced_stateless_and_epoch_walks():
    s = QuestionSampler(_pool(), per_step=8, seed=1)
    steps = [s.for_step(k) for k in range(15)]
    assert all(len(b) == 8 for b in steps)
    tot = Counter(q["qtype_by_version"]["v3"] for b in steps for q in b)
    assert set(tot.values()) == {24}                         # 15 * 8 / 5 each
    for b in steps:                                           # per step: 1 or 2 per qtype
        assert set(Counter(q["qtype_by_version"]["v3"] for q in b).values()) <= {1, 2}
    again = QuestionSampler(list(reversed(_pool())), per_step=8, seed=1)
    assert [q["id"] for q in again.for_step(11)] == [q["id"] for q in steps[11]]
    # the 3 stale questions: each seen once per epoch of 3 stale draws
    stale = [q["id"] for b in steps for q in b if q["qtype_by_version"]["v3"] == "stale"]
    for e in range(len(stale) // 3):
        assert len(set(stale[3 * e:3 * e + 3])) == 3


def test_natural_and_overfit_modes():
    nat = QuestionSampler(_pool(), per_step=10, mix="natural", seed=0)
    first_epoch = [q["id"] for k in range(6) for q in nat.for_step(k)]
    assert sorted(first_epoch) == sorted(q["id"] for q in _pool())  # 60 = one epoch
    ov = QuestionSampler(_pool(), per_step=99, overfit_n=8, seed=0)
    assert ov.for_step(0) == ov.for_step(7) and len(ov.for_step(0)) == 8
    assert len({q["qtype_by_version"]["v3"] for q in ov.for_step(0)}) == 5


def test_config_extends_deep_merges(tmp_path):
    (tmp_path / "base.yaml").write_text("a: 1\ntrain:\n  lr: 1\n  lora: {r: 16}\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\ntrain:\n  lr: 2\n")
    c = load_config(str(tmp_path / "child.yaml"))
    assert c == {"a": 1, "train": {"lr": 2, "lora": {"r": 16}}}


def test_repo_configs_load():
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("train_notes", "train_nonotes", "overfit"):
        c = load_config(os.path.join(root, "configs", f"{name}.yaml"))
        assert c["train"]["lora"]["r"] == 16 and c["rollout"]["group_size"] == 8
    assert load_config(os.path.join(root, "configs", "train_nonotes.yaml"))["train"][
        "condition"] == "no_notes"


def test_scaler_only_for_fp16_cuda(tok):
    assert make_scaler(_policy(tok)) is None          # fp32 on CPU


def test_entropy_matches_full_softmax(tok):
    policy = _policy(tok).eval()
    ids = torch.randint(0, 500, (2, 30), generator=torch.Generator().manual_seed(0))
    attn = torch.ones_like(ids)
    mask = torch.zeros(2, 30)
    mask[:, 10:20] = 1
    with torch.no_grad():
        lp, ent = sequence_logprobs(policy, ids, attn, logit_mask=mask, return_entropy=True,
                                    chunk_size=3)
        logits = policy(input_ids=ids, attention_mask=attn).logits.float()
        ref = -(F.softmax(logits, -1) * F.log_softmax(logits, -1)).sum(-1)   # [B, L]
    m = mask.bool()
    # entropy at position t is of the distribution that predicts token t (from t-1)
    assert torch.allclose(ent[:, 10:20], ref[:, 9:19], atol=1e-4)
    assert torch.all(ent[~m] == 0)
