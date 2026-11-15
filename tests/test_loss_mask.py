"""The generic `loss_mask` path (new in learn-to-remember).

A multi-turn trajectory is one row [prompt | gen | tool | gen | ...]. The loss
must count only model-generated tokens. These tests prove:
  1. GRPO-Zero's contiguous prompt/completion behavior is exactly the special
     case where the mask covers the completion block.
  2. An interleaved mask counts only the masked-in tokens, in value and in
     gradient, even when masked-out tokens hold extreme (inf) values.
  3. The Dr. GRPO normalization option matches its formula.
"""

import pytest
import torch

from learn_to_remember.grpo.core import (
    grpo_loss, per_token_logprobs, sequence_logprobs,
)


class _Out:
    def __init__(self, logits):
        self.logits = logits


class _FakeLM:
    def __init__(self, logits):
        self._logits = logits

    def __call__(self, input_ids=None, attention_mask=None, **kwargs):
        return _Out(self._logits)


def _rand_logprobs(*shape, seed):
    g = torch.Generator().manual_seed(seed)
    return -torch.rand(*shape, generator=g) * 3.0


# --------------------------------------------------------------------------- #
# 1. Contiguous special case == GRPO-Zero behavior
# --------------------------------------------------------------------------- #
def test_sequence_logprobs_alignment_and_first_position_zero():
    torch.manual_seed(1)
    B, L, V = 2, 5, 7
    logits = torch.randn(B, L, V)
    ids = torch.randint(0, V, (B, L))
    got = sequence_logprobs(_FakeLM(logits), ids, torch.ones(B, L, dtype=torch.long))
    assert got.shape == (B, L)
    assert torch.all(got[:, 0] == 0)
    lsm = torch.log_softmax(logits, dim=-1)
    for t in range(1, L):
        expected = lsm[torch.arange(B), t - 1, ids[:, t]]
        assert torch.allclose(got[:, t], expected, atol=1e-5)


def test_per_token_logprobs_equals_sequence_slice():
    torch.manual_seed(2)
    B, P, C, V = 3, 4, 5, 11
    logits = torch.randn(B, P + C, V)
    prompt, comp = torch.randint(0, V, (B, P)), torch.randint(0, V, (B, C))
    pm, cm = torch.ones(B, P, dtype=torch.long), torch.ones(B, C, dtype=torch.long)
    lm = _FakeLM(logits)
    old_api = per_token_logprobs(lm, prompt, pm, comp, cm)
    seq = sequence_logprobs(lm, torch.cat([prompt, comp], 1), torch.cat([pm, cm], 1))
    assert torch.allclose(old_api, seq[:, P:], atol=1e-6)


def test_full_sequence_mask_reproduces_completion_only_loss():
    """GRPO-Zero computed the loss on [B, C] completion log-probs with a
    completion mask. Doing it on [B, P+C] full-sequence log-probs with the mask
    zero on the prompt and equal to the completion mask on the completion must
    give the identical loss and metrics."""
    B, P, C = 4, 6, 9
    new = _rand_logprobs(B, P + C, seed=10)
    old = _rand_logprobs(B, P + C, seed=11)
    ref = _rand_logprobs(B, P + C, seed=12)
    adv = torch.tensor([1.5, -0.5, 0.0, -1.0])
    comp_mask = torch.ones(B, C)
    comp_mask[1, 5:] = 0          # an EOS mid-completion
    comp_mask[3, 2:] = 0
    full_mask = torch.cat([torch.zeros(B, P), comp_mask], dim=1)

    kw = dict(clip_eps=0.2, kl_coef=0.04)
    loss_old, m_old = grpo_loss(new[:, P:], old[:, P:], ref[:, P:], adv, comp_mask, **kw)
    loss_new, m_new = grpo_loss(new, old, ref, adv, full_mask, **kw)
    # rel 1e-6: summing over the extra (zeroed) prompt slots changes float32
    # reduction order, which moves the last bit or two; the math is identical.
    assert loss_new.item() == pytest.approx(loss_old.item(), rel=1e-6)
    for k in ("policy_loss", "kl", "clip_frac", "n_loss_tokens"):
        assert m_new[k] == pytest.approx(m_old[k], rel=1e-6, abs=1e-9)


# --------------------------------------------------------------------------- #
# 2. Interleaved mask: only generated tokens count
# --------------------------------------------------------------------------- #
# layout of one row: [prompt x3 | gen x2 | tool x3 | gen x2 | pad x2]
_MASK = torch.tensor([[0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0]], dtype=torch.float32)


def test_interleaved_mask_matches_manual_computation():
    T = _MASK.shape[1]
    new, old, ref = (_rand_logprobs(1, T, seed=s) for s in (20, 21, 22))
    adv = torch.tensor([0.7])
    clip, beta = 0.2, 0.1
    loss, m = grpo_loss(new, old, ref, adv, _MASK, clip_eps=clip, kl_coef=beta)

    idx = _MASK[0].nonzero().squeeze(1)
    ratio = torch.exp(new[0, idx] - old[0, idx])
    pol = -torch.min(ratio * 0.7, ratio.clamp(1 - clip, 1 + clip) * 0.7)
    d = ref[0, idx] - new[0, idx]
    kl = torch.exp(d) - d - 1
    assert loss.item() == pytest.approx((pol + beta * kl).mean().item(), abs=1e-6)
    assert m["n_loss_tokens"] == 4


def test_masked_out_tokens_cannot_change_loss_even_if_extreme():
    T = _MASK.shape[1]
    new, old, ref = (_rand_logprobs(1, T, seed=s) for s in (30, 31, 32))
    adv = torch.tensor([-1.2])
    base, _ = grpo_loss(new, old, ref, adv, _MASK, clip_eps=0.2, kl_coef=0.04)

    out = _MASK[0] == 0
    new2, old2, ref2 = new.clone(), old.clone(), ref.clone()
    new2[0, out] = -50.0          # tool/prompt tokens: wildly different values
    old2[0, out] = 40.0
    ref2[0, out] = float("inf")   # exp(inf) would poison a `* mask` implementation
    loss2, m2 = grpo_loss(new2, old2, ref2, adv, _MASK, clip_eps=0.2, kl_coef=0.04)
    assert torch.isfinite(loss2)
    assert loss2.item() == pytest.approx(base.item(), abs=1e-6)
    assert all(torch.isfinite(torch.tensor(v)) for v in m2.values())


def test_gradient_flows_only_to_masked_in_tokens():
    T = _MASK.shape[1]
    new = _rand_logprobs(1, T, seed=40).requires_grad_(True)
    old, ref = _rand_logprobs(1, T, seed=41), _rand_logprobs(1, T, seed=42)
    loss, _ = grpo_loss(new, old, ref, torch.tensor([0.9]), _MASK,
                        clip_eps=0.2, kl_coef=0.04)
    loss.backward()
    g = new.grad[0]
    assert torch.all(g[_MASK[0] == 0] == 0)
    assert torch.all(g[_MASK[0] == 1] != 0)


def test_all_zero_mask_row_contributes_zero_not_nan():
    new, old, ref = (_rand_logprobs(2, 5, seed=s) for s in (50, 51, 52))
    mask = torch.tensor([[1, 1, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.float32)
    loss, _ = grpo_loss(new, old, ref, torch.tensor([1.0, 1.0]), mask,
                        clip_eps=0.2, kl_coef=0.04)
    one, _ = grpo_loss(new[:1], old[:1], ref[:1], torch.tensor([1.0]), mask[:1],
                       clip_eps=0.2, kl_coef=0.04)
    assert loss.item() == pytest.approx(one.item() / 2, abs=1e-7)


# --------------------------------------------------------------------------- #
# 3. Normalization options
# --------------------------------------------------------------------------- #
def test_dr_grpo_normalization_formula():
    # on-policy, no KL -> per-token loss = -A; rows of different length
    new = torch.zeros(2, 4)
    mask = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.float32)
    adv = torch.tensor([1.0, 1.0])
    seq, _ = grpo_loss(new, new, new, adv, mask, clip_eps=0.2, kl_coef=0.0)
    dr, _ = grpo_loss(new, new, new, adv, mask, clip_eps=0.2, kl_coef=0.0,
                      norm="dr_grpo", max_tokens=8)
    assert seq.item() == pytest.approx(-1.0)            # each row's mean is -1
    assert dr.item() == pytest.approx(-5.0 / (2 * 8))   # 5 tokens of -1 over B*max_tokens


def test_unknown_norm_rejected():
    z = torch.zeros(1, 1)
    with pytest.raises(ValueError):
        grpo_loss(z, z, z, torch.zeros(1), torch.ones(1, 1), clip_eps=0.2,
                  kl_coef=0.0, norm="token_mean")
