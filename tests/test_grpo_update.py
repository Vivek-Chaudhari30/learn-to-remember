# Origin: ported from GRPO-Zero `tests/test_grpo_update.py` (main @ ce51ed9).
# Changed: imports, the fake LM accepts position_ids, and pytest.approx is used.
"""Model-free tests for the GRPO update math: group advantages, the clipped+KL
loss, and the per-token log-prob indexing. No GPU or model needed."""

import math

import pytest
import torch

from learn_to_remember.grpo.core import compute_advantages, grpo_loss, per_token_logprobs


# --------------------------------------------------------------------------- #
# compute_advantages
# --------------------------------------------------------------------------- #
def test_advantages_zero_mean_unit_scale_single_group():
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])
    gids = torch.tensor([0, 0, 0, 0])
    adv = compute_advantages(rewards, gids, eps=1e-8)
    assert adv.mean().abs() < 1e-5                 # baseline subtracted
    assert adv.std(unbiased=False).item() == pytest.approx(1.0, abs=1e-5)
    # higher reward -> positive advantage
    assert (adv[:2] > 0).all() and (adv[2:] < 0).all()


def test_advantages_per_group_independent():
    rewards = torch.tensor([1.0, 0.0, 5.0, 5.0])
    gids = torch.tensor([0, 0, 1, 1])
    adv = compute_advantages(rewards, gids, eps=1e-8)
    # group 0 has spread -> nonzero; group 1 is constant -> ~0 (nothing to learn)
    assert adv[0] > 0 and adv[1] < 0
    assert abs(adv[2]) < 1e-3 and abs(adv[3]) < 1e-3


def test_advantages_degenerate_group_is_finite():
    rewards = torch.tensor([2.0, 2.0, 2.0])
    gids = torch.tensor([0, 0, 0])
    adv = compute_advantages(rewards, gids, eps=1e-4)
    assert torch.isfinite(adv).all()
    assert adv.abs().max() < 1e-3


def test_advantages_without_std_keep_the_reward_scale():
    # The 2026-11-26 overfit collapse (docs/incidents/): on an always-wrong question the
    # samples differ only by reading cost (0.09075 with one lookup, 0.1 with none).
    wrong = torch.tensor([0.09075] * 7 + [0.1])
    mixed = torch.tensor([1.1, 1.1, 1.1, 1.1, 0.1, 0.1, 0.1, 0.1])     # right vs wrong
    g = torch.zeros(8, dtype=torch.long)
    # dividing by the std makes the 0.009 cost gap as strong as right vs wrong
    assert compute_advantages(wrong, g).abs().max() > 2.0
    assert compute_advantages(mixed, g).abs().max() == pytest.approx(1.0, abs=1e-3)
    # without it the cost gap stays tiny and right vs wrong dominates
    a_wrong = compute_advantages(wrong, g, scale_by_std=False)
    a_mixed = compute_advantages(mixed, g, scale_by_std=False)
    assert a_wrong.abs().max() < 0.01
    assert torch.allclose(a_mixed, mixed - mixed.mean())
    assert a_mixed.abs().max() / a_wrong.abs().max() > 50
    assert a_wrong[-1] > 0                          # reading less still helps, a little
    # a constant group still gives zero advantage
    assert compute_advantages(torch.full((4,), 0.1), torch.zeros(4, dtype=torch.long),
                              scale_by_std=False).abs().max() == 0


# --------------------------------------------------------------------------- #
# grpo_loss
# --------------------------------------------------------------------------- #
def _ones(*shape):
    return torch.ones(*shape)


def test_loss_on_policy_reduces_to_neg_advantage():
    # ratio == 1 (new == old), no KL -> per-token loss = -A, averaged over seqs
    new = torch.zeros(2, 1)
    old = torch.zeros(2, 1)
    ref = torch.zeros(2, 1)
    adv = torch.tensor([2.0, -1.0])
    loss, m = grpo_loss(new, old, ref, adv, _ones(2, 1), clip_eps=0.2, kl_coef=0.0)
    assert loss.item() == pytest.approx((-2.0 + 1.0) / 2, abs=1e-6)   # = -0.5
    assert m["clip_frac"] == 0.0


def test_loss_ppo_clip_caps_positive_advantage():
    # new-old = 0.5 -> ratio = e^0.5 ~ 1.649 > 1.2; positive adv -> clipped to 1.2
    new = torch.tensor([[0.5]])
    old = torch.tensor([[0.0]])
    ref = torch.tensor([[0.5]])           # diff 0 -> no KL
    adv = torch.tensor([1.0])
    loss, m = grpo_loss(new, old, ref, adv, _ones(1, 1), clip_eps=0.2, kl_coef=0.0)
    assert loss.item() == pytest.approx(-1.2, abs=1e-6)
    assert m["clip_frac"] == 1.0


def test_loss_kl_term_matches_k3_estimator():
    # ratio 1, adv 0 -> only KL contributes. ref-new = 0.5
    new = torch.tensor([[0.0]])
    old = torch.tensor([[0.0]])
    ref = torch.tensor([[0.5]])
    adv = torch.tensor([0.0])
    kl_coef = 0.04
    loss, m = grpo_loss(new, old, ref, adv, _ones(1, 1), clip_eps=0.2, kl_coef=kl_coef)
    expected_kl = math.exp(0.5) - 0.5 - 1.0
    assert m["kl"] == pytest.approx(expected_kl, rel=1e-5)
    assert loss.item() == pytest.approx(kl_coef * expected_kl, rel=1e-5)


def test_loss_respects_completion_mask():
    # token 1 carries a huge KL but is masked out -> must not affect the loss
    new = torch.tensor([[0.0, 0.0]])
    old = torch.tensor([[0.0, 0.0]])
    ref = torch.tensor([[0.0, 10.0]])     # token1 KL is enormous
    adv = torch.tensor([0.0])
    mask = torch.tensor([[1.0, 0.0]])
    loss, _ = grpo_loss(new, old, ref, adv, mask, clip_eps=0.2, kl_coef=1.0)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# per_token_logprobs (indexing): tiny fake model returning fixed logits
# --------------------------------------------------------------------------- #
class _Out:
    def __init__(self, logits):
        self.logits = logits


class _FakeLM:
    def __init__(self, logits):
        self._logits = logits

    def __call__(self, input_ids=None, attention_mask=None, **kwargs):
        return _Out(self._logits)


def test_per_token_logprobs_picks_completion_positions():
    torch.manual_seed(0)
    B, L, V = 1, 4, 6
    logits = torch.randn(B, L, V)
    full = torch.tensor([[1, 2, 3, 4]])   # P=2 prompt, C=2 completion
    prompt_ids, completion_ids = full[:, :2], full[:, 2:]
    ones = torch.ones(B, 2, dtype=torch.long)

    got = per_token_logprobs(_FakeLM(logits), prompt_ids, ones, completion_ids, ones)

    # reference: standard log_softmax + gather, then slice the completion tokens
    ref_lp = torch.log_softmax(logits[:, :-1, :], dim=-1)
    gathered = ref_lp.gather(-1, full[:, 1:].unsqueeze(-1)).squeeze(-1)  # [1, 3]
    expected = gathered[:, 1:]                                          # last C=2
    assert torch.allclose(got, expected, atol=1e-5)
    assert got.shape == (1, 2)
