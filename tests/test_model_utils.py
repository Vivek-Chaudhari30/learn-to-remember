"""Tests with a real (tiny, randomly initialized, offline) Qwen2 model + LoRA.

No download: the model is built from a config. This checks the parts that a
fake-logits model can't: position ids under padding, the reference policy via
`disable_adapter`, and checkpoint -> resume round-trips.
"""

import pytest
import torch

from learn_to_remember.grpo.core import (
    batched_logprobs, grpo_loss, per_token_logprobs, sequence_logprobs,
)
from learn_to_remember.model_utils import (
    STATE_FILE, apply_lora, get_device, maybe_resume, resolve_dtype, save_checkpoint,
    set_seed,
)

LORA = {"r": 4, "alpha": 8, "dropout": 0.0,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}


def _tiny_model(seed=0):
    from transformers import AutoModelForCausalLM, Qwen2Config

    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=128)
    model = AutoModelForCausalLM.from_config(cfg)
    model.eval()
    return model


def _perturb_lora(model, seed=0):
    """LoRA B starts at zero (adapter == identity). Give it weight so the
    adapter actually changes the output."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)


# --------------------------------------------------------------------------- #
# device / dtype
# --------------------------------------------------------------------------- #
def test_get_device_respects_explicit_pref():
    assert get_device("cpu").type == "cpu"


def test_resolve_dtype_auto_and_explicit():
    assert resolve_dtype("auto", torch.device("cpu")) == torch.float32
    assert resolve_dtype("auto", torch.device("cuda")) == torch.bfloat16
    assert resolve_dtype("auto", torch.device("mps")) == torch.float16
    assert resolve_dtype("fp16", torch.device("cpu")) == torch.float16


# --------------------------------------------------------------------------- #
# log-probs on a real model
# --------------------------------------------------------------------------- #
def test_padding_does_not_change_logprobs_of_real_tokens():
    """Left- or right-padding a row must not change the log-probs of its real
    tokens. (For RoPE models this holds even with default position ids, since a
    constant shift doesn't change relative positions. It is an invariance check
    on the whole padding/masking path, not a test of position ids alone.)"""
    model = _tiny_model()
    ids = torch.tensor([[5, 9, 3, 17, 22, 8]])
    attn = torch.ones_like(ids)
    plain = sequence_logprobs(model, ids, attn)

    pad = 0
    left = torch.cat([torch.full((1, 3), pad), ids], 1)
    left_attn = torch.cat([torch.zeros(1, 3, dtype=torch.long), attn], 1)
    right = torch.cat([ids, torch.full((1, 3), pad)], 1)
    right_attn = torch.cat([attn, torch.zeros(1, 3, dtype=torch.long)], 1)
    with torch.no_grad():
        lp_left = sequence_logprobs(model, left, left_attn)[:, 3:]
        lp_right = sequence_logprobs(model, right, right_attn)[:, :6]
    # position 0 of the left-padded row is "predicted" from a pad, so skip it
    assert torch.allclose(lp_left[:, 1:], plain[:, 1:], atol=1e-5)
    assert torch.allclose(lp_right, plain, atol=1e-5)


def test_per_token_logprobs_real_model_matches_sequence_slice():
    model = _tiny_model()
    prompt, comp = torch.tensor([[4, 5, 6]]), torch.tensor([[7, 8]])
    ones3, ones2 = torch.ones(1, 3, dtype=torch.long), torch.ones(1, 2, dtype=torch.long)
    with torch.no_grad():
        a = per_token_logprobs(model, prompt, ones3, comp, ones2)
        b = sequence_logprobs(model, torch.cat([prompt, comp], 1), torch.ones(1, 5))[:, 3:]
    assert torch.allclose(a, b, atol=1e-6)


def test_disable_adapter_gives_frozen_base_logprobs():
    model = _tiny_model()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1]])
    attn = torch.ones_like(ids)
    with torch.no_grad():
        base_lp = sequence_logprobs(model, ids, attn)

    policy = apply_lora(model, LORA)
    _perturb_lora(policy)
    policy.eval()
    pol_lp = batched_logprobs(policy, ids, attn, micro_batch=1)
    ref_lp = batched_logprobs(policy, ids, attn, micro_batch=1, disable_adapter=True)
    assert torch.allclose(ref_lp, base_lp, atol=1e-5)
    assert not torch.allclose(pol_lp, base_lp, atol=1e-3)


# --------------------------------------------------------------------------- #
# checkpoint / resume
# --------------------------------------------------------------------------- #
def _one_step(policy, opt, ids, mask):
    attn = torch.ones_like(ids)
    lp = sequence_logprobs(policy, ids, attn)
    loss, _ = grpo_loss(lp, lp.detach(), lp.detach(), torch.tensor([1.0, -1.0]), mask,
                        clip_eps=0.2, kl_coef=0.0)
    opt.zero_grad()
    loss.backward()
    opt.step()


def test_checkpoint_resume_round_trip(tmp_path):
    set_seed(0)
    policy = apply_lora(_tiny_model(), LORA)
    _perturb_lora(policy)
    policy.train()
    opt = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=1e-2)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])
    mask = torch.tensor([[0, 0, 1, 1, 0, 1], [0, 0, 1, 1, 1, 0]], dtype=torch.float32)
    _one_step(policy, opt, ids, mask)

    history = [{"step": 0, "reward_mean": 0.5}]
    save_checkpoint(str(tmp_path), policy, opt, step=0, history=history)
    assert (tmp_path / STATE_FILE).exists()
    assert (tmp_path / "adapter_model.safetensors").exists()
    expected_next = torch.rand(3)            # RNG draw right after the save

    # fresh policy (same base weights, different adapter) + fresh optimizer
    fresh = apply_lora(_tiny_model(), LORA)
    _perturb_lora(fresh, seed=99)
    fresh_opt = torch.optim.AdamW((p for p in fresh.parameters() if p.requires_grad), lr=1e-2)
    start, hist = maybe_resume(str(tmp_path), fresh, fresh_opt, torch.device("cpu"))

    assert start == 1 and hist == history
    assert torch.allclose(torch.rand(3), expected_next)   # RNG restored
    trained = {n: p for n, p in policy.named_parameters() if p.requires_grad}
    for n, p in fresh.named_parameters():
        if p.requires_grad:
            assert torch.allclose(p, trained[n]), n
    # optimizer moments restored -> the next step is identical in both copies
    _one_step(policy, opt, ids, mask)
    _one_step(fresh, fresh_opt, ids, mask)
    for n, p in fresh.named_parameters():
        if p.requires_grad:
            assert torch.allclose(p, trained[n], atol=1e-6), n


def test_maybe_resume_without_checkpoint_starts_fresh(tmp_path):
    policy = apply_lora(_tiny_model(), LORA)
    opt = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=1e-3)
    assert maybe_resume(str(tmp_path), policy, opt, torch.device("cpu")) == (0, [])


@pytest.mark.parametrize("gc", [False, True])
def test_apply_lora_only_adapter_trainable(gc):
    policy = apply_lora(_tiny_model(), LORA, grad_checkpoint=gc)
    trainable = [n for n, p in policy.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)
