"""Chunked log-probs (M3): never build the [B, L, V] logits.

`sequence_logprobs` runs the backbone once, then lm_head + logsumexp over
slices of positions (checkpointed when grads are on), optionally only at the
positions in `logit_mask`. These tests pin it to the plain full-logits
computation in value and in gradient, through LoRA and gradient checkpointing.
"""

import pytest
import torch
import torch.nn.functional as F

from learn_to_remember.grpo.core import (
    batched_logprobs, grpo_loss, position_ids_from_mask, sequence_logprobs,
)
from learn_to_remember.model_utils import apply_lora

LORA = {"r": 4, "alpha": 8, "dropout": 0.0,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}


def _tiny(seed=0, vocab=97):
    from transformers import AutoModelForCausalLM, Qwen2Config
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=vocab, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=256)
    return AutoModelForCausalLM.from_config(cfg).eval()


def _perturb(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)


def reference_logprobs(model, ids, attn):
    """The M0 implementation: full logits, then selected - logsumexp."""
    logits = model(input_ids=ids, attention_mask=attn,
                   position_ids=position_ids_from_mask(attn)).logits[:, :-1].float()
    sel = logits.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return F.pad(sel - torch.logsumexp(logits, -1), (1, 0), value=0.0)


def _batch(B=3, L=40, vocab=97, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (B, L), generator=g)
    attn = torch.ones(B, L, dtype=torch.long)
    attn[1, 30:] = 0                     # right padding on one row
    mask = (torch.rand(B, L, generator=g) > 0.6).float() * attn
    return ids, attn, mask


@pytest.mark.parametrize("chunk", [1, 7, 64, 4096])
def test_chunked_equals_full_logits(chunk):
    model = _tiny()
    ids, attn, _ = _batch()
    with torch.no_grad():
        ref = reference_logprobs(model, ids, attn)
        got = sequence_logprobs(model, ids, attn, chunk_size=chunk)
    assert got.shape == ref.shape and torch.all(got[:, 0] == 0)
    real = attn.bool()
    assert torch.allclose(got[real], ref[real], atol=1e-5)


def test_logit_mask_computes_only_those_positions():
    model = _tiny()
    ids, attn, mask = _batch()
    with torch.no_grad():
        ref = reference_logprobs(model, ids, attn)
        got = sequence_logprobs(model, ids, attn, logit_mask=mask, chunk_size=5)
    m = mask.bool().clone()
    m[:, 0] = False
    assert torch.allclose(got[m], ref[m], atol=1e-5)
    assert torch.all(got[~m] == 0)


def test_gradients_match_full_logits_through_lora_and_checkpointing():
    """The loss gradient on the LoRA params is the same as with full logits,
    with per-chunk checkpointing and layer gradient checkpointing on."""
    ids, attn, mask = _batch(seed=1)
    adv = torch.tensor([1.0, -0.5, 0.3])
    grads = []
    for mode in ("ref", "chunked"):
        policy = apply_lora(_tiny(), LORA, grad_checkpoint=(mode == "chunked"))
        _perturb(policy)
        policy.train()
        if mode == "ref":
            lp = reference_logprobs(policy, ids, attn)
        else:
            lp = sequence_logprobs(policy, ids, attn, logit_mask=mask, chunk_size=6)
        loss, _ = grpo_loss(lp, lp.detach(), lp.detach() - 0.1, adv, mask,
                            clip_eps=0.2, kl_coef=0.04)
        loss.backward()
        grads.append({n: p.grad.clone() for n, p in policy.named_parameters()
                      if p.requires_grad})
    assert grads[0].keys() == grads[1].keys() and grads[0]
    for n in grads[0]:
        assert torch.allclose(grads[0][n], grads[1][n], atol=1e-5), n


def test_reference_policy_via_disable_adapter_still_works_chunked():
    base = _tiny()
    ids, attn, mask = _batch(seed=2)
    with torch.no_grad():
        base_lp = reference_logprobs(base, ids, attn)
    policy = apply_lora(base, LORA)
    _perturb(policy)
    policy.eval()
    ref = batched_logprobs(policy, ids, attn, micro_batch=2, disable_adapter=True,
                           logit_mask=mask, chunk_size=4)
    pol = batched_logprobs(policy, ids, attn, micro_batch=2, logit_mask=mask, chunk_size=4)
    m = mask.bool().clone()
    m[:, 0] = False
    assert torch.allclose(ref[m], base_lp[m], atol=1e-5)
    assert not torch.allclose(pol[m], base_lp[m], atol=1e-3)
