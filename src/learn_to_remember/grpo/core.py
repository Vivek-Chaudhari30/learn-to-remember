# Origin: ported from GRPO-Zero `src/grpo.py`
#   (https://github.com/Vivek-Chaudhari30/GRPO-Zero, main @ ce51ed9).
#   Copied: compute_advantages, grpo_loss, per_token_logprobs, batched_logprobs,
#   make_completion_mask.
#   Changed: log-probs are computed over the WHOLE sequence (`sequence_logprobs`)
#   and the loss takes a generic `loss_mask`, so a multi-turn trajectory
#   [prompt | gen | tool | gen | ...] can be trained on as a single row. The old
#   prompt/completion API is kept as the special case. Position ids are passed
#   explicitly from the attention mask. Masked-out tokens are zeroed with
#   torch.where (inf-safe). A `dr_grpo` normalization option was added.
"""GRPO update math: group advantages -> per-token log-probs -> clipped loss + KL.

Recap (unchanged from GRPO-Zero):
  - For each question, sample a group of G trajectories and score each one.
  - Group-relative advantage A_i = (r_i - mean_g(r)) / (std_g(r) + eps). The
    group mean is the baseline, so there is no critic network.
  - PPO-style clipped policy loss with A_i, plus a k3 KL penalty to a frozen
    reference (the LoRA policy with its adapter disabled).

What is new for multi-turn tool use: a trajectory interleaves tokens the model
generated with tokens it was *given* (the prompt, tool results). Only generated
tokens are actions, so only they may carry policy-gradient or KL terms. That is
what `loss_mask` (1 = model-generated) encodes. It need not be one contiguous
block.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F

LOSS_NORMS = ("seq_mean", "dr_grpo")


# --------------------------------------------------------------------------- #
# Masks
# --------------------------------------------------------------------------- #
def make_completion_mask(completion_ids: torch.Tensor, eos_id: int) -> torch.Tensor:
    """Mask of real completion tokens: 1 up to and including the first EOS in
    each row, 0 afterwards. Rows with no EOS (hit max_new_tokens) are all 1.

    In the multi-turn rollout this trims each generated turn at its end-of-turn
    token, so padding and post-EOS junk never enter the trajectory.
    """
    is_eos = completion_ids == eos_id
    _, C = completion_ids.shape
    has_eos = is_eos.any(dim=1)
    # argmax returns the FIRST max (first True); 0 when a row has no EOS, so guard it.
    first_eos = torch.argmax(is_eos.to(torch.int), dim=1)
    end = torch.where(has_eos, first_eos, torch.full_like(first_eos, C - 1))
    idx = torch.arange(C, device=completion_ids.device).unsqueeze(0)  # [1, C]
    return (idx <= end.unsqueeze(1)).to(torch.long)


def position_ids_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Position ids that count only real tokens: 0, 1, 2, ... starting at the
    first non-pad token of each row. Pad slots get 0 (they are masked anyway).

    Why: HF `generate` derives positions this way, while a plain forward defaults
    to 0..L-1, so for a left-padded row the two disagree by a constant shift.
    For RoPE models (Qwen) a constant shift is harmless, because attention scores
    depend only on relative positions (checked: the padding test in
    tests/test_model_utils.py passes with or without this). Passing positions
    explicitly keeps the log-probs padding-independent for any position scheme,
    at no cost.
    """
    pos = attention_mask.long().cumsum(dim=-1) - 1
    return pos.clamp(min=0)


# --------------------------------------------------------------------------- #
# Advantages
# --------------------------------------------------------------------------- #
def compute_advantages(rewards: torch.Tensor, group_ids: torch.Tensor,
                       eps: float = 1e-4) -> torch.Tensor:
    """Group-relative advantage A_i = (r_i - mean_g(r)) / (std_g(r) + eps).

    The group mean is GRPO's baseline (no critic). Dividing by the group std
    whitens the signal across questions of differing difficulty. When a whole
    group has identical reward (std=0) the advantage collapses to ~0. There is
    nothing to learn from that question this step, which is the correct behavior.
    """
    advantages = torch.empty_like(rewards)
    for g in torch.unique(group_ids):
        m = group_ids == g
        rg = rewards[m]
        advantages[m] = (rg - rg.mean()) / (rg.std(unbiased=False) + eps)
    return advantages


# --------------------------------------------------------------------------- #
# Log-probs
# --------------------------------------------------------------------------- #
def sequence_logprobs(model, input_ids: torch.Tensor,
                      attention_mask: torch.Tensor) -> torch.Tensor:
    """Teacher-forced log-prob of every token in each row. Returns [B, L].

    out[b, t] = log pi(input_ids[b, t] | input_ids[b, :t]) for t >= 1, and
    out[b, 0] = 0 (the first token has nothing predicting it). Positions are
    aligned with `input_ids`, so a `loss_mask` of the same [B, L] shape selects
    the tokens to train on directly.

    Uses `selected - logsumexp` instead of a full log_softmax, so no second
    [B, L, V] tensor is materialized. (The logits themselves still are; at long
    contexts this becomes a chunked computation in M3, see DECISIONS.md.)
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask,
                   position_ids=position_ids_from_mask(attention_mask)).logits
    logits = logits[:, :-1, :].float()              # position t predicts token t+1
    targets = input_ids[:, 1:]                      # [B, L-1]
    selected = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    token_logprobs = selected - torch.logsumexp(logits, dim=-1)
    return F.pad(token_logprobs, (1, 0), value=0.0)  # [B, L], aligned to input_ids


def per_token_logprobs(model, prompt_ids, prompt_mask, completion_ids, completion_mask):
    """GRPO-Zero's single-turn API: log-probs of the *completion* tokens only.

    Returns [B, C]. This is the special case of `sequence_logprobs` where the
    loss mask is exactly the completion block.
    """
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attn = torch.cat([prompt_mask, completion_mask], dim=1)
    return sequence_logprobs(model, input_ids, attn)[:, prompt_ids.shape[1]:]


@torch.no_grad()
def batched_logprobs(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, *,
                     micro_batch: int, disable_adapter: bool = False) -> torch.Tensor:
    """`sequence_logprobs` over many rows, chunked to cap memory. Returns [B, L].

    `disable_adapter=True` turns the LoRA policy back into the frozen base model,
    giving the reference log-probs without holding a second copy in memory.
    """
    ctx = model.disable_adapter() if disable_adapter else nullcontext()
    chunks = []
    with ctx:
        for s in range(0, input_ids.shape[0], micro_batch):
            sl = slice(s, s + micro_batch)
            chunks.append(sequence_logprobs(model, input_ids[sl], attention_mask[sl]))
    return torch.cat(chunks, dim=0)


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def grpo_loss(new_logprobs, old_logprobs, ref_logprobs, advantages, loss_mask, *,
              clip_eps: float, kl_coef: float, norm: str = "seq_mean",
              max_tokens: int | None = None):
    """GRPO objective for one (micro-)batch. Returns (scalar loss, metrics).

    All log-prob tensors and `loss_mask` share one shape, [B, T]. `loss_mask` is
    1 on tokens the model generated and 0 on prompt, tool-result and padding
    tokens. It may be any pattern, not just a contiguous block.

    Per token:
        ratio  = exp(logpi_new - logpi_old)
        L_pol  = -min(ratio * A, clip(ratio, 1 +/- eps) * A)          # PPO clip
        KL     = exp(logpi_ref - logpi_new) - (logpi_ref - logpi_new) - 1   # k3, unbiased, >= 0
        L      = L_pol + kl_coef * KL
    Every masked-in token of trajectory i gets the same advantage A_i.

    Normalization (`norm`):
      - "seq_mean" (default, DeepSeekMath): mean over each row's masked tokens,
        then mean over rows.
      - "dr_grpo" (Dr. GRPO ablation): sum of masked tokens / (B * max_tokens),
        with max_tokens a constant (defaults to T). This removes the length bias
        that per-row averaging introduces.
    """
    if norm not in LOSS_NORMS:
        raise ValueError(f"norm must be one of {LOSS_NORMS}, got {norm!r}")
    mask = loss_mask.float()
    adv = advantages.unsqueeze(1)                        # [B, 1] -> broadcast over tokens

    ratio = torch.exp(new_logprobs - old_logprobs)
    policy_per_token = -torch.min(
        ratio * adv,
        torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv,
    )

    diff = ref_logprobs - new_logprobs
    kl_per_token = torch.exp(diff) - diff - 1.0          # k3 estimator

    # torch.where, not `* mask`: an inf on a masked-out token (e.g. exp of a huge
    # log-ratio on a tool token) would otherwise become inf * 0 = nan.
    keep = mask > 0
    per_token = torch.where(keep, policy_per_token + kl_coef * kl_per_token, 0.0)
    policy_per_token = torch.where(keep, policy_per_token, 0.0)
    kl_per_token = torch.where(keep, kl_per_token, 0.0)
    if norm == "seq_mean":
        seq_lens = mask.sum(dim=1).clamp(min=1.0)
        loss = (per_token.sum(dim=1) / seq_lens).mean()
    else:
        denom = mask.shape[0] * (max_tokens if max_tokens is not None else mask.shape[1])
        loss = per_token.sum() / denom

    with torch.no_grad():
        tok = mask.sum().clamp(min=1.0)
        metrics = {
            "loss": loss.item(),
            "policy_loss": ((policy_per_token * mask).sum() / tok).item(),
            "kl": ((kl_per_token * mask).sum() / tok).item(),
            # fraction of masked tokens where the ratio left the trust region
            "clip_frac": ((((ratio - 1.0).abs() > clip_eps).float() * mask).sum()
                          / tok).item(),
            "n_loss_tokens": int(mask.sum().item()),
        }
    return loss, metrics
