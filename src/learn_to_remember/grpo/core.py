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
#   `compute_advantages(scale_by_std=False)` drops the std division (Dr. GRPO);
#   the default since the 2026-11-26 overfit collapse (docs/incidents/).
"""GRPO update math: group advantages -> per-token log-probs -> clipped loss + KL.

Recap (unchanged from GRPO-Zero):
  - For each question, sample a group of G trajectories and score each one.
  - Group-relative advantage A_i = (r_i - mean_g(r)) / (std_g(r) + eps). The
    group mean is the baseline, so there is no critic network. This project
    trains with A_i = r_i - mean_g(r) (no std division, `adv_std_norm: false`):
    see compute_advantages for why.
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
                       eps: float = 1e-4, scale_by_std: bool = True) -> torch.Tensor:
    """Group-relative advantage A_i = r_i - mean_g(r), divided by (std_g(r) + eps)
    when `scale_by_std` (the original GRPO).

    The group mean is GRPO's baseline (no critic). When a whole group has
    identical reward (std=0) the advantage is ~0: nothing to learn from that
    question this step.

    Dividing by the group std makes every group's signal the same size, however
    small its reward differences are. With a reading cost in the reward, that is
    harmful: on a question the model always gets wrong, the samples differ only
    by cost (e.g. 0.09075 vs 0.1), and the division turns that 0.009 gap into a
    full ±1 "read less" signal, as strong as right vs wrong. That drove the
    2026-11-26 overfit collapse to answering without reading
    (docs/incidents/2026-11-26-overfit-collapse/). Without the division
    (Dr. GRPO, Liu et al. 2025), advantages keep the reward's own scale.
    """
    advantages = torch.empty_like(rewards)
    for g in torch.unique(group_ids):
        m = group_ids == g
        rg = rewards[m]
        a = rg - rg.mean()
        advantages[m] = a / (rg.std(unbiased=False) + eps) if scale_by_std else a
    return advantages


# --------------------------------------------------------------------------- #
# Log-probs
# --------------------------------------------------------------------------- #
def _backbone_and_head(model):
    """(decoder backbone, lm_head) of an HF causal LM, through a PEFT wrapper.
    None when the model doesn't expose them (e.g. the fake LMs in the tests).

    PEFT injects LoRA into the backbone's Linear layers in place, so calling the
    backbone directly still applies the adapter, and `disable_adapter()` still
    turns it off.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if not (hasattr(base, "get_output_embeddings") and hasattr(base, "get_decoder")):
        return None
    try:
        head, backbone = base.get_output_embeddings(), base.get_decoder()
    except (AttributeError, NotImplementedError):
        return None
    if head is None or backbone is None or backbone is base:
        return None
    return backbone, head


def _chunk_logprobs(head, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """log softmax(head(hidden))[target] for a slice of positions. [N, H] -> [N]."""
    logits = head(hidden).float()
    selected = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return selected - torch.logsumexp(logits, dim=-1)


def _chunk_logprobs_entropy(head, hidden, targets):
    """As _chunk_logprobs, plus the (detached) entropy of each position's
    distribution, for monitoring collapse. Returns [N, 2]: (log-prob, entropy)."""
    logits = head(hidden).float()
    lse = torch.logsumexp(logits, dim=-1)
    lp = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - lse
    with torch.no_grad():
        ent = lse - (torch.softmax(logits, dim=-1) * logits).sum(-1)
    return torch.stack([lp, ent.detach()], dim=-1)


def sequence_logprobs(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, *,
                      logit_mask: torch.Tensor | None = None,
                      chunk_size: int = 1024, return_entropy: bool = False):
    """Teacher-forced log-prob of every token in each row. Returns [B, L].

    out[b, t] = log pi(input_ids[b, t] | input_ids[b, :t]) for t >= 1, and
    out[b, 0] = 0 (the first token has nothing predicting it). Positions are
    aligned with `input_ids`, so a `loss_mask` of the same [B, L] shape selects
    the tokens to train on directly.

    Memory (M3, see DECISIONS.md): the full [B, L, V] logits are never built.
    The backbone runs once for hidden states, then lm_head + logsumexp run over
    slices of `chunk_size` positions. Each slice is checkpointed when grads are
    on, so the backward recomputes it instead of storing its logits. With
    `logit_mask` ([B, L], 1 = needed, e.g. the loss mask), only those positions
    are computed at all and every other entry is 0. In a trajectory most tokens
    are the prompt and tool results, so this skips most of the head's work.
    Models without a separate backbone/head fall back to the plain full-logits
    path, and `logit_mask` then only zeroes entries.

    `return_entropy=True` returns (log-probs, entropies), both [B, L]; the
    entropy is detached (a monitoring signal, not part of the loss).
    """
    pos = position_ids_from_mask(attention_mask)
    parts = _backbone_and_head(model)
    if parts is None:
        logits = model(input_ids=input_ids, attention_mask=attention_mask,
                       position_ids=pos).logits
        logits = logits[:, :-1, :].float()           # position t predicts token t+1
        targets = input_ids[:, 1:]
        selected = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        lse = torch.logsumexp(logits, dim=-1)
        out = F.pad(selected - lse, (1, 0), value=0.0)
        out = out if logit_mask is None else torch.where(logit_mask.bool(), out, 0.0)
        if not return_entropy:
            return out
        with torch.no_grad():
            ent = F.pad(lse - (torch.softmax(logits, -1) * logits).sum(-1), (1, 0), value=0.0)
            ent = ent if logit_mask is None else torch.where(logit_mask.bool(), ent, 0.0)
        return out, ent

    backbone, head = parts
    hidden = backbone(input_ids=input_ids, attention_mask=attention_mask,
                      position_ids=pos).last_hidden_state          # [B, L, H]
    B, L = input_ids.shape
    need = torch.ones(B, L, dtype=torch.bool, device=input_ids.device)
    if logit_mask is not None:
        need = logit_mask.bool()
    need = need.clone()
    need[:, 0] = False                                 # nothing predicts position 0
    b_idx, t_idx = need.nonzero(as_tuple=True)         # positions whose token we score
    h = hidden[b_idx, t_idx - 1]                       # [N, H], hidden at t-1 predicts t
    tgt = input_ids[b_idx, t_idx]                      # [N]
    use_ckpt = torch.is_grad_enabled() and h.requires_grad
    fn = _chunk_logprobs_entropy if return_entropy else _chunk_logprobs
    vals = []
    for s in range(0, h.shape[0], chunk_size):
        hs, ts = h[s:s + chunk_size], tgt[s:s + chunk_size]
        if use_ckpt:
            from torch.utils.checkpoint import checkpoint
            vals.append(checkpoint(fn, head, hs, ts, use_reentrant=False))
        else:
            vals.append(fn(head, hs, ts))
    out = torch.zeros(B, L, dtype=torch.float32, device=input_ids.device)
    if not return_entropy:
        return out.index_put((b_idx, t_idx), torch.cat(vals)) if vals else out
    ent = torch.zeros(B, L, dtype=torch.float32, device=input_ids.device)
    if vals:
        v = torch.cat(vals)
        out = out.index_put((b_idx, t_idx), v[:, 0])
        ent = ent.index_put((b_idx, t_idx), v[:, 1].detach())
    return out, ent


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
                     micro_batch: int, disable_adapter: bool = False,
                     logit_mask: torch.Tensor | None = None,
                     chunk_size: int = 1024) -> torch.Tensor:
    """`sequence_logprobs` over many rows, chunked to cap memory. Returns [B, L].

    `disable_adapter=True` turns the LoRA policy back into the frozen base model,
    giving the reference log-probs without holding a second copy in memory.
    """
    ctx = model.disable_adapter() if disable_adapter else nullcontext()
    chunks = []
    with ctx:
        for s in range(0, input_ids.shape[0], micro_batch):
            sl = slice(s, s + micro_batch)
            chunks.append(sequence_logprobs(
                model, input_ids[sl], attention_mask[sl], chunk_size=chunk_size,
                logit_mask=None if logit_mask is None else logit_mask[sl]))
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
