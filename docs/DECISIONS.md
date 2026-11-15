# Decision log

Dated log of non-obvious design decisions, with the reason for each. Deviations from `docs/PROJECT.md` are marked **[deviation]**.

## System prompt

Not written yet (M2/M3). The exact text will be recorded here once it is fixed, and it does not change after that.

---

## 2026-11-15: GRPO-Zero's "no gain" premise was out of date

PROJECT.md §1.2 cited the 2026-10-15 Kaggle T4 run (70/200 steps, lr 1e-6, 0.4397 → 0.4329). GitHub's GRPO-Zero `main` already held a later, complete run: 2026-10-29, 1× RTX A6000, 200 steps, lr 1e-5, 0.4276 → 0.4716 (+4.4 pts, McNemar p ≈ 0.0005), ~28 s/step for 64 completions of ≤512 tokens with the 0.5B model. The local checkout was 11 commits behind `origin/main`.
- Action (approved): fast-forwarded GRPO-Zero's local `main`, added only `results/kaggle-t4-2026-10-15/` as an archive (commit `ce51ed9`, README unchanged), and pushed it. That is the only change made to GRPO-Zero.
- Consequence: `configs/base.yaml` starts at **lr 1e-5**, not GRPO-Zero's config default of 1e-6, which showed no gain.

## 2026-11-15: v3 notes are imperfect **[deviation, approved]**

Spec §3.1 defines v3 as coverage 1.0 and a fresh snapshot. That version has no `gap` and no `stale` questions, and the main run (experiment 2) trains on v3, so it could never learn to fall back to raw files, which is half of RQ5. v3 becomes roughly coverage 0.9, a snapshot slightly before the final change events, and some truncation, while staying the best of the four versions. Exact numbers are fixed and logged at M1.

## 2026-11-15: Generic `loss_mask` via full-sequence log-probs

`grpo/core.py` adds `sequence_logprobs(model, input_ids, attention_mask)`, which returns `[B, L]` with position `t` holding `log π(x_t | x_<t)` (position 0 = 0, since it has no prediction). A multi-turn trajectory `[prompt | gen | tool | gen | …]` is then one row, and `loss_mask` (1 = model-generated token) picks the positions to train on. GRPO-Zero's `per_token_logprobs(prompt, completion)` becomes a thin wrapper, which is the special case where the mask covers the completion block. `grpo_loss` already worked on any mask of the log-prob shape. A test proves the two paths agree.

## 2026-11-15: Spec budget inconsistencies to resolve at M2 (open)

- K=6 × 400 tool tokens + 6 × 256 generated tokens + a system prompt with tool schemas (~700 tokens, to be measured) ≈ 4.6k tokens, over the 4096 context cap.
- With at most 2400 tool tokens readable, B = 4000 never binds, so `min(1, ·)` is a no-op and the most a run can lose to cost is λ·0.6.
- The spec doesn't say whether the `answer` call counts toward K.
- These will be resolved with measured token counts at M2 and logged here.

## 2026-11-15: Logit memory at long context (open, M3)

`sequence_logprobs` materializes `[B, L, V]` logits as GRPO-Zero did. At L = 4096 with Qwen's V ≈ 152k that is ~2.5 GB per row in fp32, which is too much for a T4. At M3 this becomes a chunked computation over hidden states (lm_head plus logsumexp in slices, only at masked positions). The M0 API is kept so that change stays internal.

## 2026-11-15: Explicit position ids; inf-safe masking

- `sequence_logprobs` passes `position_ids` computed from the attention mask, the same way `generate` does. I checked whether this fixes a real GRPO-Zero bug, and it doesn't: Qwen uses RoPE, which is invariant to a constant position shift, and the padding-invariance test passes with or without it. It's kept because it costs nothing and holds for any position scheme.
- `grpo_loss` zeroes masked-out tokens with `torch.where` instead of `* mask`, so an inf on a tool or prompt token can't turn into `inf * 0 = nan`. A test injects inf there.

## 2026-11-15: Python environment

Project venv from Homebrew `python3.12` (`/opt/homebrew/bin/python3.12`). The base anaconda is broken (numpy 2.x / pandas ABI mismatch) and is never used. Installed at M0: torch 2.15.0, transformers 5.23.0, peft 0.21.3, numpy 2.5.5. `model_utils` handles the transformers `torch_dtype`→`dtype` rename and the v5 `activation_checkpointing_enable` rename, so older Colab/Kaggle images still work.
