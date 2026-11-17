"""The hand-written multi-turn GRPO step and training loop (spec §3.4, M4).

One step:
  1. Questions: P per step from the training split (stratified by qtype by
     default, see QuestionSampler), G sampled trajectories each.
  2. Rollout with the current policy (temperature 1, nothing else on).
  3. Reward per trajectory -> group-relative advantage (compute_advantages).
     Optionally drop groups whose rewards are all equal (DAPO-style).
  4. Reference log-probs: the same model with the LoRA adapter disabled, no
     grad, only at loss-mask positions.
  5. Update: per micro-batch, new log-probs (+ entropy for monitoring), then
     grpo_loss on masked tokens only; gradients accumulate exactly (each
     micro-batch weighted by its share of rows); then clip and one AdamW step.

Old log-probs. With ppo_epochs = 1 the update runs once on the batch it just
sampled, so pi_old IS the current policy, and old = new.detach() exactly
(ratio 1, gradient = A * grad log pi). This skips a forward pass and avoids
the noise of computing old in eval mode and new in train mode, where LoRA
dropout differs. With ppo_epochs > 1, old is computed once in a separate
no-grad pass before the epochs.

TRL is not used here; it may only cross-check the math (ground rule).
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from contextlib import nullcontext

import torch

from ..env.episode import behavior
from ..workspace.formats import sub_rng
from ..workspace.questions import QTYPES
from .core import compute_advantages, grpo_loss, sequence_logprobs
from .rollout import collate, rollout


# --------------------------------------------------------------------------- #
# Question sampling
# --------------------------------------------------------------------------- #
class QuestionSampler:
    """Stateless, seeded question schedule, so resuming at step s gives exactly
    the questions an uninterrupted run would have seen.

    mix = "stratified": every step draws as evenly as possible from each qtype
      (under the training notes version), each qtype walking through its own
      reshuffled-per-epoch order. This answers the M1 open item: the natural
      training mix is ~71% lookup and has only ~37 stale facts.
    mix = "natural": one reshuffled-per-epoch walk through the whole pool.
    overfit_n > 0: the same overfit_n questions every step (sanity check).
    """

    def __init__(self, questions: list[dict], *, per_step: int, version: str = "v3",
                 mix: str = "stratified", seed: int = 0, overfit_n: int = 0):
        if mix not in ("stratified", "natural"):
            raise ValueError(mix)
        self.per_step, self.mix, self.seed, self.version = per_step, mix, seed, version
        qs = sorted(questions, key=lambda q: q["id"])
        if overfit_n:
            qs = self._stratified_pick(qs, overfit_n)
            self.mix, self.per_step = "fixed", overfit_n
        self.questions = qs
        self.by_qtype = {t: [q for q in qs if q["qtype_by_version"][version] == t]
                         for t in QTYPES}
        self.by_qtype = {t: v for t, v in self.by_qtype.items() if v}

    def _stratified_pick(self, qs, n):
        rng = sub_rng(self.seed, "overfit")
        buckets = {t: [q for q in qs if q["qtype_by_version"][self.version] == t] for t in QTYPES}
        out, t_i = [], 0
        order = [t for t in QTYPES if buckets[t]]
        for t in order:
            rng.shuffle(buckets[t])
        while len(out) < n:
            t = order[t_i % len(order)]
            if buckets[t]:
                out.append(buckets[t].pop())
            t_i += 1
        return sorted(out, key=lambda q: q["id"])

    def _walk(self, pool, key, k):
        """The k-th item of an endless walk through `pool`, reshuffled each epoch."""
        epoch, pos = divmod(k, len(pool))
        order = list(range(len(pool)))
        sub_rng(self.seed, "sampler", key, epoch).shuffle(order)
        return pool[order[pos]]

    def for_step(self, step: int) -> list[dict]:
        if self.mix == "fixed":
            return list(self.questions)
        if self.mix == "natural":
            return [self._walk(self.questions, "all", step * self.per_step + i)
                    for i in range(self.per_step)]
        types = list(self.by_qtype)
        n = len(types)
        base, extra = divmod(self.per_step, n)
        out = []
        for j, t in enumerate(types):
            # qtype j gets an extra slot on steps where it is among the `extra`
            # rotated types; count how many it got before this step
            rot = lambda s, j=j: base + (1 if (j - s * extra) % n < extra else 0)  # noqa: E731
            before = sum(rot(s) for s in range(step)) if extra else base * step
            out += [self._walk(self.by_qtype[t], t, before + i) for i in range(rot(step))]
        return out


# --------------------------------------------------------------------------- #
# One GRPO step
# --------------------------------------------------------------------------- #
def _micro_batches(rows: list[int], trajs, micro: int):
    """Group row indices into micro-batches of similar length (less padding)."""
    order = sorted(rows, key=lambda i: len(trajs[i].ids))
    return [order[s:s + micro] for s in range(0, len(order), micro)]


def train_step(policy, tok, fmt, generate, make_episodes, questions, optimizer, tcfg: dict,
               *, group_size: int, scaler=None, device=None) -> tuple[dict, list]:
    """One GRPO step. `make_episodes(questions) -> [Episode]` (one per question).
    Returns (metrics, trajectories)."""
    device = device or next(policy.parameters()).device
    micro = tcfg["micro_batch"]
    t0 = time.perf_counter()

    # 1-2) rollout: G episodes per question, all batched together
    group_qs = [q for q in questions for _ in range(group_size)]
    episodes = make_episodes(group_qs)
    trajs = rollout(generate, fmt, episodes)
    t_roll = time.perf_counter() - t0

    # 3) rewards -> advantages
    rewards = torch.tensor([t.result.reward for t in trajs], dtype=torch.float32)
    gids = torch.tensor([i // group_size for i in range(len(trajs))])
    adv = compute_advantages(rewards, gids, eps=tcfg["adv_eps"])
    grp_std = torch.stack([rewards[gids == g].std(unbiased=False) for g in range(len(questions))])
    zero_var = (grp_std < 1e-8)
    keep = list(range(len(trajs)))
    if tcfg.get("skip_zero_variance_groups"):
        keep = [i for i in keep if not zero_var[i // group_size]]
    B = len(keep)

    metrics = {"n_trajectories": len(trajs), "n_trained": B,
               "frac_zero_var_groups": zero_var.float().mean().item(),
               "skipped_groups": int(zero_var.sum().item()) if tcfg.get(
                   "skip_zero_variance_groups") else 0}
    agg = {"loss": 0.0, "policy_loss": 0.0, "kl": 0.0, "clip_frac": 0.0, "entropy": 0.0}
    grad_norm = float("nan")
    t1 = time.perf_counter()
    t_ref = t_upd = 0.0
    if B:
        mbs = _micro_batches(keep, trajs, micro)
        pad = tok.pad_token_id if tok.pad_token_id is not None else fmt.eot_id
        batches = [collate([trajs[i] for i in mb], pad, device) for mb in mbs]

        # 4) reference log-probs (adapter off), and old log-probs if ppo_epochs > 1
        policy.eval()
        refs, olds = [], []
        with torch.no_grad():
            ctx = policy.disable_adapter() if hasattr(policy, "disable_adapter") else nullcontext()
            with ctx:
                for b in batches:
                    refs.append(sequence_logprobs(policy, b["input_ids"], b["attention_mask"],
                                                  logit_mask=b["loss_mask"]))
            if tcfg["ppo_epochs"] > 1:
                for b in batches:
                    olds.append(sequence_logprobs(policy, b["input_ids"], b["attention_mask"],
                                                  logit_mask=b["loss_mask"]))
        t_ref = time.perf_counter() - t1
        policy.train()

        # 5) update
        tok_total = sum(float(b["loss_mask"].sum()) for b in batches)
        for _ in range(tcfg["ppo_epochs"]):
            optimizer.zero_grad(set_to_none=True)
            agg = {k: 0.0 for k in agg}
            for j, (mb, b) in enumerate(zip(mbs, batches)):
                new, ent = sequence_logprobs(policy, b["input_ids"], b["attention_mask"],
                                             logit_mask=b["loss_mask"], return_entropy=True)
                old = olds[j] if olds else new.detach()
                loss, m = grpo_loss(new, old, refs[j], adv[mb].to(device), b["loss_mask"],
                                    clip_eps=tcfg["clip_eps"], kl_coef=tcfg["kl_coef"],
                                    norm=tcfg.get("loss_norm", "seq_mean"),
                                    max_tokens=tcfg.get("dr_grpo_max_tokens"))
                w = len(mb) / B                        # exact accumulation over rows
                (scaler.scale(loss * w) if scaler else loss * w).backward()
                share = float(b["loss_mask"].sum()) / max(tok_total, 1.0)
                agg["loss"] += m["loss"] * w
                for k in ("policy_loss", "kl", "clip_frac"):
                    agg[k] += m[k] * share
                agg["entropy"] += float((ent * b["loss_mask"]).sum()) / max(tok_total, 1.0)
            params = [p for p in policy.parameters() if p.requires_grad]
            if scaler:
                scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, tcfg["grad_clip"]))
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        t_upd = time.perf_counter() - t1 - t_ref

    beh = [behavior(t.result) for t in trajs]
    qt = Counter(t.result.qtype for t in trajs)
    acc_qt = {k: sum(t.result.correct for t in trajs if t.result.qtype == k) / v
              for k, v in qt.items()}
    gen_lens = [g for t in trajs for g in t.gen_lens]
    metrics.update(agg)
    metrics.update({
        "reward_mean": rewards.mean().item(), "reward_std": rewards.std(unbiased=False).item(),
        "frac_correct": sum(t.result.correct for t in trajs) / len(trajs),
        "frac_valid_answer": sum(t.result.valid_answer for t in trajs) / len(trajs),
        "acc_by_qtype": acc_qt,
        "tokens_read_mean": sum(t.result.tokens_read for t in trajs) / len(trajs),
        "turns_mean": sum(b["n_turns"] for b in beh) / len(beh),
        "lookups_mean": sum(b["n_lookups"] for b in beh) / len(beh),
        "zero_lookup_rate": sum(b["zero_lookup_answer"] for b in beh) / len(beh),
        "repeated_calls_per_traj": sum(b["repeated_calls"] for b in beh) / len(beh),
        "frac_traj_with_repeats": sum(b["repeated_calls"] > 0 for b in beh) / len(beh),
        "valid_call_rate": sum(b["valid_calls"] for b in beh) / max(1, sum(
            b["n_calls_total"] for b in beh)),
        "used_files_rate": sum(b["used_files"] for b in beh) / len(beh),
        "used_notes_rate": sum(b["used_notes"] for b in beh) / len(beh),
        "ended": dict(Counter(t.result.ended for t in trajs)),
        "gen_tokens_per_turn": sum(gen_lens) / max(1, len(gen_lens)),
        "turns_hit_gen_cap": sum(c for t in trajs for c in t.hit_gen_cap) / max(1, len(gen_lens)),
        "max_traj_tokens": max(len(t.ids) for t in trajs),
        "n_loss_tokens": int(sum(sum(trajs[i].mask) for i in keep)),
        "grad_norm": grad_norm,
        "scaler_scale": float(scaler.get_scale()) if scaler else None,
        "sec_rollout": t_roll, "sec_ref": t_ref, "sec_update": t_upd,
        "sec_step": time.perf_counter() - t0,
    })
    if not math.isfinite(metrics["loss"]):
        metrics["nonfinite_loss"] = True
    return metrics, trajs


# --------------------------------------------------------------------------- #
# Loop
# --------------------------------------------------------------------------- #
def make_scaler(model):
    """GradScaler when the base runs in fp16 (T4): LoRA params are fp32 (PEFT
    autocasts adapters), but gradients flow through fp16 activations and can
    underflow without loss scaling. None for bf16/fp32."""
    dtype = next(p for p in model.parameters() if not p.requires_grad).dtype
    dev = next(model.parameters()).device
    if dtype != torch.float16 or dev.type != "cuda":
        return None
    return torch.amp.GradScaler("cuda")


def train_loop(*, policy, tok, fmt, generate, make_episodes, sampler: QuestionSampler,
               optimizer, tcfg: dict, group_size: int, steps: int, save_dir: str,
               log_path: str, start_step: int = 0, history: list | None = None,
               sample_dir: str | None = None, log=print, scaler=None):
    from ..model_utils import save_checkpoint

    history = list(history or [])
    t_start = time.perf_counter()
    for step in range(start_step, steps):
        qs = sampler.for_step(step)
        m, trajs = train_step(policy, tok, fmt, generate, make_episodes, qs, optimizer, tcfg,
                              group_size=group_size, scaler=scaler)
        m["step"] = step
        m["question_ids"] = [q["id"] for q in qs]
        history.append(m)
        log(f"step {step:4d} | reward {m['reward_mean']:+.3f} acc {m['frac_correct']:.3f} "
            f"| valid {m['valid_call_rate']:.2f} | KL {m['kl']:.4f} | ent {m['entropy']:.3f} "
            f"| clip {m['clip_frac']:.3f} | zero-var {m['frac_zero_var_groups']:.2f} "
            f"| tok {m['tokens_read_mean']:.0f} | rep {m['frac_traj_with_repeats']:.2f} "
            f"| |g| {m['grad_norm']:.3f} | {m['sec_step']:.1f}s "
            f"(roll {m['sec_rollout']:.0f} ref {m['sec_ref']:.0f} upd {m['sec_update']:.0f})")
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(history, f, indent=1)
        if sample_dir:
            os.makedirs(sample_dir, exist_ok=True)
            best = max(range(len(trajs)), key=lambda i: trajs[i].result.reward)
            worst = min(range(len(trajs)), key=lambda i: trajs[i].result.reward)
            with open(os.path.join(sample_dir, "train_samples.jsonl"), "a") as f:
                for i in sorted({best, worst}):
                    r = trajs[i].result
                    f.write(json.dumps({"step": step, "id": r.question_id, "reward": r.reward,
                                        "correct": r.correct, "ended": r.ended,
                                        "turns": [t["assistant"] for t in r.turns]}) + "\n")
        every = tcfg.get("checkpoint_every") or 0
        if every and (step + 1) % every == 0 and step + 1 < steps:
            save_checkpoint(save_dir, policy, optimizer, step, history)
            log(f"  checkpoint -> {save_dir} (step {step})")
    if steps > start_step:
        save_checkpoint(save_dir, policy, optimizer, steps - 1, history)
    log(f"done: steps {start_step}..{steps - 1} in {time.perf_counter() - t_start:.0f}s")
    return history
