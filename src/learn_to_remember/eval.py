"""Eval harness: conditions x note versions x workspaces, behavior metrics, CIs.

    rows, trajs = run_condition(gen, fmt, ws, questions, "v3", cfg, counter)
    summary = summarize(rows)

A condition is a notes version ("v0".."v3") or "no_notes". qtype labels follow
the condition's version; "no_notes" uses the v3 labels, as the oracle does.

Metrics (spec §4), overall and per qtype, each with a percentile bootstrap 95%
CI over questions (seeded, so reruns reproduce the CI exactly):
  accuracy                 mean of correct
  mean_tokens_read         tool-output tokens returned to the model per question
  lookups_per_q            successful search/read calls per question
  median_tokens_per_lookup median over all lookups (pooled; resampled by question)
  fallback_rate            gap/stale questions only: share that used a file tool
                           (search_files or read_file)
  false_alarm_rate         lookup questions only (notes correct and fresh), notes
                           conditions only: share that used a file tool
  valid_call_rate          calls that parsed and named a real tool with valid
                           arguments / all model turns (pooled ratio)
  zero_lookup_rate         answered without any successful lookup
  mean_turns               model turns used
  mean_reward, repeated_calls_per_q
Plus, without CIs: how episodes ended, generated tokens per turn, the share of
turns cut at the per-turn generation cap, and seconds per question.
"""

from __future__ import annotations

import random
from statistics import median

import numpy as np

from .env.episode import Episode, EnvConfig, behavior
from .env.tools import FILE_TOOLS
from .workspace.questions import QTYPES

CONDITIONS = ("v0", "v1", "v2", "v3", "no_notes")
METRICS = ("accuracy", "mean_tokens_read", "lookups_per_q", "median_tokens_per_lookup",
           "fallback_rate", "false_alarm_rate", "valid_call_rate", "zero_lookup_rate",
           "mean_turns", "mean_reward", "repeated_calls_per_q")

# Model-selection rule (spec §3.5), applied to the probe (50 questions, notes v3).
PROBE_MIN_VALID_CALLS = 0.80
PROBE_ACC_RANGE = (0.20, 0.60)


def label_version(condition: str) -> str:
    return "v3" if condition == "no_notes" else condition


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #
def make_episodes(ws, questions, condition: str, cfg: EnvConfig, counter):
    from .oracle import make_env
    from .reward import build_catalog

    version = None if condition == "no_notes" else condition
    env = make_env(ws, version, counter, cfg)
    catalog = build_catalog(ws.world)
    return [Episode(env, q, catalog, cfg, label_version(condition)) for q in questions]


def run_condition(generate, fmt, ws, questions, condition: str, cfg: EnvConfig, counter, *,
                  episodes_per_rollout: int = 32, progress=None):
    """Run every question once. Returns (rows, trajectories)."""
    from .grpo.rollout import rollout

    rows, trajs = [], []
    for s in range(0, len(questions), episodes_per_rollout):
        chunk = questions[s:s + episodes_per_rollout]
        eps = make_episodes(ws, chunk, condition, cfg, counter)
        out = rollout(generate, fmt, eps)
        for q, t in zip(chunk, out):
            rows.append(episode_row(t, q, condition, seconds=t.seconds / len(chunk)))
            trajs.append(t)
        if progress:
            progress(len(rows), len(questions), rows)
    return rows, trajs


def episode_row(traj, question: dict, condition: str, seconds: float | None = None) -> dict:
    """Everything the metrics need, per question (small enough to store)."""
    res = traj.result
    b = behavior(res)
    return {
        "id": question["id"], "qtype": res.qtype, "condition": condition,
        "heldout": bool(question.get("heldout")),
        "correct": bool(res.correct), "valid_answer": bool(res.valid_answer),
        "answer": res.answer, "gold": question["gold"], "reward": res.reward,
        "ended": res.ended, "tokens_read": res.tokens_read,
        "n_turns": b["n_turns"], "n_lookups": b["n_lookups"],
        "tokens_per_lookup": b["tokens_per_lookup"],
        "used_files": b["used_files"], "used_notes": b["used_notes"],
        "valid_calls": b["valid_calls"], "n_calls_total": b["n_calls_total"],
        "repeated_calls": b["repeated_calls"], "zero_lookup_answer": b["zero_lookup_answer"],
        "tools": [t["tool"] or (t["parse_error"] or "?") for t in res.turns],
        "gen_lens": list(traj.gen_lens), "hit_gen_cap": list(traj.hit_gen_cap),
        "traj_tokens": len(traj.ids), "seconds": seconds,
    }


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _stat(name: str, rows: list[dict], condition: str):
    """One metric on a list of rows; None if undefined for these rows."""
    if not rows:
        return None
    if name == "accuracy":
        return float(np.mean([r["correct"] for r in rows]))
    if name == "mean_tokens_read":
        return float(np.mean([r["tokens_read"] for r in rows]))
    if name == "lookups_per_q":
        return float(np.mean([r["n_lookups"] for r in rows]))
    if name == "median_tokens_per_lookup":
        pooled = [x for r in rows for x in r["tokens_per_lookup"]]
        return float(median(pooled)) if pooled else None
    if name == "fallback_rate":
        sel = [r for r in rows if r["qtype"] in ("gap", "stale")]
        return float(np.mean([r["used_files"] for r in sel])) if sel else None
    if name == "false_alarm_rate":
        if condition == "no_notes":
            return None
        sel = [r for r in rows if r["qtype"] == "lookup"]
        return float(np.mean([r["used_files"] for r in sel])) if sel else None
    if name == "valid_call_rate":
        tot = sum(r["n_calls_total"] for r in rows)
        return sum(r["valid_calls"] for r in rows) / tot if tot else None
    if name == "zero_lookup_rate":
        return float(np.mean([r["zero_lookup_answer"] for r in rows]))
    if name == "mean_turns":
        return float(np.mean([r["n_turns"] for r in rows]))
    if name == "mean_reward":
        return float(np.mean([r["reward"] for r in rows]))
    if name == "repeated_calls_per_q":
        return float(np.mean([r["repeated_calls"] for r in rows]))
    raise KeyError(name)


def bootstrap_ci(rows: list[dict], name: str, condition: str, *, n_boot: int = 2000,
                 seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap over questions. None if the metric is undefined."""
    if _stat(name, rows, condition) is None:
        return None
    rng = np.random.default_rng(seed)
    n = len(rows)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = _stat(name, [rows[i] for i in idx], condition)
        if v is not None:
            vals.append(v)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return [float(lo), float(hi)]


def _block(rows, condition, n_boot, seed) -> dict:
    out = {"n": len(rows)}
    for m in METRICS:
        v = _stat(m, rows, condition)
        out[m] = None if v is None else {
            "value": v, "ci95": bootstrap_ci(rows, m, condition, n_boot=n_boot, seed=seed)}
    turns = [g for r in rows for g in r["gen_lens"]]
    caps = [c for r in rows for c in r["hit_gen_cap"]]
    ended: dict[str, int] = {}
    for r in rows:
        ended[r["ended"]] = ended.get(r["ended"], 0) + 1
    out["ended"] = dict(sorted(ended.items()))
    out["gen_tokens_per_turn"] = ({"mean": float(np.mean(turns)), "median": float(median(turns)),
                                   "max": int(max(turns))} if turns else None)
    out["turns_hit_gen_cap"] = float(np.mean(caps)) if caps else None
    secs = [r["seconds"] for r in rows if r.get("seconds") is not None]
    out["seconds_per_question"] = float(np.mean(secs)) if secs else None
    out["max_traj_tokens"] = max(r["traj_tokens"] for r in rows) if rows else None
    return out


def summarize(rows: list[dict], *, n_boot: int = 2000, seed: int = 0) -> dict:
    conds = {r["condition"] for r in rows}
    assert len(conds) == 1, f"summarize one condition at a time, got {conds}"
    cond = conds.pop()
    return {"condition": cond,
            "overall": _block(rows, cond, n_boot, seed),
            "by_qtype": {t: _block([r for r in rows if r["qtype"] == t], cond, n_boot, seed)
                         for t in QTYPES if any(r["qtype"] == t for r in rows)}}


def probe_verdict(summary: dict) -> dict:
    """The §3.5 rule: valid tool calls >= 80% and accuracy in [20%, 60%]."""
    o = summary["overall"]
    valid = o["valid_call_rate"]["value"] if o["valid_call_rate"] else 0.0
    acc = o["accuracy"]["value"]
    ok_valid = valid >= PROBE_MIN_VALID_CALLS
    ok_acc = PROBE_ACC_RANGE[0] <= acc <= PROBE_ACC_RANGE[1]
    return {"valid_call_rate": valid, "accuracy": acc, "valid_ok": ok_valid,
            "accuracy_ok": ok_acc, "passes": ok_valid and ok_acc}


# --------------------------------------------------------------------------- #
# Question subsets and trajectory export
# --------------------------------------------------------------------------- #
def probe_question_ids(questions: list[dict], per_qtype: int = 10, seed: int = 0) -> list[str]:
    """Fixed, stratified probe subset: `per_qtype` questions per v3 qtype."""
    rng = random.Random(seed)
    out = []
    for t in QTYPES:
        ids = sorted(q["id"] for q in questions if q["qtype_by_version"]["v3"] == t)
        out += sorted(rng.sample(ids, min(per_qtype, len(ids))))
    return out


def trajectory_record(traj, question: dict, tok=None) -> dict:
    """A readable trajectory: question, gold, each turn's text and observation."""
    res = traj.result
    rec = {"id": question["id"], "question": question["question"], "gold": question["gold"],
           "qtype": res.qtype, "answer": res.answer, "correct": res.correct,
           "reward": res.reward, "ended": res.ended, "tokens_read": res.tokens_read,
           "turns": [{"assistant": t["assistant"], "tool": t["tool"], "args": t["args"],
                      "error": t["parse_error"] or t["tool_error"],
                      "observation": t["observation"], "tokens": t["tokens"]}
                     for t in res.turns],
           "gen_lens": traj.gen_lens}
    return rec


def sample_trajectories(rows, trajs, questions, per_qtype: int = 2, seed: int = 0) -> list:
    """A few trajectories per qtype (one correct and one wrong when possible)."""
    rng = random.Random(seed)
    qby = {q["id"]: q for q in questions}
    out = []
    for t in QTYPES:
        idx = [i for i, r in enumerate(rows) if r["qtype"] == t]
        right = [i for i in idx if rows[i]["correct"]]
        wrong = [i for i in idx if not rows[i]["correct"]]
        pick = ([rng.choice(right)] if right else []) + ([rng.choice(wrong)] if wrong else [])
        pick += [i for i in idx if i not in pick][:max(0, per_qtype - len(pick))]
        out += [trajectory_record(trajs[i], qby[rows[i]["id"]]) for i in pick[:per_qtype]]
    return out


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def _fmt(m, pct=False, nd=1):
    if m is None:
        return "-"
    v, ci = m["value"], m["ci95"]
    if pct:
        return f"{100 * v:.{nd}f} [{100 * ci[0]:.{nd}f}, {100 * ci[1]:.{nd}f}]"
    return f"{v:.{nd}f} [{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def summary_table(summaries: list[tuple[str, dict]]) -> str:
    """Markdown: one row per (label, summary), overall metrics with CIs."""
    head = ("| run | n | accuracy % | tokens read | lookups/q | median tok/lookup | "
            "fallback % (gap+stale) | false alarm % (lookup) | valid calls % | "
            "zero-lookup % | turns |")
    lines = [head, "|" + "---|" * 11]
    for label, s in summaries:
        o = s["overall"]
        lines.append(" | ".join([
            f"| {label}", str(o["n"]), _fmt(o["accuracy"], True), _fmt(o["mean_tokens_read"], nd=0),
            _fmt(o["lookups_per_q"], nd=2), _fmt(o["median_tokens_per_lookup"], nd=0),
            _fmt(o["fallback_rate"], True), _fmt(o["false_alarm_rate"], True),
            _fmt(o["valid_call_rate"], True), _fmt(o["zero_lookup_rate"], True),
            _fmt(o["mean_turns"], nd=2)]) + " |")
    return "\n".join(lines)


def qtype_table(summaries: list[tuple[str, dict]]) -> str:
    lines = ["| run | " + " | ".join(QTYPES) + " |", "|" + "---|" * (len(QTYPES) + 1)]
    for label, s in summaries:
        cells = []
        for t in QTYPES:
            b = s["by_qtype"].get(t)
            cells.append(f"{_fmt(b['accuracy'], True, 0)} (n={b['n']})" if b else "-")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def ops_line(s: dict) -> str:
    o = s["overall"]
    g = o["gen_tokens_per_turn"] or {}
    return (f"ended {o['ended']}; gen tokens/turn mean {g.get('mean', 0):.1f}, "
            f"max {g.get('max', 0)}; turns cut at gen cap {100 * (o['turns_hit_gen_cap'] or 0):.1f}%; "
            f"max trajectory {o['max_traj_tokens']} tokens; "
            f"{o['seconds_per_question'] or 0:.2f} s/question")


__all__ = ["CONDITIONS", "METRICS", "run_condition", "episode_row", "summarize",
           "bootstrap_ci", "probe_verdict", "probe_question_ids", "summary_table",
           "qtype_table", "sample_trajectories", "trajectory_record", "ops_line",
           "FILE_TOOLS"]
