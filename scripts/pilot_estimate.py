"""After `PILOT=1 bash scripts/run_all.sh`: check the pilot and estimate the full run.

    python scripts/pilot_estimate.py            # reads results/ in this clone

The pilot runs every stage with the real model and settings but few steps and few
questions, so its per-step and per-question times are real. This scales them up to the
full run's step and question counts. It also flags non-finite losses and prints peak GPU
memory. Pilot accuracies mean nothing and are never reported.
"""

from __future__ import annotations

import glob
import json
import math
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Full-run work, from scripts/run_all.sh / run_m3.sh / run_m4.sh / run_m5.sh.
TRAIN_STEPS = 3 + 60 + 200 * 5      # timing + overfit + main, exp3, exp4 (x2), exp5
N_A, N_B = 293, 300                 # data/eval/company_{a,b}_eval.jsonl
A_EVALS = 2 + 3 + 1 + 1 + 2 + 4 + 3  # baselines, drift, m4 eval, exp3, exp4, exp5, exp6
B_EVALS = 2 + 1 + 1                  # baselines, exp5, exp6
EVAL_QUESTIONS = 32 + A_EVALS * N_A + B_EVALS * N_B   # + the 32-question timing eval


def estimate_hours(sec_per_step: float, sec_per_question: float) -> dict:
    train_h = TRAIN_STEPS * sec_per_step / 3600
    eval_h = EVAL_QUESTIONS * sec_per_question / 3600
    return {"train_h": train_h, "eval_h": eval_h, "total_h": train_h + eval_h}


def step_times(log: list[dict]) -> list[float]:
    """Seconds per step, without the first step of a run (model warm-up)."""
    secs = [h["sec_step"] for h in log]
    return secs[1:] if len(secs) > 1 else secs


def main() -> int:
    res = os.path.join(ROOT, "results")
    problems: list[str] = []

    secs: list[float] = []
    for path in sorted(glob.glob(os.path.join(res, "train", "*", "train_log.json"))):
        log = json.load(open(path))
        run = os.path.basename(os.path.dirname(path))
        bad = [h.get("step") for h in log if h.get("nonfinite_loss")
               or not all(math.isfinite(v) for v in h.values() if isinstance(v, float))]
        if bad:
            problems.append(f"{run}: non-finite values at steps {bad}")
        secs += step_times(log)
        print(f"train {run:14s} {len(log):3d} steps, median {statistics.median(step_times(log)):.1f} s/step")

    spq: list[float] = []
    for path in sorted(glob.glob(os.path.join(res, "**", "timing.json"), recursive=True)):
        if os.sep + "train" + os.sep in path:
            continue
        for t in json.load(open(path)):
            spq.append(t["seconds_per_question"])

    peak = None
    tpath = os.path.join(res, "train", "timing", "timing.json")
    if os.path.exists(tpath):
        t = json.load(open(tpath))
        peak = t.get("peak_gpu_gb")
        print(f"timing stage: {t['gpu']} {t['dtype']}, peak GPU {peak} GB, micro-batch {t['micro_batch']}")

    if not secs or not spq:
        print("missing training logs or eval timings: did the pilot finish? (bash scripts/run_all.sh status)")
        return 1
    s_step, s_q = statistics.median(secs), statistics.median(spq)
    e = estimate_hours(s_step, s_q)
    print(f"\nmeasured: {s_step:.1f} s/step (median of {len(secs)} steps), "
          f"{s_q:.2f} s/question (median of {len(spq)} evals)")
    print(f"full run: {TRAIN_STEPS} steps -> {e['train_h']:.1f} h training, "
          f"{EVAL_QUESTIONS} questions -> {e['eval_h']:.1f} h evals, "
          f"about {e['total_h']:.0f} h in total ({e['total_h'] / 24:.1f} days)")
    if peak is not None and peak > 40:
        problems.append(f"peak GPU {peak} GB is close to the A6000's 48 GB: little room if someone shares it")
    if problems:
        print("\nPROBLEMS:\n  " + "\n  ".join(problems))
        return 1
    print("\nNo problems found. Pilot accuracies are not results; do not report them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
