#!/usr/bin/env bash
# The whole GPU run (M3 -> M4 -> M5 -> report) on one CUDA GPU, in order, resumable.
# Run inside tmux from anywhere; see docs/RUNBOOK.md.
#
#   bash scripts/run_all.sh probe              # step 1: tests, memory check, model probe. Then STOP:
#                                              #   read the verdicts and choose MODEL (spec §3.5).
#   MODEL=<choice> bash scripts/run_all.sh     # step 2: everything else, in order
#   bash scripts/run_all.sh status             # which stages are done
#
# Every finished stage writes results/.stages/<stage>.done and is skipped on a re-run.
# A training stage cut off part-way resumes from its last checkpoint (every 10 steps),
# so after any interruption just run the same command again.
#
# Gates (the run stops instead of wasting GPU hours):
#   - after the overfit check, the mean fraction correct over its last 10 steps must be
#     >= OVERFIT_MIN (default 0.8). GATE=0 skips the gate (only after reading the log).
#   - after the M4 timing stage it prints the measured s/step and the training hours
#     that implies; PAUSE_AFTER_TIMING=1 stops there so you can decide.
#
# DTYPE defaults to bf16 (Ampere or newer). Do not mix GPUs or dtypes across stages.
# SMOKE=1: plumbing rehearsal only (tiny steps / question limits, any device, no probe,
# no gate). Its numbers mean nothing and must never be reported.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
[ -f .venv/bin/activate ] && source .venv/bin/activate

MODE=${1:-all}
export DTYPE=${DTYPE:-bf16}
STAGES=results/.stages
mkdir -p "$STAGES" logs

if [ "${SMOKE:-0}" = "1" ]; then
  export TRAIN_EXTRA="--steps 2 --prompts-per-step 2 --group-size 2 --max-gen-tokens 48 --gen-batch 4"
  export EVAL_EXTRA="--limit 3 --n-boot 50"
  GATE=0
  echo "SMOKE=1: plumbing rehearsal, numbers are meaningless"
fi

stage() {   # stage <name> <command...>: run once, mark done
  local name=$1; shift
  if [ -f "$STAGES/$name.done" ]; then
    echo "[skip] $name (done $(cat "$STAGES/$name.done"))"
    return 0
  fi
  echo "[run ] $name  $(date '+%F %T')"
  "$@"
  date '+%F %T' > "$STAGES/$name.done"
  echo "[done] $name  $(date '+%F %T')"
}

overfit_gate() {
  [ "${GATE:-1}" = "0" ] && { echo "overfit gate skipped (GATE=0)"; return 0; }
  python - "${OVERFIT_MIN:-0.8}" <<'EOF'
import json, sys
thr = float(sys.argv[1])
h = json.load(open("results/train/overfit/train_log.json"))
last = h[-10:]
acc = sum(x["frac_correct"] for x in last) / len(last)
rew = sum(x["reward_mean"] for x in last) / len(last)
print(f"overfit check: last {len(last)} steps, fraction correct {acc:.3f}, reward {rew:.3f} "
      f"(gate: fraction correct >= {thr})")
if acc < thr:
    print("FAILED: the loop did not fit 8 questions. Do not start the main run; bring back "
          "results/train/overfit and logs/m4_overfit.log. (GATE=0 overrides.)")
    sys.exit(1)
EOF
}

timing_report() {
  python - <<'EOF'
import json, os
t = json.load(open("results/train/timing/timing.json"))
s = t["mean_sec_per_step"]
steps = 60 + 200 * 5          # overfit + main + exp3 + exp4 (x2) + exp5
print(f"measured: {s:.1f} s/step over {t['steps_measured']} steps (rollout {t['mean_sec_rollout']:.1f}"
      f" s, ref+update {t['mean_sec_update']:.1f} s), peak GPU {t.get('peak_gpu_gb')} GB, "
      f"{t['gpu']} {t['dtype']}")
print(f"training still to run: {steps} steps -> {steps * s / 3600:.1f} h at this rate "
      f"(evals not included; first steps include warm-up)")
EOF
}

case "$MODE" in
  status)
    ls -1 "$STAGES" 2>/dev/null | sed 's/\.done$//' || true
    exit 0 ;;
  probe)
    [ "${SMOKE:-0}" = "1" ] && { echo "the probe loads 1.5B+ models; not part of SMOKE"; exit 1; }
    stage m3_probe bash scripts/run_m3.sh probe
    echo
    echo "Next: read the verdicts above (results/probe/probe_summary.json), choose the model, then"
    echo "  MODEL=<choice> bash scripts/run_all.sh"
    exit 0 ;;
  all) ;;
  *) echo "usage: run_all.sh [probe|all|status]"; exit 1 ;;
esac

export MODEL=${MODEL:?set MODEL to the probe choice, e.g. MODEL=Qwen/Qwen2.5-1.5B-Instruct}
SLUG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')
if [ "${SMOKE:-0}" != "1" ] && [ ! -f "$STAGES/m3_probe.done" ]; then
  echo "run the probe first: bash scripts/run_all.sh probe"; exit 1
fi
echo "MODEL=$MODEL DTYPE=$DTYPE slug=$SLUG"

# M3: baselines (base model), then the drift baseline for RQ3
stage m3_baselines bash scripts/run_m3.sh baselines
stage m3_drift     bash scripts/run_m3.sh drift
# M4: timing, overfit sanity check (gated), main run (experiment 2), its eval
stage m4_timing    bash scripts/run_m4.sh timing
[ "${SMOKE:-0}" = "1" ] || timing_report
if [ "${PAUSE_AFTER_TIMING:-0}" = "1" ] && [ ! -f "$STAGES/m4_overfit.done" ]; then
  echo "PAUSE_AFTER_TIMING=1: stopping. Re-run without it to continue."; exit 0
fi
stage m4_overfit   bash scripts/run_m4.sh overfit
# (explicit if: `set -e` does not fire inside `a || { b && c; }`, so a failed gate must exit here)
if [ ! -f "$STAGES/m4_overfit_gate.done" ]; then
  if ! overfit_gate; then exit 1; fi
  date '+%F %T' > "$STAGES/m4_overfit_gate.done"
fi
stage m4_main      bash scripts/run_m4.sh main
stage m4_eval      bash scripts/run_m4.sh eval
# M5: experiments 3-6 (exp6 is eval only, on the main-run adapter)
stage m5_exp3      bash scripts/run_m5.sh exp3
stage m5_exp4_l0   bash scripts/run_m5.sh exp4_l0
stage m5_exp4_l0.5 bash scripts/run_m5.sh exp4_l0.5
stage m5_exp5      bash scripts/run_m5.sh exp5
stage m5_exp6      bash scripts/run_m5.sh exp6
# Report, figures, and the zip to bring back (re-made on every run, cheap)
python scripts/make_report.py --slug "$SLUG" > /dev/null
python scripts/make_figures.py --slug "$SLUG"
python scripts/pack_results.py
echo "All stages done. Bring back gpu_results.zip (results/report.md has the tables)."
