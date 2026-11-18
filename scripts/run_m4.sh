#!/usr/bin/env bash
# M4 on a CUDA GPU. Usage (default stage: all, in order):
#   MODEL=Qwen/Qwen2.5-1.5B-Instruct bash scripts/run_m4.sh [timing|overfit|main|eval|all]
# timing: 3 steps of the main config, measures s/step (no checkpoint)
# overfit: sanity check, 8 fixed questions, 60 steps (reward should climb to ~1)
# main:   experiment 2 (notes v3, lambda 0.2, 200 steps), resumable
# eval:   the main-run adapter on company A, notes v3
# DTYPE defaults to bf16 (Ampere+); use fp16 on a T4. TRAIN_EXTRA / EVAL_EXTRA
# append arguments (used by run_all.sh SMOKE=1 for a plumbing rehearsal only).
set -euo pipefail
STAGE=${1:-all}
MODEL=${MODEL:?set MODEL to the M3 choice}
DTYPE=${DTYPE:-bf16}
MICRO=${MICRO:-2}
CKPT=${CKPT:-checkpoints}
TRAIN_EXTRA=${TRAIN_EXTRA:-}
EVAL_EXTRA=${EVAL_EXTRA:-}
mkdir -p logs

timing() {
  # shellcheck disable=SC2086
  python -u scripts/train.py --config configs/train_notes.yaml --model "$MODEL" --dtype "$DTYPE" \
    --micro-batch "$MICRO" --steps 3 --timing --out results/train/timing $TRAIN_EXTRA \
    2>&1 | tee logs/m4_timing.log
}
overfit() {
  # shellcheck disable=SC2086
  python -u scripts/train.py --config configs/overfit.yaml --model "$MODEL" --dtype "$DTYPE" \
    --micro-batch "$MICRO" --out results/train/overfit --save-dir "$CKPT/overfit" $TRAIN_EXTRA \
    2>&1 | tee logs/m4_overfit.log
}
main_run() {
  # shellcheck disable=SC2086
  python -u scripts/train.py --config configs/train_notes.yaml --model "$MODEL" --dtype "$DTYPE" \
    --micro-batch "$MICRO" --out results/train/notes_l0.2 --save-dir "$CKPT/notes_l0.2" --resume \
    $TRAIN_EXTRA 2>&1 | tee -a logs/m4_main.log
}
eval_main() {
  # shellcheck disable=SC2086
  python -u scripts/run_eval.py --model "$MODEL" --adapter "$CKPT/notes_l0.2" --dtype "$DTYPE" \
    --companies company_a --conditions v3 --out results/trained/notes_l0.2 $EVAL_EXTRA \
    2>&1 | tee logs/m4_eval.log
}

case "$STAGE" in
  timing)  timing ;;
  overfit) overfit ;;
  main)    main_run ;;
  eval)    eval_main ;;
  all)     timing && overfit && main_run && eval_main ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
