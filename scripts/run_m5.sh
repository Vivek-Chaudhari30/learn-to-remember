#!/usr/bin/env bash
# M5: experiments 3-6 (spec §4), one at a time. Each training run is resumable
# (re-run the same command after a disconnect). Usage:
#   MODEL=Qwen/Qwen2.5-1.5B-Instruct bash scripts/run_m5.sh exp3
#   ... exp4_l0 | exp4_l0.5 | exp5 | exp6 | drift_base
# exp6 needs the M4 main-run adapter (checkpoints/notes_l0.2).
# Results land in results/trained/<run>/ and results/baselines/<slug>/company_a_drift/;
# then `python scripts/make_report.py` builds the tables.
# DTYPE defaults to bf16 (Ampere+); use fp16 on a T4. TRAIN_EXTRA / EVAL_EXTRA
# append arguments (run_all.sh SMOKE=1 plumbing rehearsal only).
set -euo pipefail
# shellcheck disable=SC2086  # $TRAIN_EXTRA / $EVAL_EXTRA are split on purpose
EXP=${1:?which experiment: exp3 | exp4_l0 | exp4_l0.5 | exp5 | exp6 | drift_base}
MODEL=${MODEL:?set MODEL to the M3 choice}
DTYPE=${DTYPE:-bf16}
MICRO=${MICRO:-2}
CKPT=${CKPT:-checkpoints}
TRAIN_EXTRA=${TRAIN_EXTRA:-}
EVAL_EXTRA=${EVAL_EXTRA:-}
SLUG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')
mkdir -p logs

train() {   # train <config> <run>
  python -u scripts/train.py --config "configs/$1.yaml" --model "$MODEL" --dtype "$DTYPE" \
    --micro-batch "$MICRO" --out "results/train/$2" --save-dir "$CKPT/$2" --resume $TRAIN_EXTRA \
    2>&1 | tee -a "logs/m5_train_$2.log"
}
evaluate() {   # evaluate <run> <companies> <conditions>
  [ -f "$CKPT/$1/adapter_model.safetensors" ] || { echo "missing adapter $CKPT/$1"; exit 1; }
  python -u scripts/run_eval.py --model "$MODEL" --adapter "$CKPT/$1" --dtype "$DTYPE" \
    --companies "$2" --conditions "$3" --out "results/trained/$1" $EVAL_EXTRA \
    2>&1 | tee -a "logs/m5_eval_$1.log"
}

case "$EXP" in
  exp3)       train train_nonotes nonotes_l0.2 && evaluate nonotes_l0.2 company_a no_notes ;;
  exp4_l0)    train train_l0 notes_l0 && evaluate notes_l0 company_a v3 ;;
  exp4_l0.5)  train train_l0.5 notes_l0.5 && evaluate notes_l0.5 company_a v3 ;;
  exp5)       train train_v0 v0_l0.2 && evaluate v0_l0.2 company_a v0,v1,v2,v3 \
                && evaluate v0_l0.2 company_b v3 ;;
  exp6)       # eval only: the main run (exp 2) on company B, and on the older note versions (RQ3)
              evaluate notes_l0.2 company_b v3 && evaluate notes_l0.2 company_a v0,v1,v2 ;;
  drift_base) python -u scripts/run_eval.py --model "$MODEL" --dtype "$DTYPE" --companies company_a \
                --conditions v0,v1,v2 --out "results/baselines/$SLUG/company_a_drift" $EVAL_EXTRA \
                2>&1 | tee -a logs/m5_drift_base.log ;;
  *) echo "unknown experiment $EXP"; exit 1 ;;
esac
