#!/usr/bin/env bash
# M4 on a rented CUDA GPU (A100/L4: bf16). Usage:
#   MODEL=Qwen/Qwen2.5-1.5B-Instruct DTYPE=bf16 bash scripts/run_m4.sh
# Steps: measure s/step, overfit sanity check, main run (resumable), eval of the adapter.
set -euo pipefail
MODEL=${MODEL:?set MODEL to the M3 choice}
DTYPE=${DTYPE:-bf16}
MICRO=${MICRO:-2}
CKPT=${CKPT:-checkpoints}
python -u scripts/train.py --config configs/train_notes.yaml --model "$MODEL" --dtype "$DTYPE" \
  --micro-batch "$MICRO" --steps 3 --timing --out results/train/timing
python -u scripts/train.py --config configs/overfit.yaml --model "$MODEL" --dtype "$DTYPE" \
  --micro-batch "$MICRO" --out results/train/overfit --save-dir "$CKPT/overfit"
python -u scripts/train.py --config configs/train_notes.yaml --model "$MODEL" --dtype "$DTYPE" \
  --micro-batch "$MICRO" --out results/train/notes_l0.2 --save-dir "$CKPT/notes_l0.2" --resume
python -u scripts/run_eval.py --model "$MODEL" --adapter "$CKPT/notes_l0.2" --dtype "$DTYPE" \
  --companies company_a --conditions v3 --out results/trained/notes_l0.2
