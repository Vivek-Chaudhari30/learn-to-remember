#!/usr/bin/env bash
# M3 on a plain CUDA GPU server (same steps as notebooks/kaggle_m3.ipynb, one GPU).
# Run inside tmux/screen from the repo root, with the project venv active.
#
#   bash scripts/run_m3.sh probe                    # tests, log-prob memory, model probe
#   MODEL=<choice> bash scripts/run_m3.sh baselines # timing, then company A and B baselines
#   MODEL=<choice> bash scripts/run_m3.sh drift     # optional: base model on notes v0-v2 (RQ3)
#   bash scripts/run_m3.sh pack                     # zip M3 results into m3_results.zip
#
# Choose MODEL from the probe verdict (spec §3.5): if Qwen2.5-1.5B-Instruct passes
# (or both pass), keep it. If neither passes, stop and bring back results/probe.
# DTYPE defaults to bf16 (Ampere or newer, e.g. A6000). Every later run (M4, M5)
# must use the same GPU type and dtype, so baselines and trained models compare
# like for like.
set -euo pipefail
# shellcheck disable=SC2086  # $EVAL_EXTRA is split on purpose
STAGE=${1:?which stage: probe | baselines | drift | pack}
DTYPE=${DTYPE:-bf16}
GEN_BATCH=${GEN_BATCH:-16}
EVAL_EXTRA=${EVAL_EXTRA:-}   # extra run_eval args (run_all.sh SMOKE=1 rehearsal only)
mkdir -p logs

case "$STAGE" in
  probe)
    python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; \
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
    python -m pytest -q 2>&1 | tail -3
    python -u scripts/logprob_memory.py --model Qwen/Qwen2.5-1.5B-Instruct --dtype "$DTYPE" \
      --lengths 2048,4096 --batch 2 --out results/memory 2>&1 | tee logs/m3_memory.log
    python -u scripts/probe_models.py --dtype "$DTYPE" --gen-batch "$GEN_BATCH" \
      --models Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen3-1.7B --out results/probe \
      2>&1 | tee logs/m3_probe.log
    echo "Probe done. Read the verdicts above, then run: MODEL=<choice> bash scripts/run_m3.sh baselines"
    ;;
  baselines)
    MODEL=${MODEL:?set MODEL to the probe choice}
    SLUG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')
    # 32-question timing first: seconds per question for the budget.
    python -u scripts/run_eval.py --model "$MODEL" --dtype "$DTYPE" --gen-batch "$GEN_BATCH" \
      --companies company_a --conditions v3 --limit 32 --n-boot 200 \
      --out "results/timing/$SLUG" $EVAL_EXTRA 2>&1 | tee logs/m3_timing.log
    for company in company_a company_b; do
      python -u scripts/run_eval.py --model "$MODEL" --dtype "$DTYPE" --gen-batch "$GEN_BATCH" \
        --companies "$company" --conditions v3,no_notes \
        --out "results/baselines/$SLUG/$company" $EVAL_EXTRA 2>&1 | tee "logs/m3_baseline_$company.log"
    done
    ;;
  drift)
    MODEL=${MODEL:?set MODEL to the probe choice}
    SLUG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')
    python -u scripts/run_eval.py --model "$MODEL" --dtype "$DTYPE" --gen-batch "$GEN_BATCH" \
      --companies company_a --conditions v0,v1,v2 \
      --out "results/baselines/$SLUG/company_a_drift" $EVAL_EXTRA 2>&1 | tee logs/m3_drift.log
    ;;
  pack)
    python scripts/pack_results.py --out m3_results.zip
    ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
