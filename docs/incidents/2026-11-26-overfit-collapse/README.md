# 2026-11-26: the overfit check collapsed to "answer without reading"

The M4 overfit check (`configs/overfit.yaml`) failed its gate on umibozu, so `run_all.sh` stopped before the main run. This folder keeps everything from that run: it will be overwritten when the check is re-run.

## What ran

- `MODEL=Qwen/Qwen3-1.7B bash scripts/run_all.sh`, stage `m4_overfit`, started 2026-11-26 12:08:43, done 12:45:01 (60 steps in 2,091 s).
- RTX A6000, bf16, LoRA r16/α32 (17.4M trainable params, fp32). Condition v3, P = 8, G = 8, lr 5e-5, kl_coef 0.04, λ = 0.2, budget 4,000 tokens, max 6 turns. Advantage = (r − mean) / (std + 1e-4), loss norm `seq_mean`.
- The same 8 training questions every step: `company_a-train-0072, 0430, 0773, 0816, 1226, 1305, 1595, 1599` (stratified over qtypes).
- Gate result: `overfit check: last 10 steps, fraction correct 0.000, reward 0.100 (gate: fraction correct >= 0.8)` → `FAILED`.

## What happened

1. **Steps 0–8: reading shrinks.** Tokens read 716 → 321, files read in 77% → 12% of trajectories, turn-limit endings 12 → 0. Accuracy holds at 33–45%.
2. **Steps 9–15: one note lookup, then answer.** Tokens ≈ 177, files never read, accuracy 25% (the gap question and one lookup question stay right). Stale accuracy goes from 0.75 at step 0 to 0, because stale questions need the files. Steps 10–13 have every group at zero variance (all 8 samples identical).
3. **Step 16 onward: no lookups at all.** The model calls `answer` as its first and only turn. Zero-lookup rate 1.00, accuracy 0, reward exactly 0.100 (the format bonus) on every trajectory. Almost every group has zero variance, so there is no learning signal left, and it stays there for the last 44 steps. KL to the reference ≈ 1.2.

Example, question `company_a-train-0072` ("where is the team that runs Project Condor?"):

- step 10–15: `search_notes("Project Condor")` → wrong answer, reward **0.09075**
- step 16: `answer("...East Office.")` with no lookup, still wrong, reward **0.1**
- step 59: `answer("I cannot determine the location of the team that runs Project Condor.")`, reward **0.1**

## Cause

Reward = 1.0 (correct) + 0.1 (valid answer) − 0.2 · min(1, tokens / 4000). On a question the model always gets wrong, the 8 samples differ only in the reading cost: 0.09075 with one lookup vs 0.1 with none, a 0.009 gap. The advantage divides by the group's std, so that 0.009 gap becomes ±1, the same strength as a right-vs-wrong difference. On the 8 overfit questions, multihop and aggregate were never right (0% from step 0), so most of every step's gradient said "read less." The push spilled over to the questions the model was getting right, until at step 16 it stopped reading on those too. Once all 8 samples answer the same way, the std is 0, the advantage is 0, and nothing can pull the model back. kl_coef 0.04 was not enough on its own.

The 5× higher learning rate of the overfit check (5e-5 vs 1e-5) made this happen within 16 steps. The main run has the same reward and advantage, so it would most likely drift the same way, more slowly. Its 3 timing steps (lr 1e-5) did not run long enough to show it.

Also visible: per-token entropy was very low (0.006–0.016) in the first steps even at temperature 1.0, so many groups were already all-identical (zero-var 0.38–0.75) before the collapse.

## Fix

No division by the group std: `adv_std_norm: false` in `configs/base.yaml` (Dr. GRPO), for every run. The advantage is now r − group mean, so on the example above the gap between reading once and not reading stays 0.009, while a right-vs-wrong group gives ±0.5. `tests/test_grpo_update.py::test_advantages_without_std_keep_the_reward_scale` uses this question's exact rewards. The overfit check is re-run with `bash scripts/run_all.sh redo m4_overfit`. See docs/DECISIONS.md (2026-11-26).

## Files

| file | what |
|---|---|
| `train_log.json` | one entry per step, every metric `train.py` logs |
| `run.json` | full config, environment, the 8 question ids |
| `timing.json` | s/step (mean 34.8), peak GPU 16.37 GB |
| `train_samples.jsonl` | best and worst trajectory per step (94 rows) |
| `m4_overfit.log` | the stage's console output |

Also kept, not in git:

- umibozu: `~/learn-to-remember/failed_runs/2026-11-26_overfit_collapse/` (these files plus the collapsed adapter checkpoint, 267 MB) and `~/learn-to-remember/snapshot_overfit_failed.zip`.
- Mac: `gpu_snapshots/snapshot_overfit_failed.zip` (sha1 `1966693e…`, every result so far plus all rollouts) and `gpu_snapshots/overfit_failed/`.

## Every step

| step | reward | correct | tokens read | lookups | read files | zero-lookup | turn limit | zero-var groups | KL | entropy | s/step |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.405 | 0.359 | 716 | 3.58 | 0.77 | 0.00 | 12 | 0.38 | 0.0000 | 0.015 | 106.8 |
| 1 | 0.457 | 0.391 | 540 | 2.86 | 0.48 | 0.00 | 4 | 0.00 | 0.0086 | 0.015 | 79.9 |
| 2 | 0.522 | 0.453 | 561 | 2.94 | 0.53 | 0.00 | 2 | 0.50 | 0.0070 | 0.013 | 83.7 |
| 3 | 0.433 | 0.359 | 504 | 2.66 | 0.38 | 0.00 | 1 | 0.75 | 0.0353 | 0.007 | 69.4 |
| 4 | 0.466 | 0.391 | 494 | 2.59 | 0.36 | 0.00 | 0 | 0.38 | 0.0414 | 0.006 | 69.2 |
| 5 | 0.454 | 0.375 | 420 | 2.19 | 0.16 | 0.00 | 0 | 0.62 | 0.1516 | 0.007 | 58.1 |
| 6 | 0.455 | 0.375 | 401 | 2.09 | 0.12 | 0.00 | 0 | 0.62 | 0.1738 | 0.007 | 57.9 |
| 7 | 0.410 | 0.328 | 366 | 1.95 | 0.08 | 0.00 | 0 | 0.75 | 0.2429 | 0.006 | 53.3 |
| 8 | 0.459 | 0.375 | 321 | 1.67 | 0.12 | 0.00 | 0 | 0.75 | 0.3351 | 0.008 | 47.4 |
| 9 | 0.339 | 0.250 | 217 | 1.20 | 0.00 | 0.00 | 0 | 0.62 | 0.5167 | 0.011 | 43.3 |
| 10 | 0.341 | 0.250 | 177 | 1.00 | 0.00 | 0.00 | 0 | 1.00 | 0.6093 | 0.016 | 37.6 |
| 11 | 0.341 | 0.250 | 177 | 1.00 | 0.00 | 0.00 | 0 | 1.00 | 0.6071 | 0.015 | 38.1 |
| 12 | 0.341 | 0.250 | 177 | 1.00 | 0.00 | 0.00 | 0 | 1.00 | 0.6069 | 0.014 | 37.0 |
| 13 | 0.341 | 0.250 | 177 | 1.00 | 0.00 | 0.00 | 0 | 1.00 | 0.5990 | 0.015 | 37.7 |
| 14 | 0.341 | 0.250 | 174 | 0.98 | 0.00 | 0.02 | 0 | 0.75 | 0.6004 | 0.019 | 39.0 |
| 15 | 0.345 | 0.250 | 107 | 0.61 | 0.00 | 0.39 | 0 | 0.62 | 0.7575 | 0.041 | 35.1 |
| 16 | 0.115 | 0.016 | 3 | 0.02 | 0.00 | 0.98 | 0 | 0.88 | 1.1825 | 0.093 | 27.6 |
| 17 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 1.00 | 1.1545 | 0.094 | 26.7 |
| 18 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 1.00 | 1.1508 | 0.109 | 26.6 |
| 19 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 1.00 | 1.1611 | 0.110 | 26.2 |
| 20 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 1.00 | 1.1480 | 0.111 | 26.2 |
| 21 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 1.00 | 1.1480 | 0.117 | 26.6 |
| 22 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 1.00 | 1.1911 | 0.114 | 26.5 |
| 23 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 1.00 | 1.1540 | 0.122 | 27.2 |
| 24 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 1.00 | 1.1771 | 0.115 | 26.8 |
| 25 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 1.00 | 1.2146 | 0.118 | 26.8 |
| 26 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 1.00 | 1.1830 | 0.118 | 26.8 |
| 27 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 1.00 | 1.1901 | 0.112 | 26.4 |
| 28 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 1.00 | 1.1885 | 0.115 | 26.6 |
| 29 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 1.00 | 1.2060 | 0.112 | 26.2 |
| 30 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 1.00 | 1.2062 | 0.111 | 26.6 |
| 31 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 1.00 | 1.2114 | 0.109 | 26.6 |
| 32 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 1.00 | 1.1955 | 0.103 | 26.2 |
| 33 | 0.116 | 0.016 | 0 | 0.00 | 0.00 | 1.00 | 0 | 0.88 | 1.2275 | 0.105 | 25.9 |
| 34 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 0.88 | 1.3925 | 0.105 | 27.6 |
| 35 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 0.75 | 1.1820 | 0.119 | 28.3 |
| 36 | 0.100 | 0.000 | 2 | 0.00 | 0.00 | 1.00 | 0 | 0.50 | 1.1976 | 0.120 | 28.8 |
| 37 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 0.88 | 1.3809 | 0.125 | 27.6 |
| 38 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 0.88 | 1.1309 | 0.132 | 27.7 |
| 39 | 0.100 | 0.000 | 1 | 0.00 | 0.00 | 1.00 | 0 | 0.75 | 1.1844 | 0.134 | 28.8 |
| 40 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 0.88 | 1.2149 | 0.130 | 28.4 |
| 41 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 1.00 | 1.1775 | 0.133 | 27.4 |
| 42 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 1.00 | 1.1904 | 0.136 | 26.0 |
| 43 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 0.88 | 1.2040 | 0.131 | 28.0 |
| 44 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 1.00 | 1.1838 | 0.129 | 26.8 |
| 45 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 1.00 | 1.1954 | 0.125 | 26.5 |
| 46 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 1.00 | 1.2077 | 0.126 | 26.6 |
| 47 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 1.00 | 1.2129 | 0.125 | 26.5 |
| 48 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 0.88 | 1.3530 | 0.122 | 27.3 |
| 49 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 1.00 | 1.2631 | 0.125 | 26.3 |
| 50 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 1.00 | 1.2721 | 0.129 | 26.8 |
| 51 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 0.88 | 1.2650 | 0.133 | 27.8 |
| 52 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 0.88 | 1.3124 | 0.135 | 28.9 |
| 53 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 0.88 | 1.2556 | 0.143 | 29.0 |
| 54 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 0.75 | 1.2733 | 0.138 | 28.3 |
| 55 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 0.88 | 1.2697 | 0.134 | 27.3 |
| 56 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 0.88 | 1.2619 | 0.125 | 27.4 |
| 57 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 0.88 | 1.2311 | 0.125 | 27.9 |
| 58 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 1.00 | 1.1877 | 0.121 | 26.3 |
| 59 | 0.100 | 0.000 | 0 | 0.00 | 0.00 | 1.00 | 0 | 1.00 | 1.1934 | 0.122 | 27.2 |
