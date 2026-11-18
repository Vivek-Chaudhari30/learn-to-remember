# GPU runbook: the whole experiment on one CUDA GPU (osaka, RTX A6000)

Everything from the model probe (M3) to the last experiment (M5) runs with two
commands. Each stage is skipped once it is done, and training resumes from its
last checkpoint, so after any interruption you re-run the same command.

All stages must run on the **same GPU type and dtype** (bf16 on the A6000).
Baselines and trained models are compared question by question, so do not mix in
runs from Kaggle or another card.

## 0. Requirements

- Linux, NVIDIA GPU with >= 16 GB (A6000: 48 GB). bf16 needs compute capability >= 8.0.
- Driver new enough for the torch build (osaka: driver 555, CUDA 12.5 -> torch cu124 wheels).
- Python >= 3.10 with `venv`, and pip install rights (no root needed).
- Network: github.com and huggingface.co (~7 GB of models, public, no token).
- Disk: ~15–20 GB free (models ~7 GB, torch ~5 GB, checkpoints ~0.2 GB per run, results small).
- A way to keep jobs alive for many hours: `tmux` (or `screen`).

## 1. Setup (once)

```bash
git clone https://github.com/Vivek-Chaudhari30/learn-to-remember.git
```

```bash
cd learn-to-remember && bash scripts/setup_server.sh
```

The setup script:

- creates `.venv`;
- installs the CUDA 12.4 build of torch and the project;
- checks that torch sees the GPU and whether bf16 is native;
- downloads the two candidate models;
- runs the test suite (all tests should pass).

Overrides: `PY=python3.12`, `TORCH_INDEX=<wheel index url>`, `HF_HOME=/path/with/space`.

## 2. Model probe (M3, ~minutes)

```bash
tmux new -s ltr
```

```bash
bash scripts/run_all.sh probe
```

This runs the tests, a log-prob memory check at 2k/4k context, and the probe. The probe is
Qwen2.5-1.5B-Instruct and Qwen3-1.7B on 50 questions. Read the two verdicts it prints
(`results/probe/probe_summary.json`). The rule (spec §3.5) is valid tool calls >= 80% and
accuracy in [20%, 60%]:

| Probe outcome | MODEL |
|---|---|
| Qwen2.5-1.5B passes (alone or with Qwen3) | `Qwen/Qwen2.5-1.5B-Instruct` |
| only Qwen3-1.7B passes | `Qwen/Qwen3-1.7B` |
| neither passes | **stop**: bring back `results/probe` and `logs/`; this needs a decision (e.g. an SFT warm start) |

## 3. Everything else

```bash
MODEL=Qwen/Qwen2.5-1.5B-Instruct bash scripts/run_all.sh
```

Stages, in order:

| Stage | What | Output |
|---|---|---|
| `m3_baselines` | 32-question timing, then base model on company A and B, notes v3 and no notes | `results/timing/`, `results/baselines/<slug>/company_{a,b}/` |
| `m3_drift` | base model on notes v0–v2 (RQ3 reference) | `results/baselines/<slug>/company_a_drift/` |
| `m4_timing` | 3 training steps; prints measured s/step and the training hours it implies | `results/train/timing/timing.json` |
| `m4_overfit` | sanity check: 8 fixed questions, 60 steps; **gate** (below) | `results/train/overfit/` |
| `m4_main` | experiment 2: notes v3, λ = 0.2, 200 steps | `results/train/notes_l0.2/`, `checkpoints/notes_l0.2/` |
| `m4_eval` | main-run adapter on company A, notes v3 | `results/trained/notes_l0.2/` |
| `m5_exp3` | training without notes (RQ1) + eval | `results/{train,trained}/nonotes_l0.2/` |
| `m5_exp4_l0`, `m5_exp4_l0.5` | λ sweep (RQ2) + evals | `results/{train,trained}/notes_l0{,.5}/` |
| `m5_exp5` | trained on v0 notes; eval on v0–v3 and company B (RQ3, RQ4) | `results/{train,trained}/v0_l0.2/` |
| `m5_exp6` | main-run adapter on company B and on v0–v2 (RQ3, RQ4); eval only | `results/trained/notes_l0.2/` |
| (end) | `results/report.md`, `figures/`, `gpu_results.zip` | |

Options:

- `PAUSE_AFTER_TIMING=1` stops right after `m4_timing`, so you can read the measured budget
  first. Re-run without it to continue.
- `MICRO=4`: larger training micro-batch, faster if `m4_timing` shows memory headroom
  (`peak_gpu_gb`). The result is identical: gradient accumulation is exact.
- Keep the defaults for everything else. Changing a hyperparameter part-way makes runs
  incomparable.

### The overfit gate

After `m4_overfit`, the mean fraction correct over its last 10 steps must be >= 0.8
(`OVERFIT_MIN`). If it is lower, the run stops before the main run, because a loop that
cannot fit 8 questions should not spend 200 steps. Bring back `results/train/overfit/` and
`logs/m4_overfit.log`. `GATE=0` overrides the gate; use it only after reading the log.
To redo the overfit check:

```bash
rm results/.stages/m4_overfit.done
```

## 4. While it runs

- Detach from tmux with `Ctrl-b d`. Reattach with:

  ```bash
  tmux attach -t ltr
  ```

- Show which stages are finished:

  ```bash
  bash scripts/run_all.sh status
  ```

- Follow a stage's log (every stage logs to `logs/`):

  ```bash
  tail -f logs/m4_main.log
  ```

- In the training log, watch `acc` and `tok` (tokens read). `ent` falling towards 0, or
  `rep` (repeated calls) rising, is the collapse signal.
- Check the GPU itself with `nvidia-smi`.

After a crash, reboot or disconnect, re-run the same `MODEL=... bash scripts/run_all.sh`.
A stage that was cut off re-runs: training continues from its last checkpoint, and an eval
starts over.

## 5. Bring back

When it finishes, the last step writes `gpu_results.zip`. It contains `results/`,
`figures/` and `logs/`, but no checkpoints and no per-question rollouts. Copy it to the Mac,
for example:

```bash
scp osaka:learn-to-remember/gpu_results.zip ~/Downloads/
```

`python scripts/pack_results.py` re-makes it at any point, for partial results, and
`--rollouts` adds every eval trajectory. Keep `checkpoints/` on the server until the
results are analysed.

## SMOKE rehearsal (plumbing only)

```bash
SMOKE=1 DTYPE=auto MODEL=Qwen/Qwen2.5-0.5B-Instruct bash scripts/run_all.sh
```

This runs every stage with 2 training steps and 3 questions per eval, on any device. Use
it in a scratch copy of the repo: it writes `results/` and `checkpoints/`. Its numbers mean
nothing and are never reported.
