# GPU runbook: the whole experiment on one CUDA GPU (lab RTX A6000, e.g. umibozu)

Everything from the model probe (M3) to the last experiment (M5) runs with two
commands. Each stage is skipped once it is done, and training resumes from its
last checkpoint, so after any interruption you re-run the same command.

All stages must run on the **same GPU type and dtype** (bf16 on the A6000).
Baselines and trained models are compared question by question, so do not mix in
runs from Kaggle or another card.

## 0. Requirements

- Linux, NVIDIA GPU with >= 16 GB (A6000: 48 GB). bf16 needs compute capability >= 8.0.
- Driver new enough for the torch build: >= 550 for the torch cu124 wheels (`nvidia-smi` shows it;
  osaka had driver 555, CUDA 12.5).
- Python 3.10–3.13 (3.12 recommended) with `venv`, and pip install rights (no root needed).
  torch has no CUDA wheels for 3.14 yet; umibozu's conda base is 3.14, so use a conda env.
- Network: github.com and huggingface.co (~7 GB of models, public, no token).
- Disk: ~15–20 GB free (models ~7 GB, torch ~5 GB, checkpoints ~0.2 GB per run, results small).
- A way to keep jobs alive for many hours: `tmux` (or `screen`).

## 1. Setup (once)

The lab machines are shared. Pick a GPU that shows FREE, and check it yourself before
starting (`nvidia-smi`: next to no memory used, 0–5% utilisation). Then set these in every
shell you run from (put them in `~/.bashrc` to keep them):

```bash
export CUDA_VISIBLE_DEVICES=0 HF_HOME=~/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

`CUDA_VISIBLE_DEVICES` pins the job to one GPU, `HF_HOME` keeps the ~7 GB of models in
your home folder, and `expandable_segments` makes out-of-memory errors from fragmentation
less likely with long contexts. If the server's Python is 3.14 or only conda's, make a
Python 3.12 env for this project and pass it to the setup script, which still builds its own
`.venv` (later commands use `.venv`, so the env is only needed once). With a driver >= 570
(umibozu has 580), the cu128 wheel index gives a newer torch:

```bash
conda create -n ltr python=3.12 -y && conda activate ltr
```

```bash
PY=python TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash scripts/setup_server.sh 2>&1 | tee setup.log
```

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

## 1b. CUDA smoke test (optional, a few minutes)

The full pipeline was rehearsed on a Mac, not on CUDA. Before the real run, rehearse it
once on the server with the small 0.5B model, in a **separate clone** (it writes `results/`
and `checkpoints/`, which must not mix with the real run):

```bash
git clone https://github.com/Vivek-Chaudhari30/learn-to-remember.git ~/ltr-smoke
```

```bash
cd ~/ltr-smoke && ln -s ~/learn-to-remember/.venv .venv
```

```bash
SMOKE=1 MODEL=Qwen/Qwen2.5-0.5B-Instruct bash scripts/run_all.sh
```

It runs every stage (2 training steps, 3 questions per eval) in bf16 on the GPU and
ends with `All stages done`. It exercises the CUDA, bf16, LoRA, checkpoint and eval paths.
Its numbers mean nothing and are never reported. Delete `~/ltr-smoke` afterwards. If it
fails, bring back its `logs/` before starting the real run.

## 1c. Pilot run (recommended, ~1–3 h)

The smoke test proves the code paths work. The pilot proves the **real** run works: the
real model (Qwen2.5-1.5B) at the real settings (group size 8, 8 questions per step, 4k
context, same batch sizes), so memory use and speed are what the full run will see. Each
stage is cut short: 5 training steps per run and 16 questions per eval. It runs every
stage, from baselines to the report and zip, in order. Use its own clone again:

```bash
git clone https://github.com/Vivek-Chaudhari30/learn-to-remember.git ~/ltr-pilot
```

```bash
cd ~/ltr-pilot && ln -s ~/learn-to-remember/.venv .venv
```

```bash
tmux new -s pilot
```

```bash
PILOT=1 bash scripts/run_all.sh
```

At the end it runs `scripts/pilot_estimate.py`, which prints:

- the measured seconds per training step and per eval question (medians, warm-up step dropped);
- peak GPU memory;
- the full run's hours, from those measured rates;
- `PROBLEMS:` if any loss went non-finite or memory came close to 48 GB.

If it ends with `No problems found`, start the real run (§2). If not, bring back
`~/ltr-pilot/gpu_results.zip` and `logs/`. `PILOT_STEPS` and `PILOT_LIMIT` change the
lengths (defaults 5 and 16). Pilot accuracies mean nothing and are never reported. Keep
`~/ltr-pilot` until the real run has started cleanly, then delete it.

Each clone records what kind of run it holds (`results/.stages/.kind`) and refuses a
different kind, so pilot or smoke stages can never be mixed into the real `results/`.

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

The next command checks this for you: it refuses to start unless the `MODEL` you pass
passed the probe. `PROBE_CHECK=0` overrides that check; use it only for a deviation we
have agreed on and logged.

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
scp umibozu:learn-to-remember/gpu_results.zip ~/Downloads/
```

`python scripts/pack_results.py` re-makes it at any point, for partial results, and
`--rollouts` adds every eval trajectory. Keep `checkpoints/` on the server until the
results are analysed.

## SMOKE rehearsal (plumbing only, any device)

```bash
SMOKE=1 DTYPE=auto MODEL=Qwen/Qwen2.5-0.5B-Instruct bash scripts/run_all.sh
```

This runs every stage with 2 training steps and 3 questions per eval, on any device. Use
it in a scratch copy of the repo: it writes `results/` and `checkpoints/`. Its numbers mean
nothing and are never reported.
