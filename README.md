# learn-to-remember

Teaching a small open model (~1.5B, default `Qwen/Qwen2.5-1.5B-Instruct`) to **use a text memory well** with a hand-written GRPO + LoRA loop. The memory is a notes index over a synthetic company workspace, imperfect on purpose. The model decides which notes to read, whether to trust them, and when to fall back to the raw files. Reward = correctness + format bonus − λ · tokens read.

Successor to [GRPO-Zero](https://github.com/Vivek-Chaudhari30/GRPO-Zero), which provides the GRPO engine. Inspired by Engram's [*Learning in a world of notes*](https://engram.com/blog/learning-in-a-world-of-notes).

- Design spec: [`docs/PROJECT.md`](docs/PROJECT.md)
- Decision log (every design choice, dated, with its reason): [`docs/DECISIONS.md`](docs/DECISIONS.md)
- Measured numbers: [`docs/RESULTS.md`](docs/RESULTS.md)

## Status

**All code is written and tested (M0–M6). No model has been trained or evaluated on a GPU yet**, so there are no model results. Every result below is **not run** until a GPU run produces it. Nothing here is estimated or filled in.

| Milestone | Code | Measured |
|---|---|---|
| M0–M2: GRPO core, workspace, tools, reward, oracle | done | oracle 98.7–100% on eval (the task is solvable) |
| M3: multi-turn rollout, eval harness, model probe, baselines | done | **not run** (GPU) |
| M4: hand-written multi-turn GRPO training | done | **not run** (GPU) |
| M5: experiments 3–6, paired statistics, report tables | done | **not run** (GPU) |
| M6: figures, analysis notebook, this write-up | done (code) | the written reading of the results waits for M3–M5 |

## The question

Can a ~1.5B model learn, from reward alone, *how to use* a notes index: look up the right note, notice when a note is missing or out of date, fall back to the raw files only then, and read no more than it needs?

| | Question | Hypothesis (spec) | Figure | Result |
|---|---|---|---|---|
| RQ1 | Do notes and GRPO training add up? | notes + training > notes only > base, and notes + training > training without notes | `figures/rq1_notes_x_training.png` | not run |
| RQ2 | Can a cost penalty buy efficiency without losing accuracy? | a moderate λ cuts tokens read by ≥30% for ≤2 points of accuracy | `figures/rq2_cost_tradeoff.png` | not run |
| RQ3 | Does the skill survive notes drift? | trained on v0 beats base on v1–v3; retraining on v3 helps only a little | `figures/rq3_drift.png` | not run |
| RQ4 | Strategy or memorization? | trained on company A transfers to unseen company B | `figures/rq4_transfer.png` | not run |
| RQ5 | Which behaviors change? | more file fallback on gap/stale questions, narrower lookups, fewer zero-lookup answers | `figures/rq5_*.png` | not run |

How the results will be read: every number is greedy decoding on the full eval sets, with 95% bootstrap CIs over questions. Comparisons between runs are paired (same questions), using a bootstrap CI on the difference and an exact McNemar p for accuracy. A difference counts only when its CI excludes 0. Negative and null results are reported as plainly as positive ones. Once the runs exist, this section gets a short reading of each RQ: what the CIs support, and what they do not.

## Setup

**Workspace.** A seeded generator builds a fictional company (people, projects, events, documents). Notes summarize it in four versions: v0 (sparse, old) to v3 (current, but still imperfect: ~90% coverage, a snapshot slightly before the last events, some truncation). Five question types: `lookup` (the notes have it), `multihop`, `aggregate`, `gap` (the notes lack it) and `stale` (the notes are out of date). Company A (seed 0) is for training and eval; company B (seed 1) is eval only.

**Tools.** `search_notes`, `read_note`, `search_files` (with an optional date filter), `read_file`, `answer`. Every tool output counts toward tokens read.

**Training.** Multi-turn GRPO with LoRA:
- G = 8 trajectories per question, 8 questions per step, 200 steps, lr 1e-5;
- k3 KL to the frozen reference (the adapter switched off), PPO clip 0.2;
- a loss mask on the model-sampled tokens only;
- trajectories keep the exact sampled token ids (no re-tokenization);
- log-probs are computed in checkpointed chunks, so long contexts fit in GPU memory.

## Reproducing

Dev setup (Mac / CPU, plumbing only; nothing larger than 0.5B runs locally):

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e . --no-deps
.venv/bin/python -m pytest -q
```

Workspace (rebuilt from the seed; generated files are gitignored, the eval sets are committed):

```bash
.venv/bin/python scripts/generate_workspace.py
```

GPU runs: the whole experiment runs on one CUDA GPU with two commands. The full guide is [`docs/RUNBOOK.md`](docs/RUNBOOK.md).

```bash
bash scripts/setup_server.sh
```

```bash
bash scripts/run_all.sh probe
```

```bash
MODEL=<probe choice> bash scripts/run_all.sh
```

The first command is the one-time setup. The second runs the tests, a memory check and the model probe; choose the model from its verdict. The third runs everything else in order: baselines, drift baseline, M4 (timing, gated overfit check, main run) and M5 (experiments 3–6), then the report, the figures and `gpu_results.zip`. It is resumable: after an interruption, run the same command again. The Kaggle notebooks (`notebooks/kaggle_m3/m4/m5.ipynb`) remain as a fallback for T4s, but all runs in one study must use the same GPU type and dtype.

After copying each run's result folder into `results/`:

```bash
.venv/bin/python scripts/make_report.py
```

```bash
.venv/bin/python scripts/make_figures.py
```

`make_report.py` writes `results/report.md` (the RQ tables with CIs and paired tests). `make_figures.py` writes `figures/*.png`. Anything missing shows as "not run". `notebooks/analysis.ipynb` shows the same figures and tables, plus paired base-vs-trained trajectory examples picked by a fixed rule.

## Layout

```
src/learn_to_remember/
  workspace/      generator, notes builder, questions
  env/            tools, BM25, parsing, prompt, episode;  reward.py, oracle.py
  grpo/           core (advantages, loss, chunked log-probs), rollout, train loop
  eval.py         metrics + bootstrap CIs; eval_runner.py runs a model over the eval sets
  compare.py      paired bootstrap + exact McNemar
  report.py       RQ tables;  analysis.py + figures.py  figure data and plots
configs/          base.yaml and per-experiment configs (extends:), report.yaml
scripts/          generate_workspace, run_eval, probe_models, train, make_report, make_figures
notebooks/        kaggle_m3/m4/m5 (GPU runs), analysis (write-up)
results/, figures/  measured outputs only
```
