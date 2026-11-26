# learn-to-remember — teaching a small model to use its notes with GRPO

> **One line:** Reuse the hand-written GRPO + LoRA loop from GRPO-Zero to train a small open model (~1.5B) to *use a text memory well*. The memory is a notes index over a synthetic company workspace. The model decides which notes to read, how far to trust them, and when to fall back to the raw files. Its reward is answer correctness minus a cost for every token it reads.

Status: **planning** (written 2026-11-15). Successor to [GRPO-Zero](https://github.com/Vivek-Chaudhari30/GRPO-Zero).

---

## 1. Why this project

### 1.1 The inspiration

Engram's post [**"Learning in a world of notes"**](https://engram.com/blog/learning-in-a-world-of-notes) (Nov 4, 2026) argues that **training and text memory multiply each other**. Agents already store knowledge as text: notes, wikis, skill files. What they lack is judgment about using that memory: which note to open, whether it is complete or stale, and when to check the primary source. Engram trains that judgment with **GRPO + LoRA**, using a reward of **accuracy − λ·cost**, while keeping the notes, harness and prompt fixed. Their headline results:

| Claim | Their evidence |
|---|---|
| Notes and training add up | 2×2 ablation. Law-firm workspace: base 60% → trained without notes 68% → notes only 72% → **notes + training 76%**. Data-warehouse workspace: 15 → 27 → 18 → **37%** |
| A small trained model can match a frontier model | Trained ~27B model ≈ a frontier model given the same notes (~78%), at about a third of the price per query |
| Training survives notes drift | Trained once on early notes, still beats the base model after the notes change by 23M tokens. Retraining on the final notes adds only about 1 point |
| It learns *strategies* | Gap checking (comparing notes with the workspace when something looks off), more but **narrower** lookups (median tokens per lookup 567 → 166), more following of links from notes to source files |

Caveats: these are the company's own benchmarks, and their cost model rests on assumed prices.

### 1.2 Why this is the right next step after GRPO-Zero

- GRPO-Zero built and validated the RL engine: a hand-written GRPO loop (group advantages, clipped objective, k3 KL, LoRA reference with the adapter disabled) and 53 tests. The full GSM8K run showed **no gain** (0.4329 trained vs 0.4397 base, 70 of 200 steps on a T4). The base model was already at 44%, and arithmetic skill moves slowly.
  - *Correction (2026-11-15, see DECISIONS.md):* the T4 run above was superseded. GRPO-Zero's `main` holds a complete run from 2026-10-29 on 1× RTX A6000: 200 steps at **lr 1e-5**, 0.4276 → **0.4716** (+4.4 pts, exact McNemar p ≈ 0.0005), at ~28 s/step. So the engine *does* produce a measurable gain once trained long enough at a higher lr.
- Note-use behaviors are different. Valid tool calls, narrow queries and "check the raw file when the note is missing or stale" are things a small model does **badly at first**, and RL is known to shape them quickly. There is more room to show a measurable gain.
- This is exactly the work of an RL environment: an environment, a verifiable reward and an eval dataset. That is the profile of the target role (AfterQuery: RL environments, reward signals, eval datasets).

### 1.3 Relation to GRPO-Zero

- GRPO-Zero is **finished and frozen** as "v1: the GRPO engine". Its `main` branch should hold the final Kaggle results; they currently live on `results/kaggle-gpu-run`.
- learn-to-remember is a **new repo**. It **copies** (does not depend on) the parts of GRPO-Zero that carry over unchanged, with a header comment in each copied file naming its origin:
  - `compute_advantages`, `grpo_loss`, `per_token_logprobs`, `batched_logprobs` from `src/grpo.py`
  - `src/model_utils.py`: device and dtype detection, LoRA policy loading, adapter loading
  - checkpoint and resume (`save_checkpoint`, `maybe_resume`) from `src/train.py`
  - the matching unit tests
- **Rewritten:** the rollout (becomes multi-turn with tool calls and a loss mask), the verifier and reward, the data loader (becomes a workspace generator), and evaluation.

---

## 2. Research questions and hypotheses

| # | Question | Hypothesis | Experiment |
|---|---|---|---|
| RQ1 | Do notes and GRPO training add up for a ~1.5B model? | notes + training > notes only > base, and notes + training > training without notes | 2×2 ablation |
| RQ2 | Can a cost penalty buy efficiency without losing accuracy? | A moderate λ cuts tokens read by ≥30% for ≤2 points of accuracy | λ sweep, accuracy-vs-tokens Pareto curve |
| RQ3 | Does the learned skill survive notes drift? | A model trained on v0 notes beats the base model on v1–v3 and improves as the notes improve. Retraining on v3 helps only a little | Drift experiment |
| RQ4 | Is it a strategy rather than memorization? | A model trained on company A transfers to an unseen company B | Cross-workspace transfer |
| RQ5 | Which behaviors change? | More fallback to raw files on gap and stale questions, narrower lookups, fewer answers given without any lookup | Behavior metrics from the trajectories |

All results are reported as **measured**, including negative ones. No hypothesis is "confirmed" without a number and a confidence interval.

---

## 3. System design

### 3.1 The workspace: a synthetic company (deterministic, seeded)

A generator builds a fictional company from a seed. Using a different seed gives a different company (company B for RQ4).

**Entities** (~150–250 per company):
- *People*: name, team, role, office, manager, start date
- *Teams / offices*
- *Projects*: codename, owner, team, budget, status, start and end dates, client
- *Clients / contracts*: client, value, signed date, linked project
- *Events*: budget revisions, owner changes, status changes, each with a date. These are what make notes go stale.

**Raw documents** (~400–800, 150–400 tokens each, ~150–250k tokens in total) are written from templates: meeting notes, status reports, emails, contract summaries, HR announcements, all dated.
- Facts are spread across documents, and later documents revise earlier ones.
- **The raw documents are the source of truth.** The gold answer is always the latest value in them.
- Documents include distractors, such as similar codenames and people with the same first name, so broad queries pick up noise.

**Notes** (the text memory): one page per entity plus an index page, built from a snapshot of the facts *as of date t_k*. Each page carries an `as_of:` date and links to the source documents. Note quality is controlled by settings:
- `coverage`: the fraction of entities that have a note page (missing pages → **gap** questions)
- `snapshot date`: facts that changed after it are **stale** in the notes
- `truncation`: some pages leave fields out
- `recipe`: `structured` (key: value plus links), with `prose` as an option for the blog's alternative-recipe comparison

**Note versions for the drift experiment:**
- v0 = coverage 0.4, snapshot t0
- v1 = 0.6, t1
- v2 = 0.8, t2
- v3 = 1.0, t3 (fresh)

*Deviation (2026-11-15, approved, see DECISIONS.md):* v3 is **not** perfect. A fully fresh, fully covered v3 has no `gap` or `stale` cases, and the main run trains on v3, so it would never practise falling back to raw files (RQ5). v3 becomes roughly coverage 0.9, a snapshot slightly before the last change events, and some truncation. Exact values are fixed at M1.

The raw workspace stays the same across versions.

**Question types**, each with a programmatic gold answer and a recorded `qtype`:

| qtype | Example | What it tests |
|---|---|---|
| `lookup` | "What is Project Heron's current budget?" (notes are correct) | Using the notes efficiently |
| `multihop` | "Who manages the owner of Project Heron?" | Chaining lookups |
| `aggregate` | "How many active projects does the Lisbon office own?" | Narrow versus broad search |
| `gap` | The entity has no note page | Falling back to raw files |
| `stale` | The note's value was revised after `as_of` | Noticing and checking stale notes |

Answers are entity names, numbers or dates, with a normalizer (the same idea as GRPO-Zero's verifier: case, punctuation, number formats, date formats).

**Splits:**
- ~2000 training questions
- ~300 eval questions, balanced across qtypes
- eval also includes questions about **held-out entities** that never appear in training questions
- a separate ~300-question **company-B** transfer set

Small eval JSONL files are committed for reproducibility. The full workspace regenerates from its seed.

**Validation that the environment is sound (before any RL):**
- Every gold answer can be derived from the raw documents.
- Every `gap` entity really has no note.
- Every `stale` note really disagrees with the latest raw value.
- A scripted **oracle policy** that uses the tools reaches ~100% accuracy, which proves the environment is solvable.

### 3.2 Tools (the harness: fixed and identical across all conditions)

| Tool | Returns |
|---|---|
| `search_notes(query, k=5)` | BM25 over note pages → ids, titles, short snippets |
| `read_note(note_id)` | The full note page, truncated at a token cap |
| `search_files(query, k=5)` | BM25 over raw documents → paths, dates, snippets |
| `read_file(path)` | The raw document, truncated at a token cap |
| `answer(text)` | Ends the episode |

- Tool calls use the model's **native chat-template tool format**: Qwen's `<tool_call>{json}</tool_call>`, with results in the `tool` role.
- Limits:
  - at most **K = 6** tool turns
  - per-turn generation cap ~256 tokens
  - per-call tool-output cap ~400 tokens
  - total context cap ~4096 tokens
- A malformed tool call returns an error message as the tool result and still uses up a turn.
- "No notes" conditions remove `search_notes` and `read_note` from the tool list and the system prompt.
- The system prompt is fixed for the whole project and logged in `docs/DECISIONS.md`. Engram optimized theirs with GEPA, which we skip.

### 3.3 Reward (deterministic)

```
r = 1[correct]                          # main signal
  + 0.1 · 1[episode ended with a valid answer() call]   # format shaping, independent of correctness
  − λ · min(1, tokens_read / B)         # cost: tool-output tokens returned to the model, B ≈ 4000
```

- No `answer()` within K turns → correctness 0, and the format bonus is not paid.
- λ ∈ {0, 0.2, 0.5}. λ = 0.2 is the main run.
- Watch for **reward hacking**: with a high λ, the model may answer immediately without reading anything. Track the "answered with zero lookups" rate.

### 3.4 Multi-turn GRPO (the main engineering change)

- **Trajectory as one token sequence:** `[prompt | model turn 1 | tool result 1 | model turn 2 | …]`, with a `loss_mask` that is 1 only on **tokens the model generated**. Prompt and tool-result tokens are 0.
- **Re-tokenization pitfall:** keep the model's generated token ids exactly as sampled, and append tool results by tokenizing only that segment, as rendered by the chat template. Do **not** re-render and re-tokenize the whole conversation; tokenization can shift at segment boundaries and silently corrupt the log-probs. Write a unit test for this.
- Advantages are per trajectory, relative to the group mean. Every model token in a trajectory gets that trajectory's advantage. *(Changed 2026-11-26: no division by the group std (Dr. GRPO), `adv_std_norm: false`. With the std division, the first overfit check collapsed to answering without reading. See DECISIONS.md and `docs/incidents/2026-11-26-overfit-collapse/`.)*
- The clipped objective and the k3 KL are computed **only on masked-in tokens**. Normalization: per-sequence mean over masked tokens, then batch mean. The Dr. GRPO normalization is a config option for an ablation.
- Optional (DAPO-style): skip groups whose rewards are all equal (zero advantage) to save compute. Log how many are skipped.
- Monitor collapse: per-token entropy, the fraction of groups with zero reward variance, the KL, and the rate of repeated identical tool calls. RAGEN reports an "Echo Trap" instability when GRPO is used for multi-turn training.

### 3.5 Model

- Default: **Qwen2.5-1.5B-Instruct** (same family as GRPO-Zero, native tool calling).
- Candidate: **Qwen3-1.7B** in non-thinking mode.
- Dev model on the Mac: Qwen2.5-0.5B-Instruct (plumbing only, no conclusions).
- **Selection rule (M3):** probe each candidate on 50 eval questions with notes. Pick the one with **≥80% valid tool calls** and **20–60% accuracy**, which leaves room to improve. Record the probe numbers.

---

## 4. Experiments, in priority order

If compute runs out, cut from the bottom.

1. **Baselines:** base model with notes, base model without notes (greedy, on the full eval set and on company B).
2. **Main training run:** with notes, λ = 0.2, trained on v3 notes. → RQ1, RQ5
3. **Training without notes**, λ = 0.2. → completes the 2×2 for RQ1
4. **λ sweep:** λ = 0 and λ = 0.5. → RQ2
5. **Drift:** train on **v0** notes, evaluate on v0, v1, v2 and v3, and compare with run 2 (trained on v3). → RQ3
6. **Transfer:** evaluate runs 2 and 5 on company B. Evaluation only, no new training. → RQ4
7. *Stretch:* prose-recipe notes; Dr. GRPO normalization; a frontier API model as a reference point given the same notes; joint training of note writing and note reading (see Memory-R1).

**Metrics**, reported per qtype and overall, with **bootstrap 95% CIs**:
- accuracy
- mean tool-output tokens read per question
- lookups per question
- median tokens per lookup
- fallback rate (`search_files`/`read_file` used) on `gap`/`stale` questions
- false-alarm rate (raw-file reads on `lookup` questions where the notes were correct)
- valid tool-call rate
- answers given with zero lookups
- turns used

**Figures:**
- reward and KL curves
- the 2×2 bar chart
- the accuracy-vs-tokens Pareto plot
- accuracy against note version for the drift experiment
- histograms of the behavior metrics
- 3–5 annotated before/after trajectories

---

## 5. Compute plan

| Where | What |
|---|---|
| MacBook Air M2, 8 GB, MPS | Generator, environment, reward, tests, oracle policy, tiny 0.5B rollouts for plumbing. Use the project `.venv`; the base anaconda is broken |
| Kaggle T4 (free) / Colab | Base-model probes and baseline evals, the overfit sanity check |
| Rented L4/A100 (pay as you go) | Real training runs, if T4 speed rules them out |

- In GRPO-Zero, a single-turn step took ~215 s on a T4. Multi-turn will be slower. **Measure seconds per step at M4 before committing to a run**, and write a budget table from the measured numbers. Never estimate it.
- vLLM for rollouts is an **optional** speed-up, used only if measured HF-generate speed makes the plan impossible. It needs LoRA weights synced each step. Keep the pure-HF path working either way.
- Everything is device-agnostic (cuda → mps → cpu) and configured in YAML, as in GRPO-Zero. Checkpoint and resume are on from day one.

---

## 6. Repo layout (target)

```
learn-to-remember/
  README.md
  docs/
    PROJECT.md        ← this file
    DECISIONS.md      ← dated log of design decisions + system prompt text
    RESULTS.md        ← measured numbers only, linked to result files
  configs/            base.yaml, train_notes.yaml, train_nonotes.yaml, ...
  src/learn_to_remember/
    workspace/        entities.py, documents.py, notes.py, questions.py, generate.py
    env/              bm25.py, tools.py, parsing.py, episode.py
    reward.py         answer normalizer + reward
    grpo/             core.py (ported from GRPO-Zero), rollout.py (multi-turn + mask), train.py
    eval.py           conditions × note versions × workspaces, behavior metrics
    oracle.py         scripted policy proving solvability
    model_utils.py    ported from GRPO-Zero
  scripts/            generate_workspace.py, probe_models.py, ...
  notebooks/          kaggle_train.ipynb, analysis.ipynb
  tests/
  data/               committed eval JSONL; generated workspaces are gitignored and rebuilt from the seed
  results/            per-run metrics JSON, curves, trajectories
```

---

## 7. Milestones

Work one at a time. Commit after each one, then stop for review.

| M | Deliverable | Done when |
|---|---|---|
| M0 | Repo bootstrap; GRPO core and its tests ported | Ported tests pass on the Mac; README stub; `.gitignore`; config skeleton |
| M1 | Workspace generator, notes builder, question generator | Seeded generation is deterministic (test); gold, gap and stale invariants have tests; stats table (entities, documents, tokens, qtype counts) in `docs/RESULTS.md` |
| M2 | Tool environment, parsing, reward, oracle policy | Unit tests for every tool, parser edge cases and the reward; oracle reaches ~100% on eval |
| M3 | Multi-turn rollout with loss mask; eval harness; model probe; baselines | Tests: mask is 0 on tool and prompt tokens, re-tokenization matches. Probe numbers recorded and model chosen. Base with/without notes measured on GPU |
| M4 | GRPO training | Overfit sanity: reward → ~1 on 8 questions. Measured seconds per step and a budget table. Main run (experiment 2) complete |
| M5 | Experiments 3–6 | All numbers in `docs/RESULTS.md` with CIs |
| M6 | Behavior analysis and write-up | `analysis.ipynb`, figures, README with an honest read of the results |

---

## 8. Risks and how we handle them

| Risk | Mitigation |
|---|---|
| Base model can barely call tools, so every group scores 0 and there is no learning signal | Model probe at M3. Fallback (reported openly as a deviation): a small SFT warm start on oracle trajectories before GRPO |
| Multi-turn GRPO instability (Echo Trap) | Monitor entropy, zero-variance groups and repeated calls; lower the lr; tune the KL; keep checkpoints |
| The cost penalty teaches "answer immediately" | Cap the penalty with `min(1, ·)`; track the zero-lookup rate; λ sweep |
| Compute | Measure first; priority-ordered experiments; resumable training |
| Synthetic workspace is too easy or too hard | Difficulty settings (distractors, coverage, document count); the probe targets 20–60% base accuracy |
| Silent log-prob bugs from tokenization | Exact-token-id trajectory building and a dedicated unit test |
| Eval noise with ~300 questions | Bootstrap CIs; never claim a difference that the CI doesn't support |

---

## 9. References and resources

**Primary inspiration**
- Engram, *Learning in a world of notes* (Nov 2026): https://engram.com/blog/learning-in-a-world-of-notes

**Our own prior work**
- GRPO-Zero: https://github.com/Vivek-Chaudhari30/GRPO-Zero (local: `/Users/vivek/Developer/CODES/GRPO-Zero`). Results: `results/kaggle-t4-2026-10-15/`

**RL algorithm**
- DeepSeekMath, the paper that introduced GRPO: https://arxiv.org/abs/2402.03300
- DeepSeek-R1, RL from verifiable rewards at scale: https://arxiv.org/abs/2501.12948
- PPO, the clipped objective: https://arxiv.org/abs/1707.06347
- Schulman, *Approximating KL divergence* (the k3 estimator): http://joschu.net/blog/kl-approx.html
- DAPO (dynamic sampling, token-level loss): https://arxiv.org/abs/2503.14476
- Dr. GRPO, *Understanding R1-Zero-Like Training* (normalization bias): https://arxiv.org/abs/2503.20783
- LoRA: https://arxiv.org/abs/2106.09685

**Agents, tools and memory with RL**
- Search-R1, RL for interleaved reasoning and search (closest technical template for masking retrieved tokens): https://arxiv.org/abs/2503.09516 · code: https://github.com/PeterGriffinJin/Search-R1
- RAGEN / StarPO, multi-turn agent RL and the "Echo Trap" instability: https://arxiv.org/abs/2504.20073 · code: https://github.com/RAGEN-AI/RAGEN
- Memory-R1, RL-trained memory manager and answer agent (cited by Engram): https://arxiv.org/abs/2508.19828
- A-Mem, Zettelkasten-style agentic notes: https://arxiv.org/abs/2502.12110
- GEPA, the reflective prompt optimizer Engram used for its fixed prompt: https://arxiv.org/abs/2507.19457
- Also cited by Engram, not yet located: "Librarian Agents" and "Harness Learning". Follow the links in the blog post.

**Tooling**
- Hugging Face Transformers: chat templates with `tools=` (tool-use rendering)
- Qwen2.5 function-calling format (`<tool_call>` tags, `tool` role)
- `rank_bm25` (or a small hand-written BM25) for search
- TRL `GRPOTrainer`, **as a cross-check only and never as the trainer** (same rule as GRPO-Zero)
- vLLM, optional rollout speed-up

---

## 10. Ground rules (carried over from GRPO-Zero)

1. The **GRPO update loop is hand-written**. Libraries can cross-check it but never replace it.
2. **Only real measured numbers.** No placeholders presented as results, no invented metrics, and negative results are reported.
3. **One milestone at a time.** Commit after each, then pause for review.
4. Every reward and environment component is **deterministic and unit-tested** before RL touches it.
5. Decisions go in `docs/DECISIONS.md` with a date and a reason.
