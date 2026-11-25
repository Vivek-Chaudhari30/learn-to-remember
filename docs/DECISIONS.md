# Decision log

Dated log of non-obvious design decisions, with the reason for each. Deviations from `docs/PROJECT.md` are marked **[deviation]**.

## System prompt

Fixed on 2026-11-15 (M2), in `src/learn_to_remember/env/prompt.py`, pinned by sha256 in `tests/test_prompt_and_oracle.py`. Changing it invalidates comparisons with earlier runs. The tool schemas (JSON) are rendered by the Qwen chat template's own `# Tools` block after this text.

**Notes condition** (sha256 `b26e52d4…`):

```
You answer questions about a company using its internal workspace. Today is 2025-12-31.
There are two sources. The raw files are the source of truth. The notes are a curated summary of them; they are faster to use but may be incomplete or out of date. Each note has an as_of date and lists its source files. When a note is missing, lacks the detail you need, or may have changed after its as_of date, check the raw files.
Use the tools to look things up. Each turn, make exactly one tool call. You have at most 6 tool calls in total, and the final one must be answer. Reading costs: every token a tool returns counts against you, so prefer narrow searches and read only what you need.
Give the final answer with the answer tool, as just the answer: a name, a number, a date or a short phrase, with no explanation.
```

**No-notes condition** (sha256 `d5d6ff2f…`):

```
You answer questions about a company using its internal workspace. Today is 2025-12-31.
The raw files are the source of truth.
Use the tools to look things up. Each turn, make exactly one tool call. You have at most 6 tool calls in total, and the final one must be answer. Reading costs: every token a tool returns counts against you, so prefer narrow searches and read only what you need.
Give the final answer with the answer tool, as just the answer: a name, a number, a date or a short phrase, with no explanation.
```

The user turn is the question text alone. Measured with the Qwen2.5 template including the tool schemas: about 745 prompt tokens (notes) and about 498 (no notes), for a 14-word question.

---

## 2026-11-15: GRPO-Zero's "no gain" premise was out of date

PROJECT.md §1.2 cited the 2026-10-15 Kaggle T4 run (70/200 steps, lr 1e-6, 0.4397 → 0.4329). GitHub's GRPO-Zero `main` already held a later, complete run: 2026-10-29, 1× RTX A6000, 200 steps, lr 1e-5, 0.4276 → 0.4716 (+4.4 pts, McNemar p ≈ 0.0005), ~28 s/step for 64 completions of ≤512 tokens with the 0.5B model. The local checkout was 11 commits behind `origin/main`.
- Action (approved): fast-forwarded GRPO-Zero's local `main`, added only `results/kaggle-t4-2026-10-15/` as an archive (commit `ce51ed9`, README unchanged), and pushed it. That is the only change made to GRPO-Zero.
- Consequence: `configs/base.yaml` starts at **lr 1e-5**, not GRPO-Zero's config default of 1e-6, which showed no gain.

## 2026-11-15: v3 notes are imperfect **[deviation, approved]**

Spec §3.1 defines v3 as coverage 1.0 and a fresh snapshot. That version has no `gap` and no `stale` questions, and the main run (experiment 2) trains on v3, so it could never learn to fall back to raw files, which is half of RQ5. v3 becomes roughly coverage 0.9, a snapshot slightly before the final change events, and some truncation, while staying the best of the four versions. Exact numbers are fixed and logged at M1.

## 2026-11-15: Generic `loss_mask` via full-sequence log-probs

`grpo/core.py` adds `sequence_logprobs(model, input_ids, attention_mask)`, which returns `[B, L]` with position `t` holding `log π(x_t | x_<t)` (position 0 = 0, since it has no prediction). A multi-turn trajectory `[prompt | gen | tool | gen | …]` is then one row, and `loss_mask` (1 = model-generated token) picks the positions to train on. GRPO-Zero's `per_token_logprobs(prompt, completion)` becomes a thin wrapper, which is the special case where the mask covers the completion block. `grpo_loss` already worked on any mask of the log-prob shape. A test proves the two paths agree.

## 2026-11-15: Spec budget inconsistencies (resolved at M2, see the M2 entry)

- K=6 × 400 tool tokens + 6 × 256 generated tokens + a system prompt with tool schemas (~700 tokens, to be measured) ≈ 4.6k tokens, over the 4096 context cap.
- With at most 2400 tool tokens readable, B = 4000 never binds, so `min(1, ·)` is a no-op and the most a run can lose to cost is λ·0.6.
- The spec doesn't say whether the `answer` call counts toward K.
- These will be resolved with measured token counts at M2 and logged here.

## 2026-11-15: Logit memory at long context (resolved in code at M3, see the 2026-11-16 entry)

`sequence_logprobs` materializes `[B, L, V]` logits as GRPO-Zero did. At L = 4096 with Qwen's V ≈ 152k that is ~2.5 GB per row in fp32, which is too much for a T4. At M3 this becomes a chunked computation over hidden states (lm_head plus logsumexp in slices, only at masked positions). The M0 API is kept so that change stays internal.

## 2026-11-15: Explicit position ids; inf-safe masking

- `sequence_logprobs` passes `position_ids` computed from the attention mask, the same way `generate` does. I checked whether this fixes a real GRPO-Zero bug, and it doesn't: Qwen uses RoPE, which is invariant to a constant position shift, and the padding-invariance test passes with or without it. It's kept because it costs nothing and holds for any position scheme.
- `grpo_loss` zeroes masked-out tokens with `torch.where` instead of `* mask`, so an inf on a tool or prompt token can't turn into `inf * 0 = nan`. A test injects inf there.

## 2026-11-15 (M1): Workspace generator design

- **Ground truth is a per-fact timeline** (`workspace/entities.py`). Docs, notes and gold answers are all derived from it, so they can't disagree by accident. Every doc records the facts it asserts. The tests check that each assertion equals the timeline at the doc's date (consistency), that each fact's final value is asserted on or after its last change (coverage), and that every gold answer can be re-derived from document assertions alone.
- **No fact changes twice within 14 days.** Weekly meeting notes then never contain two values of one fact. Same-day conflicts are impossible.
- **Q4 2025 planning wave:** event rates ×2–4 from 2025-10-01 to 2025-12-12. This is what makes the v3 notes (snapshot 2025-09-30) stale on real facts. Without it, company A had only ~85 v3-stale candidates, and the eval stale bucket fell short of 60.
- **Note versions** (exact values; v3 imperfect as approved):

  | version | coverage (nominal) | snapshot | P(page truncated) |
  |---|---|---|---|
  | v0 | 0.40 | 2024-09-30 | 0.30 |
  | v1 | 0.60 | 2025-02-28 | 0.20 |
  | v2 | 0.80 | 2025-06-30 | 0.15 |
  | v3 | 0.90 | 2025-09-30 | 0.10 |

  Coverage is nested: a fixed per-company priority order, so a page in v_k also exists in v_{k+1}. Entities created after a snapshot have no page, so actual coverage is lower than nominal (v3: 0.87). A truncated page leaves out 1–2 fields.
- **qtype is computed per note version** for single-fact questions: `gap` (with `gap_kind` = `missing_page` or `missing_field`; truncation-caused gaps are gaps too), `stale` (the page value differs from the latest raw value), or `lookup`. `multihop` and `aggregate` keep their structural label and record `notes_ok_by_version`. The primary `qtype` is the v3 label, and eval is balanced on it.
- **Train/eval disjointness is at the fact level:** no (entity, attr) fact read by an eval question is read by any training question, and no (family, anchor) key or question text is shared. An aggregate's "fact" is (anchor, family). The counted member facts can overlap with singles; this is accepted.
- **Held-out entities:** 10% of people, projects, clients and contracts (22 in company A). A training question may not touch one as anchor, hop or answer entity. The test also checks question and gold text with word boundaries.
- **Eval size:** 60 per qtype, capped at 60% of each v3 bucket for company A so training keeps some stale/aggregate facts. Company A ends up with 293 (aggregate 53). Company B has no training split and gets 300 (60 × 5).
- **Aggregates:** zero counts are skipped (too easy to guess). Year-end reviews (office portfolio, office headcount, team roster with its projects, client account review) make every aggregate answerable from raw files in one or two reads. The team roster's project list was added after the derivability test caught that team-project counts had no direct raw source.
- **Excluded from questions:** the CEO and the Leadership pseudo-team, and `client.industry`, which follows from the client's name ("Redwood Insurance" → insurance) and could be guessed without a lookup.
- **Distractors:** first names repeat across people, paired codenames ("Flint" / "Night Flint"), old values in "from X to Y" phrasing, and at most 2 passing mentions of other people/projects per doc. Filler sentences are drawn without replacement. No "rejected proposal" distractors in this version: they would make the task harder in a way we can't yet calibrate (revisit if the probe shows the task is too easy).
- **Determinism:** all randomness goes through `sub_rng(seed, *keys)` (sha256-keyed), never `hash()`, and sets are never iterated unsorted. Tested byte-identical across processes with different `PYTHONHASHSEED`.
- **Note page sizes vs. the 400-token tool cap:** the index pages for people/projects/contracts exceed 400 tokens by design (a broad read gets truncated, as in the blog). One office page (Seoul, 448 tokens) also exceeds it. List headers carry counts (`people (30):`), so aggregates stay readable even when truncated.
- **Open (M4): the training mix is lookup-heavy.** At v3: 1222 lookup / 158 multihop / 72 aggregate / 188 gap / 74 stale (the 74 stale are ~37 facts × 2 paraphrases). To be decided at M4: stratified sampling by qtype, or training on a mix of versions.

## 2026-11-15 (M2): Tool environment, reward, oracle

- **Search is hand-written Okapi BM25** (k1 = 1.5, b = 0.75, small stopword list, ties broken by document order). Ids like `C-1589` are single tokens; before that, "c" matched every contract doc. **Bigrams were tried and rejected:** measured top-10 recall of the target doc was 268/300 vs 283/300 for unigrams. Bigrams make long list docs longer, and the length penalty then buries them. Note search weights the title (it is indexed three times), because an office page lost to dozens of people/team pages that merely contain `office: <city>`.
- **`search_files(query, k=5, since=None)` [deviation].** The spec's tool table has no `since`. BM25 cannot express "newer", and the newest update to a fact (a meeting note, an email) was often outranked by older status reports about the same project. Those stale questions were unsolvable with natural queries. With `since`, top-10 recall of the target rose from 283/300 to 293/300. It's also the natural tool for "did this change after the note's as_of?", the behavior RQ5 measures. It exists in both conditions (the harness stays identical). It can be removed if you prefer the spec as written. Measured on the final env with the `since` fallback disabled, the oracle scores 91.7–93.9% (A: v0 92.2%, v3 93.9%, no notes 91.8%; B: 92.3%, 93.0%, 91.7%), vs 98.7–100% with it.
- **Generator changes forced by the oracle** (they change M1's data; eval sets and stats were regenerated, and all M1 invariant tests still pass):
  - at most one passing mention per doc, in ~30% of docs (before: ~20 per codename, burying real updates);
  - at most 3 decisions per weekly meeting note (Q4 weeks had bundled up to ~8);
  - roster column headers (`name | role | office | start date | manager`);
  - every template variant names its attribute (dropped "Ownership of X moves…" and "X will now finish by…", which never said "owner" / "end date").
- **Turn rules:** K = 6 model turns *including* `answer`. A non-answer call on turn 6 is not executed and costs nothing; its output would never reach the model. A malformed call or tool error returns an error observation (which is counted in tokens) and uses the turn. Only the first `<tool_call>` per turn runs.
- **Budgets, measured:** prompt ≈ 745 tokens (notes) / 498 (no notes); 18 template tokens per tool turn; tool output ≤ 400. Worst case 745 + 5 × 418 + 6 × 256 ≈ 4371 > 4096. The 4096 cap is kept and enforced: an episode ends with `context_limit` when the next turn cannot fit. Real turns generate far fewer than 256 tokens (to be measured at M3).
- **Cost budget B = 4000 kept as in the spec.** At most 5 × 400 = 2000 tool tokens can be read, so `min(1, ·)` never applies: the cost is linear, λ · tokens/4000, at most λ/2. Kept on purpose so λ values mean what the spec says; revisit at the λ sweep if the penalty is too weak.
- **Normalizer:** for names, the maximal catalog matches in the answer must be exactly {gold}. Listing several names, a first name alone, "Senior X" for "X", or "Night Heron" for "Heron" all fail. For money/date/count, exactly one distinct value. A bare integer ≥ 1000 counts as money when no `$`/`USD`/`M` marker is present. Status synonyms are mapped. Numeric d/m/y dates are rejected as ambiguous.
- **The oracle is privileged about what, not how** (see `oracle.py` and RESULTS.md). It uses natural queries with k = 10, plus two fallbacks: the same query with `since` = the fact's last change, then `"<name> team roster"` for people and teams. Contracts are searched by id alone.

## 2026-11-16 (M3): Multi-turn rollout, log-probs, eval harness, probe

- **Exact-id trajectories** (`grpo/rollout.py`). A trajectory is built by appending: the prompt ids, then each turn's sampled ids exactly as `generate` returned them (through the stop token), then each tool result tokenized as a segment by itself. The segment's framing strings are cut out of the tokenizer's own chat template with sentinels, so nothing Qwen-specific is hard-coded. For Qwen2.5 a segment is `"\n<|im_start|>user\n<tool_response>\n" + obs + "\n</tool_response><|im_end|>\n<|im_start|>assistant\n"`. It starts right after `<|im_end|>`, a special token and so a hard token boundary, which makes tokenizing it alone exact.
  - Tests prove that for Qwen2.5 with canonically tokenized turns this equals rendering and tokenizing the whole conversation. They also prove two cases where re-rendering would be **wrong**: a non-canonical sampled split (re-tokenizing merges it), and **Qwen3**, whose template drops the empty `<think></think>` block from earlier assistant turns even though the model was prompted with one before every turn. On a real (tiny, random) model, teacher-forced log-probs of the trajectory equal the sampling-time log-probs token for token (atol 1e-4).
  - `loss_mask` = 1 on sampled ids only, including the sampled stop token. A turn cut at the 256-token cap, or one that stopped on `<|endoftext|>` instead of `<|im_end|>`, gets an `<|im_end|>` appended with mask 0. The model did not sample it, so it is not trained on, but the next segment is framed correctly.
  - The context check runs on exact ids, with the same rule as the M2 text runner: a turn starts only if prompt-so-far + 256 fits in 4096; otherwise the episode ends with `context_limit`.
- **Generation parameters are all pinned [bug avoided].** transformers 5 fills every generation parameter left unset from the model's own `generation_config`, and Qwen2.5-Instruct ships `repetition_penalty 1.1, top_k 20, top_p 0.8, temperature 0.7`. A plain `generate(do_sample=False)` would therefore apply a repetition penalty to "greedy" eval, and RL sampling would draw from a truncated, re-weighted distribution that is not the policy GRPO scores. `HFGenerator` sets `repetition_penalty 1.0, top_k 0, top_p 1.0`, temperature 1.0 for sampling (spec), and greedy for eval. It also overwrites the model's generation_config. A test plants a hostile generation_config and checks that greedy output equals a plain argmax loop.
- **Chunked log-probs** (`grpo/core.py`). `sequence_logprobs` runs the backbone once, then lm_head + logsumexp over slices of 1024 positions, each checkpointed when grads are on. It never builds `[B, L, V]` logits. With `logit_mask` (the loss mask) it computes only those positions; in a trajectory most tokens are prompt and tool output. Models without a separate head (the fake LMs in the M0 tests) use the old full path. Tests: equal to the full-logits computation for chunk sizes 1/7/64/4096, equal LoRA gradients through layer gradient checkpointing, and the reference policy via `disable_adapter` still matches the base model. The measured T4 memory is pending (`scripts/logprob_memory.py`, in the Kaggle notebook).
- **Eval metric definitions** (`eval.py`):
  - `fallback_rate`: gap/stale questions that used a file tool (`search_files` or `read_file`).
  - `false_alarm_rate`: lookup questions (notes correct and fresh) that used a file tool, notes conditions only. The spec says "raw-file reads"; searching the files counts too, so the two rates are defined the same way.
  - `valid_call_rate`: turns whose call parsed and named a real tool with valid arguments / all model turns, pooled over questions. This is the "valid tool calls" of the §3.5 probe rule.
  - CIs: percentile bootstrap over questions, 2000 resamples, seed 0.
  - qtype labels follow the condition's notes version; `no_notes` uses v3 labels, as the oracle does.
  - Seconds per question = the rollout batch's wall time / its size.
- **Probe subset:** 10 company-A eval questions per v3 qtype, drawn with `random.Random(0)` over sorted ids, committed as `data/eval/probe_company_a.json`. The spec chooses the model on eval questions, so the choice touches 50 of the 293 company-A eval questions. This is accepted and logged; reported company-A baselines and trained numbers use the full set. Tie-break: if both candidates pass, keep the spec default (Qwen2.5-1.5B-Instruct). If neither passes, stop and decide with the user (the risk table's fallback is an SFT warm start on oracle trajectories, which would be a reported deviation).
- **Eval is greedy** and runs both companies in parallel on Kaggle's two T4s (`notebooks/kaggle_m3.ipynb`), in fp16 (the T4 has no native bf16).

## 2026-11-16 (M4): GRPO training loop

Written before the M3 GPU numbers exist, at the user's request. The code is model-agnostic; the model id in `configs/base.yaml` stays the spec default until the probe picks one.

- **Old log-probs = new log-probs, detached, when `ppo_epochs: 1`.** The update runs once, on the batch the current policy just sampled, so pi_old *is* the current policy: the ratio is exactly 1, clipping never fires, and the gradient is A · ∇log pi (plus KL). This saves a forward pass. It also removes a noise source GRPO-Zero had: old in eval mode, new in train mode, so LoRA dropout (0.05) made the ratio ≠ 1 at step start. With `ppo_epochs > 1`, old is computed in one no-grad pass before the epochs.
- **Exact gradient accumulation.** Each micro-batch's loss is weighted by its share of rows (`b/B`), so for `seq_mean` the accumulated gradient equals the full-batch gradient even when the last micro-batch is short. GRPO-Zero's `loss / n_micro` is slightly off then. Tested: micro-batch sizes 1, 3 (uneven) and 8 give the same gradients. Micro-batches group rows of similar length to cut padding.
- **Reference log-probs** come from the policy with the adapter disabled, no grad, at loss-mask positions only. **Entropy** (detached, per masked token) comes from the same chunked pass as the new log-probs, for collapse monitoring.
- **Training question mix: stratified by v3 qtype** (resolves the M1 open item). Each step draws P/5 questions per qtype, with the remainder rotated across steps; each qtype walks its own pool, reshuffled every epoch. The schedule is a pure function of (seed, step), so resume is exact. Consequence: over 200 steps × 8 questions each qtype gets ~320 draws, so the 74 stale questions (~37 facts) are each seen ~4 times and the 72 aggregates ~4–5 times, against 1222 lookups seen ~0.3 times. That is a real risk of memorizing stale facts, so it must be checked on the held-out eval (whose facts are disjoint from training) rather than assumed away. `question_mix: natural` is kept as an ablation.
- **Zero-variance groups** are counted every step (`frac_zero_var_groups`). They are kept by default (spec): their advantage is 0, but they still carry the KL term. `skip_zero_variance_groups: true` drops them.
- **fp16 (T4):** PEFT keeps LoRA weights in fp32 (checked: `lora_dtypes ['torch.float32']`), and a `torch.amp.GradScaler` is used automatically when the frozen base is fp16 on CUDA. bf16 and fp32 run without a scaler.
- **Optimizer:** AdamW at a constant lr (GRPO-Zero's setup, lr 1e-5 per the 2026-11-15 entry), grad clip 1.0, KL 0.04, clip ε 0.2, G = 8, P = 8, 200 steps, checkpoint every 10 steps.
- **Overfit sanity config** (`configs/overfit.yaml`): 8 fixed training questions (one or two per qtype), 60 steps, **lr 5e-5**. The higher lr is deliberate: the check is that the loop *can* fit, not a result. Done when the batch reward climbs to ~1.
- **Monitoring per step** (`train_log.json`): reward mean/std, accuracy (overall and per qtype), valid-call rate, KL, entropy, clip fraction, zero-variance share, tokens read, lookups, zero-lookup rate, repeated calls (the Echo Trap signal), how episodes ended, generated tokens per turn, the share of turns cut at the cap, grad norm, scaler scale, and seconds per step split into rollout / reference / update. The best and worst trajectory per step go to `rollouts/train_samples.jsonl`.
- **Configs use `extends:`** (deep merge, `model_utils.load_config`): `train_notes.yaml` (experiment 2), `train_nonotes.yaml` (experiment 3), `overfit.yaml`.

## 2026-11-17 (M5): Experiments 3–6 harness and paired statistics

Written before any GPU result exists (M3 and M4 numbers are still pending). No experiment has run.

- **Runs** (spec §4): exp 3 `train_nonotes.yaml`; exp 4 `train_l0.yaml` and `train_l0.5.yaml` (notes v3); exp 5 `train_v0.yaml`; exp 6 is eval only. Every run is identical to the main run (`base.yaml`: 200 steps, P = 8, G = 8, lr 1e-5, KL 0.04, stratified mix) except for the one setting being studied. `scripts/run_m5.sh <exp>` trains, then evaluates with the same greedy harness as the baselines.
- **Exp 5 trains on v0 labels.** The sampler stratifies by the training version's qtype. At v0 the training pool is 986 gap / 404 lookup / 158 multihop / 94 stale / 72 aggregate, so stratifying keeps it from being two-thirds gap. Same caveat as M4: rare qtypes repeat (~3–4× per question over 200 steps).
- **RQ3 needs three extra eval sets,** now in the plan: the base model on v0–v2 (`drift_base`), the exp-2 adapter on v0–v2 (`exp6`), and the exp-5 adapter on v0–v3.
- **RQ4:** the exp-2 and exp-5 adapters are evaluated on company B, notes v3, against the base model on company B (from the M3 baselines). Company B is never trained on.
- **Comparisons are paired** (`compare.py`), because every condition answers the same questions:
  - accuracy difference: percentile paired bootstrap, 2000 resamples, seed 0, plus an exact two-sided McNemar p;
  - tokens read: relative change mean(b)/mean(a) − 1 with a paired bootstrap CI.
  
  A difference is called one only when its 95% CI excludes 0 (spec §8). Comparing two independent CIs would be needlessly loose, because question difficulty dominates the variance. `align` refuses to pair runs over different question sets.
- **RQ2 reads against λ = 0, not the base model.** The hypothesis is about what the penalty buys, so the comparison isolates the penalty from training itself.
- **`scripts/make_report.py`** builds `results/report.md` from the result files listed in `configs/report.yaml`. Anything missing is printed "not run". It reports estimates and CIs; it does not mark hypotheses as confirmed.

## 2026-11-17 (M6, code only): figures, analysis notebook, README

The user asked to build M6 before any GPU result exists. Only the result-independent parts are built. The written reading of the results (the README "honest read") waits for real numbers, and the README says so.

- **One code path for all figures.** `analysis.py` pulls figure data out of the result files (pure functions, no plotting). `figures.py` draws it. `scripts/make_figures.py` and `notebooks/analysis.ipynb` both call these, so the notebook and the committed figures cannot disagree. The notebook is committed without outputs. `configs/report.yaml` gains `train_logs:` (run → `train_log.json` folder).
- **Missing data is never drawn as data.**
  - A figure none of whose runs exist is not written; it is listed as "not run" in `figures/INDEX.md`.
  - A figure with some runs missing is drawn with what exists, says "not run: ..." under the plot, and shows "not run" in each empty bar slot.
- **CIs on every point.** Summary metrics use the CI `eval.summarize` stored. Per-qtype rates computed in `analysis.py` (accuracy and file-tool use by qtype) use the same percentile bootstrap (2000 resamples, seed 0). Paired tests stay in the report tables (`report.py`), not the figures.
- **Figures:**
  - training curves per run: faint per-step line plus a 10-step trailing mean; rate panels fixed to 0–1 so noise is not magnified;
  - λ-sweep training curves;
  - RQ1: 2×2 bars;
  - RQ2: accuracy vs tokens read, both CIs, λ points joined in λ order;
  - RQ3: accuracy by eval notes version;
  - RQ4: company B bars;
  - RQ5: accuracy and file-tool use by qtype, and a lookups-per-question histogram.
- **Color follows the entity across all figures:** base = blue `#2a78d6`, trained on v3 = orange `#eb6834`, trained on v0 = aqua `#1baf7a`.
  - This is the reference data-viz palette's first three slots. The validator passes them all-pairs on the light surface (worst CVD ΔE 9.2, normal-vision 24.0). A grey "base" failed the chroma floor, so it was not used.
  - Aqua is under 3:1 contrast, so every series is direct-labeled or named on the axis.
  - The λ-sweep curves use one blue ordinal ramp (λ=0 light → 0.5 dark), which passes `--ordinal`.
- **Paired trajectory examples** (base vs trained, company A, v3): the questions the trained model won and lost, picked by a fixed rule (qtype order, then id), never by hand. Full trajectories come from `rollouts/` (gitignored) when present, otherwise from the saved samples.
- **matplotlib ≥ 3.8** added to `requirements.txt` (3.11.2 in the venv). The tests run the figures and the whole notebook on synthetic result files. Synthetic numbers are plumbing only and are never reported.

## 2026-11-25: GPU moves to umibozu; PILOT dress rehearsal; Python 3.10–3.13 only

- **Server: umibozu** (lab RTX A6000, 48 GB, compute 8.6, driver 580 / CUDA 13.0). osaka was busy; the user picked the free A6000. Same GPU type and bf16, so nothing in the study design changes. Every stage still runs on one card type. The home folder is on the lab's NFS share (30 TB free).
- **`PILOT=1 bash scripts/run_all.sh`**, asked for by the user: a trial of the whole run before committing ~2 days of GPU. The real model (default Qwen2.5-1.5B) at the real settings (G = 8, P = 8, 4k context, same batch sizes), so memory and speed are what the full run will see. Each stage is cut short: `PILOT_STEPS` (5) training steps per run and `PILOT_LIMIT` (16) questions per eval. No probe needed, no overfit gate. It ends with `scripts/pilot_estimate.py`: median s/step (warm-up step dropped), median s/question, peak GPU memory, the full run's hours (1,063 steps, 5,920 eval questions), and `PROBLEMS:` on non-finite losses or peak memory > 40 GB. Pilot accuracies are never reported. It needs its own clone, like SMOKE.
- **One kind of run per clone.** `run_all.sh` writes `results/.stages/.kind` (real / pilot / smoke) and refuses a different kind in the same clone, so pilot or smoke stages can never be skipped-over as "done" in the real `results/`.
- **Python 3.10–3.13 only.** umibozu's conda base is Python 3.14, which has no torch cu124 wheels, so the setup would have failed half-way. `setup_server.sh` now refuses outside 3.10–3.13 with a hint. The runbook uses a conda env (`conda create -n ltr python=3.12`) and, for driver ≥ 570, `TORCH_INDEX=.../whl/cu128`.
- **Lab-server settings in the runbook:** `CUDA_VISIBLE_DEVICES=0`, `HF_HOME=~/hf_cache`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, check `nvidia-smi` before starting, run in tmux. These came from the GRPO-Zero run on tottori; the lab's own written rules have not been read yet.
- **Revised estimate (not measured):** GRPO-Zero's tottori eval took ~30–60 min for 1,319 single-turn 0.5B questions, so our ~5,900 multi-turn 1.5B eval questions are ~5–10 h, not 1–3 h. Full run ~35–55 h. The pilot replaces this guess with measured rates.
- **Rehearsal (Mac, plumbing only):** SMOKE ran every stage to `All stages done`. `pilot_estimate.py` parsed its output correctly. A PILOT run in that smoke clone was refused (exit 1). A PILOT run with every stage already marked done exported `--steps 5` / `--limit 16`, ran report, figures and zip, then the estimate. Python 3.14 is refused and 3.12/3.13 pass.

## 2026-11-18: probe check before the full run; CUDA smoke test on osaka

- **`run_all.sh` now checks the probe verdict.** Before the first stage it reads `results/probe/probe_summary.json` and exits 1 unless the `MODEL` it was given passed the §3.5 rule. Before, `run_all.sh probe` marked itself done whatever the verdict, so a model that failed (or a typo in `MODEL`) could start hours of training on a bad learning signal. The check also notes when Qwen2.5-1.5B passed but another model was chosen, because the spec keeps Qwen2.5 when it passes. `PROBE_CHECK=0` overrides, only for an agreed and logged deviation (for example an SFT warm start if neither model passes). SMOKE runs skip the check, because they have no probe. Tested in a scratch copy with a made-up summary: a failing model, an unprobed name and a missing summary each exit 1 before any stage; a passing model runs on; `PROBE_CHECK=0` skips the check.
- **RUNBOOK §1b: optional CUDA smoke test.** The rehearsal ran on the Mac, so the CUDA/bf16 path has not run yet. `SMOKE=1` with Qwen2.5-0.5B in a separate clone on osaka (sharing the venv through a symlink; every script puts its own clone's `src/` first on `sys.path`, so the clone uses its own code and data) tests that path in minutes without touching the real `results/`.

## 2026-11-17: GPU runs on one A6000 (osaka), bf16, one pipeline

The user has a free RTX A6000 (48 GB, Ampere, driver 555 / CUDA 12.5) on a lab machine, osaka, and asked for the repo to be made ready to run the whole GPU part there. The user starts the runs.

- **All GPU work runs on osaka in bf16:** probe, baselines, training and every eval. Comparisons are paired on the same questions, so the baseline and the trained model must share the GPU type and dtype. Greedy decoding in fp16 on a T4 and in bf16 on an A6000 can pick different tokens, which would leak a hardware difference into the result. The Kaggle notebooks stay as a fallback only, and no study mixes the two.
- **`scripts/run_all.sh` runs M3 → M4 → M5 → report in order.** Each stage writes `results/.stages/<stage>.done` and is skipped on a re-run. Training resumes from its 10-step checkpoints. `scripts/setup_server.sh` builds the venv (torch cu124), downloads the models and runs the tests. `docs/RUNBOOK.md` has the step-by-step guide.
- **Human decision points stay human.**
  - The model choice: `run_all.sh probe` stops after the probe, and the user sets `MODEL` from its verdicts.
  - An automatic **overfit gate**: the main run starts only if the overfit check's mean fraction correct over its last 10 steps is ≥ 0.8. GATE=0 overrides it, after reading the log.
  - **Timing:** `m4_timing` prints the measured s/step and the training hours it implies. `PAUSE_AFTER_TIMING=1` stops there.
- **Eval batch size stays 16 everywhere** (baselines and trained models alike), so left-padding is the same for every compared run. The training micro-batch (`MICRO`) may be raised for speed: accumulation is exact, so the gradient does not change.
- **Fix:** `run_eval.py` now rebuilds `tables.md` from every result file in its `--out` folder and merges `timing.json`. Before, a second call into the same folder (exp 5 evaluates company A then company B; exp 6 adds to the main run's folder) overwrote the earlier tables and timings. The per-condition eval JSONs were never affected.
- **`requirements.txt`:** transformers ≥ 4.51, < 6 (Qwen3 needs 4.51; the Kaggle notebooks already pinned this).
- **Rehearsal:** before pushing, the whole pipeline ran end to end on the Mac with `SMOKE=1` (Qwen2.5-0.5B, 2 steps, 3 questions per eval). Every stage completed, re-runs skipped finished stages, and the report and figures filled every cell. That checks the plumbing only; its numbers are never reported. It found two bugs, both fixed:
  1. The timing run left its throwaway checkpoint in `results/train/timing/_tmp`. For 1.5B that is hundreds of MB, and it got zipped. It now goes to a temp dir that is deleted, and `pack_results.py` skips `.pt`, `.safetensors` and `.bin` in any case.
  2. The overfit gate did not stop the run. `set -e` does not fire inside `a || { gate && b; }`. It is now an explicit `if`, re-tested: a failing overfit log exits 1 before the main run, and `GATE=0` passes.

## 2026-11-15: Python environment

Project venv from Homebrew `python3.12` (`/opt/homebrew/bin/python3.12`). The base anaconda is broken (numpy 2.x / pandas ABI mismatch) and is never used. Installed at M0: torch 2.15.0, transformers 5.23.0, peft 0.21.3, numpy 2.5.5. `model_utils` handles the transformers `torch_dtype`→`dtype` rename and the v5 `activation_checkpointing_enable` rename, so older Colab/Kaggle images still work.
