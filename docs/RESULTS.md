# Results

Measured numbers only. Anything not run is marked "not run". Every number links to the result file it came from.

## Workspace stats (M1)

Measured 2026-11-15 by `scripts/generate_workspace.py` (generator `m1-2026-11-15`). Token counts use the Qwen2.5 tokenizer (`Qwen/Qwen2.5-0.5B-Instruct`, shared with 1.5B). Raw numbers: [`results/workspace_stats.json`](../results/workspace_stats.json). Eval sets: [`data/eval/`](../data/eval/), with sha256 hashes in `data/eval/manifest.json`.

qtype order everywhere: lookup / multihop / aggregate / gap / stale. For single-fact questions, qtype depends on the note version (see DECISIONS.md), so it is shown per version. "Answerable from notes alone" means every fact the question needs is present and fresh in that note version. It is a property of the notes, not a model score.

| | company_a | company_b |
|---|---|---|
| Seed | 0 | 1 |
| Entities (office/team/person/project/client/contract) | 237 (6/15/107/51/22/36) | 243 (6/15/113/51/22/36) |
| Change events | 423 | 422 |
| Raw documents | 748 | 739 |
| Raw tokens total | 181,205 | 177,414 |
| Raw tokens per doc (min / median / max) | 162 / 241 / 375 | 162 / 239 / 338 |
| Notes v0: pages (coverage, truncated, tokens) | 85 (0.36, 25, 13,626) | 85 (0.35, 24, 13,147) |
| Notes v1: pages (coverage, truncated, tokens) | 129 (0.55, 24, 21,894) | 133 (0.55, 23, 21,355) |
| Notes v2: pages (coverage, truncated, tokens) | 178 (0.76, 30, 30,816) | 181 (0.75, 24, 29,729) |
| Notes v3: pages (coverage, truncated, tokens) | 205 (0.87, 23, 35,969) | 210 (0.87, 27, 34,595) |
| Held-out entities | 22 | 0 |
| Train questions | 1714 | 0 |
| Eval questions (held-out-entity Qs) | 293 (48) | 300 (0) |
| train qtype @v3 (lookup/multihop/aggregate/gap/stale) | 1222/158/72/188/74 | 0/0/0/0/0 |
| eval qtype @v3 (lookup/multihop/aggregate/gap/stale) | 60/60/53/60/60 | 60/60/60/60/60 |
| eval qtype @v0 | 21/60/53/129/30 | 21/60/60/130/29 |
| eval qtype @v1 | 36/60/53/101/43 | 38/60/60/98/44 |
| eval qtype @v2 | 64/60/53/57/59 | 61/60/60/65/54 |
| eval qtype @v3 | 60/60/53/60/60 | 60/60/60/60/60 |
| eval answerable from notes alone @v0 | 28/293 | 31/300 |
| eval answerable from notes alone @v1 | 56/293 | 70/300 |
| eval answerable from notes alone @v2 | 116/293 | 110/300 |
| eval answerable from notes alone @v3 | 124/293 | 125/300 |

*Regenerated at M2 (2026-11-15) after the generator changes logged in DECISIONS.md (passing mentions, meeting-note size, roster headers, template wording). The M1 numbers before that were 746/739 docs and 183,960/179,790 tokens.*

## Oracle policy (M2)

Measured 2026-11-15 by `scripts/run_oracle.py` with the Qwen2.5 tokenizer, K = 6 turns (answer included), and a 400-token tool-output cap. Raw results: [`results/oracle/oracle_results.json`](../results/oracle/oracle_results.json).

The oracle is privileged about *what* to read (whether the note is fresh, which raw docs assert the final value). It is not privileged about *finding* it: it uses the same search tools with natural queries, reads through the same caps and turn limit, and its answer must be visible in the text it got back. qtype columns use the labels of the condition's note version (no_notes uses the v3 labels).

| company | condition | accuracy | lookup/multihop/aggregate/gap/stale | failures | mean tokens read | mean turns |
|---|---|---|---|---|---|---|
| company_a | v0 | 291/293 (99.3%) | 21/21 / 58/60 / 53/53 / 129/129 / 30/30 | out_of_turns 2 | 747.3 | 3.51 |
| company_a | v1 | 292/293 (99.7%) | 36/36 / 59/60 / 53/53 / 101/101 / 43/43 | out_of_turns 1 | 725.1 | 3.51 |
| company_a | v2 | 293/293 (100.0%) | 64/64 / 60/60 / 53/53 / 57/57 / 59/59 | none | 679.0 | 3.47 |
| company_a | v3 | 293/293 (100.0%) | 60/60 / 60/60 / 53/53 / 60/60 / 60/60 | none | 682.0 | 3.47 |
| company_a | no_notes | 291/293 (99.3%) | 60/60 / 58/60 / 53/53 / 60/60 / 60/60 | out_of_turns 2 | 783.9 | 3.52 |
| company_b | v0 | 298/300 (99.3%) | 21/21 / 59/60 / 60/60 / 129/130 / 29/29 | out_of_turns 1, file_not_retrieved 1 | 740.2 | 3.5 |
| company_b | v1 | 298/300 (99.3%) | 38/38 / 59/60 / 60/60 / 97/98 / 44/44 | out_of_turns 1, file_not_retrieved 1 | 717.0 | 3.49 |
| company_b | v2 | 298/300 (99.3%) | 61/61 / 59/60 / 60/60 / 64/65 / 54/54 | out_of_turns 1, file_not_retrieved 1 | 701.4 | 3.49 |
| company_b | v3 | 298/300 (99.3%) | 60/60 / 59/60 / 60/60 / 60/60 / 59/60 | out_of_turns 1, file_not_retrieved 1 | 692.5 | 3.49 |
| company_b | no_notes | 296/300 (98.7%) | 59/60 / 58/60 / 60/60 / 60/60 / 59/60 | file_not_retrieved 3, out_of_turns 1 | 780.0 | 3.5 |

Residual failures: 6 distinct questions across all 10 conditions (1–4 per condition, at most 1.3%):

| question | reason | conditions where it fails |
|---|---|---|
| A-0073 "Which office is the team behind Project Beryl based in?" | out_of_turns | v0, no_notes |
| A-0087 "Which office is the team behind Project North Falcon based in?" | out_of_turns | v0, v1, no_notes |
| B-0076 "What is the job title of Leo Molina's manager?" | out_of_turns | all |
| B-0276 "By what date is Project Night Vega scheduled to be done?" | file_not_retrieved | all |
| B-0012 "What budget is Project Nebula currently working with?" | file_not_retrieved | no_notes |
| B-0067 "How much budget does the project linked to contract C-3952 have?" | file_not_retrieved | no_notes |

- `out_of_turns`: two-hop questions answered from raw files where both hops needed a fallback query. 2 × (search, search, read) + answer = 7 turns > K = 6.
- `file_not_retrieved`: the newest doc asserting the value stayed outside the top 10 for all the oracle's queries, including the `since` fallback. Night Vega has three sibling codenames (Vega, Silver Vega, Red Vega); the other two have none, so the cause there is plain BM25 ranking among many docs about the same project.

These are left in rather than tuned away. Tuning the env further against the oracle's own queries would overfit it. At the measured rate, they cap achievable accuracy at about 98.7–100%.

## Model probe (M3)

not run

## Baselines (M3)

not run

## Training runs (M4–M5)

not run
