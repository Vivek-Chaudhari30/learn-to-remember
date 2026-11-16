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
| Raw documents | 746 | 739 |
| Raw tokens total | 183,960 | 179,790 |
| Raw tokens per doc (min / median / max) | 164 / 246 / 385 | 166 / 245 / 361 |
| Notes v0: pages (coverage, truncated, tokens) | 85 (0.36, 25, 13,625) | 85 (0.35, 24, 13,150) |
| Notes v1: pages (coverage, truncated, tokens) | 129 (0.55, 24, 21,893) | 133 (0.55, 23, 21,354) |
| Notes v2: pages (coverage, truncated, tokens) | 178 (0.76, 30, 30,814) | 181 (0.75, 24, 29,732) |
| Notes v3: pages (coverage, truncated, tokens) | 205 (0.87, 23, 35,968) | 210 (0.87, 27, 34,596) |
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

## Oracle policy (M2)

not run

## Model probe (M3)

not run

## Baselines (M3)

not run

## Training runs (M4–M5)

not run
