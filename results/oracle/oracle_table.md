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
