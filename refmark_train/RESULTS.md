# Results

This file records the retained benchmark results that the current folder is
meant to support.

## Single-Region Localization

Corpus:

- source set: `documentation_set.txt`
- source size: about `52.7k` words
- anchors: `955`
- task: predict the correct citation anchor for mostly single-anchor targets

Retained run artifacts:

- randomized baseline:
  [center_width_ensemble_20260424T102842Z_documentation_full_paragraph_random_torchcpu.json](runs/center_width_ensemble_20260424T102842Z_documentation_full_paragraph_random_torchcpu.json)
- contextual baseline:
  [center_width_ensemble_20260424T105016Z_documentation_full_paragraph_contextual_torchcpu.json](runs/center_width_ensemble_20260424T105016Z_documentation_full_paragraph_contextual_torchcpu.json)
- current best retained run:
  [center_width_ensemble_20260424T115745Z_documentation_full_paragraph_contextual_idf_lean2_torchcpu.json](runs/center_width_ensemble_20260424T115745Z_documentation_full_paragraph_contextual_idf_lean2_torchcpu.json)

| Dataset | Train Qs | Setup | Valid exact | Reform exact | Notes |
| --- | ---: | --- | ---: | ---: | --- |
| `documentation_full_paragraph_random` | 11,460 | randomized questions, light BM25 prior | 0.442 | 0.372 | weak addressability baseline |
| `documentation_full_paragraph_contextual` | 13,370 | contextual questions, light BM25 prior | 0.818 | 0.732 | large gain from better question generation |
| `documentation_full_paragraph_contextual_idf_lean2` | 3,820 | contextual + IDF phrase selection + stronger BM25 prior | 0.845 | 0.800 | best retained single-region result |

Key observations:

- the quality of question generation mattered more than raw question volume
- the best retained run used about `29%` of the training questions of the
  earlier contextual run
- train exact on the lean retained run is about `0.964`, which suggests the
  main bottleneck is reformulation/addressability rather than inability to fit
  the train set

## Refinement Loop

Retained run artifact:

- [refinement_two_model_20260424T083449Z_documentation_refinement_w1_2_broad_v3_torchcpu.json](runs/refinement_two_model_20260424T083449Z_documentation_refinement_w1_2_broad_v3_torchcpu.json)

Dataset:

- `documentation_refinement_w1_2_broad_v3`
- anchors: `239`
- train examples: `2,080`
- benchmark shape: broad-stage retrieval followed by precise refinement

Representative retained numbers:

- valid hybrid local-cross precise stage: `0.819`
- reform hybrid local-cross precise stage: `0.613`
- default predicted-broad loop precise stage: `0.756 / 0.450` valid/reform

Interpretation:

- refinement is useful when the broad region is already good
- reform remains bottlenecked by first-pass broad retrieval
- this is promising, but less mature than the single-region benchmark
- lineage is explicit in the dataset manifest and names the historical
  structure-random source dataset, which is not redistributed here

## Cross-Domain Smoke Check

Retained run artifact:

- [refinement_two_model_20260424T081958Z_corporate_refinement_w1_2_broad_smoke_torchcpu.json](runs/refinement_two_model_20260424T081958Z_corporate_refinement_w1_2_broad_smoke_torchcpu.json)

Dataset:

- `corporate_refinement_w1_2_broad_smoke`
- anchors: `399`
- train examples: `780`

Representative retained numbers:

- valid predicted-broad loop precise stage: `0.367`
- reform predicted-broad loop precise stage: `0.233`

Interpretation:

- the method is not yet robust across domains
- this retained smoke check is useful because it prevents overclaiming
- the benchmark is intentionally retained as a limit-check, while its
  historical source anchor dataset is not redistributed here

