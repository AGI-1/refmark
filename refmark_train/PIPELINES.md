# Pipelines

This file documents the retained reproducible pipelines in this folder.

All commands below assume your working directory is the parent directory that
contains `refmark_train`.

## 0. Smoke Verification

Before rerunning benchmarks, verify the artifact itself:

```bash
python -m refmark_train.verify_publish_artifact
python -m refmark_train.smoke
```

## 1. Prepare The Documentation Source Set

The retained single-region datasets were built from a local regenerated source
set:

- `refmark_train/source_docs/sets/documentation_set.txt`

Raw and normalized source text is not redistributed in the public artifact.
Use `python -m refmark_train.pull_source_docs` to rebuild local source sets from
the canonical URLs in `source_docs/manifest.json` before running the `prepare`
commands below.

## 2. Randomized Full-Document Baseline

Prepare:

```bash
python -m refmark_train.cli prepare-doc --input-path refmark_train/source_docs/sets/documentation_set.txt --name documentation_full_paragraph_random --unit paragraph --paragraph-window 1 --stride 1 --anchor-limit 1000 --seed 17 --question-mode randomized --train-mutations 2
```

Run:

```bash
python -m refmark_train.cli center-width-ensemble-saved --data-dir refmark_train/data/documentation_full_paragraph_random --seeds 13,21 --epochs 35 --backend torchcpu --mlp-hidden-dim 768 --center-exact-weight 0.75 --center-soft-weight 0.10 --center-width-weight 0.25 --bm25-weight 0.05 --bm25-top-k 5 --bm25-margin 1 --bm25-outside-penalty 0.8 --bm25-enrich-train
```

## 3. Contextual Full-Document Baseline

Prepare:

```bash
python -m refmark_train.cli prepare-doc --input-path refmark_train/source_docs/sets/documentation_set.txt --name documentation_full_paragraph_contextual --unit paragraph --paragraph-window 1 --stride 1 --anchor-limit 1000 --seed 17 --question-mode contextual --train-mutations 2
```

Run:

```bash
python -m refmark_train.cli center-width-ensemble-saved --data-dir refmark_train/data/documentation_full_paragraph_contextual --seeds 13,21 --epochs 35 --backend torchcpu --mlp-hidden-dim 768 --center-exact-weight 0.75 --center-soft-weight 0.10 --center-width-weight 0.25 --bm25-weight 0.05 --bm25-top-k 5 --bm25-margin 1 --bm25-outside-penalty 0.8 --bm25-enrich-train
```

## 4. Current Best Retained Single-Region Pipeline

This is the lean, publish-facing setup we currently want to keep pointing at.

Prepare:

```bash
python -m refmark_train.cli prepare-doc --input-path refmark_train/source_docs/sets/documentation_set.txt --name documentation_full_paragraph_contextual_idf_lean2 --unit paragraph --paragraph-window 1 --stride 1 --anchor-limit 1000 --seed 17 --question-mode contextual --train-mutations 0 --train-questions-per-phrase 2 --valid-questions-per-phrase 2 --reform-questions-per-phrase 4
```

Run:

```bash
python -m refmark_train.cli center-width-ensemble-saved --data-dir refmark_train/data/documentation_full_paragraph_contextual_idf_lean2 --seeds 13,21 --epochs 35 --backend torchcpu --mlp-hidden-dim 768 --center-exact-weight 0.75 --center-soft-weight 0.10 --center-width-weight 0.25 --bm25-weight 0.50 --bm25-top-k 5 --bm25-margin 1 --bm25-outside-penalty 0.8 --bm25-enrich-train
```

Optional diagnostics:

```bash
python -m refmark_train.cli bm25-diagnostics --data-dir refmark_train/data/documentation_full_paragraph_contextual_idf_lean2
```

## 5. Refinement Benchmark

Retained dataset:

- `documentation_refinement_w1_2_broad_v3`
- lineage: generated from the historical `documentation_structure_random` dataset

This refinement benchmark is intentionally retained at top level because it is
part of the publish-facing evidence, but its historical source dataset is not
redistributed in the public artifact.

Run:

```bash
python -m refmark_train.cli refinement-two-model-saved --data-dir refmark_train/data/documentation_refinement_w1_2_broad_v3 --seeds 13,21,34 --epochs 70 --backend torchcpu --mlp-hidden-dim 1024
```

## 6. Cross-Domain Smoke Check

Retained dataset:

- `corporate_refinement_w1_2_broad_smoke`
- lineage: generated from the historical `corporate_structure_random` dataset

Run:

```bash
python -m refmark_train.cli refinement-two-model-saved --data-dir refmark_train/data/corporate_refinement_w1_2_broad_smoke --seeds 13 --epochs 35 --backend torchcpu --mlp-hidden-dim 512
```

## 7. Notes On The Current Generator

The single-document pipeline now includes:

- deterministic contextual question generation
- configurable question counts per phrase
- IDF-aware phrase selection to improve addressability
- train/valid/reform split generation in a single step

These pieces live primarily in:

- [single_doc.py](single_doc.py)
- [experiment.py](experiment.py)

