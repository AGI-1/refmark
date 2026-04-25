# Refmark Train

`refmark_train` is a compact, CPU-friendly experiment package for testing
whether a tiny model can learn to map a question directly to a corpus-local
`refmark` citation anchor.

This folder is curated as a proof-of-concept artifact. It keeps the current
pipeline code, representative derived datasets, and retained run artifacts that
support the present conclusions. Older exploratory assets and raw source
payloads are intentionally not redistributed.

The publish-facing emphasis is on the extracted QA/anchor datasets. Source
manifests remain available for lineage and regeneration, while raw and
normalized upstream source payloads are excluded from the public artifact.

## What This Folder Supports

This package gives a strong signal for a narrow but useful claim:

> Refmarks are a practical train/eval primitive for corpus-local citation
> localization, and a tiny model can learn this mapping on a real anchored
> corpus with cheap deterministic evaluation.

What the current package supports well:

- single-region citation localization on a fixed anchored corpus
- judge-free reward computation for models that emit refmark ids
- cheap reruns when question generation or model settings change
- direct comparison between randomized, contextual, and addressability-aware
  question generation
- hybridization between a tiny learned model and enriched BM25

What it does not claim yet:

- superiority over classical retrieval in general
- robust multi-range citation prediction across large corpora
- strong cross-domain generalization

## Current Publishable Signal

On the retained documentation benchmark:

- corpus size: about `52.7k` source words
- anchors: `955`
- target task: mostly single-anchor citation retrieval

Representative progression:

1. randomized questions: `0.442 / 0.372` valid/reform exact hit
2. contextual questions: `0.818 / 0.732`
3. lean contextual + IDF phrase selection + stronger BM25 prior:
   `0.845 / 0.800`

The important result is not just the final number. The cleaner signal is:

- better anchor/question addressability mattered more than bigger models
- we improved while reducing train questions from `13,370` to `3,820`
- deterministic overlap, cover, and breadth metrics made those changes cheap
  to evaluate

See [RESULTS.md](RESULTS.md) for the exact retained
benchmark table.
See [PUBLISH_REVIEW.md](PUBLISH_REVIEW.md) for
the current readiness review and remaining gaps.
See [SOURCE_PROVENANCE.md](SOURCE_PROVENANCE.md)
for retained-source provenance and usage notes.

## Folder Layout

- [cli.py](cli.py)
  Command-line entry points for dataset prep, diagnostics, and training.
- [single_doc.py](single_doc.py)
  Single-document anchoring and deterministic question generation.
- [experiment.py](experiment.py)
  Main training/evaluation loops and report formatting.
- [diagnostics.py](diagnostics.py)
  BM25 diagnostics and retrieval-side analysis.
- [source_docs](source_docs)
  Source manifests and category manifests used to describe retained dataset
  lineage. Raw and normalized upstream source payloads are not redistributed.
- [data](data)
  Curated retained datasets only.
- [runs](runs)
  Curated retained run artifacts only.

## Retained Datasets

- `documentation_full_paragraph_random`
  Randomized full-document baseline.
- `documentation_full_paragraph_contextual`
  Contextual question-generation baseline.
- `documentation_full_paragraph_contextual_idf_lean2`
  Current strongest single-region benchmark configuration.
- `documentation_refinement_w1_2_broad_v3`
  Two-stage broad-to-precise refinement benchmark.
- `corporate_refinement_w1_2_broad_smoke`
  Small cross-domain smoke check showing the approach is not yet universally
  robust.

## Quick Start

From the parent directory that contains `refmark_train`:

```bash
python -m refmark_train.verify_publish_artifact
python -m refmark_train.smoke
python -m refmark_train.cli center-width-ensemble-saved --data-dir refmark_train/data/documentation_full_paragraph_contextual_idf_lean2 --seeds 13,21 --epochs 35 --backend torchcpu --mlp-hidden-dim 768 --center-exact-weight 0.75 --center-soft-weight 0.10 --center-width-weight 0.25 --bm25-weight 0.50 --bm25-top-k 5 --bm25-margin 1 --bm25-outside-penalty 0.8 --bm25-enrich-train
```

The first command checks that the retained manifests, datasets, runs, and CLI
entry points resolve inside the artifact. The second trains a tiny synthetic
model and verifies the training path without network or external corpora. The
third reruns the current strongest retained single-region configuration.

For the exact prep and run commands for each retained benchmark, see
[PIPELINES.md](PIPELINES.md).

## Judge-Free Reward Use

The retained `documentation_full_paragraph_contextual_idf_lean2` dataset can be
used as a small RLHF/DPO-style reward sandbox: model outputs are refmark ids,
gold labels are refmark ids, and rewards are computed with deterministic
metrics such as cover, breadth ratio, undercite, and overcite. No LLM judge is
needed for the citation-location reward.

Run:

```bash
python examples/judge_free_rewards/run.py
```

## Hardware Expectations

This package was developed to run on a weak local Windows machine.

- CPU-only runs are fine
- DirectML is optional for Windows AMD hardware
- the retained single-region benchmark runs in minutes, not hours

## Notes On Curation

This folder intentionally does not expose every exploratory branch at top
level. The goal is reproducibility and readability, not preserving every local
decision in the first screenful of files. Historical raw/source payloads are
excluded from this public artifact. Rebuild them from canonical URLs with
[pull_source_docs.py](pull_source_docs.py) if your redistribution policy allows it.

