# Training Prototype

`refmark_train` exists in the broader research workspace as an exploratory companion to the main toolkit.

## What It Is

It is a lightweight corpus-local anchor prediction prototype.

The current prototype is useful for:

- testing whether anchor localization is learnable on a fixed corpus
- generating anchored QA-style datasets with train/valid/reformulated splits
- computing judge-free citation rewards for RLHF/DPO-style experiments
- running cheap reruns when a corpus changes
- comparing tiny trainable models against lexical baselines like BM25

## What It Is Not

It is not yet evidence for a broad product claim such as:

- small-model anchor prediction works across arbitrary corpora
- learned anchor prediction beats lexical retrieval generally
- refmark-specific training is already production ready

## Why It Still Matters

It supports a plausible next-stage product/research story:

- stronger model or review pipeline creates anchored supervision
- cheap local model predicts anchors on a refreshed corpus
- deterministic rewards remain mostly automatic and do not require an LLM judge

That path is promising, but still exploratory.

## Recommended Publish Framing

Present `refmark_train` as:

- a prototype
- CPU-friendly
- useful for internal experiments and future corpus-local localization work
- not part of the core stable surface

## Suggested First Entry Point

Use:

- `refmark_train/README.md`
- `examples/judge_free_rewards/run.py`

Treat:

- `refmark_train/research.md`

as exploratory working notes rather than the main onboarding document.
