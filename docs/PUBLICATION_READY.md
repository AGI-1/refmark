# Publication-Ready Scope

This note defines the narrow release shape that is supportable by the current evidence.

## Stable Scope To Publish Now

1. Deterministic locate-only citation evaluation with public data-smell metrics
2. Highlighted source review for cited anchor regions
3. Stable same-file multi-region edits for Python and TypeScript via `apply_ref_diff`
4. Corpus-local anchored QA dataset generation and training prototype through `refmark_train`

## Supported User Journeys

### 1. Exact Citations With Bigger Models

Supported today as a research-backed workflow:

- inject or shadow-mark a corpus
- run the shipped deterministic smoke and examples
- score exact/overlap/cover plus overcite, undercite, breadth, and wrong-location behavior on returned refs
- optionally run second-pass coverage judging

Primary file:

- [CURRENT_BENCHMARK_SNAPSHOT.md](CURRENT_BENCHMARK_SNAPSHOT.md)
- `refmark/metrics.py`

Public verification command:

```bash
python -m refmark.cli smoke
python examples/citation_qa/run_eval.py
python examples/data_smells/run.py
```

Additional evidence that would strengthen this claim:

- a larger repeatability study on one stronger model family
- a denser exact-minimal benchmark (`golden_exact_v1`)
- transfer to one external QA benchmark with pre-mapped anchors

### 2. Syntax Highlighting For Cited Regions

Supported today as a practical audit workflow:

- resolve returned refs back into clean source windows
- render text or HTML review artifacts
- use persistent shadow sessions for unmarked Python/TypeScript files
- present cited snippets to human reviewers without asking them to hunt line numbers manually

Primary files:

- [GETTING_STARTED.md](GETTING_STARTED.md)
- [MCP_USAGE.md](MCP_USAGE.md)
- `refmark/highlight.py`

Additional evidence that would help:

- screenshots or tiny examples in docs
- one notebook or HTML example for QA review

### 3. Pipeline For Anchored QA Bench Creation From Corpus

Supported today, but should be presented as a prototype pipeline rather than a finalized product:

- ingest local or prepared corpora
- create anchored datasets
- export train/valid/reformulated splits
- rerun lightweight localization experiments

Primary files in the broader research workspace:

- `refmark_train/README.md`
- `refmark_train/cli.py`

Public verification command:

```bash
python -m refmark_train.smoke
```

Additional evidence that would strengthen this path:

- one end-to-end corpus refresh demo
- one reviewed exported QA set outside the synthetic path
- one comparison against BM25 on the same corpus-local anchor set

## Not Ready To Publish As A Product Claim

- universal token savings from refmarks
- universal coding improvement across agents and benchmarks
- broad SWE-bench superiority
- small-model citation accuracy improvement as a general result
- training-prototype transfer advantages across corpora

## Tooling Resilience Expectations

For the current published scope, the tools are resilient enough if we stay inside these boundaries:

- Python and TypeScript only for code-edit workflows
- same-file edits only for `apply_ref_diff`
- locate-only citation evaluation rather than free-form answer grading
- highlighted review over returned refs, not arbitrary semantic span inference
- `refmark_train` framed as exploratory and corpus-local

## Recommended Publish Position

Publish `refmark` as:

- a research toolkit with a narrow stable surface
- strong on deterministic locate-only citation evaluation and HiL review
- promising but still experimental on coding-agent multidiff
- exploratory on trainable corpus-local anchor prediction
