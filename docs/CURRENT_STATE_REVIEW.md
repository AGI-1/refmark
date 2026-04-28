# Current State Review

This is a handoff snapshot for continuing work after context compression.

## Product Claim

Best current framing:

> Refmark creates a stable addressable evidence space for AI systems. Model
> guesses become measurable because citations, edits, retrieval hits, training
> labels, and review targets all resolve to concrete refs/ranges.

The public wedge is still:

> Turn your corpus into a regression test suite for retrieval.

That is stronger and safer than claiming Refmark is a search engine, training
method, or MCP editor. Those are applications of the address space.

The image/tagline can reasonably shift toward:

```text
Stable addressable evidence space for AI models.
Cheap eval. Exact citation targets.
MCP for stable multi-diff file edits.
Model guesses become measurable.
```

## Tool Surface

### Core Marking And Resolution

| Surface | State | Value |
| --- | --- | --- |
| `inject` / `strip` | stable core | Add/remove visible markers in supported text/code formats. |
| `highlight` | stable core | Resolve refs/ranges to reviewable text/HTML/JSON snippets. |
| `Refmarker` / registry | useful, needs more docs | Pass-through library interface for refmarking data and keeping shadow registries. |
| citation parser | stable core | Strictly parses refs/ranges; keeps inclusive citation ranges explicit. |

### Corpus Pipeline And RAG Evaluation

| Surface | State | Value |
| --- | --- | --- |
| `map` | stable enough | Builds JSONL region manifests for files/directories. |
| `expand` | stable enough | Expands refs to neighbors/parents for context packing. |
| `pack-context` | stable enough | Produces deterministic evidence bundles for prompts/review. |
| `question-prompt` | stable enough | Creates overridable prompts for query/gold-ref generation. |
| `discover` | early | Builds corpus discovery manifests; local mode is deterministic, model-backed discovery is experimental. |
| `build-index` / `search-index` | usable example/product wedge | Builds local BM25 JSON indexes over source plus generated retrieval views. |
| `eval-index` | important product surface | Self-checking eval reports with provenance hashes, stale refs, heatmap/confusion diagnostics, confidence gates, and adaptation recommendations. |
| `CorpusMap` / `EvalSuite` / `EvalRun` | stable enough | Python API for `query -> gold refs/ranges` evaluation and retriever comparison. |
| `provenance.py` | stable enough | Hashes index/examples/settings so eval artifacts can self-check. |

### MCP And Editing

| Surface | State | Value |
| --- | --- | --- |
| `apply_ref_diff` | stable for bounded same-file edits | Applies replace/delete/insert/patch edits addressed by refs. |
| MCP server | useful, needs continued hardening | Exposes `read_refmarked_file`, `list_ref_regions`, `apply_ref_diff`. |
| WSL Codex check docs | current | Use Linux nvm Codex CLI with `CODEX_HOME=/home/alkon/.codex-refmark`; avoid WindowsApps binary. |

MCP is a real application, but not the main product story. Frame it as:

> Once code is addressable, multi-diff edits can target stable regions instead
> of drifting line numbers.

### Document/Demo Utilities

| Surface | State | Value |
| --- | --- | --- |
| `align` / document mapping | demo/experimental | Extracted-text alignment for request-vs-contract style examples. |
| PDF/DOCX | extracted-text only | Do not claim original-layout citation unless page/box provenance is added. |
| browser search export | example | Exports a compact local index for page/document navigation demos. |
| BGB demo scripts | research/demo | Strong evidence-retrieval playground; not the package core. |

## Current Training And Retrieval Findings

The training work is valuable mostly because Refmark makes it measurable. It
does not yet prove that tiny local models replace embeddings.

### Strong Results

| Experiment | Current evidence |
| --- | --- |
| Refmark-enriched Qwen3 embeddings | Full BGB fixed benchmark hit@10 `0.9888`; randomized 200k stress cycles around `0.9174-0.9614` hit@10. |
| Cached query embedding -> article classifier | 3-cycle held-out hit@10 `0.9377`, hit@50 `0.9819`; strong if runtime embeddings already exist. |
| Static generated views / concern aliases | Large gains on curated concern suite; useful for no-runtime-infra docs search. |
| Deterministic/confusion signatures | Small global gain, strong targeted repairs; best adapted index + teacher blend hit@10 `0.6389` on fair split. |
| Oracle query reformulation on 10-article slice | Raw BM25 hit@10 `0.6207`; learned train-derived term bank `0.6897`; per-query oracle `0.8966`. Shows real expansion ceiling. |

### Negative Or Limited Results

| Experiment | Finding |
| --- | --- |
| Direct text -> article classifier | Failed: Gemma 200k hit@10 `0.0395`. |
| Text -> Qwen3/article vector distillation | Improved with more data but still below BM25: best larger CPU run hit@10 `0.3196`, hit@50 `0.5552`. |
| Global query reformulator | Naive append learned generic/legal magnet terms and hurt BM25; fused side-channel was only barely positive. |
| Fielded BM25/RRF | Did not beat combined article views in tested BGB split. |
| Aggressive signature gating | Repairs hard rows but can damage global quality; needs confidence/confusion-aware triggers. |

### Latest Reformulation Runs

Scripts:

- `examples/bgb_browser_search/train_bgb_query_reformulator.py`
- `examples/bgb_browser_search/iterate_bgb_oracle_reformulation.py`
- `examples/bgb_browser_search/train_bgb_oracle_reformulation_predictor.py`

Artifacts:

- `examples/bgb_browser_search/output_full_qwen_turbo/bgb_query_reformulator_3cycle_small_report.json`
- `examples/bgb_browser_search/output_full_qwen_turbo/bgb_query_reformulator_3cycle_discriminative_report.json`
- `examples/bgb_browser_search/output_full_qwen_turbo/bgb_query_reformulator_fusion_probe_report.json`
- `examples/bgb_browser_search/output_full_qwen_turbo/bgb_oracle_reformulation_10articles_10iters.json`
- `examples/bgb_browser_search/output_full_qwen_turbo/bgb_oracle_reformulation_predictor_10articles_report.json`

Key numbers:

| Run | Result |
| --- | --- |
| naive global reformulator | append hit@10 `0.5042` vs BM25 `0.5900`; bad generic magnets. |
| discriminative global reformulator | append hit@10 `0.5578`; fusion `0.5917`; still weak. |
| 10-article learned term bank | hit@10 `0.6897` vs raw `0.6207`, no worsened rows. |
| 10-article per-query oracle | hit@10 `0.8966`, hit@1 `0.8276`; high ceiling. |
| 10-article oracle-term predictor | 129k params, 0.54 MB; predicted append hit@10 `0.6897`, MRR `0.5969`; best fusion hit@10 `0.6552`. |

Interpretation:

- Query expansion can improve BM25 when terms are locally relevant.
- Global unconstrained expansion learns magnet terms and can hurt.
- The promising architecture is an ensemble:

```text
query
 -> coarse surface model: likely articles/sections
 -> local term/reformulation predictor constrained to that surface
 -> BM25/reranker over narrowed candidates
 -> Refmark eval/heatmap/adapt loop
```

The next model experiment should be surface-conditioned, not a larger global
reformulator.

## What Refmark Contributes To These Experiments

Refmark is the address/evaluation substrate:

- every query has concrete gold refs/ranges;
- every retrieval variant is scored against the same targets;
- failures localize to hard refs, wrong-top refs, languages, styles, and target
  shapes;
- corpus drift can mark examples stale by source hash;
- output is reviewable because hits jump back to highlighted source.

This is why even negative training results are useful: they become structured
evidence, not vibes.

## Recommended Next Steps

1. Update README/image wording toward "stable addressable evidence space" and
   "model guesses become measurable".
2. Keep `eval-index` and `EvalSuite` as the product center.
3. Promote the adapt-loop docs: heatmap -> adapt -> re-evaluate.
4. For training, test surface-conditioned reformulation:
   - pick or train a coarse article/section router;
   - restrict expansion vocabulary to that surface;
   - evaluate append, fusion, and reranker-feature modes.
5. Optimize oracle-label generation before scaling:
   - cache `query + term -> rank`;
   - parallelize across queries/articles;
   - only run full oracle evaluation at checkpoints.
6. Keep generated outputs ignored unless a small report is intentionally
   published.

## Verification Snapshot

Before the latest reformulation scripts, the repo was pushed with:

- full pytest: `157 passed`;
- pushed commits:
  - `aaecf75` `Add evidence retrieval evaluation pipeline`
  - `6104046` `Record BGB embedding distillation result`

The latest reformulation scripts compile locally but are not committed yet.
