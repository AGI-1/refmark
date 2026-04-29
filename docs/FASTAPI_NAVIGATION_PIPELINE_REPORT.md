# FastAPI Navigation Pipeline Report

This report captures a practical English documentation run for the Refmark
evidence-retrieval pipeline. The goal was to test whether a corpus can be
turned into a searchable, measurable evidence space, then compare local lexical
search, Refmark-enriched retrieval metadata, and optional embeddings.

## Corpus

- Source: retained FastAPI documentation snapshot under
  `examples/portable_search_index/output/online_corpora/fastapi_docs`.
- Approximate source size: 309k tokens.
- Documents: 150.
- Refmark regions: 689.
- Local generated evaluation questions: 480.
- OpenRouter/Qwen generated question cache rows: 160 target regions, 480
  question rows. Some comparison scripts dedupe to one active question per
  target/cache key, so they report 160 evaluated questions.

## Pipeline Artifacts

- Local pipeline config:
  `examples/portable_search_index/fastapi_pipeline.local.yaml`
- Remote/enriched config:
  `examples/portable_search_index/fastapi_pipeline.qwen_mistral.yaml`
- Local pipeline output:
  `examples/portable_search_index/output/fastapi_pipeline`
- Qwen/Mistral output and caches:
  `examples/portable_search_index/output/fastapi_pipeline_qwen_mistral`

Generated output directories are ignored and should not be committed.

## Token And Cost Notes

Question generation used `qwen/qwen-turbo` for 160 target regions:

- Input tokens: 67,789.
- Output tokens: 8,967.
- Estimated cost: about $0.0052.
- Provider errors: 0.

Embedding comparisons used two document-vector sets, one over raw/local index
text and one over Refmark-enriched index text, plus warmed query vectors:

- Approximate document/query input tokens embedded: 637k.
- `openai/text-embedding-3-small`: 1,536 dimensions.
- `qwen/qwen3-embedding-8b`: 4,096 dimensions.

The exact embedding spend is provider-priced outside the report, but at the
listed OpenRouter prices this is still in the cents-or-less range for this
corpus.

## Evaluation Results

The 480-row `eval-index` run uses all generated question rows and reports region
hit@10 as `hit_at_k`.

| Method | Questions | Region hit@1 | Region hit@10 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Local Refmark/BM25 rerank | 480 | 0.469 | 0.773 | 0.561 |
| Enriched Refmark/BM25 rerank | 480 | 0.598 | 0.902 | 0.706 |

The 160-question comparison run dedupes the generated question cache to one
active question per target. It compares lexical, embedding, and hybrid
approaches over the same corpus and refs.

| Method | Region hit@1 | Region hit@10 | Article hit@1 | Article hit@10 | Region MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Local BM25 | 0.356 | 0.694 | 0.606 | 0.894 | 0.460 |
| Refmark-enriched BM25 | 0.506 | 0.856 | 0.756 | 0.963 | 0.631 |
| Local hashed embedding | 0.213 | 0.500 | 0.325 | 0.694 | 0.290 |
| Refmark hashed embedding | 0.319 | 0.644 | 0.544 | 0.850 | 0.411 |
| Local OpenAI small embedding | 0.450 | 0.881 | 0.750 | 0.981 | 0.584 |
| Refmark OpenAI small embedding | 0.463 | 0.888 | 0.788 | 0.981 | 0.606 |
| Refmark BM25 + OpenAI small embedding | 0.563 | 0.888 | 0.800 | 0.969 | 0.670 |
| Local Qwen3 embedding | 0.469 | 0.869 | 0.769 | 0.988 | 0.616 |
| Refmark Qwen3 embedding | 0.550 | 0.900 | 0.806 | 0.988 | 0.668 |
| Refmark BM25 + Qwen3 embedding, best weight | 0.588 | 0.894 | 0.813 | 0.981 | 0.694 |

## Interpretation

Refmark-enriched retrieval metadata produced the largest no-embedding gain:
region hit@10 improved from 0.694 to 0.856 in the deduped comparison, and from
0.773 to 0.902 in the full generated-question eval.

Real semantic embeddings helped article-level navigation substantially. For
this corpus, embeddings alone were already strong at article hit@10, but the
best region-level results came from combining Refmark-enriched metadata with
Qwen3 embeddings.

Hashed embeddings were useful as a local sanity baseline, but they did not beat
the enriched lexical index. They are not a substitute for semantic embeddings
on this corpus.

The strongest current claim from this run is not that Refmark replaces BM25 or
embeddings. The claim is that Refmark gives all of them the same measurable
target: recover the correct evidence region or article, then inspect misses and
hard zones with stable refs.

The heatmap surfaced a distinct data-smell class: release-note/changelog-style
ledger regions are query magnets. They are large, repetitive, and often contain
many shallow mentions of unrelated topics. Refmark now marks those regions with
roles such as `ledger`, `query_magnet`, and `exclude_from_default_search`. They
remain visible in manifests/heatmaps and can be searched with an explicit
`--include-excluded` opt-in, but default search can avoid treating them as
primary evidence.

## Heatmap And Adaptation Workbench

The generated FastAPI heatmap evolved from a static report into a small
evidence workbench:

- blocks are grouped by the existing documentation hierarchy, with small
  singleton branches folded into `unclustered`/`other` buckets for readability;
- color shows retrieval strength for the selected mode, while selected-block
  details show the stable range, metrics, roles, and the generated eval
  questions that target that block;
- a search field highlights all matching regions by title, doc label, stable
  ref, or embedded eval question;
- global run summary, largest excluded areas, and weakest covered areas live in
  an overview panel instead of crowding the block inspector;
- non-adaptive BM25/embedding modes remain available beside adaptive metadata
  modes, so improvements and regressions stay visible.

The matching adaptation loop is intentionally conservative. A reviewer model
diagnoses weak blocks and can recommend question rewrites, alternate valid
refs, range extension/splitting, exclusion/query-magnet roles, confusion
mapping, or retriever tuning. Only safe question/metadata changes are applied
automatically, and affected-row mini-eval runs before a full refreshed report.

That makes the heatmap useful beyond this FastAPI demo: it is the visual
control surface for Refmark's evidence-CI loop. The goal is not just "better
search", but inspectable corpus maintenance: which regions are weak, why they
are weak, which adaptation was applied, and whether the same held-out evidence
metrics improved afterward.

## Search Usage

After a pipeline run, a user can query the processed corpus directly:

```bash
python -m refmark.cli query-pipeline examples/portable_search_index/output/fastapi_pipeline "How do I configure CORS for browser clients?"
```

The response returns ranked regions, snippets, context refs, and containing
section metadata when available.

To include query-magnet/ledger regions such as release notes:

```bash
python -m refmark.cli query-pipeline examples/portable_search_index/output/fastapi_pipeline "What changed in the latest release?" --include-excluded
```

## Follow-Ups

- Make the remote retrieval-view stage fully resumable and cache each completed
  row immediately.
- Add first-class embedding configuration to `run-pipeline`, not only to the
  comparison script.
- Preserve both 480-row and deduped-160 evaluation modes explicitly in reports.
- Add article-level and region-level metrics to the default pipeline summary.
- Generalize the FastAPI heatmap shell into a reproducible renderer instead of
  treating the generated HTML as a hand-polished artifact.
- Extend heatmap output to show repeated hard refs, near-misses, ambiguous
  neighboring regions, and release-note-like low-value/noisy zones.
- Promote noisy-region roles into configurable corpus policy rather than relying
  only on built-in release-note/changelog heuristics.
- Add richer ingestion for folders containing extracted PDF/DOCX/HTML sources,
  with clear extracted-text provenance boundaries before promising layout-level
  citation.
