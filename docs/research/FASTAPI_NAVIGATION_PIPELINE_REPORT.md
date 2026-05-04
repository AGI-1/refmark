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
- Style-aware clean output and fixed comparison suite:
  `examples/portable_search_index/output/fastapi_pipeline_styleaware_clean`

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

## Style-Aware Question Planning

A later run made question generation explicit before calling the model. Each
selected non-excluded region received one planned `direct`, one `concern`, and
one `adversarial` query request. The run used:

- Config: `examples/portable_search_index/fastapi_pipeline.styleaware.yaml`
- Output: `examples/portable_search_index/output/fastapi_pipeline_styleaware_clean`
- Corpus: 689 regions, 150 docs, about 309k source tokens.
- Discovery: windowed local discovery, 8 region-safe windows.
- Default-excluded refs: 135 release-note/query-magnet regions skipped from
  question planning.
- Generated eval rows: 336, balanced as 112 per style.
- Question generation spend: about `$0.0082` estimated.

The first style-aware run exposed a useful bug: release notes were excluded by
the search index, but local discovery had not marked every release-note region
as excluded from question planning. That created gold refs the default retriever
was intentionally not allowed to return. Discovery now imports the same
query-magnet/default-search-exclusion role policy used by the index.

Clean style-aware rerank results:

| Query style | Questions | Region hit@1 | Region hit@10 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Direct | 112 | 0.589 | 0.911 | 0.708 |
| Concern | 112 | 0.518 | 0.821 | 0.618 |
| Adversarial | 112 | 0.330 | 0.670 | 0.453 |
| Overall | 336 | 0.479 | 0.801 | 0.593 |

Interpretation: direct lexical lookup is mostly solved by the enriched BM25
stack, concern-style user wording remains decent, and lower-overlap adversarial
paraphrases are the main weakness. The style split is therefore not cosmetic:
it tells the adaptation loop whether to add metadata/query rewrites, improve
range boundaries, or compare a semantic embedding/hybrid teacher.

Weak zones in the clean run included:

| Area | Rows | Region hit@10 | Observed smell |
| --- | ---: | ---: | --- |
| `reference/parameters` | 3 | 0.000 | broad parameter-class queries confuse nearby reference pages |
| `index` | 6 | 0.167 | overview/index regions are weak gold targets and often better treated as parent/article navigation |
| `advanced/sub-applications` | 3 | 0.333 | API-docs wording competes with OpenAPI/reference pages |
| `tutorial/body/multiple-params` | 6 | 0.333 | adjacent same-article snippets compete; range/neighbor scoring likely better than exact single-ref |
| `tutorial/index` | 6 | 0.333 | generic onboarding wording attracts feature/history/deployment pages |
| `advanced/python-types` | 6 | 0.500 | concern wording maps to general Python types instead of advanced type pages |

The fixed style-aware comparison uses the exported `eval_questions.jsonl`
directly: 336 rows, balanced as 112 direct, 112 concern, and 112 adversarial
questions. It compares lexical, embedding, and hybrid approaches over the same
corpus and refs.

| Method | Region hit@1 | Region hit@10 | Article hit@1 | Article hit@10 | Region MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Local BM25 | 0.372 | 0.691 | 0.616 | 0.845 | 0.478 |
| Refmark-enriched BM25 | 0.497 | 0.821 | 0.694 | 0.926 | 0.607 |
| Local hashed embedding | 0.241 | 0.542 | 0.381 | 0.747 | 0.331 |
| Refmark hashed embedding | 0.319 | 0.658 | 0.485 | 0.777 | 0.431 |
| Local Qwen3 embedding | 0.563 | 0.929 | 0.747 | 0.982 | 0.694 |
| Refmark Qwen3 embedding | 0.607 | 0.946 | 0.792 | 0.976 | 0.727 |
| Refmark BM25 + Qwen3 embedding, weight 0.10 | 0.625 | 0.961 | 0.816 | 0.985 | 0.741 |
| Refmark BM25 + Qwen3 embedding, weight 0.25 | 0.607 | 0.949 | 0.804 | 0.985 | 0.727 |

The Qwen3 comparison warms query embeddings before timing. The latency fields in
the JSON are useful for relative local CPU work, not provider benchmarking:
cached embedding search itself is sub-millisecond in-process, while hybrid
evaluation also runs BM25 candidate scoring.

Style split for the strongest fixed-suite hybrid:

| Query style | Region hit@1 | Region hit@10 | MRR |
| --- | ---: | ---: | ---: |
| Direct | 0.634 | 0.964 | 0.748 |
| Concern | 0.679 | 0.964 | 0.778 |
| Adversarial | 0.563 | 0.955 | 0.696 |

## Interpretation

Refmark-enriched retrieval metadata produced the largest no-embedding gain:
region hit@10 improved from 0.691 to 0.821 in the fixed style-aware
comparison, and from 0.773 to 0.902 in the earlier full generated-question
eval.

Real semantic embeddings helped article-level navigation substantially. For
this corpus, embeddings alone were already strong at article hit@10, but the
best region-level results came from combining Refmark-enriched metadata with
Qwen3 embeddings. The best fixed-suite hybrid reached 0.961 region hit@10 and
0.985 article hit@10, including 0.955 region hit@10 on adversarial questions.

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
- Preserve both full-pipeline and exported-eval-JSONL comparison modes
  explicitly in reports.
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
