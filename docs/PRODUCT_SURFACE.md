# Product Surface

Refmark's product surface is the address space and the evaluation/lifecycle
machinery around it. Search engines, trained routers, browser demos, and
centroid preselectors are consumers of that surface.

## What Refmark Is

- an address layer for source regions and ranges;
- an evidence-evaluation layer for retrieval and citations;
- a lifecycle layer for stale/changed evidence labels;
- a review layer for data smells and adaptation plans;
- a small set of CLI/API entry points that can attach to existing RAG stacks.

## What Refmark Is Not

- not a vector database;
- not a replacement for BM25, embeddings, rerankers, or search services;
- not an answer-quality judge;
- not a guarantee that a model chose the right evidence;
- not a standalone training framework;
- not a polished browser search product;
- not a general coding-agent superiority claim.

## Inventory

| Surface | Main Entry Points | Produces | Use When | Maturity |
| --- | --- | --- | --- | --- |
| Evidence CI | `refmark ci`, `map`, `build-index`, `eval-index`, `compare-index` | manifest, index, eval report, smell report, adaptation plan, comparison report | You want a default local evidence-regression loop over a corpus | Product path |
| Library evaluation | `CorpusMap`, `EvalSuite`, `EvalRun` | evidence metrics, stale examples, comparable run artifacts | You already have a retriever or vector DB and want Refmark metrics | Product path |
| Saved-run comparison | `compare-runs` | cross-run metric deltas and compatibility checks | Retrieval jobs run elsewhere or in separate CI stages | Product path |
| Lifecycle validation | `manifest-diff`, `lifecycle-validate-labels`, `lifecycle-git` | added/removed/changed refs, stale examples, revision reports | A corpus changed and you need to know which eval rows need review | Product path |
| Data smells | `build_data_smell_report`, `eval-index --smell-report-output` | stale labels, hard refs, confusions, query-style gaps, query magnets | You need an actionable review queue instead of only aggregate hit@k | Product path |
| Adaptation planning | `adapt-plan`, `eval-index --adapt-plan-output` | review-required adaptation actions | You want suggestions for metadata, stale-label, range, or gating review | Product path |
| Eval-tool exports | `export_ragas_rows`, `export_deepeval_cases`, `export_trace_events` | dependency-free rows/cases/traces with ref metadata | You want Ragas/DeepEval/Phoenix/Langfuse-style tools to carry evidence refs | Product path |
| Citation scoring | `parse_citation_refs`, metrics helpers, `highlight` | exact/overlap/cover/overcite/undercite scores and snippets | A model returns refs and you need deterministic evidence scoring | Product path |
| Shadow mapping | `Refmarker`, `map --marked-dir`, manifest JSONL | external registry and optional marked copies | You do not want to mutate source files but need addressable regions | Product path |
| MCP edits | `apply_ref_diff`, MCP server tools | bounded same-file region edits | You want agent/code edits addressed by refs instead of drifting line numbers | Bounded product path |
| Discovery maps | `discover`, `discovery-map`, `repair-discovery-clusters` | corpus overview cards/maps and repairable clusters | You need review/navigation context for a corpus before question generation | Early product/review tool |
| Portable search | `build-index`, `search-index`, `export-browser-index` | local BM25 JSON and browser payload | You need a no-infra demo or baseline searchable artifact | Demo/application |
| Training/research | `refmark_train`, BGB scripts, centroid/route experiments | trained or static corpus-local navigation experiments | You are studying new retrieval/navigation strategies using Refmark labels | Research |

## Adoption Shape

The easiest integration path is:

```text
1. map corpus -> manifest
2. attach stable refs to chunks or retrieval hits
3. create/import query -> gold_refs rows
4. run EvalSuite/eval-index
5. inspect smells/adaptation plan
6. rerun after corpus or retriever changes
```

That path works whether the retriever is BM25, embeddings, a vector database,
a reranker, a hosted service, or a research model. Refmark's job is to keep the
evidence target stable and measurable.
