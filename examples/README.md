# Refmark Examples

These examples are intentionally small and inspectable. They are meant for
researchers and developers who want to poke at the core claims without running
a broad benchmark harness.

Run them from the repository root:

```bash
python examples/citation_qa/run_eval.py
python examples/data_smells/run.py
python examples/judge_free_rewards/run.py
python examples/multidiff_demo/run.py
python examples/pipeline_primitives/run.py
python examples/coverage_alignment/run.py
python examples/docs_navigation_pipeline/run.py
python -m refmark.cli build-index examples/portable_search_index/sample_corpus -o examples/portable_search_index/output/index_local.json
python examples/rag_retrieval_benchmark/run.py
```

## Citation QA

`citation_qa` demonstrates the loop behind cheap citation evaluation:

1. copy a tiny corpus into `output/`
2. inject addressable refs
3. score predicted refs against gold refs with exact/overlap/cover metrics
4. render cited regions as text and HTML for audit

Edit `examples/citation_qa/predictions.json` to see the scores change.

## Data Smells

`data_smells` compares two mock models:

- one predicts completely wrong locations
- one finds the right neighborhood but overcites or undercites

This demonstrates why locate-only metrics are more informative than exact
match alone.

## Judge-Free Rewards

`judge_free_rewards` uses the retained
`refmark_train/data/documentation_full_paragraph_contextual_idf_lean2` dataset
to show deterministic continuous rewards for exact, overbroad, wrong-location,
and missing citation outputs. No LLM judge or API call is used.

## Multidiff Demo

`multidiff_demo` demonstrates bounded same-file edits:

1. copy a source file into `output/`
2. inject live refs
3. apply two successful edits in one payload
4. try one intentionally stale edit and show that the file is unchanged

Edit `examples/multidiff_demo/good_edits.json` to try your own ref-addressed
patches.

## Pipeline Primitives

`pipeline_primitives` demonstrates the plug-and-play document surface:

1. map documents into a JSONL region manifest
2. expand a retrieved region into neighboring context
3. align source document regions to target document regions
4. emit a paste-ready prompt that asks a general chat model for cited region ranges

## Coverage Alignment

`coverage_alignment` generates small `.docx` and `.pdf` input documents and
runs two review flows:

- customer request vs offer/contract
- tender requirements vs technical specification

The example writes marked text, region manifests, naive and expanded coverage
JSON, and an HTML review page that highlights covered items and gaps by stable
Refmark regions.

## Portable Search Index

`portable_search_index` demonstrates the product-shaped retrieval flow:

1. map a folder of docs into Refmark regions
2. enrich each region with local or OpenRouter-generated retrieval metadata
3. write a single portable JSON index
4. search locally with BM25 and return stable region ids plus optional neighbor context

This is the "corpus plus cheap LLM becomes searchable corpus" path. The build
step can use a cheap model once; query-time search needs no API, embeddings,
GPU, vector database, or server.

## Documentation Navigation Pipeline

`docs_navigation_pipeline` is the frozen small end-to-end recipe for software
documentation navigation:

1. map Markdown docs into a region manifest
2. build a local portable index
3. export a browser-search payload
4. evaluate query -> gold-ref examples against the index
5. run a free-text navigation query that returns stable refs and snippets

It intentionally avoids vector databases and runtime model calls. Larger
comparisons against embeddings or hosted retrievers can still be scored through
`eval-index --retriever-results`.

## Browser Page Search

`browser_page_search` demonstrates the smallest browser-facing use case:
semantic find inside the current page. A tiny BM25 payload plus
`refmark/browser_search.js` can power an in-page query box that jumps to
elements with matching `data-refmark-ref` anchors.

## BGB Browser Search

`bgb_browser_search` builds a larger offline demo from the official
Gesetze-im-Internet BGB HTML page. It turns the German Civil Code into a static
browser search app with stable paragraph-level anchors, jump/highlight behavior,
and a small break suite for expected, ambiguous, and out-of-domain queries.

## Static Search Benchmark

`static_search_benchmark` compares Refmark browser BM25 with common static
browser search engines such as MiniSearch, Lunr, and FlexSearch using the same
Refmark-labeled question cache. It reports localization quality together with
latency, index size, licensing, and deployment requirements.

## RAG Retrieval Benchmark

`rag_retrieval_benchmark` compares naive fixed-window chunks with Refmark
regions, deterministic neighbor expansion, train-question-enriched region
retrieval, cached generated retrieval views, and synthetic distractor scaling
over the retained training corpus. It reports hit@k, MRR, token cost, returned
refs, and sample misses.
