# Refmark Examples

These examples are intentionally small and inspectable. They are meant for
researchers and developers who want to poke at the core claims without running
a broad benchmark harness.

For the canonical product-shaped path, run
`python examples/docs_navigation_pipeline/run.py` or follow
[Evidence CI Quickstart](../docs/QUICKSTART_EVIDENCE_CI.md).

Run them from the repository root:

```bash
python examples/citation_qa/run_eval.py
python examples/data_smells/run.py
python examples/eval_tool_integrations_demo/run.py
python examples/library_integration_demo/run.py
python examples/ragas_refmark_mutation_demo/run.py
python examples/lifecycle_tool_integrations_demo/run.py
python examples/judge_free_rewards/run.py
python examples/lifecycle_ci_demo/run.py
python examples/multidiff_demo/run.py
python examples/pipeline_primitives/run.py
python examples/coverage_alignment/run.py
python examples/docs_navigation_pipeline/run.py
python examples/evidence_lifecycle_benchmark/evaluate_ref_stability_mutations.py --dataset beir/scifact/test
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

## Library Integration

`library_integration_demo` shows the attach point for an existing retrieval
system. A normal Python callback returns stable refs or hit dictionaries, then
`EvalSuite.evaluate(...)` produces evidence metrics, data smells, an adaptation
plan, and a comparable run artifact.

This is the path for teams that already have BM25, embeddings, a vector DB, or
hosted retrieval service and only want Refmark as the evidence-evaluation layer.

## Data Smells

`data_smells` compares two mock models:

- one predicts completely wrong locations
- one finds the right neighborhood but overcites or undercites

This demonstrates why locate-only metrics are more informative than exact
match alone.

The same example also writes a first-class retrieval smell report with
`build_data_smell_report(...)`. That second report demonstrates stale labels,
hard refs, confusion pairs, query-style gaps, over/undercitation, low-confidence
hits, and query magnets as reviewable adaptation inputs. It also writes an
`adaptation_plan.json` artifact, equivalent to running `refmark adapt-plan` on
the smell report.

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

`portable_search_index` demonstrates a retrieval artifact built on the
evidence-eval substrate:

1. map a folder of docs into Refmark regions
2. enrich each region with local or OpenRouter-generated retrieval metadata
3. write a single portable JSON index
4. search locally with BM25 and return stable region ids plus optional neighbor context

This is the "corpus plus cheap LLM becomes searchable corpus" demo path. The
build step can use a cheap model once; query-time search needs no API,
embeddings, GPU, vector database, or server. It is useful as a baseline and
application example, not as a separate product promise.

## Documentation Navigation Pipeline

`docs_navigation_pipeline` is the frozen small end-to-end recipe for evidence
CI over software documentation:

1. map Markdown docs into a region manifest
2. build a local portable index
3. export a browser-search payload
4. evaluate query -> gold-ref examples against the index
5. write a data-smell report and adaptation plan for the evaluated run
6. run a free-text query that returns stable refs and snippets

It intentionally avoids vector databases and runtime model calls. Larger
comparisons against embeddings or hosted retrievers can still be scored through
`eval-index --retriever-results`.

## Evidence Lifecycle Benchmark

`evidence_lifecycle_benchmark` tests what happens to maintained evidence labels
when a corpus changes. It compares stable ref migration against naive same-path
or same-ordinal chunk ids on controlled mutations and Git-backed documentation
revision pairs.

This is the "corpus-as-test-suite over time" path: old eval labels, citations,
metadata, and review notes should be preserved, migrated, reviewed, or marked
stale instead of silently pointing to wrong evidence after re-chunking.

`lifecycle_ci_demo` is the smaller product-shaped version: it maps two document
revisions, validates a saved `query -> gold_refs` row, and emits a lifecycle
report that can fail CI when stale examples exceed a configured threshold.

## Eval Tool Integrations

`eval_tool_integrations_demo` exports the same Refmark eval run as
Ragas-style rows, DeepEval-style cases, and Phoenix/Langfuse-style trace events.
The adapters are dependency-free handoff formats: existing eval/observability
tools can ingest the rows while Refmark preserves exact refs, source hashes,
stale-state metadata, and corpus/run fingerprints.

`real_sdk_smoke.py` is an optional check for environments that already have
those SDKs installed. It constructs native Ragas, DeepEval, Langfuse, and
Phoenix/OpenInference objects when available, while skipping hosted ingestion
that would require external credentials or a running observability server.

`ragas_refmark_mutation_demo` shows the lifecycle difference directly. A saved
`query -> gold_ref` eval suite is created on one corpus revision, the corpus is
mutated, and then both plain Ragas-style rows and Refmark-enriched rows are
exported. The plain rows still contain valid strings for answer/context scoring;
the Refmark-enriched rows additionally flag stale labels through preserved
source hashes and corpus fingerprints.

`lifecycle_tool_integrations_demo` uses compact summary rows from five
Git-backed documentation lifecycle runs. It exports tracker-style rows and a
short report showing how Ragas-style answer/context evaluation can be paired
with Refmark lifecycle metrics such as stale labels, review-needed refs, and
naive silent-wrong rates.

Data-smell reports can be generated from any `eval-index` run with
`--smell-report-output`. They consolidate stale labels, hard refs, confusion
pairs, query-style gaps, citation breadth issues, confidence gaps, and query
magnets into one reviewable JSON artifact for humans or adaptation agents.

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
