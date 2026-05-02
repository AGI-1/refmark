# Refmark Project Roadmap

This note keeps the product and research tracks separated. Refmark should be
published as a stable evidence-addressing toolkit first; training, browser
search, and agentic repair loops are important applications, but they should not
be the first claim.

## Core Thesis

Refmark turns a corpus into a stable, addressable evidence space.

Once source regions have durable refs, retrieval, citation, answer-generation,
human review, and training-data maintenance can all target the same concrete
object:

```text
query -> expected evidence refs/ranges -> retrieved or cited refs -> score/report/adapt
```

The central public claim should be:

> Refmark turns your corpus into a regression test suite for retrieval.

This is deliberately narrower than "better search". Refmark makes retrieval
quality measurable and maintainable; it can then be used to improve search
systems, compare search systems, or detect when supervision became stale.

## What Is Publishable Now

The current package is ready to publish as a research toolkit if the README and
examples stay honest about maturity.

Strong, implemented surfaces:

- corpus mapping into stable refs/ranges;
- `EvalSuite` / `eval-index` evidence-region evaluation;
- source-hash stale-label detection;
- comparable eval artifacts with corpus/eval/run fingerprints;
- data-smell reports from eval runs;
- `adapt-plan` review actions from data-smell reports;
- deterministic citation scoring and highlighting;
- lifecycle benchmarks over real documentation revisions;
- dependency-free handoff rows for Ragas, DeepEval, Langfuse, and Phoenix-style
  tooling;
- bounded same-file multi-region edit tooling and MCP server.

Useful demos, but not headline claims:

- portable browser/documentation search;
- FastAPI heatmap and question-improvement workbench;
- BGB/static-search/training experiments;
- PDF/DOCX extracted-text coverage alignment.

## Near Term

Goal: make the repository clear enough for public inspection.

1. Keep README focused on corpus CI and evidence evaluation.
2. Preserve a tiny command-line loop:

   ```bash
   refmark map corpus/ -o corpus.refmark.jsonl
   refmark build-index corpus/ -o corpus.index.json
   refmark eval-index corpus.index.json eval.jsonl --manifest corpus.refmark.jsonl \
     --smell-report-output smells.json -o eval.json
   refmark compare-index corpus.index.json eval.jsonl --manifest corpus.refmark.jsonl \
     --strategies flat,hierarchical,rerank -o compare_index.json
   refmark compare-runs runs/eval_*.json -o compare_runs.json
   refmark adapt-plan smells.json -o adaptation_plan.json
   ```

3. Keep examples small and reproducible; leave large generated outputs ignored.
4. Make `docs/DATA_SMELLS.md`, `docs/EVAL_TOOL_INTEGRATIONS.md`, and
   `docs/EVIDENCE_EVAL_ARTIFACTS.md` the main supporting docs.
5. Ensure tests pass before each push.
6. Avoid claims that Refmark replaces vector databases, answer judges, or search
   engines.

Near-term success means a reader can understand the concept, run a small eval,
see stale labels/data smells, and inspect an adaptation plan in under ten
minutes.

## Mid Term

Goal: make Refmark easy to attach to existing retrieval and documentation
systems.

1. Stabilize library APIs around:
   - `CorpusMap`
   - `EvalSuite`
   - `DataSmellReport`
   - `AdaptationPlan`
   - retriever adapters
2. Add richer integration examples for lifecycle/eval tooling:
   - Ragas rows plus Refmark stale-label metadata;
   - DeepEval cases plus evidence metrics;
   - Phoenix/Langfuse traces with ref/corpus fingerprints.
3. Build a polished FastAPI-style demo:
   - eval report;
   - heatmap;
   - data-smell report;
   - adaptation plan;
   - before/after rerun.
4. Expand production-feedback ingestion:
   - query/click/manual-selection events;
   - no-answer signals;
   - query-magnet and coverage-gap candidates.
5. Strengthen shadow manifests:
   - store refs outside source documents;
   - map refs to revisions;
   - detect changed/removed/moved evidence;
   - keep lifecycle reports CI-friendly.
6. Improve input provenance:
   - Markdown/text as first-class;
   - PDF/DOCX as extracted text unless page/layout provenance is explicitly
     stored.

Mid-term success means an existing RAG stack can keep its retriever and answer
evaluator, add Refmark metadata, and get evidence-region CI plus stale-label
maintenance.

## Long Term

Goal: turn the concept into a repeatable research and product platform.

Research directions:

- compare Refmark evidence eval against answer-level RAG evaluation;
- apply Refmark to external QA/retrieval benchmarks;
- measure label drift and silent wrong evidence across corpus revisions;
- evaluate whether data-smell-driven adaptation improves held-out retrieval;
- use refs/ranges as labels for corpus-local routers, rerankers, and small
  navigation models;
- study browser/local "semantic Ctrl+F" over fixed documentation corpora.

Product directions:

- dashboard for heatmaps, smells, stale labels, and adaptation plans;
- agent workflow where discovery, diagnosis, adaptation, and mini-eval are
  explicit tools;
- richer corpus clustering and overview maps for flat wiki-style corpora;
- optional build-time teacher models that enrich shadow metadata without
  requiring runtime model infrastructure;
- maintained adapters for popular RAG/eval/observability tools.

Long-term success means Refmark becomes a lifecycle substrate: the shared
address space that lets corpora, retrievers, LLM citations, training examples,
and human reviews stay connected as documents change.

## Publication Strategy

Publish the repository before trying to publish broad research claims.

A defensible technical report or arXiv preprint would currently be a
systems/position paper, not a benchmark-superiority paper. A good title shape:

> Refmark: Stable Evidence Addresses for Retrieval Evaluation and Corpus
> Lifecycle CI

Defensible evidence today:

- real documentation revision experiments show naive labels can silently point
  at wrong evidence after corpus changes;
- Refmark distinguishes auto-preserved, review-needed, and stale evidence;
- evidence-region metrics reveal hard refs, confusions, query-style gaps, and
  citation breadth failures below generated prose;
- data-smell reports and adaptation plans make improvement loops auditable.

Evidence still needed before stronger claims:

- held-out retrieval adaptation results across multiple corpora;
- comparison against existing lifecycle/eval tools in their native workflows;
- manually curated or independently generated query sets;
- clearer costs for setup, maintenance, and reviewer effort;
- a polished public demo that does not depend on large ignored artifacts.

The safe public posture is: Refmark makes corpus evidence measurable and
maintainable; retrieval improvements are an important downstream application.
