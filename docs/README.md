# Refmark Docs

Start here if you are trying to understand the publishable surface rather than
the research history.

## Main Path

- [Evidence CI Quickstart](QUICKSTART_EVIDENCE_CI.md): the shortest
  `map -> eval -> compare -> smells -> adapt -> manifest diff` workflow.
- [Evidence Summary](EVIDENCE_SUMMARY.md): conservative claims currently
  backed by reproducible artifacts.
- [Publication-Ready Scope](PUBLICATION_READY.md): what is stable enough to
  publish now and what should stay experimental.
- [Product Surface](PRODUCT_SURFACE.md): command/API inventory, maturity, and
  non-goals.

## Core Concepts

- [Address Space Contract](ADDRESS_SPACE_CONTRACT.md): the formal layers,
  prior-art positioning, and claim discipline behind the public framing.
- [Range And Citation Semantics](RANGE_AND_CITATION_SEMANTICS.md): inclusive
  citation ranges, edit boundaries, and parser behavior.
- [Evidence Eval Artifacts](EVIDENCE_EVAL_ARTIFACTS.md): run fingerprints,
  comparison keys, and self-checking report shapes.
- [Data Smells](DATA_SMELLS.md): stale labels, hard refs, confusions, query
  magnets, and other retrieval review signals.
- [Discovery Adapt Loop](DISCOVERY_ADAPT_LOOP.md): discovery, question
  generation, heatmaps, and review-required adaptation loops.

## Integration And Lifecycle

- [Eval Tool Integrations](EVAL_TOOL_INTEGRATIONS.md): Ragas-style rows,
  DeepEval-style cases, and observability traces.
- [Evidence Lifecycle Benchmark](EVIDENCE_LIFECYCLE_BENCHMARK.md): what happens
  to evidence labels when documentation revisions change.
- [Production Feedback Loop](PRODUCTION_FEEDBACK_LOOP.md): turning real search
  interactions into reviewable improvement candidates.
- [Evidence Retrieval Pipeline](EVIDENCE_RETRIEVAL_PIPELINE.md): larger design
  notes for searchable corpus pipelines.

## Other Surfaces

- [Getting Started](GETTING_STARTED.md): install variants and basic CLI checks.
- [Ephemeral Mode](EPHEMERAL_MODE.md): disposable refs for one-off document
  review, citation cleanup, and bounded edits.
- [MCP Usage](MCP_USAGE.md): same-file region editing via MCP.
- [Document Provenance](DOCUMENT_PROVENANCE.md): PDF/DOCX extracted-text
  boundaries and provenance caveats.
- [Project Roadmap](PROJECT_ROADMAP.md): near-, mid-, and long-term work.
- [Next Goals Tasklist](NEXT_GOALS_TASKLIST.md): living deliverable checklist
  with test and advisory-review gates.
- [Research Angles](RESEARCH_ANGLES.md): paper/research directions that should
  stay separate from product claims.

## Research Notes

The BGB, browser search, lifecycle, and training notes are useful context, but
they are not the first public promise. Read them when you want the experiment
history behind the current roadmap:

- [BGB Retrieval Experiment Report](BGB_RETRIEVAL_EXPERIMENT_REPORT.md)
- [Search And Training Findings](SEARCH_AND_TRAINING_FINDINGS.md)
- [Refmark Benchmarking Notes](REFMARK_BENCHMARKING_NOTES.md)

## Tiny Examples

- [Evidence CI Quickstart](../examples/evidence_ci_quickstart): smallest
  local `ci` fixture for map/index/eval/compare/smells/adaptation.
- [Ephemeral Mode Quickstart](../examples/ephemeral_mode_quickstart):
  disposable refs for one-off contract-style edits.
- [Heterogeneous Corpus Stress](../examples/heterogeneous_corpus_stress):
  small mixed-format fixture for query magnets, duplicate support, and uneven
  source styles.
