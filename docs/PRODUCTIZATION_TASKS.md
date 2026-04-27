# Refmark Productization Tasks

This is a working task list for turning Refmark from a research artifact into
small, useful library and CLI surfaces. The goal is plug-and-play adoption in
review, RAG, and model-evaluation workflows without hiding the limits of the
current implementation.

## Current Core Surfaces

- `refmark.rag_eval`: evaluate `query -> gold refs/ranges` examples against
  arbitrary retriever callbacks and detect stale examples after corpus changes.
- `enrich-prompt`: wrap a document so general chat models can answer with
  concrete region and range citations.
- `map`: create JSONL manifests of addressable document regions.
- `expand`: expand a retrieved/cited region into neighboring context.
- `align`: map source document regions to target document regions.
- `coverage_alignment` example: generated DOCX/PDF review flow with covered
  and gap regions.

## Priority 1: Make The Library Feel Obvious

- [done] Add a top-level `refmark.documents` convenience module.
  - `map_document(path) -> DocumentMap`
  - `align_documents(source, target) -> AlignmentReport`
  - `write_coverage_report(report, html_path)`
- [done] Keep dataclasses serializable and predictable.
  - Stable `.to_dict()` on every report type.
  - JSONL manifest schema documented in one place.
- [done] Add a small `DocumentMap` wrapper.
  - Holds `marked_text`, `records`, `doc_id`, `source_path`.
  - Methods: `expand(refs, before=0, after=1)`, `to_jsonl(path)`.
- [done] Add a small `AlignmentReport` wrapper.
  - Holds alignments, coverage items, and summary.
  - Methods: `to_json(path)`, `to_html(path)`.
- [done] Add a top-level attachable `Refmarker` wrapper.
  - `mark_text(...) -> RefmarkResult`
  - live mode embeds markers into output content.
  - shadow mode preserves source content and stores the marked view in a registry.
- [partial] Decide whether headings/titles are first-class regions.
  - Current CLI maps them.
  - Coverage workflows can set `include_headings: false`.
  - Product API should expose an explicit `include_headings` or
    `region_filter` option.

## Priority 2: Improve CLI Ergonomics

- [done] Add `--ignore-title` / `--min-words` to `map` and `align`.
- [done] Add `--summary-json` to `align`.
- Add `--marked-source` and `--marked-target` output options to `align`.
- Add `--no-expanded-evidence` for compact HTML reports.
- Print short human summaries to stderr when writing JSON/HTML artifacts.
- Add `refmark coverage` as a clearer alias over `align --coverage-*`.
- Add examples of common commands to `refmark --help` epilog.

## Priority 3: Prompt-Enrichment Use Case

- Add examples for:
  - long policy document
  - pasted contract excerpt
  - source region set plus question
- Add `--copy-safe` mode that avoids markdown fences and extra formatting.
- Add `--answer-format` presets:
  - `claims`
  - `table`
  - `audit`
  - `json`
- [done] Add an answer parser that extracts `[P01]`, `[P01-P03]`, and `[P01,P04]`.
- [done] Add a deterministic evaluation helper:
  - compare model-cited refs to expected refs
  - emit overcite / undercite / wrong-location data smells

## Priority 4: RAG Enhancement

- [done] Treat Refmark as metadata, not just injected text.
  - Store `doc_id`, `region_id`, `hash`, `prev_region_id`, `next_region_id`.
  - `parent_region_id` is present in the manifest schema but not yet populated
    by section/chapter hierarchy.
- [done] Add a first-pass eval-suite API.
  - `CorpusMap`
  - `EvalExample`
  - `EvalSuite.evaluate(...)`
  - `EvalSuite.compare(...)`
  - stale-example detection via stored source hashes
- Add `expand_policy` helpers:
  - [done] fixed neighbors: before/after
  - [done] same heading parent for Markdown-style heading regions
  - until token budget
  - stop at heading boundary
- Add a `context_pack` function:
  - [done] `CorpusMap.context_pack(...)`
  - [done] CLI `pack-context`
  - input: retrieved region ids
  - output: ordered text bundle with stable refs
- Add a small benchmark example:
  - [done] CLI `eval-index` evaluates portable search indexes against JSONL
    `query -> gold_refs` suites.
  - retrieval hit only
  - hit plus neighbor expansion
  - hit plus parent expansion
  - measure gold-support coverage and token cost
- [done] Document the large evidence-retrieval pipeline.
  - `docs/EVIDENCE_RETRIEVAL_PIPELINE.md`
  - covers structured Markdown corpus preparation above 200k tokens
  - separates single-region, contiguous-range, and distributed-ref targets
  - reports BM25, generated/local views, reranking, embeddings, hybrid search,
    and trained-resolver evidence where available
  - frames the output as CI evidence, not as a one-off demo score

## Priority 4A: Evidence Retrieval CI

- [partial] Add discovery stage.
  - CLI: `refmark discover manifest.jsonl -o corpus.discovery.json`
  - local deterministic source implemented for tests and smoke runs
  - OpenRouter source implemented for whole-corpus model-backed discovery with
    local fallback
  - hierarchical/windowed discovery still needs real merge/reconcile passes
- Add first-class `gold_mode` summaries to `EvalSuite` and `eval-index`.
  - `single`
  - `range`
  - `distributed`
  - `parent` / `section`
- [partial] Add first-class `gold_mode` summaries to the portable real-corpus
  evaluator.
  - reports `by_gold_mode`
  - still needs the same shape in library `EvalSuite` and CLI `eval-index`
- Add CI thresholds:
  - maximum stale examples
  - minimum hit@k / MRR
  - minimum gold coverage
  - maximum overcitation breadth
  - maximum undercitation rate
- Add a cached question-generation runner.
  - Input: manifest + refs/ranges + prompt template
  - Output: JSONL eval rows with source hashes and target-shape metadata
  - Cache key: ref/range, source hashes, provider, model, prompt version
- Add a compact run manifest for reproducibility.
  - corpus manifest hash
  - eval suite hash
  - search index hash
  - generated-view cache hash
  - model/provider settings
- Add per-method failure exports.
  - misses
  - partial-range hits
  - wrong-neighbor hits
  - overbroad but supported results
  - distributed-ref partial coverage

## Priority 5: Review / HiL Coverage Pipelines

- Improve coverage status model.
  - Current: `covered` / `gap`
  - Proposed: `covered`, `partial`, `conflict`, `gap`, `needs_review`
- Make numeric checks explicit and documented.
  - Current: tiny unit-aware baseline for days/years/hours/kWh/percent.
  - Add currency, dates, ranges, greater-than/less-than language.
- Add negation/conflict detection.
  - Example: "not included", "separately purchased", "subject to approval".
- Add review annotations.
  - Human can accept/reject candidate coverage.
  - Output can become a reviewed alignment dataset.
- [done] Add side-by-side HTML report.
  - Source requirement on left, target candidates/expanded evidence on right.
  - Keep region refs highly visible.
- Add CSV export for procurement/legal review teams.

## Priority 6: Document Format Support

- Current DOCX extraction is plain OOXML paragraph extraction.
- Current PDF extraction uses `pypdf`.
- [done] Document current provenance boundary: PDF/DOCX refs resolve to
  extracted text regions, not original-layout page boxes.
- Improve DOCX:
  - tables
  - headers/footers
  - numbered lists
  - section headings
  - comments/tracked changes as optional metadata
- Improve PDF:
  - page numbers
  - layout-aware blocks where available
  - scanned PDF warning
- Add extracted text QA artifacts.
  - `source_extracted.txt`
  - `target_extracted.txt`
  - region count and extraction warnings

## Priority 7: Model-Assisted Experiments

- Add optional OpenRouter runner for prompt-enrichment evaluation.
  - Input: marked prompt + question
  - Output: answer + cited refs
  - Metrics: exact/overlap/cover/data-smell
- Keep it optional and environment-driven.
  - No hard dependency on API access.
  - Redact keys from logs.
- Compare models on:
  - unmarked prompt
  - marked prompt
  - marked prompt with citation format examples
- [done] Add model-agnostic question-generation prompt builder.
  - CLI `question-prompt`
  - overridable template
  - emits prompts for any external LLM endpoint or manual curation
- For coding agents, test:
  - baseline patch
  - live marked file
  - shadow MCP workflow
  - WSL child-process validation where useful

## Priority 8: Package / Install Polish

- Add optional extras:
  - `documents = ["pypdf"]`
  - maybe `docx = []` if OOXML remains stdlib-only
- Ensure examples run from fresh checkout after `pip install -e .[dev]`.
- Decide whether generated `examples/*/output/` should be gitignored.
- Add a short public-facing "What Refmark is not" section:
  - not a vector database
  - not a semantic matcher
  - not proof of broad coding-agent superiority
  - not a guarantee that the model picked the right region

## Open Discussion Questions

- Should default prose markers be `P01` or a document-specific prefix like
  `D001`?
- Should headings become regions, parents, or metadata?
- Should region ids remain ordinal after document mutation, or should manifests
  prefer content hashes plus stable local ids?
- Where should the line be between deterministic helpers and model-assisted
  review?
- Is the most compelling public demo prompt citation, RAG expansion, or
  coverage alignment?
