# Refmark Next Goals Tasklist

Living execution plan for turning Refmark from a promising research/tooling
surface into a clean release. Update this file as chunks are completed,
reprioritized, or deliberately dropped.

## Working Thesis

Refmark is not a better quote selector, a vector database, a browser search
engine, or a training trick.

Refmark formalizes a mutable corpus as a versioned address space of evidence
units. Retrieval, citations, context packing, eval labels, review workflows,
bounded edits, and training experiments can then share the same lifecycle-aware
contract.

The core claim to preserve:

> Layered selectors solve reattachment. Refmark solves evidence lifecycle.

## Operating Rules

- Keep product claims tied to implemented, tested behavior.
- Treat training, browser search, and corpus-local navigation as consumers of
  the address space unless promoted by evidence.
- Run focused tests after each chunk and the full suite before push.
- Get architectural/advisory review after every chunk that changes public
  claims, APIs, lifecycle semantics, or benchmark methodology.
- Keep generated outputs, caches, and model artifacts out of commits unless a
  task explicitly promotes a small fixture.
- Update this file at the end of each chunk with status, evidence, open risks,
  and next action.

## Status Legend

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[!]` blocked or needs decision
- `[?]` candidate, may be dropped

## Current Baseline

Already implemented and tested:

- evidence-region mapping and manifests;
- `CorpusMap`, `EvalSuite`, `EvalRun`;
- `eval-index`, `compare-index`, `compare-runs`;
- data-smell reports and adaptation plans;
- lifecycle validation and five-corpus lifecycle benchmark;
- competent lifecycle baselines, including quote selectors;
- layered selector with conservative quote/similarity gates;
- review-card worksheet and formatted side-by-side review HTML;
- citation parsing/scoring/highlighting;
- integration rows for Ragas/DeepEval/Phoenix/Langfuse-style tools;
- ephemeral map/apply for one-off text/DOCX/extracted-text workflows;
- MCP/code edit surface as bounded application layer.

Current strongest evidence:

- over 51,624 labels and 15 documentation revision comparisons, `chunk_id_only`
  produced 30.0% silent drift under the current migration oracle;
- `chunk_hash_quote_selector` preserved 51.1% with 0.8% silent drift;
- `refmark_layered_selector` preserved 51.4% with 0.0% observed silent drift
  under the current migration oracle and 48.6% review workload;
- oracle independence is not solved; human-reviewed disagreement slices remain
  the decisive evidence gap.

## Chunk 1: Conceptual Spine And Public Narrative

**Goal:** Make the address-space abstraction unmistakable and separate it from
selector novelty.

Tasks:

- [x] Define the formal layers in public docs:
  - source corpus;
  - address space;
  - addressable unit/region/range;
  - anchor bundle/resolver;
  - lifecycle state;
  - evidence obligation;
  - application layers.
- [x] Update README lead to emphasize `corpus -> address space -> evidence
  obligations -> lifecycle states -> consumers`.
- [x] Add a compact prior-art positioning section:
  - qrels / passage-level IR judgments;
  - Web Annotation selectors;
  - robust anchoring;
  - content hashes;
  - QA evidence spans;
  - Refmark's contribution as composition and lifecycle contract.
- [x] Ensure wording never implies that span addressing, quote selectors, qrels,
  or robust anchoring are invented here.
- [x] Add a one-screen diagram, preferably Mermaid, showing the address-space
  stack and consumers.

Acceptance criteria:

- A new reader can explain Refmark in one sentence without mentioning training.
- README clearly says selectors are primitives Refmark can use, not the whole
  product.
- The oracle caveat remains visible near lifecycle benchmark results.

Tests/checks:

- [x] `git diff --check`
- [x] link/path sanity check for docs touched

Advisory review:

- [x] Ask at least two models to review claim safety and prior-art positioning.
- [x] Record accepted/rejected feedback in this file.

Status notes:

- Owner: Codex
- Date started: 2026-05-04
- Date completed: 2026-05-04
- Advisory feedback accepted:
  - clarified "address space" as a logical manifest, not fixed offsets;
  - added that selector/resolver evidence must be versioned and bounded;
  - bounded the no-silent-drift wording to current benchmark tables, migration
    oracle, and corpus fingerprinting policy.
- Advisory feedback not taken:
  - did not weaken the benchmark phrasing to only "reduced silent drift",
    because the measured table is legitimately 0 observed silent drift for the
    layered selector; the caveat now carries the burden.
- Open risks: final reader sanity pass still useful before release.

## Chunk 2: Lifecycle State Model V2

**Goal:** Move from coarse `preserved/review/stale` to actionable lifecycle
states that justify Refmark as workflow infrastructure.

Target state vocabulary:

- `unchanged`
- `moved`
- `rewritten`
- `split_support`
- `merged`
- `deleted`
- `ambiguous`
- `alternative_support`
- `duplicate_support`
- `contradictory_support`
- `low_confidence`
- `partial_overlap`
- `semantic_drift`
- `superseded`
- `deprecated`
- `externalized`
- `invalidated`

Tasks:

- [x] Design a typed lifecycle-state schema.
- [x] Add lifecycle state fields to review cards and lifecycle outputs without
  breaking existing consumers.
- [x] Detect at least:
  - moved exact;
  - fuzzy rewritten;
  - ambiguous quote/selector hits;
  - split-support candidate;
  - deleted/stale;
  - alternative support candidate.
- [x] Update review HTML to surface state, reason, confidence, and suggested
  next action.
- [x] Update `adapt-plan` or a lifecycle-specific planner to consume these
  states.

Acceptance criteria:

- Review packet tells the reviewer what kind of decision they are making, not
  just that review is required.
- Baselines can still be compared through the old method-comparison table.

Tests/checks:

- [x] unit tests for each lifecycle state emitted by small fixtures;
- [x] snapshot or structural test for review-card JSON schema;
- [x] CLI smoke for worksheet/render/calibrate;
- [x] full `python -m pytest -q`.

Advisory review:

- [x] Ask for schema critique and missing state categories.
- [x] Ask for failure modes: duplicated support, split evidence, semantic rewrite.

Status notes:

- Owner: Codex
- Date started: 2026-05-04
- Date completed: 2026-05-04
- Implemented:
  - `LifecycleDecision` and `lifecycle_by_ref` in Git lifecycle reports;
  - `lifecycle_state_counts` in stable migration reports;
  - split-support detection as review/range-repair instead of stale;
  - review card schema `refmark.lifecycle_review_card.v2`;
  - HTML/CSV review fields for state, reason, confidence, priority, and next
    action;
  - review summary counts by lifecycle state and suggested action.
- Advisory feedback accepted:
  - reserved future states for partial overlap, semantic drift, superseded,
    deprecated, externalized, and invalidated labels;
  - added lifecycle priority to decisions/review cards;
  - documented that confidence is selector/resolver confidence, not semantic
    truth.
- Advisory feedback not taken:
  - did not merge `ambiguous` and `low_confidence`; they route different review
    work today;
  - did not add post-review resolution states yet, because that belongs with the
    human-review utility study.
- Verification:
  - focused lifecycle tests: `16 passed`;
  - full suite: `240 passed`, with the existing pytest cache permission warning.
- Open risks: post-review resolution states still belong to Chunk 3.

## Chunk 3: Human Review Utility Study

**Goal:** Test whether Refmark review packets reduce human effort or improve
decisions compared with a competent selector registry.

Research question:

> At similar review rates, does Refmark make review faster, safer, or more
> actionable than chunk-hash plus quote selector output?

Tasks:

- [x] Build paired review packets:
  - baseline selector view;
  - Refmark lifecycle view.
- [x] Sample 300-500 cases focused on method disagreement:
  - quote selector preserved, Refmark review/stale;
  - Refmark preserved, quote selector review/stale;
  - quote-selector silent-drift candidates;
  - split-support candidates;
  - fuzzy moved/rewritten candidates.
- [x] Add a lightweight review form that records:
  - verdict;
  - confidence;
  - seconds spent;
  - suggested action;
  - whether alternative support was found;
  - whether split/range repair is needed.
- [x] Generate summary metrics:
  - seconds per label;
  - reviewer confidence;
  - accepted auto-migrations;
  - stale labels caught;
  - alternative supports discovered;
  - split/range repairs discovered.
- [x] Keep LLM-review as a calibration aid, not the final oracle.

Acceptance criteria:

- We can compare review utility, not only review percentage.
- Result supports or falsifies the claim that lifecycle states are operationally
  useful.

Tests/checks:

- [x] review packet generation test;
- [x] timing field parsing/aggregation tests;
- [x] calibration report test on a small fixture;
- [x] HTML render smoke.

Advisory review:

- [x] Ask advisors to critique study design before running full review.
- [ ] Ask advisors to critique conclusions after results.

Status notes:

- Owner: Codex
- Date started: 2026-05-04
- Date completed:
- Implemented:
  - `paired-worksheet` command emits baseline-selector and Refmark-lifecycle
    rows for the same cards;
  - `utility` command summarizes filled review rows by view;
  - CSV/HTML rows include review view, seconds, suggested human action,
    alternative-support flag, and split/range-repair flag.
  - counterbalanced paired-row ordering with optional blind public labels;
  - 360-card disagreement-focused sample for utility review;
  - 40-card / 80-judgment LLM calibration run across DeepSeek and Qwen.
- Advisory feedback accepted:
  - paired studies must control for order/carryover effects;
  - baseline and Refmark rows need comparable evidence context, otherwise
    review-speed results only measure information density;
  - disagreement-only sampling is valuable for failure analysis but biased for
    population-level workload claims;
  - LLM judgments are calibration and triage aids, not an independent oracle.
- Verification:
  - focused lifecycle/review tests: `18 passed`;
  - `py_compile` passed for the review script and lifecycle tests;
  - paired worksheet and utility smoke succeeded on sample cards;
  - LLM calibration reliability sample completed with `80 ok / 0 failed`.
- Open risks:
  - 300-500 case sample is generated but not human-filled;
  - LLM calibration was run on review cards, not final paired packets;
  - true blinding still needs a human-facing export that hides internal
    `review_view` while preserving it for analysis;
  - full suite should be rerun before commit/push.

## Chunk 4: Human-Reviewed Disagreement Slice

**Goal:** Close the main oracle-independence gap for the lifecycle benchmark.

Tasks:

- [ ] Select disagreement cases with stratified sampling.
- [ ] Prepare human-readable side-by-side formatted evidence, including:
  - old evidence;
  - candidate evidence;
  - changed spans;
  - method decisions;
  - selector/hash/fuzzy signals;
  - source path/ref metadata.
- [ ] Define human verdict taxonomy:
  - valid unchanged;
  - valid moved;
  - valid rewritten;
  - split support;
  - alternative support;
  - stale;
  - ambiguous;
  - invalid candidate.
- [ ] Review at least a first tranche of 100-150 cases manually.
- [ ] Expand to 300-500 cases if the first tranche is stable.
- [ ] Report agreement/disagreement against:
  - Refmark migration oracle;
  - quote selector baseline;
  - layered selector;
  - LLM adjudication.

Acceptance criteria:

- We can state which lifecycle numbers are oracle-only and which are supported
  by human-reviewed samples.
- The preprint/report can include a credible independent validation section.

Tests/checks:

- [x] reviewer CSV schema test;
- [ ] calibration aggregation test;
- [x] HTML review rendering smoke;
- [ ] no generated review outputs committed unless converted to a small fixture.

Advisory review:

- [ ] Review sampling strategy before manual work.
- [ ] Review result interpretation before publication.

Status notes:

- Owner:
- Date started:
- Date completed:
- Implemented:
  - regression coverage for human-review worksheet utility fields;
  - regression coverage for formatted side-by-side Markdown evidence diffs.
- Verification:
  - focused lifecycle/review tests: `20 passed`;
  - `py_compile` passed for the review script and lifecycle tests.
- Open risks:
  - actual human-reviewed tranche still not performed;
  - disagreement sample design needs final balancing before publication use.

## Chunk 5: Product CLI/API Happy Path

**Goal:** Make the default lifecycle/evidence CI flow feel like a small usable
tool rather than a bag of scripts.

Target command shape:

```bash
refmark map corpus/ -o corpus.refmark.jsonl
refmark build-index corpus/ -o corpus.index.json
refmark eval-index corpus.index.json eval.jsonl --manifest corpus.refmark.jsonl -o eval.json
refmark lifecycle-validate-labels corpus.refmark.jsonl eval.jsonl -o lifecycle.json
refmark adapt-plan smells.json -o adaptation_plan.json
```

Tasks:

- [ ] Decide whether to add wrappers/aliases for common flows or keep current
  commands.
- [x] Add a compact `refmark quickstart` or `refmark ci` command if it reduces
  friction.
- [ ] Ensure outputs contain stable schema names and fingerprints.
- [x] Add a tiny example corpus and eval suite that runs in under one minute.
- [ ] Add docs showing how to attach Refmark to an existing retriever.
- [x] Add a "what to commit vs what to ignore" section.

Acceptance criteria:

- A new user can run one end-to-end evidence CI demo in under ten minutes.
- CI output clearly identifies pass/fail and next review action.

Tests/checks:

- [x] CLI smoke tests for the happy path;
- [x] fixture-based output schema tests;
- [ ] full suite.

Advisory review:

- [x] Ask for API ergonomics and confusing command names.
- [x] Ask for docs-first review from a skeptical RAG engineer.

Status notes:

- Owner: Codex
- Date started: 2026-05-04
- Date completed:
- Implemented:
  - `examples/evidence_ci_quickstart/` with a tiny policy corpus and eval JSONL;
  - quickstart docs now point to the direct `refmark.cli ci` command;
  - quickstart docs now include what to commit vs ignore;
  - local smoke verified the quickstart writes manifest, index, eval,
    comparison, smell report, and adaptation plan.
- Verification:
  - quickstart CLI smoke passed with `--source local`, `--min-hit-at-k 1.0`,
    `--min-best-hit-at-k 1.0`, and `--fail-on-regression`;
  - fixture-backed quickstart schema test added;
  - full suite after preceding review changes: `244 passed`.
- Open risks:
  - happy-path fixture proves command shape, not realistic retrieval quality;
  - wrapper/alias ergonomics still need a skeptical user pass.
  - advisor warned that examples must distinguish committed fixture inputs from
    generated outputs; quickstart docs now include that boundary.

## Chunk 6: Ephemeral Mode Product Wedge

**Goal:** Make one-off address maps useful enough to demonstrate immediate
risk reduction versus full-document LLM rewrites.

Tasks:

- [ ] Add a clean example:
  - [x] input `.md` or `.docx`;
  - [x] generated temporary refs;
  - [x] model-style edit JSON;
  - [x] dry-run;
  - [x] patched output.
- [ ] Strengthen safety reporting:
  - [x] unknown ref;
  - [x] unsupported action;
  - [x] duplicate source match;
  - [x] DOCX multi-paragraph rejection;
  - [x] PDF extracted-text warning.
- [ ] Consider a side-by-side review HTML for ephemeral patches.
- [ ] Document limitations around DOCX rich formatting and PDF layout.
- [x] Add optional manifest output examples for one-off workflows that want an
  audit trail.

Acceptance criteria:

- A user can understand and run ephemeral mode without understanding RAG.
- The workflow is clearly safer than blind rewrite and clearly not a full DOCX
  layout editor.

Tests/checks:

- [x] CLI tests for invalid JSON, missing refs, duplicate matches, dry-run;
- [x] DOCX exact paragraph replacement test;
- [x] text/Markdown replacement test;
- [x] no output file on failed apply.

Advisory review:

- [x] Ask for safety risks around document corruption and misleading claims.
- [x] Ask for UX feedback on edit JSON and dry-run output.

Status notes:

- Owner: Codex
- Date started: 2026-05-04
- Date completed:
- Open risks:
  - side-by-side patch preview HTML is still a candidate, not implemented;
  - DOCX support remains exact single-paragraph replacement only.
  - advisor warned that ephemeral mode can be mistaken for durable workflow
    state; docs now distinguish disposable mode from lifecycle mode.

## Chunk 7: Integration Positioning

**Goal:** Make clear how Refmark complements existing lifecycle/eval tools
instead of competing with them.

Tasks:

- [ ] Add integration examples for:
  - [x] Ragas-style rows;
  - [x] qrels/trec_eval style rows;
  - [x] DeepEval-style test cases;
  - [x] Phoenix/Langfuse-style traces;
  - [x] LlamaIndex/LangChain document metadata shape.
- [x] Add docs explaining:
  - [x] existing tools can store IDs;
  - [x] Refmark owns/version-controls the address space and lifecycle states;
  - [x] answer judges and evidence lifecycle checks answer different questions.
- [ ] Add a "competitor/prior art mapping" table.
- [ ] Include quote selectors and Web Annotation as primitives Refmark can use.

Acceptance criteria:

- Existing RAG/eval tool users can see where Refmark attaches.
- Refmark is not presented as replacing their stack.

Tests/checks:

- [x] adapter output schema tests;
- [ ] optional SDK construction smoke if package is installed;
- [ ] docs command examples checked.

Advisory review:

- [x] Ask for integration/prior-art critique.
- [x] Ask whether the positioning feels additive or defensive.

Status notes:

- Owner: Codex
- Date started: 2026-05-04
- Date completed:
- Open risks:
  - no hosted ingestion is tested by default;
  - qrels export is JSONL-shaped handoff data, not a full trec_eval CLI wrapper;
  - metadata rows are dependency-free, not native LangChain/LlamaIndex object
    constructors.
  - advisor warned that "integration" can imply live sync or SDK ownership; docs
    now frame these as versioned handoff schemas, not hosted integrations.

## Chunk 8: Heterogeneous Corpus Stress

**Goal:** Find where the address-space abstraction helps beyond clean Markdown
docs and where it breaks.

Candidate corpora:

- Markdown API docs;
- duplicate-heavy generated docs;
- flat wiki export;
- legal/policy docs;
- DOCX extracted text;
- PDF extracted text;
- tables or semi-structured records;
- mixed code plus docs.

Tasks:

- [x] Select 3-5 corpus styles with different failure modes.
- [x] Run mapping/eval/smell inspection where applicable.
- [x] Record extraction/provenance limitations.
- [~] Record where quote selectors alone fail or become ambiguous.
- [~] Identify corpus-specific resolver needs.

Acceptance criteria:

- We know which corpus types Refmark handles well today.
- We know which corpus types require stronger provenance/resolvers before
  public claims.

Tests/checks:

- [x] small fixture for each new resolver/provenance behavior;
- [x] no raw copyrighted/large corpora committed;
- [x] benchmark scripts cache and fail clearly.

Advisory review:

- [x] Ask advisors to identify missing corpus classes and likely failure modes.

Status notes:

- Owner: Codex
- Date started: 2026-05-04
- Date completed:
- Implemented:
  - `examples/heterogeneous_corpus_stress/` with Markdown, plain text, RST,
    table-like Markdown, duplicate support, and release-note query magnets;
  - fixture README documents build, inspect, and eval commands;
  - docs index links the fixture as a tiny example.
- Verification:
  - build-index smoke: 5 documents, 15 regions, 3 default-excluded query magnets;
  - inspect-index smoke: 3 query magnets, 1 exact duplicate group;
  - eval-index smoke: rerank `hit_at_k=1.0`, with query-style gap and
    overcitation smell visible;
  - focused pipeline tests: `26 passed`.
- Open risks:
  - this fixture is intentionally tiny and should not be treated as benchmark
    evidence;
  - DOCX/PDF remain extracted-text concerns and are not represented by committed
    binaries here;
  - quote-selector ambiguity on heterogeneous files still needs a dedicated
    revision/mutation slice.
  - advisors emphasized that broader heterogeneous claims need real DOCX/PDF
    provenance, malformed inputs, mutation tests, scale, latency/memory, and
    precision/false-citation metrics.

## Chunk 9: Data Smells And Adaptation Loop V2

**Goal:** Turn retrieval/evidence failures into actionable, auditable review
queues.

Tasks:

- [x] Expand smell taxonomy:
  - [x] stale label;
  - [x] hard ref;
  - [x] confusion pair;
  - [x] query magnet;
  - [x] duplicate support;
  - [x] contradictory support;
  - [x] undercitation;
  - [x] overcitation;
  - [x] low confidence;
  - [~] split support;
  - [x] uncovered region.
- [~] Link smells to lifecycle states.
- [x] Add action classes:
  - [x] add eval row;
  - [x] add alias/doc2query metadata;
  - [x] split/merge region;
  - [x] mark query magnet;
  - [x] add alternative support;
  - [x] review contradiction;
  - [x] tighten/expand range policy.
- [x] Add before/after run comparison for adaptation loops.

Acceptance criteria:

- Smell report is useful to a human editor, not only an aggregate metric dump.
- Each suggested adaptation has a reason, evidence refs, and review status.

Tests/checks:

- [~] smell fixtures for each category;
- [x] adaptation-plan fixtures;
- [x] compare-smells regression test;
- [ ] HTML/dashboard smoke if visualized.

Advisory review:

- [ ] Ask for missing data-smell categories.
- [ ] Ask whether actions are operationally clear.

Status notes:

- Owner: Codex
- Date started: 2026-05-04
- Date completed:
- Implemented:
  - eval-run data-smell report now emits `duplicate_support`,
    `contradictory_support`, and `uncovered_region` in addition to the existing
    stale/hard/confusion/style/range/citation/confidence/query-magnet smells;
  - adaptation planner maps the new smells to support topology, corpus
    consistency, and coverage-planning actions;
  - data-smell docs list the expanded taxonomy and clarify the relation between
    index/corpus smells and eval-run smells.
  - `compare-smells` CLI/API emits `refmark.data_smell_comparison.v1` for
    before/after adaptation loops.
- Verification:
  - focused data-smell/adaptation tests: `5 passed`;
  - py_compile passed for changed modules/tests.
- Open risks:
  - split-support is represented in lifecycle/review paths but is not yet a
    dedicated data-smell type in `build_data_smell_report`;
  - contradictory support detection is lexical/cue-based and intentionally only
    a review signal;
  - visual dashboard smoke remains open.
  - advisors asked for more executable agent handoff payloads, resolution
    effort/status fields, conflict precedence, and immutable audit trails; those
    are intentionally deferred to a review-workflow chunk.
  - advisors suggested `fragmented_support`, temporal supersession, confidence
    inflation, citation dilution, and context-boundary smells as future
    candidates.

## Chunk 10: Release Candidate Pass

**Goal:** Decide whether the project is releaseable, needs another cycle, or
should be frozen as a research toolkit.

Releaseable when:

- [x] README leads with corpus address space and evidence lifecycle.
- [x] Public claims distinguish implemented behavior from research directions.
- [x] Lifecycle benchmark includes competent baselines and visible oracle caveat.
- [x] At least one runnable small example demonstrates map/eval/lifecycle/review.
- [x] Ephemeral mode has a safe, documented happy path.
- [x] Tests pass locally.
- [x] Generated outputs are ignored or absent.
- [x] Advisory review finds no major claim/API blocker.

Closed/frozen when:

- [x] Stable address-space API is documented.
- [x] Lifecycle validation and review packet workflow are usable.
- [x] Citation/eval/highlight and ephemeral edit workflows are usable.
- [~] Remaining ideas are research applications rather than core substrate.
- [~] The project can be maintained with small bugfixes instead of broad churn.

Stop or narrow if:

- [ ] lifecycle states do not reduce review time or improve decisions;
- [ ] human-reviewed disagreement slices show the migration oracle is unreliable;
- [ ] quote-selector registries match the workflow value with much less surface;
- [ ] users only perceive Refmark as a wrapper around hashes/selectors;
- [ ] citation, eval, lifecycle, and ephemeral-edit use cases fail to create
  independent value.

Final verification:

- [x] `python -m pytest -q`
- [x] `git diff --check`
- [x] CLI smoke: evidence CI quickstart
- [x] CLI smoke: lifecycle method aggregation
- [x] CLI smoke: review packet render
- [x] CLI smoke: ephemeral map/apply
- [x] advisory review summary filed

Status notes:

- Owner: Codex
- Date started: 2026-05-04
- Date completed: 2026-05-04
- Release decision: releaseable as an evidence-address/evidence-lifecycle
  toolkit with explicit maturity boundaries. Not closed as a research topic:
  heterogeneous provenance, human-reviewed disagreement evidence, review
  workflow tickets/audit trails, and training/navigation experiments remain
  separate next-stage work.

## Advisory Review Log

Append short entries here after each advisor pass.

```text
Date: 2026-05-04
Chunk: 5-7 product surface
Models/reviewers: DeepSeek v3.2 via OpenRouter; Qwen request reached the API but
local Unicode printing failed before capture.
Accepted feedback: clarify that eval-tool exports are snapshot handoff schemas,
not hosted/live integrations; document schema-version boundaries; distinguish
ephemeral one-off mode from durable lifecycle workflows; keep committed fixture
inputs separate from generated outputs.
Rejected feedback: none from captured DeepSeek review.
Follow-up tasks: add native SDK constructors only if users need them; consider
ephemeral side-by-side patch preview as a future UX improvement.
Decision: product surface remains additive and releaseable once full tests pass.
```

```text
Date: 2026-05-04
Chunk: 8 heterogeneous corpus stress
Models/reviewers: DeepSeek v3.2 and Qwen 3.6 Plus via OpenRouter
Accepted feedback: keep the fixture scoped as smoke/stress, not benchmark
evidence; avoid robust heterogeneous support claims; call out extracted-text
boundaries for PDF/DOCX; require mutation tests, negative controls, real
document formats, scale, and precision/false-citation metrics before broader
claims.
Rejected feedback: none.
Follow-up tasks: add heterogeneous mutation slice and negative controls before
claiming more than smoke coverage; keep real DOCX/PDF provenance as a later
resolver/provenance chunk.
Decision: fixture is useful release smoke coverage, not release evidence for
heterogeneous corpus robustness.
```

```text
Date: 2026-05-04
Chunk: 9 data smells and adaptation loop
Models/reviewers: DeepSeek v3.2 and Qwen 3.6 Plus via OpenRouter
Accepted feedback: keep contradiction as lexical triage, not proof; make
contradiction adaptation category more precise as corpus consistency; document
exact-duplicate limitation; record missing future smells and agent-handoff
requirements.
Rejected feedback: did not add confidence-gated claim blocking, audit trails, or
precedence engine in this chunk because they belong to a larger review-workflow
surface.
Follow-up tasks: add fragmented support and context-boundary smells; add
resolution effort/status fields when adaptation actions become workflow tickets.
Decision: chunk improves actionable smell coverage without overclaiming.
```

Template:

```text
Date:
Chunk:
Models/reviewers:
Accepted feedback:
Rejected feedback:
Follow-up tasks:
Decision:
```

## Open Questions

- Should lifecycle states become a first-class public enum now or remain schema
  strings until they settle?
- Should ephemeral mode support patch previews before applying?
- Should review packets become a public CLI command or remain an example script?
- How much human review evidence is enough for the first preprint?
- Do we need a formal prior-art section in README or only in docs/preprint?
- Which external corpus types are essential before a broader release?
