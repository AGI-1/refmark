# Review Follow-Ups

This table merges valid/actionable points from `review_*.txt` and
`cgpt_review_re.txt`. It is intended for prioritization, not as a commitment to
implement everything.

Current priority interpretation:

- The core abstraction remains **addressable regions for documents/code/corpora**.
- The strongest product framing is **corpus-as-test-suite**: once a corpus has
  stable refs/ranges, it becomes CI-testable infrastructure for retrieval,
  citations, training data, corpus drift, and human review.
- The strongest near-term wedge inside that framing is **RAG evidence
  evaluation**: attach stable refs/ranges to a corpus, then compare chunkers,
  retrievers, rerankers, query rewriters, expansion policies, and
  vector/BM25/hybrid stacks by whether they recover the correct evidence
  regions.
- MCP editing, citation highlighting, browser search, and training are important
  applications of the same address space, but should not displace the evidence
  lifecycle/evaluation substrate as the crisp lead.

## Current Status Snapshot

Done or good-enough for current publish pass:

- R02: README has a 30-second mental model and concrete example.
- R03: `AGENTS.md` defines canonical terminology for future edits.
- R07/R08/R10/R12/R13/R14: `refmark.rag_eval` now exposes `CorpusMap`,
  `EvalExample`, `EvalSuite`, retriever callback comparison, evidence metrics,
  source-hash capture, stale-example detection, and corpus diffing.
- R21/R23/R24: existing tests cover many `apply_ref_diff`, premarked detection,
  and shadow-session cases; keep expanding, but no longer treat as absent.
- R26: highlight/resolve exists through `highlight_refs` and CLI `highlight`.
- R04/R25: `docs/RANGE_AND_CITATION_SEMANTICS.md` now defines inclusive
  citation ranges vs exclusive edit boundary ranges, and `refmark.citations`
  provides strict parsing/validation helpers.
- R15/R17/R18/R19/R20: MCP docs now describe read/list/apply flow, ordinal
  region ordering, metadata-only logging, path roots, and truncation controls;
  code enforces allowed roots when configured and redacts edit bodies by default.
- R27/R28: `docs/DOCUMENT_PROVENANCE.md` now states that PDF/DOCX refs resolve
  to extracted text regions, not original-layout page boxes.
- R29/R30/R31/R32/R34/R35: portable search now has local/openrouter views,
  metadata-only mode, cached generated views, hierarchical/rerank strategies,
  browser export, Unicode tokenization, and neighbor recomputation after filtered
  regions.
- R33: Markdown-style heading regions now populate `parent_region_id`, and
  `expand_region_context(..., same_parent=True)` / CLI `expand --same-parent`
  can expand within the parent section.
- R42/R43: README now leads with corpus CI/evidence recovery and includes a
  small `CorpusMap`/`EvalSuite` example.
- R46: `.gitignore`/`.diffignore` cover generated outputs and review artifacts.

Still high-priority open work:

- R11/R33: parent/section-aware expansion exists for lightweight heading
  metadata, but richer PDF/DOCX/HTML hierarchy is still open.
- R16/R17: MCP shadow/live session contract is stronger, but dry-run/preview
  diff/base-hash guards still need a dedicated hardening pass.
- R25: citation grammar/parser exists; next step is integrating it into every
  CLI/example that accepts refs.
- R41/R44/R45: package/narrative pruning and examples organization need final
  polish before publishing.

| ID | Area | Valid point | Follow-up to review/prioritize | Suggested priority | Notes |
| --- | --- | --- | --- | --- | --- |
| R01 | Narrative | Refmark should be framed as an addressable document/text/code space that makes corpora testable, with RAG evidence evaluation as the clearest product wedge. | Rewrite README opening around stable evidence addresses for corpus CI: evaluate whether a pipeline recovered the exact support regions/ranges, then detect stale examples after corpus changes. Keep cite/edit/search/train as downstream uses. | P0 | Updated after `cgpt_review_re.txt` and follow-up review: concept stays broad, lead becomes corpus-as-test-suite/evidence evaluation. |
| R02 | Narrative | Marker syntax is implementation detail; the public concept should be region/address/resolver. | Add a 30-second mental model and diagram: source -> regions -> refs -> resolve -> operate. | P0 | This also addresses earlier README feedback. |
| R03 | Terminology | Terms are mixed: anchor, marker, refmark, ref, region, range, block. | Define canonical terminology: marker = visible token, ref/region_id = ID, region = span, range = ordered refs, manifest/registry = mapping. | P0 | Should be reflected in README, docs, API names, examples. |
| R04 | Range Semantics | Citation ranges and edit boundary ranges can be confused. | Document inclusive citation ranges (`P03-P05`) vs exclusive edit spans (`start_ref` to before `end_ref`). Consider renaming user-facing `end_ref` to `stop_before_ref`. | P0 | Important for citations and MCP correctness. |
| R05 | Public Scope | `refmark_train` is valuable but distracts from core package maturity. | Keep training clearly experimental/research. Avoid making it a primary README claim or default product story. | P0 | User already agrees training is not yet publishable value. |
| R06 | Product Lanes | The project has several valid lanes but currently feels blended. | Explicitly split docs into lanes: corpus-as-test-suite, RAG evidence evaluation harness, region metadata layer, citation/highlight review, MCP editing, browser/local search, training research. | P0 | Does not necessarily require separate packages immediately. |
| R07 | RAG Eval | The strongest commercial/useful wedge is retriever-agnostic evidence localization evaluation inside a CI-testable corpus lifecycle. | Promote a first-class RAG eval harness: query -> gold refs/ranges, run retrieval stack variants, score hit@k/MRR/coverage/neighbor/parent/token cost. | P0 | This is stronger than "Refmark improves retrieval"; it makes retrieval quality measurable and repeatable. |
| R08 | RAG Eval | Refmark should compare arbitrary retrieval stacks, not only its own BM25/index. | Design a `RagEval.compare({...})` style API that accepts external retriever callbacks returning stable refs/chunks. | P0 | Key integration story for existing RAG systems. |
| R09 | RAG Eval | Evidence evaluation is narrower and more reliable than answer judging. | Add README/example showing evidence-localization eval before answer generation: did the right evidence enter context? | P0 | Use BGB as supporting artifact, not main premise. |
| R10 | RAG Eval | Metrics should include coverage and context cost, not only exact hit. | Add/standardize metrics: hit@k, MRR, gold coverage, neighbor hit, parent hit, region precision/recall, overretrieval/context tokens, wrong-section/missing-evidence. | P1 | Some metrics exist; package them around RAG eval. |
| R11 | RAG Eval | Context expansion/parent-neighbor packing is part of the evaluation target. | Implement/document expansion policy evaluation: before/after neighbors, same-parent expansion, section expansion, token cost. | P1 | Bridges deterministic anchor metrics and natural answer usefulness. |
| R12 | Lifecycle | Corpus mutations/churn should be regression-tested with stable refs. | Add corpus lifecycle features: changed/removed/added refs, stale gold refs, examples affected by changed refs, embedding churn, and data-smell diagnostics. | P0 | This is central to the "corpus-as-test-suite" story, not just enterprise polish. |
| R13 | API Design | A canonical region/manifest abstraction would make the system easier to understand. | Design public `Region`, `RegionRange`, `CorpusMap`/`RegionManifest`, `EvalExample`, and `EvalSuite` or equivalent API; keep current internals compatible. | P1 | Needed by corpus CI, RAG eval, and citations. |
| R14 | Library Surface | Users need a simple attach/pass-through interface for existing systems. | Add small API examples: map a corpus, validate/stale-check eval examples, compare retrievers, resolve refs, score retrieval/citations, context-pack refs. | P0 | This is now central to the corpus lifecycle/eval wedge. |
| R15 | MCP | MCP should expose lightweight region discovery. | Ensure `list_ref_regions`/`get_ref_region` style workflow is documented and fast; maybe add `get_ref_region` if absent. | P1 | Current `list_ref_regions` exists, but workflow docs can improve. |
| R16 | MCP | Read/list/apply must share the same shadow registry/session semantics. | Audit MCP in-memory shadow sessions vs filesystem-backed shadow session. Unify or explicitly bridge them. | P1 | Important to preserve ref consistency across tools. |
| R17 | MCP | Writes should have stronger safety controls. | Add/verify dry-run, base source hash/session hash guard, structured errors, changed-region output, preview diff. | P1 | Some safety exists, but reviews consistently ask for stronger contract. |
| R18 | MCP | Logs may contain sensitive source/edit content. | Review MCP logging defaults; consider metadata-only or opt-in full payload logging. | P1 | Important for corporate/internal adoption. |
| R19 | MCP | Region ordering should be ordinal, not string-sorted. | Check `list_ref_regions` ordering and fix if string sort can produce `F10` before `F2`. | P1 | Quick correctness check. |
| R20 | Security | MCP/editor tools should enforce workspace roots/path constraints. | Add allowed-root validation and path traversal checks where local file edit tools run. | P1 | Especially relevant if MCP is public-facing. |
| R21 | Edit Robustness | `apply_ref_diff` should be tested hard around overlap, expected_text, syntax failure, and patch formats. | Add focused unit tests for replace/delete/insert/patch_within/search_replace/unified_diff/overlaps/stale hashes. | P1 | Existing tests are good but can be expanded. |
| R22 | Marker Robustness | Inject/strip roundtrip must be extremely trustworthy. | Add broader fixtures/fuzz tests: Python, TS, Markdown fences, Unicode, comments, docstrings, whitespace. | P1 | Core trust issue. |
| R23 | Marker Detection | Premarked detection can misclassify examples in strings. | Ensure `_detect_premarked` only counts syntactic/live markers, not arbitrary `[@` in string literals/docs; add `force_live`/`force_shadow` override docs. | P1 | We discussed/fixed one version, but keep as follow-up regression area. |
| R24 | Registry | Shadow registry invalidation after external edits is central. | Review hash/config invalidation and refresh behavior; add tests for human edits between read/apply. | P1 | Important for real agent workflows. |
| R25 | Citations | Need a strict citation contract/schema. | Define JSON/Markdown citation grammar and parser: refs, ranges, disjoint sets, doc ids, validation errors. | P1 | Supports hallucination-resistant citation use case. |
| R26 | Citations | Highlight/resolve should be first-class and easy to demo. | Add citation-first cookbook: inject/mark, model cites refs, resolve/highlight, score over/under/wrong-location. | P1 | Strong downstream use case after RAG eval. |
| R27 | Documents | PDF/DOCX extraction should not overpromise original layout citation. | Document that current PDF/DOCX refs resolve to extracted text unless layout/page provenance exists. | P1 | Avoids credibility trap. |
| R28 | Documents | PDF/DOCX parsing may be better as integration/example than core responsibility. | Review dependency/package boundary for `document_io`; consider extras and docs showing upstream parsers. | P2 | Not necessarily remove, but scope clearly. |
| R29 | Retrieval | Refmark is better framed as localization/evaluation layer, not a new search engine. | Adjust docs: it complements BM25/embeddings/RAG and makes results resolvable/testable. | P0 | Matches RAG eval harness framing. |
| R30 | Retrieval | Unicode tokenization matters for German/BGB and multilingual corpora. | Audit `tokenize` in `search_index.py`; add Unicode-aware tokenization and regression tests. | P1 | High value after BGB work. |
| R31 | Retrieval | Neighbor pointers can become stale if regions are filtered. | Check build pipeline and recompute `prev_region_id`/`next_region_id` after filtering/removal. | P1 | Could affect context expansion. |
| R32 | Retrieval | Source omission/metadata-only mode should be explicit. | Add option or docs for index payloads that omit raw source text for privacy/size. | P2 | Useful for browser/package distribution. |
| R33 | Retrieval | Parent/section hierarchy is underdeveloped. | Add parent/section metadata and section-aware context expansion for docs/search/RAG. | P1 | Up-ranked because coarse/parent coverage is central to RAG eval. |
| R34 | Retrieval | Generated view failures need robust retry/error cache behavior. | Add retry/backoff and per-region failure metadata for view/question generation. | P2 | BGB OpenRouter runs showed provider errors matter. |
| R35 | Retrieval | Python/JS BM25 parity should be tested. | Add fixture ensuring `search_index.py` and `browser_search.js` return equivalent ranking on small corpus. | P2 | Important for browser claims. |
| R36 | Browser Search | Browser/local search is promising but should remain an optional demo until UX is polished. | Keep browser exporter, but do not make it the headline. Add minimal browser-ready docs only after cleanup. | P2 | Avoids overloading README. |
| R37 | Training | Tiny resolver is exploratory; strongest value is deterministic supervision/evaluation and stale-data maintenance. | Keep training docs honest; next experiments should compare against Qwen3/refmark embedding baseline and reuse the same `query -> gold refs/ranges` lifecycle artifacts. | P0 | Training is a consumer of the address space, not the current headline claim. |
| R38 | Training | Need hard negatives and coarse-to-fine/range targets. | Design experiments for BM25-near, embedding-near, same-section negatives; exact region vs range vs section targets. | P2 | Research track. |
| R39 | Training | LLM judge labels can complement deterministic labels. | Cache judge labels for usefulness/underbroad/overbroad as secondary labels, not primary eval. | P2 | Matches natural-query judge results. |
| R40 | Training | Cross-domain/cross-corpus tests are needed before training claims. | Run train-on-one-corpus/eval-on-another and mutation-stability experiments. | P2 | Needed to prove/falsify training idea. |
| R41 | Examples | Some examples are too research-heavy for a publishable first impression. | Separate stable examples from research examples in docs and folder naming. | P1 | Could be docs-only initially. |
| R42 | README | README should lead with a tiny corpus-CI/RAG evidence-eval example before theory. | Show gold `query -> policy:P13`, two retrieval pipelines, hit@k/coverage/context-token comparison, then a corpus update marking examples stale. | P0 | Replaces citation-only first example as lead. |
| R43 | README | Add action-oriented quickstarts ordered by product wedge. | "Turn a corpus into an eval suite", "compare retrievers by evidence refs", "detect stale eval examples", "attach refs to existing chunks", "audit citations/highlight evidence", "bounded code edit via MCP". | P0 | MCP remains important but no longer first wedge. |
| R44 | Packaging | Optional/dependency-heavy pieces should be behind extras or examples. | Review `pyproject.toml` dependencies/extras for documents, MCP, search, train separation. | P1 | Avoid package feeling heavy. |
| R45 | Public API | Top-level exports should be smaller and concept-centered. | Review `refmark/__init__.py` and public CLI for too-broad surface; avoid exposing training internals. | P2 | Polish after narrative/API design. |
| R46 | Generated Outputs | Generated outputs and run artifacts must stay out of commits. | Keep/verify `.gitignore` patterns for `examples/*/output_*/`, train runs/data as appropriate. | P0 | Already partly handled; verify before push. |
| R47 | Review Hygiene | Some review claims need verification before action. | Treat model reviews as input, not truth; verify claims like missing assets, current tests, implemented tools. | P0 | Example: `LICENSE` and `refmark.png` already exist. |

## Likely Non-Actions / Do Not Prioritize Now

| Point | Reason |
| --- | --- |
| Move `refmark_train` to a separate repository immediately | Good eventual separation, but current priority is public narrative and package boundary. It can remain clearly experimental for now. |
| Remove all PDF/DOCX support from core immediately | The capability is useful; better first step is clearer extras/dependency and provenance documentation. |
| Claim browser search as primary product | Useful demo, but current strongest story is addressable regions across citation/edit/retrieval/evaluation. |
| Claim broad coding-agent superiority | Current evidence supports bounded edit targeting, not SWE-bench-level superiority. |
| Claim tiny model is current product value | Current resolver gains are not strong enough versus embedding baselines. |
| Claim Refmark itself improves retrieval | Better claim: Refmark makes evidence-region retrieval quality measurable; generated views/metadata can improve specific retrieval stacks. |
