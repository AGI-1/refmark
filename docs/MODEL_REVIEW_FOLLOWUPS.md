# External Model Review Follow-Ups

This note summarizes the 2026-04-29 review pass across:

- `qwen/qwen3.6-max-preview`
- `deepseek/deepseek-v4-pro`
- `xiaomi/mimo-v2.5-pro`
- `moonshotai/kimi-k2.6`
- `minimax/minimax-m2.7`
- `z-ai/glm-5.1`

The models were given a code2prompt bundle focused on the public package,
README, evidence-eval docs, tests, and training findings. Xiaomi and GLM needed
a short "do not overthink" retry; both then returned usable final markdown.
Raw responses are local scratch artifacts under `.refmark/model_reviews/`.

## Consensus

| Theme | Review signal | Status |
| --- | --- | --- |
| Corpus CI is the strongest framing | All usable reviews converged on evidence-region evaluation and corpus lifecycle as the credible wedge. | Accepted framing. |
| `eval-index` must behave like a CI gate | Qwen, Kimi, GLM, Xiaomi, and DeepSeek all asked for threshold flags and nonzero exits. | Implemented: `--min-hit-at-k`, `--min-hit-at-1`, `--min-mrr`, `--min-gold-coverage`, `--max-stale`, `--fail-on-regression`. |
| External retriever scoring is adoption-critical | Kimi and DeepSeek explicitly called out the CLI being locked to the built-in index. | Implemented: `--retriever-endpoint` HTTP scoring path. |
| Offline exported hit scoring lowers integration friction | DeepSeek suggested JSONL input for per-query hits; this is easier than running a service in CI. | Implemented: `--retriever-results`. |
| Manifest/stale checks must be decoupled from the index | DeepSeek identified that stale checks tied to the index weaken corpus lifecycle claims. | Implemented: `--manifest` for current region map validation/staleness. |
| Saved source hashes must survive reload | DeepSeek called out the missing persistence loop. | Implemented: `EvalSuite.to_jsonl`; `with_source_hashes` now preserves existing hashes by default. |
| Gold target shape needs separate reporting | DeepSeek, GLM, Xiaomi, and Minimax asked for single/range/distributed breakdowns. | Implemented as `run.diagnostics["by_gold_mode"]`. |
| Preferred retriever hit shape should be public | Qwen and DeepSeek flagged loose return shapes as confusing. | Partly implemented: `NormalizedHit` is exported. Further adapter docs remain. |

## Still Valuable

| Item | Why it matters | Priority |
| --- | --- | --- |
| Retriever adapter protocol/docs | Makes LangChain, LlamaIndex, vector DB, and custom retriever integration less ad hoc. | P1 |
| Region overlap mapping for existing chunks | Existing RAG systems need to map chunk spans/results back to refs. | P1 |
| Manifest schema versioning | Reduces future breakage as `RegionRecord` evolves. | P1 |
| More `apply_ref_diff` and region parser tests | Multiple reviews flagged edit/parser paths as higher-risk than eval paths. | P1 |
| Context packing by token budget/heading boundary | Needed for real RAG context assembly beyond fixed neighbor expansion. | P1 |
| Parent/section population | Required for section-level gold targets and parent-hit scoring. | P1 |
| `enrich-prompt --answer-format claims` | Makes citation grammar easier for general chat workflows. | P2 |
| Training: hard-negative or concern-alias reranker | Best near-term BGB experiment is targeted, heatmap-driven repair, not broad "tiny model replaces embeddings." | Experimental |

## 2026-04-29 Heatmap/Adaptation Review Pass

A compact follow-up prompt was sent to:

- `qwen/qwen3.6-max-preview`
- `moonshotai/kimi-k2.6`
- `minimax/minimax-m2.7`
- `z-ai/glm-5.1`
- `xiaomi/mimo-v2.5-pro`

The prompt focused on the current evidence-retrieval framework, not the whole
codebase. Qwen returned the cleanest structured JSON; Minimax returned useful
content but truncated before strict JSON closure; the other providers returned
empty content through OpenRouter in this run. Useful repeated themes:

| Theme | Action |
| --- | --- |
| Query-style averages can hide failures | Implemented query-layer heatmap controls in the FastAPI dashboard and core `EvalSuite` diagnostics: `by_query_style` and `query_style_gap`. |
| Local adaptation can regress unrelated rows | Implemented a deterministic blast-radius mini-eval probe before accepting FastAPI shadow-metadata changes. |
| Shadow metadata can overfit generated questions | Documented hold-out and curated/manual probe requirements; keep adaptive and non-adaptive modes visible side by side. |
| Query magnets can dominate retrieval and adaptation | Keep query-magnet roles/exclusions visible in heatmaps; future work should add a magnet-domination metric. |
| Reviewer model can misclassify retrieval failures as gold ambiguity | Keep embedding-teacher disagreement checks as a next implementation target. |
| Alias provenance matters | Current shadow metadata carries provenance; future work should expose alias-level provenance and collision/leakage diagnostics in reports. |

## 2026-04-29 Data-Smell Review Pass

OpenRouter model availability was checked before the run. Relevant low-cost
options available at the time included `deepseek/deepseek-v4-flash`,
`deepseek/deepseek-v4-pro`, `x-ai/grok-4.1-fast`, and `x-ai/grok-4-fast`.
A compact data-smell prompt was sent to:

- `deepseek/deepseek-v4-flash`
- `x-ai/grok-4.1-fast`
- `qwen/qwen3.6-max-preview`

Useful feedback:

| Theme | Action |
| --- | --- |
| Data smells need severity, not only counts | Implemented a deterministic `weighted_smell_score` and severity bucket in `analyze_index_smells`. |
| Exact duplicates and near duplicates are different | Implemented exact duplicate groups beside near-duplicate candidate pairs. |
| Contradictions should start as candidates | Implemented cheap potential-conflict cue pairs; LLM confirmation remains optional future work. |
| Staleness and shadow metadata overfit are separate smell classes | Keep stale-ref checks in `EvalSuite`/`eval-index`; keep blast-radius probing for adaptation overfit. |
| Smells should be dashboard-visible | FastAPI heatmap overview now includes a data-smell block. |

## 2026-04-30 Discovery/Question-Planning Review Pass

A compact discovery-flow prompt was sent to:

- `deepseek/deepseek-v4-flash`
- `x-ai/grok-4.1-fast`
- `qwen/qwen3.6-max-preview`

Useful repeated themes:

| Theme | Action |
| --- | --- |
| Do not cut model prompts through the middle of a region | Keep discovery windows region-safe; current whole-corpus packer already stops between records, but true windowed discovery is still future work. |
| Discovery should feed generation through compact cards | Implemented `DiscoveryContextCard` and `refmark cli discovery-card`; full pipeline question generation now includes the card and hashes it into the question cache key. |
| Local discovery noise needs review queues | Implemented deterministic `review_discovery` and `refmark cli review-discovery` for broad terms, singleton terms, excluded regions, heading-boundary issues, broad query families, and stale refs. |
| Preserve provenance through normalization/clustering | Context cards and review issues keep stable refs/ranges instead of free-form summaries only. |
| Heading detection can pollute roles | Tightened the short-line heading heuristic so one-line definitions ending in punctuation are not marked as headings. |

## Rejected Or Deferred

| Suggestion | Decision |
| --- | --- |
| Make browser search a headline product | Keep as demo/application until eval pipeline is boringly solid. |
| Bundle vector database clients in core | Keep Refmark runtime-agnostic; adapters can live in optional extras or examples. |
| Claim training superiority | Training remains exploratory; evidence eval is the product. |
| Layout-accurate PDF/DOCX citation | Out of current core scope; extracted-text provenance only unless explicitly implemented. |
| Multi-file MCP edits | Defer until same-file semantics and tests are hardened further. |
