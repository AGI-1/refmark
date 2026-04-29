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

## Rejected Or Deferred

| Suggestion | Decision |
| --- | --- |
| Make browser search a headline product | Keep as demo/application until eval pipeline is boringly solid. |
| Bundle vector database clients in core | Keep Refmark runtime-agnostic; adapters can live in optional extras or examples. |
| Claim training superiority | Training remains exploratory; evidence eval is the product. |
| Layout-accurate PDF/DOCX citation | Out of current core scope; extracted-text provenance only unless explicitly implemented. |
| Multi-file MCP edits | Defer until same-file semantics and tests are hardened further. |
