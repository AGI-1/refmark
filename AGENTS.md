# AGENTS.md

Internal guidance for AI agents working in this repository.

Refmark is not primarily a training experiment, browser search engine, or MCP
editor. Those are applications of the core idea: Refmark gives text, documents,
code, and corpora stable addressable regions so models and tools can cite, edit,
retrieve, evaluate, and train against concrete refs/ranges. The strongest
product framing is corpus-as-test-suite: a refmarked corpus becomes testable
infrastructure for retrieval, citations, corpus drift, training data refresh,
and human review. The clearest near-term wedge inside that framing is RAG
evidence evaluation: use those refs/ranges to measure whether any retrieval
pipeline found the right supporting evidence.

## Repository Shape

- `refmark/` is the publishable package surface. Keep it lightweight and tested.
- `refmark_train/` is research/prototype work. Do not make training the main
  public claim unless a task explicitly asks for that direction.
- `examples/` contains demos and experiments. Do not treat every demo capability
  as a release promise.
- `docs/` contains public and internal design notes. Keep public-facing docs
  honest about maturity and limits.
- Generated outputs under `examples/*/output_*`, benchmark caches, model
  handoff bundles, and `review_*.txt` files are research artifacts, not product
  source.

## Current Product Framing

Use this framing unless the user changes direction:

> Refmark creates addressable regions for documents and code. A model can refer
> to `P03-P05` or `F04`; Refmark resolves that address to exact source spans for
> highlighting, deterministic edits, retrieval context, citation scoring, or
> training labels.

For product/README work, lead with corpus CI and the RAG evidence-evaluation
harness:

> Refmark turns documents into stable evidence regions so teams can evaluate
> any RAG stack by whether it retrieves the correct support refs/ranges, then
> detect which evaluation or training examples became stale after corpus
> changes. It is independent of BM25, embeddings, rerankers, query rewriting,
> or vector database choice.

Preferred terms:

- **Marker**: visible injected token, such as `[@P03]`.
- **Region id / ref**: symbolic id, such as `P03`.
- **Stable ref**: fully qualified id, such as `policy:P03`.
- **Region**: the source span addressed by one ref.
- **Range**: ordered set of regions, usually inclusive for citations.
- **Registry/manifest**: mapping from refs to source spans and metadata.
- **Shadow mode**: hidden marked view/registry without modifying source files.

Avoid mixing `anchor`, `marker`, `region`, and `refmark` casually. In public docs,
lead with regions and refs, not marker syntax.

## Maturity Boundaries

Credible today:

- Region marking, resolving, highlighting, and citation scoring.
- Evidence-region evaluation for retrieval/RAG pipelines.
- Corpus lifecycle evaluation as a direction: stable refs make changed,
  removed, stale, and ambiguous evidence examples inspectable. Be precise about
  which pieces are already implemented.
- Bounded same-file edits via `apply_ref_diff` / MCP, with caveats.
- Portable search indexes and browser export as optional examples.
- Refmark-enriched retrieval as a measurable localization/evaluation layer.

Promising but not a headline claim:

- Browser/local search as a polished product.
- Corpus-specific tiny resolver training.
- Broad superiority over coding agents, search engines, or vector databases.
- PDF/DOCX original-layout citation. Current support is primarily extracted-text
  oriented unless page/layout provenance is explicit.

When adding claims, ground them in a reproducible artifact or mark them as
experimental.

## Engineering Rules

- Keep changes scoped. Do not refactor unrelated modules.
- Preserve user or prior-agent work in the dirty tree.
- Prefer `rg`/`rg --files` for search.
- Use `apply_patch` for manual edits.
- Do not commit or push unless explicitly asked.
- Keep generated benchmark/model artifacts out of commits.
- If a demo requires API keys or remote models, cache results and handle provider
  failures explicitly.
- For provider/model comparisons, preflight a tiny request before a long run.
- For judge runs, report judge model, success/failure rate, and whether fallback
  judges were used.

## WSL Codex/MCP Checks

When validating Codex inside WSL, use the Linux npm Codex CLI installed through
nvm, not the packaged Windows desktop binary under `C:\Program Files\WindowsApps`.
The WindowsApps binary is not the intended WSL launch path and may fail with
`Permission denied`, which is not evidence that WSL Codex is unavailable.

Correct non-interactive WSL shape:

```bash
source /home/alkon/.nvm/nvm.sh
CODEX_HOME=/home/alkon/.codex-refmark codex mcp list
CODEX_HOME=/home/alkon/.codex-refmark codex mcp get refmark_apply_ref_diff
```

The expected refmark profile is `/home/alkon/.codex-refmark/config.toml`.
It should expose all three MCP tools:

- `read_refmarked_file`
- `list_ref_regions`
- `apply_ref_diff`

The MCP server should run via the WSL venv Python at
`/home/alkon/.venvs/refmark-mcp/bin/python` with `PYTHONPATH=/mnt/c/aider`.
For benchmark runs, prefer `python C:\aider\scripts\run_codex_wsl_bench.py ...`;
that runner already sources nvm before `codex exec`.

## Testing Expectations

Keep the repo clean after each iteration.

- For code changes in `refmark/`, run focused tests and then `python -m pytest -q`
  when feasible.
- For example-script changes, run `python -m py_compile` on changed scripts plus
  a small smoke command if practical.
- For docs-only changes, run at least a sanity read or `git diff --check`.
- Always report what was and was not verified.

High-value regression areas:

- Inject/strip round trips, especially whitespace, Markdown fences, Unicode,
  Python comments/docstrings, and TypeScript.
- Ref/range parsing and inclusive vs exclusive range semantics.
- `apply_ref_diff` atomicity, stale refs, overlaps, expected text, and syntax
  validation.
- Shadow registry consistency after external file edits.
- MCP read/list/apply using the same region namespace.
- Unicode tokenization and Python/JS browser-search parity.

High-value product/API roadmap areas:

- `CorpusMap` / region manifest stability.
- `EvalSuite` schema for `query -> gold refs/ranges`.
- Stale-example detection after corpus updates.
- Retriever adapter interface and `compare({...})` reports.
- Context expansion scoring: exact region, neighbor hit, parent hit, token cost.
- CI command that exits nonzero on retrieval/evidence regression.

## Documentation Rules

- Lead with one concrete example before theory.
- Prefer the tagline "turn your corpus into a regression test suite for
  retrieval" when it fits the audience.
- Separate stable package features from demos and research.
- Be explicit about range semantics:
  - citation ranges like `P03-P05` are inclusive;
  - edit boundary spans may be exclusive and should say so.
- For PDF/DOCX, say refs resolve to extracted text unless page/layout
  provenance is explicitly implemented.
- For BGB/training demos, frame the value as instant corpus navigation and
  evidence-eval instrumentation; do not imply the tiny model is the core
  product.
- Do not overclaim hallucination prevention. Refmark makes refs resolvable and
  scoreable; it does not guarantee the model chose the right evidence.
- Do not overclaim training value. Current training work is exploratory.

## Review Artifacts

The repository may contain `review_*.txt`, `*_review*.txt`, model critiques, and
generated `docs/refmark_model_handoff*.md` bundles. Treat them as input material,
not ground truth. Verify claims before acting; some reviews may reference stale
or nonexistent repository state.
