# Corpus Eval Flow

This is the core Refmark loop for a documentation corpus. It keeps model calls
optional: Refmark prepares addressable evidence and evaluation artifacts; you
can send prompts to any LLM endpoint or curate the rows by hand.

## 1. Map A Corpus

```bash
python -m refmark.cli map docs/ -o corpus.refmark.jsonl --marked-dir marked_docs/
```

The manifest contains stable refs such as `guide.md:P03`, source hashes,
neighbor refs, and optional parent refs.

## 2. Pack Evidence For A Region Or Range

```bash
python -m refmark.cli pack-context corpus.refmark.jsonl \
  --refs "guide.md:P03-guide.md:P06" \
  --format text
```

Use this when you want a deterministic evidence bundle for review, prompt
generation, or context injection.

## 3. Discover Corpus Context

```bash
python -m refmark.cli discover corpus.refmark.jsonl \
  -o corpus.discovery.json \
  --source local
```

Discovery records corpus summaries, terms, abbreviations, region roles,
candidate ranges, and training exclusions. Local discovery is deterministic;
model-backed discovery can be run with `--source openrouter` when a whole corpus
or section fits in context.

For the broader loop and hierarchical plan, see
`docs/DISCOVERY_ADAPT_LOOP.md`.

## 4. Generate Or Curate Questions

```bash
python -m refmark.cli question-prompt corpus.refmark.jsonl \
  --refs "guide.md:P03-guide.md:P06" \
  --language English \
  --count 5 \
  -o prompts/guide_p03_p06_questions.txt
```

Override the prompt with `--template`. Supported template fields:

- `{refs}`: comma-separated stable refs.
- `{refs_json}`: JSON list of stable refs.
- `{context}`: packed evidence text.
- `{language}`: requested question language.
- `{count}`: requested question count.

The generated or curated JSONL rows become the eval suite:

```json
{"query":"...","gold_refs":["guide.md:P03","guide.md:P04"],"notes":"..."}
```

## 5. Build A Searchable Index

```bash
python -m refmark.cli build-index docs/ \
  -o corpus.index.json \
  --source local
```

Use `--source openrouter --view-cache views.jsonl` when you want generated
retrieval views. Keep API keys in environment variables.

## 6. Evaluate Retrieval

```bash
python -m refmark.cli eval-index corpus.index.json eval_questions.jsonl \
  --top-k 10 \
  --expand-after 1 \
  --provenance-out runs/local_bm25.provenance.json \
  -o runs/local_bm25.json
```

The report includes hit@1, hit@k, MRR, gold coverage, region precision,
average context refs, stale examples, per-query retrieved/context refs, and a
self-checking provenance block with input hashes and settings hashes. Re-run
with `--expect-provenance runs/local_bm25.provenance.json` when you need to
prove that the index, examples, and evaluation knobs are still the same.

The report also includes two operational diagnostics:

- `diagnostics.heatmap`: hard refs, missed queries, and repeated gold-ref ->
  wrong-top-ref confusions.
- `diagnostics.selective_jump`: score-margin thresholds with accepted coverage
  and precision, useful for deciding when a UI can jump directly versus showing
  candidates.
- `diagnostics.adaptation`: first-pass actions such as adding
  query/alias coverage, reviewing region granularity, or adding hard negatives
  for repeated confusions.

Those diagnostics are the first productized form of the adapt loop:

```text
evaluate -> heatmap/confidence gate -> adapt hard zones -> regenerate stale rows -> evaluate again
```

## 7. Use The Same Refs Elsewhere

The same stable refs support citation parsing and highlighting, RAG context
packing, stale-example detection after corpus updates, MCP/editor-region
operations, and training labels for corpus-specific resolvers.

The point is not that Refmark replaces BM25, embeddings, rerankers, or LLMs. It
gives those systems a shared, testable evidence address space.

## 8. Add Motivation Metadata When Needed

Some real user questions do not share words with the correct evidence. Refmark
can make that visible because the wrong top refs and missed gold refs are
addressable.

One adaptation is to attach retrieval-only aliases to the target refs/ranges:

```json
{
  "query": "I bought a cellphone and found it is broken. What should I do?",
  "gold_refs": ["bgb:S_434", "bgb:S_437", "bgb:S_439"],
  "aliases": ["defective phone", "repair replacement refund", "kaputtes Handy"]
}
```

Those aliases may feed BM25 views, embedding text, question generation, or a
small resolver's training rows. They are not citation evidence; the final answer
must still cite real corpus refs/ranges.

## Large Evidence-Retrieval Pipeline

For the current large-corpus benchmark shape, see
`docs/EVIDENCE_RETRIEVAL_PIPELINE.md`. It documents a reproducible local
Markdown run over a retained OSHA legal corpus above 200k tokens, plus the
stronger BGB run with bilingual generated views, Qwen3 embeddings, natural-query
judging, and a small trained resolver.

The key evolution from a simple `query -> gold_ref` suite is to keep target
shape explicit:

```json
{"query":"...","gold_refs":["manual:P10"],"metadata":{"gold_mode":"single"}}
{"query":"...","gold_refs":["manual:P10-manual:P12"],"metadata":{"gold_mode":"range"}}
{"query":"...","gold_refs":["manual:P10","manual:P44"],"metadata":{"gold_mode":"distributed"}}
```

CI reports should split those modes. A single-region hit, a contiguous range
that needs expansion, and a distributed answer that only gets one support region
are different engineering signals.

## Retrieval CI Gates

Treat the corpus manifest and eval suite as test inputs:

- validate that every gold ref/range still resolves;
- mark examples stale when stored source hashes changed;
- compare methods by hit@k, MRR, gold coverage, precision, undercitation,
  overcitation, breadth, and token/context cost;
- report generated-view provider/model/cache hashes so runs are reproducible;
- fail or warn when a protected baseline regresses beyond an agreed threshold.

Current CLI gate:

```bash
python -m refmark.cli eval-index docs.refmark-index.json eval_questions.jsonl \
  --manifest corpus.refmark.jsonl \
  --top-k 10 \
  --min-hit-at-k 0.80 \
  --max-stale 0 \
  --fail-on-regression
```

For an existing retrieval service, keep the same eval suite and current
manifest, but replace the built-in index search with an HTTP retriever:

```bash
python -m refmark.cli eval-index docs.refmark-index.json eval_questions.jsonl \
  --manifest corpus.refmark.jsonl \
  --retriever-endpoint http://localhost:8000/retrieve \
  --min-hit-at-k 0.80 \
  --fail-on-regression
```

For batch systems, export retriever hits as JSONL and score the file directly:

```jsonl
{"query":"...","hits":[{"stable_ref":"manual:P10","score":0.91}]}
```

```bash
python -m refmark.cli eval-index docs.refmark-index.json eval_questions.jsonl \
  --manifest corpus.refmark.jsonl \
  --retriever-results exported_hits.jsonl \
  --min-hit-at-k 0.80
```

Training is one consumer of the same loop. A trained resolver should be accepted
only if it beats the configured retrieval baseline on the held-out suite without
breaking coverage or overcitation limits.
