# Full Pipeline Configuration

The full evidence-retrieval pipeline has more moving parts than the small
no-infra navigation example. A single config file should control model tiers,
caches, budgets, and loop stopping criteria so runs are repeatable.

Create a template:

```bash
python -m refmark.cli init-pipeline-config -o refmark_pipeline.yaml --check
```

Validate an edited config:

```bash
python -m refmark.cli check-pipeline-config refmark_pipeline.yaml
```

Run the easy-mode pipeline:

```bash
python -m refmark.cli run-pipeline refmark_pipeline.yaml
```

That command writes a shadow manifest, section map, discovery file, generated
question suite, portable index, browser index, eval report, and run summary into
`artifacts.output_dir`. It reuses existing artifacts by default; set
`artifacts.overwrite: true` when you intentionally want to regenerate them.

Query a processed knowledge source:

```bash
python -m refmark.cli query-pipeline .refmark/pipeline \
  "How do I configure CORS for browser clients?" \
  --top-k 5
```

The query result includes the concrete hit ref, source path, snippet, neighbor
context refs, and the containing section/range from `sections.json` when
available. That gives callers both granular region navigation and article-level
navigation from the same processed output.

The config separates three model tiers:

| Tier | Purpose | Default shape |
| --- | --- | --- |
| `question_generation` | generate query -> gold-ref examples and stress questions | OpenRouter chat model, parallel, cached |
| `retrieval_views` | generate summaries/questions/keywords attached to refs | cheap OpenRouter chat model, cached by ref hash |
| `judge` | optional natural-query/result review | disabled by default, smarter model when enabled |

Embeddings are listed separately because a run may evaluate several models or
leave them disabled entirely. The no-infra runtime path remains valid when
`include_embeddings: false`.

Important idempotency rules:

- generated views, questions, judgements, and embeddings must be cached by
  source ref/range hash plus provider/model/options;
- output artifacts should be written to a run directory without overwriting
  prior reports unless `artifacts.overwrite` is true;
- eval reports include provenance for corpus fingerprint, config hash, split
  seed, and source artifact hashes;
- provider failures should be explicit in reports, not silently converted into
  successful quality claims.

Budget controls live under `budget`. Token counts are approximate unless a
provider-specific tokenizer is available, but even approximate input/output
tracking is useful for repeatability and cost reviews.

The intended full loop is:

```text
corpus
-> shadow manifest + section map
-> discovery
-> question/retrieval-view generation
-> candidate indexes/retrievers
-> eval reports + heatmap
-> adapt/regenerate
-> frozen searchable artifact
```

Most users will consume only slices of this pipeline. The config exists so the
full run can be reproduced when a team wants to compare chunking, generated
views, embeddings, rerankers, or adaptation loops against the same ref targets.
