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

Discovery is configured separately from model tiers because it defines the
planning surface for question generation and review. Small corpora can use one
whole-corpus pass. Larger corpora should use region-safe windows so model
context limits never split a refmark region:

```yaml
discovery:
  mode: windowed
  source: local
  window_tokens: 40000
  overlap_regions: 2
  review_enabled: true
```

The runner writes both `discovery.json` and `discovery_review.json`. The review
artifact is deterministic today: it flags noisy broad terms, oversized regions,
empty windows, broad clusters, unclustered refs, and stale range candidates
before those issues can quietly shape generated questions.

Question planning is also an explicit artifact. By default each selected region
gets one `direct`, one `concern`, and one `adversarial` query request:

```yaml
question_plan:
  direct_per_region: 1
  concern_per_region: 1
  adversarial_per_region: 1
  include_excluded: false
```

The runner writes `question_plan.json` before calling any generator. Generated
eval rows preserve `metadata.query_style`, so reports and heatmaps can separate
easy lexical lookup from user-concern wording and lower-overlap paraphrases.

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
-> explicit question plan
-> question/retrieval-view generation
-> candidate indexes/retrievers
-> eval reports + heatmap
-> adapt/regenerate
-> frozen searchable artifact
```

Most users will consume only slices of this pipeline. The config exists so the
full run can be reproduced when a team wants to compare chunking, generated
views, embeddings, rerankers, or adaptation loops against the same ref targets.
