# RAG Retrieval Benchmark

This example compares classical fixed-window chunk retrieval with Refmark
region retrieval on the retained `refmark_train` documentation corpus.

It is intentionally CPU-only and API-free. The goal is to measure whether
Refmark regions and retained anchor supervision improve retrieval quality
before adding heavier embedding or neural-reranking systems.

Run the baseline benchmark:

```bash
python examples/rag_retrieval_benchmark/run.py
python examples/rag_retrieval_benchmark/run.py --split reformulated --limit 100000
```

Generate and validate retrieval views:

```bash
python examples/rag_retrieval_benchmark/generate_views.py --source local --output examples/rag_retrieval_benchmark/output/views_local_full.jsonl
python examples/rag_retrieval_benchmark/validate_views.py --views-jsonl examples/rag_retrieval_benchmark/output/views_local_full.jsonl
python examples/rag_retrieval_benchmark/run.py --views-jsonl examples/rag_retrieval_benchmark/output/views_local_full.jsonl --view-name local_views_full
```

Probe an OpenRouter-backed generator, if `OPENROUTER_API_KEY` is set:

```bash
python examples/rag_retrieval_benchmark/generate_views.py --source openrouter --limit 5 --output examples/rag_retrieval_benchmark/output/views_openrouter_probe_5.jsonl --questions-per-anchor 3
python examples/rag_retrieval_benchmark/validate_views.py --views-jsonl examples/rag_retrieval_benchmark/output/views_openrouter_probe_5.jsonl
```

Stress retrieval under synthetic distractor scale:

```bash
python examples/rag_retrieval_benchmark/run.py --distractor-copies 4 --limit 1000
```

Compared retrieval views:

- `naive_fixed_chunks`: overlapping fixed-token chunks over the anchored corpus.
- `refmark_regions`: one retrieval unit per Refmark region.
- `refmark_regions_plus_neighbor_expansion`: retrieve by region, then include
  adjacent refs as deterministic context.
- `refmark_regions_enriched_with_train_questions`: index each region together
  with generated training questions for that region.
- `--views-jsonl`: index cached generated summaries, questions, and keywords.

The benchmark reports hit@k, MRR, average token cost, average returned refs, and
sample misses. Chunk hit rates should always be read together with token cost
and returned-ref count because large chunks can include the gold ref while
returning much more context.

Representative full retained-corpus results:

| Split | Method | hit@1 | hit@5 | MRR | avg tokens @1 | avg refs @1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| valid | naive fixed chunks | 0.6785 | 0.8715 | 0.7587 | 236.41 | 8.53 |
| valid | refmark regions | 0.7565 | 0.9332 | 0.8297 | 28.85 | 1.00 |
| valid | refmark + neighbors | 0.7738 | 0.9403 | 0.8425 | 84.52 | 3.00 |
| valid | refmark + train-question enrichment | 0.8620 | 0.9832 | 0.9140 | 27.70 | 1.00 |
| reformulated | naive fixed chunks | 0.6504 | 0.9047 | 0.7579 | 233.69 | 8.83 |
| reformulated | refmark regions | 0.7259 | 0.9084 | 0.8046 | 27.50 | 1.00 |
| reformulated | refmark + neighbors | 0.7445 | 0.9131 | 0.8173 | 81.94 | 3.00 |
| reformulated | refmark + train-question enrichment | 0.8624 | 0.9881 | 0.9165 | 28.06 | 1.00 |

The first practical takeaway is that Refmark regions give higher precision at
far lower token cost than naive chunks on this corpus. The second is that
question/claim views generated during training can be reused as retrieval
metadata and produce a large retrieval lift without neural training.

Additional local-view checks:

- full local extractive/question views validate their own generated questions
  at `0.9327` hit@1 and `0.9982` hit@5.
- on the full validation split, `local_views_full` reaches `0.8262` hit@1 and
  `0.9751` hit@5. This is above raw Refmark regions, but below the retained
  train-question view, which suggests the generated-view quality matters.
- a 5-anchor OpenRouter probe produced 15 questions and validated at `1.0`
  hit@1. This is only a path check, not a benchmark claim.

The next useful experiment is a modest OpenRouter batch over a representative
anchor sample, followed by view validation and scaled retrieval. Keep the
generated `views.jsonl` files cached; the region hash in each row is the hook
for later stale-view detection when a corpus changes.
