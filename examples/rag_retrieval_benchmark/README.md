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
python examples/rag_retrieval_benchmark/generate_views.py --source openrouter --limit 20 --sample-mode even --concurrency 4 --output examples/rag_retrieval_benchmark/output/views_openrouter_probe_20.jsonl --questions-per-anchor 3
python examples/rag_retrieval_benchmark/validate_views.py --views-jsonl examples/rag_retrieval_benchmark/output/views_openrouter_probe_20.jsonl
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

## Side Benefit: Testable Naive RAG

Refmark does not require replacing an existing RAG stack. If a system keeps its
current naive chunking, adding refmark ids to chunk text or metadata already
turns anonymous retrieved blobs into testable evidence:

- retrieval tests can ask whether chunks contain the expected refs, not just
  whether answer text looks plausible
- citation validation can detect refs outside the retrieved evidence
- source churn can be localized by region hash instead of re-embedding or
  re-reviewing a whole document blindly
- repeated retrieval runs can be compared by stable evidence ids
- anomaly checks, such as data smells, duplicate regions, missing refs, stale
  manifests, or unusually broad chunks, become concrete corpus tests

This mode does not promise better ranking by itself. Its value is that the
existing ranking behavior becomes measurable and reviewable. The stronger mode
is evidence resolution: use refmarks actively to narrow a retrieved page or
chunk into an exact range, expand deterministic context, and score
overcitation/undercitation.

Run an embedding/vector-store probe:

```bash
python examples/rag_retrieval_benchmark/embedding_benchmark.py --limit 300 --unit-mode anchors --source openrouter --model perplexity/pplx-embed-v1-0.6b
python examples/rag_retrieval_benchmark/embedding_benchmark.py --limit 300 --unit-mode enriched --source openrouter --model perplexity/pplx-embed-v1-0.6b
```

`embedding_benchmark.py` caches embeddings in JSONL and evaluates cosine search
over normalized vectors. The cache is deliberately simple: it behaves like a
small local vector store for experiments and can later be swapped for FAISS,
Qdrant, sqlite-vss, or an application-owned store.

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

Larger local corpus smoke checks:

| Corpus | Examples | Method | hit@1 | hit@5 | MRR | avg tokens @1 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| corporate | 2,880 | naive fixed chunks | 0.6122 | 0.8865 | 0.7295 | 237.42 |
| corporate | 2,880 | refmark regions | 0.6517 | 0.8885 | 0.7546 | 40.62 |
| corporate | 2,880 | refmark + train-question enrichment | 0.8816 | 0.9931 | 0.9317 | 40.56 |
| medical | 4,628 | naive fixed chunks | 0.4989 | 0.8274 | 0.6324 | 236.26 |
| medical | 4,628 | refmark regions | 0.6333 | 0.8758 | 0.7379 | 39.32 |
| medical | 4,628 | refmark + train-question enrichment | 0.8838 | 0.9907 | 0.9302 | 38.66 |
| legal | 4,328 | naive fixed chunks | 0.5437 | 0.7736 | 0.6398 | 244.17 |
| legal | 4,328 | refmark regions | 0.5615 | 0.7743 | 0.6515 | 41.00 |
| legal | 4,328 | refmark + train-question enrichment | 0.8403 | 0.9794 | 0.9007 | 41.66 |
| combined nonfiction | 8,000 | naive fixed chunks | 0.5854 | 0.8193 | 0.6832 | 239.40 |
| combined nonfiction | 8,000 | refmark regions | 0.6730 | 0.8464 | 0.7492 | 36.62 |
| combined nonfiction | 8,000 | refmark + train-question enrichment | 0.8654 | 0.9810 | 0.9164 | 36.93 |

Early embedding result on the documentation corpus with
`perplexity/pplx-embed-v1-0.6b`:

| Examples | Unit mode | hit@1 | hit@5 | MRR |
| ---: | --- | ---: | ---: | ---: |
| 300 | raw refmark regions | 0.4300 | 0.7267 | 0.5426 |
| 300 | refmark + train-question enrichment | 0.7400 | 0.9567 | 0.8290 |

That result should not be read as an "embeddings beat BM25" contest. The more
useful product shape is that enriched refmark views make each region easier to
find, and those views can feed either lexical search, embeddings, or a hybrid
backend. BM25 is especially interesting because it keeps query-time search tiny
enough to embed inside a package or documentation viewer.

Full combined-corpus embedding check:

| Examples | Unit mode | hit@1 | hit@5 | hit@10 | MRR |
| ---: | --- | ---: | ---: | ---: | ---: |
| 15,708 | raw refmark regions | 0.3846 | 0.6360 | 0.7206 | 0.4915 |
| 15,708 | refmark + train-question enrichment | 0.7643 | 0.9496 | 0.9766 | 0.8437 |

The embedding cache uses a simple lock file so parallel benchmark processes do
not corrupt the JSONL cache.

Candidate refinement and uncertainty:

```bash
python examples/rag_retrieval_benchmark/refinement_uncertainty.py --data-dir refmark_train/data/documentation_full_paragraph_contextual_idf_lean2 --negative-data-dir refmark_train/data/corporate_full --negative-data-dir refmark_train/data/medical_full --negative-data-dir refmark_train/data/legal_full
```

This trains a tiny logistic reranker over top-k BM25 candidates and then samples
the reranker repeatedly. The repeated samples do not improve exact-hit on the
current synthetic single-ref task, but they expose useful stability signals.

| Corpus | Eval | candidate recall@10 | single hit | vote hit | avg max prob | avg vote entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| documentation | 1,000 | 0.9910 | 0.8710 | 0.8710 | 0.9278 | 0.1379 |
| combined nonfiction | 1,000 | 0.9970 | 0.8720 | 0.8720 | 0.9329 | 0.1184 |

For docs-vs-other-domain no-answer probes, the same reranker gives a
max-probability AUC around `0.85`. With noisier 25-sample self-consistency,
vote-share AUC rises to `0.7920` and low-entropy AUC to `0.7734`, which suggests
that scattered repeated citations are a useful absence-of-citation signal.

The current citation shape is still single-ref: undercitation is simply
`1 - hit`, while overcitation is mostly represented by candidate breadth
(`candidate_k` refs before refinement). Multi-ref/range examples are the next
step for real overcitation, undercitation, breadth, and density metrics.

Hybrid retrieval against classical chunks:

```bash
python examples/rag_retrieval_benchmark/hybrid_retrieval.py --data-dir refmark_train/data/combined_nonfiction_full --limit 3000 --source openrouter --model perplexity/pplx-embed-v1-0.6b --hybrid-alpha 0.8
```

On a 3,000-example combined-corpus sample, a BM25-heavy hybrid improves the
naive chunk baseline slightly, but refmark-enriched retrieval remains much more
precise and much cheaper to inspect:

| Method | hit@1 | hit@5 | hit@10 | MRR | avg tokens @1 | avg refs @1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| naive chunks, BM25 | 0.5760 | 0.8187 | 0.8910 | 0.6828 | 239.33 | 6.94 |
| naive chunks, embedding | 0.2380 | 0.4617 | 0.5597 | 0.3421 | 240.08 | 6.44 |
| naive chunks, hybrid | 0.6027 | 0.8357 | 0.8967 | 0.7048 | 239.83 | 6.83 |
| refmark regions, BM25 | 0.6630 | 0.8447 | 0.8907 | 0.7465 | 36.67 | 1.00 |
| refmark regions, embedding | 0.3863 | 0.6423 | 0.7200 | 0.4989 | 38.54 | 1.00 |
| refmark regions, hybrid | 0.6833 | 0.8650 | 0.9033 | 0.7650 | 36.15 | 1.00 |
| refmark-enriched, BM25 | 0.8580 | 0.9790 | 0.9943 | 0.9108 | 37.01 | 1.00 |
| refmark-enriched, embedding | 0.7697 | 0.9470 | 0.9757 | 0.8483 | 36.02 | 1.00 |
| refmark-enriched, hybrid | 0.8670 | 0.9860 | 0.9980 | 0.9188 | 37.03 | 1.00 |

This is the first result aimed at the stronger claim: refmark metadata can help
push more retrieval quality out of the same corpus and same cheap retrieval
backends. It is still a retained synthetic benchmark, so the product-facing
path is now `examples/portable_search_index`: build enriched region metadata
from real user corpora, then search the resulting JSON index locally.
