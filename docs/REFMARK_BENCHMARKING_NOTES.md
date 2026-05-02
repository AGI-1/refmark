# Refmark Benchmarking Notes

This is a research notebook, not a release claim sheet. It records intermediate
experiments, failed ideas, and hardware-specific notes so future work can resume
without reconstructing the path. Public-facing claims should cite the smaller
reproducible examples or explicitly say when a result is preliminary.

Working question:

> Does Refmark-based evaluation let us build retrieval systems that perform
> better at finding relevant information?

This is an experiment track, not a product claim. The current goal is to compare
ordinary document retrieval against addressable-region retrieval and parent
aggregation on external corpora where held-out qrels already exist.

A second, more Refmark-native question has emerged:

> Do stable refs/ranges make retrieved evidence more durable and auditable when
> the corpus changes?

## Current Harness

Ubuntu experiment host:

- GPU: NVIDIA RTX 3090, 24 GB VRAM.
- Python venv: `/srv/codex-work/venvs/refmark-bench`.
- Datasets: BEIR through `ir_datasets`.
- Fast evaluator: `examples/external_qa_benchmark/evaluate_beir_refmark_fast.py`
  on the Ubuntu research checkout.

Compared variants:

- `doc_tfidf`: document-level sparse lexical baseline.
- `doc_enriched_tfidf`: document-level lexical with source-only title/keyword
  expansion.
- `refmark_region_tfidf`: retrieve addressable child regions directly.
- `refmark_region_doc_max_tfidf`: retrieve child regions, rank parent document
  by best child-region score.
- `refmark_enriched_region_doc_max_tfidf`: same, with source-only region
  metadata.
- `doc_embedding`: document-level sentence-transformer retrieval.
- `refmark_region_doc_max_embedding`: embedding retrieval over child regions,
  parent document ranked by best child-region score.
- `hybrid_doc_region_embedding_tuned`: dev-selected weighted blend of document
  embedding score and best child-region embedding score.

No benchmark questions were used for enrichment or training in these runs.
For adaptation runs, benchmark questions are split deterministically into dev
and held-out halves; selected weights/settings are learned only from dev.

## Preliminary Results

All numbers are document-level hit@10 unless otherwise stated.

| Dataset | Size | Baseline | Refmark Region Variant | Result |
| --- | --- | --- | --- | --- |
| SciFact | 5,183 docs, 300 queries, 14,498 regions | MiniLM doc embedding `0.793` | MiniLM region->doc max `0.833` | region evidence helped |
| NFCorpus | 3,633 docs, 323 queries, 10,995 regions | MiniLM doc embedding `0.690` | MiniLM enriched region->doc max `0.712` | region evidence helped |
| FiQA | 57,638 docs, 648 queries, 106,740 regions | MiniLM doc embedding `0.657` | MiniLM region->doc max `0.668` | region evidence helped slightly |
| ArguAna | 8,674 docs, 1,406 queries, 19,157 regions | TF-IDF doc `0.792` | TF-IDF region->doc max `0.775` | no win |
| ArguAna | 8,674 docs, 1,406 queries, 19,157 regions | MiniLM doc embedding `0.765` | MiniLM region->doc max `0.740` | no win |

SciFact with BGE-small shows the opposite tradeoff:

| Variant | hit@1 | hit@10 | MRR |
| --- | ---: | ---: | ---: |
| BGE-small doc embedding | `0.603` | `0.857` | `0.688` |
| BGE-small region->doc max | `0.610` | `0.843` | `0.694` |

Interpretation: a stronger document embedder can already capture enough context
that regionization improves rank sharpness but not top-k recall. This is useful
evidence against overclaiming.

## Region Granularity Signal

SciFact with MiniLM:

| Region setting | Regions | hit@1 | hit@10 | MRR |
| --- | ---: | ---: | ---: | ---: |
| 60 tokens / 40 stride | 27,747 | `0.550` | `0.810` | `0.644` |
| 110 tokens / 80 stride | 14,498 | `0.533` | `0.833` | `0.636` |
| 220 tokens / 160 stride | 7,527 | `0.500` | `0.813` | `0.615` |

This is a real Refmark design knob. Smaller regions sharpen top-1/MRR, while
mid-sized regions gave the best top-10 recall on this corpus.

Region-size sweeps were then promoted into the dev/held-out loop. The sweep
tried `60/40`, `110/80`, and `220/160` token/stride settings, selected the best
setting on dev, and reported held-out metrics.

| Dataset | Objective | Selected Region | Doc hit@10 | Region hit@10 | Tuned Hybrid hit@10 | Hybrid MRR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SciFact | hit@10 | `110/80` | `0.767` | `0.820` | `0.813` | `0.642` |
| NFCorpus | hit@10 | `220/160` | `0.727` | `0.739` | `0.752` | `0.516` |
| FiQA | balanced | `110/80` | `0.614` | `0.627` | `0.633` | `0.431` |

Interpretation: region size is corpus- and objective-dependent. NFCorpus wanted
larger regions for top-10 recall but smaller regions for MRR; FiQA preferred the
mid-sized setting. This supports treating granularity as a benchmarked corpus
configuration rather than a hard-coded default.

## What Failed

- Naive summing of all child-region scores into a parent document is harmful.
  It over-rewards documents with many moderately matching regions.
- Source-only keyword enrichment is mixed. It helps some lexical cases but can
  hurt embedding retrieval by adding noisy repeated terms.
- A 100-query ArguAna slice looked positive, but the full run did not. Sliced
  benchmarks are useful for debugging only, not claims.
- Refmark region retrieval alone does not automatically beat document retrieval.
  The aggregation policy, corpus type, embedder, and region size all matter.

## Hardware / Infra Boundaries

- The original pure-Python BM25 loop became the bottleneck before training or
  embeddings did.
- Vectorized TF-IDF made full FiQA feasible: 57k documents, 106k regions, and
  11M source tokens in roughly 30 seconds lexical-only.
- MiniLM embedding on full FiQA took roughly 200 seconds on the RTX 3090,
  including four unit sets. This is suitable for build-time evaluation, not a
  browser runtime.
- Larger BEIR corpora or multiple embedding models should use cached embeddings
  and run artifacts before becoming routine CI.

## Current Takeaway

Refmark-based evaluation is already useful because it shows which retrieval
choices help or hurt at the evidence-address level. Early external benchmarks
suggest child-region retrieval plus parent max aggregation can improve document
ranking for some corpora and embedders, but it is not universal.

## Versioned Evidence Robustness

External model feedback converged on the same critique: static BEIR retrieval
can make Refmark look like named chunking. To test the distinctive stability
claim, a controlled mutation benchmark was added on the Ubuntu research host.

Protocol:

1. split corpus documents into addressable regions at version `t`;
2. synthetically mutate the corpus with insertions, deletions, and light edits;
3. compare stable content-based ref migration against naive same-ordinal chunk
   identity;
4. mark deleted/unmatched regions as stale.

This is not yet a natural revision benchmark, but it directly tests whether
stored evidence addresses survive corpus drift better than disposable chunk ids.

| Dataset | Base Regions | Mutated Regions | Stable Exact | Stable Fuzzy | Stable Stale | Naive Correct | Naive Wrong Same-ID |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SciFact | 12,687 | 13,878 | `80.6%` | `8.8%` | `10.6%` | `58.8%` | `35.0%` |
| FiQA | 97,524 | 107,344 | `80.9%` | `9.1%` | `10.0%` | `62.0%` | `31.6%` |

Interpretation:

- stable refs can migrate unchanged or lightly edited evidence after insertions
  and deletions;
- stale regions are explicit rather than silently wrong;
- naive ordinal chunk ids often still resolve, but to the wrong content;
- this is a better test of Refmark's unique value than document-level hit@10
  alone.

The next stronger version should use natural revisions: documentation versions,
Wikipedia revision pairs, legal/regulatory updates, or Git-backed docs.

Natural Git-backed documentation revision checks were then run:

| Corpus | Revision Pair | Old Regions | New Regions | Stable Exact | Stable Fuzzy/Moved | Stable Stale | Naive Correct | Naive Wrong Same-ID |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FastAPI docs | `0.100.0 -> 0.110.0` | 1,282 | 1,380 | `41.4%` | `26.6%` | `32.0%` | `65.4%` | `27.3%` |
| Django docs | `4.2 -> 5.0` | 6,519 | 6,613 | `52.3%` | `23.9%` | `23.9%` | `73.1%` | `26.6%` |

The natural revision result is more nuanced than the synthetic mutation result:
some old evidence really changes or disappears, so the stale rate is higher.
The important safety property remains: naive path+ordinal chunk ids often still
resolve but point to different evidence, while stable ref migration can either
find the same/moved/edited evidence or mark it stale for review.

The more practical framing is curated eval/training maintenance: treat every old
`query -> evidence ref` mapping as an expensive maintained label. After a
revision, Refmark can say which labels are automatically preserved/migrated,
which need fuzzy review, and which are stale. Naive chunk labels may appear to
resolve while silently targeting wrong evidence.

Multi-revision curves from one base version:

| Corpus | Base -> Target | Refmark Auto | Refmark Review | Refmark Stale | Naive Correct | Naive Silent Wrong | Review Workload Reduction |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FastAPI | `0.100.0 -> 0.105.0` | `54.2%` | `39.1%` | `6.7%` | `71.5%` | `21.3%` | `50.6%` |
| FastAPI | `0.100.0 -> 0.110.0` | `42.5%` | `25.5%` | `32.0%` | `65.4%` | `27.3%` | `38.0%` |
| FastAPI | `0.100.0 -> 0.115.0` | `6.6%` | `34.0%` | `59.4%` | `29.8%` | `61.5%` | `-2.4%` |
| Django | `4.2 -> 5.0` | `52.3%` | `23.9%` | `23.9%` | `73.1%` | `26.6%` | `52.1%` |
| Django | `4.2 -> 5.1` | `42.3%` | `28.4%` | `29.3%` | `64.3%` | `35.3%` | `42.0%` |
| Django | `4.2 -> 5.2` | `36.3%` | `30.5%` | `33.2%` | `59.4%` | `40.2%` | `36.0%` |

This is the strongest evidence so far for the lifecycle claim. The win is not
that old evidence always survives. The win is that the system distinguishes
preserved, moved/edited, and stale evidence, while naive labels accumulate
silent corruption as revision distance grows.

## First Adaptation-Loop Proof

The next check tuned a simple hybrid score on dev queries only:

```text
score(doc) = a * document_embedding_score + b * best_child_region_embedding_score
```

The selected `a/b` weights were then evaluated on held-out queries. This is a
minimal test of the core research question: whether Refmark diagnostics and
addressable child evidence can guide a retrieval change that survives held-out
evaluation.

Held-out results with MiniLM:

| Dataset | Doc Embedding hit@10 | Region Max hit@10 | Dev-Tuned Hybrid hit@10 | Hybrid MRR | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| SciFact | `0.767` | `0.820` | `0.813` | `0.642` | hybrid improves over doc, region-only best top-10 |
| NFCorpus | `0.727` | `0.739` | `0.727` | `0.527` | hybrid improves doc MRR, region-only best top-10 |
| FiQA | `0.614` | `0.627` | `0.630` | `0.429` | hybrid best top-10 and MRR |
| ArguAna | `0.762` | `0.734` | `0.764` | `0.256` | hybrid recovers region regression and slightly beats doc |

This is encouraging but not yet a broad claim. The first adaptation loop shows
that region evidence is a useful additional signal, especially when tuned on
dev and checked on held-out data. It also shows that the tuning objective must
be explicit: optimizing top-1/MRR may trade off against hit@10.

The run artifacts now include corpus/query fingerprints, region-setting
fingerprints, dev query ids, held-out query ids, selected weights by objective
(`hit10`, `mrr`, `balanced`), and held-out baseline comparisons. This is the
minimum structure needed for reproducible retrieval-CI style experiments.

## Diagnostic Buckets

The evaluator now classifies held-out queries by which signal found the gold
document in top 10:

- `all_hit`: doc, region, and hybrid all found it.
- `region_wins_over_doc`: region evidence found it and document embedding did
  not.
- `doc_wins_over_region`: document embedding found it and region evidence did
  not.
- `shared_misses`: all variants missed.

Held-out MiniLM diagnostics:

| Dataset | All Hit | Region Wins | Doc Wins | Shared Misses | Mean Doc-Rank Minus Hybrid-Rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| SciFact | 114 | 9 | 1 | 26 | `0.149` |
| NFCorpus | 112 | 7 | 5 | 37 | `0.087` |
| FiQA | 184 | 19 | 15 | 106 | `0.171` |

These buckets make the next adaptation step concrete:

- `region_wins_over_doc` examples show where child evidence is rescuing long or
  noisy parent documents.
- `doc_wins_over_region` examples show where region boundaries are too narrow or
  local context is insufficient.
- `shared_misses` are candidates for metadata aliases, alternate embedders,
  query rewriting, or benchmark-label inspection.
- repeated wrong top-3 documents are query-magnet/confusion candidates.

FiQA is the best next adaptation target because it has the largest held-out
surface: many shared misses plus a meaningful split between doc wins and region
wins.

## Dev-Only Alias Adaptation

As a first metadata adaptation test, dev query text was appended as shadow
retrieval aliases to documents labeled relevant on the dev split. Held-out
queries were never added as aliases.

Held-out MiniLM result:

| Dataset | Base Region hit@10 | Alias Region hit@10 | Base Region MRR | Alias Region MRR | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| SciFact | `0.820` | `0.820` | `0.640` | `0.643` | tiny MRR gain |
| NFCorpus | `0.739` | `0.745` | `0.547` | `0.557` | small gain |
| FiQA | `0.627` | `0.627` | `0.421` | `0.420` | no gain |

This is intentionally conservative. Blindly adding supervised aliases is not a
general solution for embedding retrieval. The likely next step is targeted
metadata adaptation: use diagnostic buckets to add aliases only for shared
misses, repeated confusions, or regions where child evidence consistently beats
parent retrieval.

The next research step is a true adaptation loop:

1. use train/dev qrels only;
2. inspect missed refs, query magnets, and confusion pairs;
3. apply bounded policies such as metadata aliases, hierarchy, exclusion, or
   region-size changes;
4. evaluate once on held-out qrels;
5. report both wins and regressions.
