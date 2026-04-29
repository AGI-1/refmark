# Portable Search Index

This example is the direct product shape for Refmark retrieval:

```text
your corpus -> refmark regions -> cheap LLM retrieval views -> local BM25 index
```

The generated index is a single JSON file. Searching it needs no API key,
embedding model, GPU, vector database, or server process. That makes it a good
fit for package documentation, internal tools, support playbooks, and other
corpora where a small index can ship with the software itself.

Build a local zero-cost index over the sample docs:

```bash
python -m refmark.cli build-index examples/portable_search_index/sample_corpus -o examples/portable_search_index/output/index_local.json
python -m refmark.cli search-index examples/portable_search_index/output/index_local.json "How do I rotate tokens?"
python -m refmark.cli export-browser-index examples/portable_search_index/output/index_local.json -o examples/portable_search_index/output/index_browser.json
```

Run the easy-mode full pipeline over the fetched FastAPI documentation corpus
and query the processed output:

```bash
python -m refmark.cli run-pipeline examples/portable_search_index/fastapi_pipeline.local.yaml
python -m refmark.cli query-pipeline examples/portable_search_index/output/fastapi_pipeline "How do I configure CORS for browser clients?"
```

That corpus is about 309k approximate source tokens in the retained benchmark
snapshot. The local config does not need an API key or vector database.

Build an LLM-enriched index through OpenRouter:

```bash
python -m refmark.cli build-index examples/portable_search_index/sample_corpus -o examples/portable_search_index/output/index_openrouter.json --source openrouter --model mistralai/mistral-nemo --concurrency 4
python -m refmark.cli search-index examples/portable_search_index/output/index_openrouter.json "Which setting keeps audit logs longer?"
```

For repeat builds, keep generated retrieval views in a JSONL cache. The cache
key includes stable ref, region hash, provider, and model, so unchanged regions
are reused and changed regions are regenerated:

```bash
python -m refmark.cli build-index docs -o docs.refmark-index.json --source openrouter --model mistralai/mistral-nemo --view-cache .refmark/view-cache.jsonl
```

The index stores, per region:

- `doc_id` and `region_id`, giving a stable evidence id like
  `security_guide:P03`
- original region text and region hash
- neighbor ids for deterministic expansion
- retrieval metadata: summary, likely user questions, and keywords

The search path is deliberately boring: BM25 over source text plus generated
retrieval metadata. The useful claim is not that embeddings disappear forever;
it is that a one-time cheap LLM pass can make lexical search much more semantic
while keeping runtime search tiny and embeddable.

Changelog/release-note-like regions are treated as query magnets. They remain
in the index and visual reports, but generated indexes mark them as
`exclude_from_default_search` so normal search can skip them unless the caller
passes `--include-excluded`.

For browser usage, `export-browser-index` writes a compact BM25 payload with
postings, region metadata, and short snippets. Pair it with
`refmark/browser_search.js` to run search fully on the client. On a single
documentation page, elements can expose `data-refmark-ref` and the runtime can
jump/highlight them directly, turning Refmark into semantic find for the
current page.

The FastAPI navigation experiment also includes an evidence heatmap generated
from the same eval artifacts. It is intentionally a workbench, not just a
chart: structural clusters follow the documentation path hierarchy, weak
regions are visible by retrieval mode, a search field highlights matching
sections/refs/questions, and the side panel pins the selected block's refs,
metrics, and generated eval questions. The matching adaptation script
(`improve_fastapi_questions.py`) can review weak blocks, write shadow
Doc2Query metadata, rerun affected-row mini-evals, and refresh the report.

That loop is the product-shaped part of this example:

```text
build index -> generate/evaluate questions -> inspect heatmap
            -> adapt questions or shadow metadata -> compare modes again
```

On the current 2.1M-token public-docs corpus, the OpenRouter-enriched portable
index is `12.6 MiB` raw / `2.9 MiB` gzip, while the browser BM25 export with
600-character snippets is `8.2 MiB` raw / `1.9 MiB` gzip. That is already in
static-site territory before any binary packing or payload splitting.

With `mistralai/mistral-nemo` pricing around `$0.01/M` input tokens and
`$0.03/M` output tokens on OpenRouter, this style should stay far under `$1`
per million original corpus tokens unless prompts are made very wasteful. The
CLI prints an approximate cost estimate into the build log.

For an application integration, call the library surface directly:

```python
from refmark.search_index import build_search_index, load_search_index

build_search_index("docs", "docs.refmark-index.json", source="openrouter")
index = load_search_index("docs.refmark-index.json")
hits = index.search("How do I configure retention?", top_k=3, expand_after=1)
```

## Real-Corpus Evaluation

The important evaluation question is scale degradation: does a cheap enriched
region index still localize evidence as the corpus grows, and when do we need a
second stage such as hierarchy, neighbor expansion, reranking, or embeddings?

Run a local zero-cost sweep:

```bash
python examples/portable_search_index/evaluate_real_corpus.py refmark_train/source_docs/text --budgets 50000,100000,250000,500000,1000000 --sample-size 200 --question-source local --index-view-source local
```

Fetch public documentation corpora for reproducibility checks:

```bash
python examples/portable_search_index/fetch_online_corpora.py --output examples/portable_search_index/output/online_corpora
```

Run a smaller held-out natural-question probe with OpenRouter:

```bash
python examples/portable_search_index/evaluate_real_corpus.py refmark_train/source_docs/text --budgets 50000,250000,1000000 --sample-size 40 --question-source openrouter --index-view-source local --model mistralai/mistral-nemo --concurrency 8
```

Compare retrieval modes:

```bash
python examples/portable_search_index/evaluate_real_corpus.py refmark_train/source_docs/text --budgets 1000000 --sample-size 40 --question-source openrouter --index-view-source local --strategies flat,hierarchical,rerank --expand-after 1
```

Run an OpenRouter-enriched public-docs pass with cached views:

```bash
python examples/portable_search_index/evaluate_real_corpus.py examples/portable_search_index/output/online_corpora/kubernetes_docs --budgets 1000000 --sample-size 200 --question-source openrouter --index-view-source openrouter --model mistralai/mistral-nemo --concurrency 12 --strategies flat,rerank --candidate-k 50 --expand-after 1 --gold-mode single --cache examples/portable_search_index/output/eval_question_cache_online.jsonl --view-cache examples/portable_search_index/output/view_cache_online_mistral_nemo.jsonl
```

Run a mixed 1M-token public-docs pass from the fetched corpus root. The
`--exclude-glob` option keeps generated aggregate files out of the sample:

```bash
python examples/portable_search_index/evaluate_real_corpus.py examples/portable_search_index/output/online_corpora --exclude-glob "*_combined.txt" --budgets 1000000 --sample-size 250 --question-source openrouter --index-view-source openrouter --model mistralai/mistral-nemo --concurrency 12 --strategies flat,rerank --candidate-k 50 --expand-after 1 --gold-mode single --cache examples/portable_search_index/output/eval_question_cache_online_mixed.jsonl --view-cache examples/portable_search_index/output/view_cache_online_mixed_mistral_nemo.jsonl
```

The evaluator reports:

- exact `hit@k`: the returned region is the generated gold region
- `context_hit@k`: the gold region is inside deterministic neighbor expansion
- `doc_hit@k`: the right document was found, even if paragraph localization missed
- MRR and sample misses for failure analysis

For a second evaluation lens, run random cached questions through retrieval and
ask an LLM judge whether the retrieved top-k evidence would support an answer:

```bash
python -m refmark.cli build-index examples/portable_search_index/output/online_corpora/kubernetes_docs -o examples/portable_search_index/output/kubernetes_openrouter_index.json --source openrouter --model mistralai/mistral-nemo --concurrency 12 --view-cache examples/portable_search_index/output/view_cache_online_mistral_nemo.jsonl
python examples/portable_search_index/judge_random_queries.py examples/portable_search_index/output/kubernetes_openrouter_index.json --question-cache examples/portable_search_index/output/eval_question_cache_online.jsonl --limit 30 --top-k 3 --expand-after 1 --model mistralai/mistral-nemo
```

First real-doc observations:

| Corpus slice | Questions | Method | hit@1 | hit@10 | context hit@1 | doc hit@1 | MRR |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| extracted docs, ~648k tokens | 40 OpenRouter-heldout | raw region BM25 | 0.475 | 0.900 | 0.500 | 0.950 | 0.607 |
| extracted docs, ~648k tokens | 40 OpenRouter-heldout | local-enriched BM25 | 0.575 | 0.900 | 0.650 | 0.950 | 0.666 |
| set corpus, ~999k tokens | 200 local-heldout | raw region BM25 | 0.255 | 0.880 | 0.255 | 0.575 | 0.454 |
| set corpus, ~999k tokens | 200 local-heldout | local-enriched BM25 | 0.305 | 0.900 | 0.305 | 0.575 | 0.524 |

Mode comparison on the ~648k extracted-doc slice with OpenRouter-heldout
questions and one-region neighbor expansion:

| Method | hit@1 | hit@10 | context hit@1 | undercite@1 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw BM25, flat | 0.475 | 0.900 | 0.500 | 0.500 | 0.607 |
| raw BM25, rerank | 0.475 | 0.900 | 0.500 | 0.500 | 0.614 |
| local-enriched BM25, flat | 0.575 | 0.900 | 0.650 | 0.350 | 0.666 |
| local-enriched BM25, hierarchical | 0.575 | 0.900 | 0.650 | 0.350 | 0.669 |
| local-enriched BM25, rerank | 0.575 | 0.925 | 0.625 | 0.375 | 0.664 |

First LLM-enriched index-view result on a smaller 50k-token slice:

| Method | hit@1 | hit@3 | hit@10 | context hit@1 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw BM25 | 0.500 | 0.739 | 0.891 | 0.587 | 0.638 |
| OpenRouter-enriched BM25 | 0.870 | 0.957 | 0.978 | 0.902 | 0.915 |

That is currently the cleanest evidence for the product claim: cheap one-time
LLM metadata can substantially improve local lexical search. It is still only
a 50k-token slice, so the next run should repeat this at 250k and 1M with the
same cached view-generation path.

Public documentation sweep, OpenRouter-generated questions and OpenRouter
retrieval views, 200 held-out questions per corpus:

| Corpus | Approx tokens | Regions | raw hit@1 | enriched hit@1 | delta | raw MRR | enriched MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FastAPI docs | 309k | 689 | 0.440 | 0.650 | +0.210 | 0.540 | 0.743 |
| Django docs | 876k | 262 | 0.625 | 0.725 | +0.100 | 0.718 | 0.803 |
| Kubernetes docs | 666k | 1480 | 0.540 | 0.690 | +0.150 | 0.641 | 0.783 |
| Rust book | 196k | 824 | 0.580 | 0.710 | +0.130 | 0.687 | 0.807 |
| TypeScript handbook | 62k | 423 | 0.515 | 0.720 | +0.205 | 0.614 | 0.810 |
| Mixed public docs | 1,000k | 554 | 0.548 | 0.756 | +0.208 | 0.663 | 0.830 |

That is a more reproducible current claim than the small single-corpus probe:
for public technical documentation, cached LLM region views consistently moved
rank-1 evidence localization up by 10-21 points while keeping runtime search
local and cheap. The largest absolute gains appeared in API/reference-style
docs where users ask task-shaped questions that differ from the section text.
Django improved less, partly because this extraction produced only 262 large
regions over 876k tokens, which makes the task closer to document lookup than
fine evidence localization.

Scaling sweep over the combined public-docs set:

| Approx tokens | Docs | Regions | raw hit@1 | enriched hit@1 | raw hit@10 | enriched hit@10 | raw MRR | enriched MRR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 249k | 94 | 94 | 0.702 | 0.766 | 0.926 | 0.936 | 0.770 | 0.830 |
| 499k | 164 | 166 | 0.681 | 0.753 | 0.910 | 0.940 | 0.758 | 0.826 |
| 1,000k | 358 | 554 | 0.523 | 0.743 | 0.877 | 0.950 | 0.645 | 0.832 |
| 1,500k | 528 | 1380 | 0.517 | 0.733 | 0.793 | 0.933 | 0.611 | 0.806 |
| 2,109k | 767 | 3678 | 0.553 | 0.723 | 0.853 | 0.963 | 0.649 | 0.817 |

This is the first degradation picture. Enriched retrieval stays strong through
2.1M tokens, but rank-1 localization bends as duplicate-like regions and
release-note fragments enter the corpus. Candidate recall remains high, which
is exactly where a trained resolver can help.

LLM-as-judge spot checks on random held-out queries provide a useful reality
check beyond exact gold-ref matching:

| Corpus | Queries | top-k | answer-supported | avg gold coverage | irrelevant-extra rate | invalid judge JSON |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FastAPI docs | 30 | 3 | 0.900 | 0.800 | 0.367 | 0.000 |
| Kubernetes docs | 30 | 3 | 0.900 | 0.826 | 0.433 | 0.000 |
| Mixed public docs | 40 | 3 | 0.775 | 0.674 | 0.450 | 0.125 |

The judge results are encouraging but also point at the next quality lever:
top-3 evidence often supports the answer even when it is not an exact region
match, but the returned context is still too broad. Refmark makes that visible
as a concrete density/overcitation metric instead of hiding it inside a final
answer score. The mixed run also shows why the judge is a supporting signal
rather than the primary metric: five of forty cheap-model judge responses were
not valid JSON, and some generated queries are noisy or underspecified.

## Trained Resolver

The portable index can also train a small candidate resolver. This is not a
model that memorizes the whole corpus. The runtime shape is:

```text
query -> enriched BM25 top-50 candidates -> tiny pair scorer -> best refmark region
```

Train on cached held-out questions plus generated retrieval-view questions:

```bash
python examples/portable_search_index/train_refmark_resolver.py examples/portable_search_index/output/mixed_1m_openrouter_index.json --question-cache examples/portable_search_index/output/eval_question_cache_online_mixed.jsonl --candidate-k 50 --epochs 14 --layers 3 --embed-dim 128 --hidden-dim 256 --loss hybrid --train-from-index-views --view-questions-per-region 4
```

Current first positive training results:

| Corpus | Train questions | Eval questions | Params | Artifact | BM25 hit@1 | Best hit@1 | BM25 MRR | Best MRR | Best blend | Candidate scoring |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mixed public docs, 1M-token slice | 2614 | 156 | 1.31M | 5.43 MB | 0.8205 | 0.8526 | 0.8777 | 0.9015 | 0.50 | 1.28 ms/query |
| Kubernetes docs | 6064 | 56 | 1.31M | 5.43 MB | 0.6964 | 0.8036 | 0.8036 | 0.8564 | 0.65 | 1.60 ms/query |
| Full public docs, 2.1M-token set | 15319 | 237 | 1.31M | 5.43 MB | 0.6582 | 0.6920 | 0.7658 | 0.7819 | 0.50 | 1.61 ms/query |

A smaller 0.57M-parameter resolver also worked, but the gain was thinner on
the mixed corpus: hit@1 moved from 0.8205 to 0.8333 and MRR from 0.8777 to
0.8865. The larger resolver is still small enough to ship as a local package
artifact.

The best result comes from blending normalized BM25 score with the neural
resolver score. That preserves more of BM25's lower-rank stability while using
the resolver for the final evidence choice. The important caveat remains: this
improves rank-1 localization, but can slightly disturb lower ranks on some
corpora. Use it as an evidence-resolution layer over a high-recall candidate
set, while keeping the original candidate list available for fallback, range
expansion, and anomaly detection.

First vector-signal probe: adding sparse TF-IDF cosine features between query
and refmark-range text/metadata did not beat the non-vector blended resolver on
the mixed 1M slice. That is a useful negative result. Lexical vectors mostly
repeat information BM25 already provides; semantic embedding vectors may still
be useful for candidate generation, hard-negative mining, or resolver features
if they add signal that BM25 and generated index views do not already capture.

## Navigation Search Baseline

For application documentation, article-level navigation is often the primary
success metric: getting the user to the right page is enough, and the exact
anchor is a bonus. Refmark labels let us evaluate both.

On the full 2.1M-token public-docs corpus with 844 cached questions:

| Method | Anchor hit@1 | Anchor hit@10 | Article hit@1 | Article hit@10 | Anchor MRR | Article MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw BM25 | 0.583 | 0.867 | 0.725 | 0.936 | 0.681 | 0.800 |
| Refmark BM25 | 0.720 | 0.949 | 0.831 | 0.973 | 0.810 | 0.888 |
| raw hashed embedding | 0.286 | 0.589 | 0.385 | 0.683 | 0.377 | 0.477 |
| Refmark hashed embedding | 0.350 | 0.627 | 0.476 | 0.715 | 0.440 | 0.554 |
| raw BM25 + hashed embedding | 0.474 | 0.821 | 0.606 | 0.903 | 0.583 | 0.699 |
| Refmark BM25 + hashed embedding | 0.551 | 0.889 | 0.673 | 0.930 | 0.652 | 0.750 |

The comparison is deliberately modest: the embedding rows use a dependency-free
hashed lexical vector baseline, not a strong semantic embedding model. It is
still useful as a floor. In this setup, Refmark-generated retrieval views beat
raw BM25 clearly, while cheap lexical vectors and naive hybridization hurt.
The same script can also call an OpenAI-compatible embedding endpoint through
OpenRouter and evaluate it with the exact same Refmark article/anchor labels:

```bash
python examples/portable_search_index/compare_navigation_search.py examples/portable_search_index/output/full_2m_raw_index.json examples/portable_search_index/output/full_2m_openrouter_index.json --question-cache examples/portable_search_index/output/eval_question_cache_online_scaling.jsonl --real-embeddings --embedding-model openai/text-embedding-3-small --embedding-cache examples/portable_search_index/output/openrouter_embedding_cache_text_embedding_3_small.jsonl
```

First real-embedding comparison on the same 2.1M-token corpus:

| Method | Anchor hit@1 | Anchor hit@10 | Article hit@1 | Article hit@10 | Anchor MRR | Article MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw BM25 | 0.583 | 0.867 | 0.725 | 0.936 | 0.681 | 0.800 |
| Refmark BM25 | 0.720 | 0.949 | 0.831 | 0.973 | 0.810 | 0.888 |
| OpenRouter embedding | 0.620 | 0.928 | 0.768 | 0.967 | 0.730 | 0.837 |
| Refmark BM25 + OpenRouter embedding | 0.725 | 0.959 | 0.839 | 0.981 | 0.815 | 0.894 |

This is a better comparison shape. Semantic embeddings alone beat raw BM25, but
they do not beat cached Refmark retrieval views on this documentation corpus.
The hybrid gives the best article navigation and a small anchor-localization
gain, which supports the practical design: Refmark is the evaluation ruler and
the cheap lexical navigation layer; embeddings are optional extra signal when
latency, dependency, and cache size are acceptable.

Coarse target training is supported with `--coarse-mode article`, where any
candidate anchor in the gold document counts as positive. On the full 2.1M
public-doc set, an article-mode resolver moved article hit@1 from 0.802 to
0.814 on its held-out split, but MRR and top-k were already very strong with
Refmark BM25. That suggests the cheaper default for application docs should be:
Refmark BM25 for article navigation, then optional resolver/range expansion for
section selection.

## Coarse Range Localization

For section-level navigation, generate range-level questions instead of
single-anchor questions:

```bash
python examples/portable_search_index/evaluate_coarse_ranges.py examples/portable_search_index/output/full_2m_raw_index.json examples/portable_search_index/output/full_2m_openrouter_index.json --range-type window --window-size 6 --sample-size 120 --question-source openrouter --expand-after 5
```

First coarse-window run on the full 2.1M public-doc corpus:

| Method | Any anchor in range@1 | Any anchor in range@5 | Article hit@1 | Full range context@1 | Full range context@5 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw BM25 | 0.625 | 0.842 | 0.700 | 0.508 | 0.717 | 0.733 |
| Refmark BM25 | 0.758 | 0.975 | 0.783 | 0.567 | 0.850 | 0.846 |

This is the most product-aligned signal so far: for a broad section/range
query, Refmark only needs to land on any anchor inside the target range to
support navigation, and deterministic expansion can recover much of the range
for highlighting or answer generation.

Range probes show the harder boundary:

| Gold shape | Method | hit@1 | hit@10 | range cover@1 | range cover@10 | wrong-location@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| adjacent 2-region | local-enriched BM25 | 0.730 | 0.980 | 0.550 | 0.865 | 0.020 |
| disjoint 2-region | local-enriched BM25 | 0.660 | 0.950 | 0.335 | 0.770 | 0.040 |

So disjoint questions do not fail as pure candidate retrieval, but they do fail
as complete evidence retrieval: the system often finds one relevant region or
the right document, while underciting the second support span. This is useful
because regular single-vector retrieval has the same structural problem, and
Refmark makes the failure measurable.

These numbers are not final product claims. They are useful because they show
where quality starts to bend. At larger scale, candidate recall remains
workable (`hit@10` stays high), while rank-1 localization and duplicate-like
regions become the pressure points. Neighbor expansion already recovers some
near misses. The first hierarchy/reranker pass is intentionally cheap and only
moves the curve a little; the next stronger version should use cached
LLM-generated index views at larger sample size and a learned reranker over the
already-good top-10/top-20 candidates.

One negative result is also worth keeping: the first naive learned reranker
overfit badly and collapsed out of split. Keep the hand-built reranker as a
diagnostic, but do not claim learned reranking until it uses a better objective
and a cleaner train/eval split.
