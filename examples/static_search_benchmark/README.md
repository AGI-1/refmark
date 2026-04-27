# Static Search Benchmark

This benchmark compares Refmark browser BM25 against common static/browser
search engines using the same Refmark-labeled questions.

The goal is not to prove that one library is universally better. The goal is to
measure information localization under the same evidence ruler:

- did the engine find the right article?
- did it find the exact anchor?
- how large is the deployable index?
- how fast are build, startup, and query?
- can it run offline in a browser without a server or API key?

## Engines

Current adapters:

| Variant | Dependency | License | Deployment |
| --- | --- | --- | --- |
| `raw_refmark_bm25` | none | MIT | static/browser |
| `refmark_bm25` | none | MIT | static/browser |
| `raw_minisearch` / `refmark_minisearch` | `minisearch` | MIT | static/browser |
| `raw_lunr` / `refmark_lunr` | `lunr` | MIT | static/browser |
| `raw_flexsearch` / `refmark_flexsearch` | `flexsearch` | Apache-2.0 | static/browser |

The `raw_*` variants index source region text only. The `refmark_*` variants
index source text plus generated Refmark retrieval views: summary, likely
questions, and keywords. This keeps the comparison focused on whether Refmark
metadata improves localization across engines, not only inside Refmark's own
BM25 runtime.

Pagefind, Meilisearch, Typesense, Elasticsearch, and embedding search are good
next adapters, but they need separate handling because they introduce a build
CLI, server process, or remote/model dependency.

## Setup

Install only the benchmark dependencies:

```bash
cd examples/static_search_benchmark
npm.cmd install
```

The Refmark-only path does not need npm dependencies.

## Inputs

Build browser BM25 exports from the already generated portable indexes:

```bash
python -m refmark.cli export-browser-index examples/portable_search_index/output/full_2m_raw_index.json -o examples/portable_search_index/output/full_2m_raw_browser_index.json --max-text-chars 600
python -m refmark.cli export-browser-index examples/portable_search_index/output/full_2m_openrouter_index.json -o examples/portable_search_index/output/full_2m_browser_index.json --max-text-chars 600
```

## Run

From the repository root:

```bash
node examples/static_search_benchmark/compare_static_search_engines.js --output examples/static_search_benchmark/output/static_search_benchmark.json
```

To run only the dependency-free Refmark BM25 baseline:

```bash
node examples/static_search_benchmark/compare_static_search_engines.js --engines refmark-bm25 --output examples/static_search_benchmark/output/refmark_only_benchmark.json
```

## Metrics

Quality:

- `anchor_hit_at_k`: exact gold ref was returned in top-k
- `article_hit_at_k`: any hit in the gold document was returned in top-k
- `anchor_mrr` and `article_mrr`
- sample misses for qualitative inspection

Deployment/runtime:

- `build_ms`
- `raw_size_bytes`
- `gzip_size_bytes`
- query latency `avg`, `p50`, `p95`, `max`
- `server_required`
- `api_key_required`
- `offline`
- `license`

These deployment metrics are part of the fairness story. A hosted search engine
may beat a static browser index on some quality metric, but that should be
weighed against server operations, index serving, privacy, offline behavior, and
whether it can be embedded into a documentation page or application package.

## Methodology Notes

The comparison is intentionally built around stable Refmark labels rather than
final answer quality. Every engine receives the same region set and the same
question cache. A hit is counted as an exact anchor hit only when the returned
result's `stable_ref` matches the generated gold ref. A looser article hit
counts any returned region in the gold document.

For each engine there are two variants:

- `raw_*`: source region text only
- `refmark_*`: source region text plus Refmark retrieval views

This means the benchmark asks two separate questions:

1. Which static search engine localizes evidence best under the same labels?
2. Does adding Refmark retrieval metadata improve localization across engines?

Adapter caveats:

- Refmark BM25 uses a pre-exported browser postings payload, so `build_ms` is
  `0` for the query-time artifact. Index generation is measured separately by
  the CLI/evaluation pipeline.
- MiniSearch uses its default query ranking. Prefix/fuzzy expansion was tested
  and rejected here because it produced noisy localization on this corpus.
- Lunr queries are punctuation-sanitized before normal parsing so natural user
  questions are not interpreted as Lunr query syntax.
- FlexSearch uses a single concatenated index because its multi-field document
  results do not provide a directly comparable global BM25-style score in this
  simple adapter. Its size is currently reported as `null` until export sizing
  is implemented.
- These are reasonable first adapters, not vendor-tuned best possible
  configurations. Results should be treated as reproducible evidence for this
  setup, not universal engine rankings.

## Current Refmark-Only Smoke

On the 2.1M-token public-doc corpus with 844 cached questions, the
dependency-free JS Refmark browser runtime reproduces the Python BM25 numbers:

| Variant | Anchor@1 | Anchor@10 | Article@1 | Article@10 | p50 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw Refmark BM25 | 0.583 | 0.867 | 0.725 | 0.936 | 1.045 ms |
| Refmark BM25 | 0.720 | 0.949 | 0.831 | 0.973 | 1.114 ms |

The browser BM25 exports were about `1.9-2.0 MiB` gzip with 600-character
snippets over 3,678 regions.

## Current Static-Engine Run

Full run over the 2.1M-token public-doc corpus, 3,678 regions, and 844 cached
OpenRouter-generated questions:

| Variant | Anchor@1 | Anchor@10 | Article@1 | Anchor MRR | p50 latency | gzip size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw Refmark BM25 | 0.583 | 0.867 | 0.725 | 0.681 | 1.043 ms | 2.00 MB |
| Refmark BM25 | 0.720 | 0.949 | 0.831 | 0.810 | 1.171 ms | 2.02 MB |
| raw MiniSearch | 0.312 | 0.646 | 0.448 | 0.419 | 4.194 ms | 1.31 MB |
| Refmark MiniSearch | 0.675 | 0.931 | 0.801 | 0.769 | 8.266 ms | 1.73 MB |
| raw Lunr | 0.518 | 0.846 | 0.680 | 0.635 | 3.012 ms | 3.63 MB |
| Refmark Lunr | 0.619 | 0.919 | 0.770 | 0.723 | 8.040 ms | 4.59 MB |
| raw FlexSearch | 0.129 | 0.213 | 0.180 | 0.158 | 0.244 ms | n/a |
| Refmark FlexSearch | 0.328 | 0.615 | 0.422 | 0.424 | 0.306 ms | n/a |

The important pattern is not only that Refmark BM25 is strong. It is that
Refmark retrieval views improve every tested engine's localization:

| Engine | Anchor@1 raw | Anchor@1 + Refmark views | Delta |
| --- | ---: | ---: | ---: |
| Refmark BM25 | 0.583 | 0.720 | +0.138 |
| MiniSearch | 0.312 | 0.675 | +0.364 |
| Lunr | 0.518 | 0.619 | +0.101 |
| FlexSearch | 0.129 | 0.328 | +0.199 |

That supports a product claim narrower and stronger than "new search engine":
Refmark provides a reusable localization/evaluation layer that can improve and
measure multiple search backends.
