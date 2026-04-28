# BGB Browser Search Demo

This example builds a fully offline browser search demo for the German
`Bürgerliches Gesetzbuch` (BGB).

It demonstrates the "semantic Ctrl+F" use case:

```text
official legal text -> stable Refmark regions -> browser BM25 index -> jump/highlight
```

The source is the official Gesetze-im-Internet BGB HTML page:

https://www.gesetze-im-internet.de/bgb/BJNR001950896.html

This is a navigation/search demo, not legal advice.

## Build

```bash
python examples/bgb_browser_search/build_bgb_demo.py
```

Generated files are written to `examples/bgb_browser_search/output/`, which is
ignored by git:

- `bgb_official.html`
- `bgb_refmark_index.json`
- `bgb_browser_index.json`
- `bgb_demo_data.js`
- `index.html`
- `manifest.json`

Open `examples/bgb_browser_search/output/index.html` directly in a browser.
The index is loaded through `bgb_demo_data.js`, so the demo also works from
`file://` without a local server.

Try queries such as:

- `Kündigungsfrist Wohnung`
- `Geschäftsfähigkeit Minderjährige`
- `Erbschaft ausschlagen`
- `Verbraucher Widerruf Fernabsatz`
- `Mietminderung Mangel`
- `digitale Produkte Mangel`

Run the small break suite:

```bash
node examples/bgb_browser_search/run_bgb_break_suite.js
```

It contains expected demo queries, ambiguous one-word probes, and out-of-domain
no-answer probes. The no-answer rows are not scored as failures yet; they expose
where a browser search UI should avoid pretending that the top hit is a legal
answer.

## Article-Level Concern Navigation

For the "I have a real-world concern, jump me to the relevant article" version,
build an article-level index from the generated BGB search index:

```bash
python examples/bgb_browser_search/build_bgb_article_navigation.py
```

This aggregates Absatz-level regions such as `bgb:S_437_A01` into article refs
such as `bgb:S_437`, then optionally injects curated concern aliases from
`concern_aliases.json`. The aliases are not source text and should be treated as
retrieval metadata: they make motivation-level wording measurable without
changing the BGB text.

The full BGB pipeline also consumes the same alias file:

```bash
python examples/bgb_browser_search/run_bgb_pipeline.py \
  --view-source local \
  --question-source local \
  --languages de,en
```

Concern rows are added as `gold_mode=concern` eval/training questions, and the
aliases are injected only into the enriched Refmark retrieval view. Raw BM25
stays a clean baseline. Use `--skip-concern-questions` to disable this layer.

To compare the same concern queries without alias injection:

```bash
python examples/bgb_browser_search/build_bgb_article_navigation.py \
  --without-aliases \
  --output-dir examples/bgb_browser_search/output_article_nav_raw
```

Current extended article-level smoke result with
`concern_aliases_extended.json` has 104 expected concern rows plus five
ambiguous/no-answer probes. Held-out query text is not injected into the index;
only the separate `aliases` field is retrieval metadata.

| Index | expected hit@1 | expected hit@5 |
| --- | ---: | ---: |
| raw article index | 0.7885 | 0.9327 |
| article index + concern aliases | 0.8173 | 0.9327 |

The practical break probe is:

```text
Ich habe ein Handy gekauft und es ist kaputt. Was kann ich tun?
```

Without concern aliases this drifts into unrelated finder/property provisions.
With aliases it lands at `bgb:S_437` / `bgb:S_439` / `bgb:S_474`, which is the
expected buyer-defect neighborhood. This is not a legal-answer claim; it is a
retrieval-localization claim that Refmark can score directly.

The full cached Qwen Turbo pipeline also writes a combined eval suite at
`output_full_qwen_turbo/bgb_eval_questions.jsonl`. In the latest full rerun it
contained 899 rows, including 99 held-out concern rows from
`concern_aliases_extended.json`.

| Method | overall hit@1 | overall hit@10 | concern hit@1 | concern hit@10 |
| --- | ---: | ---: | ---: | ---: |
| raw BM25 | 0.3471 | 0.5384 | 0.1717 | 0.2828 |
| Refmark BM25 + generated views + concern aliases | 0.7642 | 0.9366 | 0.8182 | 0.9495 |
| Refmark rerank | 0.7608 | 0.9333 | 0.8081 | 0.9596 |

The remaining concern misses are now visible in the report under
`misses_by_mode.concern`, which makes the next adapt loop concrete instead of
guessy.

## Randomized Stress Questions

For a less friendly benchmark, run randomized question generation over sampled
article blocks:

```bash
python examples/bgb_browser_search/run_bgb_stress_eval.py \
  --sample-size 32 \
  --models qwen/qwen-turbo,mistralai/mistral-nemo,mistralai/mistral-small-3.2-24b-instruct,google/gemma-3-27b-it,x-ai/grok-4-fast \
  --languages de,en \
  --styles direct,concern,adversarial \
  --questions-per-style 1 \
  --preflight
```

This asks each generator model to create fresh questions from an article block;
the answer key remains the sampled block refs. The report contains a `sections`
array so every sampled article keeps the generated question list that attacked
it.

Current mixed stress run:

| Method | rows | article hit@1 | article hit@10 |
| --- | ---: | ---: | ---: |
| raw BM25 | 953 | 0.1427 | 0.2445 |
| Refmark BM25 | 953 | 0.3578 | 0.5824 |
| Refmark rerank | 953 | 0.3526 | 0.5887 |

Current concern-heavy stress run:

| Method | rows | article hit@1 | article hit@10 |
| --- | ---: | ---: | ---: |
| raw BM25 | 873 | 0.1924 | 0.3116 |
| Refmark BM25 | 873 | 0.4719 | 0.7320 |
| Refmark rerank | 873 | 0.4708 | 0.7308 |

These numbers are intentionally lower than the curated concern suite. They are
better evidence for the adapt loop because they reveal weak blocks such as
`bgb:S_14`, `bgb:S_2316`, `bgb:S_308`, `bgb:S_2268`, `bgb:S_536b`, and
`bgb:S_965`.

The 200k-token stress cycles use about half of the BGB source corpus per run and
compare static Refmark BM25 against Qdrant/Qwen3 embeddings as build-time
teacher infrastructure:

| Cycle | Generator | Questions | Refmark BM25 hit@10 | Qwen3 hit@10 | Best hybrid hit@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Qwen Turbo | 4,008 | 0.6914 | 0.9474 | 0.9506 |
| 2 | Mistral Small 3.2 24B | 3,916 | 0.6586 | 0.9614 | 0.9620 |
| 3 | Gemma 3 27B | 3,874 | 0.3446 | 0.9174 | 0.9177 |

This is the strongest current evidence that embeddings are valuable as an
offline teacher. The runtime target remains a static Refmark/BM25 index plus a
small reranker or distilled metadata, not Qdrant in production.

First static adaptation attempt: splitting Gemma-cycle questions within each
article and injecting half as raw aliases made held-out BM25 worse. The article
baseline was `hit@10 = 0.3750`; aliases mixed into the article view dropped to
`0.3193`, and an alias-only side index reached only `0.0768`. So the next
adaptation step should compress/weight teacher signals or train a reranker with
hard negatives rather than paste full generated questions into metadata.

A 3-cycle heatmap over all 11,798 stress questions now reports static Refmark
BM25 at `0.5920` hit@10, `0.7434` hit@50, and `0.9517` candidate recall at
top 1000. That shape is useful: most gold articles are somewhere in the deep
candidate set, but not high enough for a polished instant-navigation UI.

The first compressed adaptation works better than raw aliases. Local
held-out-safe intent signatures improved the combined 3-cycle held-out split
from `0.5912` to `0.6251` hit@10 and from `0.7435` to `0.7849` hit@50. The
signal helped German, English, concern, and adversarial rows. Signature-only
retrieval was poor, so the useful pattern is compact intent metadata mixed into
the source/generated-view index, not a separate replacement index.

Cached LLM-generated signatures were then tried on the 40 hardest heatmap
articles. On those selected rows, baseline BM25 had `0.0000` hit@10 by
construction, deterministic signatures reached `0.1616`, Qwen Turbo signatures
reached `0.2828`, and Gemma 3 27B signatures reached `0.3030`. Across all
held-out rows, deterministic + Gemma hard-40 signatures slightly edged the
deterministic baseline (`0.6253` vs. `0.6251` hit@10, `0.7866` vs. `0.7849`
hit@50), while Qwen was slightly noisier. The practical lesson is that LLM
signatures look useful for heatmap repair, but they need gating or careful
weighting before becoming a corpus-wide default.

The LLM-signature evaluator is cached per article hash/model/prompt:

```bash
python examples/bgb_browser_search/evaluate_bgb_llm_intent_signatures.py \
  --stress-report examples/bgb_browser_search/output_full_qwen_turbo/bgb_stress_eval_200k_cycle1_qwen_v2.json \
  --stress-report examples/bgb_browser_search/output_full_qwen_turbo/bgb_stress_eval_200k_cycle2_mistral_small.json \
  --stress-report examples/bgb_browser_search/output_full_qwen_turbo/bgb_stress_eval_200k_cycle3_gemma.json \
  --report examples/bgb_browser_search/output_full_qwen_turbo/bgb_llm_intent_signatures_hard40.json \
  --max-hard-articles 40 \
  --model google/gemma-3-27b-it \
  --preflight
```

The first gating experiment uses deterministic signatures by default and
switches to the LLM-repaired index only when a signature-only side index has
enough confidence:

```bash
python examples/bgb_browser_search/evaluate_bgb_signature_gating.py \
  --stress-report examples/bgb_browser_search/output_full_qwen_turbo/bgb_stress_eval_200k_cycle1_qwen_v2.json \
  --stress-report examples/bgb_browser_search/output_full_qwen_turbo/bgb_stress_eval_200k_cycle2_mistral_small.json \
  --stress-report examples/bgb_browser_search/output_full_qwen_turbo/bgb_stress_eval_200k_cycle3_gemma.json \
  --report examples/bgb_browser_search/output_full_qwen_turbo/bgb_signature_gating_hard40_gemma.json \
  --model google/gemma-3-27b-it
```

The result is a useful warning sign. Aggressive gating repairs hard rows but
hurts global quality; conservative gating preserves global quality but barely
rescues. For example, threshold `2.0` improves hard-row hit@10 by `+0.1313` but
drops global hit@10 by `-0.0108`. Threshold `4.0` has no global hit@10 loss but
only improves hard-row hit@10 by `+0.0404`. So the next trigger should use more
than side-index score: baseline confidence, deep-candidate hard-article hits,
and known confusion pairs.

Adding a low-deterministic-score condition did not help this run. The hard
queries can still have confident lexical matches, just to the wrong article.
That makes confusion-aware gating the more interesting next step.

Fielded static retrieval was also tested on the same Gemma split. Splitting
source, generated summaries, questions, keywords, and held-out-safe train
aliases into separate BM25 fields did not improve the result; the original
combined article view stayed best at `hit@10 = 0.3750`, while fielded RRF was
`0.2803`.

A tiny article reranker is more promising but candidate-limited. With 80 BM25
candidates it improved held-out `hit@10` from `0.3728` to `0.4080`; with 200
candidates it reached `0.4161`. The model artifact was 7.27 MB and reranked
200 candidates in about 6.4 ms/query on this machine. The bottleneck is
candidate recall: BM25 reaches only `0.73` recall at top 200 on this hard split,
so the next useful work is better static candidate generation or teacher-
compressed views, not larger reranker layers.

Direct local candidate generation was also tested. A plain query-to-article
classifier failed (`hit@10 = 0.0395`). Distilling into cached Qwen3 article
vectors worked better but still did not beat BM25: on the hard Gemma split it
reached `hit@10 = 0.1661`, and on all three 200k cycles combined it reached
`0.2816` (`0.5207` hit@50). Training against cached Qwen3 query embeddings was
similar at `0.2771` hit@10. This means the current tiny bag/MLP student has not
captured embedding-level navigation quality; the embedding teacher is still an
upper-bound/evaluation signal, not yet a replaced runtime component.

One embedding-runtime detour is much stronger: if Qwen3 query embeddings are
already available, a small supervised `query embedding -> article ref` head
reached `0.9377` hit@10 and `0.9819` hit@50 on the combined 3-cycle held-out
set. The artifact was about 20.4 MB and the classifier pass took about
0.63 ms/query after embedding. This does not satisfy the browser-offline goal,
but it shows that Refmark supervision can train a cheap corpus-local address
head on top of an existing embedding system.

Query reformulation was tested as a smaller offline path. Global query ->
expansion-term prediction is not good enough yet: naive appending hurt BM25,
and discriminative side-channel fusion only nudged hit@10 from `0.5900` to
`0.5917`. A tiny 10-article oracle loop is much more encouraging: raw BM25
hit@10 was `0.6207`, learned train-derived term banks reached `0.6897`, and
per-query oracle expansion reached `0.8966`. A 129k-param predictor trained on
oracle-improving terms matched the learned-bank hit@10 on that tiny slice.
This suggests the next shape should be surface-conditioned: first narrow to a
small article/section surface, then predict local expansion terms inside that
surface.

```bash
python examples/bgb_browser_search/train_bgb_query_embedding_classifier.py \
  --stress-report examples/bgb_browser_search/output_full_qwen_turbo/bgb_stress_eval_200k_cycle1_qwen_v2.json \
  --stress-report examples/bgb_browser_search/output_full_qwen_turbo/bgb_stress_eval_200k_cycle2_mistral_small.json \
  --stress-report examples/bgb_browser_search/output_full_qwen_turbo/bgb_stress_eval_200k_cycle3_gemma.json \
  --query-cache examples/bgb_browser_search/output_full_qwen_turbo/bgb_stress_query_embeddings_200k_cycle1_qwen_v2.jsonl \
  --query-cache examples/bgb_browser_search/output_full_qwen_turbo/bgb_stress_query_embeddings_200k_cycle2_mistral_small.jsonl \
  --query-cache examples/bgb_browser_search/output_full_qwen_turbo/bgb_stress_query_embeddings_200k_cycle3_gemma.jsonl \
  --output examples/bgb_browser_search/output_full_qwen_turbo/bgb_query_embedding_classifier_3cycle.pt \
  --report examples/bgb_browser_search/output_full_qwen_turbo/bgb_query_embedding_classifier_3cycle_report.json
```

The small resolver trained on that combined suite remains a secondary demo. In
one bounded run it produced a 2.49 MB artifact and improved held-out hit@1 from
`0.7257` to `0.7566`, while preserving the held-out concern rows. Treat that as
encouraging instrumentation, not a polished model claim.

## Why BGB Is Useful

The BGB is large enough to feel real, but still small enough to ship as a
static browser artifact. It also has a natural hierarchy of paragraphs,
sections, and titles, so every hit can jump to a concrete stable region.

The point is not that a local lexical model gives legal reasoning. The point is
that Refmark can make a dense reference work navigable in-browser and can
evaluate whether search lands on the right region.

A later version of this demo can pair the browser BM25 index with a tiny
corpus-local resolver trained on the same ref targets. That should be framed as
an instant-navigation demo over a bounded corpus, not as the core Refmark
product claim.

## Current Smoke

The current full build parses 4,988 BGB regions. Generated artifact sizes:

| Artifact | Raw | gzip |
| --- | ---: | ---: |
| portable Refmark index | 6.87 MiB | 0.87 MiB |
| browser BM25 index | 5.64 MiB | 1.21 MiB |
| browser data JS | 5.64 MiB | 1.21 MiB |
| demo HTML | 2.38 MiB | 0.46 MiB |

The expected-query break suite currently lands all five expected probes in the
top three, with four of five at rank one. That is good enough for a demo, and
the ambiguous/no-answer rows remain useful pressure tests.
