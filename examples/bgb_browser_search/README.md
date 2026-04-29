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

Provenance matters here. An early LLM-signature cache keyed only on
article/model/limit even though the prompt could include train-query snippets.
That was hardened so the cache now includes prompt context. With a cleaner
cycle1-only, article-text-only Qwen run, transfer is modest rather than
spectacular: cycle2 hit@10 stays essentially neutral (`0.6885 -> 0.6890`) while
the harder Gemma cycle3 improves from `0.3561` to `0.3707` hit@10 and from
`0.6531` to `0.6824` hit@100. Treat same-split hard40 numbers as targeted
repair/upper-bound evidence unless split provenance is explicit.

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

Fresh full-cycle fielded and index-fusion loops kept the same lesson. On a
small 32x5 slice, raw train-question aliases looked spectacular
(`hit@10 = 0.8700`), but on the combined 3-cycle held-out run they collapsed to
`0.2880`. That marks raw aliases as overfit metadata, not a reusable static
adaptation. Durable indexes transferred more honestly: the base generated-view
index reached `0.5912` hit@10, deterministic intent signatures reached
`0.6250`, and confusion-signature repair reached `0.6289`. Cycle1-only
signatures evaluated against later cycles gave a smaller but cleaner lift:
`0.5297 -> 0.5525` hit@10 when blended with the base index.

BM25 parameter tuning was also checked and is not the missing lever. A sampled
grid over `k1` and `b` suggested only tiny movement, and full validation showed
the default `k1=1.5, b=0.75` essentially tied the best setting. For the
confusion-signature index, default hit@10 was `0.6289`; the sampled best
`k1=1.2, b=0.75` reached `0.6287`.

The more useful product metric is local neighborhood recovery. With the
confusion-signature index, exact article hit@10 on the combined held-out split
is `0.6289`, but accepting a tight adjacent-article window gives:

| Article window | hit@1 | hit@10 | hit@50 | MRR |
| --- | ---: | ---: | ---: | ---: |
| exact | 0.3389 | 0.6289 | 0.7886 | 0.4363 |
| +/-1 article | 0.3852 | 0.6998 | 0.8547 | 0.4908 |
| +/-2 articles | 0.4108 | 0.7387 | 0.8900 | 0.5204 |
| +/-5 articles | 0.4525 | 0.7928 | 0.9330 | 0.5680 |

This does not replace exact evidence scoring, but it supports a better browser
UX: direct jump only when confident, otherwise open a small local cluster and
highlight the ranked refs. The repeatable report is generated with:

```bash
python examples/bgb_browser_search/evaluate_bgb_neighbor_windows.py \
  --index examples/bgb_browser_search/output_full_qwen_turbo/bgb_confusion_signatures_gemma_top300_index.json \
  --stress-report examples/bgb_browser_search/output_full_qwen_turbo/bgb_stress_eval_200k_cycle1_qwen_v2.json \
  --stress-report examples/bgb_browser_search/output_full_qwen_turbo/bgb_stress_eval_200k_cycle2_mistral_small.json \
  --stress-report examples/bgb_browser_search/output_full_qwen_turbo/bgb_stress_eval_200k_cycle3_gemma.json \
  --output examples/bgb_browser_search/output_full_qwen_turbo/bgb_neighbor_window_eval_3cycle.json \
  --split eval
```

The first explicit coarse-to-fine area router is now also measurable. It builds
overlapping 50-article windows with stride 25, ranks areas first, then reranks
articles inside the selected area union:

```bash
python examples/bgb_browser_search/evaluate_bgb_area_router.py \
  --index examples/bgb_browser_search/output_full_qwen_turbo/bgb_confusion_signatures_gemma_top300_index.json \
  --stress-report examples/bgb_browser_search/output_full_qwen_turbo/bgb_stress_eval_200k_cycle1_qwen_v2.json \
  --stress-report examples/bgb_browser_search/output_full_qwen_turbo/bgb_stress_eval_200k_cycle2_mistral_small.json \
  --stress-report examples/bgb_browser_search/output_full_qwen_turbo/bgb_stress_eval_200k_cycle3_gemma.json \
  --output examples/bgb_browser_search/output_full_qwen_turbo/bgb_area_router_w50_s25_3cycle.json \
  --split eval \
  --area-size 50 \
  --stride 25
```

Hard filtering through top areas was too lossy: top 8 areas covered the gold
article `81.34%` of the time, but final exact hit@10 reached only `0.6059`,
below flat article search at `0.6289`. As a soft prior, however, area routing
does help candidate-list recall: boosting articles in the top 5 areas by `0.1`
raised hit@10 to `0.6456` and hit@50 to `0.8010`, but lowered hit@1 to
`0.2874` and MRR to `0.4026`. So the current lesson is:

```text
area routing is useful as a candidate-list / fallback prior,
not yet as a direct-jump gate.
```

An explicit equal-window area classifier makes the limit clearer. With 51
non-overlapping areas of about 50 articles each, a tiny text-only
query-to-area model overfits train and reaches only `0.3024` top-1,
`0.6477` top-5, and `0.7461` top-8 on held-out questions. A cached Qwen3
query-embedding head is much stronger: `0.7609` top-1, `0.8988` top-2,
`0.9435` top-3, and `0.9726` top-5. So a single-area `0.9` router is not
there yet, but a two-to-three-area coarse router is realistic if embeddings or
an embedding-quality student are allowed.

The coarser 17-area version, about 150 articles per area, is much more viable:

| Area router | top-1 | top-2 | top-3 | top-5 | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| full 4096-d Qwen3 embedding + MLP | 0.8036 | 0.9313 | 0.9642 | 0.9855 | Requires query embedding at runtime. |
| PCA-1024 + MLP | 0.8102 | 0.9308 | 0.9676 | 0.9869 | Best top-1 in this run. |
| PCA-256 + MLP | 0.7735 | 0.9127 | 0.9563 | 0.9843 | 105k-param head; PCA matrix storage not included. |
| PCA-128 + MLP | 0.7373 | 0.8971 | 0.9468 | 0.9806 | 56k-param head; useful two-area reducer. |
| PCA-128 int8 centroid | 0.6243 | 0.8067 | 0.8793 | 0.9473 | About 2.2 KB centroid table, but weaker precision. |
| PCA-16 int8 centroid | 0.5091 | 0.7165 | 0.8164 | 0.9103 | Tiny coarse fallback, not enough for direct jump. |

This is still not browser-offline search by itself: these routers consume
cached Qwen3-style query embeddings. The useful product shape is a cheap
coarse layer on top of an existing embedding system, or a teacher target for a
future local text/embedding student.

Area size is the main tradeoff:

| Area size | Area count | best top-1 | best top-2 | best top-5 | PCA-128 MLP top-2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 75 articles | 34 | 0.7404 | 0.8861 | 0.9742 | 0.8353 |
| 100 articles | 26 | 0.7792 | 0.9081 | 0.9812 | 0.8769 |
| 150 articles | 17 | 0.8102 | 0.9308 | 0.9869 | 0.8971 |
| 200 articles | 13 | 0.8138 | 0.9379 | 0.9898 | 0.9081 |
| 300 articles | 9 | 0.8401 | 0.9513 | 0.9950 | 0.9301 |

So the hierarchy should start broad: choose two or three areas first, then run
article/paragraph retrieval inside the union. Very narrow first-stage areas
are still too brittle.

For the 9-area setup, `evaluate_bgb_area_reducer_sweep.py` also checks whether
dimension reduction or shuffled coordinate slices can improve the coarse
router. In the current run, reduction did not beat the full embedding head, but
it preserved most of the signal:

| Reducer | dims | top-1 | top-2 | top-5 | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| full embedding + MLP | 4096 | 0.8353 | 0.9465 | 0.9964 | Highest ceiling in this sweep. |
| PCA + MLP | 512 | 0.8277 | 0.9427 | 0.9950 | Nearly full quality. |
| PCA + MLP | 256 | 0.8096 | 0.9334 | 0.9936 | Strong compact head. |
| PCA + MLP | 128 | 0.7869 | 0.9255 | 0.9936 | Good broad-area reducer. |
| PCA + MLP | 64 | 0.7554 | 0.9098 | 0.9914 | Still viable for top-2/top-5 routing. |
| PCA int8 centroid | 128 | 0.6747 | 0.8437 | 0.9773 | About 1.1 KB centroid table; projection excluded. |

Random coordinate samples at 512 dimensions got close to PCA, but did not beat
it. The practical takeaway is that the area signal is broadly distributed in
the embedding, and int8 centroid quantization is almost free at this coarse
level.

At higher granularity, PCA-128/256 still works as a neighborhood generator:

| Area size | Areas | full top-5 | PCA128 top-5 | PCA256 top-5 |
| ---: | ---: | ---: | ---: | ---: |
| 5 articles | 506 | 0.9153 | 0.8902 | 0.8924 |
| 10 articles | 253 | 0.9324 | 0.8914 | 0.9002 |
| 25 articles | 102 | 0.9511 | 0.9236 | 0.9343 |
| 50 articles | 51 | 0.9697 | 0.9525 | 0.9594 |
| 100 articles | 26 | 0.9821 | 0.9714 | 0.9757 |

So compressed embeddings look useful for first-stage candidate generation even
near 5-article neighborhoods. They are still weaker for precise top-1/top-2
jumps, so the next sensible demo path is compressed area candidates followed by
a local resolver/reranker.

`evaluate_bgb_local_embedder.py` tests the no-Qwen runtime path with frozen
local SentenceTransformer models. On the same held-out stress questions,
MiniLM was weak, while `intfloat/multilingual-e5-small` over generated Refmark
views was much more useful:

| Local embedder path | article hit@10 | article hit@50 | area25 top-5 | area50 top-5 | area100 top-5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MiniLM, combined text | 0.3360 | 0.5391 | 0.5561 | 0.6515 | 0.7613 |
| e5-small, source text | 0.5077 | 0.7003 | 0.6539 | 0.7243 | 0.8036 |
| e5-small, source + view | 0.6069 | 0.7566 | 0.7668 | 0.8320 | 0.8912 |
| e5-small, Refmark view only | 0.7272 | 0.8818 | 0.8391 | 0.8873 | 0.9263 |

The view-only result matters: generated bilingual Refmark retrieval views make
a frozen local embedder much stronger and remove Qwen/OpenRouter from query
runtime. It is still below Qwen3 embeddings, but it is plausible for browser or
edge candidate generation followed by local refinement.

`evaluate_bgb_embedding_area_bm25_stack.py` tests that staged shape directly:

```text
local e5 over Refmark views -> top article areas -> BM25/signature resolver inside the area union
```

On the 3-cycle 200k stress split (`5,810` held-out concern/adversarial rows),
flat no-embedding BM25/signature search reached hit@10 `0.6289`, hit@50
`0.7886`. The best e5-area + inside-area fusion reached hit@10 `0.6995`,
hit@50 `0.8587`, but with a broad top-10 `100`-article area union of about
`979` candidate articles. A smaller top-10 `25`-article setup searched about
`250` articles and reached hit@10 `0.6947`, hit@50 `0.8408`.

On the older mixed direct/concern/adversarial split, the stack improved hit@10
from `0.6771` to `0.7631`; by style it reached direct `0.9576`, concern
`0.7152`, and adversarial `0.6039` hit@10. This supports the current hierarchy:
use embeddings as a semantic area router, then use Refmark BM25/signatures for
the final article ranking and evidence refs.

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

The first no-leak surface-conditioned runs are modest but valid. On a 32-article
slice with BM25 surface-k 20, a 2.78 MB local model improved BM25 from hit@10
`0.6122`, MRR `0.4424` to hit@10 `0.6226`, MRR `0.4564`. On a 12-article slice
with oracle-derived term labels, fusion improved hit@10 `0.6554 -> 0.6723` and
MRR `0.4978 -> 0.5299`. This is a useful reranker/local-booster signal, not yet
a replacement for the coarse router.

The current bottleneck is measurable with `report_bgb_surface_recall.py`.
On the 32x5 split, enriched article BM25 reaches hit@100 `0.8008` but only
hit@10 `0.6122`. Direct queries are much easier (`0.9108` hit@10) than concern
queries (`0.5394`) or adversarial wording (`0.3871`). The next useful router
work should therefore improve broad surface recall for motivation-style and
adversarial questions.

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
