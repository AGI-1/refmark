# BGB Retrieval and Training Experiment Report

This report is a handoff document for reviewing the current Refmark retrieval
experiments with external models or collaborators. It focuses on the BGB
browser-search / evidence-retrieval work because that is currently the most
concrete stress test of the idea:

> Use Refmark refs/ranges to turn a corpus into a measurable evidence space,
> then compare retrieval, adaptation, and training strategies by whether they
> recover the right source region.

The important framing is that Refmark is not the retriever by itself. It gives
the corpus stable addresses, gold targets, highlighting targets, drift targets,
and evaluation metrics. BM25, embeddings, rerankers, LLM-generated views, and
tiny trained models are all interchangeable retrieval strategies over the same
address space.

## Corpus

Primary corpus:

- Source: German Civil Code / BGB HTML export used by the local demo.
- Domain: legal reference text.
- Full parsed corpus: `4,988` Refmark regions.
- Approximate source size: `421k` source tokens.
- Browser/static demo target: a dense but bounded corpus that can be shipped as
  static search assets.

Generated full-demo artifact sizes:

| Artifact | Raw | gzip |
| --- | ---: | ---: |
| portable Refmark index | 6.87 MiB | 0.87 MiB |
| browser BM25 index | 5.64 MiB | 1.21 MiB |
| browser data JS | 5.64 MiB | 1.21 MiB |
| demo HTML | 2.38 MiB | 0.46 MiB |

Granularity:

- Fine regions are paragraph/Absatz-like BGB spans such as
  `bgb:S_437_A01`.
- Article-level refs aggregate those regions to broader anchors such as
  `bgb:S_437`.
- Most current stress metrics use article/block hits, because the first product
  demo target is "jump me to the relevant article"; exact paragraph/range
  scoring remains the next granularity target.

## Question Types

We used several question families because friendly generated questions can make
retrieval look better than it really is.

| Type | Purpose | Notes |
| --- | --- | --- |
| direct | Ask about the article content directly. | Usually easiest and more lexically aligned. |
| concern | User describes a real-world situation or motivation. | Harder and closer to application search. |
| adversarial | Deliberately indirect, rephrased, or confusable. | Useful for breaking lexical search. |
| curated concern | Manually authored/curated concerns. | Useful for demo and targeted regression. |
| natural judge queries | Fresh natural concerns judged by LLM. | Small sample, good sanity check. |
| mixed target | Single, adjacent range, and disjoint support. | Used in OSHA markdown harness, not BGB main run. |

Languages:

- German.
- English.
- The multilingual setup is deliberate: if retrieval views are generated in
  multiple languages, a German legal corpus can still answer English user
  questions.

Question generation models used across experiments:

- `qwen/qwen-turbo`
- `mistralai/mistral-nemo`
- `mistralai/mistral-small-3.2-24b-instruct`
- `google/gemma-3-27b-it`
- `x-ai/grok-4-fast` in the smaller mixed stress run

Embedding model used for the strongest semantic retrieval runs:

- `qwen/qwen3-embedding-8b`

Judge models:

- DeepSeek v4-pro was attempted but unstable through OpenRouter during the run.
- Valid v4-pro judgments were retained where available.
- DeepSeek Chat v3.1 was used as fallback for failed/time-out batches.

## Evaluation Strategy

The core deterministic metric is whether a retrieval method returns the gold
article/ref in top-k.

Common metrics:

- `hit@1`, `hit@10`, `hit@50`.
- `hit@1000` for deep candidate recall.
- `MRR`.
- Per-language splits.
- Per-style splits: direct, concern, adversarial.
- Hard articles and repeated misses.
- Wrong-top confusion pairs.
- Sample misses for qualitative inspection.

Important evaluation rule:

- Generated question text is not trusted as a label. The sampled source
  article/block is the gold label.

Held-out-safe adaptation rule:

- When generated questions are used as training/adaptation metadata, questions
  are split within each article/block.
- Train-half questions may influence static metadata.
- Held-out-half questions evaluate retrieval.
- This avoids exact query leakage while testing whether generated metadata
  generalizes to new phrasings.

## Full BGB Fixed Generated Benchmark

Full BGB fixed generated questions:

- 800 fixed Qwen Turbo generated evaluation questions.
- German and English.
- Qwen Turbo bilingual generated retrieval views.
- Qwen3 embeddings used for semantic comparison.

| Method | hit@1 | hit@10 | MRR |
| --- | ---: | ---: | ---: |
| raw BM25 | 0.3638 | 0.5737 | 0.4344 |
| Refmark BM25 | 0.7638 | 0.9287 | 0.8242 |
| raw Qwen3 embedding | 0.8163 | 0.9750 | 0.8777 |
| Refmark-enriched Qwen3 embedding | 0.8313 | 0.9888 | 0.8953 |
| best low-BM25 hybrid | 0.8263 | 0.9888 | 0.8919 |

Interpretation:

- Refmark-generated views strongly improve BM25 over raw lexical search.
- Qwen3 embeddings are the strongest tested retrieval signal.
- Refmark-enriched embedding text improves over raw source embeddings.
- The exact-anchor benchmark favors Refmark-enriched Qwen3 embeddings.

## Curated Concern Benchmark

Full BGB with generated questions plus 99 held-out curated concern rows:

| Method | overall hit@1 | overall hit@10 | concern hit@1 | concern hit@10 | concern misses |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw BM25 | 0.3471 | 0.5384 | 0.1717 | 0.2828 | 71 |
| Refmark BM25 + generated views + concern aliases | 0.7642 | 0.9366 | 0.8182 | 0.9495 | 5 |
| Refmark rerank | 0.7608 | 0.9333 | 0.8081 | 0.9596 | 4 |

Notes:

- Concern rows were held out from retrieval aliases, so the query text itself
  was not directly pasted into the index.
- Remaining concern misses became concrete adapt-loop targets, including:
  defective purchased goods, limitation periods for defects, pledge expiration,
  partnership exit rights, and finder duties.
- This benchmark is useful for demo/regression but too friendly as the only
  quality measure.

## Randomized Small Stress Runs

Mixed stress run:

- sampled article blocks: 32
- generator models: Qwen Turbo, Mistral Nemo, Mistral Small 3.2 24B, Gemma 3
  27B, Grok 4 Fast
- rows: 953
- styles: direct, concern, adversarial
- languages: German and English

| Method | article hit@1 | article hit@10 | MRR |
| --- | ---: | ---: | ---: |
| raw BM25 | 0.1427 | 0.2445 | 0.1715 |
| Refmark BM25 | 0.3578 | 0.5824 | 0.4304 |
| Refmark rerank | 0.3526 | 0.5887 | 0.4277 |

Refmark BM25 style split:

| Style | Rows | hit@1 | hit@10 |
| --- | ---: | ---: | ---: |
| direct | 318 | 0.6667 | 0.8962 |
| concern | 317 | 0.2145 | 0.4921 |
| adversarial | 318 | 0.1918 | 0.3585 |

Concern-heavy stress run:

- sampled article blocks: 40
- generator models: Qwen Turbo, Mistral Nemo, Mistral Small 3.2 24B, Gemma 3
  27B
- rows: 873

| Method | article hit@1 | article hit@10 | MRR |
| --- | ---: | ---: | ---: |
| raw BM25 | 0.1924 | 0.3116 | 0.2277 |
| Refmark BM25 | 0.4719 | 0.7320 | 0.5533 |
| Refmark rerank | 0.4708 | 0.7308 | 0.5525 |

Repeated hard blocks in the concern-heavy run included:

- `bgb:S_2316`
- `bgb:S_308`
- `bgb:S_2268`
- `bgb:S_14`
- `bgb:S_536b`
- `bgb:S_904`
- `bgb:S_955`
- `bgb:S_965`

Interpretation:

- Refmark BM25 is much better than raw BM25.
- Direct questions are often solved.
- Concern/adversarial questions expose real lexical brittleness.

## 200k-Token Stress Cycles

We then moved to representative 200k-token slices, roughly half of the BGB
source corpus per cycle. These runs use only concern and adversarial questions
in German and English.

Qdrant in WSL was used as build/evaluation infrastructure for Qwen3 embeddings.
This does not change the intended no-vector-runtime deployment target.

| Cycle | Generator | Slice tokens | Blocks | Questions | Refmark BM25 hit@10 | Qwen3 hit@10 | Best hybrid hit@10 | Best hybrid hit@50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Qwen Turbo | 200,136 | 1,002 | 4,008 | 0.6914 | 0.9474 | 0.9506 | 0.9943 |
| 2 | Mistral Small 3.2 24B | 200,020 | 1,038 | 3,916 | 0.6586 | 0.9614 | 0.9620 | 0.9936 |
| 3 | Gemma 3 27B | 200,070 | 1,022 | 3,874 | 0.3446 | 0.9174 | 0.9177 | 0.9845 |

Cycle 3 is the most useful breaker:

- Gemma generated harder or less lexically aligned questions.
- Refmark BM25 dropped to `0.3446` hit@10.
- Qwen3 embeddings still reached `0.9174` hit@10.

Interpretation:

- Embeddings are currently the upper-bound teacher signal.
- Static Refmark BM25 can be useful but degrades sharply on hard phrasing.
- The key research question is how much embedding-teacher quality can be
  transferred into static metadata, hard negatives, and small local models.

## 3-Cycle Heatmap

`report_bgb_retrieval_heatmap.py` aggregates the three 200k stress cycles into
article-level diagnostics.

Baseline static Refmark BM25 over all three cycles:

| Scope | Rows | hit@1 | hit@10 | hit@50 | hit@1000 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3-cycle stress heatmap | 11,798 | 0.3193 | 0.5920 | 0.7434 | 0.9517 | 0.4123 |

Why `hit@1000` matters:

- The correct article is often somewhere in the deep lexical candidate pool.
- It is not ranked high enough for a polished instant-navigation experience.
- This suggests two paths:
  - improve candidate generation;
  - rerank/correct a deep candidate pool.

Heatmap dimensions available or intended:

- Per-article miss rate.
- Per-article low-rank gold positions.
- Per-language performance.
- Per-style performance.
- Per-generator-model performance.
- Wrong-top confusion pairs.
- Articles with high hit@1000 but poor hit@10.
- Articles with no deep candidate recall.
- Sample misses for qualitative review.

Heatmap-to-adapt possibilities:

- Add questions for blind-spot articles.
- Add compact intent signatures for repeatedly missed articles.
- Add hard negatives for repeated wrong-top confusions.
- Merge adjacent regions if gold ranges are too fragmented.
- Expand valid gold ranges when multiple adjacent/distributed refs are genuinely
  valid evidence.
- Exclude low-value boilerplate, summaries, or table-of-contents-like regions
  from training/eval where appropriate.
- Train specialized rescue models only for hard clusters.

## Selective Jump Diagnostics

The next product-facing diagnostic asks a different question from hit@k:

> When is the system confident enough to jump directly, and when should it show
> a candidate list or trigger a rescue path?

`evaluate_bgb_selective_jump.py` computes candidate recall ceilings,
conditional fallback quality, confidence calibration, and precision/coverage
tradeoffs for direct jumps.

Base Refmark BM25 over the same three 200k stress cycles:

| Metric | Value |
| --- | ---: |
| queries | 11,798 |
| hit@1 | 0.3193 |
| hit@10 | 0.5920 |
| hit@50 | 0.7434 |
| hit@200 | 0.8551 |
| hit@1000 | 0.9517 |

Candidate ceiling:

| Candidate pool | recall | `hit@10` if gold is in pool |
| ---: | ---: | ---: |
| 200 | 0.8551 | 0.6924 |
| 1000 | 0.9517 | 0.6221 |

This confirms the earlier diagnosis: there is a large candidate-generation
reservoir, but the correct article is often too deep. A reranker over top-80 or
top-200 cannot approach the embedding teacher unless candidate generation or
candidate expansion improves.

Best direct-jump tradeoff by raw top-score confidence:

| Target jump precision | Jump coverage | Wrong jumps | Fallback hit@10 |
| ---: | ---: | ---: | ---: |
| 0.80 | 0.1048 | 247 | 0.5495 |
| 0.85 | 0.0633 | 112 | 0.5665 |
| 0.90 | 0.0401 | 47 | 0.5755 |
| 0.95 | 0.0121 | 7 | 0.5870 |

Best direct-jump tradeoff by top1/top2 margin:

| Target jump precision | Jump coverage | Wrong jumps | Fallback hit@10 |
| ---: | ---: | ---: | ---: |
| 0.80 | 0.1638 | 385 | 0.5346 |
| 0.85 | 0.1316 | 233 | 0.5429 |
| 0.90 | 0.0883 | 104 | 0.5581 |
| 0.95 | 0.0543 | 32 | 0.5704 |

Best direct-jump tradeoff by normalized margin ratio:

| Target jump precision | Jump coverage | Wrong jumps | Fallback hit@10 |
| ---: | ---: | ---: | ---: |
| 0.80 | 0.1290 | 304 | 0.5521 |
| 0.85 | 0.0961 | 170 | 0.5602 |
| 0.90 | 0.0566 | 66 | 0.5713 |
| 0.95 | 0.0315 | 18 | 0.5797 |

Interpretation:

- Safe direct jumps are possible, but the coverage is modest with simple
  confidence features.
- Margin is the best simple feature for high-precision jumps in this run:
  about `5.43%` of queries can jump at `95%` precision.
- Top score gives lower coverage at high precision but fewer total wrong
  jumps.
- Entropy is more conservative and does not dominate margin/top-score.
- This supports a product shape where the UI auto-jumps only when calibrated
  confidence is high; otherwise it shows top candidates or uses a rescue path.
- A learned gate over richer diagnostics is now more compelling than another
  hand-tuned score threshold.

Split-aware held-out selective-jump runs were then added so adapted indexes can
be evaluated with the same train/eval split used to create their metadata. On
the `5,810` held-out rows from the three-cycle stress suite:

| Index | hit@1 | hit@10 | hit@50 | hit@200 | hit@1000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| base source + views | 0.3196 | 0.5912 | 0.7435 | 0.8594 | 0.9534 |
| deterministic intent signatures | 0.3392 | 0.6250 | 0.7849 | 0.8910 | 0.9738 |
| deterministic + targeted confusion signatures | 0.3389 | 0.6289 | 0.7886 | 0.8929 | 0.9731 |

At `90%` target precision using the top1/top2 margin, direct-jump coverage is:

| Index | Jump coverage | Wrong jumps | Fallback hit@10 | Expected success |
| --- | ---: | ---: | ---: | ---: |
| base source + views | 0.0880 | 51 | 0.5575 | 0.5876 |
| deterministic intent signatures | 0.0417 | 24 | 0.6121 | 0.6241 |
| deterministic + targeted confusion signatures | 0.0420 | 24 | 0.6157 | 0.6277 |

The adapted indexes improve fallback quality and candidate recall, but they do
not expand high-confidence direct-jump coverage. That is useful product
guidance: adaptation currently makes the candidate list/rescue path better; a
separate learned confidence gate is still needed for more automatic jumps.

## Static Adaptation Experiments

### Raw Alias Injection

Hardest Gemma 200k cycle:

1. Split generated questions within each article block.
2. Inject half as static aliases.
3. Evaluate BM25 on the held-out half.

| Static path | held-out hit@10 | held-out hit@50 | MRR |
| --- | ---: | ---: | ---: |
| article baseline | 0.3750 | 0.5633 | 0.2219 |
| aliases mixed into article view | 0.3193 | 0.5184 | 0.1844 |
| alias-only side index | 0.0768 | 0.1499 | 0.0503 |
| source + alias side-index hybrid | 0.3723 | 0.5633 | 0.1468 |

Conclusion:

- Pasting raw generated questions into the index hurts.
- The signal exists, but it must be compressed, weighted, or used with hard
  negative logic.

### Fielded Static Retrieval

Same hard Gemma held-out split. Separate BM25 fields:

- source text
- generated summaries
- generated questions
- generated keywords
- held-out-safe train aliases

Then weighted reciprocal-rank fusion.

| Static path | held-out hit@10 | held-out hit@50 | MRR |
| --- | ---: | ---: | ---: |
| baseline source + generated views | 0.3750 | 0.5633 | 0.2219 |
| fielded original RRF | 0.2803 | 0.4762 | 0.1378 |
| fielded + train aliases RRF | 0.2716 | 0.4800 | 0.1302 |

Conclusion:

- Simple field splitting/RRF did not solve metadata noise.
- The original combined article view stayed better.

### Deterministic Compressed Intent Signatures

Instead of injecting full generated questions, train-half questions were
compressed into short local intent signatures. No LLM was used in this pass.

Held-out half of all three 200k stress cycles:

| Static path | held-out hit@1 | held-out hit@10 | held-out hit@50 | MRR |
| --- | ---: | ---: | ---: | ---: |
| baseline source + generated views | 0.3196 | 0.5912 | 0.7435 | 0.4104 |
| compressed intent signatures mixed in | 0.3392 | 0.6251 | 0.7849 | 0.4350 |
| signature-only side index | 0.1003 | 0.2676 | 0.4294 | 0.1556 |
| source/signature RRF | 0.2084 | 0.5719 | 0.7556 | 0.3135 |

Splits:

- German hit@10: `0.5956 -> 0.6344`
- English hit@10: `0.5869 -> 0.6159`
- adversarial hit@10: `0.6053 -> 0.6408`
- concern hit@10: `0.5772 -> 0.6095`

Conclusion:

- Compact metadata helps.
- Signature-only retrieval is poor.
- Useful pattern: compact intent metadata mixed into source/generated article
  views, not a replacement index.

### LLM Hard-Article Signatures

Targeted the 40 hardest heatmap articles. Generated compact article signatures
from source text plus held-out-safe train questions.

All held-out rows:

| Target | Method | hit@1 | hit@10 | hit@50 | MRR |
| --- | --- | ---: | ---: | ---: | ---: |
| all held-out rows | baseline source + views | 0.3196 | 0.5912 | 0.7435 | 0.4104 |
| all held-out rows | deterministic signatures | 0.3392 | 0.6251 | 0.7849 | 0.4350 |
| all held-out rows | Qwen hard-40 LLM signatures | 0.3127 | 0.5928 | 0.7444 | 0.4059 |
| all held-out rows | deterministic + Qwen hard-40 | 0.3367 | 0.6243 | 0.7843 | 0.4334 |
| all held-out rows | Gemma hard-40 LLM signatures | 0.3189 | 0.5941 | 0.7454 | 0.4110 |
| all held-out rows | deterministic + Gemma hard-40 | 0.3391 | 0.6253 | 0.7866 | 0.4353 |

Selected hard-40 subset:

| Target | Method | hit@1 | hit@10 | hit@50 | MRR |
| --- | --- | ---: | ---: | ---: | ---: |
| hard-40 held-out rows | baseline source + views | 0.0000 | 0.0000 | 0.3434 | 0.0162 |
| hard-40 held-out rows | deterministic signatures | 0.0202 | 0.1616 | 0.4040 | 0.0675 |
| hard-40 held-out rows | Qwen hard-40 LLM signatures | 0.0808 | 0.2828 | 0.5455 | 0.1486 |
| hard-40 held-out rows | Gemma hard-40 LLM signatures | 0.0707 | 0.3030 | 0.5556 | 0.1518 |

Conclusion:

- LLM signatures repair targeted hard zones.
- Unrestricted global use can add noise.
- This is a strong argument for heatmap-guided adapt loops rather than global
  metadata generation everywhere.

### Confusion-Conditioned Signatures

The next pass generated signatures for articles that appeared both in hard
misses and wrong-top confusion pairs. The first run used the default top-40
heatmap and selected only `39` useful articles; it was mostly a negative result
because the target selector was too narrow.

A wider top-300 heatmap selected `80` articles, with `67` overlap articles
between hard misses and confusion golds. This was more informative:

| Scope | Method | hit@1 | hit@10 | hit@50 | MRR |
| --- | --- | ---: | ---: | ---: | ---: |
| all held-out rows | baseline source + views | 0.3196 | 0.5912 | 0.7435 | 0.4104 |
| all held-out rows | deterministic signatures | 0.3392 | 0.6251 | 0.7849 | 0.4350 |
| all held-out rows | confusion signatures only | 0.3200 | 0.5998 | 0.7523 | 0.4132 |
| all held-out rows | deterministic + confusion | 0.3389 | 0.6293 | 0.7886 | 0.4355 |
| selected hard-confusion rows | baseline source + views | 0.0805 | 0.1889 | 0.4180 | 0.1273 |
| selected hard-confusion rows | deterministic signatures | 0.1672 | 0.3406 | 0.5635 | 0.2290 |
| selected hard-confusion rows | confusion signatures only | 0.1672 | 0.3870 | 0.6068 | 0.2376 |
| selected hard-confusion rows | deterministic + confusion | 0.1579 | 0.4087 | 0.6378 | 0.2413 |
| selected hard-confusion rows | deterministic + confusion side RRF | 0.2693 | 0.4241 | 0.6161 | 0.3151 |
| hard-40 rows | baseline source + views | 0.0000 | 0.0000 | 0.3434 | 0.0162 |
| hard-40 rows | deterministic + confusion | 0.0202 | 0.1818 | 0.4646 | 0.0773 |

Interpretation:

- Confusion-conditioned phrases are not a global replacement for deterministic
  signatures.
- They do help the selected hard-confusion subset, especially when treated as a
  side signal.
- The active loop is now concrete: heatmap selects hard/confused articles,
  targeted metadata improves those articles, and held-out metrics show whether
  the repair stayed local or polluted the global index.
- The selector itself must be part of the evaluated artifact. Using too-small
  heatmaps makes the adaptation look worse than it is.

### Signature Gating

Default index: deterministic signatures.

Rescue index: LLM-repaired index.

Gate: switch to rescue index when a signature-only side index score exceeds a
threshold.

| Gate threshold | Global switch rate | Global hit@10 delta vs deterministic | Hard-40 hit@10 delta vs deterministic |
| ---: | ---: | ---: | ---: |
| 0.5 | 0.9628 | -0.0279 | +0.1414 |
| 1.0 | 0.7310 | -0.0199 | +0.1313 |
| 1.5 | 0.5005 | -0.0139 | +0.1313 |
| 2.0 | 0.3327 | -0.0108 | +0.1313 |
| 3.0 | 0.0905 | -0.0012 | +0.1111 |
| 4.0 | 0.0134 | +0.0000 | +0.0404 |
| 5.0 | 0.0017 | +0.0000 | +0.0101 |

Conclusion:

- Aggressive rescue helps hard rows but hurts global quality.
- Conservative rescue preserves global quality but barely repairs.
- Score confidence alone is insufficient because wrong lexical matches can look
  confident.
- More promising gates:
  - known wrong-top confusion pairs;
  - whether deep candidates include a hard article;
  - margin between top candidates;
  - article-level heatmap history;
  - embedding-teacher disagreement.

## Embedding and Hybrid Experiments

### Qwen3 Embeddings

Qwen3 embeddings are currently the strongest retrieval signal.

Best fixed full-BGB benchmark:

- raw Qwen3 embedding hit@10: `0.9750`
- Refmark-enriched Qwen3 embedding hit@10: `0.9888`

200k stress cycles:

- cycle 1 Qwen3 hit@10: `0.9474`
- cycle 2 Qwen3 hit@10: `0.9614`
- cycle 3 Qwen3 hit@10: `0.9174`

Interpretation:

- Embeddings handle concern/adversarial phrasing much better than BM25.
- Refmark views improve embedding retrieval too, not only lexical retrieval.
- Embeddings are a strong teacher for adaptation and hard-negative mining.

### Best Hybrid

200k stress cycle best hybrid:

| Cycle | Best hybrid hit@10 | Best hybrid hit@50 |
| --- | ---: | ---: |
| 1 | 0.9506 | 0.9943 |
| 2 | 0.9620 | 0.9936 |
| 3 | 0.9177 | 0.9845 |

Natural query judging favored a low-BM25 hybrid for answer usefulness and
citation quality, even when exact-anchor scoring favored pure embeddings.

Interpretation:

- Exact localization and answer usefulness are related but not identical.
- Small lexical/Refmark signals may help the top evidence look more citeable
  and useful even when embeddings dominate hit@k.

## Natural Query Judge

12 fresh natural user concerns in German and English, not from the deterministic
test set. A DeepSeek-family judge scored whether each method returned useful
evidence.

| Method | Supported | Avg usefulness | Avg citation quality | Wins |
| --- | ---: | ---: | ---: | ---: |
| raw BM25 | 0.3333 | 0.2083 | 0.2000 | 1 |
| Refmark BM25 | 0.5000 | 0.4792 | 0.4917 | 2 |
| raw Qwen3 embedding | 0.8333 | 0.6500 | 0.6583 | 4 |
| Refmark Qwen3 embedding | 0.9167 | 0.7167 | 0.7500 | 2 |
| Refmark hybrid w0.05 | 0.9167 | 0.8000 | 0.8333 | 3 |

Observed failure:

- For a German property-damage query, all methods missed the central
  `Paragraph 823 I BGB` liability basis and mostly retrieved `Paragraph 249`
  restoration/compensation rules.

Conclusion:

- A judge-based natural query layer is useful because it catches failures that
  exact-anchor generated evaluation may not represent.
- Judge runs must report the judge model, provider failures, and fallbacks.

## Tiny Training Experiments

### Full BGB Tiny Resolver

Small local resolver:

- About `1.31M` parameters.
- About `5.43 MB` artifact.
- About `21.9 minutes` CPU training time on the local mini PC.
- Low-millisecond inference in the current evaluation loop.

Full BGB resolver result:

- BM25 baseline on resolver eval split: hit@1 `0.6562`, hit@10 `0.9152`,
  MRR `0.7457`.
- Best blend: hit@1 `0.6652`, hit@10 `0.9018`, MRR `0.7530`.

Interpretation:

- Some MRR/rank-1 signal appears.
- Not enough to claim a tiny model is the main product value.
- Training is currently an experimental consumer of the Refmark evidence space.

### Article Pair Reranker

Trained a small article-level pair reranker over BM25 candidates on the hard
Gemma split.

| Candidate depth | BM25 candidate recall | Best blended hit@10 | Best blended hit@50 | Artifact |
| ---: | ---: | ---: | ---: | ---: |
| 80 | 0.6272 | 0.4080 | 0.6039 | 7.27 MB |
| 200 | 0.7300 | 0.4161 | 0.6136 | 7.27 MB |

Runtime:

- About `6.4 ms/query` to rerank 200 candidates on this machine.

Candidate recall on the same hard split:

- BM25 top 200 recall: `0.73`
- BM25 top 500 recall: `0.84`
- BM25 top 1000 recall: `0.91`
- Qwen3 embedding retrieval hit@10: `0.9174`

Interpretation:

- Reranking helps inside the candidate pool.
- Candidate generation is the limiting factor.
- A reranker cannot recover gold refs absent from top-k.

### Embedding-Teacher Text Reranker

Next we tested whether Qwen3 embeddings can act as offline supervision for a
text-only reranker. The runtime path remains lexical:

```text
query -> Refmark BM25 candidates -> small text pair reranker -> article refs
```

During training only, cached Qwen3 query vectors and cached Qwen3 article
vectors provide soft teacher scores. The model sees query text, article text,
BM25 rank/score, and lexical overlap features. It does not receive embeddings
at runtime.

Clean base-index run:

- Index: base `bgb_openrouter_index.json`.
- Stress data: all three 200k cycles.
- Train questions: `5,988`.
- Eval questions: `5,810`.
- Candidate depth: `80`.
- Candidate rows: `479,618` train, `464,800` eval.
- Model size: `3.53M` parameters, `14.74 MB` artifact.
- Training time: `368.35 s` for 3 epochs.
- Inference: about `4.53 ms/query` for the best blend over 80 candidates.

| Method | hit@1 | hit@10 | hit@50 | MRR |
| --- | ---: | ---: | ---: | ---: |
| base Refmark BM25 | 0.3127 | 0.5678 | 0.7263 | 0.4008 |
| pure teacher reranker | 0.3145 | 0.5816 | 0.7475 | 0.4078 |
| best BM25 + teacher-reranker blend | 0.3133 | 0.5997 | 0.7509 | 0.4100 |

Best blend splits:

| Split | hit@10 | hit@50 | MRR |
| --- | ---: | ---: | ---: |
| German | 0.6034 | 0.7451 | 0.4241 |
| English | 0.5960 | 0.7566 | 0.3962 |
| adversarial | 0.6158 | 0.7575 | 0.4304 |
| concern | 0.5829 | 0.7441 | 0.3888 |

Interpretation:

- This is a real but modest positive: `+0.0319` hit@10 and `+0.0246` hit@50
  over base Refmark BM25.
- The reranker improves deeper placement more than rank-1.
- It still sits far below Qwen3 embedding retrieval, so teacher supervision has
  not been distilled into a full local semantic retriever.
- Candidate generation remains the ceiling.

A warning from this run: using the prebuilt deterministic-signature index with a
different train/eval split produced a much higher baseline (`0.8129` hit@10),
which is likely split leakage because held-out stress phrasings may already be
present in the adapted index. Treat that run as invalid except as a reminder
that adapted indexes must carry their exact split seed and source reports.

The same warning reappeared in a larger reranker pass: evaluating an adapted
index built with split seed `1515` through a reranker split seeded with `3535`
produced an apparent `0.8205` blended hit@10. That number is invalid as a
quality claim. Rerunning with the matching `1515` split gives the clean result:

| Method | hit@1 | hit@10 | hit@50 | MRR |
| --- | ---: | ---: | ---: | ---: |
| adapted deterministic + confusion index | 0.3389 | 0.6293 | 0.7886 | 0.4362 |
| pure teacher reranker | 0.3394 | 0.6050 | 0.7807 | 0.4320 |
| best adapted index + teacher-reranker blend | 0.3399 | 0.6389 | 0.7928 | 0.4405 |

Clean run details:

- Index: deterministic intent signatures plus top-300 confusion signatures.
- Split seed: `1515`, matching the adapted index.
- Candidate depth: `80`.
- Model size: `3.51M` parameters, `14.65 MB` artifact.
- Training time: `393.71 s` for 3 epochs.
- Inference: about `4.62 ms/query` for the best blend over 80 candidates.

This is a small but real compound gain over the adapted static index. It also
shows that the pure teacher-reranker score can hurt hit@10; blending is doing
the useful work.

### Direct Candidate Generation and Distillation

Tested whether the embedding teacher can be replaced by a tiny local model.

Variants:

1. Query text -> article class.
2. Query text -> vector close to Qwen3 article vector.
3. Query text -> vector close to cached Qwen3 query embedding.
4. Cached Qwen3 query embedding -> article class.

| Local generator | Eval split | Held-out hit@10 | Held-out hit@50 | Runtime note |
| --- | --- | ---: | ---: | --- |
| direct text label classifier | Gemma 200k | 0.0395 | 0.1012 | 15.8 MB artifact, ~0.24 ms/query |
| text -> gold article-vector distill | Gemma 200k | 0.1661 | 0.3653 | 38.2 MB artifact, ~1.42 ms/query |
| text -> gold article-vector distill | 3-cycle combined | 0.2816 | 0.5207 | 38.2 MB artifact, ~1.45 ms/query |
| text -> Qwen3 query-embedding distill | 3-cycle combined | 0.2771 | 0.5270 | 38.2 MB artifact, ~1.45 ms/query |
| cached Qwen3 query embedding -> article classifier | 3-cycle combined | 0.9377 | 0.9819 | 20.4 MB artifact, ~0.63 ms/query after embedding |
| BM25 article baseline | Gemma 200k | 0.3712 | 0.5714 | static lexical |
| BM25 article baseline | 3-cycle combined | 0.5974 | 0.7491 | static lexical, same split as embedding-classifier run |

Cached embedding classifier details:

- Input: already-computed Qwen3 query embedding.
- Output: article/ref class.
- Parameters: `5.09M`.
- Artifact: `20.4 MB`.
- Train questions: `5,988`.
- Eval questions: `5,810`.
- Train article count: `1,756`.
- Eval article count: `1,755`.
- hit@1: `0.6910`.
- hit@3: `0.8547`.
- hit@5: `0.8960`.
- hit@10: `0.9377`.
- hit@20: `0.9642`.
- hit@50: `0.9819`.
- MRR: `0.7833`.
- Classifier inference after embedding: about `0.63 ms/query`.

Split details for the cached embedding classifier:

- German hit@10: `0.9326`, hit@50: `0.9767`, MRR: `0.7776`.
- English hit@10: `0.9428`, hit@50: `0.9872`, MRR: `0.7891`.
- adversarial hit@10: `0.9372`, hit@50: `0.9799`, MRR: `0.7816`.
- concern hit@10: `0.9382`, hit@50: `0.9840`, MRR: `0.7850`.

Important interpretation:

- Text-only local distillation has not worked well yet.
- The embedding space itself contains a strong learnable address signal.
- If an application already computes query embeddings, a small corpus-local
  `embedding -> ref` head is practical and fast.
- This does not satisfy the fully browser-offline goal because the embedding
  model is still required at query time.

## Mixed Target / Range Evidence

Separate local OSHA Markdown run:

- Generated Markdown legal corpus.
- 250k-token budget.
- 120 mixed targets.

Gold modes:

| Gold mode | Count |
| --- | ---: |
| single | 62 |
| adjacent range | 49 |
| disjoint refs | 9 |

Selected results:

| Method | hit@1 | hit@10 | context hit@10 | MRR | range cover@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw BM25 flat | 0.275 | 0.742 | 0.633 | 0.412 | 0.704 |
| local-view BM25 flat | 0.325 | 0.792 | 0.692 | 0.476 | 0.754 |
| local-view BM25 rerank | 0.308 | 0.800 | 0.700 | 0.457 | 0.758 |
| local-view learned rerank | 0.233 | 0.883 | 0.650 | 0.396 | 0.783 |

Interpretation:

- Refmark can evaluate single refs, adjacent ranges, and disjoint evidence in
  one harness.
- Learned reranking increased top-10/range coverage but hurt rank-1 and MRR.
- This is exactly the kind of over/under-citation tradeoff that Refmark makes
  visible.

## What Helped

- Refmark-generated local views substantially improved BM25.
- Qwen3 embeddings were the strongest tested retrieval signal.
- Refmark-enriched embedding text improved over raw source embeddings.
- Multilingual generated views enabled English queries over German legal text.
- Deterministic compressed intent signatures improved static BM25.
- LLM signatures helped targeted hard heatmap zones.
- Curated concern aliases improved demo-style motivation queries when held out
  properly.
- Low-BM25 hybrid helped natural query usefulness/citation quality.
- Cached embedding -> ref classifier worked well when query embeddings were
  available.
- Stable refs made every failure inspectable: exact misses, neighbor misses,
  wrong-top confusions, stale labels, overbroad ranges, and underbroad ranges.

## What Did Not Help Yet

- Raw generated aliases hurt.
- Alias-only side indexes were poor.
- Fielded BM25/RRF over generated fields was worse than the combined baseline.
- Simple score-based rescue gating was too blunt.
- Low deterministic top score was not a reliable rescue trigger.
- Direct query-text -> article classifier failed.
- Text -> embedding/vector distillation remained below BM25.
- Tiny rerankers are candidate-recall limited.
- DeepSeek v4-pro was too unstable through OpenRouter for unattended judging in
  this run.
- Fixed generated benchmarks can be too friendly unless complemented by random
  stress and natural judge checks.

## Plausible Next Experiments

### 1. Embedding-Teacher Reranker With Hard Negatives

Train a small reranker on lexical/deep candidates using Qwen3 teacher ranks.

Key design:

- Candidate pool from Refmark BM25 + deterministic signatures.
- Positives from gold refs and Qwen3 top hits.
- Hard negatives from repeated wrong-top BM25 candidates.
- Evaluate on held-out stress questions.

Why plausible:

- BM25 hit@1000 is high (`0.9517`) across the 3-cycle heatmap.
- A reranker could be useful if candidate generation gets gold into the pool.

Risk:

- Top 200 recall is not enough on hard splits.
- Need candidate expansion before reranking can close the gap.

### 2. Confusion-Aware Gating

Current gates switch based on side-index score only. Better gate candidates:

- known wrong-top pair appears;
- hard article appears in deep candidates;
- top BM25 margin is suspicious;
- embedding teacher strongly disagrees with BM25;
- article has repeated historical misses in heatmap.

Goal:

- Use LLM signatures or rescue index only where it repairs known weak zones.
- Avoid global metadata noise.

### 3. Per-Article Intent Prototypes

Use successful query embeddings to build compact article prototypes.

Possible forms:

- centroid per article;
- multiple centroids per article from clustered query intents;
- compressed float16/quantized vectors;
- learned low-dimensional projection.

Why plausible:

- Cached query embedding -> article classifier performed very well.
- The address signal is visible in embedding space.

Open question:

- Can prototypes be used without a full vector database?
- Can a tiny text model map queries into this prototype space better than the
  current bag/MLP distillation?

### 4. Stronger Text Encoder Student

The current text-only student is too weak. Options:

- pretrained multilingual encoder small enough for browser/edge;
- character n-gram / hashed subword architecture;
- two-tower contrastive loss instead of MSE vector distillation;
- article coarse head + fine rerank head;
- larger local model, still under 20-50 MB.

Acceptance bar:

- Must beat static Refmark BM25 on held-out concern/adversarial stress.
- Must preserve rank-1/MRR, not only top-50.

### 5. Active Adapt Loop

Formalize:

```text
train -> evaluate -> heatmap -> adapt -> retrain -> compare
```

Adapt actions:

- add hard questions;
- add compact signatures;
- add hard negatives;
- merge adjacent refs;
- allow multi-positive gold ranges;
- exclude boilerplate or summaries;
- regenerate stale examples after corpus drift.

This may be more product-relevant than any single model improvement.

### 6. Range and Multi-Positive Labels

Current BGB main stress is mostly article-level. Next scoring should include:

- exact article hit;
- exact paragraph hit;
- neighbor hit;
- parent/section hit;
- valid range coverage;
- overcitation;
- undercitation;
- disjoint support detection.

This better matches real RAG evidence needs.

### 7. Cross-Corpus Reproducibility

BGB is promising but domain-specific. Need at least one technical docs corpus.

Good target:

- structured product/API docs;
- 200k to 1M tokens;
- clear headings/sections;
- natural user questions.

Compare:

- raw BM25;
- Refmark BM25;
- Refmark BM25 + deterministic signatures;
- embeddings;
- hybrid;
- trained reranker;
- cached embedding address head if embeddings are available.

### 8. Browser Demo Integration

Current static assets are small enough for browser use.

Useful demo features:

- query box;
- jump to article/ref;
- highlight region;
- show top candidates;
- show confidence/debug traces;
- show why a hard query failed;
- optionally load tiny resolver or static signatures.

Do not overclaim:

- BM25/static browser path is useful and inspectable, not currently equal to
  Qwen3 embeddings on hard concern/adversarial questions.

## Current Claims That Are Fair

Fair:

- Refmark turns retrieval evaluation into evidence-region recovery.
- Refmark-generated views can substantially improve lexical search.
- Refmark-enriched embeddings improved over raw embeddings on BGB.
- Qwen3 embeddings are currently the best tested teacher signal.
- Static compact intent signatures improved held-out retrieval.
- Heatmaps identify concrete weak articles and confusion patterns.
- Targeted LLM signatures repair hard heatmap rows but need careful gating.
- Tiny training is promising as a secondary/rescue signal, not yet a main
  product claim.
- If query embeddings already exist, a small corpus-local `embedding -> ref`
  head can be very accurate and fast.

Not fair yet:

- "Tiny offline model beats embeddings."
- "Browser-only search matches semantic vector search."
- "Refmark replaces BM25/vector databases."
- "Training is the main value."
- "The BGB result generalizes without testing other corpora."

## Useful Source Files and Artifacts

Main documentation:

- `docs/SEARCH_AND_TRAINING_FINDINGS.md`
- `examples/bgb_browser_search/README.md`

Main scripts:

- `examples/bgb_browser_search/run_bgb_pipeline.py`
- `examples/bgb_browser_search/run_bgb_stress_eval.py`
- `examples/bgb_browser_search/evaluate_bgb_stress_embeddings.py`
- `examples/bgb_browser_search/report_bgb_retrieval_heatmap.py`
- `examples/bgb_browser_search/evaluate_bgb_selective_jump.py`
- `examples/bgb_browser_search/evaluate_bgb_intent_signatures.py`
- `examples/bgb_browser_search/evaluate_bgb_llm_intent_signatures.py`
- `examples/bgb_browser_search/evaluate_bgb_signature_gating.py`
- `examples/bgb_browser_search/evaluate_bgb_fielded_static.py`
- `examples/bgb_browser_search/train_bgb_article_resolver.py`
- `examples/bgb_browser_search/train_bgb_article_candidate_generator.py`
- `examples/bgb_browser_search/train_bgb_embedding_distilled_generator.py`
- `examples/bgb_browser_search/train_bgb_query_embedding_classifier.py`

Generated reports live under:

- `examples/bgb_browser_search/output_full_qwen_turbo/`

Generated outputs are research artifacts and should not be committed unless
explicitly promoted.
