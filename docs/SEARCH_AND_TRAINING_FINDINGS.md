# Refmark Search and Training Findings

This note captures the current experimental state. It is meant as a handoff for
reviewers or external models, not as a polished product claim.

## Current Mental Model

Refmark gives documents stable, inspectable region identifiers. That changes
retrieval work in two ways:

- Search systems can return a concrete region or range, not only a text blob.
- Evaluation can be deterministic: a query generated from a known region can be
  scored by whether the retrieval system recovers that region or a permitted
  range.

The strongest product angle so far is:

> Refmark adds a low-cost evidence-resolution layer to existing RAG/search
> pipelines.

This does not replace BM25, embeddings, or answer generation. It gives them a
stable navigation/evaluation layer and makes it cheap to compare them.

For the large-corpus pipeline framing, see
`docs/EVIDENCE_RETRIEVAL_PIPELINE.md`. That note makes the evidence-retrieval
loop explicit: map a structured corpus to refs/ranges, generate or curate
questions, evaluate multiple retrieval approaches, and keep the resulting
metrics as CI evidence.

## Experiment Ledger

Tried so far:

| Track | Status | Finding |
| --- | --- | --- |
| Full-corpus exact-anchor retrieval | strong positive | Refmark-enriched Qwen3 embeddings reached `0.9888` hit@10 on the fixed generated benchmark. |
| Randomized 200k stress cycles | strong positive for embeddings, mixed for BM25 | Qwen3 embeddings stayed around `0.9174-0.9614` hit@10; static Refmark BM25 ranged from `0.3446` to `0.6914` depending on generator hardness. |
| Raw generated aliases | negative | Pasting full synthetic questions into metadata hurt held-out BM25. |
| Fielded BM25 / RRF over generated fields | negative | Splitting source/summary/questions/keywords into fields did not recover the lost signal. |
| Deterministic compressed intent signatures | positive | Improved 3-cycle held-out hit@10 from `0.5912` to `0.6251`. |
| LLM hard-article signatures | targeted positive | Repaired hard-40 hit@10 from `0.0000` baseline to `0.3030`, but needs gating to avoid global noise. |
| Confusion-conditioned signatures | targeted positive, modest globally | With a wider top-300 heatmap, selected hard-confusion rows improved from `0.1889` to `0.4087` hit@10 when combined with deterministic signatures; global held-out hit@10 moved from `0.6251` to `0.6293`. |
| Split-aware selective jump | product diagnostic | Adapted indexes improve held-out fallback/candidate quality, but high-precision auto-jump coverage remains small: around `4.2%` coverage at `90%` precision for the best adapted margin gate. |
| Side-score and low-confidence gating | partial/negative | Aggressive gates repair hard rows but hurt global quality; low deterministic score is not enough because wrong matches can look confident. |
| Tiny BM25-candidate reranker | modest positive | Improved hard Gemma split hit@10 from about `0.3728` to `0.4080`, candidate-recall limited. |
| Embedding-teacher text reranker | modest positive | Training from Qwen3 soft scores improved clean 3-cycle base BM25 hit@10 from `0.5678` to `0.5997`, but remains candidate-limited and far below embeddings. |
| Embedding-teacher reranker over adapted index | small positive after fairness fix | With matching split seed, adapted static hit@10 `0.6293` improved to `0.6389` in the best blend; a seed-mismatched run gave an invalid `0.8205` warning case. |
| Direct text -> article classifier | negative | Hit@10 only `0.0395` on Gemma 200k. |
| Text -> article-vector/query-vector distillation | weak positive | 3-cycle hit@10 around `0.28`, below BM25. |
| Cached query embedding -> article classifier | strong but different runtime | Hit@10 `0.9377`, hit@50 `0.9819`; proves query embeddings contain learnable address signal, but still requires query embeddings at runtime. |

Not tried yet or only lightly touched:

- Confusion-aware gating: switch to repair signatures when top candidates match known wrong-top pairs or heatmap clusters.
- Stronger embedding-teacher reranker over deeper/better candidates with hard negatives from Qwen3 ranks.
- Compact per-ref learned vectors or prototypes distilled from query clusters, used as static metadata/features.
- Multi-positive/range labels where several adjacent or alternative refs count as valid evidence.
- Active adapt loop: heatmap -> add/merge/exclude/expand examples -> retrain/re-evaluate.
- Broader corpus reproducibility beyond BGB, especially technical documentation with article/page hierarchy.
- Browser-side demo integration of the best static path plus visible heatmap/debug traces.

## BGB Stress Test

The current real-world stress corpus is the German Civil Code (BGB), parsed from
the official HTML source.

Full BGB run:

- 4,988 regions.
- About 421k source tokens.
- 800 fixed Qwen Turbo generated evaluation questions, German and English.
- 99 held-out curated concern questions in the latest concern run.
- Qwen Turbo bilingual generated retrieval views.
- Qwen3 Embedding 8B for semantic retrieval.

Important artifacts:

- `examples/bgb_browser_search/output_full_qwen_turbo/bgb_pipeline_report.json`
- `examples/bgb_browser_search/output_full_qwen_turbo/qwen3_embedding_compare.json`
- `examples/bgb_browser_search/output_full_qwen_turbo/deepseek_natural_judge_combined.json`

The generated outputs are ignored and should not be committed.

## Deterministic Retrieval Results

Full BGB, fixed generated questions:

| Method | hit@1 | hit@10 | MRR |
| --- | ---: | ---: | ---: |
| raw BM25 | 0.3638 | 0.5737 | 0.4344 |
| Refmark BM25 | 0.7638 | 0.9287 | 0.8242 |
| raw Qwen3 embedding | 0.8163 | 0.9750 | 0.8777 |
| Refmark-enriched Qwen3 embedding | 0.8313 | 0.9888 | 0.8953 |
| best low-BM25 hybrid | 0.8263 | 0.9888 | 0.8919 |

The exact-anchor benchmark currently favors Qwen3 embeddings over
Refmark-enriched region text. This is the strongest full-corpus result.

On the 600-region slice, the low-BM25 hybrid slightly improved over pure
Refmark-enriched embeddings, but that did not hold as the best exact-anchor
method on full BGB.

Full BGB, generated questions plus 99 held-out concern rows:

| Method | overall hit@1 | overall hit@10 | concern hit@1 | concern hit@10 | concern misses |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw BM25 | 0.3471 | 0.5384 | 0.1717 | 0.2828 | 71 |
| Refmark BM25 + generated views + concern aliases | 0.7642 | 0.9366 | 0.8182 | 0.9495 | 5 |
| Refmark rerank | 0.7608 | 0.9333 | 0.8081 | 0.9596 | 4 |

The concern benchmark now separates held-out `queries` from retrieval
`aliases`, so concern rows are not directly injected into the index. The
remaining concern misses are useful adapt-loop targets: defective purchased
goods, limitation periods for defects, pledge expiration, partnership exit
rights, and finder duties.

## Randomized BGB Stress Results

The curated concern suite is useful, but too friendly. A new stress harness,
`examples/bgb_browser_search/run_bgb_stress_eval.py`, samples article blocks and
asks multiple cheap OpenRouter models to generate fresh direct, concern, and
adversarial questions. The generated question text is not trusted as a label:
the gold target is the sampled article block.

Mixed stress run:

- sampled article blocks: 32
- generator models: Qwen Turbo, Mistral Nemo, Mistral Small 3.2 24B, Gemma 3
  27B, Grok 4 Fast
- generated/evaluated rows: 953
- styles: direct, concern, adversarial
- languages: German and English

| Method | article hit@1 | article hit@10 | MRR |
| --- | ---: | ---: | ---: |
| raw BM25 | 0.1427 | 0.2445 | 0.1715 |
| Refmark BM25 | 0.3578 | 0.5824 | 0.4304 |
| Refmark rerank | 0.3526 | 0.5887 | 0.4277 |

Style split for Refmark BM25:

| Style | rows | hit@1 | hit@10 |
| --- | ---: | ---: | ---: |
| direct | 318 | 0.6667 | 0.8962 |
| concern | 317 | 0.2145 | 0.4921 |
| adversarial | 318 | 0.1918 | 0.3585 |

Concern-heavy stress run:

- sampled article blocks: 40
- generator models: Qwen Turbo, Mistral Nemo, Mistral Small 3.2 24B, Gemma 3
  27B
- generated/evaluated concern rows: 873

| Method | article hit@1 | article hit@10 | MRR |
| --- | ---: | ---: | ---: |
| raw BM25 | 0.1924 | 0.3116 | 0.2277 |
| Refmark BM25 | 0.4719 | 0.7320 | 0.5533 |
| Refmark rerank | 0.4708 | 0.7308 | 0.5525 |

This is a healthier stress result than the curated `0.95` concern hit@10. It
shows a clear Refmark gain, but also shows that randomized layperson questions
still break exact lexical navigation. The hardest repeated blocks in the
concern-heavy run included `bgb:S_2316`, `bgb:S_308`, `bgb:S_2268`, `bgb:S_14`,
`bgb:S_536b`, `bgb:S_904`, `bgb:S_955`, and `bgb:S_965`.

The first attempt to compare Qwen3 embeddings on this stress suite timed out
locally while loading the existing large embedding cache. That is an
implementation bottleneck, not a negative retrieval result. The next comparison
should use a compact vector store or memory-mapped cache before making a stress
claim about embeddings/hybrids.

### 200k-Token Stress Cycles

We then moved from tiny sampled stress slices to representative 200k-token
slices, roughly half of the BGB source corpus per run. The generator models
created only concern and adversarial questions in German and English. Qdrant
ran inside WSL as build/eval infrastructure for Qwen3 embeddings; this does not
change the intended no-vector-runtime deployment path.

| Cycle | Generator | Slice tokens | Blocks | Questions | Refmark BM25 hit@10 | Qwen3 hit@10 | Best hybrid hit@10 | Best hybrid hit@50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Qwen Turbo | 200,136 | 1,002 | 4,008 | 0.6914 | 0.9474 | 0.9506 | 0.9943 |
| 2 | Mistral Small 3.2 24B | 200,020 | 1,038 | 3,916 | 0.6586 | 0.9614 | 0.9620 | 0.9936 |
| 3 | Gemma 3 27B | 200,070 | 1,022 | 3,874 | 0.3446 | 0.9174 | 0.9177 | 0.9845 |

Cycle 3 is especially useful as a breaker: Gemma generated much harder or less
lexically aligned questions, dropping Refmark BM25 to `0.3446` hit@10, while
Qwen3 embeddings still recovered `0.9174` hit@10. Across all three cycles, the
embedding result is the current upper-bound teacher signal. The next question is
how much of that gain can be distilled into static generated views, aliases,
hard negatives, and a tiny reranker so runtime can remain BM25/model-only.

### First Static Adaptation Attempt

We tested the simplest adaptation idea on the hardest Gemma 200k cycle:

1. Split generated questions within each article block.
2. Inject half as static aliases.
3. Evaluate BM25 on the held-out half.

This deliberately avoids exact-query leakage while testing whether generated
section questions generalize as retrieval metadata.

Result: naive full-question alias injection hurt.

| Static path | held-out hit@10 | held-out hit@50 | MRR |
| --- | ---: | ---: | ---: |
| article baseline | 0.3750 | 0.5633 | 0.2219 |
| aliases mixed into article view | 0.3193 | 0.5184 | 0.1844 |
| alias-only side index | 0.0768 | 0.1499 | 0.0503 |
| source + alias side-index hybrid | 0.3723 | 0.5633 | 0.1468 |

Interpretation: adding raw generated questions increases noise and BM25 length
normalization penalties. The adaptation signal is real, but it should be
compressed and weighted, not pasted wholesale into the source index. Better next
adapt steps:

- teacher-compressed aliases: short intent phrases rather than full questions;
- fielded/static multi-index retrieval: source BM25, generated-view BM25, alias
  BM25 scored separately;
- hard negatives from repeated wrong-top refs;
- train a reranker on BM25 candidates using embedding/hybrid ranks as soft
  labels, but only after BM25 candidate recall is high enough;
- for low-recall slices like Gemma cycle 3, first improve candidate generation
  because a reranker cannot recover refs absent from top-k.

### Heatmap and Compressed Intent Signatures

To make the adapt loop less anecdotal, `report_bgb_retrieval_heatmap.py` now
aggregates multiple stress reports into article-level retrieval diagnostics:
overall hit@k, language/style/model splits, hard articles, wrong-top confusion
pairs, and sample misses. Across the three 200k-token cycles, baseline static
Refmark BM25 had:

| Scope | Rows | hit@1 | hit@10 | hit@50 | hit@1000 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3-cycle stress heatmap | 11,798 | 0.3193 | 0.5920 | 0.7434 | 0.9517 | 0.4123 |

The top-1000 number matters: the correct article is often somewhere in the
candidate pool, but not high enough for a browser-quality instant navigation
experience. That makes the heatmap a useful target selector for adaptation:
work first on article blocks with repeated misses, repeated wrong-top neighbors,
or weak candidate-depth recall.

`evaluate_bgb_selective_jump.py` now adds a UI-facing layer on top of that
heatmap: when should the browser jump directly, and when should it show
candidates? On all `11,798` rows, base Refmark BM25 had `0.8551` candidate
recall at top 200 and `0.9517` at top 1000. On the matching held-out split
(`5,810` rows), deterministic signatures improved hit@10 from `0.5912` to
`0.6250`; deterministic plus top-300 confusion signatures reached `0.6289`.
Top-200 candidate recall moved from `0.8594` to `0.8929`.

Simple confidence features can produce high-precision direct jumps, but only at
modest coverage. With a held-out top1/top2 margin threshold:

| Index | hit@10 | top200 recall | jump coverage at 90% precision | fallback hit@10 |
| --- | ---: | ---: | ---: | ---: |
| base source + views | 0.5912 | 0.8594 | 0.0880 | 0.5575 |
| deterministic intent signatures | 0.6250 | 0.8910 | 0.0417 | 0.6121 |
| deterministic + confusion signatures | 0.6289 | 0.8929 | 0.0420 | 0.6157 |

This is valuable product behavior even though it does not solve auto-jump by
itself: the search UI can jump only when calibrated confidence is high and fall
back to a stronger candidate list or rescue path otherwise. It also shows that
static adaptation improves the fallback/candidate-list mode more than the
direct-jump mode.

The next static adaptation compressed held-out-safe training questions into
short local intent signatures. This is still local and deterministic: no LLM
was used for this pass. It extracts compact phrases from the train half of the
generated questions and injects those phrases as weighted article metadata.

On the held-out half of all three 200k stress cycles:

| Static path | held-out hit@1 | held-out hit@10 | held-out hit@50 | MRR |
| --- | ---: | ---: | ---: | ---: |
| baseline source + generated views | 0.3196 | 0.5912 | 0.7435 | 0.4104 |
| compressed intent signatures mixed in | 0.3392 | 0.6251 | 0.7849 | 0.4350 |
| signature-only side index | 0.1003 | 0.2676 | 0.4294 | 0.1556 |
| source/signature RRF | 0.2084 | 0.5719 | 0.7556 | 0.3135 |

This is the first static metadata adaptation that helped consistently across
top-k. It improved hit@10 by `+0.0339` and hit@50 by `+0.0414` without using a
runtime vector database. It also helped both languages and both hard styles:
German hit@10 moved from `0.5956` to `0.6344`, English from `0.5869` to
`0.6159`, adversarial from `0.6053` to `0.6408`, and concern from `0.5772` to
`0.6095`.

This suggests the next LLM-backed adaptation should not ask for full synthetic
questions as index text. It should ask for compact multilingual intent
signatures, synonyms, layperson formulations, and hard-negative distinctions,
then cache them per article/ref range.

The first cached LLM signature pass targeted the 40 hardest heatmap articles
instead of the whole corpus. It used the same held-out split as the deterministic
signature experiment and generated compact article signatures from source text
plus held-out-safe train questions.

| Target | Method | hit@1 | hit@10 | hit@50 | MRR |
| --- | --- | ---: | ---: | ---: | ---: |
| all held-out rows | baseline source + views | 0.3196 | 0.5912 | 0.7435 | 0.4104 |
| all held-out rows | deterministic signatures | 0.3392 | 0.6251 | 0.7849 | 0.4350 |
| all held-out rows | Qwen hard-40 LLM signatures | 0.3127 | 0.5928 | 0.7444 | 0.4059 |
| all held-out rows | deterministic + Qwen hard-40 | 0.3367 | 0.6243 | 0.7843 | 0.4334 |
| all held-out rows | Gemma hard-40 LLM signatures | 0.3189 | 0.5941 | 0.7454 | 0.4110 |
| all held-out rows | deterministic + Gemma hard-40 | 0.3391 | 0.6253 | 0.7866 | 0.4353 |

On the selected hard-40 eval subset, the shape is much clearer:

| Target | Method | hit@1 | hit@10 | hit@50 | MRR |
| --- | --- | ---: | ---: | ---: | ---: |
| hard-40 held-out rows | baseline source + views | 0.0000 | 0.0000 | 0.3434 | 0.0162 |
| hard-40 held-out rows | deterministic signatures | 0.0202 | 0.1616 | 0.4040 | 0.0675 |
| hard-40 held-out rows | Qwen hard-40 LLM signatures | 0.0808 | 0.2828 | 0.5455 | 0.1486 |
| hard-40 held-out rows | Gemma hard-40 LLM signatures | 0.0707 | 0.3030 | 0.5556 | 0.1518 |

So LLM signatures are useful as targeted heatmap repair. They are not yet a
drop-in replacement for deterministic corpus-wide signatures: unrestricted
mixed metadata can attract unrelated queries and slightly depress top-1. A
side-index/RRF fusion with baseline BM25 was too conservative to recover most
hard rows. The next gating shape should probably be confidence/heatmap-aware:
use LLM signatures as a rescue/candidate-expansion path when baseline retrieval
has weak lexical confidence or repeated known confusion, not as a uniformly
weighted side index.

We then made that gating hypothesis explicit with
`evaluate_bgb_signature_gating.py`. The tested gate uses deterministic
signatures as the default index, then switches to the LLM-repaired index when a
signature-only side index exceeds a score threshold.

The tradeoff was clear:

| Gate threshold | Global switch rate | Global hit@10 delta vs deterministic | Hard-40 hit@10 delta vs deterministic |
| ---: | ---: | ---: | ---: |
| 0.5 | 0.9628 | -0.0279 | +0.1414 |
| 1.0 | 0.7310 | -0.0199 | +0.1313 |
| 1.5 | 0.5005 | -0.0139 | +0.1313 |
| 2.0 | 0.3327 | -0.0108 | +0.1313 |
| 3.0 | 0.0905 | -0.0012 | +0.1111 |
| 4.0 | 0.0134 | +0.0000 | +0.0404 |
| 5.0 | 0.0017 | +0.0000 | +0.0101 |

This validates the product loop more than the exact gate: the heatmap identifies
real weak zones, LLM signatures repair those zones, and the evaluation harness
shows when the repair leaks noise into global retrieval. The current side-score
gate is too blunt. A better next trigger should combine multiple signals:
baseline confidence/margin, whether top deep candidates include known hard
articles, wrong-top confusion pairs, and side-index confidence.

Adding a second condition, "only switch when deterministic BM25 top score is
below a ceiling", did not improve the tradeoff in this run. The selected hard
queries often still had confident-looking lexical scores, just for the wrong
articles. That is another useful data smell: score confidence alone is not
enough; we need confusion-aware or teacher-aware gating.

### Static Fielding and Tiny Reranking

The next two no-vector-runtime checks were deliberately run on the same hard
Gemma held-out split.

Fielded static retrieval separated source text, generated summaries, generated
questions, generated keywords, and held-out-safe train aliases into separate
BM25 indexes, then fused them with weighted reciprocal-rank fusion. This did
not help:

| Static path | held-out hit@10 | held-out hit@50 | MRR |
| --- | ---: | ---: | ---: |
| baseline source + generated views | 0.3750 | 0.5633 | 0.2219 |
| fielded original RRF | 0.2803 | 0.4762 | 0.1378 |
| fielded + train aliases RRF | 0.2716 | 0.4800 | 0.1302 |

So the problem is not only BM25 length normalization. The generated question
field contains useful signal, but simple rank fusion and raw train aliases add
noise.

We then trained a small article-level pair reranker over BM25 candidates. This
does help, but only inside the BM25 candidate pool:

| Candidate depth | BM25 candidate recall | Best blended hit@10 | Best blended hit@50 | Artifact |
| ---: | ---: | ---: | ---: | ---: |
| 80 | 0.6272 | 0.4080 | 0.6039 | 7.27 MB |
| 200 | 0.7300 | 0.4161 | 0.6136 | 7.27 MB |

The best k=200 run used a 1.72M-parameter resolver and measured about
`6.4 ms/query` to rerank 200 candidates on this machine. Training itself was
not the limiting cost; candidate construction/evaluation dominated wall time.

Current conclusion: a tiny static reranker is useful as a lightweight
distillation layer, but it cannot close the gap to Qwen3 embeddings unless
candidate generation improves. On the same held-out split, plain BM25 recall is
only `0.73` at top 200, `0.84` at top 500, and `0.91` at top 1000, while Qwen3
embedding retrieval reached `0.9174` hit@10. The next serious improvement should
therefore focus on candidate generation: compressed teacher views, better
static multilingual terms, or a two-stage local candidate generator trained
from embedding-teacher positives and hard negatives.

We also tried that teacher idea directly as a text-only reranker. Cached Qwen3
query/article vectors were used only during training to provide soft targets;
runtime still used base Refmark BM25 candidates plus a small query/article text
pair model. On the combined 3-cycle held-out set with 80 candidates, base
Refmark BM25 reached `0.5678` hit@10 and `0.7263` hit@50. The pure
teacher-shaped reranker reached `0.5816` hit@10 and `0.7475` hit@50. The best
BM25/model blend reached `0.5997` hit@10 and `0.7509` hit@50, with a 14.74 MB
artifact and about `4.53 ms/query` over 80 candidates. This is a modest
positive, not a distilled replacement for embeddings.

One invalid run is worth recording as a test hygiene warning: using the
prebuilt deterministic-signature index with a different train/eval split gave a
baseline of `0.8129` hit@10, which is likely split leakage from adapted
metadata. Adapted indexes need to carry and match their exact stress reports
and split seeds.

### Direct Candidate Generation and Embedding Distillation

We then tested the question Boris raised explicitly: can the strong embedding
retriever be used as an offline teacher so a tiny local model can later produce
good candidates without embeddings or a vector database at runtime?

Four direct candidate-generator variants were tried:

1. **Label classifier**: query -> article class.
2. **Gold article-vector distillation**: query -> vector close to the Qwen3
   article vector for the gold ref.
3. **Query-embedding distillation**: query -> vector close to the cached Qwen3
   query embedding, then local dot product against Qwen3 article vectors.
4. **Cached query embedding classifier**: cached Qwen3 query embedding ->
   article class. This still requires query embeddings at runtime, so it is not
   a browser-offline replacement; it tests whether the embedding space is
   directly learnable as an address space.

Results:

| Local generator | Eval split | Held-out hit@10 | Held-out hit@50 | Runtime note |
| --- | --- | ---: | ---: | --- |
| direct text label classifier | Gemma 200k | 0.0395 | 0.1012 | 15.8 MB artifact, ~0.24 ms/query |
| text -> gold article-vector distill | Gemma 200k | 0.1661 | 0.3653 | 38.2 MB artifact, ~1.42 ms/query |
| text -> gold article-vector distill | 3-cycle combined | 0.2816 | 0.5207 | 38.2 MB artifact, ~1.45 ms/query |
| text -> Qwen3 query-embedding distill | 3-cycle combined | 0.2771 | 0.5270 | 38.2 MB artifact, ~1.45 ms/query |
| cached Qwen3 query embedding -> article classifier | 3-cycle combined | 0.9377 | 0.9819 | 20.4 MB artifact, ~0.63 ms/query after embedding |
| BM25 article baseline | Gemma 200k | 0.3712 | 0.5714 | static lexical |
| BM25 article baseline | 3-cycle combined | 0.5974 | 0.7491 | static lexical, same split as embedding-classifier run |

This is a useful negative result. Distillation is not magic yet. More generated
supervision clearly helps (`0.1661 -> 0.2816` hit@10), and vector targets are
far better than an opaque article-label classifier, but the tiny bag/MLP query
encoder does not approximate Qwen3 embedding quality. Training against cached
Qwen3 query embeddings also did not beat training against gold article vectors.

The cached query-embedding classifier changes the picture, but also changes the
runtime assumptions. It does not replace embeddings; it adds a small supervised
address head on top of an embedding model. That head reached `0.6910` hit@1,
`0.9377` hit@10, and `0.9819` hit@50 on the combined held-out stress set, with
balanced German/English and concern/adversarial performance. This shows that
the embedding space is highly compatible with Refmark supervision. It also
suggests a practical deployment mode for systems that already compute query
embeddings: add a corpus-local `embedding -> ref/range` head for fast
candidate generation, diagnostics, or hybrid reranking.

Current interpretation:

- The embedding teacher remains valuable for evaluation and upper-bound
  candidate generation.
- The no-runtime-embedding local text student needs either far more query
  coverage, a stronger/pretrained multilingual query encoder, better compressed
  teacher features, or a different objective.
- If query embeddings are available at runtime, a small corpus-local address
  head is already strong enough to be interesting.
- A tiny local model can currently act as a supplementary signal, not as a
  replacement for embeddings in browser-offline mode.
- This strengthens the product framing: Refmark makes the failure measurable
  and prevents us from overclaiming the training story.

## Natural Query Judge Results

We also ran 12 fresh natural user concerns in German and English. These were not
from the deterministic test set. A DeepSeek-family judge scored whether each
method produced useful evidence.

DeepSeek v4-pro was unstable via OpenRouter during this run, so the combined
report uses valid v4-pro judgments where available and falls back to
DeepSeek Chat v3.1 for provider failures/timeouts.

Judge summary:

| Method | Supported | Avg usefulness | Avg citation quality | Wins |
| --- | ---: | ---: | ---: | ---: |
| raw BM25 | 0.3333 | 0.2083 | 0.2000 | 1 |
| Refmark BM25 | 0.5000 | 0.4792 | 0.4917 | 2 |
| raw Qwen3 embedding | 0.8333 | 0.6500 | 0.6583 | 4 |
| Refmark Qwen3 embedding | 0.9167 | 0.7167 | 0.7500 | 2 |
| Refmark hybrid w0.05 | 0.9167 | 0.8000 | 0.8333 | 3 |

Interpretation:

- Exact-anchor scoring favors pure Refmark-enriched Qwen3 embeddings.
- Natural query judging favors the low-BM25 hybrid for usefulness and citation
  quality.
- This suggests two separate modes:
  - exact localization: Refmark-enriched embeddings;
  - answer-oriented navigation: embedding-first with a small lexical/Refmark
    signal.

The judge surfaced a valuable failure: for a German property-damage query, all
methods missed the central `§ 823 I BGB` liability basis and mostly retrieved
`§ 249` restoration/compensation rules. That is a real retrieval gap, not just a
metric artifact.

## What Helped

- Bilingual generated views helped cross-language retrieval substantially.
- Qwen Turbo was the strongest practical view/question generator in the tested
  slice.
- Qwen3 Embedding 8B was the strongest retrieval signal tested so far.
- Refmark-enriched embedding text improved over raw text embeddings on full BGB.
- Refmark BM25 is strong enough to be useful as a browser/local fallback.
- Deterministic compressed intent signatures improved held-out static BM25 on
  the 3-cycle stress suite.
- Confusion-conditioned signatures helped selected hard-confusion rows when
  selected from a wide enough heatmap.
- Curated concern aliases improved motivation-query localization when query
  text was held out from the retrieval view.
- Low-BM25 hybrid helps on natural concerns when exact-anchor scoring is not the
  whole story.
- Split-aware selective-jump metrics translated retrieval quality into a UI
  policy: direct jump when calibrated, otherwise show candidates.
- Stable regions made it cheap to compare methods and inspect concrete failures.

## What Did Not Help Yet

- The simple default rank fusion was too BM25-heavy and often degraded embedding
  results.
- The tiny trained resolver only gave a small gain as a reranker and is not yet
  strong enough to be a product claim.
- Confusion signatures did not beat deterministic signatures as a global
  default; they need selection and gating.
- Adapted-index split/provenance mismatch can create impressive but invalid
  numbers. Adapted indexes must record and match source reports plus split seed.
- Article-level metrics are not meaningful for the BGB single-document setup.
- DeepSeek v4-pro was too unstable through OpenRouter for unattended judging.
- Deterministic generated-question benchmarks can overstate confidence if not
  complemented by natural user queries and judge checks.

## Tiny Resolver / Training State

The full bilingual BGB resolver experiment produced a small model:

- About 1.31M parameters.
- About 5.43 MB artifact.
- About 21.9 minutes training time on the local CPU machine.
- Inference in the low millisecond range in the current evaluation loop.

Full BGB resolver result:

- BM25 baseline on resolver eval split: hit@1 0.6562, hit@10 0.9152, MRR 0.7457.
- Best blend: hit@1 0.6652, hit@10 0.9018, MRR 0.7530.

This is encouraging for a local reranker, but not yet enough to claim that the
trained tiny model is the main value. The current value is the evaluation and
navigation layer; training remains exploratory.

## Training Evidence Interpretation

Training should be evaluated as a second-stage evidence resolver over a
high-recall candidate set. The current evidence is mixed:

- On BGB, the 1.31M-parameter multilingual resolver gave only a small blended
  MRR gain over the BM25 candidate baseline.
- On public documentation, small resolver runs did improve rank-1 localization
  on some corpora while keeping artifacts around 5.43 MB.
- On the local OSHA Markdown mixed-target pressure test, the simple learned
  reranker improved top-10 recall for local-view BM25 but hurt rank-1 precision
  and MRR.
- On the BGB 3-cycle held-out split, an embedding-teacher text reranker over
  the adapted deterministic+confusion index gave a small blended gain:
  hit@10 `0.6293 -> 0.6389`, hit@50 `0.7886 -> 0.7928`, MRR
  `0.4362 -> 0.4405`, with a `14.65 MB` artifact and about `4.62 ms/query`.
- A seed-mismatched reranker/adapted-index run produced an invalid `0.8205`
  hit@10, which is now treated as a provenance hygiene warning rather than a
  result.

That last negative result is valuable. Refmark makes it obvious when training
mostly moves the right evidence somewhere into top-k while making the first
answer worse. The acceptance rule should be: keep a trained resolver only when
it improves held-out evidence localization without violating coverage,
undercitation, or overcitation gates.

## Mixed Target Evidence

The local OSHA Markdown run packages a retained legal corpus as generated
Markdown and evaluates 120 mixed targets over a 250k-token budget:

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

This is not a public quality claim because the questions are local keyword
questions. It is a useful harness result: single, range, and distributed support
can be measured in one run, and over/under-citation becomes visible immediately.

## Training Questions To Investigate

Useful next questions for external review:

- Can the tiny resolver learn from embedding-neighborhood features, not just
  lexical/refmark features?
- Should we train separate coarse and fine heads?
  - coarse: document/article/section range;
  - fine: exact region or small neighboring range.
- Can hard-negative mining use Refmark failures directly?
  - same title, wrong paragraph;
  - same statute family, wrong subsection;
  - correct topic, missing adjacent provision.
- Can natural-query judge outcomes become training labels?
  - answer-supported;
  - underbroad/overbroad;
  - useful but non-exact anchor.
- Is the best browser-deployable stack:
  - generated Refmark views + static BM25;
  - generated views + quantized/local embeddings;
  - or generated views + tiny learned reranker?
- What is the smallest useful model that improves over static search on a
  new corpus after cheap corpus-specific training?

## Product Direction

The most concrete user-facing feature is:

> Convert a documentation corpus into a searchable, browser-ready asset with
> stable jump targets and measurable retrieval quality.

A practical stack today:

- Generate bilingual/multilingual retrieval views per Refmark region.
- Build a portable static index for browser/local search.
- Optionally precompute embeddings for high-quality semantic search.
- Use Refmark ids/ranges for highlighting, citation, drift checks, and
  evaluation.

The training story should stay experimental until it reliably improves over the
Qwen3 embedding baseline or enables an offline/browser deployment that embeddings
cannot satisfy cheaply.
