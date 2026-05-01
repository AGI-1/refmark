# Refmark Research Angles

This note keeps research directions distinct from the current product surface.
The common theme is that Refmark creates a stable evidence address space, then
different systems can be evaluated, adapted, or trained against that same
coordinate system.

## 1. Refmark Eval vs Classical RAG Eval

Question: does ref-based evidence evaluation give more actionable feedback than
answer-level judging alone?

Compare:

- answer correctness judged by an LLM
- generated-answer similarity metrics
- ordinary retrieval hit@k over chunks
- Refmark evidence-region hit@k, coverage, precision, and stale-ref checks

Expected value:

- separates answer generation failures from evidence retrieval failures
- exposes overcitation, undercitation, and neighbor-hit behavior
- makes corpus drift and stale evaluation examples measurable
- creates heatmaps for weak regions instead of only aggregate scores

## 2. Refmark for External QA Benchmarks

Question: can Refmark improve results on existing QA benchmarks without training
on the benchmark questions themselves?

Target benchmark types:

- documentation QA
- legal/statutory QA
- technical manual QA
- scientific or guideline QA
- finance/regulatory QA
- enterprise/wiki-style QA when a source corpus is available

Method:

1. Refmark the benchmark source corpus.
2. Map answer support to refs/ranges, using provided spans when available.
3. Adapt/train retrieval from corpus-derived signals only.
4. Keep benchmark questions as held-out evaluation.
5. Compare BM25, embeddings, hybrid retrieval, refmark metadata adaptation,
   trained bi-encoders, and rerankers.

Key metric: did the system retrieve the source evidence needed to answer the
benchmark question, not only did it generate a plausible answer?

## 3. Corpus Classification and Coarse Navigation

Question: can small local models learn to route questions to corpus areas,
articles, sections, or ref ranges?

Research variants:

- area classifier over coarse region clusters
- article classifier over known corpus units
- hierarchical router: area -> article -> region
- bi-encoder candidate generator plus tiny reranker
- compact student models trained from stronger embedding teachers

Current BGB evidence suggests:

- opaque ref-string generation is weak
- classification and contrastive retrieval are much stronger
- hard masks are risky; soft top-k hierarchy is safer
- bi-encoder plus bounded reranker is a promising local navigation stack

## 4. Adaptation Loops and Data Smells

Question: can Refmark make retrieval improvement inspectable enough to support
safe adaptation loops?

Adaptation types:

- generated retrieval-view metadata
- doc2query aliases in shadow metadata
- hard-negative/confusion signatures
- region merge/split suggestions
- query-magnet exclusions
- stale example refresh after corpus edits

Data smells to track:

- duplicate or near-duplicate evidence
- contradictory regions
- release-note or changelog query magnets
- overly broad sections
- missing question coverage
- metadata that improves one region but hurts neighbors

The key requirement is a before/after run artifact with the same corpus,
eval-suite, and settings fingerprints unless the changed component is explicitly
the subject of the experiment.

## 5. Refmark as Training/Eval Label Infrastructure

Question: is the main value of Refmark for training the labels themselves, or
the lifecycle around labels?

Hypothesis:

Refmark is valuable less because it invents a new model objective and more
because it gives every training example a stable evidence target that can be
validated, diffed, refreshed, and scored after corpus changes.

Important tests:

- source-only train, external-question eval
- generated-question train, manual-question eval
- cross-generator train/eval splits
- corpus revision drift and stale-label detection
- whether region-level labels outperform ordinary chunk ids

## 6. Browser/Edge Retrieval

Question: can a refmarked corpus be shipped as a local evidence navigator with
minimal infrastructure?

Variants:

- static BM25 index
- compact metadata-adapted index
- small local bi-encoder candidate cache
- tiny reranker or classifier
- section/article jump UI with ref-highlighted evidence

The research risk is model/runtime size. The product value is clearer when the
system can run as a static documentation asset or in-browser search surface.

## 7. Human Review and Corpus CI

Question: can Refmark make corpus quality and retrieval quality reviewable in
CI-like workflows?

Useful outputs:

- evidence heatmaps
- stale-ref reports
- weak-region queues
- confusion pairs
- adaptation proposals
- held-out regression reports
- human approval queues for metadata changes

This is likely the strongest long-term framing: a refmarked corpus becomes a
regression-testable evidence space.
