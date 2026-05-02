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

### Active Benchmarking Track

Core question:

> Does Refmark-based evaluation let us build retrieval systems that perform
> better at finding relevant information?

This should be tested as an optimization loop, not only as a reporting format.
The experiment should compare a plain baseline against systems adapted using
only training/dev evidence diagnostics produced from a refmarked corpus.

Minimum design:

1. Pick an external QA/retrieval benchmark with a source corpus and held-out
   questions.
2. Convert the corpus into a `CorpusMap` with stable refs/ranges.
3. Map benchmark gold passages, documents, or support facts to refs/ranges.
4. Split questions into train/dev/test or use the benchmark's existing split.
5. Run baseline retrievers without Refmark adaptation.
6. Use Refmark metrics on train/dev only to identify weak regions, query
   magnets, stale/ambiguous labels, missing metadata, and confusion pairs.
7. Apply a bounded adaptation policy: metadata/doc2query aliases, source
   hierarchy, query-magnet exclusion, context expansion, hybrid weighting, or
   reranker training.
8. Evaluate once on held-out questions using the same evidence metrics and, when
   useful, answer-level metrics.

The claim is valid only if held-out evidence retrieval improves, for example:

- higher gold-document/ref hit@k;
- higher evidence coverage for ranges or multi-hop support;
- better MRR/top-1 without losing hit@k;
- lower context cost at the same evidence recall;
- fewer wrong-top confusions or query-magnet hits;
- improved downstream grounded answer quality.

Do not claim that Refmark improves QA just because it makes failures visible.
Visibility is the mechanism. The research question is whether that visibility
can guide changes that improve held-out retrieval or answer grounding.

Preliminary no-training BEIR checks, run on the Ubuntu experiment host, are
mixed in the useful way:

| Corpus | Scope | Baseline | Best Refmark Variant | Result |
| --- | --- | --- | --- | --- |
| SciFact | 5,183 docs, 300 queries, 14,498 regions | doc BM25 hit@10 0.803 | enriched region->doc max hit@10 0.790 | regionization alone did not beat document BM25 |
| NFCorpus | 3,633 docs, 323 queries, 10,995 regions | doc BM25 hit@10 0.690 | enriched region->doc max hit@10 0.669 | regionization alone did not beat document BM25 |
| ArguAna | 8,674 docs, first 100 queries, 19,157 regions | doc BM25 hit@10 0.690 | enriched region->doc max hit@10 0.810 | region evidence improved top-10 on this slice |

Early interpretation:

- naive summing of child-region scores into a parent document is harmful;
- max-region parent aggregation is the first sane default;
- source-only enrichment has small or corpus-dependent effects;
- Refmark does not automatically improve document-level retrieval, but it makes
  aggregation and adaptation choices measurable;
- larger datasets such as FiQA need a faster/vectorized benchmark harness or
  stronger hardware because the current pure-Python BM25 loop becomes the
  bottleneck before training or embedding costs matter.

See `docs/REFMARK_BENCHMARKING_NOTES.md` for the current external benchmark
summary, including vectorized lexical runs and MiniLM/BGE embedding baselines.

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

## 8. Evidence Lifecycle Under Corpus Change

Question: can stable refs/ranges reduce silent corruption in maintained
retrieval, evaluation, citation, and training labels after corpus revisions?

This is currently the most Refmark-specific benchmark direction. Static
retrieval benchmarks can make Refmark look like named chunking; versioned
evidence benchmarks test the addressability claim directly.

See `docs/EVIDENCE_LIFECYCLE_BENCHMARK.md` and
`examples/evidence_lifecycle_benchmark/`.
