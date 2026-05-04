# Evidence Summary

This document summarizes the evidence that currently supports the public
Refmark framing. It is intentionally conservative: Refmark is presented as an
evidence-addressing and evaluation substrate, not as a universal search,
training, or coding-agent win.

## Main Supported Claim

Refmark turns a corpus into a stable, addressable evidence space that can be
used to evaluate and maintain retrieval/citation pipelines.

The strongest current claim is:

> A refmarked corpus can act like a regression test suite for retrieval.

This means:

- retrieval outputs can be scored by exact source refs/ranges;
- stale labels can be detected after corpus changes;
- hard refs, confusions, query-style gaps, query magnets, and citation breadth
  issues become inspectable data-smell reports;
- adaptation plans can be generated as review-required actions instead of
  silent automatic mutations.

## Reproducible Product Loop

The smallest public loop is:

```bash
python examples/docs_navigation_pipeline/run.py
```

That example runs:

```text
map -> toc -> build-index -> export-browser-index
    -> eval-index -> data-smell report -> adaptation plan
    -> sample navigation query
```

It writes ignored artifacts under
`examples/docs_navigation_pipeline/output/`, including `smells.json` and
`adaptation_plan.json`.

Equivalent CLI shape:

```bash
python -m refmark.cli eval-index docs.index.json eval_questions.jsonl \
  --manifest corpus.refmark.jsonl \
  --smell-report-output smells.json \
  --adapt-plan-output adaptation_plan.json \
  -o eval.json

python -m refmark.cli compare-index docs.index.json eval_questions.jsonl \
  --manifest corpus.refmark.jsonl \
  --strategies flat,hierarchical,rerank \
  -o compare_index.json
```

For the full compact command path, see
[Evidence CI Quickstart](QUICKSTART_EVIDENCE_CI.md).

## Lifecycle Evidence

Five Git-backed documentation corpora were evaluated across natural release
revisions:

- FastAPI
- Django
- Flask
- HTTPX
- Kubernetes docs

Across 15 revision comparisons:

| Metric | Average |
| --- | ---: |
| Refmark auto-preserved labels | 45.0% |
| Refmark review-needed labels | 22.9% |
| Refmark stale labels | 32.1% |
| Naive silent-wrong labels | 30.1% |
| Workload reduction vs full audit | 37.7% |

Most dramatic observed case:

| Corpus revision | Naive silent wrong | Refmark stale | Refmark review |
| --- | ---: | ---: | ---: |
| FastAPI `0.100.0 -> 0.115.0` | 61.5% | 59.4% | 34.0% |

Interpretation:

- ordinary same-path/same-ordinal chunk ids can silently point to wrong
  evidence after documentation changes;
- Refmark exposes whether labels are unchanged, review-needed, or stale;
- high stale rates are not necessarily bad when a corpus was heavily rewritten;
  they mean old supervision should not be trusted automatically.

Limit:

- the naive baseline is simple. Stronger registry or semantic migration systems
  may do better. The evidence supports the need for lifecycle validation, not a
  universal win over every possible chunking scheme.

The stronger five-corpus lifecycle benchmark now includes competent baselines:
chunk id plus content hash, qrels/source-file hash, chunk hash plus quote
selector, and a Refmark layered selector. In that aggregate, the exact Refmark
migration layer was more conservative than the quote-selector baseline, but the
layered Refmark selector borrowed the useful quote-selector signal while keeping
unsafe cases explicit. It preserved slightly more labels than the quote-selector
baseline (`26,520` vs `26,389`) with no observed silent drift under the current
migration oracle, at the cost of a slightly larger explicit review queue
(`48.6%` vs `48.1%`).

This oracle caveat is central: the aggregate benchmark still uses Refmark's
exact/fuzzy migration heuristic to decide whether old evidence remains valid.
The benchmark is therefore strong evidence about lifecycle instrumentation and
method tradeoffs, not final independent proof of semantic correctness.

An LLM-adjudicated sample of 200 disagreement/review cards further suggests
that Refmark's review queue is meaningful rather than just conservative noise.
Most review-needed/fuzzy cases were judged as valid rewritten or moved
evidence, while quote-selector silent-drift candidates included stale and
split-support cases. This points to the next improvement target: increase safe
auto-preservation from the review queue without reintroducing silent drift.

A smaller Codex-filled adjudication pass over the top 60 review cards was used
as a calibration aid, not as independent human evidence. It found 46/60
preservable labels (`valid_unchanged`, `valid_moved`, or `valid_rewritten`),
7 stale labels, 3 split-support labels, and 4 ambiguous labels. The split was
especially useful for rule design: 15/16 Refmark fuzzy/layered review cases
looked safely preservable, while 44 quote-selector silent-drift candidates
contained 6 stale, 3 split-support, and 4 ambiguous cases. That supports the
current vNext direction: borrow quote-selector recovery, but gate it through
Refmark lifecycle state and keep split evidence as an explicit range-repair
state.

See [Evidence Lifecycle Benchmark](EVIDENCE_LIFECYCLE_BENCHMARK.md).

## Retrieval/Evaluation Evidence

The core retrieval evidence is structural rather than a single leaderboard
number: Refmark lets any retriever be evaluated by source-region recovery.

Current implemented diagnostics include:

- hit@1 / hit@k;
- MRR;
- gold coverage;
- region precision;
- stale examples;
- hard-ref heatmaps;
- wrong-top confusion pairs;
- query-style gaps;
- gold-mode/range gaps;
- low-confidence top hits;
- overcitation and undercitation.

The FastAPI pipeline work showed that a documentation corpus can be turned into
an inspectable heatmap where weak areas become concrete action candidates. The
small public docs-navigation example now demonstrates the same mechanics in a
CI-sized form.

Limit:

- the public small example is intentionally easy. It proves integration shape,
  not retrieval superiority.
- larger retrieval experiments are useful research context but should not be
  presented as a broad benchmark claim until rerun with held-out/manual query
  sets and stronger baselines.

See [Evidence Retrieval Pipeline](EVIDENCE_RETRIEVAL_PIPELINE.md) and
[Data Smells](DATA_SMELLS.md).

## Data-Smell And Adaptation Evidence

`build_data_smell_report(...)` and `refmark cli eval-index
--smell-report-output` produce `refmark.data_smells.v1` reports. `adapt-plan`
converts those reports into `refmark.adaptation_plan.v1` actions.

Current smell/action categories:

| Smell | Typical adaptation action |
| --- | --- |
| `stale_label` | review or refresh gold refs/source hashes |
| `hard_ref` | add aliases/doc2query, add eval rows, review boundaries |
| `confusion_pair` | add disambiguators, hard negatives, alternate gold review |
| `query_style_gap` | generate weak-style questions or concern aliases |
| `gold_mode_gap` | improve range/distributed evidence policy |
| `undercitation` | increase candidate/context recall |
| `overcitation` | tighten expansion or split broad regions |
| `low_confidence` | gate auto-jumps or route to reranker |
| `query_magnet` | mark/downweight hub content |

This is important because it makes improvement loops auditable:

```text
evaluate -> smell report -> adaptation plan -> reviewed change -> rerun eval
```

Limit:

- `adapt-plan` is deliberately conservative. It suggests review-required
  actions and does not mutate a corpus automatically.

## Integration Evidence

Refmark exports rows for existing RAG/eval/observability tools while preserving
evidence metadata:

- Ragas-style rows;
- DeepEval-style cases;
- Phoenix/Langfuse-style trace events;
- lifecycle summary rows for tracking stale/ref-review/silent-wrong rates.

Optional real-SDK smoke checks construct native objects when those packages are
installed, but hosted ingestion is intentionally skipped unless the user
configures credentials and services.

The main integration finding:

> Existing answer/context evaluators can score generated prose, while Refmark
> tracks whether the underlying evidence labels are still valid and recoverable.

See [Eval Tool Integrations](EVAL_TOOL_INTEGRATIONS.md).

## Research Context

The broader research workspace contains BGB, browser search, static search,
small-model, and citation-localization experiments.

Useful current read:

- BGB is a strong stress test for corpus navigation and German legal structure;
- small local/search demos are promising, but not yet a headline product claim;
- training/corpus-local resolvers remain exploratory;
- citation localization often recovers the right support neighborhood before it
  recovers exact minimal ranges.

See [Current Benchmark Snapshot](CURRENT_BENCHMARK_SNAPSHOT.md),
[BGB Retrieval Experiment Report](BGB_RETRIEVAL_EXPERIMENT_REPORT.md), and
[Search And Training Findings](SEARCH_AND_TRAINING_FINDINGS.md).

## Claims To Avoid For Now

Do not claim:

- Refmark universally beats vector databases, BM25, embeddings, or search
  engines;
- Refmark eliminates hallucinations;
- tiny trained models replace semantic retrieval in general;
- browser search is already a polished product;
- coding agents broadly outperform baselines because of Refmark;
- PDF/DOCX refs preserve original page-layout boxes unless explicit layout
  provenance exists.

## Strongest Next Evidence

The highest-value next evidence is:

1. rerun the evidence loop on one realistic external docs corpus with held-out
   or manually written queries;
2. compare BM25, BM25+metadata, embeddings, and hybrid retrieval through the
   same Refmark `EvalSuite`;
3. apply one reviewed adaptation plan and show before/after held-out metrics;
4. export the same run to Ragas/DeepEval/Phoenix-style rows to demonstrate
   lifecycle-tool compatibility;
5. write a compact technical report around lifecycle evidence and data-smell
   adaptation, not benchmark superiority.
