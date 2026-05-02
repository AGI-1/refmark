# Evidence Eval Artifacts

Refmark retrieval experiments should stay comparable across corpus changes,
retriever variants, metadata adaptations, and training runs. The core rule is:
do not compare aggregate metrics unless the corpus, eval suite, and run settings
are explicitly identifiable.

## Stable Artifacts

### Corpus Snapshot

`CorpusMap.snapshot()` records:

- `revision_id`
- corpus `fingerprint`
- region count
- stable refs
- source metadata

The corpus fingerprint is derived from stable refs, region hashes, ordinals,
source paths, and parent ids. It is the comparison boundary for a refmarked
address space.

### Eval Suite Summary

`EvalSuite.summary()` records:

- suite `fingerprint`
- example count
- query-style distribution
- gold-mode distribution
- source-hash coverage

The suite fingerprint includes query text, gold refs/ranges, source hashes, and
metadata. This keeps direct, concern, adversarial, manual, and generated
question sets separable.

### Eval Run Artifact

`EvalSuite.run_artifact(run, settings=..., artifacts=...)` writes a comparable
run object:

```json
{
  "schema": "refmark.eval_run_artifact.v1",
  "run_fingerprint": "...",
  "comparison_key": "...",
  "corpus": {"fingerprint": "..."},
  "eval_suite": {"fingerprint": "..."},
  "settings": {"strategy": "rerank", "top_k": 10},
  "metrics": {},
  "diagnostics": {},
  "results": []
}
```

The `comparison_key` is built from corpus fingerprint, eval-suite fingerprint,
settings, and run name. It intentionally excludes incidental artifact paths, so
the same run can be reproduced in another folder without changing the key.

## Candidate Caches

Training and reranking experiments should cache expensive candidate pools when
possible. A candidate cache must include enough metadata to reject stale reuse:

- corpus/index hash
- question/eval file hashes
- train/eval split seed and fraction
- model id or model path
- model/view settings
- candidate-k
- article/ref digest
- train/eval query digests

The BGB bi-encoder reranker implements this as
`refmark.bgb_biencoder_candidate_cache.v1`. The same shape should be generalized
before exposing it as a public API.

## Comparability Rules

Use these rules for research notes and CI:

1. Compare retriever quality only when `comparison_key` is the same, or explain
   which component changed.
2. Treat metadata adaptation as a new corpus/search-view condition, not as the
   same baseline.
3. Keep train/adapt/eval question sets separate.
4. Report query-style and gold-mode breakdowns next to aggregate metrics.
5. Report candidate recall separately from reranker ordering quality.
6. Preserve stale-ref checks when a corpus revision changes.

This is the main reason Refmark matters in the training experiments: the refs
are not just labels. They are the shared coordinate system that makes retrieval,
adaptation, model training, and regression checks inspectable in the same terms.

## Existing Eval Tools

Refmark evidence metrics can be logged beside answer-level metrics in tools such
as RAGAS, LangSmith, MLflow, or internal dashboards. Use
`export_ragas_rows(...)` when a tool expects question/answer/context rows, and
use `refmark_evidence_metrics(...)` or `eval_tool_summary(...)` when a tool
expects run metadata. See [Eval Tool Integrations](EVAL_TOOL_INTEGRATIONS.md).
