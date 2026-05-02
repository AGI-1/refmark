# Eval Tool Integrations

Refmark is meant to sit below existing RAG evaluation tools, not replace them.
Tools such as RAGAS, LangSmith, MLflow, or custom dashboards can continue to
measure answer quality, faithfulness, latency, and cost. Refmark adds the
evidence-address layer:

- did the retriever return the expected source refs/ranges?
- did the final context include too little or too much evidence?
- did a corpus update make a curated eval row stale?
- are direct questions passing while concern/adversarial questions fail?
- are failures concentrated in a small set of regions or confusions?

This is useful because answer-level evaluation can say that an answer is weak,
but it often cannot localize whether the failure came from the retriever,
reranker, context expansion policy, stale labels, or the generator.

## RAGAS-Style Rows

The adapter is dependency-free. It emits ordinary dictionaries with common
RAGAS-style columns plus Refmark metadata:

```python
from refmark import CorpusMap, EvalSuite, export_ragas_rows, refmark_evidence_metrics

corpus = CorpusMap.from_manifest(".refmark/docs.jsonl", revision_id="git:abc123")
suite = EvalSuite.from_jsonl("eval_questions.jsonl", corpus=corpus, attach_source_hashes=True)

def my_retriever(query: str):
    # Your existing retriever can be BM25, embeddings, a vector DB, or hybrid.
    # The only requirement is that hits expose stable refs.
    hits = existing_search_service(query, top_k=10)
    return [
        {
            "stable_ref": hit["ref"],
            "score": hit.get("score"),
            "context_refs": hit.get("expanded_refs", [hit["ref"]]),
        }
        for hit in hits
    ]

run = suite.evaluate(my_retriever, name="hybrid", k=10)

rows = export_ragas_rows(
    suite,
    run,
    answers={"How do I configure auth?": "Use the OAuth2 settings section."},
)
metrics = refmark_evidence_metrics(suite, run)
```

Each exported row includes:

- `user_input`
- `response`
- `retrieved_contexts`
- `reference`
- `gold_refs`
- `retrieved_refs`
- `context_refs`
- `refmark.hit_at_1`
- `refmark.hit_at_k`
- `refmark.gold_coverage`
- `refmark.region_precision`
- `refmark.query_style`
- `refmark.gold_mode`
- `refmark.source_hashes`

The `retrieved_contexts` and `reference` fields are resolved source text, so the
same rows can be sent to answer-level evaluators. The refs stay attached so a
dashboard can still jump back to exact regions.

## Refmark Metrics

`refmark_evidence_metrics(suite, run)` emits a compact metrics payload:

```json
{
  "schema": "refmark.evidence_metrics.v1",
  "hit_at_1": 0.72,
  "hit_at_k": 0.91,
  "gold_coverage": 0.88,
  "region_precision": 0.64,
  "avg_overcitation_refs": 1.3,
  "avg_undercitation_refs": 0.2,
  "avg_support_tokens": 940.0,
  "stale_example_count": 4,
  "stale_ref_count": 7
}
```

These metrics are intentionally lower-level than generated-answer scores. They
answer whether the right evidence entered the context window and whether the
evaluation data itself is still valid for the current corpus revision.

## Experiment Trackers

`eval_tool_summary(suite, run, tool="mlflow")` returns a compact run summary with:

- corpus snapshot and fingerprint
- eval-suite summary and fingerprint
- run fingerprint
- Refmark evidence metrics
- heatmap/confusion diagnostics

That payload can be logged as experiment metadata next to normal latency,
answer-quality, and model-cost measurements.

## Adoption Pattern

1. Keep the existing RAG system.
2. Add a Refmark manifest for the corpus.
3. Store eval rows as `query -> gold_refs`.
4. Wrap each retriever/reranker variant so it returns stable refs.
5. Export rows to the existing evaluator.
6. Log Refmark evidence metrics beside answer-level metrics.
7. On corpus update, run stale checks before trusting old labels.

Refmark does not require BM25, embeddings, vector databases, rerankers, or
training. It gives those choices a shared testable target.

## Caveats

- Evidence metrics do not replace answer-level evaluation. A system can recover
  the right evidence and still generate a bad answer.
- Stale-ref checks require eval rows to preserve source hashes, for example via
  `attach_source_hashes=True`.
- Overcitation and undercitation depend on the chosen gold refs/ranges. If gold
  labels are too narrow or too broad, the metrics will reflect that choice.
