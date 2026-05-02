# Eval Tool Integrations

Refmark is meant to sit below existing RAG evaluation and lifecycle tools, not
replace them. Tools such as Ragas, DeepEval, Phoenix, Langfuse, LangSmith,
MLflow, or custom dashboards can continue to measure answer quality,
faithfulness, latency, cost, traces, and production behavior. Refmark adds the
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

## DeepEval-Style Rows

`export_deepeval_cases(...)` emits dependency-free dictionaries shaped like
DeepEval `LLMTestCase` inputs:

```python
from refmark import export_deepeval_cases

cases = export_deepeval_cases(
    suite,
    run,
    answers={"How do I configure auth?": "Use OAuth2 settings."},
)
```

Each row includes:

- `input`
- `actual_output`
- `expected_output`
- `retrieval_context`
- `context`
- `refmark.gold_refs`
- `refmark.retrieved_refs`
- `refmark.context_refs`
- `refmark.hit_at_k`
- `refmark.gold_coverage`
- `refmark.source_hashes`

Callers that already use DeepEval can turn those dictionaries into native test
case objects. Refmark intentionally does not import DeepEval directly in the
core package.

## Phoenix / Langfuse / Trace-Style Rows

`export_trace_events(...)` emits one retrieval event per query:

```python
from refmark import export_trace_events

events = export_trace_events(suite, run, tool="phoenix")
```

Each event contains stable fingerprints and attributes that fit naturally into
trace metadata or span attributes:

- `refmark.corpus_fingerprint`
- `refmark.eval_suite_fingerprint`
- `refmark.run_fingerprint`
- `refmark.gold_refs`
- `refmark.retrieved_refs`
- `refmark.context_refs`
- `refmark.hit_at_1`
- `refmark.hit_at_k`
- `refmark.gold_coverage`
- `refmark.region_precision`
- `refmark.source_hashes`
- `refmark.stale`

For Langfuse-style usage, store the event as a trace/span with the `attributes`
object as metadata. For Phoenix/OpenInference-style usage, the same attributes
can be attached to retrieval spans or dataset examples.

## Lifecycle Rows

`export_lifecycle_summary_rows(...)` converts `lifecycle-git` `summary_rows`
into tracker rows:

```python
from refmark import export_lifecycle_summary_rows

rows = export_lifecycle_summary_rows(lifecycle_payload, tool="mlflow")
```

Promoted metrics include:

- `refmark_auto_rate`
- `refmark_review_rate`
- `refmark_stale_rate`
- `naive_correct_rate`
- `naive_silent_wrong_rate`
- `naive_missing_rate`
- `workload_reduction_vs_audit`

This is the lifecycle integration point: a tracker can store how much of an eval
suite survived a corpus revision, how much needs review, and how much would have
silently pointed to the wrong place under naive chunk identity.

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

## What Was Tested

The adapter layer is currently tested as dependency-free handoff data, not as
live SDK calls into each vendor package. The tests verify that:

- resolved source text is present for answer-level evaluators;
- gold/retrieved/context refs remain attached;
- source hashes survive export for stale checks;
- trace rows contain stable fingerprints and stale-state metadata;
- lifecycle rows preserve corpus revision metrics.

This is deliberate. It keeps Refmark's package surface lightweight while making
the integration boundary explicit. Thin optional SDK-specific shims can be added
later without changing the evidence metrics or row schemas.

## Caveats

- Evidence metrics do not replace answer-level evaluation. A system can recover
  the right evidence and still generate a bad answer.
- Stale-ref checks require eval rows to preserve source hashes, for example via
  `attach_source_hashes=True`.
- Overcitation and undercitation depend on the chosen gold refs/ranges. If gold
  labels are too narrow or too broad, the metrics will reflect that choice.
