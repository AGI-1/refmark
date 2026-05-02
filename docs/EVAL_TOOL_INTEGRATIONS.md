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

The core adapter layer is tested as dependency-free handoff data. That keeps
Refmark lightweight and prevents Ragas, DeepEval, Phoenix, Langfuse, or MLflow
from becoming required package dependencies. The normal tests verify that:

- resolved source text is present for answer-level evaluators;
- gold/retrieved/context refs remain attached;
- source hashes survive export for stale checks;
- trace rows contain stable fingerprints and stale-state metadata;
- lifecycle rows preserve corpus revision metrics.

There is also an optional real-SDK smoke test:

```bash
python examples/eval_tool_integrations_demo/real_sdk_smoke.py
```

With no optional packages installed, the script reports `not_installed` for each
tool. With the current SDK packages installed in an isolated environment, this
was checked against:

| Tool | SDK object/path checked | Result |
| --- | --- | --- |
| Ragas 0.4.3 | `EvaluationDataset` with `SingleTurnSample` rows | native construction ok; refs and source hashes preserved |
| DeepEval 3.9.9 | `LLMTestCase` with Refmark metadata | native construction ok; refs and source hashes preserved |
| Langfuse 4.5.1 | `Langfuse` retriever observation metadata | local construction ok; refs and fingerprints preserved |
| Phoenix/OpenInference | in-memory OpenTelemetry retriever span | span construction ok; refs and fingerprints preserved |

Hosted ingestion is intentionally not part of the default check. Langfuse needs
caller-owned `LANGFUSE_*` credentials and Phoenix needs a running server or OTLP
endpoint. Ragas and DeepEval answer-quality metrics may also need model/judge
configuration. Refmark's role in these integrations is the evidence layer: refs,
resolved contexts, source hashes, fingerprints, stale-state metadata, and
evidence metrics.

The full `arize-phoenix` server package was not made a project dependency. A
one-shot install attempt pulled a broad server stack and hit pip resolver depth;
the lighter Phoenix/OpenInference client path was enough to validate the span
metadata shape locally.

## Mutation Comparison: Ragas Alone vs Ragas + Refmark

The most useful comparison is not "Ragas versus Refmark." Ragas evaluates
answer/context quality. Refmark adds evidence identity and corpus lifecycle
checks before those answer/context rows are trusted.

`examples/ragas_refmark_mutation_demo/run.py` demonstrates the failure mode:

1. Create eval rows on `policy-v1` with `query -> gold_refs` and source hashes.
2. Mutate the corpus into `policy-v2` while keeping the same visible ref ids.
3. Export plain Ragas-style rows from the current corpus.
4. Export Refmark-enriched Ragas-style rows from the same run.

In that demo, retrieval against the mutated corpus still has perfect evidence
hit metrics because the current strings and current refs are self-consistent:

```json
{
  "hit_at_k": 1.0,
  "gold_coverage": 1.0
}
```

But Refmark also reports:

```json
{
  "stale_example_count": 2,
  "changed_refs": ["policy:P01", "policy:P02"]
}
```

That is the practical integration point. Ragas can still score generated
answers and retrieved context text, while Refmark tells the lifecycle system
that the maintained eval labels were created against old source text and need
review or refresh.

## Real-Corpus Lifecycle Rows

`examples/lifecycle_tool_integrations_demo/run.py` consumes compact summary rows
from five Git-backed documentation revision runs: FastAPI, Django, Flask, HTTPX,
and Kubernetes. It does not ship raw cloned corpora; it ships only the compact
metrics needed to demonstrate the integration surface.

The demo exports `lifecycle_tool_rows.jsonl`, where each row can be logged to an
experiment tracker or observability system beside normal Ragas/DeepEval answer
metrics:

```json
{
  "schema": "refmark.lifecycle_tool_row.v1",
  "name": "lifecycle:0.100.0->0.115.0",
  "metrics": {
    "refmark_auto_rate": 0.0655,
    "refmark_review_rate": 0.3401,
    "refmark_stale_rate": 0.5944,
    "naive_silent_wrong_rate": 0.6147
  }
}
```

The current compact fixture covers 15 revision comparisons over 5 documentation
corpora. In those rows, the average naive silent-wrong rate is about `30.1%`.
That number is not a universal claim about all chunking systems. It is evidence
for the lifecycle failure mode: persisted evidence labels can keep resolving
while pointing to different content.

The operational recipe is:

1. Run Ragas or DeepEval on current answer/context rows.
2. Run Refmark stale/lifecycle validation on the same eval suite.
3. Log both surfaces to the same tracker.
4. Gate or annotate answer metrics when stale/review rates exceed thresholds.

This makes Refmark additive: existing evaluators keep their answer-quality role,
while Refmark supplies evidence identity and corpus-change safety.

## Caveats

- Evidence metrics do not replace answer-level evaluation. A system can recover
  the right evidence and still generate a bad answer.
- Stale-ref checks require eval rows to preserve source hashes, for example via
  `attach_source_hashes=True`.
- Overcitation and undercitation depend on the chosen gold refs/ranges. If gold
  labels are too narrow or too broad, the metrics will reflect that choice.
