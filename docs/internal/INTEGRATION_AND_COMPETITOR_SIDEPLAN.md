# Integration And Competitor Sideplan

Refmark should not be positioned as a replacement for RAG evaluators,
observability platforms, citation engines, search libraries, annotation
standards, or coding agents. The stronger story is narrower:

> Refmark supplies stable, resolvable, revision-aware evidence refs that make
> existing retrieval, citation, observability, and agent systems easier to test.

## Positioning

Preferred claim:

```text
Regression tests for retrieval and citations over changing corpora.
```

Avoid leading with:

```text
another RAG evaluator
another citation generator
another local search engine
another coding agent
```

Refmark owns the corpus/ref intelligence layer:

```text
source corpus -> stable refs/ranges -> discovery/review -> eval rows
-> deterministic evidence metrics -> stale-ref maintenance -> safe adaptations
```

Other tools should keep owning traces, production monitoring, model calls,
answer-quality judging, app dashboards, and framework-native retrieval.

## Compare/Integrate With

| Ecosystem | Refmark integration angle |
| --- | --- |
| Ragas / DeepEval | Add deterministic `gold_refs`, hit@k, range coverage, stale-ref metrics beside LLM-judged faithfulness/relevance. |
| TruLens / Phoenix / Langfuse | Export recovered refs, wrong refs, stale refs, data-smell labels, and highlighted evidence artifacts into traces/experiments. |
| LlamaIndex / Haystack | Provide document/node preprocessors that attach stable refs/ranges as metadata and eval adapters that score at ref granularity. |
| W3C Web Annotation / Hypothesis / text fragments | Export/import selector-like targets where useful; be clear Refmark is not a general annotation standard. |
| Aider / OpenHands / SWE-agent | Offer MCP/addressed-region editing as a primitive these agents can use, not a full competing coding agent. |
| DSPy / prompt optimizers | Supply cheap deterministic evidence-recovery metrics as optimizer objectives. |

## Product Boundary

Refmark should provide:

- region manifests and shadow registries;
- context cards and discovery review queues;
- ref/range evaluation metrics;
- stale-ref detection and affected-example refresh;
- data-smell findings tied to refs;
- adaptation proposals with mini-eval and blast-radius checks;
- import/export adapters for existing eval/observability tools.

Refmark should avoid becoming:

- a full LLM observability platform;
- a general annotation UI;
- a full vector database or search platform;
- a general answer-quality judge;
- a complete coding agent.

## Observed Query Events

A future production bridge should be small and adapter-friendly. Capture only
corpus-native fields that generic observability tools usually lack:

```json
{
  "event_id": "evt_...",
  "timestamp": "2026-04-30T12:00:00Z",
  "user_query": "How do I run code before my FastAPI app starts?",
  "query_family": "concern",
  "retriever_mode": "hybrid_bm25_embedding",
  "resolved_refmarks": [
    {"ref": "advanced/events:P01-P03", "rank": 1, "score": 0.83}
  ],
  "opened_targets": [
    {"url": "/advanced/events/", "ref": "advanced/events:P01-P03", "dwell_seconds": 48}
  ],
  "answer_refs": ["advanced/events:P01-P03"],
  "feedback": {"explicit": null, "implicit": "opened_top_result"}
}
```

Important: opened links are evidence, not truth. They should become review
candidates, not automatic gold labels.

## Future Sideplan

1. Write a public "Compared To..." page.
2. Add adapter schemas for Ragas/Haystack/LlamaIndex/Phoenix/Langfuse.
3. Add observed-query event ingestion.
4. Convert event clusters into eval candidates and data-smell review issues.
5. Keep all production-derived adaptations reviewable and regression-tested.

