"""Adapters from Refmark evidence evals to common RAG evaluation surfaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import json
from typing import Any

from refmark.rag_eval import CorpusMap, EvalRun, EvalSuite


def refmark_evidence_metrics(suite: EvalSuite, run: EvalRun) -> dict[str, Any]:
    """Return Refmark-native metrics for external eval dashboards.

    The returned payload is intentionally dependency-free. Tools such as RAGAS,
    LangSmith, MLflow, or custom CI jobs can attach it as extra run metadata
    while still using their own answer-quality metrics.
    """

    _ensure_aligned(suite, run)
    overcitation_counts: list[int] = []
    undercitation_counts: list[int] = []
    support_tokens: list[int] = []
    for result in run.examples:
        gold_set = set(result.gold_refs)
        context_set = set(result.context_refs)
        overcitation_counts.append(len(context_set - gold_set))
        undercitation_counts.append(len(gold_set - context_set))
        if result.context_refs:
            support_tokens.append(suite.corpus.context_pack(result.context_refs).token_estimate)
        else:
            support_tokens.append(0)
    stale = suite.stale_examples()
    return {
        "schema": "refmark.evidence_metrics.v1",
        "corpus_fingerprint": suite.corpus.fingerprint,
        "eval_suite_fingerprint": suite.fingerprint,
        "run_fingerprint": run.fingerprint,
        "run_name": run.name,
        "count": len(run.examples),
        "hit_at_1": run.metrics.get("hit_at_1", 0.0),
        "hit_at_k": run.metrics.get("hit_at_k", 0.0),
        "mrr": run.metrics.get("mrr", 0.0),
        "gold_coverage": run.metrics.get("gold_coverage", 0.0),
        "region_precision": run.metrics.get("region_precision", 0.0),
        "avg_context_refs": run.metrics.get("avg_context_refs", 0.0),
        "avg_overcitation_refs": _mean(overcitation_counts),
        "avg_undercitation_refs": _mean(undercitation_counts),
        "avg_support_tokens": _mean(support_tokens),
        "stale_example_count": len(stale),
        "stale_ref_count": sum(len(item.missing_refs) + len(item.changed_refs) for item in stale),
        "query_style_gap": run.diagnostics.get("query_style_gap", {}),
        "gold_mode_breakdown": run.diagnostics.get("by_gold_mode", {}),
    }


def export_ragas_rows(
    suite: EvalSuite,
    run: EvalRun,
    *,
    answers: Mapping[str, str] | Sequence[str] | None = None,
    references: Mapping[str, str] | Sequence[str] | None = None,
    include_refmark_fields: bool = True,
) -> list[dict[str, Any]]:
    """Return RAGAS-style rows with Refmark evidence metadata.

    The shape uses the common RAGAS dataset columns `user_input`, `response`,
    `retrieved_contexts`, and `reference`. It does not import RAGAS; callers can
    feed these dicts into their chosen dataframe/dataset layer.
    """

    _ensure_aligned(suite, run)
    rows: list[dict[str, Any]] = []
    for index, result in enumerate(run.examples):
        example = suite.examples[index]
        row: dict[str, Any] = {
            "user_input": result.query,
            "response": _lookup_text(answers, index, result.query, default=""),
            "retrieved_contexts": _context_texts(suite.corpus, result.context_refs),
            "reference": _lookup_text(
                references,
                index,
                result.query,
                default=suite.corpus.context_pack(result.gold_refs).text,
            ),
        }
        if include_refmark_fields:
            row.update(
                {
                    "gold_refs": result.gold_refs,
                    "retrieved_refs": result.retrieved_refs,
                    "context_refs": result.context_refs,
                    "refmark": {
                        "gold_refs": result.gold_refs,
                        "retrieved_refs": result.retrieved_refs,
                        "context_refs": result.context_refs,
                        "hit_at_1": result.hit_at_1,
                        "hit_at_k": result.hit_at_k,
                        "gold_coverage": result.gold_coverage,
                        "region_precision": result.region_precision,
                        "query_style": result.query_style,
                        "gold_mode": result.gold_mode,
                        "source_hashes": dict(example.source_hashes),
                        "top_ref": result.top_ref,
                        "score_margin": result.score_margin,
                    },
                }
            )
        rows.append(row)
    return rows


def write_ragas_jsonl(
    path: str | Path,
    suite: EvalSuite,
    run: EvalRun,
    *,
    answers: Mapping[str, str] | Sequence[str] | None = None,
    references: Mapping[str, str] | Sequence[str] | None = None,
    include_refmark_fields: bool = True,
) -> None:
    """Write RAGAS-style rows as JSONL."""

    rows = export_ragas_rows(
        suite,
        run,
        answers=answers,
        references=references,
        include_refmark_fields=include_refmark_fields,
    )
    Path(path).write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def export_deepeval_cases(
    suite: EvalSuite,
    run: EvalRun,
    *,
    answers: Mapping[str, str] | Sequence[str] | None = None,
    expected_outputs: Mapping[str, str] | Sequence[str] | None = None,
    include_refmark_fields: bool = True,
) -> list[dict[str, Any]]:
    """Return DeepEval-style LLM test-case dictionaries.

    The rows use the common `input`, `actual_output`, `expected_output`,
    `retrieval_context`, and `context` fields while preserving Refmark evidence
    metadata. This avoids a hard dependency on DeepEval but keeps the handoff
    shape obvious for callers that want to construct `LLMTestCase` objects.
    """

    _ensure_aligned(suite, run)
    rows: list[dict[str, Any]] = []
    for index, result in enumerate(run.examples):
        row: dict[str, Any] = {
            "input": result.query,
            "actual_output": _lookup_text(answers, index, result.query, default=""),
            "expected_output": _lookup_text(
                expected_outputs,
                index,
                result.query,
                default=suite.corpus.context_pack(result.gold_refs).text,
            ),
            "retrieval_context": _context_texts(suite.corpus, result.context_refs),
            "context": _context_texts(suite.corpus, suite.corpus.expand_refs(result.gold_refs)),
        }
        if include_refmark_fields:
            row["refmark"] = _refmark_row_metadata(suite, run, index)
        rows.append(row)
    return rows


def write_deepeval_jsonl(
    path: str | Path,
    suite: EvalSuite,
    run: EvalRun,
    *,
    answers: Mapping[str, str] | Sequence[str] | None = None,
    expected_outputs: Mapping[str, str] | Sequence[str] | None = None,
    include_refmark_fields: bool = True,
) -> None:
    """Write DeepEval-style rows as JSONL."""

    rows = export_deepeval_cases(
        suite,
        run,
        answers=answers,
        expected_outputs=expected_outputs,
        include_refmark_fields=include_refmark_fields,
    )
    _write_jsonl(path, rows)


def export_trace_events(
    suite: EvalSuite,
    run: EvalRun,
    *,
    tool: str = "generic",
    answers: Mapping[str, str] | Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Return observability-trace events for tools such as Phoenix/Langfuse.

    Each row is a dependency-free trace/event payload with stable IDs,
    fingerprints, retrieved refs, context refs, gold refs, stale status, and
    score/margin metadata. Adapters can log this as span attributes, trace
    metadata, or dataset examples depending on the destination tool.
    """

    _ensure_aligned(suite, run)
    stale_by_query = {item.example.query: item.to_dict() for item in suite.stale_examples()}
    events: list[dict[str, Any]] = []
    for index, result in enumerate(run.examples):
        example = suite.examples[index]
        trace_id = _digest(
            {
                "corpus": suite.corpus.fingerprint,
                "suite": suite.fingerprint,
                "run": run.fingerprint,
                "index": index,
                "query": result.query,
            }
        )
        events.append(
            {
                "schema": "refmark.trace_event.v1",
                "tool": tool,
                "trace_id": trace_id,
                "span_name": f"retrieval:{run.name}",
                "input": result.query,
                "output": _lookup_text(answers, index, result.query, default=""),
                "attributes": {
                    "refmark.corpus_fingerprint": suite.corpus.fingerprint,
                    "refmark.eval_suite_fingerprint": suite.fingerprint,
                    "refmark.run_fingerprint": run.fingerprint,
                    "refmark.run_name": run.name,
                    "refmark.gold_refs": result.gold_refs,
                    "refmark.retrieved_refs": result.retrieved_refs,
                    "refmark.context_refs": result.context_refs,
                    "refmark.hit_at_1": result.hit_at_1,
                    "refmark.hit_at_k": result.hit_at_k,
                    "refmark.gold_coverage": result.gold_coverage,
                    "refmark.region_precision": result.region_precision,
                    "refmark.query_style": result.query_style,
                    "refmark.gold_mode": result.gold_mode,
                    "refmark.top_ref": result.top_ref,
                    "refmark.top_score": result.top_score,
                    "refmark.score_margin": result.score_margin,
                    "refmark.source_hashes": dict(example.source_hashes),
                    "refmark.stale": result.query in stale_by_query,
                    "refmark.stale_detail": stale_by_query.get(result.query),
                },
            }
        )
    return events


def write_trace_jsonl(
    path: str | Path,
    suite: EvalSuite,
    run: EvalRun,
    *,
    tool: str = "generic",
    answers: Mapping[str, str] | Sequence[str] | None = None,
) -> None:
    """Write observability trace/event payloads as JSONL."""

    _write_jsonl(path, export_trace_events(suite, run, tool=tool, answers=answers))


def export_lifecycle_summary_rows(
    lifecycle_payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    tool: str = "generic",
) -> list[dict[str, Any]]:
    """Return lifecycle-summary rows shaped for experiment trackers.

    Accepts either a full `lifecycle-git` payload with `summary_rows` or a list
    of summary rows. The output keeps the original row under `refmark.lifecycle`
    while promoting the most useful tracker dimensions to top-level fields.
    """

    if isinstance(lifecycle_payload, Mapping):
        rows = [dict(row) for row in lifecycle_payload.get("summary_rows", [])]
    else:
        rows = [dict(row) for row in lifecycle_payload]
    result: list[dict[str, Any]] = []
    for row in rows:
        run_id = _digest(
            {
                "repo_url": row.get("repo_url"),
                "subdir": row.get("subdir"),
                "old_ref": row.get("old_ref"),
                "new_ref": row.get("new_ref"),
            }
        )
        result.append(
            {
                "schema": "refmark.lifecycle_tool_row.v1",
                "tool": tool,
                "run_id": run_id,
                "name": f"lifecycle:{row.get('old_ref')}->{row.get('new_ref')}",
                "dataset": row.get("repo_url"),
                "metadata": {
                    "subdir": row.get("subdir"),
                    "old_ref": row.get("old_ref"),
                    "new_ref": row.get("new_ref"),
                    "old_labels": row.get("old_labels"),
                    "new_regions": row.get("new_regions"),
                    "new_tokens": row.get("new_tokens"),
                },
                "metrics": {
                    "refmark_auto_rate": row.get("refmark_auto_rate", 0.0),
                    "refmark_review_rate": row.get("refmark_review_rate", 0.0),
                    "refmark_stale_rate": row.get("refmark_stale_rate", 0.0),
                    "naive_correct_rate": row.get("naive_correct_rate", 0.0),
                    "naive_silent_wrong_rate": row.get("naive_silent_wrong_rate", 0.0),
                    "naive_missing_rate": row.get("naive_missing_rate", 0.0),
                    "workload_reduction_vs_audit": row.get("workload_reduction_vs_audit", 0.0),
                },
                "refmark": {"lifecycle": row},
            }
        )
    return result


def write_lifecycle_tool_jsonl(
    path: str | Path,
    lifecycle_payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    tool: str = "generic",
) -> None:
    """Write lifecycle summary rows as generic tracker JSONL."""

    _write_jsonl(path, export_lifecycle_summary_rows(lifecycle_payload, tool=tool))


def eval_tool_summary(suite: EvalSuite, run: EvalRun, *, tool: str = "generic") -> dict[str, Any]:
    """Return a compact run summary suitable for external experiment trackers."""

    _ensure_aligned(suite, run)
    return {
        "schema": "refmark.eval_tool_summary.v1",
        "tool": tool,
        "corpus": suite.corpus.snapshot().to_dict(),
        "eval_suite": suite.summary(),
        "run_name": run.name,
        "run_fingerprint": run.fingerprint,
        "metrics": refmark_evidence_metrics(suite, run),
        "diagnostics": run.diagnostics,
    }


def _refmark_row_metadata(suite: EvalSuite, run: EvalRun, index: int) -> dict[str, Any]:
    result = run.examples[index]
    example = suite.examples[index]
    return {
        "gold_refs": result.gold_refs,
        "retrieved_refs": result.retrieved_refs,
        "context_refs": result.context_refs,
        "hit_at_1": result.hit_at_1,
        "hit_at_k": result.hit_at_k,
        "gold_coverage": result.gold_coverage,
        "region_precision": result.region_precision,
        "query_style": result.query_style,
        "gold_mode": result.gold_mode,
        "source_hashes": dict(example.source_hashes),
        "top_ref": result.top_ref,
        "score_margin": result.score_margin,
    }


def _context_texts(corpus: CorpusMap, refs: Sequence[str]) -> list[str]:
    by_ref = corpus.by_stable_ref
    texts: list[str] = []
    for ref in refs:
        record = by_ref.get(ref)
        if record is not None:
            texts.append(f"[{ref}]\n{record.text.rstrip()}")
    return texts


def _lookup_text(
    values: Mapping[str, str] | Sequence[str] | None,
    index: int,
    query: str,
    *,
    default: str,
) -> str:
    if values is None:
        return default
    if isinstance(values, Mapping):
        return values.get(query, default)
    if isinstance(values, str):
        return values
    return values[index] if index < len(values) else default


def _ensure_aligned(suite: EvalSuite, run: EvalRun) -> None:
    if len(suite.examples) != len(run.examples):
        raise ValueError("EvalSuite and EvalRun must contain the same number of examples")


def _mean(values: Sequence[int | float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    Path(path).write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    import hashlib

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
