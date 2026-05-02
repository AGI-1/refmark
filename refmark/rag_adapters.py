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
