"""Data-smell reports for evidence retrieval runs."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from refmark.rag_eval import CorpusMap, EvalRun, EvalSuite


@dataclass(frozen=True)
class DataSmell:
    """A reviewable retrieval/corpus smell with evidence and suggested actions."""

    type: str
    severity: str
    message: str
    refs: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    suggested_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DataSmellReport:
    """First-class report for retrieval/evidence data-smell analysis."""

    schema: str
    summary: dict[str, Any]
    smells: list[DataSmell]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "summary": self.summary,
            "smells": [smell.to_dict() for smell in self.smells],
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def build_data_smell_report(
    suite: EvalSuite,
    run: EvalRun,
    *,
    corpus: CorpusMap | None = None,
    include_text: bool = True,
    max_text_chars: int = 700,
    max_smells: int = 80,
) -> DataSmellReport:
    """Build a consolidated data-smell report from an evaluated retrieval run."""

    if len(suite.examples) != len(run.examples):
        raise ValueError("EvalSuite and EvalRun must contain the same number of examples")
    corpus = corpus or suite.corpus
    smells: list[DataSmell] = []
    smells.extend(_stale_label_smells(suite, corpus))
    smells.extend(_hard_ref_smells(run, corpus, include_text=include_text, max_text_chars=max_text_chars))
    smells.extend(_confusion_smells(run, corpus, include_text=include_text, max_text_chars=max_text_chars))
    smells.extend(_query_style_gap_smells(run))
    smells.extend(_gold_mode_gap_smells(run))
    smells.extend(_over_under_citation_smells(run))
    smells.extend(_low_confidence_smells(run))
    smells.extend(_query_magnet_smells(run, corpus, include_text=include_text, max_text_chars=max_text_chars))
    smells = sorted(smells, key=lambda smell: (_severity_rank(smell.severity), smell.type, ",".join(smell.refs)))[:max_smells]
    summary = _summary(suite, run, smells)
    return DataSmellReport(schema="refmark.data_smells.v1", summary=summary, smells=smells)


def _stale_label_smells(suite: EvalSuite, corpus: CorpusMap) -> list[DataSmell]:
    smells = []
    for item in corpus.stale_examples(suite.examples):
        refs = [*item.changed_refs, *item.missing_refs]
        severity = "high" if item.missing_refs else "medium"
        smells.append(
            DataSmell(
                type="stale_label",
                severity=severity,
                message="Saved eval label points to changed or missing source evidence.",
                refs=refs,
                evidence={
                    "query": item.example.query,
                    "changed_refs": item.changed_refs,
                    "missing_refs": item.missing_refs,
                    "gold_refs": item.example.gold_refs,
                },
                suggested_actions=[
                    "review_label_against_current_corpus",
                    "refresh_source_hashes_after_human_or_judge_approval",
                    "regenerate_question_if_evidence_changed_semantically",
                ],
            )
        )
    return smells


def _hard_ref_smells(run: EvalRun, corpus: CorpusMap, *, include_text: bool, max_text_chars: int) -> list[DataSmell]:
    smells = []
    for row in run.diagnostics.get("heatmap", {}).get("hard_refs", []):
        if int(row.get("miss_at_k", 0)) <= 0:
            continue
        hit_at_k = float(row.get("hit_at_k", 0.0))
        severity = "high" if hit_at_k == 0.0 else "medium"
        ref = str(row["ref"])
        smells.append(
            DataSmell(
                type="hard_ref",
                severity=severity,
                message=f"Region missed {row['miss_at_k']} of {row['count']} eval queries at k.",
                refs=[ref],
                evidence={
                    **_ref_packet(corpus, ref, include_text=include_text, max_text_chars=max_text_chars),
                    "count": row.get("count"),
                    "miss_at_1": row.get("miss_at_1"),
                    "miss_at_k": row.get("miss_at_k"),
                    "hit_at_k": hit_at_k,
                    "sample_misses": row.get("sample_misses", []),
                },
                suggested_actions=[
                    "add_or_rewrite_eval_queries",
                    "add_shadow_metadata_or_doc2query_aliases",
                    "review_region_boundaries",
                    "add_hard_negatives_for_competing_refs",
                ],
            )
        )
    return smells


def _confusion_smells(run: EvalRun, corpus: CorpusMap, *, include_text: bool, max_text_chars: int) -> list[DataSmell]:
    smells = []
    for row in run.diagnostics.get("heatmap", {}).get("confusions", []):
        gold_ref = str(row["gold_ref"])
        top_ref = str(row["top_ref"])
        count = int(row["count"])
        smells.append(
            DataSmell(
                type="confusion_pair",
                severity="high" if count >= 3 else "medium",
                message=f"Gold ref is repeatedly confused with wrong top ref {top_ref}.",
                refs=[gold_ref, top_ref],
                evidence={
                    "count": count,
                    "gold": _ref_packet(corpus, gold_ref, include_text=include_text, max_text_chars=max_text_chars),
                    "competing": _ref_packet(corpus, top_ref, include_text=include_text, max_text_chars=max_text_chars),
                    "sample_queries": _queries_for_confusion(run, gold_ref, top_ref),
                },
                suggested_actions=[
                    "add_disambiguating_shadow_metadata",
                    "add_hard_negative_for_competing_ref",
                    "mark_alternate_gold_if_competing_ref_is_valid",
                    "merge_or_link_equivalent_sections",
                ],
            )
        )
    return smells


def _query_style_gap_smells(run: EvalRun) -> list[DataSmell]:
    gap = run.diagnostics.get("query_style_gap", {})
    hit_gap = float(gap.get("hit_at_1_gap", 0.0) or 0.0)
    if hit_gap < 0.25:
        return []
    weakest = gap.get("weakest_by_hit_at_1") or {}
    return [
        DataSmell(
            type="query_style_gap",
            severity="high" if hit_gap >= 0.5 else "medium",
            message="Retrieval quality differs strongly by query style.",
            evidence={"gap": gap, "weakest_style": weakest},
            suggested_actions=[
                "generate_more_questions_for_weak_style",
                "add_concern_or_adversarial_doc2query_metadata",
                "evaluate_style_specific_thresholds",
            ],
        )
    ]


def _gold_mode_gap_smells(run: EvalRun) -> list[DataSmell]:
    by_mode = run.diagnostics.get("by_gold_mode", {})
    if len(by_mode) < 2:
        return []
    values = {mode: float(metrics.get("gold_coverage", 0.0)) for mode, metrics in by_mode.items()}
    if max(values.values()) - min(values.values()) < 0.25:
        return []
    weakest = min(values.items(), key=lambda item: (item[1], item[0]))
    return [
        DataSmell(
            type="gold_mode_gap",
            severity="medium",
            message="Single/range/distributed evidence modes have uneven coverage.",
            evidence={"by_gold_mode": by_mode, "weakest_mode": {"mode": weakest[0], "gold_coverage": weakest[1]}},
            suggested_actions=[
                "add_range_or_distributed_eval_rows",
                "review_context_expansion_policy",
                "score_neighbor_and_parent_hits_separately",
            ],
        )
    ]


def _over_under_citation_smells(run: EvalRun) -> list[DataSmell]:
    under = []
    over = []
    for result in run.examples:
        gold = set(result.gold_refs)
        context = set(result.context_refs)
        if gold - context:
            under.append({"query": result.query, "missing_refs": sorted(gold - context), "context_refs": result.context_refs})
        if context - gold and result.region_precision < 0.5:
            over.append({"query": result.query, "extra_refs": sorted(context - gold), "context_refs": result.context_refs})
    smells = []
    if under:
        smells.append(
            DataSmell(
                type="undercitation",
                severity="high",
                message="Retrieved context misses one or more gold evidence refs.",
                refs=sorted({ref for row in under for ref in row["missing_refs"]}),
                evidence={"count": len(under), "samples": under[:8]},
                suggested_actions=["expand_context_neighbors", "increase_candidate_k", "review_gold_range_boundaries"],
            )
        )
    if over:
        smells.append(
            DataSmell(
                type="overcitation",
                severity="medium",
                message="Retrieved context includes many non-gold refs.",
                refs=sorted({ref for row in over for ref in row["extra_refs"]})[:30],
                evidence={"count": len(over), "samples": over[:8]},
                suggested_actions=["tighten_context_expansion", "split_broad_regions", "rerank_context_refs"],
            )
        )
    return smells


def _low_confidence_smells(run: EvalRun) -> list[DataSmell]:
    low = [
        result
        for result in run.examples
        if result.score_margin_ratio is not None and result.score_margin_ratio < 0.08
    ]
    if not low:
        return []
    correct = sum(1 for result in low if result.hit_at_1)
    return [
        DataSmell(
            type="low_confidence",
            severity="medium",
            message="Many accepted top hits have narrow score margins.",
            refs=sorted({result.top_ref for result in low if result.top_ref})[:30],
            evidence={
                "count": len(low),
                "precision_at_1": round(correct / len(low), 4),
                "samples": [
                    {
                        "query": result.query,
                        "top_ref": result.top_ref,
                        "score_margin_ratio": result.score_margin_ratio,
                        "hit_at_1": result.hit_at_1,
                    }
                    for result in low[:8]
                ],
            },
            suggested_actions=["route_low_margin_queries_to_reranker", "show_multiple_candidates", "raise_auto_jump_threshold"],
        )
    ]


def _query_magnet_smells(run: EvalRun, corpus: CorpusMap, *, include_text: bool, max_text_chars: int) -> list[DataSmell]:
    top_refs = Counter(result.top_ref for result in run.examples if result.top_ref)
    threshold = max(4, int(len(run.examples) * 0.15))
    smells = []
    for ref, count in top_refs.most_common(20):
        if count < threshold:
            continue
        hit_count = sum(1 for result in run.examples if result.top_ref == ref and result.hit_at_1)
        if hit_count / max(count, 1) >= 0.8:
            continue
        smells.append(
            DataSmell(
                type="query_magnet",
                severity="medium",
                message=f"Ref appears as top result for {count} queries but is often not the gold evidence.",
                refs=[str(ref)],
                evidence={
                    **_ref_packet(corpus, str(ref), include_text=include_text, max_text_chars=max_text_chars),
                    "top_count": count,
                    "top_precision": round(hit_count / max(count, 1), 4),
                    "sample_queries": [result.query for result in run.examples if result.top_ref == ref and not result.hit_at_1][:8],
                },
                suggested_actions=[
                    "mark_query_magnet_or_hub_role",
                    "exclude_or_downweight_in_default_search",
                    "add_disambiguating_metadata_to_neighbors",
                ],
            )
        )
    return smells


def _summary(suite: EvalSuite, run: EvalRun, smells: list[DataSmell]) -> dict[str, Any]:
    counts = Counter(smell.type for smell in smells)
    severities = Counter(smell.severity for smell in smells)
    return {
        "example_count": len(suite.examples),
        "run_name": run.name,
        "run_fingerprint": run.fingerprint,
        "corpus_fingerprint": suite.corpus.fingerprint,
        "metric_hit_at_k": run.metrics.get("hit_at_k", 0.0),
        "metric_gold_coverage": run.metrics.get("gold_coverage", 0.0),
        "smell_count": len(smells),
        "by_type": dict(sorted(counts.items())),
        "by_severity": dict(sorted(severities.items())),
    }


def _queries_for_confusion(run: EvalRun, gold_ref: str, top_ref: str) -> list[str]:
    return [
        result.query
        for result in run.examples
        if result.gold_refs and result.gold_refs[0] == gold_ref and result.top_ref == top_ref
    ][:8]


def _ref_packet(corpus: CorpusMap, ref: str, *, include_text: bool, max_text_chars: int) -> dict[str, Any]:
    record = corpus.by_stable_ref.get(ref)
    if record is None:
        return {"ref": ref, "missing": True}
    packet = {
        "ref": ref,
        "doc_id": record.doc_id,
        "region_id": record.region_id,
        "source_path": record.source_path,
        "hash": record.hash,
        "ordinal": record.ordinal,
    }
    if include_text:
        text = " ".join(record.text.split())
        if max_text_chars > 0 and len(text) > max_text_chars:
            text = text[: max_text_chars - 3].rstrip() + "..."
        packet["text"] = text
    return packet


def _severity_rank(severity: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(severity, 3)
