"""Adaptation planning from Refmark data-smell reports."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AdaptationAction:
    """A reviewable action suggested by a data-smell report."""

    action: str
    adaptation_type: str
    priority: str
    review_required: bool
    target_refs: list[str] = field(default_factory=list)
    source_smell: str = ""
    rationale: str = ""
    proposed_changes: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AdaptationPlan:
    """A conservative plan for humans or agents to review before mutating metadata."""

    schema: str
    summary: dict[str, Any]
    actions: list[AdaptationAction]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "summary": self.summary,
            "actions": [action.to_dict() for action in self.actions],
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def read_smell_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Smell report must be a JSON object")
    if payload.get("schema") != "refmark.data_smells.v1":
        raise ValueError(f"Unsupported smell report schema: {payload.get('schema')!r}")
    return payload


def build_adaptation_plan(
    smell_report: dict[str, Any],
    *,
    max_actions: int = 80,
) -> AdaptationPlan:
    """Convert a `refmark.data_smells.v1` report into reviewable next actions."""

    smells = list(smell_report.get("smells", []))
    actions: list[AdaptationAction] = []
    for smell in smells:
        if not isinstance(smell, dict):
            continue
        actions.extend(_actions_for_smell(smell))
    actions = _dedupe_actions(actions)
    actions.sort(key=lambda action: (_priority_rank(action.priority), action.adaptation_type, ",".join(action.target_refs), action.action))
    actions = actions[:max_actions]
    counts = Counter(action.adaptation_type for action in actions)
    by_priority = Counter(action.priority for action in actions)
    return AdaptationPlan(
        schema="refmark.adaptation_plan.v1",
        summary={
            "source_schema": smell_report.get("schema"),
            "source_run_fingerprint": smell_report.get("summary", {}).get("run_fingerprint"),
            "source_corpus_fingerprint": smell_report.get("summary", {}).get("corpus_fingerprint"),
            "source_smell_count": smell_report.get("summary", {}).get("smell_count", len(smells)),
            "action_count": len(actions),
            "by_adaptation_type": dict(sorted(counts.items())),
            "by_priority": dict(sorted(by_priority.items())),
            "review_required": True,
        },
        actions=actions,
    )


def _actions_for_smell(smell: dict[str, Any]) -> list[AdaptationAction]:
    smell_type = str(smell.get("type", ""))
    refs = [str(ref) for ref in smell.get("refs", [])]
    severity = str(smell.get("severity", "medium"))
    priority = "high" if severity == "high" else "medium"
    evidence = dict(smell.get("evidence", {})) if isinstance(smell.get("evidence"), dict) else {}
    message = str(smell.get("message", ""))

    if smell_type == "stale_label":
        return [
            _action(
                "review_or_refresh_stale_eval_label",
                "lifecycle_validation",
                priority,
                refs,
                smell_type,
                message,
                ["review_current_source_hashes", "regenerate_or_reapprove_gold_refs", "preserve_unaffected_examples"],
                evidence,
            )
        ]
    if smell_type == "hard_ref":
        return [
            _action(
                "review_hard_ref_retrievability",
                "retrieval_metadata",
                priority,
                refs,
                smell_type,
                message,
                ["add_shadow_aliases_or_doc2query", "add_more_eval_queries_for_ref", "review_region_boundaries"],
                evidence,
            )
        ]
    if smell_type == "confusion_pair":
        return [
            _action(
                "record_confusion_pair",
                "confusion_mapping",
                priority,
                refs,
                smell_type,
                message,
                ["add_disambiguators_to_gold_and_competing_refs", "add_hard_negative_for_reranker", "mark_alternate_gold_if_valid"],
                evidence,
            )
        ]
    if smell_type == "query_style_gap":
        weakest = evidence.get("weakest_style", {})
        return [
            _action(
                "add_style_targeted_questions_or_aliases",
                "query_style_coverage",
                priority,
                refs,
                smell_type,
                f"{message} Weakest style: {weakest}",
                ["generate_more_weak_style_questions", "add_concern_or_adversarial_aliases", "track_style_metrics_separately"],
                evidence,
            )
        ]
    if smell_type == "gold_mode_gap":
        return [
            _action(
                "review_range_and_distributed_evidence_policy",
                "range_semantics",
                priority,
                refs,
                smell_type,
                message,
                ["add_range_eval_rows", "tune_context_expansion", "score_parent_or_neighbor_hits"],
                evidence,
            )
        ]
    if smell_type == "undercitation":
        return [
            _action(
                "increase_candidate_or_context_recall",
                "context_expansion",
                priority,
                refs[:20],
                smell_type,
                message,
                ["increase_candidate_k", "expand_neighbors_for_range_refs", "review_missing_gold_boundaries"],
                evidence,
            )
        ]
    if smell_type == "overcitation":
        return [
            _action(
                "tighten_overbroad_context",
                "context_expansion",
                priority,
                refs[:20],
                smell_type,
                message,
                ["reduce_neighbor_expansion", "split_broad_regions", "rerank_context_refs"],
                evidence,
            )
        ]
    if smell_type == "low_confidence":
        return [
            _action(
                "gate_low_confidence_auto_jumps",
                "confidence_gating",
                priority,
                refs[:20],
                smell_type,
                message,
                ["route_low_margin_queries_to_reranker", "show_multiple_candidates", "raise_auto_jump_threshold"],
                evidence,
            )
        ]
    if smell_type == "query_magnet":
        return [
            _action(
                "review_query_magnet_role",
                "corpus_roles",
                priority,
                refs,
                smell_type,
                message,
                ["mark_hub_or_query_magnet", "exclude_or_downweight_from_default_search", "add_disambiguating_metadata_to_neighbors"],
                evidence,
            )
        ]
    if smell_type == "duplicate_support":
        return [
            _action(
                "review_duplicate_or_alternate_support",
                "support_topology",
                priority,
                refs,
                smell_type,
                message,
                ["mark_alternate_support_refs", "merge_or_link_equivalent_sections", "deduplicate_eval_gold_refs"],
                evidence,
            )
        ]
    if smell_type == "contradictory_support":
        return [
            _action(
                "review_possible_contradiction",
                "corpus_consistency",
                priority,
                refs,
                smell_type,
                message,
                ["add_scope_or_date_metadata", "mark_superseded_or_deprecated_region", "route_to_subject_matter_review"],
                evidence,
            )
        ]
    if smell_type == "uncovered_region":
        return [
            _action(
                "add_or_exclude_eval_coverage",
                "coverage_planning",
                priority,
                refs[:20],
                smell_type,
                message,
                ["generate_eval_questions_for_uncovered_regions", "mark_low_value_regions_excluded_from_eval", "review_coverage_targets"],
                evidence,
            )
        ]
    return [
        _action(
            "review_data_smell",
            "manual_review",
            priority,
            refs,
            smell_type,
            message,
            [str(action) for action in smell.get("suggested_actions", [])],
            evidence,
        )
    ]


def _action(
    action: str,
    adaptation_type: str,
    priority: str,
    refs: list[str],
    source_smell: str,
    rationale: str,
    proposed_changes: list[str],
    evidence: dict[str, Any],
) -> AdaptationAction:
    return AdaptationAction(
        action=action,
        adaptation_type=adaptation_type,
        priority=priority,
        review_required=True,
        target_refs=refs,
        source_smell=source_smell,
        rationale=rationale,
        proposed_changes=proposed_changes,
        evidence=_trim_evidence(evidence),
    )


def _dedupe_actions(actions: list[AdaptationAction]) -> list[AdaptationAction]:
    result: list[AdaptationAction] = []
    seen: set[tuple[str, str, tuple[str, ...], str]] = set()
    for action in actions:
        key = (action.action, action.adaptation_type, tuple(action.target_refs), action.source_smell)
        if key in seen:
            continue
        seen.add(key)
        result.append(action)
    return result


def _trim_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Keep plan files small while preserving enough context for review."""

    result = dict(evidence)
    for key in ("samples", "sample_misses", "sample_queries"):
        value = result.get(key)
        if isinstance(value, list):
            result[key] = value[:5]
    return result


def _priority_rank(priority: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(priority, 3)
