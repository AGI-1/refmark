"""Question-planning helpers for evidence retrieval evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable

from refmark.discovery import DiscoveryManifest, build_discovery_context_card, discovery_excluded_refs
from refmark.pipeline import RegionRecord


@dataclass(frozen=True)
class QuestionPlanItem:
    stable_ref: str
    query_style: str
    count: int = 1
    gold_refs: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)
    guidance: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_question_plan(
    discovery: DiscoveryManifest,
    records: Iterable[RegionRecord],
    *,
    direct_per_region: int = 1,
    concern_per_region: int = 1,
    adversarial_per_region: int = 1,
    include_excluded: bool = False,
) -> list[QuestionPlanItem]:
    """Build an inspectable per-ref plan before generating eval questions.

    The plan is deliberately small: it tells a generator which query styles to
    produce, while preserving the same stable gold refs used by evaluation.
    """

    excluded_refs = discovery_excluded_refs(discovery)
    items: list[QuestionPlanItem] = []
    records_list = list(records)
    for record in records_list:
        stable_ref = _stable_ref(record)
        if stable_ref in excluded_refs and not include_excluded:
            continue
        card = build_discovery_context_card(discovery, record, records=records_list)
        roles = list(card.roles)
        terms = list(card.terms)
        style_counts = {
            "direct": direct_per_region,
            "concern": concern_per_region,
            "adversarial": adversarial_per_region,
        }
        if {"navigation_only", "boilerplate"} & set(roles):
            style_counts = {"direct": max(direct_per_region, 1), "concern": 0, "adversarial": 0}
        for style, count in style_counts.items():
            if count <= 0:
                continue
            items.append(
                QuestionPlanItem(
                    stable_ref=stable_ref,
                    query_style=style,
                    count=count,
                    gold_refs=[stable_ref],
                    roles=roles,
                    terms=terms,
                    guidance=_style_guidance(style, roles, terms),
                )
            )
    return items


def question_plan_to_dict(plan: Iterable[QuestionPlanItem]) -> dict[str, object]:
    items = list(plan)
    return {
        "schema": "refmark.question_plan.v1",
        "items": [item.to_dict() for item in items],
        "summary": {
            "items": len(items),
            "questions": sum(item.count for item in items),
            "by_style": _count_by_style(items),
        },
    }


def _style_guidance(style: str, roles: list[str], terms: list[str]) -> list[str]:
    topic_hint = f"Use these terms when natural: {', '.join(terms[:4])}." if terms else "Use the section's concrete topic."
    if style == "direct":
        return [
            "Ask a straightforward lookup question that shares normal terminology with the source.",
            topic_hint,
        ]
    if style == "concern":
        return [
            "Ask from a user's problem, goal, or symptom rather than from the section title.",
            "Prefer natural wording a user might type into documentation search.",
            topic_hint,
        ]
    if style == "adversarial":
        return [
            "Use a valid paraphrase with lower lexical overlap, while keeping the answer unambiguous.",
            "Do not mention the ref id or copy the heading verbatim.",
            topic_hint,
        ]
    if "definition" in roles:
        return ["Ask what the term means and why this section is the right evidence.", topic_hint]
    return ["Ask a natural retrieval question for this evidence region.", topic_hint]


def _count_by_style(items: list[QuestionPlanItem]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item.query_style] = counts.get(item.query_style, 0) + item.count
    return dict(sorted(counts.items()))


def _stable_ref(record: RegionRecord) -> str:
    return f"{record.doc_id}:{record.region_id}"
