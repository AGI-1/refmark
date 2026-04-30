"""Production query-feedback aggregation for Refmark evidence search."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable

from refmark.rag_eval import CorpusMap


@dataclass(frozen=True)
class FeedbackEvent:
    """One production search/query interaction.

    The event is intentionally provider-agnostic. A caller can log only a
    query and top refs, or include richer click/manual-selection feedback when
    available. Refmark treats these rows as review signals, not automatic gold.
    """

    query: str
    top_refs: list[str] = field(default_factory=list)
    clicked_ref: str | None = None
    selected_ref: str | None = None
    useful: bool | None = None
    no_answer: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "FeedbackEvent":
        top_refs = row.get("top_refs") or row.get("shown_refs") or row.get("refs") or []
        if not isinstance(top_refs, list):
            top_refs = []
        useful = row.get("useful")
        if useful is None and "feedback" in row:
            feedback = str(row.get("feedback", "")).lower()
            useful = True if feedback in {"positive", "useful", "thumbs_up", "accepted"} else False if feedback in {"negative", "not_useful", "thumbs_down"} else None
        return cls(
            query=str(row["query"]),
            top_refs=[str(ref) for ref in top_refs],
            clicked_ref=_optional_str(row.get("clicked_ref") or row.get("click_ref")),
            selected_ref=_optional_str(row.get("selected_ref") or row.get("manual_ref") or row.get("correct_ref")),
            useful=useful if isinstance(useful, bool) else None,
            no_answer=bool(row.get("no_answer", False)),
            metadata=dict(row.get("metadata", {})) if isinstance(row.get("metadata"), dict) else {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def target_ref(self) -> str | None:
        """Best available user-implied target ref, if any."""

        return self.selected_ref or (self.clicked_ref if self.useful is not False else None)


@dataclass(frozen=True)
class FeedbackCluster:
    query: str
    count: int
    top_refs: list[dict[str, Any]]
    target_refs: list[dict[str, Any]]
    no_answer_count: int
    negative_count: int
    missing_refs: list[str] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FeedbackReport:
    schema: str
    events: int
    unique_queries: int
    clusters: list[FeedbackCluster]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "events": self.events,
            "unique_queries": self.unique_queries,
            "summary": self.summary,
            "clusters": [cluster.to_dict() for cluster in self.clusters],
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def read_feedback_jsonl(path: str | Path) -> list[FeedbackEvent]:
    events: list[FeedbackEvent] = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Feedback row in {path} must be a JSON object.")
        events.append(FeedbackEvent.from_dict(row))
    return events


def analyze_feedback(
    events: Iterable[FeedbackEvent],
    *,
    corpus: CorpusMap | None = None,
    min_count: int = 2,
    top_n: int = 25,
) -> FeedbackReport:
    rows = list(events)
    grouped: dict[str, list[FeedbackEvent]] = defaultdict(list)
    for event in rows:
        grouped[_normalize_query(event.query)].append(event)

    clusters: list[FeedbackCluster] = []
    for normalized_query, query_events in grouped.items():
        if len(query_events) < min_count:
            continue
        clusters.append(_cluster(normalized_query, query_events, corpus=corpus))

    clusters.sort(key=lambda item: (_severity(item), item.query))
    clusters = clusters[:top_n]
    action_counts = Counter(action["action"] for cluster in clusters for action in cluster.actions)
    return FeedbackReport(
        schema="refmark.feedback_report.v1",
        events=len(rows),
        unique_queries=len(grouped),
        clusters=clusters,
        summary={
            "min_count": min_count,
            "reported_clusters": len(clusters),
            "action_counts": dict(sorted(action_counts.items())),
        },
    )


def _cluster(query: str, events: list[FeedbackEvent], *, corpus: CorpusMap | None) -> FeedbackCluster:
    top_counter: Counter[str] = Counter()
    target_counter: Counter[str] = Counter()
    no_answer_count = 0
    negative_count = 0
    missing_refs: set[str] = set()
    for event in events:
        if event.top_refs:
            top_counter[event.top_refs[0]] += 1
        if event.target_ref:
            target_counter[event.target_ref] += 1
        if event.no_answer:
            no_answer_count += 1
        if event.useful is False:
            negative_count += 1

    if corpus is not None:
        refs = [*top_counter, *target_counter]
        missing_refs.update(corpus.validate_refs(refs)["missing"])

    top_refs = _counter_rows(top_counter)
    target_refs = _counter_rows(target_counter)
    actions = _actions(
        query=query,
        count=len(events),
        top_counter=top_counter,
        target_counter=target_counter,
        no_answer_count=no_answer_count,
        negative_count=negative_count,
        missing_refs=sorted(missing_refs),
    )
    return FeedbackCluster(
        query=query,
        count=len(events),
        top_refs=top_refs,
        target_refs=target_refs,
        no_answer_count=no_answer_count,
        negative_count=negative_count,
        missing_refs=sorted(missing_refs),
        actions=actions,
    )


def _actions(
    *,
    query: str,
    count: int,
    top_counter: Counter[str],
    target_counter: Counter[str],
    no_answer_count: int,
    negative_count: int,
    missing_refs: list[str],
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if missing_refs:
        actions.append(
            {
                "action": "review_missing_refs",
                "adaptation_type": "feedback_validation",
                "refs": missing_refs,
                "reason": "Feedback references refs that are not present in the current corpus map.",
            }
        )

    likely_target, target_count = target_counter.most_common(1)[0] if target_counter else (None, 0)
    likely_top, top_count = top_counter.most_common(1)[0] if top_counter else (None, 0)
    if likely_target and likely_target != likely_top:
        actions.append(
            {
                "action": "add_shadow_alias_or_doc2query",
                "adaptation_type": "retrieval_metadata",
                "target_ref": likely_target,
                "query": query,
                "supporting_events": target_count,
                "competing_top_ref": likely_top,
                "reason": "Users repeatedly selected or clicked a ref that was not the dominant top result.",
            }
        )
        if likely_top:
            actions.append(
                {
                    "action": "record_confusion_pair",
                    "adaptation_type": "confusion_mapping",
                    "target_ref": likely_target,
                    "competing_ref": likely_top,
                    "supporting_events": min(target_count, top_count),
                    "reason": "Production feedback shows a repeated query-level confusion.",
                }
            )

    if likely_top and negative_count >= max(2, count // 2) and not target_counter:
        actions.append(
            {
                "action": "review_query_magnet",
                "adaptation_type": "data_smell",
                "target_ref": likely_top,
                "supporting_events": negative_count,
                "reason": "The same top ref receives repeated negative/no-click feedback without a better selected ref.",
            }
        )

    if no_answer_count >= max(2, count // 2):
        actions.append(
            {
                "action": "review_no_answer_or_missing_coverage",
                "adaptation_type": "coverage_gap",
                "query": query,
                "supporting_events": no_answer_count,
                "reason": "Users repeatedly indicated no answer for this query.",
            }
        )

    if len(target_counter) >= 2:
        actions.append(
            {
                "action": "review_ambiguous_query",
                "adaptation_type": "feedback_validation",
                "query": query,
                "target_refs": [ref for ref, _count in target_counter.most_common()],
                "reason": "Different users selected different target refs for the same normalized query.",
            }
        )
    return actions or [
        {
            "action": "track_feedback_cluster",
            "adaptation_type": "observation",
            "query": query,
            "reason": "Repeated feedback cluster did not meet an automatic diagnostic rule.",
        }
    ]


def _counter_rows(counter: Counter[str]) -> list[dict[str, Any]]:
    return [{"ref": ref, "count": count} for ref, count in counter.most_common()]


def _severity(cluster: FeedbackCluster) -> tuple[int, int, str]:
    priority = {
        "add_shadow_alias_or_doc2query": 0,
        "record_confusion_pair": 1,
        "review_query_magnet": 2,
        "review_no_answer_or_missing_coverage": 3,
        "review_ambiguous_query": 4,
        "review_missing_refs": 5,
    }
    best = min((priority.get(action["action"], 9) for action in cluster.actions), default=9)
    return (best, -cluster.count, cluster.query)


def _normalize_query(query: str) -> str:
    return " ".join(query.lower().split())


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
