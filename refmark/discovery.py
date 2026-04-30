"""Corpus discovery helpers for evidence-retrieval evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable
from urllib import request

from refmark.pipeline import RegionRecord
from refmark.search_index import OPENROUTER_CHAT_URL, approx_tokens, classify_region_roles


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
ABBREVIATION_RE = re.compile(r"\b[A-Z][A-Z0-9-]{2,}\b")
GENERIC_TERMS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "been",
    "before",
    "being",
    "can",
    "does",
    "each",
    "find",
    "for",
    "from",
    "has",
    "have",
    "how",
    "include",
    "into",
    "may",
    "must",
    "not",
    "osha",
    "other",
    "passage",
    "paragraph",
    "provide",
    "rule",
    "section",
    "shall",
    "should",
    "standard",
    "support",
    "that",
    "the",
    "their",
    "this",
    "under",
    "what",
    "when",
    "where",
    "which",
    "with",
}


@dataclass(frozen=True)
class DiscoveryTerm:
    term: str
    count: int
    refs: list[str] = field(default_factory=list)
    kind: str = "term"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegionDiscovery:
    stable_ref: str
    roles: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)
    summary: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RangeCandidate:
    refs: list[str]
    reason: str
    kind: str = "adjacent"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QueryFamily:
    name: str
    refs: list[str]
    terms: list[str]
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiscoveryWindow:
    window_id: str
    refs: list[str]
    tokens: int
    doc_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiscoveryCluster:
    cluster_id: str
    name: str
    refs: list[str]
    terms: list[str] = field(default_factory=list)
    source: str = "deterministic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiscoveryContextCard:
    stable_ref: str
    corpus_summary: str
    region_summary: str
    roles: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)
    abbreviations: list[str] = field(default_factory=list)
    query_families: list[str] = field(default_factory=list)
    range_candidates: list[list[str]] = field(default_factory=list)
    neighboring_refs: list[str] = field(default_factory=list)
    parent_ref: str | None = None
    generation_guidance: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiscoveryReviewIssue:
    kind: str
    severity: str
    refs: list[str]
    summary: str
    recommendation: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiscoveryManifest:
    schema: str
    created_at: str
    mode: str
    source: str
    model: str
    corpus_summary: str
    corpus_tokens: int
    regions: int
    terms: list[DiscoveryTerm] = field(default_factory=list)
    abbreviations: list[DiscoveryTerm] = field(default_factory=list)
    region_roles: list[RegionDiscovery] = field(default_factory=list)
    range_candidates: list[RangeCandidate] = field(default_factory=list)
    query_families: list[QueryFamily] = field(default_factory=list)
    windows: list[DiscoveryWindow] = field(default_factory=list)
    clusters: list[DiscoveryCluster] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["terms"] = [item.to_dict() for item in self.terms]
        payload["abbreviations"] = [item.to_dict() for item in self.abbreviations]
        payload["region_roles"] = [item.to_dict() for item in self.region_roles]
        payload["range_candidates"] = [item.to_dict() for item in self.range_candidates]
        payload["query_families"] = [item.to_dict() for item in self.query_families]
        payload["windows"] = [item.to_dict() for item in self.windows]
        payload["clusters"] = [item.to_dict() for item in self.clusters]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DiscoveryManifest":
        return cls(
            schema=str(payload.get("schema", "refmark.discovery.v1")),
            created_at=str(payload.get("created_at", "")),
            mode=str(payload.get("mode", "whole")),
            source=str(payload.get("source", "local")),
            model=str(payload.get("model", "local")),
            corpus_summary=str(payload.get("corpus_summary", "")),
            corpus_tokens=int(payload.get("corpus_tokens", 0)),
            regions=int(payload.get("regions", 0)),
            terms=[DiscoveryTerm(**item) for item in payload.get("terms", [])],
            abbreviations=[DiscoveryTerm(**item) for item in payload.get("abbreviations", [])],
            region_roles=[RegionDiscovery(**item) for item in payload.get("region_roles", [])],
            range_candidates=[RangeCandidate(**item) for item in payload.get("range_candidates", [])],
            query_families=[QueryFamily(**item) for item in payload.get("query_families", [])],
            windows=[DiscoveryWindow(**item) for item in payload.get("windows", [])],
            clusters=[DiscoveryCluster(**item) for item in payload.get("clusters", [])],
            notes=[str(item) for item in payload.get("notes", [])],
        )


def discover_corpus(
    records: Iterable[RegionRecord],
    *,
    mode: str = "whole",
    source: str = "local",
    model: str = "local",
    endpoint: str = OPENROUTER_CHAT_URL,
    api_key_env: str = "OPENROUTER_API_KEY",
    max_input_tokens: int = 180_000,
    window_tokens: int | None = None,
    overlap_regions: int = 1,
) -> DiscoveryManifest:
    """Create a corpus-level discovery manifest.

    ``source="local"`` is deterministic and meant for CI/tests. ``source`` set
    to ``openrouter`` asks a model for the high-level summary/roles but keeps a
    local fallback so discovery never blocks the pipeline.
    """

    items = list(records)
    if mode not in {"whole", "hierarchical", "windowed"}:
        raise ValueError(f"unsupported discovery mode: {mode}")
    if source not in {"local", "openrouter"}:
        raise ValueError(f"unsupported discovery source: {source}")
    if mode == "windowed" or (window_tokens is not None and window_tokens > 0 and sum(approx_tokens(record.text) for record in items) > window_tokens):
        return discover_corpus_windowed(
            items,
            source=source,
            model=model,
            endpoint=endpoint,
            api_key_env=api_key_env,
            max_input_tokens=max_input_tokens,
            window_tokens=window_tokens or max_input_tokens,
            overlap_regions=overlap_regions,
        )
    if source == "openrouter":
        try:
            return _openrouter_discovery(
                items,
                mode=mode,
                model=model,
                endpoint=endpoint,
                api_key_env=api_key_env,
                max_input_tokens=max_input_tokens,
            )
        except Exception as exc:
            fallback = _local_discovery(items, mode=mode, model="local-fallback")
            return DiscoveryManifest(
                schema=fallback.schema,
                created_at=fallback.created_at,
                mode=fallback.mode,
                source="local-fallback",
                model="local-fallback",
                corpus_summary=fallback.corpus_summary,
                corpus_tokens=fallback.corpus_tokens,
                regions=fallback.regions,
                terms=fallback.terms,
                abbreviations=fallback.abbreviations,
                region_roles=fallback.region_roles,
                range_candidates=fallback.range_candidates,
                query_families=fallback.query_families,
                notes=[*fallback.notes, f"OpenRouter discovery failed: {exc}"],
            )
    return _local_discovery(items, mode=mode, model=model)


def discover_corpus_windowed(
    records: Iterable[RegionRecord],
    *,
    source: str = "local",
    model: str = "local",
    endpoint: str = OPENROUTER_CHAT_URL,
    api_key_env: str = "OPENROUTER_API_KEY",
    max_input_tokens: int = 180_000,
    window_tokens: int = 40_000,
    overlap_regions: int = 1,
) -> DiscoveryManifest:
    """Discover a large corpus in windows without cutting through regions."""

    items = list(records)
    windows = build_discovery_windows(items, window_tokens=window_tokens, overlap_regions=overlap_regions)
    manifests: list[DiscoveryManifest] = []
    for window in windows:
        window_records = [record for record in items if _stable_ref(record) in set(window.refs)]
        if source == "openrouter":
            try:
                manifests.append(
                    _openrouter_discovery(
                        window_records,
                        mode="windowed",
                        model=model,
                        endpoint=endpoint,
                        api_key_env=api_key_env,
                        max_input_tokens=max_input_tokens,
                    )
                )
            except Exception as exc:
                fallback = _local_discovery(window_records, mode="windowed", model="local-fallback")
                manifests.append(
                    DiscoveryManifest(
                        schema=fallback.schema,
                        created_at=fallback.created_at,
                        mode=fallback.mode,
                        source="local-fallback",
                        model="local-fallback",
                        corpus_summary=fallback.corpus_summary,
                        corpus_tokens=fallback.corpus_tokens,
                        regions=fallback.regions,
                        terms=fallback.terms,
                        abbreviations=fallback.abbreviations,
                        region_roles=fallback.region_roles,
                        range_candidates=fallback.range_candidates,
                        query_families=fallback.query_families,
                        notes=[*fallback.notes, f"OpenRouter window discovery failed for {window.window_id}: {exc}"],
                    )
                )
        else:
            manifests.append(_local_discovery(window_records, mode="windowed", model=model))
    return merge_discovery_manifests(manifests, windows=windows, records=items, source=source, model=model)


def build_discovery_windows(
    records: Iterable[RegionRecord],
    *,
    window_tokens: int,
    overlap_regions: int = 1,
) -> list[DiscoveryWindow]:
    """Pack records into token-budget windows; records are never split."""

    items = list(records)
    if window_tokens <= 0:
        raise ValueError("window_tokens must be positive")
    windows: list[DiscoveryWindow] = []
    start = 0
    while start < len(items):
        refs: list[str] = []
        doc_ids: set[str] = set()
        total = 0
        end = start
        while end < len(items):
            record = items[end]
            cost = max(1, approx_tokens(record.text))
            if refs and total + cost > window_tokens:
                break
            refs.append(_stable_ref(record))
            doc_ids.add(record.doc_id)
            total += cost
            end += 1
            if total >= window_tokens:
                break
        if not refs:
            record = items[start]
            refs = [_stable_ref(record)]
            doc_ids = {record.doc_id}
            total = max(1, approx_tokens(record.text))
            end = start + 1
        windows.append(
            DiscoveryWindow(
                window_id=f"W{len(windows) + 1:04d}",
                refs=refs,
                tokens=total,
                doc_ids=sorted(doc_ids),
            )
        )
        if end >= len(items):
            break
        start = max(start + 1, end - max(0, overlap_regions))
    return windows


def merge_discovery_manifests(
    manifests: Iterable[DiscoveryManifest],
    *,
    windows: list[DiscoveryWindow],
    records: list[RegionRecord],
    source: str,
    model: str,
) -> DiscoveryManifest:
    """Merge per-window discovery outputs while preserving refs and provenance."""

    items = list(manifests)
    terms = _merge_terms([term for manifest in items for term in manifest.terms])
    abbreviations = _merge_terms([term for manifest in items for term in manifest.abbreviations], kind="abbreviation")
    region_roles = _merge_region_roles([role for manifest in items for role in manifest.region_roles])
    range_candidates = _merge_ranges([candidate for manifest in items for candidate in manifest.range_candidates])
    query_families = _merge_query_families([family for manifest in items for family in manifest.query_families])
    clusters = _build_discovery_clusters(records, terms)
    notes = [
        "Windowed discovery merged region-safe windows.",
        "Window-level discoveries preserve refs; global normalization is deterministic and conservative.",
        "Review the discovery output before treating clusters or broad query families as accepted taxonomy.",
    ]
    for manifest in items:
        notes.extend(manifest.notes)
    return DiscoveryManifest(
        schema="refmark.discovery.v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        mode="windowed",
        source=source,
        model=model,
        corpus_summary=_local_summary(records, terms),
        corpus_tokens=sum(approx_tokens(record.text) for record in records),
        regions=len(records),
        terms=terms,
        abbreviations=abbreviations,
        region_roles=region_roles,
        range_candidates=range_candidates,
        query_families=query_families,
        windows=windows,
        clusters=clusters,
        notes=_dedupe(notes),
    )


def load_discovery(path: str | Path) -> DiscoveryManifest:
    return DiscoveryManifest.from_dict(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def write_discovery(discovery: DiscoveryManifest, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(discovery.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def build_discovery_context_card(
    discovery: DiscoveryManifest,
    record: RegionRecord,
    *,
    records: Iterable[RegionRecord] = (),
    max_terms: int = 10,
    max_families: int = 5,
    max_ranges: int = 5,
) -> DiscoveryContextCard:
    """Build a compact generation card for a single target region.

    The card is intentionally small enough to pass into question-generation
    prompts. It preserves stable refs for every discovery hint so generation can
    use corpus-level context without turning into ungrounded prose summary.
    """

    stable_ref = _stable_ref(record)
    roles_by_ref = {item.stable_ref: item for item in discovery.region_roles}
    region = roles_by_ref.get(stable_ref)
    wanted_refs = {stable_ref}
    if record.prev_region_id:
        wanted_refs.add(f"{record.doc_id}:{record.prev_region_id}")
    if record.next_region_id:
        wanted_refs.add(f"{record.doc_id}:{record.next_region_id}")
    if record.parent_region_id:
        wanted_refs.add(f"{record.doc_id}:{record.parent_region_id}")

    terms = []
    if region:
        terms.extend(region.terms)
    terms.extend(discovery_terms_for_refs(discovery, wanted_refs, limit=max_terms * 2))
    abbreviations = [
        item.term
        for item in discovery.abbreviations
        if wanted_refs & set(item.refs)
    ]
    families = [
        family.name
        for family in discovery.query_families
        if stable_ref in family.refs or wanted_refs & set(family.refs)
    ]
    ranges = [
        candidate.refs
        for candidate in discovery.range_candidates
        if stable_ref in candidate.refs
    ]
    neighbor_refs = [
        ref
        for ref in [
            f"{record.doc_id}:{record.prev_region_id}" if record.prev_region_id else None,
            f"{record.doc_id}:{record.next_region_id}" if record.next_region_id else None,
        ]
        if ref
    ]
    roles = region.roles if region else _region_roles(record)
    guidance = _generation_guidance(roles, ranges, families)
    return DiscoveryContextCard(
        stable_ref=stable_ref,
        corpus_summary=discovery.corpus_summary,
        region_summary=region.summary if region and region.summary else _first_sentence(record.text),
        roles=roles,
        terms=_dedupe(terms)[:max_terms],
        abbreviations=_dedupe(abbreviations)[:max_terms],
        query_families=_dedupe(families)[:max_families],
        range_candidates=ranges[:max_ranges],
        neighboring_refs=neighbor_refs,
        parent_ref=f"{record.doc_id}:{record.parent_region_id}" if record.parent_region_id else None,
        generation_guidance=guidance,
    )


def review_discovery(
    discovery: DiscoveryManifest,
    *,
    records: Iterable[RegionRecord] = (),
    max_issues: int = 50,
) -> list[DiscoveryReviewIssue]:
    """Return deterministic review targets for noisy discovery output."""

    issues: list[DiscoveryReviewIssue] = []
    record_by_ref = {_stable_ref(record): record for record in records}
    region_by_ref = {item.stable_ref: item for item in discovery.region_roles}

    for window in discovery.windows:
        if not window.refs:
            issues.append(
                DiscoveryReviewIssue(
                    kind="empty_window",
                    severity="high",
                    refs=[],
                    summary=f"Discovery window {window.window_id} has no refs.",
                    recommendation="Regenerate discovery windows before question planning.",
                    evidence=window.to_dict(),
                )
            )
        elif window.tokens > 0 and len(window.refs) == 1 and window.tokens >= 10_000:
            issues.append(
                DiscoveryReviewIssue(
                    kind="oversized_source_region",
                    severity="medium",
                    refs=window.refs,
                    summary=f"Discovery window {window.window_id} contains one very large region.",
                    recommendation="Review chunking density or split the source region before generating fine-grained questions.",
                    evidence=window.to_dict(),
                )
            )

    for item in discovery.terms:
        ref_count = len(set(item.refs))
        if ref_count >= max(12, int(discovery.regions * 0.2)):
            issues.append(
                DiscoveryReviewIssue(
                    kind="broad_term",
                    severity="medium",
                    refs=item.refs[:12],
                    summary=f"Term '{item.term}' appears across {ref_count} refs and may be too broad for question planning.",
                    recommendation="Review as a corpus/global term or add it to a domain stoplist if it creates generic questions.",
                    evidence={"term": item.term, "count": item.count, "ref_count": ref_count},
                )
            )
        elif item.count <= 1 and ref_count <= 1:
            issues.append(
                DiscoveryReviewIssue(
                    kind="singleton_term",
                    severity="low",
                    refs=item.refs[:3],
                    summary=f"Term '{item.term}' appears once; it may be useful jargon or local noise.",
                    recommendation="Keep only if it is domain-specific and useful for user queries.",
                    evidence={"term": item.term, "count": item.count},
                )
            )

    for region in discovery.region_roles:
        roles = set(region.roles)
        text = record_by_ref.get(region.stable_ref).text if region.stable_ref in record_by_ref else ""
        if "heading" in roles and len(text.split()) > 40:
            issues.append(
                DiscoveryReviewIssue(
                    kind="heading_detection",
                    severity="medium",
                    refs=[region.stable_ref],
                    summary="Heading role was assigned to a relatively long region.",
                    recommendation="Check whether heading detection split the document at the right boundary.",
                    evidence={"roles": region.roles, "word_count": len(text.split())},
                )
            )
        if {"navigation_only", "exclude_from_training"} & roles:
            issues.append(
                DiscoveryReviewIssue(
                    kind="excluded_region",
                    severity="low",
                    refs=[region.stable_ref],
                    summary="Region is marked navigation/boilerplate and excluded from training.",
                    recommendation="Confirm this should not receive normal generated questions.",
                    evidence={"roles": region.roles, "summary": region.summary},
                )
            )

    for candidate in discovery.range_candidates:
        if len(candidate.refs) > 6:
            issues.append(
                DiscoveryReviewIssue(
                    kind="large_range_candidate",
                    severity="medium",
                    refs=candidate.refs[:12],
                    summary=f"Range candidate spans {len(candidate.refs)} refs.",
                    recommendation="Review whether this should become a parent/section target rather than a single evidence range.",
                    evidence={"kind": candidate.kind, "reason": candidate.reason},
                )
            )
        missing_refs = [ref for ref in candidate.refs if ref not in region_by_ref and record_by_ref and ref not in record_by_ref]
        if missing_refs:
            issues.append(
                DiscoveryReviewIssue(
                    kind="stale_or_unknown_ref",
                    severity="high",
                    refs=missing_refs,
                    summary="Discovery range candidate references refs not present in the current manifest.",
                    recommendation="Regenerate or repair discovery before question generation.",
                    evidence={"candidate_refs": candidate.refs},
                )
            )

    for family in discovery.query_families:
        if len(set(family.refs)) >= max(10, int(discovery.regions * 0.15)):
            issues.append(
                DiscoveryReviewIssue(
                    kind="broad_query_family",
                    severity="medium",
                    refs=family.refs[:12],
                    summary=f"Query family '{family.name}' spans many refs and may create query magnets.",
                    recommendation="Split this family into narrower clusters before generating questions.",
                    evidence={"name": family.name, "terms": family.terms, "ref_count": len(set(family.refs))},
                )
            )

    covered_refs = {ref for cluster in discovery.clusters for ref in cluster.refs}
    expected_refs = set(record_by_ref) if record_by_ref else {role.stable_ref for role in discovery.region_roles}
    unclustered = sorted(expected_refs - covered_refs)
    if unclustered:
        issues.append(
            DiscoveryReviewIssue(
                kind="unclustered_regions",
                severity="medium",
                refs=unclustered[:20],
                summary=f"{len(unclustered)} refs are not assigned to any discovery cluster.",
                recommendation="Review cluster synthesis before using clusters for navigation or board-level heatmaps.",
                evidence={"count": len(unclustered)},
            )
        )

    for cluster in discovery.clusters:
        if len(cluster.refs) >= max(30, int(discovery.regions * 0.25)):
            issues.append(
                DiscoveryReviewIssue(
                    kind="broad_cluster",
                    severity="medium",
                    refs=cluster.refs[:20],
                    summary=f"Cluster '{cluster.name}' covers {len(cluster.refs)} refs.",
                    recommendation="Consider splitting this cluster before using it as a navigation/eval planning layer.",
                    evidence={"cluster_id": cluster.cluster_id, "terms": cluster.terms[:12]},
                )
            )

    severity_order = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda issue: (severity_order.get(issue.severity, 9), issue.kind, issue.refs))
    return issues[:max_issues]


def discovery_terms_for_refs(discovery: DiscoveryManifest, refs: Iterable[str], *, limit: int = 8) -> list[str]:
    wanted = set(refs)
    terms: list[str] = []
    for item in discovery.region_roles:
        if item.stable_ref in wanted:
            terms.extend(item.terms)
    for item in discovery.terms:
        if wanted & set(item.refs):
            terms.append(item.term)
    for item in discovery.abbreviations:
        if wanted & set(item.refs):
            terms.append(item.term)
    return _dedupe(terms)[:limit]


def discovery_excluded_refs(discovery: DiscoveryManifest) -> set[str]:
    return {
        item.stable_ref
        for item in discovery.region_roles
        if {"exclude_from_training", "navigation_only", "boilerplate"} & set(item.roles)
    }


def _local_discovery(records: list[RegionRecord], *, mode: str, model: str) -> DiscoveryManifest:
    stable_refs = [_stable_ref(record) for record in records]
    token_counts = Counter()
    token_refs: dict[str, set[str]] = {}
    abbreviation_counts = Counter()
    abbreviation_refs: dict[str, set[str]] = {}
    region_roles: list[RegionDiscovery] = []
    range_candidates: list[RangeCandidate] = []
    query_families: list[QueryFamily] = []

    for record in records:
        stable_ref = _stable_ref(record)
        tokens = _important_tokens(record.text)
        token_counts.update(tokens)
        for token in set(tokens):
            token_refs.setdefault(token, set()).add(stable_ref)
        abbreviations = ABBREVIATION_RE.findall(record.text)
        abbreviation_counts.update(abbreviations)
        for abbreviation in set(abbreviations):
            abbreviation_refs.setdefault(abbreviation, set()).add(stable_ref)
        roles = _region_roles(record)
        region_roles.append(
            RegionDiscovery(
                stable_ref=stable_ref,
                roles=roles,
                terms=_distinctive_terms(tokens, limit=8),
                summary=_first_sentence(record.text),
            )
        )

    terms = [
        DiscoveryTerm(term=term, count=count, refs=sorted(token_refs.get(term, []))[:12])
        for term, count in token_counts.most_common(40)
        if len(token_refs.get(term, [])) >= 1
    ]
    abbreviations = [
        DiscoveryTerm(term=term, count=count, refs=sorted(abbreviation_refs.get(term, []))[:12], kind="abbreviation")
        for term, count in abbreviation_counts.most_common(30)
    ]

    for index, record in enumerate(records[:-1]):
        current_terms = set(_important_tokens(record.text))
        next_terms = set(_important_tokens(records[index + 1].text))
        overlap = current_terms & next_terms
        if len(overlap) >= 2 or (record.parent_region_id and record.parent_region_id == records[index + 1].parent_region_id):
            range_candidates.append(
                RangeCandidate(
                    refs=[_stable_ref(record), _stable_ref(records[index + 1])],
                    reason="adjacent regions share terms or parent context",
                )
            )
        if len(range_candidates) >= 80:
            break

    for term in [item.term for item in terms[:12]]:
        refs = sorted(token_refs.get(term, []))[:8]
        query_families.append(QueryFamily(name=f"questions about {term}", refs=refs, terms=[term]))

    return DiscoveryManifest(
        schema="refmark.discovery.v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        mode=mode,
        source="local",
        model=model,
        corpus_summary=_local_summary(records, terms),
        corpus_tokens=sum(approx_tokens(record.text) for record in records),
        regions=len(records),
        terms=terms,
        abbreviations=abbreviations,
        region_roles=region_roles,
        range_candidates=range_candidates,
        query_families=query_families,
        notes=[
            "Local discovery is deterministic and heuristic.",
            "Use an LLM discovery pass for stronger section summaries and role labels.",
            "Hierarchical mode currently emits the same schema; large-corpus windowing is planned.",
        ],
    )


def _openrouter_discovery(
    records: list[RegionRecord],
    *,
    mode: str,
    model: str,
    endpoint: str,
    api_key_env: str,
    max_input_tokens: int,
) -> DiscoveryManifest:
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set")
    packed = _pack_records_for_prompt(records, max_input_tokens=max_input_tokens)
    prompt = f"""Discover retrieval-evaluation structure for this refmarked corpus.

Return strict JSON with keys:
- corpus_summary: short summary
- terms: list of {{"term": str, "kind": "term", "refs": [stable refs]}}
- abbreviations: list of {{"term": str, "kind": "abbreviation", "refs": [stable refs]}}
- region_roles: list of {{"stable_ref": str, "roles": [str], "terms": [str], "summary": str, "notes": str}}
- range_candidates: list of {{"refs": [stable refs], "kind": "adjacent|distributed|section", "reason": str}}
- query_families: list of {{"name": str, "refs": [stable refs], "terms": [str], "notes": str}}
- notes: list of strings

Prefer practical training/evaluation roles:
content, definition, procedure, exception, example, table, heading,
summary_region, navigation_only, boilerplate, exclude_from_training,
range_candidate, distributed_candidate.

Corpus mode: {mode}
Regions:
{packed}
"""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You analyze refmarked corpora for retrieval evaluation and training data preparation."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 5000,
    }
    req = request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/b-imenitov/refmark",
            "X-Title": "refmark discovery",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=180) as response:
        payload = json.loads(response.read().decode("utf-8"))
    content = payload["choices"][0]["message"]["content"].strip()
    parsed = _parse_json_object(content)
    local = _local_discovery(records, mode=mode, model="local-seed")
    return DiscoveryManifest(
        schema="refmark.discovery.v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        mode=mode,
        source="openrouter",
        model=model,
        corpus_summary=str(parsed.get("corpus_summary") or local.corpus_summary),
        corpus_tokens=local.corpus_tokens,
        regions=len(records),
        terms=_terms_from_payload(parsed.get("terms"), fallback=local.terms),
        abbreviations=_terms_from_payload(parsed.get("abbreviations"), fallback=local.abbreviations, kind="abbreviation"),
        region_roles=_region_roles_from_payload(parsed.get("region_roles"), fallback=local.region_roles),
        range_candidates=_range_candidates_from_payload(parsed.get("range_candidates"), fallback=local.range_candidates),
        query_families=_query_families_from_payload(parsed.get("query_families"), fallback=local.query_families),
        notes=[str(item) for item in parsed.get("notes", [])] or local.notes,
    )


def _pack_records_for_prompt(records: list[RegionRecord], *, max_input_tokens: int) -> str:
    chunks: list[str] = []
    total = 0
    for record in records:
        text = record.text.strip()
        cost = approx_tokens(text) + 8
        if chunks and total + cost > max_input_tokens:
            chunks.append("[TRUNCATED: corpus exceeded configured discovery token budget]")
            break
        chunks.append(f"[{_stable_ref(record)}]\n{text}")
        total += cost
    return "\n\n".join(chunks)


def _parse_json_object(content: str) -> dict[str, Any]:
    if content.startswith("```"):
        content = content.strip("`").removeprefix("json").strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end < start:
            raise
        parsed = json.loads(content[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("discovery model did not return a JSON object")
    return parsed


def _terms_from_payload(payload: Any, *, fallback: list[DiscoveryTerm], kind: str = "term") -> list[DiscoveryTerm]:
    if not isinstance(payload, list):
        return fallback
    terms: list[DiscoveryTerm] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        term = str(item.get("term", "")).strip()
        if not term:
            continue
        terms.append(
            DiscoveryTerm(
                term=term,
                count=int(item.get("count", 0) or 0),
                refs=[str(ref) for ref in item.get("refs", [])],
                kind=str(item.get("kind", kind)),
            )
        )
    return terms or fallback


def _region_roles_from_payload(payload: Any, *, fallback: list[RegionDiscovery]) -> list[RegionDiscovery]:
    if not isinstance(payload, list):
        return fallback
    roles: list[RegionDiscovery] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        stable_ref = str(item.get("stable_ref", "")).strip()
        if not stable_ref:
            continue
        roles.append(
            RegionDiscovery(
                stable_ref=stable_ref,
                roles=[str(value) for value in item.get("roles", [])],
                terms=[str(value) for value in item.get("terms", [])],
                summary=str(item.get("summary", "")),
                notes=str(item.get("notes", "")),
            )
        )
    return roles or fallback


def _range_candidates_from_payload(payload: Any, *, fallback: list[RangeCandidate]) -> list[RangeCandidate]:
    if not isinstance(payload, list):
        return fallback
    ranges: list[RangeCandidate] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        refs = [str(value) for value in item.get("refs", []) if str(value).strip()]
        if not refs:
            continue
        ranges.append(RangeCandidate(refs=refs, reason=str(item.get("reason", "")), kind=str(item.get("kind", "adjacent"))))
    return ranges or fallback


def _query_families_from_payload(payload: Any, *, fallback: list[QueryFamily]) -> list[QueryFamily]:
    if not isinstance(payload, list):
        return fallback
    families: list[QueryFamily] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        families.append(
            QueryFamily(
                name=name,
                refs=[str(value) for value in item.get("refs", [])],
                terms=[str(value) for value in item.get("terms", [])],
                notes=str(item.get("notes", "")),
            )
        )
    return families or fallback


def _merge_terms(items: list[DiscoveryTerm], *, kind: str = "term") -> list[DiscoveryTerm]:
    grouped: dict[str, dict[str, Any]] = {}
    for item in items:
        key = _normalize_discovery_term(item.term)
        if not key:
            continue
        row = grouped.setdefault(key, {"term": item.term, "count": 0, "refs": set(), "kind": item.kind or kind})
        row["count"] += int(item.count)
        row["refs"].update(item.refs)
        if len(item.term) < len(str(row["term"])):
            row["term"] = item.term
    merged = [
        DiscoveryTerm(term=str(row["term"]), count=int(row["count"]), refs=sorted(row["refs"]), kind=str(row["kind"]))
        for row in grouped.values()
    ]
    merged.sort(key=lambda item: (-item.count, item.term.lower()))
    return merged


def _merge_region_roles(items: list[RegionDiscovery]) -> list[RegionDiscovery]:
    grouped: dict[str, dict[str, Any]] = {}
    for item in items:
        row = grouped.setdefault(item.stable_ref, {"roles": [], "terms": [], "summary": "", "notes": []})
        row["roles"].extend(item.roles)
        row["terms"].extend(item.terms)
        if item.summary and (not row["summary"] or len(item.summary) > len(row["summary"])):
            row["summary"] = item.summary
        if item.notes:
            row["notes"].append(item.notes)
    merged = [
        RegionDiscovery(
            stable_ref=stable_ref,
            roles=_dedupe(row["roles"]),
            terms=_dedupe(row["terms"])[:12],
            summary=str(row["summary"]),
            notes=" | ".join(_dedupe(row["notes"])[:3]),
        )
        for stable_ref, row in grouped.items()
    ]
    merged.sort(key=lambda item: item.stable_ref)
    return merged


def _merge_ranges(items: list[RangeCandidate]) -> list[RangeCandidate]:
    grouped: dict[tuple[str, ...], RangeCandidate] = {}
    for item in items:
        key = tuple(item.refs)
        if key not in grouped:
            grouped[key] = item
    merged = list(grouped.values())
    merged.sort(key=lambda item: (item.refs, item.kind))
    return merged


def _merge_query_families(items: list[QueryFamily]) -> list[QueryFamily]:
    grouped: dict[str, dict[str, Any]] = {}
    for item in items:
        key = _normalize_discovery_term(item.name)
        row = grouped.setdefault(key, {"name": item.name, "refs": set(), "terms": [], "notes": []})
        row["refs"].update(item.refs)
        row["terms"].extend(item.terms)
        if item.notes:
            row["notes"].append(item.notes)
    merged = [
        QueryFamily(
            name=str(row["name"]),
            refs=sorted(row["refs"]),
            terms=_dedupe(row["terms"])[:12],
            notes=" | ".join(_dedupe(row["notes"])[:3]),
        )
        for row in grouped.values()
    ]
    merged.sort(key=lambda item: (-len(item.refs), item.name.lower()))
    return merged


def _build_discovery_clusters(records: list[RegionRecord], terms: list[DiscoveryTerm]) -> list[DiscoveryCluster]:
    by_doc: dict[str, list[str]] = {}
    for record in records:
        by_doc.setdefault(record.doc_id, []).append(_stable_ref(record))
    term_by_ref: dict[str, list[str]] = {}
    for term in terms:
        for ref in term.refs:
            term_by_ref.setdefault(ref, []).append(term.term)
    clusters = []
    for doc_id, refs in sorted(by_doc.items()):
        cluster_terms: list[str] = []
        for ref in refs:
            cluster_terms.extend(term_by_ref.get(ref, [])[:3])
        clusters.append(
            DiscoveryCluster(
                cluster_id=f"doc:{doc_id}",
                name=doc_id,
                refs=refs,
                terms=_dedupe(cluster_terms)[:12],
                source="doc_id",
            )
        )
    return clusters


def _normalize_discovery_term(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"[^a-z0-9_-]+", " ", lowered)
    return " ".join(part for part in lowered.split() if part not in GENERIC_TERMS)


def _generation_guidance(roles: list[str], ranges: list[list[str]], families: list[str]) -> list[str]:
    role_set = set(roles)
    guidance: list[str] = []
    if {"navigation_only", "boilerplate", "exclude_from_training"} & role_set:
        guidance.append("avoid normal eval questions unless explicitly testing navigation/query-magnet behavior")
    if "definition" in role_set:
        guidance.append("include at least one definition-style query")
    if "obligation" in role_set:
        guidance.append("include at least one requirement or compliance query")
    if "exception" in role_set:
        guidance.append("include at least one exception/edge-case query")
    if "example" in role_set:
        guidance.append("include at least one practical example or troubleshooting query")
    if ranges:
        guidance.append("consider range-style questions that require neighboring evidence")
    if families:
        guidance.append("align wording with the listed query families while avoiding generic repeats")
    return guidance or ["generate a mix of direct and user-intent questions grounded in this region"]


def _local_summary(records: list[RegionRecord], terms: list[DiscoveryTerm]) -> str:
    doc_ids = sorted({record.doc_id for record in records})
    top_terms = ", ".join(item.term for item in terms[:8])
    return f"{len(records)} regions across {len(doc_ids)} document(s). Frequent domain terms: {top_terms}."


def _region_roles(record: RegionRecord) -> list[str]:
    text = record.text.strip()
    lower = text.lower()
    roles: list[str] = list(classify_region_roles(record))
    looks_like_short_heading = (
        len(text.splitlines()) == 1
        and len(text.split()) <= 10
        and not re.search(r"[.;:]$", text)
        and " means " not in lower
    )
    if text.startswith("#") or looks_like_short_heading:
        roles.append("heading")
    if "table of contents" in lower or lower.startswith("contents"):
        roles.extend(["navigation_only", "exclude_from_training"])
    if len(ABBREVIATION_RE.findall(text)) >= 8 or _language_menu_score(text) >= 5:
        roles.extend(["boilerplate", "exclude_from_training"])
    if " means " in lower or lower.endswith(" means") or re.match(r"^[A-Z][A-Za-z -]+ means\b", text):
        roles.append("definition")
    if any(word in lower for word in ["shall", "must", "required", "requirement"]):
        roles.append("obligation")
    if any(word in lower for word in ["except", "unless", "exempt"]):
        roles.append("exception")
    if any(word in lower for word in ["for example", "e.g.", "example"]):
        roles.append("example")
    if "exclude_from_default_search" in roles:
        roles.append("exclude_from_training")
    return _dedupe(roles or ["content"])


def _language_menu_score(text: str) -> int:
    markers = ["english", "français", "kreyòl", "한국어", "繁体", "简体", "العربية", "tagalog"]
    lower = text.lower()
    return sum(1 for marker in markers if marker in lower)


def _important_tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 3 and token.lower() not in GENERIC_TERMS and not token.isdigit()
    ]


def _distinctive_terms(tokens: list[str], *, limit: int) -> list[str]:
    return [term for term, _count in Counter(tokens).most_common(limit)]


def _first_sentence(text: str) -> str:
    compact = " ".join(text.split())
    if len(compact) <= 180:
        return compact
    for delimiter in [". ", "; ", ": "]:
        index = compact.find(delimiter)
        if 30 <= index <= 180:
            return compact[: index + 1]
    return compact[:180].rstrip() + "..."


def _stable_ref(record: RegionRecord) -> str:
    return f"{record.doc_id}:{record.region_id}"


def _dedupe(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output
