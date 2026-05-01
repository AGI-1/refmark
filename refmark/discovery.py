"""Corpus discovery helpers for evidence-retrieval evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable
from urllib import request

from refmark.pipeline import RegionRecord
from refmark.search_index import OPENROUTER_CHAT_URL, approx_tokens, classify_region_roles


CLUSTER_STRATEGIES = {"doc_id", "source_tree", "tag_graph", "balanced_terms", "llm_topics", "llm_intents"}
LLM_CLUSTER_STRATEGIES = {"llm_topics", "llm_intents"}


# Unicode word tokenizer: starts with a letter, then allows letters/digits and
# hyphenated parts. This keeps terms like "Gläubiger" intact for non-English
# corpora while still filtering snake_case/file ids later.
TOKEN_RE = re.compile(r"[^\W\d_][^\W_]*(?:-[^\W_]+)*", re.UNICODE)
ABBREVIATION_RE = re.compile(r"\b[A-Z][A-Z0-9-]{2,}\b")
GENERIC_TERMS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "available",
    "been",
    "before",
    "being",
    "can",
    "combined",
    "code",
    "contain",
    "data",
    "document",
    "docs",
    "documentation",
    "does",
    "each",
    "example",
    "examples",
    "explain",
    "fastapi",
    "find",
    "for",
    "from",
    "github",
    "guide",
    "has",
    "have",
    "http",
    "https",
    "how",
    "include",
    "including",
    "into",
    "may",
    "more",
    "must",
    "not",
    "nonfiction",
    "osha",
    "other",
    "overview",
    "passage",
    "paragraph",
    "provide",
    "rule",
    "section",
    "shall",
    "should",
    "standard",
    "still",
    "support",
    "that",
    "the",
    "they",
    "their",
    "than",
    "this",
    "under",
    "what",
    "when",
    "where",
    "which",
    "will",
    "with",
    "would",
    "your",
    "pass",
    "read",
    "related",
    "set",
    "source",
    "such",
    "these",
    "toctree",
    "url",
    "used",
    "work",
    "even",
    "base",
    "currentmodule",
    "maxdepth",
    "several",
    "welcome",
    "aber",
    "alle",
    "allem",
    "allen",
    "aller",
    "als",
    "auch",
    "auf",
    "aus",
    "bei",
    "beim",
    "bis",
    "dass",
    "dem",
    "den",
    "der",
    "des",
    "die",
    "dies",
    "diese",
    "diesem",
    "diesen",
    "dieser",
    "durch",
    "ein",
    "eine",
    "einem",
    "einen",
    "einer",
    "eines",
    "erfolgt",
    "gegen",
    "gemäß",
    "für",
    "gegen\u00fcber",
    "gilt",
    "haben",
    "hat",
    "ist",
    "jedoch",
    "kann",
    "k\u00f6nnen",
    "mit",
    "nachdem",
    "nach",
    "nicht",
    "oder",
    "sich",
    "seiner",
    "sind",
    "soweit",
    "sowie",
    "dessen",
    "deren",
    "dabei",
    "darauf",
    "darf",
    "ohne",
    "eines",
    "anderen",
    "unter",
    "\u00fcber",
    "und",
    "von",
    "wenn",
    "werden",
    "wird",
    "zur",
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
    strategy: str = "doc_id"
    parent_id: str | None = None
    notes: str = ""

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
            clusters=[_cluster_from_payload(item) for item in payload.get("clusters", [])],
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
    cluster_strategy: str = "doc_id",
    target_clusters: int = 40,
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
    if cluster_strategy not in CLUSTER_STRATEGIES:
        raise ValueError(f"unsupported cluster strategy: {cluster_strategy}")
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
            cluster_strategy=cluster_strategy,
            target_clusters=target_clusters,
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
                cluster_strategy=cluster_strategy,
                target_clusters=target_clusters,
            )
        except Exception as exc:
            fallback = _local_discovery(
                items,
                mode=mode,
                model="local-fallback",
                cluster_strategy=cluster_strategy,
                target_clusters=target_clusters,
            )
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
                clusters=fallback.clusters,
                notes=[*fallback.notes, f"OpenRouter discovery failed: {exc}"],
            )
    return _local_discovery(items, mode=mode, model=model, cluster_strategy=cluster_strategy, target_clusters=target_clusters)


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
    cluster_strategy: str = "doc_id",
    target_clusters: int = 40,
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
                        cluster_strategy=cluster_strategy,
                        target_clusters=target_clusters,
                    )
                )
            except Exception as exc:
                fallback = _local_discovery(
                    window_records,
                    mode="windowed",
                    model="local-fallback",
                    cluster_strategy=cluster_strategy,
                    target_clusters=target_clusters,
                )
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
                        clusters=fallback.clusters,
                        notes=[*fallback.notes, f"OpenRouter window discovery failed for {window.window_id}: {exc}"],
                    )
                )
        else:
            manifests.append(
                _local_discovery(
                    window_records,
                    mode="windowed",
                    model=model,
                    cluster_strategy=cluster_strategy,
                    target_clusters=target_clusters,
                )
            )
    return merge_discovery_manifests(
        manifests,
        windows=windows,
        records=items,
        source=source,
        model=model,
        cluster_strategy=cluster_strategy,
        target_clusters=target_clusters,
    )


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
    cluster_strategy: str = "doc_id",
    target_clusters: int = 40,
) -> DiscoveryManifest:
    """Merge per-window discovery outputs while preserving refs and provenance."""

    items = list(manifests)
    terms = _merge_terms([term for manifest in items for term in manifest.terms])
    abbreviations = _merge_terms([term for manifest in items for term in manifest.abbreviations], kind="abbreviation")
    region_roles = _merge_region_roles([role for manifest in items for role in manifest.region_roles])
    range_candidates = _merge_ranges([candidate for manifest in items for candidate in manifest.range_candidates])
    query_families = _merge_query_families([family for manifest in items for family in manifest.query_families])
    effective_cluster_strategy = "tag_graph" if cluster_strategy in LLM_CLUSTER_STRATEGIES else cluster_strategy
    clusters = _build_discovery_clusters(records, terms, strategy=effective_cluster_strategy, target_clusters=target_clusters)
    notes = [
        "Windowed discovery merged region-safe windows.",
        "Window-level discoveries preserve refs; global normalization is deterministic and conservative.",
        "Review the discovery output before treating clusters or broad query families as accepted taxonomy.",
        f"Cluster strategy: {cluster_strategy}.",
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


def repair_discovery_clusters(
    discovery: DiscoveryManifest,
    records: Iterable[RegionRecord],
    *,
    source: str = "openrouter",
    model: str = "qwen/qwen-turbo",
    endpoint: str = OPENROUTER_CHAT_URL,
    api_key_env: str = "OPENROUTER_API_KEY",
    cluster_strategy: str | None = None,
    target_clusters: int | None = None,
    max_input_tokens: int = 40_000,
) -> DiscoveryManifest:
    """Ask a discovery agent to repair only the cluster layer.

    This is the tool form of the review-board loop: keep refs and local
    discovery stable, give the agent current clusters plus compact region cards,
    then sanitize its proposed cluster layer against the manifest.
    """

    items = list(records)
    strategy = cluster_strategy or _dominant_cluster_strategy(discovery)
    target = target_clusters or max(1, min(40, len(discovery.clusters) or 12))
    if strategy not in CLUSTER_STRATEGIES:
        raise ValueError(f"unsupported cluster strategy: {strategy}")
    if source == "local":
        clusters = _build_discovery_clusters(
            items,
            discovery.terms,
            strategy="tag_graph" if strategy in LLM_CLUSTER_STRATEGIES else strategy,
            target_clusters=target,
        )
        return _replace_discovery_clusters(discovery, clusters, ["Cluster repair used deterministic local synthesis."])
    if source != "openrouter":
        raise ValueError(f"unsupported repair source: {source}")
    clusters, notes = _openrouter_repair_clusters(
        discovery,
        items,
        model=model,
        endpoint=endpoint,
        api_key_env=api_key_env,
        cluster_strategy=strategy,
        target_clusters=target,
        max_input_tokens=max_input_tokens,
    )
    return _replace_discovery_clusters(discovery, clusters, notes)


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


def _local_discovery(
    records: list[RegionRecord],
    *,
    mode: str,
    model: str,
    cluster_strategy: str = "doc_id",
    target_clusters: int = 40,
) -> DiscoveryManifest:
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

    clusters = _build_discovery_clusters(records, terms, strategy=cluster_strategy, target_clusters=target_clusters)
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
        clusters=clusters,
        notes=[
            "Local discovery is deterministic and heuristic.",
            "Use an LLM discovery pass for stronger section summaries and role labels.",
            "Hierarchical mode currently emits the same schema; large-corpus windowing is planned.",
            f"Cluster strategy: {cluster_strategy}.",
            *(
                [f"Cluster strategy {cluster_strategy} requires source=openrouter; local fallback used tag_graph."]
                if cluster_strategy in LLM_CLUSTER_STRATEGIES
                else []
            ),
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
    cluster_strategy: str = "doc_id",
    target_clusters: int = 40,
) -> DiscoveryManifest:
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set")
    compact_records = cluster_strategy in LLM_CLUSTER_STRATEGIES
    packed = _pack_records_for_prompt(records, max_input_tokens=max_input_tokens, compact=compact_records)
    prompt = f"""Discover retrieval-evaluation structure for this refmarked corpus.

Return strict JSON with keys:
- corpus_summary: short summary
- terms: list of {{"term": str, "kind": "term", "refs": [stable refs]}}
- abbreviations: list of {{"term": str, "kind": "abbreviation", "refs": [stable refs]}}
- region_roles: list of {{"stable_ref": str, "roles": [str], "terms": [str], "summary": str, "notes": str}}
- range_candidates: list of {{"refs": [stable refs], "kind": "adjacent|distributed|section", "reason": str}}
- query_families: list of {{"name": str, "refs": [stable refs], "terms": [str], "notes": str}}
- clusters: list of {{"cluster_id": str, "name": str, "refs": [stable refs], "terms": [str], "strategy": "{cluster_strategy}", "notes": str}}
- notes: list of strings

Prefer practical training/evaluation roles:
content, definition, procedure, exception, example, table, heading,
summary_region, navigation_only, boilerplate, exclude_from_training,
range_candidate, distributed_candidate.

Cluster strategy: {cluster_strategy}
Target clusters: about {target_clusters}
If the cluster strategy starts with "llm_":
- create human-reviewable navigation clusters over the provided refs;
- use normalized labels, not raw noisy words;
- prefer rare/domain terms and user-intent phrases over generic common words;
- assign every provided ref to exactly one primary cluster unless it is genuinely unclassifiable;
- use llm_topics for topical/editor overview clusters;
- use llm_intents for user-task/question-intent clusters;
- include notes for ambiguous or mixed clusters.
- keep all summaries and notes concise; valid JSON matters more than detail.

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
        "max_tokens": 8000 if cluster_strategy in LLM_CLUSTER_STRATEGIES else 5000,
        "response_format": {"type": "json_object"},
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
    local = _local_discovery(
        records,
        mode=mode,
        model="local-seed",
        cluster_strategy=cluster_strategy,
        target_clusters=target_clusters,
    )
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
        clusters=(
            _clusters_from_payload(
                parsed.get("clusters") or _query_families_as_cluster_payload(parsed.get("query_families"), strategy=cluster_strategy),
                records=records,
                fallback=local.clusters,
                strategy=cluster_strategy,
                target_clusters=target_clusters,
            )
            if cluster_strategy in LLM_CLUSTER_STRATEGIES
            else local.clusters
        ),
        notes=[
            *_sanitize_model_notes([str(item) for item in parsed.get("notes", [])] or local.notes),
            f"Cluster strategy: {cluster_strategy}.",
            *(
                ["LLM-supplied clusters were sanitized against the manifest refs."]
                if cluster_strategy in LLM_CLUSTER_STRATEGIES
                else []
            ),
        ],
    )


def _sanitize_model_notes(notes: list[str]) -> list[str]:
    return [note for note in notes if "requires source=openrouter" not in note]


def _openrouter_repair_clusters(
    discovery: DiscoveryManifest,
    records: list[RegionRecord],
    *,
    model: str,
    endpoint: str,
    api_key_env: str,
    cluster_strategy: str,
    target_clusters: int,
    max_input_tokens: int,
) -> tuple[list[DiscoveryCluster], list[str]]:
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set")
    current_clusters = _cluster_repair_summary(discovery, records)
    review_issues = [issue.to_dict() for issue in review_discovery(discovery, records=records, max_issues=25)]
    packed = _pack_records_for_prompt(records, max_input_tokens=max_input_tokens, compact=True)
    prompt = f"""Repair the discovery cluster layer for this refmarked corpus.

Return strict JSON with keys:
- clusters: list of {{"cluster_id": str, "name": str, "refs": [stable refs], "terms": [str], "strategy": "{cluster_strategy}", "notes": str}}
- notes: list of concise strings explaining the repair.

Rules:
- Do not invent refs. Use only refs shown in Region cards.
- Assign every ref to exactly one primary cluster unless impossible.
- Prefer coherent navigation clusters over one-ref clusters.
- Split huge "other/unassigned" buckets into meaningful reviewable groups.
- Split any cluster that covers more than about 35 percent of refs when the
  target cluster budget allows it.
- Do not exceed the target cluster count unless there is a strong reason.
- Avoid one-ref clusters in high-level maps; group them under a coherent
  broader label when possible.
- Keep labels human-readable and normalized.
- Only apply legal/German clustering when the corpus is actually legal text or mostly German.
- Never label technical documentation as legal, regulatory, contractual, or compliance content unless those words are explicit source topics.
- If the corpus is technical documentation, cluster by task/topic users can navigate.
- Valid JSON matters more than detailed prose.

Target cluster count: about {target_clusters}
Cluster strategy: {cluster_strategy}

Current clusters:
{json.dumps(current_clusters, ensure_ascii=False, indent=2)}

Deterministic review issues:
{json.dumps(review_issues, ensure_ascii=False, indent=2)}

Region cards:
{packed}
"""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You repair Refmark discovery cluster manifests for corpus navigation and evaluation."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.15,
        "max_tokens": 12000,
        "response_format": {"type": "json_object"},
    }
    req = request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/b-imenitov/refmark",
            "X-Title": "refmark cluster repair",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=180) as response:
        payload = json.loads(response.read().decode("utf-8"))
    parsed = _parse_json_object(payload["choices"][0]["message"]["content"].strip())
    fallback = _build_discovery_clusters(records, discovery.terms, strategy="tag_graph", target_clusters=target_clusters)
    clusters = _clusters_from_payload(
        parsed.get("clusters"),
        records=records,
        fallback=fallback,
        strategy=cluster_strategy,
        target_clusters=target_clusters,
    )
    notes = _sanitize_model_notes([str(item) for item in parsed.get("notes", [])])
    return clusters, [
        *notes,
        f"Cluster repair source: openrouter.",
        f"Cluster repair model: {model}.",
        f"Cluster repair strategy: {cluster_strategy}.",
        "Cluster repair output was sanitized against the manifest refs.",
    ]


def _replace_discovery_clusters(
    discovery: DiscoveryManifest,
    clusters: list[DiscoveryCluster],
    notes: list[str],
) -> DiscoveryManifest:
    return DiscoveryManifest(
        schema=discovery.schema,
        created_at=datetime.now(timezone.utc).isoformat(),
        mode=discovery.mode,
        source=discovery.source,
        model=discovery.model,
        corpus_summary=discovery.corpus_summary,
        corpus_tokens=discovery.corpus_tokens,
        regions=discovery.regions,
        terms=discovery.terms,
        abbreviations=discovery.abbreviations,
        region_roles=discovery.region_roles,
        range_candidates=discovery.range_candidates,
        query_families=discovery.query_families,
        windows=discovery.windows,
        clusters=clusters,
        notes=[*discovery.notes, *notes],
    )


def _cluster_repair_summary(discovery: DiscoveryManifest, records: list[RegionRecord]) -> list[dict[str, Any]]:
    by_ref = {_stable_ref(record): record for record in records}
    summary: list[dict[str, Any]] = []
    for cluster in discovery.clusters:
        rows = [by_ref[ref] for ref in cluster.refs if ref in by_ref]
        summary.append(
            {
                "cluster_id": cluster.cluster_id,
                "name": cluster.name,
                "strategy": cluster.strategy,
                "source": cluster.source,
                "refs": cluster.refs[:80],
                "ref_count": len(cluster.refs),
                "terms": cluster.terms[:16],
                "notes": cluster.notes,
                "sample_titles": [_first_sentence(record.text)[:160] for record in rows[:8]],
            }
        )
    return summary


def _dominant_cluster_strategy(discovery: DiscoveryManifest) -> str:
    counts = Counter(cluster.strategy for cluster in discovery.clusters if cluster.strategy)
    return counts.most_common(1)[0][0] if counts else "doc_id"


def _pack_records_for_prompt(records: list[RegionRecord], *, max_input_tokens: int, compact: bool = False) -> str:
    chunks: list[str] = []
    total = 0
    for record in records:
        text = _compact_record_for_prompt(record) if compact else record.text.strip()
        cost = approx_tokens(text) + 8
        if chunks and total + cost > max_input_tokens:
            chunks.append("[TRUNCATED: corpus exceeded configured discovery token budget]")
            break
        chunks.append(f"[{_stable_ref(record)}]\n{text}")
        total += cost
    return "\n\n".join(chunks)


def _compact_record_for_prompt(record: RegionRecord, *, excerpt_chars: int = 900) -> str:
    title = _first_sentence(record.text).strip()[:180]
    excerpt = record.text.strip().replace("\r\n", "\n")[:excerpt_chars]
    return "\n".join(
        part
        for part in [
            f"title: {title}" if title else "",
            f"source_path: {record.source_path}" if record.source_path else "",
            f"excerpt: {excerpt}",
        ]
        if part
    )


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


def _clusters_from_payload(
    payload: Any,
    *,
    records: list[RegionRecord],
    fallback: list[DiscoveryCluster],
    strategy: str,
    target_clusters: int,
) -> list[DiscoveryCluster]:
    if not isinstance(payload, list):
        return fallback
    ref_aliases = _ref_aliases(records)
    valid_refs = set(ref_aliases.values())
    assigned: set[str] = set()
    clusters: list[DiscoveryCluster] = []
    for index, item in enumerate(payload[: max(1, target_clusters * 2)]):
        if not isinstance(item, dict):
            continue
        refs = []
        for raw_ref in item.get("refs", []):
            ref = ref_aliases.get(_normalize_ref_token(str(raw_ref)))
            if ref in valid_refs and ref not in assigned:
                refs.append(ref)
                assigned.add(ref)
        if not refs:
            continue
        name = str(item.get("name", "")).strip() or f"LLM cluster {len(clusters) + 1}"
        cluster_id = str(item.get("cluster_id", "")).strip() or f"{strategy}:{index + 1:02d}"
        clusters.append(
            DiscoveryCluster(
                cluster_id=_safe_cluster_id(cluster_id, strategy=strategy, index=index),
                name=name[:120],
                refs=refs,
                terms=[str(term).strip() for term in item.get("terms", []) if str(term).strip()][:12],
                source="openrouter",
                strategy=str(item.get("strategy", strategy) or strategy),
                parent_id=str(item["parent_id"]) if item.get("parent_id") is not None else None,
                notes=str(item.get("notes", "")),
            )
        )
    unassigned_records = [record for record in records if _stable_ref(record) not in assigned]
    if unassigned_records:
        backfill_target = max(1, min(target_clusters, max(4, len(unassigned_records) // 6)))
        backfill = _tag_graph_clusters(unassigned_records, target_clusters=backfill_target)
        for index, cluster in enumerate(backfill):
            clusters.append(
                DiscoveryCluster(
                    cluster_id=f"{strategy}:unassigned:{index + 1:02d}",
                    name=f"unassigned: {cluster.name}",
                    refs=cluster.refs,
                    terms=cluster.terms,
                    source="openrouter_sanitizer",
                    strategy=strategy,
                    notes="Deterministic backfill for refs the LLM did not assign to a primary cluster.",
                )
            )
    if not clusters:
        return fallback
    return _limit_cluster_count(clusters, target_clusters=target_clusters, strategy=strategy)


def _query_families_as_cluster_payload(payload: Any, *, strategy: str) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        return []
    clusters: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        refs = [str(ref).strip() for ref in item.get("refs", []) if str(ref).strip()]
        if not refs:
            continue
        name = str(item.get("name", "")).strip() or f"LLM family {index + 1}"
        clusters.append(
            {
                "cluster_id": f"{strategy}:{_slugify(name) or index + 1}",
                "name": name,
                "refs": refs,
                "terms": [str(term).strip() for term in item.get("terms", []) if str(term).strip()],
                "strategy": strategy,
                "notes": f"Derived from query family because the model did not return explicit {strategy} clusters.",
            }
        )
    return clusters


def _ref_aliases(records: list[RegionRecord]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    by_region_id: dict[str, list[str]] = {}
    for record in records:
        stable_ref = _stable_ref(record)
        aliases[stable_ref] = stable_ref
        by_region_id.setdefault(record.region_id, []).append(stable_ref)
    for region_id, stable_refs in by_region_id.items():
        if len(stable_refs) == 1:
            aliases[region_id] = stable_refs[0]
    return aliases


def _normalize_ref_token(value: str) -> str:
    return value.strip().strip("[]`'\"")


def _safe_cluster_id(value: str, *, strategy: str, index: int) -> str:
    cleaned = value.strip().replace(" ", "-")
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]+", "-", cleaned).strip("-")
    if not cleaned:
        return f"{strategy}:{index + 1:02d}"
    if ":" not in cleaned:
        return f"{strategy}:{cleaned}"
    return cleaned


def _limit_cluster_count(
    clusters: list[DiscoveryCluster],
    *,
    target_clusters: int,
    strategy: str,
) -> list[DiscoveryCluster]:
    if target_clusters <= 0 or len(clusters) <= target_clusters:
        return clusters
    coalesced = _coalesce_by_semantic_bucket(clusters, strategy=strategy)
    if 1 < len(coalesced) <= target_clusters:
        return coalesced
    ranked = sorted(clusters, key=_cluster_keep_key)
    kept = ranked[: max(1, target_clusters - 1)]
    overflow = ranked[max(1, target_clusters - 1) :]
    refs = [ref for cluster in overflow for ref in cluster.refs]
    terms = _dedupe(term for cluster in overflow for term in cluster.terms)[:16]
    names = [cluster.name for cluster in overflow[:12]]
    kept.append(
        DiscoveryCluster(
            cluster_id=f"{strategy}:overflow",
            name=_merged_cluster_name(names, terms),
            refs=refs,
            terms=terms,
            source="cluster_count_sanitizer",
            strategy=strategy,
            notes="Merged smaller clusters to respect the target cluster count. Original labels: " + "; ".join(names),
        )
    )
    return kept


def _coalesce_by_semantic_bucket(clusters: list[DiscoveryCluster], *, strategy: str) -> list[DiscoveryCluster]:
    buckets: dict[str, list[DiscoveryCluster]] = {}
    for cluster in clusters:
        label = _semantic_bucket_label(cluster)
        buckets.setdefault(label, []).append(cluster)
    if len(buckets) >= len(clusters):
        return clusters
    output: list[DiscoveryCluster] = []
    for index, (label, members) in enumerate(sorted(buckets.items(), key=lambda item: (-sum(len(c.refs) for c in item[1]), item[0]))):
        if len(members) == 1 and not members[0].name.lower().startswith(("unassigned", "other ")):
            output.append(members[0])
            continue
        refs = [ref for cluster in members for ref in cluster.refs]
        terms = _dedupe(term for cluster in members for term in cluster.terms)[:16]
        output.append(
            DiscoveryCluster(
                cluster_id=f"{strategy}:bucket:{index + 1:02d}",
                name=label,
                refs=refs,
                terms=terms,
                source="semantic_bucket_sanitizer",
                strategy=strategy,
                notes="Merged clusters into a semantic bucket to respect the target cluster count.",
            )
        )
    return output


def _semantic_bucket_label(cluster: DiscoveryCluster) -> str:
    text = " ".join([cluster.name, *cluster.terms]).lower()
    if any(word in text for word in ["tenant", "landlord", "mieter", "vermieter", "modernisierung", "kündigung"]):
        return "Landlord-Tenant Relations"
    if any(word in text for word in ["contract", "vertrag", "widerruf", "rücktritt", "formmängel", "garantie", "payment", "zahlung", "debt", "schuld", "retention", "zurückbehaltung"]):
        return "Contract Remedies and Obligations"
    if any(word in text for word in ["property", "possession", "vorkauf", "besitz", "grundstück", "claims", "anspruch", "beseitigung"]):
        return "Property Rights and Claims"
    if any(word in text for word in ["family", "ehegatten", "kind", "mündel", "verwandtschaft", "domicile", "residency", "wohnsitz"]):
        return "Family and Personal Status"
    if any(word in text for word in ["foundation", "stiftung", "business", "geschäftsführung", "authority", "gesellschaft"]):
        return "Organizations and Legal Capacity"
    if any(word in text for word in ["liability", "haftung", "healthcare", "kranken", "duty", "pflicht"]):
        return "Duties and Liability"
    if cluster.name.lower().startswith(("unassigned", "other ")):
        return "Mixed Legal Topics"
    return cluster.name


def _cluster_keep_key(cluster: DiscoveryCluster) -> tuple[int, int, str]:
    source_rank = 0 if cluster.source == "openrouter" else 1
    weak_name = cluster.name.lower().startswith(("unassigned", "other "))
    return (source_rank + (1 if weak_name else 0), -len(cluster.refs), cluster.name)


def _merged_cluster_name(names: list[str], terms: list[str]) -> str:
    text = " ".join([*names, *terms]).lower()
    labels: list[str] = []
    if any(word in text for word in ["contract", "vertrag", "widerruf", "rücktritt", "formmängel", "garantie", "payment", "zahlung", "debt", "schuld", "retention", "zurückbehaltung"]):
        labels.append("contract remedies and obligations")
    if any(word in text for word in ["property", "possession", "vorkauf", "besitz", "grundstück", "claims", "anspruch", "beseitigung"]):
        labels.append("property rights and claims")
    if any(word in text for word in ["domicile", "residency", "foundation", "stiftung", "family", "ehegatten", "kind", "mündel"]):
        labels.append("legal status and family matters")
    if any(word in text for word in ["liability", "haftung", "authority", "geschäftsführung", "healthcare", "kranken"]):
        labels.append("duties and liability")
    if labels:
        return "; ".join(_dedupe(labels)[:3]).title()
    clean_names = [
        name.removeprefix("unassigned: ").strip()
        for name in names
        if name and not name.lower().startswith(("other ", "unassigned: other"))
    ]
    if clean_names:
        return "Mixed: " + ", ".join(clean_names[:3])
    return "mixed reviewed topics"


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


def _cluster_from_payload(payload: dict[str, Any]) -> DiscoveryCluster:
    return DiscoveryCluster(
        cluster_id=str(payload.get("cluster_id", "")),
        name=str(payload.get("name", "")),
        refs=[str(ref) for ref in payload.get("refs", [])],
        terms=[str(term) for term in payload.get("terms", [])],
        source=str(payload.get("source", "deterministic")),
        strategy=str(payload.get("strategy", payload.get("source", "doc_id"))),
        parent_id=str(payload["parent_id"]) if payload.get("parent_id") is not None else None,
        notes=str(payload.get("notes", "")),
    )


def _build_discovery_clusters(
    records: list[RegionRecord],
    terms: list[DiscoveryTerm],
    *,
    strategy: str = "doc_id",
    target_clusters: int = 40,
) -> list[DiscoveryCluster]:
    if strategy == "doc_id":
        return _doc_id_clusters(records, terms, strategy=strategy)
    if strategy == "source_tree":
        return _source_tree_clusters(records)
    if strategy == "tag_graph":
        return _tag_graph_clusters(records, target_clusters=target_clusters)
    if strategy == "balanced_terms":
        return _balanced_term_clusters(records, target_clusters=target_clusters)
    if strategy in LLM_CLUSTER_STRATEGIES:
        return _tag_graph_clusters(records, target_clusters=target_clusters)
    raise ValueError(f"unsupported cluster strategy: {strategy}")


def _doc_id_clusters(records: list[RegionRecord], terms: list[DiscoveryTerm], *, strategy: str) -> list[DiscoveryCluster]:
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
                strategy=strategy,
                notes="Document-id cluster; usually best when source structure is already meaningful.",
            )
        )
    return clusters


def _source_tree_clusters(records: list[RegionRecord]) -> list[DiscoveryCluster]:
    """Build a navigable hierarchy from source paths or structured doc ids."""

    if not records:
        return []
    record_terms = _source_tree_record_terms(records)
    groups: dict[tuple[str, str], list[RegionRecord]] = {}
    top_order: list[str] = []
    child_order: list[tuple[str, str]] = []
    for record in records:
        top, child = _source_tree_parts(record)
        if top not in top_order:
            top_order.append(top)
        key = (top, child)
        if key not in groups:
            child_order.append(key)
        groups.setdefault(key, []).append(record)
    clusters: list[DiscoveryCluster] = []
    for top in top_order:
        rows = [record for key, members in groups.items() if key[0] == top for record in members]
        child_count = sum(1 for key in groups if key[0] == top)
        clusters.append(
            _cluster_from_records(
                f"source:{_slugify(top)}",
                _source_tree_parent_name(top, rows),
                rows,
                strategy="source_tree",
                source="source_path",
                record_terms=record_terms,
            )
        )
        clusters[-1] = DiscoveryCluster(
            cluster_id=clusters[-1].cluster_id,
            name=clusters[-1].name,
            refs=clusters[-1].refs,
            terms=clusters[-1].terms,
            source=clusters[-1].source,
            strategy=clusters[-1].strategy,
            notes=f"Source-tree parent with {child_count} child topic group(s).",
        )
    for top, child in child_order:
        rows = groups[(top, child)]
        parent_id = f"source:{_slugify(top)}"
        child_name = _source_tree_child_name(child, rows)
        cluster = _cluster_from_records(
            f"{parent_id}:{_slugify(child)}",
            child_name,
            rows,
            strategy="source_tree",
            source="source_path",
            record_terms=record_terms,
        )
        clusters.append(
            DiscoveryCluster(
                cluster_id=cluster.cluster_id,
                name=cluster.name,
                refs=cluster.refs,
                terms=cluster.terms,
                source=cluster.source,
                strategy=cluster.strategy,
                parent_id=parent_id,
                notes="Source-tree child topic group.",
            )
        )
    return clusters


def _source_tree_record_terms(records: list[RegionRecord]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for record in records:
        title = _region_title_text(record.text)
        phrases = _title_phrase_tags(title)
        output[_stable_ref(record)] = phrases[:10]
    return output


def _title_phrase_tags(title: str) -> list[str]:
    title = _clean_source_tree_title(title)
    if not title:
        return []
    raw_tokens = [
        _normalize_variant(token.lower())
        for token in TOKEN_RE.findall(title)
        if _is_candidate_discovery_token(token)
    ]
    generic = GENERIC_TERMS | {
        "appendix",
        "chapter",
        "book",
        "page",
        "section",
        "useful",
        "following",
        "current",
        "future",
        "made",
        "make",
        "making",
        "various",
    }
    product_terms = {"django", "fastapi", "kubernetes", "rust"}
    content_tokens = [token for token in raw_tokens if token not in generic]
    if not content_tokens:
        return []
    phrases: list[str] = []
    for size in (3, 2):
        for index in range(0, max(0, len(content_tokens) - size + 1)):
            window = content_tokens[index : index + size]
            if all(token in product_terms for token in window):
                continue
            if len(set(window)) < len(window):
                continue
            phrases.append("-".join(window))
    if phrases:
        return _dedupe(phrases)[:6]
    return []


def _source_tree_parts(record: RegionRecord) -> tuple[str, str]:
    source = Path(record.source_path).stem if record.source_path else record.doc_id
    parts = [part for part in re.split(r"[_/\-.]+", source.lower()) if part]
    parts = [part for part in parts if part not in {"docs", "doc", "en", "content", "src", "txt", "md", "rst", "index"}]
    if not parts:
        parts = [record.doc_id.lower()]
    if parts[0] in {"ref", "reference"} and len(parts) > 1:
        return "reference", f"reference/{parts[1]}"
    if len(parts) > 2 and parts[0] == "how" and parts[1] == "to":
        return "how-to", f"how-to/{parts[2]}"
    if parts[0] == "howto":
        return "how-to", f"how-to/{parts[1]}" if len(parts) > 1 else "how-to"
    if parts[0] in {"how", "howto"} and len(parts) > 1:
        return "how-to", f"how-to/{parts[1]}"
    if parts[0] == "ref":
        return "reference", f"reference/{parts[1]}" if len(parts) > 1 else "reference"
    if parts[0] in {"topic", "topics", "tutorial", "tutorials", "guide", "guides"} and len(parts) > 1:
        return parts[0].removesuffix("s"), f"{parts[0].removesuffix('s')}/{parts[1]}"
    if parts[0] in {"api", "apis"} and len(parts) > 1:
        return "api", f"api/{parts[1]}"
    top = parts[0]
    child = "/".join(parts[:2]) if len(parts) > 1 else top
    return top, child


def _titleize_source_part(value: str) -> str:
    aliases = {
        "faq": "FAQ",
        "api": "API",
        "ref": "Reference",
        "reference": "Reference",
        "how-to": "How To",
    }
    return " / ".join(aliases.get(part, part.replace("_", " ").replace("-", " ").title()) for part in value.split("/"))


def _source_tree_parent_name(top: str, records: list[RegionRecord]) -> str:
    label = _titleize_source_part(top)
    if re.fullmatch(r"ch\d+", top) or "_" in top or _looks_like_source_shorthand(label):
        title = _first_region_title(records)
        if title and not _looks_like_source_shorthand(title):
            return _clean_source_tree_title(title)
    return label


def _source_tree_child_name(child: str, records: list[RegionRecord]) -> str:
    source_keys = {record.source_path or record.doc_id for record in records}
    if len(source_keys) == 1:
        title = _first_region_title(records)
        if title and not _looks_like_source_shorthand(title):
            return _clean_source_tree_title(title)
    label = _titleize_source_part(child)
    if len(records) <= 4:
        titles = [
            _clean_source_tree_title(title)
            for record in records
            if (title := _region_title_text(record.text)) and not _looks_like_source_shorthand(title)
        ]
        if titles:
            common = _common_title_prefix(titles)
            if common:
                return common
            if len(titles) == 1:
                return titles[0]
    return label


def _first_region_title(records: list[RegionRecord]) -> str:
    for record in sorted(records, key=lambda item: item.ordinal):
        title = _region_title_text(record.text)
        if title:
            return title
    return ""


def _clean_source_tree_title(title: str) -> str:
    title = re.sub(r"\s+", " ", title).strip()
    title = re.sub(r"^\{\{[#%].*?\}\}\s*", "", title).strip()
    return title[:90]


def _looks_like_source_shorthand(title: str) -> bool:
    lowered = title.strip().lower()
    return bool(
        re.fullmatch(r"(ch|chapter|src|docs?|content|ref|howto|appendix)[\w\s/_-]*", lowered)
        or re.search(r"\{\{[#%]", lowered)
    )


def _common_title_prefix(titles: list[str]) -> str:
    if not titles:
        return ""
    words = [title.split() for title in titles]
    prefix: list[str] = []
    for column in zip(*words, strict=False):
        lowered = {word.lower().strip(":-") for word in column}
        if len(lowered) != 1:
            break
        prefix.append(column[0].strip(":-"))
    if 2 <= len(prefix) <= 8:
        return " ".join(prefix)
    return ""


def _tag_graph_clusters(records: list[RegionRecord], *, target_clusters: int) -> list[DiscoveryCluster]:
    """Cluster refs by normalized high-signal tags and fold small buckets.

    This is intentionally explainable: each ref is assigned to its strongest
    corpus-level tag, then small tags are merged into an ``other`` cluster. It
    is a good flat-wiki starting point because labels are human-reviewable.
    """

    target = _safe_target_clusters(target_clusters, len(records))
    record_terms = _record_terms(records, corpus_filter=True)
    term_counts = Counter(term for terms in record_terms.values() for term in set(terms))
    selected_terms = [term for term, count in term_counts.most_common(max(target * 3, target)) if count >= 2]
    if not selected_terms:
        return _doc_id_clusters(records, [], strategy="tag_graph")
    selected = set(selected_terms[: max(target * 2, target)])
    buckets: dict[str, list[RegionRecord]] = {}
    other: list[RegionRecord] = []
    for record in records:
        terms = record_terms[_stable_ref(record)]
        candidates = [term for term in terms if term in selected]
        if not candidates:
            other.append(record)
            continue
        best = min(candidates, key=lambda term: (-term_counts[term], term))
        buckets.setdefault(best, []).append(record)
    min_size = max(2, len(records) // max(target * 3, 1))
    small_keys = [key for key, rows in buckets.items() if len(rows) < min_size]
    for key in small_keys:
        other.extend(buckets.pop(key))
    clusters = _clusters_from_buckets(
        buckets,
        strategy="tag_graph",
        source="normalized_tags",
        record_terms=record_terms,
    )
    if other:
        clusters.append(
            _cluster_from_records(
                "tag:other",
                "other topics",
                other,
                strategy="tag_graph",
                source="normalized_tags",
                record_terms=record_terms,
            )
        )
    clusters.sort(key=lambda item: (-len(item.refs), item.name))
    if len(clusters) <= target:
        return clusters
    kept = clusters[: max(1, target - 1)]
    overflow_refs = set(ref for cluster in kept for ref in cluster.refs)
    overflow_records = [record for record in records if _stable_ref(record) not in overflow_refs]
    if overflow_records:
        kept.append(
            _cluster_from_records(
                "tag:other",
                "other topics",
                overflow_records,
                strategy="tag_graph",
                source="normalized_tags",
                record_terms=record_terms,
            )
        )
    return kept


def _balanced_term_clusters(records: list[RegionRecord], *, target_clusters: int) -> list[DiscoveryCluster]:
    """Create balanced clusters from local term-vector similarity.

    This is the no-embedding baseline for a dashboard-friendly semantic view.
    It picks diverse seed articles, assigns nearby regions under a soft capacity,
    and labels each cluster by its most common terms.
    """

    if not records:
        return []
    target = _safe_target_clusters(target_clusters, len(records))
    record_terms = _record_terms(records, corpus_filter=True, limit=80, prefer_rare=True)
    global_counts = Counter(term for terms in record_terms.values() for term in set(terms))
    vectors = {record: Counter(record_terms[_stable_ref(record)]) for record in records}
    seeds = _choose_balanced_seeds(records, vectors, target)
    buckets: dict[str, list[RegionRecord]] = {f"balanced:{index + 1:02d}": [] for index in range(len(seeds))}
    seed_by_key = dict(zip(buckets, seeds, strict=True))
    capacity = max(1, (len(records) + len(seeds) - 1) // len(seeds))
    for record in sorted(records, key=lambda item: (-sum(vectors[item].values()), _stable_ref(item))):
        ranked_keys = sorted(
            buckets,
            key=lambda key: (
                len(buckets[key]) >= capacity,
                -_cosine(vectors[record], vectors[seed_by_key[key]]),
                len(buckets[key]),
                key,
            ),
        )
        buckets[ranked_keys[0]].append(record)
    clusters = []
    for key, rows in buckets.items():
        if not rows:
            continue
        cluster = _cluster_from_records(
            key,
            _label_for_records(rows, record_terms=record_terms, global_counts=global_counts),
            rows,
            strategy="balanced_terms",
            source="term_vector_partition",
            record_terms=record_terms,
        )
        clusters.append(
            DiscoveryCluster(
                cluster_id=cluster.cluster_id,
                name=cluster.name,
                refs=cluster.refs,
                terms=cluster.terms,
                source=cluster.source,
                strategy=cluster.strategy,
                notes="Balanced local term-vector cluster; use as an overview layer for flat corpora.",
            )
        )
    clusters.sort(key=lambda item: item.cluster_id)
    return clusters


def _safe_target_clusters(target_clusters: int, record_count: int) -> int:
    if record_count <= 0:
        return 0
    return max(1, min(max(1, target_clusters), record_count))


def _record_terms(
    records: list[RegionRecord],
    *,
    corpus_filter: bool = False,
    limit: int = 12,
    prefer_rare: bool = False,
) -> dict[str, list[str]]:
    raw = {
        _stable_ref(record): _dedupe(
            normalized
            for term in _important_tokens(_record_cluster_text(record))
            if (normalized := _normalize_discovery_term(_normalize_variant(term)))
        )
        for record in records
    }
    if not corpus_filter or not records:
        return {ref: terms[:limit] for ref, terms in raw.items()}
    counts = Counter(term for terms in raw.values() for term in set(terms))
    max_docs = max(2, int(len(records) * 0.16))
    min_docs = 1 if len(records) < 20 else (max(3, int(len(records) * 0.01)) if prefer_rare else 2)
    output: dict[str, list[str]] = {}
    for ref, terms in raw.items():
        filtered = [term for term in terms if min_docs <= counts[term] <= max_docs]
        if prefer_rare:
            filtered.sort(key=lambda term: (counts[term], term))
        output[ref] = filtered[:limit]
    return output


def _record_cluster_text(record: RegionRecord) -> str:
    text = record.text
    text = re.sub(r"^\s*\[[A-Za-z]\d{2,}\]\s*", "", text)
    text = re.sub(r"^\s*[\w.-]+(?:_set|_corpus|_combined)?\s*:\s*", "", text)
    text = re.sub(r"(?im)^#\s*source url:\s*\S+\s*$", "", text)
    text = re.sub(r"https?://\S+", " ", text)
    title = _region_title_text(text)
    if title and approx_tokens(text) > 800:
        return f"{title}\n{title}\n{title}\n{text[:2500]}"
    return f"{title}\n{title}\n{text}" if title else text


def _region_title_text(text: str) -> str:
    lines = text.splitlines()
    for line in lines[:20]:
        cleaned = line.strip()
        if re.match(r"^#{1,6}\s+\S", cleaned):
            return cleaned.lstrip("#").strip()[:180]
    for index, line in enumerate(lines[:-1]):
        cleaned = line.strip().lstrip("#").strip()
        next_line = lines[index + 1].strip()
        if cleaned and _is_rst_adornment(next_line):
            return cleaned[:180]
        if _is_rst_adornment(cleaned) and next_line and not _is_rst_adornment(next_line):
            return next_line.strip().lstrip("#").strip()[:180]
    for line in lines[:8]:
        cleaned = line.strip().lstrip("#").strip()
        if cleaned and not _is_rst_adornment(cleaned) and not _looks_like_generated_or_code_line(cleaned):
            return cleaned[:180]
    return ""


def _looks_like_generated_or_code_line(value: str) -> bool:
    return bool(
        value.startswith(("```", "$ ", "> ", "{{#", "{{%", "<a ", "<span "))
        or re.fullmatch(r"[-*]\s+\[[^\]]+\]\(.+\)", value)
    )


def _is_rst_adornment(value: str) -> bool:
    return bool(re.fullmatch(r"([=\-~`:#\"'^_*+])\1{2,}", value.strip()))


def _clusters_from_buckets(
    buckets: dict[str, list[RegionRecord]],
    *,
    strategy: str,
    source: str,
    record_terms: dict[str, list[str]] | None = None,
) -> list[DiscoveryCluster]:
    return [
        _cluster_from_records(
            f"tag:{_slugify(name)}",
            name,
            rows,
            strategy=strategy,
            source=source,
            record_terms=record_terms,
        )
        for name, rows in buckets.items()
        if rows
    ]


def _cluster_from_records(
    cluster_id: str,
    name: str,
    records: list[RegionRecord],
    *,
    strategy: str,
    source: str,
    record_terms: dict[str, list[str]] | None = None,
) -> DiscoveryCluster:
    refs = [_stable_ref(record) for record in records]
    terms_by_ref = record_terms or _record_terms(records, corpus_filter=False, limit=40)
    counter = Counter(term for record in records for term in terms_by_ref.get(_stable_ref(record), []))
    terms = [term for term, _count in counter.most_common(12)]
    return DiscoveryCluster(
        cluster_id=cluster_id,
        name=name,
        refs=refs,
        terms=terms,
        source=source,
        strategy=strategy,
    )


def _choose_balanced_seeds(
    records: list[RegionRecord],
    vectors: dict[RegionRecord, Counter[str]],
    target: int,
) -> list[RegionRecord]:
    ranked = sorted(records, key=lambda record: (-len(vectors[record]), _stable_ref(record)))
    seeds: list[RegionRecord] = []
    for record in ranked:
        if not seeds:
            seeds.append(record)
        else:
            nearest = max(_cosine(vectors[record], vectors[seed]) for seed in seeds)
            if nearest < 0.72:
                seeds.append(record)
        if len(seeds) >= target:
            break
    for record in ranked:
        if len(seeds) >= target:
            break
        if record not in seeds:
            seeds.append(record)
    return seeds


def _label_for_records(
    records: list[RegionRecord],
    *,
    record_terms: dict[str, list[str]] | None = None,
    global_counts: Counter[str] | None = None,
) -> str:
    terms_by_ref = record_terms or _record_terms(records, corpus_filter=False, limit=40)
    counter = Counter(term for record in records for term in terms_by_ref.get(_stable_ref(record), []))
    if global_counts:
        min_cluster_count = max(2, min(6, len(records) // 16))
        candidates = [term for term in counter if counter[term] >= min_cluster_count]
        if not candidates:
            candidates = [term for term in counter if counter[term] >= 2] or list(counter)
        total_docs = max(global_counts.values(), default=1)
        ranked = sorted(
            candidates,
            key=lambda term: (
                -(counter[term] * math.log((total_docs + 1) / (global_counts[term] + 1))),
                -counter[term],
                term,
            ),
        )
        label_terms = ranked[:3]
    else:
        label_terms = [term for term, _count in counter.most_common(3)]
    return ", ".join(label_terms) if label_terms else "misc"


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = set(left) & set(right)
    numerator = sum(left[key] * right[key] for key in overlap)
    left_norm = sum(value * value for value in left.values()) ** 0.5
    right_norm = sum(value * value for value in right.values()) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "misc"


def _normalize_discovery_term(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"[^\w-]+", " ", lowered, flags=re.UNICODE)
    return " ".join(part for part in lowered.split() if part not in GENERIC_TERMS)


def _normalize_variant(value: str) -> str:
    term = value.strip().lower()
    if len(term) <= 4 or not re.fullmatch(r"[a-z][a-z-]+", term):
        return term
    if term.endswith("ies") and len(term) > 5:
        return term[:-3] + "y"
    if term.endswith("sses"):
        return term[:-2]
    if term.endswith("xes") and len(term) > 5:
        return term[:-2]
    if term.endswith("ses") and len(term) > 5:
        return term[:-2]
    if term.endswith("s") and not term.endswith(("ss", "us")):
        return term[:-1]
    return term


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
        if _is_candidate_discovery_token(token)
    ]


def _is_candidate_discovery_token(token: str) -> bool:
    lowered = token.lower()
    return (
        len(lowered) > 3
        and "_" not in lowered
        and lowered not in GENERIC_TERMS
        and not lowered.isdigit()
        and not re.search(r"\d", lowered)
        and not re.fullmatch(r"[a-z]+-[a-z]+-[a-z0-9-]+", lowered)
    )


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
