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
from refmark.search_index import OPENROUTER_CHAT_URL, approx_tokens


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
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["terms"] = [item.to_dict() for item in self.terms]
        payload["abbreviations"] = [item.to_dict() for item in self.abbreviations]
        payload["region_roles"] = [item.to_dict() for item in self.region_roles]
        payload["range_candidates"] = [item.to_dict() for item in self.range_candidates]
        payload["query_families"] = [item.to_dict() for item in self.query_families]
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
) -> DiscoveryManifest:
    """Create a corpus-level discovery manifest.

    ``source="local"`` is deterministic and meant for CI/tests. ``source`` set
    to ``openrouter`` asks a model for the high-level summary/roles but keeps a
    local fallback so discovery never blocks the pipeline.
    """

    items = list(records)
    if mode not in {"whole", "hierarchical"}:
        raise ValueError(f"unsupported discovery mode: {mode}")
    if source not in {"local", "openrouter"}:
        raise ValueError(f"unsupported discovery source: {source}")
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


def load_discovery(path: str | Path) -> DiscoveryManifest:
    return DiscoveryManifest.from_dict(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def write_discovery(discovery: DiscoveryManifest, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(discovery.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


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


def _local_summary(records: list[RegionRecord], terms: list[DiscoveryTerm]) -> str:
    doc_ids = sorted({record.doc_id for record in records})
    top_terms = ", ".join(item.term for item in terms[:8])
    return f"{len(records)} regions across {len(doc_ids)} document(s). Frequent domain terms: {top_terms}."


def _region_roles(record: RegionRecord) -> list[str]:
    text = record.text.strip()
    lower = text.lower()
    roles: list[str] = []
    if text.startswith("#") or len(text.splitlines()) == 1 and len(text.split()) <= 10:
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
