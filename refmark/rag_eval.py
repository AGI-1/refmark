"""Evidence-region evaluation helpers for RAG and corpus lifecycle checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from collections import Counter, defaultdict
from pathlib import Path
import re
from typing import Any, Callable, Iterable

from refmark.citations import CitationRef, parse_citation_refs
from refmark.metrics import normalize_ref
from refmark.pipeline import RegionRecord, read_manifest
from refmark.search_index import SUPPORTED_EXTENSIONS, map_corpus


Retriever = Callable[[str], Iterable[Any]]


@dataclass(frozen=True)
class CorpusMap:
    """A stable ref -> region manifest for an addressable corpus."""

    records: list[RegionRecord]

    @classmethod
    def from_records(cls, records: Iterable[RegionRecord]) -> "CorpusMap":
        return cls(list(records))

    @classmethod
    def from_manifest(cls, path: str | Path) -> "CorpusMap":
        return cls(read_manifest(path))

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        extensions: Iterable[str] = SUPPORTED_EXTENSIONS,
        marker_format: str = "typed_bracket",
        chunker: str = "paragraph",
        tokens_per_chunk: int | None = None,
        lines_per_chunk: int | None = None,
        min_words: int = 8,
        exclude_globs: Iterable[str] = (),
    ) -> "CorpusMap":
        return cls(
            map_corpus(
                path,
                extensions=extensions,
                marker_format=marker_format,
                chunker=chunker,
                tokens_per_chunk=tokens_per_chunk,
                lines_per_chunk=lines_per_chunk,
                min_words=min_words,
                exclude_globs=exclude_globs,
            )
        )

    @property
    def by_stable_ref(self) -> dict[str, RegionRecord]:
        return {_stable_ref(record): record for record in self.records}

    def validate_refs(self, refs: Iterable[str]) -> dict[str, list[str]]:
        missing: list[str] = []
        ambiguous: list[str] = []
        for ref in refs:
            expanded = self.expand_refs([ref])
            if expanded:
                continue
            if ":" not in ref and len({record.doc_id for record in self.records if record.region_id == ref}) > 1:
                ambiguous.append(ref)
            else:
                missing.append(ref)
        return {"missing": missing, "ambiguous": ambiguous}

    def expand_refs(self, refs: Iterable[str]) -> list[str]:
        """Resolve strict citation refs/ranges to stable refs."""
        expanded: list[str] = []
        for citation in parse_citation_refs(refs):
            expanded.extend(self._expand_citation(citation))
        return _dedupe(expanded)

    def source_hashes(self, refs: Iterable[str]) -> dict[str, str]:
        by_ref = self.by_stable_ref
        return {stable_ref: by_ref[stable_ref].hash for stable_ref in self.expand_refs(refs) if stable_ref in by_ref}

    def changed_refs(self, previous: "CorpusMap") -> dict[str, list[str]]:
        current = self.by_stable_ref
        old = previous.by_stable_ref
        added = sorted(set(current) - set(old))
        removed = sorted(set(old) - set(current))
        changed = sorted(ref for ref in set(current) & set(old) if current[ref].hash != old[ref].hash)
        unchanged = sorted(ref for ref in set(current) & set(old) if current[ref].hash == old[ref].hash)
        return {"added": added, "removed": removed, "changed": changed, "unchanged": unchanged}

    def context_pack(
        self,
        refs: Iterable[str],
        *,
        separator: str = "\n\n",
        include_headers: bool = True,
    ) -> "ContextPack":
        """Return an ordered evidence bundle for refs/ranges in this corpus."""

        by_ref = self.by_stable_ref
        stable_refs = self.expand_refs(refs)
        records = [by_ref[stable_ref] for stable_ref in stable_refs if stable_ref in by_ref]
        chunks: list[str] = []
        for stable_ref, record in zip(stable_refs, records, strict=False):
            text = record.text.rstrip()
            chunks.append(f"[{stable_ref}]\n{text}" if include_headers else text)
        return ContextPack(refs=stable_refs, records=records, text=separator.join(chunks))

    def stale_examples(self, examples: Iterable["EvalExample"]) -> list["StaleExample"]:
        by_ref = self.by_stable_ref
        stale: list[StaleExample] = []
        for example in examples:
            missing_refs: list[str] = []
            changed_refs: list[str] = []
            for stable_ref in self.expand_refs(example.gold_refs):
                record = by_ref.get(stable_ref)
                if record is None:
                    missing_refs.append(stable_ref)
                    continue
                expected_hash = example.source_hashes.get(stable_ref)
                if expected_hash and expected_hash != record.hash:
                    changed_refs.append(stable_ref)
            if missing_refs or changed_refs:
                stale.append(StaleExample(example=example, missing_refs=missing_refs, changed_refs=changed_refs))
        return stale

    def _resolve_ref(self, ref: str) -> str | None:
        by_ref = self.by_stable_ref
        if ref in by_ref:
            return ref
        if ":" in ref:
            doc_id, region_id = ref.split(":", 1)
            candidate = f"{doc_id}:{_normalize_region_id(region_id)}"
            return candidate if candidate in by_ref else None
        matches = [_stable_ref(record) for record in self.records if record.region_id == ref or record.region_id == _normalize_region_id(ref)]
        return matches[0] if len(matches) == 1 else None

    def _expand_range(self, start: str, end: str) -> list[str]:
        start_ref = self._resolve_ref(start)
        end_ref = self._resolve_ref(end)
        if start_ref is None or end_ref is None:
            return []
        start_record = self.by_stable_ref[start_ref]
        end_record = self.by_stable_ref[end_ref]
        if start_record.doc_id != end_record.doc_id:
            return [start_ref, end_ref]
        doc_records = [record for record in self.records if record.doc_id == start_record.doc_id]
        doc_records.sort(key=lambda record: record.ordinal)
        start_index = next(index for index, record in enumerate(doc_records) if record.region_id == start_record.region_id)
        end_index = next(index for index, record in enumerate(doc_records) if record.region_id == end_record.region_id)
        lo, hi = sorted((start_index, end_index))
        return [_stable_ref(record) for record in doc_records[lo : hi + 1]]

    def _expand_citation(self, citation: CitationRef) -> list[str]:
        if citation.is_range:
            end_ref = citation.stable_end_ref
            if end_ref is None:
                return []
            return self._expand_range(citation.stable_ref, end_ref)
        resolved = self._resolve_ref(citation.stable_ref)
        return [resolved] if resolved is not None else []


@dataclass(frozen=True)
class EvalExample:
    query: str
    gold_refs: list[str]
    source_hashes: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvalExample":
        return cls(
            query=str(payload["query"]),
            gold_refs=[str(ref) for ref in payload.get("gold_refs", [])],
            source_hashes={str(key): str(value) for key, value in payload.get("source_hashes", {}).items()},
            metadata=dict(payload.get("metadata", {})),
        )

    def with_source_hashes(self, corpus: CorpusMap) -> "EvalExample":
        return EvalExample(
            query=self.query,
            gold_refs=self.gold_refs,
            source_hashes=corpus.source_hashes(self.gold_refs),
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StaleExample:
    example: EvalExample
    missing_refs: list[str]
    changed_refs: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "example": self.example.to_dict(),
            "missing_refs": self.missing_refs,
            "changed_refs": self.changed_refs,
        }


@dataclass(frozen=True)
class NormalizedHit:
    stable_ref: str
    score: float | None = None
    context_refs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ContextPack:
    refs: list[str]
    records: list[RegionRecord]
    text: str

    @property
    def token_estimate(self) -> int:
        return max(1, len(self.text.split()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "refs": self.refs,
            "text": self.text,
            "token_estimate": self.token_estimate,
            "records": [record.to_dict() for record in self.records],
        }


@dataclass(frozen=True)
class EvalExampleResult:
    query: str
    gold_refs: list[str]
    retrieved_refs: list[str]
    context_refs: list[str]
    hit_at_1: bool
    hit_at_k: bool
    reciprocal_rank: float
    gold_coverage: float
    region_precision: float
    context_ref_count: int
    top_ref: str | None = None
    top_score: float | None = None
    second_score: float | None = None
    score_margin: float | None = None
    score_margin_ratio: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvalRun:
    name: str
    metrics: dict[str, float]
    examples: list[EvalExampleResult]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metrics": self.metrics,
            "diagnostics": self.diagnostics,
            "examples": [item.to_dict() for item in self.examples],
        }


@dataclass(frozen=True)
class EvalSuite:
    examples: list[EvalExample]
    corpus: CorpusMap

    @classmethod
    def from_rows(cls, rows: Iterable[dict[str, Any]], *, corpus: CorpusMap) -> "EvalSuite":
        return cls([EvalExample.from_dict(row) for row in rows], corpus)

    def with_source_hashes(self) -> "EvalSuite":
        return EvalSuite([example.with_source_hashes(self.corpus) for example in self.examples], self.corpus)

    def validate_refs(self) -> dict[str, list[dict[str, Any]]]:
        missing: list[dict[str, Any]] = []
        ambiguous: list[dict[str, Any]] = []
        for index, example in enumerate(self.examples):
            result = self.corpus.validate_refs(example.gold_refs)
            if result["missing"]:
                missing.append({"index": index, "query": example.query, "refs": result["missing"]})
            if result["ambiguous"]:
                ambiguous.append({"index": index, "query": example.query, "refs": result["ambiguous"]})
        return {"missing": missing, "ambiguous": ambiguous}

    def stale_examples(self) -> list[StaleExample]:
        return self.corpus.stale_examples(self.examples)

    def evaluate(self, retriever: Retriever, *, name: str = "retriever", k: int = 10) -> EvalRun:
        results = [_evaluate_example(example, self.corpus, retriever, k=k) for example in self.examples]
        return EvalRun(
            name=name,
            metrics=_summarize_results(results),
            diagnostics=diagnose_results(results),
            examples=results,
        )

    def compare(self, retrievers: dict[str, Retriever], *, k: int = 10) -> dict[str, EvalRun]:
        return {name: self.evaluate(retriever, name=name, k=k) for name, retriever in retrievers.items()}


def _evaluate_example(example: EvalExample, corpus: CorpusMap, retriever: Retriever, *, k: int) -> EvalExampleResult:
    gold_refs = corpus.expand_refs(example.gold_refs)
    hits = [_normalize_hit(hit, corpus) for hit in list(retriever(example.query))[:k]]
    retrieved_refs = [hit.stable_ref for hit in hits]
    context_refs = _dedupe(ref for hit in hits for ref in (hit.context_refs or [hit.stable_ref]))
    gold_set = set(gold_refs)
    context_set = set(context_refs)
    retrieved_set = set(retrieved_refs)
    top_score = hits[0].score if hits else None
    second_score = hits[1].score if len(hits) > 1 else None
    score_margin = _score_margin(top_score, second_score)
    reciprocal_rank = 0.0
    for index, hit in enumerate(hits, start=1):
        hit_refs = set(hit.context_refs or [hit.stable_ref])
        if hit_refs & gold_set:
            reciprocal_rank = 1.0 / index
            break
    intersection = context_set & gold_set
    return EvalExampleResult(
        query=example.query,
        gold_refs=gold_refs,
        retrieved_refs=retrieved_refs,
        context_refs=context_refs,
        hit_at_1=bool(hits and set(hits[0].context_refs or [hits[0].stable_ref]) & gold_set),
        hit_at_k=bool(intersection),
        reciprocal_rank=reciprocal_rank,
        gold_coverage=(len(intersection) / len(gold_set)) if gold_set else 0.0,
        region_precision=(len(intersection) / len(context_set)) if context_set else 0.0,
        context_ref_count=len(context_set),
        top_ref=hits[0].stable_ref if hits else None,
        top_score=top_score,
        second_score=second_score,
        score_margin=score_margin,
        score_margin_ratio=(score_margin / max(abs(top_score or 0.0), 1e-9)) if score_margin is not None else None,
    )


def _summarize_results(results: list[EvalExampleResult]) -> dict[str, float]:
    count = len(results)
    if not count:
        return {
            "count": 0.0,
            "hit_at_1": 0.0,
            "hit_at_k": 0.0,
            "mrr": 0.0,
            "gold_coverage": 0.0,
            "region_precision": 0.0,
            "avg_context_refs": 0.0,
        }
    return {
        "count": float(count),
        "hit_at_1": _mean(1.0 if result.hit_at_1 else 0.0 for result in results),
        "hit_at_k": _mean(1.0 if result.hit_at_k else 0.0 for result in results),
        "mrr": _mean(result.reciprocal_rank for result in results),
        "gold_coverage": _mean(result.gold_coverage for result in results),
        "region_precision": _mean(result.region_precision for result in results),
        "avg_context_refs": _mean(float(result.context_ref_count) for result in results),
    }


def diagnose_results(
    results: list[EvalExampleResult],
    *,
    hard_ref_limit: int = 25,
    confusion_limit: int = 25,
) -> dict[str, Any]:
    """Return reusable heatmap and confidence-gating diagnostics for an eval run."""

    heatmap = failure_heatmap(results, hard_ref_limit=hard_ref_limit, confusion_limit=confusion_limit)
    return {
        "heatmap": heatmap,
        "selective_jump": selective_jump_diagnostics(results),
        "adaptation": adaptation_recommendations(heatmap),
    }


def failure_heatmap(
    results: list[EvalExampleResult],
    *,
    hard_ref_limit: int = 25,
    confusion_limit: int = 25,
) -> dict[str, Any]:
    """Summarize refs and ref confusions that repeatedly underperform."""

    totals: Counter[str] = Counter()
    misses_at_1: Counter[str] = Counter()
    misses_at_k: Counter[str] = Counter()
    confusions: Counter[tuple[str, str]] = Counter()
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        primary_gold = result.gold_refs[0] if result.gold_refs else ""
        if not primary_gold:
            continue
        for gold_ref in result.gold_refs:
            totals[gold_ref] += 1
            if not result.hit_at_1:
                misses_at_1[gold_ref] += 1
            if not result.hit_at_k:
                misses_at_k[gold_ref] += 1
        if result.top_ref and result.top_ref not in result.gold_refs:
            confusions[(primary_gold, result.top_ref)] += 1
        if not result.hit_at_k and len(samples[primary_gold]) < 3:
            samples[primary_gold].append(
                {
                    "query": result.query,
                    "gold_refs": result.gold_refs,
                    "top_ref": result.top_ref,
                    "retrieved_refs": result.retrieved_refs[:5],
                }
            )
    hard_refs = []
    for ref, total in totals.items():
        miss_k = misses_at_k[ref]
        hard_refs.append(
            {
                "ref": ref,
                "count": total,
                "miss_at_1": misses_at_1[ref],
                "miss_at_k": miss_k,
                "hit_at_k": 1.0 - (miss_k / total if total else 0.0),
                "sample_misses": samples.get(ref, []),
            }
        )
    hard_refs.sort(key=lambda row: (-row["miss_at_k"], row["hit_at_k"], -row["miss_at_1"], row["ref"]))
    confusion_rows = [
        {"gold_ref": gold_ref, "top_ref": top_ref, "count": count}
        for (gold_ref, top_ref), count in confusions.most_common(confusion_limit)
    ]
    return {
        "hard_refs": hard_refs[:hard_ref_limit],
        "confusions": confusion_rows,
        "missed_queries": sum(1 for result in results if not result.hit_at_k),
    }


def selective_jump_diagnostics(
    results: list[EvalExampleResult],
    *,
    confidence_field: str = "score_margin_ratio",
    thresholds: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Estimate coverage/precision tradeoffs for accepting only confident top hits."""

    candidates = [
        result
        for result in results
        if getattr(result, confidence_field, None) is not None and result.top_ref is not None
    ]
    if thresholds is None:
        thresholds = _quantile_thresholds([float(getattr(result, confidence_field)) for result in candidates])
    rows = []
    total = len(results)
    for threshold in thresholds:
        accepted = [result for result in candidates if float(getattr(result, confidence_field)) >= threshold]
        rows.append(
            {
                "threshold": threshold,
                "accepted": len(accepted),
                "coverage": len(accepted) / total if total else 0.0,
                "precision_at_1": _mean(1.0 if result.hit_at_1 else 0.0 for result in accepted),
                "hit_at_k": _mean(1.0 if result.hit_at_k else 0.0 for result in accepted),
            }
        )
    return {
        "confidence_field": confidence_field,
        "thresholds": rows,
    }


def adaptation_recommendations(heatmap: dict[str, Any], *, limit: int = 20) -> list[dict[str, Any]]:
    """Turn heatmap symptoms into concrete next adaptation actions."""

    recommendations: list[dict[str, Any]] = []
    for row in heatmap.get("hard_refs", []):
        if row.get("miss_at_k", 0) <= 0:
            continue
        action = "add_queries_or_aliases"
        if row.get("hit_at_k", 1.0) == 0.0:
            action = "review_gold_or_region_granularity"
        recommendations.append(
            {
                "ref": row["ref"],
                "action": action,
                "reason": f"missed {row['miss_at_k']} of {row['count']} eval queries at k",
                "sample_queries": [sample["query"] for sample in row.get("sample_misses", [])],
            }
        )
    for row in heatmap.get("confusions", []):
        recommendations.append(
            {
                "ref": row["gold_ref"],
                "action": "add_hard_negative_or_disambiguating_signature",
                "reason": f"wrong top ref {row['top_ref']} appeared {row['count']} times",
                "negative_ref": row["top_ref"],
            }
        )
    return recommendations[:limit]


def _normalize_hit(hit: Any, corpus: CorpusMap) -> NormalizedHit:
    if isinstance(hit, NormalizedHit):
        return hit
    if isinstance(hit, str):
        stable_ref = corpus._resolve_ref(hit) or hit
        return NormalizedHit(stable_ref=stable_ref, context_refs=[stable_ref])
    if isinstance(hit, dict):
        ref = str(hit.get("stable_ref") or hit.get("ref") or hit.get("region_id"))
        stable_ref = corpus._resolve_ref(ref) or ref
        context_refs = [corpus._resolve_ref(str(ref_value)) or str(ref_value) for ref_value in hit.get("context_refs", [])]
        return NormalizedHit(stable_ref=stable_ref, score=_as_float(hit.get("score")), context_refs=context_refs or [stable_ref])
    ref = str(getattr(hit, "stable_ref", getattr(hit, "ref", "")))
    stable_ref = corpus._resolve_ref(ref) or ref
    context_refs = [
        corpus._resolve_ref(str(ref_value)) or str(ref_value)
        for ref_value in getattr(hit, "context_refs", [])
    ]
    return NormalizedHit(stable_ref=stable_ref, score=_as_float(getattr(hit, "score", None)), context_refs=context_refs or [stable_ref])


def _stable_ref(record: RegionRecord) -> str:
    return f"{record.doc_id}:{record.region_id}"


def _normalize_region_id(region_id: str) -> str:
    if re.fullmatch(r"[A-Za-z]+\d+", region_id):
        return normalize_ref(region_id)
    return region_id


def _dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_margin(top_score: float | None, second_score: float | None) -> float | None:
    if top_score is None:
        return None
    if second_score is None:
        return top_score
    return top_score - second_score


def _quantile_thresholds(values: list[float]) -> list[float]:
    if not values:
        return []
    sorted_values = sorted(values)
    thresholds = {0.0}
    for quantile in (0.25, 0.5, 0.75, 0.9):
        index = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * quantile))))
        thresholds.add(round(sorted_values[index], 6))
    return sorted(thresholds)
