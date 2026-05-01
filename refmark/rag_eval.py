"""Evidence-region evaluation helpers for RAG and corpus lifecycle checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
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
    revision_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_records(
        cls,
        records: Iterable[RegionRecord],
        *,
        revision_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "CorpusMap":
        return cls(list(records), revision_id=revision_id, metadata=dict(metadata or {}))

    @classmethod
    def from_manifest(
        cls,
        path: str | Path,
        *,
        revision_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "CorpusMap":
        return cls(read_manifest(path), revision_id=revision_id, metadata=dict(metadata or {"manifest_path": str(path)}))

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
        revision_id: str | None = None,
        metadata: dict[str, Any] | None = None,
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
            ),
            revision_id=revision_id,
            metadata=dict(metadata or {"source_path": str(path)}),
        )

    @property
    def by_stable_ref(self) -> dict[str, RegionRecord]:
        return {_stable_ref(record): record for record in self.records}

    @property
    def fingerprint(self) -> str:
        """Stable digest for the manifest's address space and region content."""

        rows = [
            {
                "stable_ref": _stable_ref(record),
                "hash": record.hash,
                "ordinal": record.ordinal,
                "source_path": record.source_path,
                "parent_region_id": record.parent_region_id,
            }
            for record in sorted(self.records, key=lambda record: (_stable_ref(record), record.ordinal))
        ]
        return _json_digest(rows)

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

    def diff_revision(self, previous: "CorpusMap") -> "CorpusRevisionDiff":
        """Compare this external map to a previous revision of the same corpus."""

        refs = self.changed_refs(previous)
        return CorpusRevisionDiff(
            previous_revision_id=previous.revision_id,
            current_revision_id=self.revision_id,
            previous_fingerprint=previous.fingerprint,
            current_fingerprint=self.fingerprint,
            added_refs=refs["added"],
            removed_refs=refs["removed"],
            changed_refs=refs["changed"],
            unchanged_refs=refs["unchanged"],
        )

    def snapshot(self) -> "CorpusMapSnapshot":
        """Return portable metadata for an out-of-band/shadow corpus map."""

        return CorpusMapSnapshot(
            revision_id=self.revision_id,
            fingerprint=self.fingerprint,
            region_count=len(self.records),
            stable_refs=sorted(self.by_stable_ref),
            metadata=dict(self.metadata),
        )

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

    def with_source_hashes(self, corpus: CorpusMap, *, preserve_existing: bool = True) -> "EvalExample":
        source_hashes = dict(self.source_hashes) if preserve_existing else {}
        for stable_ref, source_hash in corpus.source_hashes(self.gold_refs).items():
            source_hashes.setdefault(stable_ref, source_hash)
        return EvalExample(
            query=self.query,
            gold_refs=self.gold_refs,
            source_hashes=source_hashes,
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
class CorpusMapSnapshot:
    """Metadata for a refmark map that lives outside the source document."""

    revision_id: str | None
    fingerprint: str
    region_count: int
    stable_refs: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CorpusRevisionDiff:
    """Ref-level lifecycle diff between two external corpus maps."""

    previous_revision_id: str | None
    current_revision_id: str | None
    previous_fingerprint: str
    current_fingerprint: str
    added_refs: list[str]
    removed_refs: list[str]
    changed_refs: list[str]
    unchanged_refs: list[str]

    @property
    def has_changes(self) -> bool:
        return bool(self.added_refs or self.removed_refs or self.changed_refs)

    def affected_refs(self) -> list[str]:
        return _dedupe([*self.added_refs, *self.removed_refs, *self.changed_refs])

    def stale_examples(self, examples: Iterable["EvalExample"]) -> list["StaleExample"]:
        changed = set(self.changed_refs)
        removed = set(self.removed_refs)
        stale: list[StaleExample] = []
        for example in examples:
            refs = list(example.source_hashes) if example.source_hashes else _literal_refs_from_gold(example.gold_refs)
            missing_refs = sorted(ref for ref in refs if ref in removed)
            changed_refs = sorted(ref for ref in refs if ref in changed)
            if missing_refs or changed_refs:
                stale.append(StaleExample(example=example, missing_refs=missing_refs, changed_refs=changed_refs))
        return stale

    def to_dict(self) -> dict[str, Any]:
        return {
            "previous_revision_id": self.previous_revision_id,
            "current_revision_id": self.current_revision_id,
            "previous_fingerprint": self.previous_fingerprint,
            "current_fingerprint": self.current_fingerprint,
            "added_refs": self.added_refs,
            "removed_refs": self.removed_refs,
            "changed_refs": self.changed_refs,
            "unchanged_refs": self.unchanged_refs,
            "affected_refs": self.affected_refs(),
            "has_changes": self.has_changes,
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
    gold_mode: str
    query_style: str
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

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @property
    def fingerprint(self) -> str:
        """Stable digest for this run's observable retrieval output."""

        return _json_digest(
            {
                "name": self.name,
                "metrics": self.metrics,
                "diagnostics": self.diagnostics,
                "examples": [item.to_dict() for item in self.examples],
            }
        )


@dataclass(frozen=True)
class EvalSuite:
    examples: list[EvalExample]
    corpus: CorpusMap

    @classmethod
    def from_rows(cls, rows: Iterable[dict[str, Any]], *, corpus: CorpusMap) -> "EvalSuite":
        return cls([EvalExample.from_dict(row) for row in rows], corpus)

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        corpus: CorpusMap,
        attach_source_hashes: bool = False,
    ) -> "EvalSuite":
        """Load JSONL rows with `query` and `gold_refs` fields."""

        rows: list[dict[str, Any]] = []
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        suite = cls.from_rows(rows, corpus=corpus)
        return suite.with_source_hashes() if attach_source_hashes else suite

    def with_source_hashes(self, *, preserve_existing: bool = True) -> "EvalSuite":
        return EvalSuite(
            [example.with_source_hashes(self.corpus, preserve_existing=preserve_existing) for example in self.examples],
            self.corpus,
        )

    def to_jsonl(self, path: str | Path) -> None:
        """Persist eval rows, including source hashes, for later stale checks."""

        lines = [json.dumps(example.to_dict(), ensure_ascii=False) for example in self.examples]
        Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

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

    @property
    def fingerprint(self) -> str:
        """Stable digest for eval queries, gold refs, source hashes, and metadata."""

        return _json_digest([example.to_dict() for example in self.examples])

    def summary(self) -> dict[str, Any]:
        """Portable eval-suite metadata used to keep runs comparable."""

        query_styles = Counter(_query_style(example) for example in self.examples)
        gold_modes = Counter(_gold_mode(example, self.corpus.expand_refs(example.gold_refs)) for example in self.examples)
        return {
            "schema": "refmark.eval_suite_summary.v1",
            "fingerprint": self.fingerprint,
            "example_count": len(self.examples),
            "query_styles": dict(sorted(query_styles.items())),
            "gold_modes": dict(sorted(gold_modes.items())),
            "source_hash_coverage": round(
                sum(1 for example in self.examples if example.source_hashes) / max(len(self.examples), 1),
                4,
            ),
        }

    def run_artifact(
        self,
        run: EvalRun,
        *,
        settings: dict[str, Any] | None = None,
        artifacts: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a self-contained comparable evidence-eval artifact."""

        return {
            "schema": "refmark.eval_run_artifact.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_name": run.name,
            "run_fingerprint": run.fingerprint,
            "comparison_key": _json_digest(
                {
                    "corpus_fingerprint": self.corpus.fingerprint,
                    "eval_suite_fingerprint": self.fingerprint,
                    "settings": settings or {},
                    "run_name": run.name,
                }
            ),
            "corpus": self.corpus.snapshot().to_dict(),
            "eval_suite": self.summary(),
            "settings": dict(settings or {}),
            "artifacts": dict(artifacts or {}),
            "metrics": run.metrics,
            "diagnostics": run.diagnostics,
            "results": [item.to_dict() for item in run.examples],
        }


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
        gold_mode=_gold_mode(example, gold_refs),
        query_style=_query_style(example),
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


def _summarize_by_gold_mode(results: list[EvalExampleResult]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[EvalExampleResult]] = defaultdict(list)
    for result in results:
        grouped[result.gold_mode].append(result)
    return {mode: _summarize_results(items) for mode, items in sorted(grouped.items())}


def _summarize_by_query_style(results: list[EvalExampleResult]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[EvalExampleResult]] = defaultdict(list)
    for result in results:
        grouped[result.query_style].append(result)
    return {style: _summarize_results(items) for style, items in sorted(grouped.items())}


def query_style_gap(results: list[EvalExampleResult]) -> dict[str, Any]:
    """Return cross-style deltas so easy direct queries cannot hide hard styles."""

    by_style = _summarize_by_query_style(results)
    return {
        "styles": sorted(by_style),
        "hit_at_1_gap": _metric_gap(by_style, "hit_at_1"),
        "hit_at_k_gap": _metric_gap(by_style, "hit_at_k"),
        "mrr_gap": _metric_gap(by_style, "mrr"),
        "weakest_by_hit_at_1": _weakest_metric(by_style, "hit_at_1"),
        "weakest_by_hit_at_k": _weakest_metric(by_style, "hit_at_k"),
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
        "by_gold_mode": _summarize_by_gold_mode(results),
        "by_query_style": _summarize_by_query_style(results),
        "query_style_gap": query_style_gap(results),
        "selective_jump": selective_jump_diagnostics(results),
        "adaptation": adaptation_recommendations(heatmap),
    }


def _gold_mode(example: EvalExample, expanded_gold_refs: list[str]) -> str:
    mode = example.metadata.get("gold_mode")
    if mode:
        return str(mode)
    citations = list(parse_citation_refs(example.gold_refs))
    if any(citation.is_range for citation in citations):
        return "range"
    if len(expanded_gold_refs) <= 1:
        return "single"
    return "distributed"


def _query_style(example: EvalExample) -> str:
    for key in ("query_style", "variant", "style", "source"):
        value = example.metadata.get(key)
        if value:
            return str(value)
    return "unspecified"


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
                    "query_style": result.query_style,
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
    """Turn heatmap symptoms into concrete next adaptation actions.

    These are recommendations, not automatic corpus mutations. Refmark keeps the
    evidence address space explicit so a review loop can decide whether a weak
    area needs eval-data repair, region-boundary changes, exclusion/role marks,
    or retriever/ranker tuning.
    """

    recommendations: list[dict[str, Any]] = []
    for row in heatmap.get("hard_refs", []):
        if row.get("miss_at_k", 0) <= 0:
            continue
        action = "add_or_rewrite_eval_queries"
        adaptation_type = "validation"
        if row.get("hit_at_k", 1.0) == 0.0:
            action = "review_gold_refs_or_region_boundaries"
            adaptation_type = "region_or_validation"
        recommendations.append(
            {
                "ref": row["ref"],
                "adaptation_type": adaptation_type,
                "action": action,
                "reason": f"missed {row['miss_at_k']} of {row['count']} eval queries at k",
                "sample_queries": [sample["query"] for sample in row.get("sample_misses", [])],
                "candidate_actions": [
                    "rewrite_drifted_questions",
                    "add_valid_alternate_gold_refs",
                    "split_broad_gold_range",
                    "extend_gold_range_to_retrieved_neighbor",
                    "mark_query_magnet_or_hub_for_exclusion",
                ],
            }
        )
    for row in heatmap.get("confusions", []):
        recommendations.append(
            {
                "ref": row["gold_ref"],
                "adaptation_type": "confusion_mapping",
                "action": "record_confusion_pair_and_review_resolution",
                "reason": f"wrong top ref {row['top_ref']} appeared {row['count']} times",
                "confused_with_ref": row["top_ref"],
                "candidate_actions": [
                    "add_hard_negative_or_disambiguating_signature",
                    "add_alternate_gold_if_competing_ref_is_valid",
                    "merge_or_link_equivalent_sections",
                    "tune_retriever_or_reranker_for_confusion_pair",
                ],
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


def _json_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _literal_refs_from_gold(gold_refs: Iterable[str]) -> list[str]:
    refs: list[str] = []
    for citation in parse_citation_refs(gold_refs):
        refs.append(citation.stable_ref)
        if citation.stable_end_ref:
            refs.append(citation.stable_end_ref)
    return _dedupe(refs)


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _metric_gap(by_group: dict[str, dict[str, float]], metric: str) -> float:
    values = [float(row.get(metric, 0.0)) for row in by_group.values()]
    return max(values) - min(values) if values else 0.0


def _weakest_metric(by_group: dict[str, dict[str, float]], metric: str) -> dict[str, Any] | None:
    if not by_group:
        return None
    style, metrics = min(by_group.items(), key=lambda item: (float(item[1].get(metric, 0.0)), item[0]))
    return {"style": style, "value": float(metrics.get(metric, 0.0))}


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
