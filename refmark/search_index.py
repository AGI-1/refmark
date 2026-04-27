"""Portable Refmark search indexes for documentation-style corpora."""

from __future__ import annotations

import concurrent.futures
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Iterable
from urllib import request

from refmark.document_io import extract_document_text, text_mapping_extension
from refmark.pipeline import RegionRecord, build_region_manifest


SUPPORTED_EXTENSIONS = {".txt", ".md", ".rst", ".html", ".htm", ".docx", ".pdf"}
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


@dataclass(frozen=True)
class RetrievalView:
    summary: str
    questions: list[str]
    keywords: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SearchRegion:
    doc_id: str
    region_id: str
    text: str
    hash: str
    source_path: str | None
    ordinal: int
    prev_region_id: str | None
    next_region_id: str | None
    view: RetrievalView

    @property
    def stable_ref(self) -> str:
        return f"{self.doc_id}:{self.region_id}"

    def index_text(self, *, include_source: bool = True) -> str:
        parts: list[str] = []
        if include_source:
            parts.append(self.text)
        if self.view.summary:
            parts.append(self.view.summary)
        parts.extend(self.view.questions)
        parts.extend(self.view.keywords)
        return "\n".join(part for part in parts if part)

    def to_dict(self) -> dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "region_id": self.region_id,
            "stable_ref": self.stable_ref,
            "text": self.text,
            "hash": self.hash,
            "source_path": self.source_path,
            "ordinal": self.ordinal,
            "prev_region_id": self.prev_region_id,
            "next_region_id": self.next_region_id,
            "view": self.view.to_dict(),
        }


@dataclass(frozen=True)
class SearchHit:
    rank: int
    score: float
    doc_id: str
    region_id: str
    stable_ref: str
    text: str
    summary: str
    source_path: str | None
    context_refs: list[str]
    matched_doc_score: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class PortableBM25Index:
    """Small in-process BM25 index over enriched Refmark regions."""

    def __init__(self, regions: list[SearchRegion], *, include_source: bool = True, k1: float = 1.5, b: float = 0.75):
        self.regions = regions
        self.include_source = include_source
        self.k1 = k1
        self.b = b
        self.region_tokens = [tokenize(region.index_text(include_source=include_source)) for region in regions]
        self.region_counts = [Counter(tokens) for tokens in self.region_tokens]
        self.region_lengths = [len(tokens) for tokens in self.region_tokens]
        self.avg_len = sum(self.region_lengths) / max(len(self.region_lengths), 1)
        doc_freq = _document_frequency(self.region_counts)
        count = max(len(regions), 1)
        self.idf = {
            token: math.log(((count - freq + 0.5) / (freq + 0.5)) + 1.0)
            for token, freq in doc_freq.items()
        }
        self.postings: dict[str, list[int]] = defaultdict(list)
        for index, counts in enumerate(self.region_counts):
            for token in counts:
                self.postings[token].append(index)
        self.doc_indices: dict[str, list[int]] = defaultdict(list)
        for index, region in enumerate(regions):
            self.doc_indices[region.doc_id].append(index)
        self.doc_ids = sorted(self.doc_indices)
        self.doc_counts = [_merge_counts(self.region_counts[index] for index in self.doc_indices[doc_id]) for doc_id in self.doc_ids]
        self.doc_lengths = [sum(counts.values()) for counts in self.doc_counts]
        self.doc_avg_len = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        doc_unit_freq = _document_frequency(self.doc_counts)
        doc_count = max(len(self.doc_ids), 1)
        self.doc_idf = {
            token: math.log(((doc_count - freq + 0.5) / (freq + 0.5)) + 1.0)
            for token, freq in doc_unit_freq.items()
        }
        self.doc_postings: dict[str, list[int]] = defaultdict(list)
        for index, counts in enumerate(self.doc_counts):
            for token in counts:
                self.doc_postings[token].append(index)

    def search(self, query: str, *, top_k: int = 5, expand_before: int = 0, expand_after: int = 0) -> list[SearchHit]:
        scored = self._rank_regions(query, top_k=top_k)
        return [
            self._hit(rank, region_index, score, expand_before=expand_before, expand_after=expand_after, doc_score=None)
            for rank, (region_index, score) in enumerate(scored[:top_k], start=1)
        ]

    def search_hierarchical(
        self,
        query: str,
        *,
        top_k: int = 5,
        doc_top_k: int = 5,
        candidate_k: int = 50,
        expand_before: int = 0,
        expand_after: int = 0,
    ) -> list[SearchHit]:
        doc_scores = self._rank_docs(query, top_k=doc_top_k)
        candidate_indices = {
            region_index
            for doc_index, _score in doc_scores
            for region_index in self.doc_indices[self.doc_ids[doc_index]]
        }
        region_scores = self._rank_regions(query, top_k=candidate_k, candidate_indices=candidate_indices)
        doc_score_by_id = {self.doc_ids[index]: score for index, score in doc_scores}
        return [
            self._hit(
                rank,
                region_index,
                score,
                expand_before=expand_before,
                expand_after=expand_after,
                doc_score=doc_score_by_id.get(self.regions[region_index].doc_id),
            )
            for rank, (region_index, score) in enumerate(region_scores[:top_k], start=1)
        ]

    def search_reranked(
        self,
        query: str,
        *,
        top_k: int = 5,
        candidate_k: int = 30,
        expand_before: int = 0,
        expand_after: int = 0,
    ) -> list[SearchHit]:
        candidates = self._rank_regions(query, top_k=candidate_k)
        if not candidates:
            return []
        max_bm25 = max(score for _index, score in candidates) or 1.0
        query_terms = set(tokenize(query))
        reranked = []
        for region_index, bm25_score in candidates:
            region = self.regions[region_index]
            text_terms = set(tokenize(region.text))
            metadata_terms = set(tokenize("\n".join([region.view.summary, *region.view.questions, *region.view.keywords])))
            text_cover = len(query_terms & text_terms) / max(len(query_terms), 1)
            metadata_cover = len(query_terms & metadata_terms) / max(len(query_terms), 1)
            bigram_bonus = _bigram_overlap(query, region.index_text(include_source=self.include_source))
            score = (bm25_score / max_bm25) + (0.18 * text_cover) + (0.12 * metadata_cover) + (0.08 * bigram_bonus)
            reranked.append((region_index, score))
        reranked.sort(key=lambda item: (-item[1], self.regions[item[0]].stable_ref))
        return [
            self._hit(rank, region_index, score, expand_before=expand_before, expand_after=expand_after, doc_score=None)
            for rank, (region_index, score) in enumerate(reranked[:top_k], start=1)
        ]

    def _rank_regions(
        self,
        query: str,
        *,
        top_k: int,
        candidate_indices: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        query_terms = set(tokenize(query))
        if candidate_indices is None:
            candidate_indices = {idx for token in query_terms for idx in self.postings.get(token, [])}
        scored = _score_bm25(
            query_terms,
            candidate_indices,
            counts=self.region_counts,
            lengths=self.region_lengths,
            avg_len=self.avg_len,
            idf=self.idf,
            k1=self.k1,
            b=self.b,
        )
        scored.sort(key=lambda item: (-item[1], self.regions[item[0]].stable_ref))
        return scored[:top_k]

    def _rank_docs(self, query: str, *, top_k: int) -> list[tuple[int, float]]:
        query_terms = set(tokenize(query))
        candidate_indices = {idx for token in query_terms for idx in self.doc_postings.get(token, [])}
        scored = _score_bm25(
            query_terms,
            candidate_indices,
            counts=self.doc_counts,
            lengths=self.doc_lengths,
            avg_len=self.doc_avg_len,
            idf=self.doc_idf,
            k1=self.k1,
            b=self.b,
        )
        scored.sort(key=lambda item: (-item[1], self.doc_ids[item[0]]))
        return scored[:top_k]

    def _hit(
        self,
        rank: int,
        region_index: int,
        score: float,
        *,
        expand_before: int,
        expand_after: int,
        doc_score: float | None,
    ) -> SearchHit:
        region = self.regions[region_index]
        context_refs = _context_refs(self.regions, region_index, before=expand_before, after=expand_after)
        return SearchHit(
            rank=rank,
            score=round(score, 6),
            doc_id=region.doc_id,
            region_id=region.region_id,
            stable_ref=region.stable_ref,
            text=region.text,
            summary=region.view.summary,
            source_path=region.source_path,
            context_refs=context_refs,
            matched_doc_score=round(doc_score, 6) if doc_score is not None else None,
        )


def build_search_index(
    corpus_path: str | Path,
    output_path: str | Path,
    *,
    source: str = "local",
    model: str = "mistralai/mistral-nemo",
    endpoint: str = OPENROUTER_CHAT_URL,
    api_key_env: str = "OPENROUTER_API_KEY",
    extensions: Iterable[str] = SUPPORTED_EXTENSIONS,
    marker_format: str = "typed_bracket",
    chunker: str = "paragraph",
    tokens_per_chunk: int | None = None,
    lines_per_chunk: int | None = None,
    min_words: int = 8,
    questions_per_region: int = 4,
    keywords_per_region: int = 8,
    include_source: bool = True,
    limit: int | None = None,
    concurrency: int = 4,
    sleep: float = 0.0,
    view_cache_path: str | Path | None = None,
    exclude_globs: Iterable[str] = (),
) -> dict[str, object]:
    """Build a portable JSON search index from a file or directory corpus."""
    source_path = Path(corpus_path)
    output = Path(output_path)
    records = map_corpus(
        source_path,
        extensions=extensions,
        marker_format=marker_format,
        chunker=chunker,
        tokens_per_chunk=tokens_per_chunk,
        lines_per_chunk=lines_per_chunk,
        min_words=min_words,
        exclude_globs=exclude_globs,
    )
    if limit is not None:
        records = records[:limit]
    views = generate_views(
        records,
        source=source,
        model=model,
        endpoint=endpoint,
        api_key_env=api_key_env,
        questions_per_region=questions_per_region,
        keywords_per_region=keywords_per_region,
        concurrency=concurrency,
        sleep=sleep,
        cache_path=view_cache_path,
    )
    regions = [
        SearchRegion(
            doc_id=record.doc_id,
            region_id=record.region_id,
            text=record.text,
            hash=record.hash,
            source_path=record.source_path,
            ordinal=record.ordinal,
            prev_region_id=record.prev_region_id,
            next_region_id=record.next_region_id,
            view=views[(record.doc_id, record.region_id)],
        )
        for record in records
    ]
    approx_input_tokens = sum(approx_tokens(record.text) for record in records)
    approx_output_tokens = sum(
        approx_tokens(region.view.summary)
        + sum(approx_tokens(question) for question in region.view.questions)
        + sum(approx_tokens(keyword) for keyword in region.view.keywords)
        for region in regions
    )
    payload = {
        "schema": "refmark.portable_search_index.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_corpus": str(source_path),
        "settings": {
            "view_source": source,
            "model": model if source == "openrouter" else source,
            "marker_format": marker_format,
            "chunker": chunker,
            "tokens_per_chunk": tokens_per_chunk,
            "lines_per_chunk": lines_per_chunk,
            "min_words": min_words,
            "questions_per_region": questions_per_region,
            "keywords_per_region": keywords_per_region,
            "include_source_in_index": include_source,
            "exclude_globs": list(exclude_globs),
        },
        "stats": {
            "documents": len({record.doc_id for record in records}),
            "regions": len(regions),
            "approx_input_tokens": approx_input_tokens,
            "approx_output_tokens": approx_output_tokens,
            "approx_openrouter_cost_usd_at_mistral_nemo": round(
                (approx_input_tokens / 1_000_000 * 0.01) + (approx_output_tokens / 1_000_000 * 0.03),
                4,
            ),
        },
        "regions": [region.to_dict() for region in regions],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def export_browser_search_index(
    index_path: str | Path,
    output_path: str | Path,
    *,
    include_text: bool = True,
    max_text_chars: int = 900,
) -> dict[str, object]:
    """Export a compact BM25 payload that can be searched directly in a browser."""
    index = load_search_index(index_path)
    postings: dict[str, dict[str, object]] = {}
    for token in sorted(index.postings):
        rows = []
        for region_index in index.postings[token]:
            tf = index.region_counts[region_index].get(token, 0)
            if tf > 0:
                rows.append([region_index, tf])
        if rows:
            postings[token] = {"idf": round(index.idf.get(token, 0.0), 8), "p": rows}
    payload = {
        "schema": "refmark.browser_search_index.v1",
        "source_index": str(index_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "k1": index.k1,
            "b": index.b,
            "include_source_in_index": index.include_source,
            "max_text_chars": max_text_chars if include_text else 0,
        },
        "stats": {
            "regions": len(index.regions),
            "documents": len({region.doc_id for region in index.regions}),
            "tokens": len(postings),
        },
        "avg_len": index.avg_len,
        "lengths": index.region_lengths,
        "regions": [
            _browser_region_payload(region, include_text=include_text, max_text_chars=max_text_chars)
            for region in index.regions
        ],
        "postings": postings,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    return payload


def load_search_index(path: str | Path) -> PortableBM25Index:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    settings = payload.get("settings", {})
    include_source = bool(settings.get("include_source_in_index", True))
    regions: list[SearchRegion] = []
    for row in payload.get("regions", []):
        view_payload = row.get("view", {}) if isinstance(row, dict) else {}
        view = RetrievalView(
            summary=str(view_payload.get("summary", "")),
            questions=[str(item) for item in view_payload.get("questions", [])],
            keywords=[str(item) for item in view_payload.get("keywords", [])],
        )
        regions.append(
            SearchRegion(
                doc_id=str(row["doc_id"]),
                region_id=str(row["region_id"]),
                text=str(row["text"]),
                hash=str(row["hash"]),
                source_path=row.get("source_path"),
                ordinal=int(row["ordinal"]),
                prev_region_id=row.get("prev_region_id"),
                next_region_id=row.get("next_region_id"),
                view=view,
            )
        )
    return PortableBM25Index(regions, include_source=include_source)


def _browser_region_payload(region: SearchRegion, *, include_text: bool, max_text_chars: int) -> dict[str, object]:
    payload = {
        "doc_id": region.doc_id,
        "region_id": region.region_id,
        "stable_ref": region.stable_ref,
        "summary": region.view.summary,
        "source_path": region.source_path,
        "ordinal": region.ordinal,
        "prev_region_id": region.prev_region_id,
        "next_region_id": region.next_region_id,
    }
    if include_text:
        text = " ".join(region.text.split())
        if max_text_chars > 0 and len(text) > max_text_chars:
            text = text[: max_text_chars - 3].rstrip() + "..."
        payload["text"] = text
    return payload


def map_corpus(
    corpus_path: str | Path,
    *,
    extensions: Iterable[str] = SUPPORTED_EXTENSIONS,
    marker_format: str = "typed_bracket",
    chunker: str = "paragraph",
    tokens_per_chunk: int | None = None,
    lines_per_chunk: int | None = None,
    min_words: int = 8,
    exclude_globs: Iterable[str] = (),
) -> list[RegionRecord]:
    """Extract documents and map them to Refmark regions."""
    wanted = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    paths = _corpus_files(Path(corpus_path), wanted)
    patterns = list(exclude_globs)
    if patterns:
        paths = [path for path in paths if not _matches_any_glob(path, patterns)]
    chunker_kwargs = {}
    if tokens_per_chunk is not None:
        chunker_kwargs["tokens_per_chunk"] = tokens_per_chunk
    if lines_per_chunk is not None:
        chunker_kwargs["lines_per_chunk"] = lines_per_chunk

    records: list[RegionRecord] = []
    used_doc_ids: Counter[str] = Counter()
    for path in paths:
        text = extract_document_text(path)
        doc_id = _doc_id(path, root=Path(corpus_path) if Path(corpus_path).is_dir() else path.parent)
        used_doc_ids[doc_id] += 1
        if used_doc_ids[doc_id] > 1:
            doc_id = f"{doc_id}_{used_doc_ids[doc_id]}"
        _marked, doc_records = build_region_manifest(
            text,
            text_mapping_extension(path),
            doc_id=doc_id,
            source_path=str(path),
            marker_format=marker_format,
            chunker=chunker,
            chunker_kwargs=chunker_kwargs or None,
            min_words=min_words,
        )
        records.extend(doc_records)
    return records


def _matches_any_glob(path: Path, patterns: list[str]) -> bool:
    normalized = path.as_posix()
    return any(fnmatch.fnmatch(normalized, pattern) or fnmatch.fnmatch(path.name, pattern) for pattern in patterns)


def generate_views(
    records: list[RegionRecord],
    *,
    source: str,
    model: str,
    endpoint: str,
    api_key_env: str,
    questions_per_region: int,
    keywords_per_region: int,
    concurrency: int,
    sleep: float,
    cache_path: str | Path | None = None,
) -> dict[tuple[str, str], RetrievalView]:
    if source == "local":
        return {
            (record.doc_id, record.region_id): local_view(
                record.text,
                questions_per_region=questions_per_region,
                keywords_per_region=keywords_per_region,
            )
            for record in records
        }
    if source != "openrouter":
        raise ValueError("source must be 'local' or 'openrouter'.")
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set.")
    cache = _read_view_cache(cache_path)
    output: dict[tuple[str, str], RetrievalView] = {}
    missing = []
    for record in records:
        cache_key = _view_cache_key(record, source=source, model=model)
        if cache_key in cache:
            output[(record.doc_id, record.region_id)] = cache[cache_key]
        else:
            missing.append(record)

    def generate(record: RegionRecord) -> tuple[tuple[str, str], RetrievalView]:
        if sleep:
            time.sleep(sleep)
        return (
            (record.doc_id, record.region_id),
            openrouter_view(
                record,
                model=model,
                endpoint=endpoint,
                api_key=api_key,
                questions_per_region=questions_per_region,
                keywords_per_region=keywords_per_region,
            ),
        )

    workers = max(1, concurrency)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        generated_rows = []
        for key, view in executor.map(generate, missing):
            output[key] = view
            doc_id, region_id = key
            generated_rows.append(
                {
                    "doc_id": doc_id,
                    "region_id": region_id,
                    "stable_ref": f"{doc_id}:{region_id}",
                    "hash": next(record.hash for record in missing if record.doc_id == doc_id and record.region_id == region_id),
                    "source": source,
                    "model": model,
                    "view": view.to_dict(),
                }
            )
        _append_view_cache(cache_path, generated_rows)
    return output


def local_view(text: str, *, questions_per_region: int = 4, keywords_per_region: int = 8) -> RetrievalView:
    keywords = keywords_for(text, limit=keywords_per_region)
    summary = extractive_summary(text)
    topic = ", ".join(keywords[:3]) if keywords else "this region"
    templates = [
        f"What does this section say about {topic}?",
        f"Where is {topic} described?",
        f"What details are given for {topic}?",
        f"Which source region explains {topic}?",
    ]
    return RetrievalView(
        summary=summary,
        questions=templates[:questions_per_region],
        keywords=keywords,
    )


def openrouter_view(
    record: RegionRecord,
    *,
    model: str,
    endpoint: str,
    api_key: str,
    questions_per_region: int,
    keywords_per_region: int,
) -> RetrievalView:
    prompt = f"""Create retrieval metadata for this documentation region.

Return strict JSON with keys:
- summary: one short sentence
- questions: {questions_per_region} natural user questions this region could answer
- keywords: {keywords_per_region} concise retrieval keywords or phrases

Do not include markdown fences.

Document: {record.doc_id}
Region: {record.region_id}
Text:
{record.text}
"""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You create concise retrieval metadata for search indexes."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 320,
    }
    payload = json.dumps(body).encode("utf-8")
    req = request.Request(
        endpoint,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/b-imenitov/refmark",
            "X-Title": "refmark portable search index",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=90) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
    content = response_payload["choices"][0]["message"]["content"]
    try:
        parsed = _parse_json_object(content)
    except (json.JSONDecodeError, ValueError):
        return local_view(record.text, questions_per_region=questions_per_region, keywords_per_region=keywords_per_region)
    return RetrievalView(
        summary=str(parsed.get("summary", "")).strip(),
        questions=[str(item).strip() for item in parsed.get("questions", []) if str(item).strip()][:questions_per_region],
        keywords=[str(item).strip() for item in parsed.get("keywords", []) if str(item).strip()][:keywords_per_region],
    )


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def keywords_for(text: str, *, limit: int) -> list[str]:
    stopwords = {
        "the", "and", "for", "that", "with", "this", "from", "into", "are", "was", "were",
        "has", "have", "not", "you", "can", "will", "may", "when", "where", "which", "what",
        "using", "used", "use", "all", "any", "but", "its", "their", "there", "then", "than",
        "your", "our", "about", "section", "document",
    }
    counts = Counter(token for token in tokenize(text) if len(token) > 2 and token not in stopwords)
    return [token for token, _count in counts.most_common(limit)]


def extractive_summary(text: str, *, max_words: int = 42) -> str:
    clean = " ".join(text.split())
    words = clean.split()
    if len(words) <= max_words:
        return clean
    return " ".join(words[:max_words])


def approx_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _corpus_files(path: Path, extensions: set[str]) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(path)
    return sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in extensions)


def _doc_id(path: Path, *, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path.name
    stem = str(relative).replace("\\", "/")
    stem = re.sub(r"\.[^.]+$", "", stem)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()
    return slug or hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]


def _context_refs(regions: list[SearchRegion], index: int, *, before: int, after: int) -> list[str]:
    region = regions[index]
    start = index
    while start > 0 and before > 0 and regions[start - 1].doc_id == region.doc_id:
        start -= 1
        before -= 1
    end = index + 1
    while end < len(regions) and after > 0 and regions[end].doc_id == region.doc_id:
        end += 1
        after -= 1
    return [item.stable_ref for item in regions[start:end]]


def _document_frequency(counts_by_unit: list[Counter[str]]) -> Counter[str]:
    doc_freq: Counter[str] = Counter()
    for counts in counts_by_unit:
        doc_freq.update(counts.keys())
    return doc_freq


def _merge_counts(counts_list: Iterable[Counter[str]]) -> Counter[str]:
    merged: Counter[str] = Counter()
    for counts in counts_list:
        merged.update(counts)
    return merged


def _score_bm25(
    query_terms: set[str],
    candidate_indices: set[int],
    *,
    counts: list[Counter[str]],
    lengths: list[int],
    avg_len: float,
    idf: dict[str, float],
    k1: float,
    b: float,
) -> list[tuple[int, float]]:
    scored: list[tuple[int, float]] = []
    for index in candidate_indices:
        unit_counts = counts[index]
        length = lengths[index]
        score = 0.0
        for token in query_terms:
            tf = unit_counts.get(token, 0)
            if tf <= 0:
                continue
            norm = k1 * (1.0 - b + b * (length / max(avg_len, 1e-6)))
            score += idf.get(token, 0.0) * ((tf * (k1 + 1.0)) / (tf + norm))
        if score > 0.0:
            scored.append((index, score))
    return scored


def _bigram_overlap(query: str, text: str) -> float:
    query_tokens = tokenize(query)
    if len(query_tokens) < 2:
        return 0.0
    query_bigrams = set(zip(query_tokens, query_tokens[1:], strict=False))
    text_tokens = tokenize(text)
    text_bigrams = set(zip(text_tokens, text_tokens[1:], strict=False))
    return len(query_bigrams & text_bigrams) / max(len(query_bigrams), 1)


def _view_cache_key(record: RegionRecord, *, source: str, model: str) -> tuple[str, str, str, str]:
    return (f"{record.doc_id}:{record.region_id}", record.hash, source, model)


def _read_view_cache(path: str | Path | None) -> dict[tuple[str, str, str, str], RetrievalView]:
    if path is None:
        return {}
    source = Path(path)
    if not source.exists():
        return {}
    cache: dict[tuple[str, str, str, str], RetrievalView] = {}
    for line in source.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        view_payload = row.get("view", {})
        cache[(str(row["stable_ref"]), str(row["hash"]), str(row["source"]), str(row["model"]))] = RetrievalView(
            summary=str(view_payload.get("summary", "")),
            questions=[str(item) for item in view_payload.get("questions", [])],
            keywords=[str(item) for item in view_payload.get("keywords", [])],
        )
    return cache


def _append_view_cache(path: str | Path | None, rows: list[dict[str, object]]) -> None:
    if path is None or not rows:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_json_object(text: str) -> dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end < start:
            raise
        payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("OpenRouter response did not contain a JSON object.")
    return payload
