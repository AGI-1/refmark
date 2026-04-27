"""Compare Refmark navigation search against local lexical/vector baselines."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from urllib import error
from urllib import request

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.portable_search_index.evaluate_real_corpus import _read_question_cache  # noqa: E402
from refmark.search_index import PortableBM25Index, SearchRegion, load_search_index, tokenize  # noqa: E402


@dataclass(frozen=True)
class RankedHit:
    stable_ref: str
    doc_id: str
    score: float


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare app-doc navigation search approaches with Refmark labels.")
    parser.add_argument("raw_index", help="Portable index using raw source text only.")
    parser.add_argument("enriched_index", help="Portable index with generated Refmark retrieval views.")
    parser.add_argument("--question-cache", required=True)
    parser.add_argument("--output", default="examples/portable_search_index/output/navigation_compare.json")
    parser.add_argument("--top-ks", default="1,3,5,10")
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--real-embeddings", action="store_true", help="Also compare OpenRouter embedding search.")
    parser.add_argument("--embedding-model", default="openai/text-embedding-3-small")
    parser.add_argument("--embedding-endpoint", default="https://openrouter.ai/api/v1/embeddings")
    parser.add_argument("--embedding-cache", default="examples/portable_search_index/output/embedding_cache.jsonl")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-max-chars", type=int, default=8000)
    parser.add_argument("--hybrid-weights", default="0.1,0.25,0.35,0.5,0.65,0.75,0.9")
    parser.add_argument(
        "--warm-query-embeddings",
        action="store_true",
        help="Embed all evaluation queries before measuring search latency.",
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    top_ks = tuple(int(part) for part in args.top_ks.split(",") if part.strip())
    hybrid_weights = tuple(float(part) for part in args.hybrid_weights.split(",") if part.strip())
    raw = load_search_index(args.raw_index)
    enriched = load_search_index(args.enriched_index)
    questions = _questions_for_index(enriched, Path(args.question_cache))
    raw_embed = HashedEmbeddingIndex(raw.regions, dim=args.embedding_dim, include_view=False)
    enriched_embed = HashedEmbeddingIndex(enriched.regions, dim=args.embedding_dim, include_view=True)

    methods = {
        "raw_bm25": lambda query: _bm25_hits(raw, query, args.candidate_k),
        "refmark_bm25": lambda query: _bm25_hits(enriched, query, args.candidate_k),
        "raw_hashed_embedding": lambda query: raw_embed.search(query, top_k=args.candidate_k),
        "refmark_hashed_embedding": lambda query: enriched_embed.search(query, top_k=args.candidate_k),
        "raw_bm25_plus_embedding": lambda query: _hybrid(
            _bm25_hits(raw, query, args.candidate_k),
            raw_embed.search(query, top_k=args.candidate_k),
        ),
        "refmark_bm25_plus_embedding": lambda query: _hybrid(
            _bm25_hits(enriched, query, args.candidate_k),
            enriched_embed.search(query, top_k=args.candidate_k),
        ),
    }
    if args.real_embeddings:
        raw_real_embed = OpenRouterEmbeddingIndex(
            raw.regions,
            model=args.embedding_model,
            endpoint=args.embedding_endpoint,
            api_key=os.environ.get(args.api_key_env, ""),
            cache_path=Path(args.embedding_cache),
            batch_size=args.embedding_batch_size,
            max_chars=args.embedding_max_chars,
            include_view=False,
        )
        enriched_real_embed = OpenRouterEmbeddingIndex(
            enriched.regions,
            model=args.embedding_model,
            endpoint=args.embedding_endpoint,
            api_key=os.environ.get(args.api_key_env, ""),
            cache_path=Path(args.embedding_cache),
            batch_size=args.embedding_batch_size,
            max_chars=args.embedding_max_chars,
            include_view=True,
        )
        if args.warm_query_embeddings:
            raw_real_embed.warm_queries([question.query for question in questions])
            enriched_real_embed.warm_queries([question.query for question in questions])
        methods["raw_openrouter_embedding"] = lambda query: raw_real_embed.search(query, top_k=args.candidate_k)
        methods["refmark_openrouter_embedding"] = lambda query: enriched_real_embed.search(query, top_k=args.candidate_k)
        methods["raw_bm25_plus_openrouter_embedding"] = lambda query: _hybrid(
            _bm25_hits(raw, query, args.candidate_k),
            raw_real_embed.search(query, top_k=args.candidate_k),
        )
        methods["refmark_bm25_plus_openrouter_embedding"] = lambda query: _hybrid(
            _bm25_hits(enriched, query, args.candidate_k),
            enriched_real_embed.search(query, top_k=args.candidate_k),
        )
        for weight in hybrid_weights:
            method_name = f"refmark_bm25_plus_openrouter_embedding_w{weight:g}"
            methods[method_name] = (
                lambda query, weight=weight: _hybrid(
                    _bm25_hits(enriched, query, args.candidate_k),
                    enriched_real_embed.search(query, top_k=args.candidate_k),
                    first_weight=weight,
                )
            )
    reports = {name: evaluate(search_fn, questions, top_ks) for name, search_fn in methods.items()}
    report = {
        "raw_index": args.raw_index,
        "enriched_index": args.enriched_index,
        "question_cache": args.question_cache,
        "questions": len(questions),
        "regions": len(enriched.regions),
        "documents": len({region.doc_id for region in enriched.regions}),
        "embedding_dim": args.embedding_dim,
        "real_embedding_model": args.embedding_model if args.real_embeddings else None,
        "embedding_queries_warmed": bool(args.real_embeddings and args.warm_query_embeddings),
        "latency_note": (
            "Accuracy fields are provider/cache independent. Latency fields include the configured execution path; "
            "remote embedding latency is only comparable when query embeddings are warmed consistently."
        ),
        "reports": reports,
        "accuracy_reports": {name: _accuracy_only(result) for name, result in reports.items()},
        "latency_reports": {name: _latency_only(result) for name, result in reports.items()},
        "best_hybrid": _best_hybrid(reports),
        "coarse_article_reports": {
            name: evaluate_coarse(search_fn, questions, top_ks, mode="article")
            for name, search_fn in methods.items()
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


class HashedEmbeddingIndex:
    def __init__(self, regions: list[SearchRegion], *, dim: int, include_view: bool) -> None:
        self.regions = regions
        self.dim = dim
        rows = [_hashed_tfidf(_region_text(region, include_view=include_view), dim=dim) for region in regions]
        self.matrix = np.vstack(rows).astype(np.float32) if rows else np.zeros((0, dim), dtype=np.float32)
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        self.matrix = np.divide(self.matrix, np.maximum(norms, 1e-9))

    def search(self, query: str, *, top_k: int) -> list[RankedHit]:
        if self.matrix.size == 0:
            return []
        start = time.perf_counter()
        query_vec = _hashed_tfidf(query, dim=self.dim)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm
        scores = self.matrix @ query_vec
        count = min(top_k, len(scores))
        if count <= 0:
            return []
        indices = np.argpartition(-scores, count - 1)[:count]
        ordered = sorted(indices.tolist(), key=lambda idx: (-float(scores[idx]), self.regions[idx].stable_ref))
        _elapsed = time.perf_counter() - start
        return [
            RankedHit(stable_ref=self.regions[index].stable_ref, doc_id=self.regions[index].doc_id, score=float(scores[index]))
            for index in ordered
        ]


class OpenRouterEmbeddingIndex:
    def __init__(
        self,
        regions: list[SearchRegion],
        *,
        model: str,
        endpoint: str,
        api_key: str,
        cache_path: Path,
        batch_size: int,
        max_chars: int,
        include_view: bool,
    ) -> None:
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")
        self.regions = regions
        self.model = model
        self.endpoint = endpoint
        self.api_key = api_key
        self.cache_path = cache_path
        self.batch_size = max(batch_size, 1)
        self.max_chars = max_chars
        self.include_view = include_view
        self.cache = _read_embedding_cache(cache_path)
        vectors = self._region_vectors()
        self.matrix = np.vstack(vectors).astype(np.float32) if vectors else np.zeros((0, 1), dtype=np.float32)
        self.matrix = _normalize_matrix(self.matrix)

    def search(self, query: str, *, top_k: int) -> list[RankedHit]:
        query_vec = np.asarray(self._embed_text(query, input_type="search_query"), dtype=np.float32)
        query_vec = _normalize_matrix(query_vec.reshape(1, -1))[0]
        scores = self.matrix @ query_vec
        count = min(top_k, len(scores))
        if count <= 0:
            return []
        indices = np.argpartition(-scores, count - 1)[:count]
        ordered = sorted(indices.tolist(), key=lambda idx: (-float(scores[idx]), self.regions[idx].stable_ref))
        return [
            RankedHit(stable_ref=self.regions[index].stable_ref, doc_id=self.regions[index].doc_id, score=float(scores[index]))
            for index in ordered
        ]

    def warm_queries(self, queries: list[str]) -> None:
        for query in queries:
            self._embed_text(query, input_type="search_query")

    def _region_vectors(self) -> list[list[float]]:
        vectors: list[list[float] | None] = []
        missing: list[tuple[int, SearchRegion, str, str]] = []
        for index, region in enumerate(self.regions):
            text = self._embedding_text(_region_text(region, include_view=self.include_view))
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            key = _embedding_key(self.model, "search_document", region.stable_ref, text_hash)
            cached = self.cache.get(key)
            if cached is None:
                vectors.append(None)
                missing.append((index, region, text, text_hash))
            else:
                vectors.append(cached)
        generated_rows = []
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            texts = [text for _index, _region, text, _hash in batch]
            embeddings = self._embed_batch(texts, input_type="search_document")
            for (index, region, _text, text_hash), vector in zip(batch, embeddings, strict=True):
                vectors[index] = vector
                key = _embedding_key(self.model, "search_document", region.stable_ref, text_hash)
                generated_rows.append(
                    {
                        "model": self.model,
                        "input_type": "search_document",
                        "stable_ref": region.stable_ref,
                        "hash": text_hash,
                        "embedding": vector,
                    }
                )
        _append_embedding_cache(self.cache_path, generated_rows)
        return [vector for vector in vectors if vector is not None]

    def _embed_text(self, text: str, *, input_type: str) -> list[float]:
        text = self._embedding_text(text)
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        key = _embedding_key(self.model, input_type, "__query__", text_hash)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        vector = self._embed_batch([text], input_type=input_type)[0]
        row = {
            "model": self.model,
            "input_type": input_type,
            "stable_ref": "__query__",
            "hash": text_hash,
            "embedding": vector,
        }
        _append_embedding_cache(self.cache_path, [row])
        self.cache[key] = vector
        return vector

    def _embed_batch(self, texts: list[str], *, input_type: str) -> list[list[float]]:
        texts = [self._embedding_text(text) for text in texts]
        body = {
            "model": self.model,
            "input": texts if len(texts) > 1 else texts[0],
            "input_type": input_type,
        }
        req = request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/b-imenitov/refmark",
                "X-Title": "refmark navigation compare",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=120) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if len(texts) > 1:
                return self._embed_split(texts, input_type=input_type)
            raise RuntimeError(f"Embedding request failed with HTTP {exc.code}: {detail}") from exc
        if "data" not in payload:
            if len(texts) > 1:
                return self._embed_split(texts, input_type=input_type)
            raise RuntimeError(f"Embedding response missing data: {json.dumps(payload)[:2000]}")
        rows = sorted(payload["data"], key=lambda item: int(item.get("index", 0)))
        if len(rows) != len(texts):
            raise RuntimeError(f"Embedding response row count mismatch: expected {len(texts)}, got {len(rows)}")
        return [[float(value) for value in row["embedding"]] for row in rows]

    def _embed_split(self, texts: list[str], *, input_type: str) -> list[list[float]]:
        if len(texts) == 1:
            return self._embed_batch(texts, input_type=input_type)
        midpoint = len(texts) // 2
        return [
            *self._embed_batch(texts[:midpoint], input_type=input_type),
            *self._embed_batch(texts[midpoint:], input_type=input_type),
        ]

    def _embedding_text(self, text: str) -> str:
        if self.max_chars <= 0 or len(text) <= self.max_chars:
            return text
        return text[: self.max_chars]


def evaluate(search_fn, questions, top_ks: tuple[int, ...]) -> dict[str, object]:
    max_k = max(top_ks)
    ref_hits = Counter({k: 0 for k in top_ks})
    doc_hits = Counter({k: 0 for k in top_ks})
    reciprocal = 0.0
    doc_reciprocal = 0.0
    latencies = []
    misses = []
    for question in questions:
        start = time.perf_counter()
        hits = search_fn(question.query)[:max_k]
        latencies.append(time.perf_counter() - start)
        gold_refs = set(question.gold_refs)
        ref_rank = None
        doc_rank = None
        for rank, hit in enumerate(hits, start=1):
            if ref_rank is None and hit.stable_ref in gold_refs:
                ref_rank = rank
            if doc_rank is None and hit.doc_id == question.doc_id:
                doc_rank = rank
        if ref_rank is not None:
            reciprocal += 1.0 / ref_rank
        else:
            misses.append({"query": question.query, "gold": question.gold_refs, "top_refs": [hit.stable_ref for hit in hits[:3]]})
        if doc_rank is not None:
            doc_reciprocal += 1.0 / doc_rank
        for k in top_ks:
            if ref_rank is not None and ref_rank <= k:
                ref_hits[k] += 1
            if doc_rank is not None and doc_rank <= k:
                doc_hits[k] += 1
    total = max(len(questions), 1)
    return {
        "anchor_hit_at_k": {str(k): round(ref_hits[k] / total, 4) for k in top_ks},
        "article_hit_at_k": {str(k): round(doc_hits[k] / total, 4) for k in top_ks},
        "anchor_mrr": round(reciprocal / total, 4),
        "article_mrr": round(doc_reciprocal / total, 4),
        "avg_latency_ms": round((sum(latencies) / max(len(latencies), 1)) * 1000, 4),
        "sample_misses": misses[:8],
    }


def evaluate_coarse(search_fn, questions, top_ks: tuple[int, ...], *, mode: str) -> dict[str, object]:
    max_k = max(top_ks)
    hits_by_k = Counter({k: 0 for k in top_ks})
    reciprocal = 0.0
    for question in questions:
        hits = search_fn(question.query)[:max_k]
        rank = None
        for idx, hit in enumerate(hits, start=1):
            if _coarse_match(hit, question, mode=mode):
                rank = idx
                break
        if rank is not None:
            reciprocal += 1.0 / rank
        for k in top_ks:
            if rank is not None and rank <= k:
                hits_by_k[k] += 1
    total = max(len(questions), 1)
    return {
        "hit_at_k": {str(k): round(hits_by_k[k] / total, 4) for k in top_ks},
        "mrr": round(reciprocal / total, 4),
        "mode": mode,
    }


def _coarse_match(hit: RankedHit, question, *, mode: str) -> bool:
    if mode == "article":
        return hit.doc_id == question.doc_id
    raise ValueError(f"Unsupported coarse mode: {mode}")


def _questions_for_index(index: PortableBM25Index, cache_path: Path):
    refs = {region.stable_ref for region in index.regions}
    questions = [
        question
        for question in _read_question_cache(cache_path).values()
        if set(question.gold_refs).issubset(refs)
    ]
    questions.sort(key=lambda item: (item.stable_ref, item.query))
    return questions


def _bm25_hits(index: PortableBM25Index, query: str, top_k: int) -> list[RankedHit]:
    return [
        RankedHit(stable_ref=hit.stable_ref, doc_id=hit.doc_id, score=hit.score)
        for hit in index.search(query, top_k=top_k)
    ]


def _hybrid(first: list[RankedHit], second: list[RankedHit], *, first_weight: float = 0.65) -> list[RankedHit]:
    scores: dict[str, float] = {}
    docs: dict[str, str] = {}
    second_weight = 1.0 - first_weight
    for weight, hits in ((first_weight, first), (second_weight, second)):
        for rank, hit in enumerate(hits, start=1):
            scores[hit.stable_ref] = scores.get(hit.stable_ref, 0.0) + (weight / (rank + 60.0))
            docs[hit.stable_ref] = hit.doc_id
    return [
        RankedHit(stable_ref=stable_ref, doc_id=docs[stable_ref], score=score)
        for stable_ref, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ]


def _accuracy_only(result: dict[str, object]) -> dict[str, object]:
    return {
        "anchor_hit_at_k": result["anchor_hit_at_k"],
        "article_hit_at_k": result["article_hit_at_k"],
        "anchor_mrr": result["anchor_mrr"],
        "article_mrr": result["article_mrr"],
        "sample_misses": result["sample_misses"],
    }


def _latency_only(result: dict[str, object]) -> dict[str, object]:
    return {"avg_latency_ms": result["avg_latency_ms"]}


def _best_hybrid(reports: dict[str, dict[str, object]]) -> dict[str, object] | None:
    hybrid_items = [
        (name, result)
        for name, result in reports.items()
        if name.startswith("refmark_bm25_plus_openrouter_embedding")
    ]
    if not hybrid_items:
        return None
    name, result = max(
        hybrid_items,
        key=lambda item: (
            float(item[1]["anchor_mrr"]),
            float(item[1]["anchor_hit_at_k"].get("1", 0.0)),
            float(item[1]["anchor_hit_at_k"].get("10", 0.0)),
        ),
    )
    return {"method": name, **_accuracy_only(result), **_latency_only(result)}


def _region_text(region: SearchRegion, *, include_view: bool) -> str:
    if not include_view:
        return region.text
    return "\n".join([region.text, region.view.summary, *region.view.questions, *region.view.keywords])


def _hashed_tfidf(text: str, *, dim: int) -> np.ndarray:
    counts = Counter(tokenize(text))
    vector = np.zeros(dim, dtype=np.float32)
    for token, count in counts.items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[bucket] += sign * (1.0 + math.log(count))
    return vector


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, np.maximum(norms, 1e-9))


def _embedding_key(model: str, input_type: str, stable_ref: str, text_hash: str) -> tuple[str, str, str, str]:
    return (model, input_type, stable_ref, text_hash)


def _read_embedding_cache(path: Path) -> dict[tuple[str, str, str, str], list[float]]:
    if not path.exists():
        return {}
    cache = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = _embedding_key(str(row["model"]), str(row["input_type"]), str(row["stable_ref"]), str(row["hash"]))
        cache[key] = [float(value) for value in row["embedding"]]
    return cache


def _append_embedding_cache(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
