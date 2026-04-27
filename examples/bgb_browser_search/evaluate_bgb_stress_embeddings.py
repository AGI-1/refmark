"""Evaluate BGB stress questions with Qdrant-backed embeddings.

Embeddings are treated as build/eval infrastructure. This script reuses the
existing OpenRouter embedding cache when possible, upserts current Refmark
regions into Qdrant, embeds stress queries in batches, and compares embedding
and hybrid retrieval against the same refmarked stress suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from collections import Counter
from dataclasses import dataclass
from urllib import request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.bgb_browser_search.build_bgb_article_navigation import article_id_for  # noqa: E402
from refmark.search_index import PortableBM25Index, SearchHit, SearchRegion, load_search_index  # noqa: E402


EMBEDDING_ENDPOINT = "https://openrouter.ai/api/v1/embeddings"


@dataclass(frozen=True)
class RankedHit:
    stable_ref: str
    score: float


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Evaluate BGB stress suite with Qdrant embeddings and hybrids.")
    parser.add_argument("--stress-report", required=True)
    parser.add_argument("--raw-index", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_raw_index.json")
    parser.add_argument("--enriched-index", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_openrouter_index.json")
    parser.add_argument("--embedding-cache", default="examples/bgb_browser_search/output_scratch_multi_full/embedding_cache_qwen3_8b.jsonl")
    parser.add_argument("--query-cache", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_stress_query_embeddings.jsonl")
    parser.add_argument("--output", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_stress_embedding_eval.json")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="bgb_refmark_qwen3_enriched")
    parser.add_argument("--embedding-model", default="qwen/qwen3-embedding-8b")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--top-ks", default="1,3,5,10,20,50")
    parser.add_argument("--hybrid-weights", default="0.02,0.05,0.1,0.2,0.35")
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set.")

    raw = load_search_index(args.raw_index)
    enriched = load_search_index(args.enriched_index)
    stable_refs = [region.stable_ref for region in enriched.regions]
    ref_to_region = {region.stable_ref: region for region in enriched.regions}
    stress = json.loads(Path(args.stress_report).read_text(encoding="utf-8"))
    questions = stress_questions(stress)
    top_ks = tuple(int(part) for part in args.top_ks.split(",") if part.strip())
    hybrid_weights = tuple(float(part) for part in args.hybrid_weights.split(",") if part.strip())

    ensure_collection(
        args.qdrant_url,
        args.collection,
        enriched.regions,
        embedding_cache=Path(args.embedding_cache),
        model=args.embedding_model,
        endpoint=EMBEDDING_ENDPOINT,
        api_key=api_key,
        recreate=args.recreate,
        batch_size=args.batch_size,
    )
    query_vectors = embed_queries(
        questions,
        model=args.embedding_model,
        endpoint=EMBEDDING_ENDPOINT,
        api_key=api_key,
        cache_path=Path(args.query_cache),
        batch_size=args.batch_size,
    )
    reports = {
        "raw_bm25": evaluate(lambda query, _vector: bm25_hits(raw, query, args.candidate_k), questions, top_ks=top_ks),
        "refmark_bm25": evaluate(lambda query, _vector: bm25_hits(enriched, query, args.candidate_k), questions, top_ks=top_ks),
        "qwen3_embedding": evaluate(
            lambda _query, vector: qdrant_search(args.qdrant_url, args.collection, vector, args.candidate_k),
            questions,
            query_vectors=query_vectors,
            top_ks=top_ks,
        ),
    }
    for weight in hybrid_weights:
        reports[f"refmark_bm25_plus_qwen3_w{weight:g}"] = evaluate(
            lambda query, vector, weight=weight: hybrid(
                bm25_hits(enriched, query, args.candidate_k),
                qdrant_search(args.qdrant_url, args.collection, vector, args.candidate_k),
                first_weight=weight,
            ),
            questions,
            query_vectors=query_vectors,
            top_ks=top_ks,
        )

    report = {
        "schema": "refmark.bgb_stress_embedding_eval.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stress_report": args.stress_report,
        "questions": len(questions),
        "regions": len(enriched.regions),
        "embedding_model": args.embedding_model,
        "collection": args.collection,
        "reports": reports,
        "best_hybrid": best_hybrid(reports),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def ensure_collection(
    qdrant_url: str,
    collection: str,
    regions: list[SearchRegion],
    *,
    embedding_cache: Path,
    model: str,
    endpoint: str,
    api_key: str,
    recreate: bool,
    batch_size: int,
) -> None:
    vectors_by_ref = read_region_vectors(embedding_cache, regions, model=model)
    missing = [region for region in regions if region.stable_ref not in vectors_by_ref]
    if missing:
        generated_rows = []
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            texts = [region_text(region) for region in batch]
            vectors = openrouter_embed(texts, model=model, endpoint=endpoint, api_key=api_key, input_type="search_document")
            for region, text, vector in zip(batch, texts, vectors, strict=True):
                vectors_by_ref[region.stable_ref] = vector
                generated_rows.append(
                    {
                        "model": model,
                        "input_type": "search_document",
                        "stable_ref": region.stable_ref,
                        "hash": text_hash(text),
                        "embedding": vector,
                    }
                )
        append_jsonl(embedding_cache, generated_rows)
    first_vector = next(iter(vectors_by_ref.values()))
    if recreate:
        http_json(f"{qdrant_url}/collections/{collection}", method="DELETE", tolerate_404=True)
    existing = http_json(f"{qdrant_url}/collections/{collection}", method="GET", tolerate_404=True)
    if not existing:
        http_json(
            f"{qdrant_url}/collections/{collection}",
            method="PUT",
            body={"vectors": {"size": len(first_vector), "distance": "Cosine"}},
        )
    points = []
    for point_id, region in enumerate(regions):
        points.append(
            {
                "id": point_id,
                "vector": vectors_by_ref[region.stable_ref],
                "payload": {
                    "stable_ref": region.stable_ref,
                    "article_ref": article_ref(region.stable_ref),
                    "ordinal": region.ordinal,
                },
            }
        )
        if len(points) >= batch_size:
            upsert_points(qdrant_url, collection, points)
            points = []
    if points:
        upsert_points(qdrant_url, collection, points)


def read_region_vectors(path: Path, regions: list[SearchRegion], *, model: str) -> dict[str, list[float]]:
    wanted = {region.stable_ref: region_hash(region) for region in regions}
    output: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            stable_ref = str(row.get("stable_ref", ""))
            if (
                row.get("model") == model
                and row.get("input_type") == "search_document"
                and stable_ref in wanted
                and row.get("hash") == wanted[stable_ref]
            ):
                output[stable_ref] = [float(value) for value in row["embedding"]]
                if len(output) == len(wanted):
                    break
    return output


def region_hash(region: SearchRegion) -> str:
    return text_hash(region_text(region))


def region_text(region: SearchRegion) -> str:
    return "\n".join([region.text, region.view.summary, *region.view.questions, *region.view.keywords])


def upsert_points(qdrant_url: str, collection: str, points: list[dict[str, object]]) -> None:
    http_json(f"{qdrant_url}/collections/{collection}/points?wait=true", method="PUT", body={"points": points})


def embed_queries(
    questions: list[dict[str, object]],
    *,
    model: str,
    endpoint: str,
    api_key: str,
    cache_path: Path,
    batch_size: int,
) -> dict[str, list[float]]:
    cache = read_query_cache(cache_path, model=model)
    missing = [str(question["query"]) for question in questions if str(question["query"]) not in cache]
    generated = []
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        vectors = openrouter_embed(batch, model=model, endpoint=endpoint, api_key=api_key, input_type="search_query")
        for query, vector in zip(batch, vectors, strict=True):
            cache[query] = vector
            generated.append({"model": model, "query": query, "hash": text_hash(query), "embedding": vector})
    append_query_cache(cache_path, generated)
    return {str(question["query"]): cache[str(question["query"])] for question in questions}


def openrouter_embed(texts: list[str], *, model: str, endpoint: str, api_key: str, input_type: str) -> list[list[float]]:
    body = {"model": model, "input": texts if len(texts) > 1 else texts[0], "input_type": input_type}
    req = request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/b-imenitov/refmark",
            "X-Title": "refmark BGB stress embedding eval",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = sorted(payload["data"], key=lambda item: int(item.get("index", 0)))
    return [[float(value) for value in row["embedding"]] for row in rows]


def read_query_cache(path: Path, *, model: str) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    output = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("model") == model:
            output[str(row["query"])] = [float(value) for value in row["embedding"]]
    return output


def append_query_cache(path: Path, rows: list[dict[str, object]]) -> None:
    append_jsonl(path, rows)


def append_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def qdrant_search(qdrant_url: str, collection: str, vector: list[float], top_k: int) -> list[RankedHit]:
    payload = http_json(
        f"{qdrant_url}/collections/{collection}/points/search",
        method="POST",
        body={"vector": vector, "limit": top_k, "with_payload": True},
    )
    return [
        RankedHit(stable_ref=str(item["payload"]["stable_ref"]), score=float(item["score"]))
        for item in payload.get("result", [])
    ]


def bm25_hits(index: PortableBM25Index, query: str, top_k: int) -> list[RankedHit]:
    return [RankedHit(stable_ref=hit.stable_ref, score=hit.score) for hit in index.search(query, top_k=top_k)]


def hybrid(first: list[RankedHit], second: list[RankedHit], *, first_weight: float) -> list[RankedHit]:
    scores: dict[str, float] = {}
    second_weight = 1.0 - first_weight
    for weight, hits in ((first_weight, first), (second_weight, second)):
        for rank, hit in enumerate(hits, start=1):
            scores[hit.stable_ref] = scores.get(hit.stable_ref, 0.0) + (weight / (rank + 60.0))
    return [RankedHit(ref, score) for ref, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def evaluate(
    search_fn,
    questions: list[dict[str, object]],
    *,
    query_vectors: dict[str, list[float]] | None = None,
    top_ks: tuple[int, ...],
) -> dict[str, object]:
    max_k = max(top_ks)
    ranks = []
    by_style: dict[str, list[int | None]] = {}
    by_language: dict[str, list[int | None]] = {}
    misses = []
    wrong_top_articles: Counter[str] = Counter()
    for question in questions:
        query = str(question["query"])
        vector = query_vectors[query] if query_vectors is not None else None
        hits = search_fn(query, vector)[:max_k]
        gold_articles = {article_ref(ref) for ref in question["gold_refs"]}
        rank = None
        for index, hit in enumerate(hits, start=1):
            if article_ref(hit.stable_ref) in gold_articles:
                rank = index
                break
        ranks.append(rank)
        by_style.setdefault(str(question.get("style", "")), []).append(rank)
        by_language.setdefault(str(question.get("language", "")), []).append(rank)
        if rank is None:
            if hits:
                wrong_top_articles[article_ref(hits[0].stable_ref)] += 1
            if len(misses) < 30:
                misses.append(
                    {
                        "query": query,
                        "block_id": question["block_id"],
                        "language": question.get("language"),
                        "style": question.get("style"),
                        "top_articles": [article_ref(hit.stable_ref) for hit in hits[:5]],
                    }
                )
    return {
        "article_hit_at_k": summarize_ranks(ranks, top_ks=top_ks),
        "by_style": {key: summarize_ranks(value, top_ks=top_ks) for key, value in sorted(by_style.items())},
        "by_language": {key: summarize_ranks(value, top_ks=top_ks) for key, value in sorted(by_language.items())},
        "misses": sum(1 for rank in ranks if rank is None),
        "miss_rate": round(sum(1 for rank in ranks if rank is None) / max(len(ranks), 1), 4),
        "wrong_top_articles_top": [{"ref": ref, "count": count} for ref, count in wrong_top_articles.most_common(20)],
        "sample_misses": misses,
    }


def summarize_ranks(ranks: list[int | None], *, top_ks: tuple[int, ...]) -> dict[str, object]:
    total = max(len(ranks), 1)
    return {
        "count": len(ranks),
        "hit_at_k": {str(k): round(sum(1 for rank in ranks if rank is not None and rank <= k) / total, 4) for k in top_ks},
        "mrr": round(sum(1.0 / rank for rank in ranks if rank is not None) / total, 4),
    }


def best_hybrid(reports: dict[str, dict[str, object]]) -> dict[str, object] | None:
    candidates = [(name, report) for name, report in reports.items() if name.startswith("refmark_bm25_plus_qwen3")]
    if not candidates:
        return None
    name, report = max(
        candidates,
        key=lambda item: (
            float(item[1]["article_hit_at_k"]["hit_at_k"].get("10", 0.0)),
            float(item[1]["article_hit_at_k"]["mrr"]),
        ),
    )
    return {"method": name, **report["article_hit_at_k"]}


def stress_questions(report: dict[str, object]) -> list[dict[str, object]]:
    output = []
    for section in report.get("sections", []):
        for question in section.get("questions", []):
            output.append(dict(question))
    return output


def article_ref(stable_ref: str) -> str:
    doc_id, region_id = stable_ref.split(":", 1)
    return f"{doc_id}:{article_id_for(region_id)}"


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def http_json(url: str, *, method: str, body: dict[str, object] | None = None, tolerate_404: bool = False) -> dict[str, object]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method=method)
    try:
        with request.urlopen(req, timeout=120) as response:
            raw = response.read().decode("utf-8")
    except Exception as exc:
        if tolerate_404 and "404" in str(exc):
            return {}
        raise
    return json.loads(raw) if raw else {}


if __name__ == "__main__":
    main()
