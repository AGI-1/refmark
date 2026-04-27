"""Compare BM25, embeddings, and hybrid retrieval over chunks and refmark units."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.rag_retrieval_benchmark.common import (  # noqa: E402
    BM25Index,
    DEFAULT_DATA_DIR,
    OUTPUT_DIR,
    RetrievalUnit,
    anchor_units,
    enriched_anchor_units,
    fixed_window_units,
    load_jsonl,
)
from examples.rag_retrieval_benchmark.embedding_benchmark import (  # noqa: E402
    OPENROUTER_EMBEDDINGS_URL,
    Embedder,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark hybrid retrieval over naive chunks and refmark units.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--split", choices=["valid", "reformulated"], default="valid")
    parser.add_argument("--limit", type=int, default=3000)
    parser.add_argument("--sample-mode", choices=["first", "even", "random"], default="even")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--top-ks", default="1,3,5,10")
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--hybrid-alpha", type=float, default=0.5, help="BM25 weight; embedding weight is 1-alpha.")
    parser.add_argument("--chunk-tokens", type=int, default=220)
    parser.add_argument("--chunk-stride", type=int, default=110)
    parser.add_argument("--source", choices=["hashing", "openrouter"], default="openrouter")
    parser.add_argument("--hash-dim", type=int, default=512)
    parser.add_argument("--model", default="perplexity/pplx-embed-v1-0.6b")
    parser.add_argument("--endpoint", default=OPENROUTER_EMBEDDINGS_URL)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--cache", default=str(OUTPUT_DIR / "hybrid_embedding_cache.jsonl"))
    parser.add_argument("--output", default=str(OUTPUT_DIR / "hybrid_retrieval.json"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    anchors = load_jsonl(data_dir / "anchors.jsonl")
    train = load_jsonl(data_dir / "train.jsonl")
    examples = _select_examples(load_jsonl(data_dir / f"{args.split}.jsonl"), limit=args.limit, mode=args.sample_mode, seed=args.seed)
    top_ks = tuple(int(part) for part in args.top_ks.split(",") if part.strip())

    embedder = Embedder(args)
    query_embeddings = embedder.embed_many([str(row["question"]) for row in examples])
    unit_sets = {
        "naive_chunks": fixed_window_units(anchors, chunk_tokens=args.chunk_tokens, stride=args.chunk_stride),
        "refmark_regions": anchor_units(anchors),
        "refmark_enriched": enriched_anchor_units(anchors, train),
    }
    reports = {}
    for name, units in unit_sets.items():
        print(f"evaluating {name} ({len(units)} units)")
        unit_embeddings = embedder.embed_many([unit.text for unit in units])
        bm25_index = BM25Index(units)
        bm25_orders = _bm25_orders(bm25_index, examples, top_k=args.candidate_k)
        embedding_orders = _embedding_orders(unit_embeddings, query_embeddings, top_k=args.candidate_k)
        reports[f"{name}_bm25"] = _evaluate_orders(units, examples, bm25_orders, top_ks=top_ks)
        reports[f"{name}_embedding"] = _evaluate_orders(units, examples, embedding_orders, top_ks=top_ks)
        hybrid_orders = _hybrid_orders(
            units,
            examples,
            bm25_orders,
            embedding_orders,
            alpha=args.hybrid_alpha,
            top_k=args.candidate_k,
        )
        reports[f"{name}_hybrid"] = _evaluate_orders(units, examples, hybrid_orders, top_ks=top_ks)

    payload = {
        "data_dir": str(data_dir),
        "split": args.split,
        "examples": len(examples),
        "anchors": len(anchors),
        "settings": {
            "top_ks": top_ks,
            "candidate_k": args.candidate_k,
            "hybrid_alpha": args.hybrid_alpha,
            "sample_mode": args.sample_mode,
            "seed": args.seed,
            "embedding_source": args.source,
            "embedding_model": args.model if args.source == "openrouter" else f"hashing-{args.hash_dim}",
            "cache": args.cache,
        },
        "reports": reports,
        "interpretation": [
            "naive_chunks represents a classical fixed-window RAG baseline.",
            "refmark_regions tests addressable source units without generated metadata.",
            "refmark_enriched tests addressable source units plus retained retrieval-view questions.",
            "hybrid linearly combines per-query normalized BM25 and embedding candidate scores.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nWrote hybrid retrieval benchmark to {output}")


def _bm25_orders(index: BM25Index, examples: list[dict], *, top_k: int) -> list[list[tuple[int, float]]]:
    by_id = {unit.unit_id: idx for idx, unit in enumerate(index.units)}
    orders = []
    for row in examples:
        result = []
        for unit, score in index.search(str(row["question"]), top_k=top_k):
            result.append((by_id[unit.unit_id], float(score)))
        orders.append(result)
    return orders


def _embedding_orders(unit_embeddings: np.ndarray, query_embeddings: np.ndarray, *, top_k: int) -> list[list[tuple[int, float]]]:
    scores = query_embeddings @ unit_embeddings.T
    orders = []
    k = min(top_k, scores.shape[1])
    for row_scores in scores:
        if k >= len(row_scores):
            order = np.argsort(-row_scores, kind="stable")
        else:
            order = np.argpartition(-row_scores, k - 1)[:k]
            order = order[np.argsort(-row_scores[order], kind="stable")]
        orders.append([(int(index), float(row_scores[index])) for index in order[:k]])
    return orders


def _hybrid_orders(
    units: list[RetrievalUnit],
    examples: list[dict],
    bm25_orders: list[list[tuple[int, float]]],
    embedding_orders: list[list[tuple[int, float]]],
    *,
    alpha: float,
    top_k: int,
) -> list[list[tuple[int, float]]]:
    del units, examples
    output = []
    for bm25, embedding in zip(bm25_orders, embedding_orders, strict=True):
        bm25_scores = dict(bm25)
        embedding_scores = dict(embedding)
        candidate_ids = sorted(set(bm25_scores) | set(embedding_scores))
        max_bm25 = max(bm25_scores.values(), default=1.0)
        emb_values = [embedding_scores.get(idx, -1.0) for idx in candidate_ids]
        min_emb = min(emb_values, default=-1.0)
        max_emb = max(emb_values, default=1.0)
        spread_emb = max(max_emb - min_emb, 1e-6)
        scored = []
        for idx in candidate_ids:
            bm25_norm = bm25_scores.get(idx, 0.0) / max(max_bm25, 1e-6)
            emb_norm = (embedding_scores.get(idx, min_emb) - min_emb) / spread_emb
            scored.append((idx, alpha * bm25_norm + (1.0 - alpha) * emb_norm))
        scored.sort(key=lambda item: (-item[1], item[0]))
        output.append(scored[:top_k])
    return output


def _evaluate_orders(
    units: list[RetrievalUnit],
    examples: list[dict],
    orders: list[list[tuple[int, float]]],
    *,
    top_ks: tuple[int, ...],
) -> dict[str, object]:
    hits = {k: 0 for k in top_ks}
    reciprocal_sum = 0.0
    token_cost = {k: 0 for k in top_ks}
    refs_returned = {k: 0 for k in top_ks}
    misses: list[dict[str, object]] = []
    for row, order in zip(examples, orders, strict=True):
        gold = str(row["refmark"])
        rank = None
        for idx, (unit_index, _score) in enumerate(order, start=1):
            if gold in units[unit_index].refs:
                rank = idx
                break
        if rank is not None:
            reciprocal_sum += 1.0 / rank
        else:
            misses.append(
                {
                    "question": row["question"],
                    "gold": gold,
                    "top_units": [units[unit_index].unit_id for unit_index, _score in order[:3]],
                    "top_refs": [list(units[unit_index].refs)[:6] for unit_index, _score in order[:3]],
                }
            )
        for k in top_ks:
            selected = [units[unit_index] for unit_index, _score in order[:k]]
            returned_refs = {ref for unit in selected for ref in unit.refs}
            if gold in returned_refs:
                hits[k] += 1
            token_cost[k] += sum(unit.token_count for unit in selected)
            refs_returned[k] += len(returned_refs)
    total = max(len(examples), 1)
    return {
        "unit_count": len(units),
        "avg_unit_tokens": round(sum(unit.token_count for unit in units) / max(len(units), 1), 2),
        "mrr": round(reciprocal_sum / total, 4),
        "hit_at_k": {str(k): round(hits[k] / total, 4) for k in top_ks},
        "avg_token_cost_at_k": {str(k): round(token_cost[k] / total, 2) for k in top_ks},
        "avg_refs_returned_at_k": {str(k): round(refs_returned[k] / total, 2) for k in top_ks},
        "sample_misses": misses[:8],
    }


def _select_examples(examples: list[dict], *, limit: int, mode: str, seed: int) -> list[dict]:
    if limit <= 0 or limit >= len(examples):
        return examples
    if mode == "first":
        return examples[:limit]
    if mode == "random":
        rng = random.Random(seed)
        indexes = sorted(rng.sample(range(len(examples)), limit))
        return [examples[index] for index in indexes]
    if limit == 1:
        return [examples[0]]
    indexes = sorted({round(index * (len(examples) - 1) / (limit - 1)) for index in range(limit)})
    return [examples[index] for index in indexes[:limit]]


if __name__ == "__main__":
    main()
