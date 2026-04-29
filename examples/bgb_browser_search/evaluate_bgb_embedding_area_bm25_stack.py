"""Evaluate an embedding-router -> BM25-resolver stack for BGB navigation.

The stack is intentionally split into measurable stages:

1. embed each article-level Refmark retrieval view;
2. use query embeddings to select coarse article areas;
3. run the existing no-embedding BM25/signature resolver only inside the
   selected area union.

This tells us whether embeddings are useful as a coarse router even when the
final answer is still produced by the portable lexical Refmark index.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.bgb_browser_search.adapt_bgb_static_views import article_regions, summarize_ranks  # noqa: E402
from examples.bgb_browser_search.train_bgb_article_candidate_generator import CandidateQuestion, first_rank, load_split_questions  # noqa: E402
from refmark.search_index import PortableBM25Index, SearchRegion, load_search_index  # noqa: E402

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install sentence-transformers to run this experiment.") from exc


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Evaluate embedding-area routing followed by BM25 article resolution.")
    parser.add_argument("--index", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_confusion_signatures_gemma_top300_index.json")
    parser.add_argument("--stress-report", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--view", choices=("source", "refmark_view", "combined"), default="refmark_view")
    parser.add_argument("--cache-dir", default="examples/bgb_browser_search/output_full_qwen_turbo/local_embedder_cache")
    parser.add_argument("--area-sizes", default="5,25,50,100")
    parser.add_argument("--area-top-ks", default="1,2,5,10")
    parser.add_argument("--article-top-ks", default="1,3,5,10,20,50")
    parser.add_argument("--bm25-candidate-k", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--encode-cache-chunk-size", type=int, default=512)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1515)
    parser.add_argument("--query-prefix", default=None)
    parser.add_argument("--passage-prefix", default=None)
    parser.add_argument("--query-prompt", default=None, help="Optional SentenceTransformer encode prompt for queries.")
    parser.add_argument("--passage-prompt", default=None, help="Optional SentenceTransformer encode prompt for passages.")
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom HF model code for models that require it.")
    parser.add_argument("--truncate-dim", type=int, default=None, help="Optional Matryoshka embedding dimension truncation.")
    args = parser.parse_args()

    started = time.perf_counter()
    source_index = load_search_index(args.index)
    articles = article_regions(source_index.regions)
    article_refs = [region.stable_ref for region in articles]
    article_ref_to_index = {ref: index for index, ref in enumerate(article_refs)}
    _train_rows, eval_rows = load_split_questions(args.stress_report, train_fraction=args.train_fraction, seed=args.seed)
    eval_rows = [row for row in eval_rows if row.article_ref in article_ref_to_index]
    if not eval_rows:
        raise SystemExit("No evaluation rows match the index.")

    model = SentenceTransformer(args.model, trust_remote_code=args.trust_remote_code, truncate_dim=args.truncate_dim)
    if args.max_seq_length:
        model.max_seq_length = args.max_seq_length
    query_prefix = infer_query_prefix(args.model, args.query_prefix)
    passage_prefix = infer_passage_prefix(args.model, args.passage_prefix)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    article_texts = [passage_prefix + region_text(region, view=args.view) for region in articles]
    query_texts = [query_prefix + row.query for row in eval_rows]
    article_vectors = encode_cached(
        model,
        article_texts,
        cache_dir / cache_name(
            args.model,
            args.view,
            "articles",
            args.max_seq_length,
            len(article_texts),
            prompt=args.passage_prompt,
            truncate_dim=args.truncate_dim,
        ),
        batch_size=args.batch_size,
        cache_chunk_size=args.encode_cache_chunk_size,
        prompt=args.passage_prompt,
        truncate_dim=args.truncate_dim,
    )
    query_vectors = encode_cached(
        model,
        query_texts,
        cache_dir / cache_name(
            args.model,
            args.view,
            "queries_" + reports_key(args.stress_report, args.seed),
            args.max_seq_length,
            len(query_texts),
            prompt=args.query_prompt,
            truncate_dim=args.truncate_dim,
        ),
        batch_size=args.batch_size,
        cache_chunk_size=args.encode_cache_chunk_size,
        prompt=args.query_prompt,
        truncate_dim=args.truncate_dim,
    )
    embedding_scores = query_vectors @ article_vectors.T

    article_index = PortableBM25Index(articles, include_source=True)
    flat_bm25_ranks: list[int | None] = []
    embedding_article_ranks: list[int | None] = []
    flat_bm25_refs_by_row: list[list[str]] = []
    for row_index, row in enumerate(eval_rows):
        bm25_refs = [hit.stable_ref for hit in article_index.search(row.query, top_k=args.bm25_candidate_k)]
        flat_bm25_refs_by_row.append(bm25_refs)
        flat_bm25_ranks.append(first_rank(bm25_refs, row.article_ref))
        embedding_article_ranks.append(rank_from_scores(embedding_scores[row_index], article_ref_to_index[row.article_ref]))

    area_sizes = [int(part) for part in args.area_sizes.split(",") if part.strip()]
    area_top_ks = [int(part) for part in args.area_top_ks.split(",") if part.strip()]
    article_top_ks = tuple(int(part) for part in args.article_top_ks.split(",") if part.strip())
    area_results = []
    for area_size in area_sizes:
        area_by_article = np.array([index // area_size for index in range(len(article_refs))])
        area_count = int(area_by_article.max()) + 1
        area_scores = np.full((embedding_scores.shape[0], area_count), -1.0, dtype=np.float32)
        for area_index in range(area_count):
            mask = area_by_article == area_index
            area_scores[:, area_index] = embedding_scores[:, mask].max(axis=1)
        area_order = sorted_top_indices(area_scores, max(area_top_ks))
        per_top_k = {}
        for area_top_k in area_top_ks:
            area_recall_ranks: list[int | None] = []
            bm25_inside_ranks: list[int | None] = []
            fusion_inside_ranks: list[int | None] = []
            union_sizes: list[int] = []
            by_style = defaultdict(list)
            by_language = defaultdict(list)
            by_source_report = defaultdict(list)
            for row_index, row in enumerate(eval_rows):
                gold_index = article_ref_to_index[row.article_ref]
                selected_areas = set(int(value) for value in area_order[row_index, :area_top_k].tolist())
                candidate_indices = {index for index, area in enumerate(area_by_article.tolist()) if int(area) in selected_areas}
                union_sizes.append(len(candidate_indices))
                area_rank = first_area_rank(area_order[row_index], int(area_by_article[gold_index]))
                area_recall_ranks.append(area_rank if area_rank is not None and area_rank <= area_top_k else None)
                bm25_refs = [ref for ref in flat_bm25_refs_by_row[row_index] if article_ref_to_index.get(ref) in candidate_indices]
                bm25_rank = first_rank(bm25_refs[: max(article_top_ks)], row.article_ref)
                bm25_inside_ranks.append(bm25_rank)
                embedding_refs = refs_by_embedding_scores(embedding_scores[row_index], article_refs, candidate_indices, top_k=max(article_top_ks))
                fused_refs = rrf_fuse(bm25_refs[: max(article_top_ks) * 4], embedding_refs, first_weight=0.7)[: max(article_top_ks)]
                fusion_rank = first_rank(fused_refs, row.article_ref)
                fusion_inside_ranks.append(fusion_rank)
                for groups, rank in ((by_style, fusion_rank), (by_language, fusion_rank), (by_source_report, fusion_rank)):
                    key = row.style if groups is by_style else row.language if groups is by_language else row.source_report
                    groups[str(key)].append(rank)
            per_top_k[str(area_top_k)] = {
                "area_recall": summarize_ranks(area_recall_ranks, top_ks=tuple(area_top_ks)),
                "bm25_inside_area": {
                    "article_hit_at_k": summarize_ranks(bm25_inside_ranks, top_ks=article_top_ks),
                },
                "fusion_inside_area": {
                    "article_hit_at_k": summarize_ranks(fusion_inside_ranks, top_ks=article_top_ks),
                    "by_style": summarize_groups(by_style, top_ks=article_top_ks),
                    "by_language": summarize_groups(by_language, top_ks=article_top_ks),
                    "by_source_report": summarize_groups(by_source_report, top_ks=article_top_ks),
                },
                "mean_union_size": round(sum(union_sizes) / max(len(union_sizes), 1), 2),
            }
        area_results.append({"area_size": area_size, "area_count": area_count, "results": per_top_k})

    report = {
        "schema": "refmark.bgb_embedding_area_bm25_stack.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": vars(args),
        "article_count": len(articles),
        "eval_questions": len(eval_rows),
        "embedding_dim": int(article_vectors.shape[1]),
        "flat_bm25": summarize_with_groups(flat_bm25_ranks, eval_rows, top_ks=article_top_ks),
        "embedding_article": summarize_with_groups(embedding_article_ranks, eval_rows, top_ks=article_top_ks),
        "area_results": area_results,
        "seconds": round(time.perf_counter() - started, 3),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def region_text(region: SearchRegion, *, view: str) -> str:
    if view == "source":
        return region.text
    refmark_view = retrieval_view_text(region)
    if view == "refmark_view":
        return refmark_view
    return "\n".join(part for part in (region.text, refmark_view) if part)


def retrieval_view_text(region: SearchRegion) -> str:
    parts = [region.view.summary]
    parts.extend(region.view.keywords)
    parts.extend(region.view.questions)
    return "\n".join(part for part in parts if part)


def infer_query_prefix(model_name: str, override: str | None) -> str:
    if override is not None:
        return override
    return "query: " if "e5" in model_name.lower() else ""


def infer_passage_prefix(model_name: str, override: str | None) -> str:
    if override is not None:
        return override
    return "passage: " if "e5" in model_name.lower() else ""


def encode_cached(
    model: SentenceTransformer,
    texts: list[str],
    path: Path,
    *,
    batch_size: int,
    cache_chunk_size: int,
    prompt: str | None,
    truncate_dim: int | None,
) -> np.ndarray:
    if path.exists():
        return np.load(path)
    part_dir = path.with_name(path.stem + "_parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    part_paths = []
    chunk_size = max(cache_chunk_size, 1)
    for start in range(0, len(texts), chunk_size):
        end = min(start + chunk_size, len(texts))
        part_path = part_dir / f"{start:06d}_{end:06d}.npy"
        part_paths.append(part_path)
        if part_path.exists():
            continue
        chunk = model.encode(
            texts[start:end],
            prompt=prompt,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
            truncate_dim=truncate_dim,
        ).astype(np.float32)
        np.save(part_path, chunk)
    vectors = np.concatenate([np.load(part_path) for part_path in part_paths], axis=0)
    np.save(path, vectors)
    return vectors


def cache_name(
    model: str,
    view: str,
    role: str,
    max_seq_length: int,
    count: int,
    *,
    prompt: str | None,
    truncate_dim: int | None,
) -> str:
    digest = hashlib.sha1(
        f"{model}|{view}|{role}|{max_seq_length}|{count}|{prompt or ''}|{truncate_dim or ''}".encode("utf-8")
    ).hexdigest()[:16]
    dim_suffix = f"_d{truncate_dim}" if truncate_dim else ""
    return f"{safe_name(model)}_{view}{dim_suffix}_{role}_{digest}.npy"


def reports_key(paths: list[str], seed: int) -> str:
    return hashlib.sha1((str(seed) + "|" + "|".join(Path(path).name for path in paths)).encode("utf-8")).hexdigest()[:12]


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


def sorted_top_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    max_k = min(top_k, scores.shape[1])
    top = np.argpartition(-scores, kth=max_k - 1, axis=1)[:, :max_k]
    return np.take_along_axis(top, np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1), axis=1)


def rank_from_scores(scores: np.ndarray, label: int) -> int | None:
    better = int(np.sum(scores > scores[label]))
    return better + 1


def first_area_rank(predicted_areas: np.ndarray, gold_area: int) -> int | None:
    for rank, area in enumerate(predicted_areas.tolist(), start=1):
        if int(area) == int(gold_area):
            return rank
    return None


def refs_by_embedding_scores(scores: np.ndarray, article_refs: list[str], candidate_indices: set[int], *, top_k: int) -> list[str]:
    ordered = sorted(candidate_indices, key=lambda index: (-float(scores[index]), article_refs[index]))
    return [article_refs[index] for index in ordered[:top_k]]


def rrf_fuse(first: list[str], second: list[str], *, first_weight: float) -> list[str]:
    scores: dict[str, float] = {}
    for weight, refs in ((first_weight, first), (1.0 - first_weight, second)):
        for rank, stable_ref in enumerate(refs, start=1):
            scores[stable_ref] = scores.get(stable_ref, 0.0) + weight / (60.0 + rank)
    return sorted(scores, key=lambda stable_ref: (-scores[stable_ref], stable_ref))


def summarize_with_groups(ranks: list[int | None], rows: list[CandidateQuestion], *, top_ks: tuple[int, ...]) -> dict[str, object]:
    by_style = defaultdict(list)
    by_language = defaultdict(list)
    by_source_report = defaultdict(list)
    for rank, row in zip(ranks, rows, strict=True):
        by_style[row.style].append(rank)
        by_language[row.language].append(rank)
        by_source_report[row.source_report].append(rank)
    return {
        "article_hit_at_k": summarize_ranks(ranks, top_ks=top_ks),
        "by_style": summarize_groups(by_style, top_ks=top_ks),
        "by_language": summarize_groups(by_language, top_ks=top_ks),
        "by_source_report": summarize_groups(by_source_report, top_ks=top_ks),
    }


def summarize_groups(groups: dict[str, list[int | None]], *, top_ks: tuple[int, ...]) -> dict[str, object]:
    return {name: summarize_ranks(values, top_ks=top_ks) for name, values in sorted(groups.items())}


if __name__ == "__main__":
    main()
