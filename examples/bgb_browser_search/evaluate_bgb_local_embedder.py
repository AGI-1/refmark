"""Evaluate frozen local embedding models on BGB Refmark retrieval.

This benchmark intentionally removes OpenRouter/Qwen from runtime. It uses a
local SentenceTransformer model to embed query text and BGB article views, then
measures article retrieval and coarse-area routing against the same Refmark
stress questions used by the Qwen embedding experiments.
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

from examples.bgb_browser_search.train_bgb_article_candidate_generator import load_split_questions  # noqa: E402
from refmark.search_index import SearchRegion, load_search_index  # noqa: E402

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install sentence-transformers to run this experiment.") from exc


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Evaluate frozen local embedders for BGB retrieval.")
    parser.add_argument("--index", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_confusion_signatures_gemma_top300_index.json")
    parser.add_argument("--stress-report", action="append", required=True)
    parser.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default="examples/bgb_browser_search/output_full_qwen_turbo/local_embedder_cache")
    parser.add_argument("--view", choices=("source", "refmark_view", "combined"), default="combined")
    parser.add_argument("--area-sizes", default="5,25,50,100")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=4141)
    parser.add_argument("--query-prefix", default=None)
    parser.add_argument("--passage-prefix", default=None)
    parser.add_argument("--max-seq-length", type=int, default=256)
    args = parser.parse_args()

    started = time.perf_counter()
    index = load_search_index(args.index)
    regions = index.regions
    stable_refs = [f"{region.doc_id}:{region.region_id}" for region in regions]
    region_by_ref = dict(zip(stable_refs, regions, strict=True))
    train_rows, eval_rows = load_split_questions(args.stress_report, train_fraction=args.train_fraction, seed=args.seed)
    eval_rows = [row for row in eval_rows if row.article_ref in region_by_ref]
    if not eval_rows:
        raise SystemExit("No evaluation rows match the index.")

    model = SentenceTransformer(args.model)
    if args.max_seq_length:
        model.max_seq_length = args.max_seq_length

    passage_prefix = infer_passage_prefix(args.model, args.passage_prefix)
    query_prefix = infer_query_prefix(args.model, args.query_prefix)
    article_texts = [passage_prefix + region_text(region, view=args.view) for region in regions]
    query_texts = [query_prefix + row.query for row in eval_rows]

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    article_vectors = encode_cached(
        model,
        article_texts,
        cache_dir / cache_name(args.model, args.view, "articles", args.max_seq_length, len(article_texts)),
        batch_size=args.batch_size,
    )
    query_vectors = encode_cached(
        model,
        query_texts,
        cache_dir / cache_name(args.model, args.view, "queries_" + reports_key(args.stress_report), args.max_seq_length, len(query_texts)),
        batch_size=args.batch_size,
    )

    scores = query_vectors @ article_vectors.T
    article_metrics = retrieval_metrics(scores, [stable_refs.index(row.article_ref) for row in eval_rows], eval_rows)
    area_sizes = [int(part) for part in args.area_sizes.split(",") if part.strip()]
    area_results = []
    for area_size in area_sizes:
        area_by_article_index = np.array([index // area_size for index in range(len(stable_refs))])
        area_count = int(area_by_article_index.max()) + 1
        area_scores = np.full((scores.shape[0], area_count), -1.0, dtype=np.float32)
        for area_index in range(area_count):
            mask = area_by_article_index == area_index
            area_scores[:, area_index] = scores[:, mask].max(axis=1)
        labels = [int(stable_refs.index(row.article_ref) // area_size) for row in eval_rows]
        area_results.append(
            {
                "area_size": area_size,
                "area_count": area_count,
                "scoring": "max_article_similarity",
                "metrics": retrieval_metrics(area_scores, labels, eval_rows),
            }
        )

    report = {
        "schema": "refmark.bgb_local_embedder_eval.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "view": args.view,
        "query_prefix": query_prefix,
        "passage_prefix": passage_prefix,
        "max_seq_length": args.max_seq_length,
        "index": args.index,
        "stress_reports": args.stress_report,
        "article_count": len(regions),
        "eval_questions": len(eval_rows),
        "embedding_dim": int(article_vectors.shape[1]),
        "article_metrics": article_metrics,
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
    if view == "refmark_view":
        return retrieval_view_text(region)
    return "\n".join(part for part in (region.text, retrieval_view_text(region)) if part)


def retrieval_view_text(region: SearchRegion) -> str:
    if region.view is None:
        return ""
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


def encode_cached(model: SentenceTransformer, texts: list[str], path: Path, *, batch_size: int) -> np.ndarray:
    if path.exists():
        return np.load(path)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    np.save(path, vectors)
    return vectors


def cache_name(model: str, view: str, role: str, max_seq_length: int, count: int) -> str:
    digest = hashlib.sha1(f"{model}|{view}|{role}|{max_seq_length}|{count}".encode("utf-8")).hexdigest()[:16]
    return f"{safe_name(model)}_{view}_{role}_{digest}.npy"


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


def reports_key(paths: list[str]) -> str:
    return hashlib.sha1("|".join(Path(path).name for path in paths).encode("utf-8")).hexdigest()[:12]


def retrieval_metrics(scores: np.ndarray, labels: list[int], rows) -> dict[str, object]:
    top_ks = (1, 2, 5, 10, 50)
    max_k = min(max(top_ks), scores.shape[1])
    top = np.argpartition(-scores, kth=max_k - 1, axis=1)[:, :max_k]
    sorted_top = np.take_along_axis(top, np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1), axis=1)
    return {
        "all": summarize(sorted_top, labels, top_ks=top_ks),
        "by_style": grouped_metrics(sorted_top, labels, rows, key=lambda row: row.style, top_ks=top_ks),
        "by_language": grouped_metrics(sorted_top, labels, rows, key=lambda row: row.language, top_ks=top_ks),
        "by_source_report": grouped_metrics(sorted_top, labels, rows, key=lambda row: row.source_report, top_ks=top_ks),
    }


def grouped_metrics(top: np.ndarray, labels: list[int], rows, *, key, top_ks: tuple[int, ...]) -> dict[str, object]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(key(row))].append(index)
    return {name: summarize(top[indices], [labels[index] for index in indices], top_ks=top_ks) for name, indices in groups.items()}


def summarize(top: np.ndarray, labels: list[int], *, top_ks: tuple[int, ...]) -> dict[str, object]:
    ranks = []
    for predicted, label in zip(top.tolist(), labels, strict=True):
        rank = None
        for index, value in enumerate(predicted, start=1):
            if int(value) == int(label):
                rank = index
                break
        ranks.append(rank)
    total = max(len(ranks), 1)
    output = {"rows": len(ranks)}
    for k in top_ks:
        output[f"hit_at_{k}"] = round(sum(1 for rank in ranks if rank is not None and rank <= k) / total, 4)
    output["mrr"] = round(sum(1 / rank for rank in ranks if rank is not None) / total, 4)
    return output


if __name__ == "__main__":
    main()
