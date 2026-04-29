"""Report BGB coarse-surface recall before local reranking/training.

Surface-conditioned models can only help when the correct article enters the
candidate surface. This diagnostic measures that ceiling for different coarse
routers and breaks it down by language/style/source report.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.bgb_browser_search.adapt_bgb_static_views import article_regions  # noqa: E402
from examples.bgb_browser_search.train_bgb_article_candidate_generator import CandidateQuestion, load_split_questions, summarize_groups  # noqa: E402
from refmark.search_index import PortableBM25Index, SearchHit, load_search_index  # noqa: E402


Router = Callable[[str, int], list[SearchHit]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure BGB article surface recall for router candidates.")
    parser.add_argument("--raw-index", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_raw_index.json")
    parser.add_argument("--enriched-index", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_openrouter_index.json")
    parser.add_argument("--adapted-index", action="append", default=[], help="Optional adapted article/region indexes to compare.")
    parser.add_argument("--stress-report", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=3031)
    parser.add_argument("--surface-ks", default="5,10,20,50,100")
    parser.add_argument("--doc-top-k", type=int, default=5)
    parser.add_argument(
        "--routers",
        default="raw_article_bm25,enriched_article_bm25,enriched_article_rerank,enriched_article_hierarchical,adapted",
        help="Comma-separated router names to run; use adapted to include all --adapted-index values.",
    )
    parser.add_argument("--sample-misses", type=int, default=25)
    args = parser.parse_args()

    surface_ks = tuple(int(part) for part in args.surface_ks.split(",") if part.strip())
    max_k = max(surface_ks)
    _train_rows, eval_rows = load_split_questions(args.stress_report, train_fraction=args.train_fraction, seed=args.seed)

    raw_article_index = PortableBM25Index(article_regions(load_search_index(args.raw_index).regions), include_source=True)
    enriched_article_index = PortableBM25Index(article_regions(load_search_index(args.enriched_index).regions), include_source=True)
    requested = {part.strip() for part in args.routers.split(",") if part.strip()}
    routers: dict[str, Router] = {}
    available: dict[str, Router] = {
        "raw_article_bm25": lambda query, top_k: raw_article_index.search(query, top_k=top_k),
        "enriched_article_bm25": lambda query, top_k: enriched_article_index.search(query, top_k=top_k),
        "enriched_article_rerank": lambda query, top_k: enriched_article_index.search_reranked(query, top_k=top_k, candidate_k=max(top_k * 3, 30)),
        "enriched_article_hierarchical": lambda query, top_k: enriched_article_index.search_hierarchical(
            query,
            top_k=top_k,
            doc_top_k=args.doc_top_k,
            candidate_k=max(top_k * 3, 30),
        ),
    }
    for name, router in available.items():
        if name in requested:
            routers[name] = router
    for path in args.adapted_index:
        if "adapted" not in requested:
            continue
        index = PortableBM25Index(article_regions(load_search_index(path).regions), include_source=True)
        name = f"adapted:{Path(path).stem}"
        routers[name] = lambda query, top_k, index=index: index.search(query, top_k=top_k)

    reports = {
        name: evaluate_router(router, eval_rows, surface_ks=surface_ks, max_k=max_k, sample_misses=args.sample_misses)
        for name, router in routers.items()
    }
    best_by_hit10 = sorted(
        (
            {
                "router": name,
                "hit_at_10": report["article_hit_at_k"]["hit_at_k"].get("10", 0.0),
                "mrr": report["article_hit_at_k"]["mrr"],
            }
            for name, report in reports.items()
        ),
        key=lambda row: (-float(row["hit_at_10"]), -float(row["mrr"]), row["router"]),
    )
    output = {
        "schema": "refmark.bgb_surface_recall_report.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": vars(args),
        "eval_questions": len(eval_rows),
        "surface_ks": surface_ks,
        "reports": reports,
        "best_by_hit10": best_by_hit10,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


def evaluate_router(
    router: Router,
    rows: list[CandidateQuestion],
    *,
    surface_ks: tuple[int, ...],
    max_k: int,
    sample_misses: int,
) -> dict[str, object]:
    ranks: list[int | None] = []
    by_language: dict[str, list[int | None]] = {}
    by_style: dict[str, list[int | None]] = {}
    by_report: dict[str, list[int | None]] = {}
    wrong_top: Counter[str] = Counter()
    misses = []
    for row in rows:
        hits = router(row.query, max_k)
        refs = [hit.stable_ref for hit in hits]
        rank = first_rank(refs, row.article_ref)
        ranks.append(rank)
        by_language.setdefault(row.language, []).append(rank)
        by_style.setdefault(row.style, []).append(rank)
        by_report.setdefault(row.source_report, []).append(rank)
        if hits and rank != 1:
            wrong_top[hits[0].stable_ref] += 1
        if rank is None and len(misses) < sample_misses:
            misses.append(
                {
                    "query": row.query,
                    "gold_ref": row.article_ref,
                    "language": row.language,
                    "style": row.style,
                    "top_refs": refs[:10],
                }
            )
    return {
        "article_hit_at_k": summarize_ranks(ranks, surface_ks=surface_ks),
        "by_language": summarize_groups(by_language, top_ks=surface_ks),
        "by_style": summarize_groups(by_style, top_ks=surface_ks),
        "by_report": summarize_groups(by_report, top_ks=surface_ks),
        "wrong_top_refs": [{"ref": ref, "count": count} for ref, count in wrong_top.most_common(20)],
        "sample_misses": misses,
    }


def first_rank(refs: list[str], gold_ref: str) -> int | None:
    for index, ref in enumerate(refs, start=1):
        if ref == gold_ref:
            return index
    return None


def summarize_ranks(ranks: list[int | None], *, surface_ks: tuple[int, ...]) -> dict[str, object]:
    count = len(ranks)
    hit_at_k = {}
    for k in surface_ks:
        hit_at_k[str(k)] = round(sum(1 for rank in ranks if rank is not None and rank <= k) / max(count, 1), 4)
    reciprocal = [1.0 / rank for rank in ranks if rank is not None]
    return {
        "count": count,
        "hit_at_k": hit_at_k,
        "mrr": round(sum(reciprocal) / max(count, 1), 4),
        "misses_at_max_k": sum(1 for rank in ranks if rank is None),
    }


if __name__ == "__main__":
    main()
