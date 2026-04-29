"""Evaluate a coarse-to-fine area router for BGB article navigation.

The flat BGB task ranks roughly 2,500 article-level regions at once. This
experiment tests a simpler hierarchy:

1. rank overlapping article windows ("areas");
2. union the articles from the top areas;
3. rerank only those articles for the final answer.

The goal is high-recall coarse routing, not perfect area precision. If the
gold article lands inside a small top-area union, the fine article ranker has a
much easier job than the flat global search.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.bgb_browser_search.adapt_bgb_static_views import (  # noqa: E402
    article_id_from_ref,
    article_regions,
    split_questions_by_block,
    stress_questions,
    summarize_ranks,
)
from examples.bgb_browser_search.run_bgb_stress_eval import StressQuestion  # noqa: E402
from refmark.search_index import PortableBM25Index, RetrievalView, SearchRegion, load_search_index  # noqa: E402


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Evaluate BGB sliding-window area routing.")
    parser.add_argument("--index", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_confusion_signatures_gemma_top300_index.json")
    parser.add_argument("--stress-report", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("all", "train", "eval"), default="eval")
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1515)
    parser.add_argument("--area-size", type=int, default=50)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--area-top-ks", default="1,2,3,5,8")
    parser.add_argument("--article-top-ks", default="1,3,5,10,20,50")
    parser.add_argument("--soft-boosts", default="0.02,0.05,0.1,0.2")
    parser.add_argument("--flat-candidate-k", type=int, default=1000)
    parser.add_argument("--include-source", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    started = time.perf_counter()
    source_index = load_search_index(args.index)
    articles = article_regions(source_index.regions)
    article_index = PortableBM25Index(articles, include_source=args.include_source)
    article_index_by_ref = {region.stable_ref: index for index, region in enumerate(articles)}
    areas, area_members = build_sliding_areas(articles, area_size=args.area_size, stride=args.stride)
    area_index = PortableBM25Index(areas, include_source=args.include_source)
    questions = load_questions(
        args.stress_report,
        split=args.split,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    area_top_ks = tuple(int(part) for part in args.area_top_ks.split(",") if part.strip())
    article_top_ks = tuple(int(part) for part in args.article_top_ks.split(",") if part.strip())
    soft_boosts = tuple(float(part) for part in args.soft_boosts.split(",") if part.strip())
    max_area_k = max(area_top_ks)
    max_article_k = max(article_top_ks)

    flat_ranks: list[int | None] = []
    area_ranks: dict[int, list[int | None]] = {top_k: [] for top_k in area_top_ks}
    routed_ranks: dict[int, list[int | None]] = {top_k: [] for top_k in area_top_ks}
    soft_ranks: dict[tuple[int, float], list[int | None]] = {
        (top_k, boost): [] for top_k in area_top_ks for boost in soft_boosts
    }
    union_sizes: dict[int, list[int]] = {top_k: [] for top_k in area_top_ks}
    by_style: dict[int, dict[str, list[int | None]]] = {top_k: {} for top_k in area_top_ks}
    by_language: dict[int, dict[str, list[int | None]]] = {top_k: {} for top_k in area_top_ks}
    miss_examples = []

    for question in questions:
        gold_article = article_id_from_ref(question.block_id)
        flat_hits = article_index.search(question.query, top_k=args.flat_candidate_k)
        flat_refs = [hit.stable_ref for hit in flat_hits]
        flat_rank = first_rank(flat_refs, gold_article)
        flat_ranks.append(flat_rank)

        area_hits = area_index.search(question.query, top_k=max_area_k)
        area_refs = [hit.stable_ref for hit in area_hits]
        for top_k in area_top_ks:
            selected_areas = area_refs[:top_k]
            candidate_refs = sorted({ref for area_ref in selected_areas for ref in area_members[area_ref]})
            union_sizes[top_k].append(len(candidate_refs))
            area_ranks[top_k].append(1 if gold_article in candidate_refs else None)
            routed_refs = rerank_articles_inside(
                article_index,
                article_index_by_ref,
                question.query,
                candidate_refs,
                top_k=max_article_k,
            )
            routed_rank = first_rank(routed_refs, gold_article)
            routed_ranks[top_k].append(routed_rank)
            by_style[top_k].setdefault(question.style, []).append(routed_rank)
            by_language[top_k].setdefault(question.language, []).append(routed_rank)
            for boost in soft_boosts:
                soft_refs = soft_area_prior_refs(
                    flat_hits,
                    selected_areas,
                    area_members,
                    boost=boost,
                    top_k=max_article_k,
                )
                soft_ranks[(top_k, boost)].append(first_rank(soft_refs, gold_article))
        if len(miss_examples) < 30 and (flat_rank is None or flat_rank > 10):
            miss_examples.append(
                {
                    "query": question.query,
                    "gold_article": gold_article,
                    "language": question.language,
                    "style": question.style,
                    "flat_rank": flat_rank,
                    "top_areas": area_refs[:3],
                }
            )

    routed = {
        str(top_k): {
            "area_recall": round(sum(1 for rank in area_ranks[top_k] if rank is not None) / max(len(questions), 1), 4),
            "article_hit_at_k": summarize_ranks(routed_ranks[top_k], top_ks=article_top_ks),
            "mean_union_size": round(sum(union_sizes[top_k]) / max(len(union_sizes[top_k]), 1), 2),
            "by_style": summarize_groups(by_style[top_k], top_ks=article_top_ks),
            "by_language": summarize_groups(by_language[top_k], top_ks=article_top_ks),
        }
        for top_k in area_top_ks
    }
    report = {
        "schema": "refmark.bgb_area_router_eval.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_index": args.index,
        "stress_reports": args.stress_report,
        "settings": vars(args),
        "question_count": len(questions),
        "article_count": len(articles),
        "area_count": len(areas),
        "seconds": round(time.perf_counter() - started, 3),
        "flat_article": summarize_ranks(flat_ranks, top_ks=article_top_ks),
        "routed": routed,
        "soft_area_prior": {
            f"areas_{top_k}_boost_{boost:g}": summarize_ranks(ranks, top_ks=article_top_ks)
            for (top_k, boost), ranks in sorted(soft_ranks.items())
        },
        "sample_flat_misses": miss_examples,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def build_sliding_areas(
    articles: list[SearchRegion],
    *,
    area_size: int,
    stride: int,
) -> tuple[list[SearchRegion], dict[str, list[str]]]:
    if area_size <= 0 or stride <= 0:
        raise SystemExit("--area-size and --stride must be positive.")
    areas: list[SearchRegion] = []
    members: dict[str, list[str]] = {}
    starts = list(range(0, len(articles), stride))
    if starts and starts[-1] + area_size < len(articles):
        starts.append(max(len(articles) - area_size, 0))
    for ordinal, start in enumerate(starts):
        chunk = articles[start : start + area_size]
        if not chunk:
            continue
        first = chunk[0].region_id
        last = chunk[-1].region_id
        region_id = f"W{ordinal:04d}_{first}_to_{last}"
        text = "\n\n".join(region.text for region in chunk)
        view = RetrievalView(
            summary=f"BGB article window from {first} to {last}.",
            questions=unique(question for region in chunk for question in region.view.questions)[:300],
            keywords=unique(keyword for region in chunk for keyword in region.view.keywords)[:300],
        )
        area = SearchRegion(
            doc_id=chunk[0].doc_id,
            region_id=region_id,
            text=text,
            hash=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
            source_path=chunk[0].source_path,
            ordinal=ordinal,
            prev_region_id=None,
            next_region_id=None,
            view=view,
        )
        areas.append(area)
        members[area.stable_ref] = [region.stable_ref for region in chunk]
    return [
        replace(
            area,
            prev_region_id=areas[index - 1].region_id if index > 0 else None,
            next_region_id=areas[index + 1].region_id if index + 1 < len(areas) else None,
        )
        for index, area in enumerate(areas)
    ], members


def rerank_articles_inside(
    article_index: PortableBM25Index,
    article_index_by_ref: dict[str, int],
    query: str,
    candidate_refs: list[str],
    *,
    top_k: int,
) -> list[str]:
    candidate_indices = {article_index_by_ref[ref] for ref in candidate_refs if ref in article_index_by_ref}
    if not candidate_indices:
        return []
    ranked = article_index._rank_regions(query, top_k=top_k, candidate_indices=candidate_indices)
    return [article_index.regions[index].stable_ref for index, _score in ranked]


def soft_area_prior_refs(
    flat_hits,
    selected_areas: list[str],
    area_members: dict[str, list[str]],
    *,
    boost: float,
    top_k: int,
) -> list[str]:
    scores = {hit.stable_ref: 1.0 / (60.0 + rank) for rank, hit in enumerate(flat_hits, start=1)}
    for area_rank, area_ref in enumerate(selected_areas, start=1):
        area_boost = boost / (60.0 + area_rank)
        for ref in area_members.get(area_ref, []):
            if ref in scores:
                scores[ref] += area_boost
    return sorted(scores, key=lambda ref: (-scores[ref], ref))[:top_k]


def load_questions(paths: list[str], *, split: str, train_fraction: float, seed: int) -> list[StressQuestion]:
    rows: list[StressQuestion] = []
    for offset, path in enumerate(paths):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        questions = stress_questions(payload)
        if split != "all":
            train, eval_rows = split_questions_by_block(questions, train_fraction=train_fraction, seed=seed + offset)
            questions = train if split == "train" else eval_rows
        rows.extend(questions)
    return rows


def first_rank(stable_refs: list[str], gold_article: str) -> int | None:
    for rank, stable_ref in enumerate(stable_refs, start=1):
        if article_id_from_ref(stable_ref) == gold_article:
            return rank
    return None


def summarize_groups(groups: dict[str, list[int | None]], *, top_ks: tuple[int, ...]) -> dict[str, object]:
    return {name: summarize_ranks(values, top_ks=top_ks) for name, values in sorted(groups.items())}


def unique(values) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


if __name__ == "__main__":
    main()
