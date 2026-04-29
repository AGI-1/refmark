"""Grid-search fielded BGB article retrieval fusion.

This is a cheap no-new-LLM experiment for the article-level router. It keeps
source text, generated summaries, generated questions, keywords, and
train-split aliases as separate BM25 fields, precomputes ranks once, then
searches weighted RRF plans on the train split and evaluates them on held-out
queries.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.bgb_browser_search.adapt_bgb_static_views import (  # noqa: E402
    alias_text,
    article_id_from_ref,
    article_regions,
    keyword_terms,
    split_questions_by_block,
    stress_questions,
    summarize_ranks,
    unique,
)
from examples.bgb_browser_search.run_bgb_stress_eval import StressQuestion  # noqa: E402
from refmark.search_index import PortableBM25Index, RetrievalView, SearchHit, SearchRegion, load_search_index  # noqa: E402


FIELDS = ("source", "summary", "questions", "keywords", "train_aliases")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Grid-search BGB fielded weighted-RRF article retrieval.")
    parser.add_argument("--index", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_openrouter_index.json")
    parser.add_argument("--stress-report", action="append", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=515)
    parser.add_argument("--top-ks", default="1,3,5,10,20,50,100")
    parser.add_argument("--candidate-k", type=int, default=120)
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--max-aliases-per-article", type=int, default=16)
    parser.add_argument("--source-weights", default="0,0.5,1,1.5")
    parser.add_argument("--summary-weights", default="0,0.25,0.5,0.75")
    parser.add_argument("--questions-weights", default="0,0.5,1,1.5")
    parser.add_argument("--keywords-weights", default="0,0.25,0.5")
    parser.add_argument("--train-aliases-weights", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--max-plans", type=int, default=None, help="Optional cap for quick probes.")
    args = parser.parse_args()

    top_ks = tuple(int(part) for part in args.top_ks.split(",") if part.strip())
    train_rows, eval_rows = load_split(args.stress_report, train_fraction=args.train_fraction, seed=args.seed)
    base_regions = article_regions(load_search_index(args.index).regions)
    fields = build_field_indexes(base_regions, train_rows, max_aliases_per_article=args.max_aliases_per_article)
    grid = weight_grid(args)
    if args.max_plans is not None:
        grid = grid[: args.max_plans]

    train_rank_maps = precompute_rank_maps(fields, train_rows, candidate_k=args.candidate_k)
    eval_rank_maps = precompute_rank_maps(fields, eval_rows, candidate_k=args.candidate_k)
    baseline_weights = {"source": 1.0, "summary": 1.0, "questions": 1.0, "keywords": 1.0, "train_aliases": 0.0}
    baseline_train = evaluate_plan(train_rank_maps, train_rows, baseline_weights, top_ks=top_ks, rrf_k=args.rrf_k)
    baseline_eval = evaluate_plan(eval_rank_maps, eval_rows, baseline_weights, top_ks=top_ks, rrf_k=args.rrf_k)

    scored = []
    for weights in grid:
        train = evaluate_plan(train_rank_maps, train_rows, weights, top_ks=top_ks, rrf_k=args.rrf_k)
        scored.append({"weights": weights, "train": train, "score": objective(train)})
    scored.sort(key=lambda row: (-float(row["score"]), weights_key(row["weights"])))
    best_global = scored[0]
    best_global["eval"] = evaluate_plan(eval_rank_maps, eval_rows, best_global["weights"], top_ks=top_ks, rrf_k=args.rrf_k)

    best_by_style = {}
    eval_style_fused_ranks = []
    for style in sorted({row.style for row in train_rows + eval_rows}):
        train_indices = [index for index, row in enumerate(train_rows) if row.style == style]
        eval_indices = [index for index, row in enumerate(eval_rows) if row.style == style]
        if not train_indices or not eval_indices:
            continue
        style_rank_maps = subset_rank_maps(train_rank_maps, train_indices)
        style_rows = [train_rows[index] for index in train_indices]
        style_scored = []
        for weights in grid:
            train = evaluate_plan(style_rank_maps, style_rows, weights, top_ks=top_ks, rrf_k=args.rrf_k)
            style_scored.append({"weights": weights, "train": train, "score": objective(train)})
        style_scored.sort(key=lambda row: (-float(row["score"]), weights_key(row["weights"])))
        best = style_scored[0]
        eval_subset_maps = subset_rank_maps(eval_rank_maps, eval_indices)
        eval_subset_rows = [eval_rows[index] for index in eval_indices]
        best["eval"] = evaluate_plan(eval_subset_maps, eval_subset_rows, best["weights"], top_ks=top_ks, rrf_k=args.rrf_k)
        best_by_style[style] = best

    for index, row in enumerate(eval_rows):
        weights = best_by_style.get(row.style, best_global)["weights"]
        rank = rank_for_query({name: maps[index] for name, maps in eval_rank_maps.items()}, row, weights, rrf_k=args.rrf_k)
        eval_style_fused_ranks.append(rank)
    style_routed_eval = summarize_split(eval_style_fused_ranks, eval_rows, top_ks=top_ks)

    report = {
        "schema": "refmark.bgb_fielded_fusion_grid.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": vars(args),
        "train_questions": len(train_rows),
        "eval_questions": len(eval_rows),
        "article_count": len(base_regions),
        "plans_tested": len(grid),
        "baseline_weights": baseline_weights,
        "baseline_train": baseline_train,
        "baseline_eval": baseline_eval,
        "best_global": best_global,
        "best_by_style": best_by_style,
        "style_routed_eval": style_routed_eval,
        "top_plans": scored[:20],
    }
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def load_split(paths: list[str], *, train_fraction: float, seed: int) -> tuple[list[StressQuestion], list[StressQuestion]]:
    train: list[StressQuestion] = []
    eval_rows: list[StressQuestion] = []
    for offset, path in enumerate(paths):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = stress_questions(payload)
        split_train, split_eval = split_questions_by_block(rows, train_fraction=train_fraction, seed=seed + offset)
        train.extend(split_train)
        eval_rows.extend(split_eval)
    return train, eval_rows


def build_field_indexes(
    base_regions: list[SearchRegion],
    train_questions: list[StressQuestion],
    *,
    max_aliases_per_article: int,
) -> dict[str, PortableBM25Index]:
    aliases_by_ref: dict[str, list[str]] = {}
    for question in train_questions:
        aliases_by_ref.setdefault(question.block_id, []).append(alias_text(question))
    return {
        "source": PortableBM25Index([source_region(region) for region in base_regions], include_source=True),
        "summary": PortableBM25Index([summary_region(region) for region in base_regions], include_source=False),
        "questions": PortableBM25Index([questions_region(region) for region in base_regions], include_source=False),
        "keywords": PortableBM25Index([keywords_region(region) for region in base_regions], include_source=False),
        "train_aliases": PortableBM25Index(
            [
                alias_region(region, unique(aliases_by_ref.get(region.stable_ref, []))[:max_aliases_per_article])
                for region in base_regions
            ],
            include_source=False,
        ),
    }


def source_region(region: SearchRegion) -> SearchRegion:
    return replace(region, view=RetrievalView(summary="", questions=[], keywords=[]))


def summary_region(region: SearchRegion) -> SearchRegion:
    return replace(region, text="", view=RetrievalView(summary=region.view.summary, questions=[], keywords=[]))


def questions_region(region: SearchRegion) -> SearchRegion:
    return replace(region, text="", view=RetrievalView(summary="", questions=region.view.questions, keywords=[]))


def keywords_region(region: SearchRegion) -> SearchRegion:
    return replace(region, text="", view=RetrievalView(summary="", questions=[], keywords=region.view.keywords))


def alias_region(region: SearchRegion, aliases: list[str]) -> SearchRegion:
    return replace(region, text="", view=RetrievalView(summary="", questions=aliases, keywords=keyword_terms(aliases)))


def weight_grid(args: argparse.Namespace) -> list[dict[str, float]]:
    values = {
        "source": parse_floats(args.source_weights),
        "summary": parse_floats(args.summary_weights),
        "questions": parse_floats(args.questions_weights),
        "keywords": parse_floats(args.keywords_weights),
        "train_aliases": parse_floats(args.train_aliases_weights),
    }
    grid = []
    for combo in itertools.product(*(values[field] for field in FIELDS)):
        weights = dict(zip(FIELDS, combo, strict=True))
        if any(value > 0 for value in weights.values()):
            grid.append(weights)
    return grid


def parse_floats(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part.strip()]


def precompute_rank_maps(
    fields: dict[str, PortableBM25Index],
    questions: list[StressQuestion],
    *,
    candidate_k: int,
) -> dict[str, list[dict[str, int]]]:
    output: dict[str, list[dict[str, int]]] = {name: [] for name in fields}
    for question in questions:
        for name, index in fields.items():
            output[name].append({hit.stable_ref: rank for rank, hit in enumerate(index.search(question.query, top_k=candidate_k), start=1)})
    return output


def subset_rank_maps(rank_maps: dict[str, list[dict[str, int]]], indices: list[int]) -> dict[str, list[dict[str, int]]]:
    return {name: [rows[index] for index in indices] for name, rows in rank_maps.items()}


def evaluate_plan(
    rank_maps: dict[str, list[dict[str, int]]],
    rows: list[StressQuestion],
    weights: dict[str, float],
    *,
    top_ks: tuple[int, ...],
    rrf_k: float,
) -> dict[str, object]:
    ranks = [
        rank_for_query({name: maps[index] for name, maps in rank_maps.items()}, row, weights, rrf_k=rrf_k)
        for index, row in enumerate(rows)
    ]
    return summarize_split(ranks, rows, top_ks=top_ks)


def rank_for_query(field_ranks: dict[str, dict[str, int]], row: StressQuestion, weights: dict[str, float], *, rrf_k: float) -> int | None:
    scores: dict[str, float] = {}
    for field, ranks in field_ranks.items():
        weight = weights.get(field, 0.0)
        if weight <= 0:
            continue
        for ref, rank in ranks.items():
            scores[ref] = scores.get(ref, 0.0) + weight / (rrf_k + rank)
    ordered_refs = sorted(scores, key=lambda ref: (-scores[ref], ref))
    gold_article = article_id_from_ref(row.block_id)
    for rank, ref in enumerate(ordered_refs, start=1):
        if article_id_from_ref(ref) == gold_article:
            return rank
    return None


def summarize_split(ranks: list[int | None], rows: list[StressQuestion], *, top_ks: tuple[int, ...]) -> dict[str, object]:
    by_language: dict[str, list[int | None]] = {}
    by_style: dict[str, list[int | None]] = {}
    misses_by_block: Counter[str] = Counter()
    for rank, row in zip(ranks, rows, strict=True):
        by_language.setdefault(row.language, []).append(rank)
        by_style.setdefault(row.style, []).append(rank)
        if rank is None or rank > 10:
            misses_by_block[row.block_id] += 1
    return {
        "article_hit_at_k": summarize_ranks(ranks, top_ks=top_ks),
        "by_language": {name: summarize_ranks(values, top_ks=top_ks) for name, values in sorted(by_language.items())},
        "by_style": {name: summarize_ranks(values, top_ks=top_ks) for name, values in sorted(by_style.items())},
        "misses_by_block_top": [{"block_id": block, "misses_or_late": count} for block, count in misses_by_block.most_common(20)],
    }


def objective(summary: dict[str, object]) -> float:
    metrics = summary["article_hit_at_k"]
    hits = metrics["hit_at_k"]
    return (2.0 * float(hits.get("10", 0.0))) + float(hits.get("50", 0.0)) + float(metrics["mrr"])


def weights_key(weights: dict[str, float]) -> tuple[float, ...]:
    return tuple(weights[field] for field in FIELDS)


if __name__ == "__main__":
    main()
