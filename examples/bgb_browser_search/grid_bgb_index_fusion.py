"""Grid-search weighted fusion across existing BGB article indexes.

Unlike `grid_bgb_fielded_fusion.py`, this script does not build train-question
aliases. It treats existing indexes as fixed retrieval variants and searches
for a weighted RRF blend on the train split, then evaluates the chosen blend on
held-out questions. This is a safer check for whether static durable signals
compound without injecting generated eval-like text.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.bgb_browser_search.adapt_bgb_static_views import (  # noqa: E402
    article_id_from_ref,
    article_regions,
    split_questions_by_block,
    stress_questions,
    summarize_ranks,
)
from examples.bgb_browser_search.run_bgb_stress_eval import StressQuestion  # noqa: E402
from refmark.search_index import PortableBM25Index, load_search_index  # noqa: E402


DEFAULT_INDEXES = [
    "base=examples/bgb_browser_search/output_full_qwen_turbo/bgb_openrouter_index.json",
    "deterministic=examples/bgb_browser_search/output_full_qwen_turbo/bgb_intent_signatures_3cycle_index.json",
    "confusion=examples/bgb_browser_search/output_full_qwen_turbo/bgb_confusion_signatures_gemma_top300_index.json",
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Grid-search fusion across existing BGB indexes.")
    parser.add_argument("--index", action="append", default=[], help="name=path. Defaults to base/deterministic/confusion indexes.")
    parser.add_argument("--stress-report", action="append", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1515)
    parser.add_argument("--top-ks", default="1,3,5,10,20,50,100")
    parser.add_argument("--candidate-k", type=int, default=200)
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--weights", default="0,0.25,0.5,0.75,1,1.5")
    parser.add_argument("--max-plans", type=int, default=None)
    args = parser.parse_args()

    specs = args.index or DEFAULT_INDEXES
    indexes = load_indexes(specs)
    top_ks = tuple(int(part) for part in args.top_ks.split(",") if part.strip())
    weights = [float(part) for part in args.weights.split(",") if part.strip()]
    train_rows, eval_rows = load_split(args.stress_report, train_fraction=args.train_fraction, seed=args.seed)
    grid = weight_grid(tuple(indexes), weights)
    if args.max_plans is not None:
        grid = grid[: args.max_plans]

    train_rank_maps = precompute(indexes, train_rows, candidate_k=args.candidate_k)
    eval_rank_maps = precompute(indexes, eval_rows, candidate_k=args.candidate_k)
    baseline_name = next(iter(indexes))
    baseline_weights = {name: (1.0 if name == baseline_name else 0.0) for name in indexes}
    baseline_train = evaluate_plan(train_rank_maps, train_rows, baseline_weights, top_ks=top_ks, rrf_k=args.rrf_k)
    baseline_eval = evaluate_plan(eval_rank_maps, eval_rows, baseline_weights, top_ks=top_ks, rrf_k=args.rrf_k)

    scored = []
    for plan in grid:
        train = evaluate_plan(train_rank_maps, train_rows, plan, top_ks=top_ks, rrf_k=args.rrf_k)
        scored.append({"weights": plan, "train": train, "score": objective(train)})
    scored.sort(key=lambda row: (-float(row["score"]), tuple(row["weights"][name] for name in indexes)))
    best = scored[0]
    best["eval"] = evaluate_plan(eval_rank_maps, eval_rows, best["weights"], top_ks=top_ks, rrf_k=args.rrf_k)

    single_index_eval = {
        name: evaluate_plan(
            eval_rank_maps,
            eval_rows,
            {candidate: (1.0 if candidate == name else 0.0) for candidate in indexes},
            top_ks=top_ks,
            rrf_k=args.rrf_k,
        )
        for name in indexes
    }
    report = {
        "schema": "refmark.bgb_index_fusion_grid.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": vars(args),
        "indexes": specs,
        "train_questions": len(train_rows),
        "eval_questions": len(eval_rows),
        "plans_tested": len(grid),
        "baseline_name": baseline_name,
        "baseline_train": baseline_train,
        "baseline_eval": baseline_eval,
        "single_index_eval": single_index_eval,
        "best_global": best,
        "top_plans": scored[:20],
    }
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def load_indexes(specs: list[str]) -> dict[str, PortableBM25Index]:
    output = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"Expected name=path index spec, got {spec!r}.")
        name, path = spec.split("=", 1)
        output[name] = PortableBM25Index(article_regions(load_search_index(path).regions), include_source=True)
    return output


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


def weight_grid(names: tuple[str, ...], weights: list[float]) -> list[dict[str, float]]:
    rows = []
    for combo in itertools.product(weights, repeat=len(names)):
        if any(value > 0 for value in combo):
            rows.append(dict(zip(names, combo, strict=True)))
    return rows


def precompute(indexes: dict[str, PortableBM25Index], rows: list[StressQuestion], *, candidate_k: int) -> dict[str, list[dict[str, int]]]:
    output = {name: [] for name in indexes}
    for row in rows:
        for name, index in indexes.items():
            output[name].append({hit.stable_ref: rank for rank, hit in enumerate(index.search(row.query, top_k=candidate_k), start=1)})
    return output


def evaluate_plan(
    rank_maps: dict[str, list[dict[str, int]]],
    rows: list[StressQuestion],
    weights: dict[str, float],
    *,
    top_ks: tuple[int, ...],
    rrf_k: float,
) -> dict[str, object]:
    ranks = []
    for index, row in enumerate(rows):
        scores: dict[str, float] = {}
        for name, maps in rank_maps.items():
            weight = weights.get(name, 0.0)
            if weight <= 0:
                continue
            for ref, rank in maps[index].items():
                scores[ref] = scores.get(ref, 0.0) + weight / (rrf_k + rank)
        ordered_refs = sorted(scores, key=lambda ref: (-scores[ref], ref))
        ranks.append(first_rank(ordered_refs, article_id_from_ref(row.block_id)))
    return summarize_split(ranks, rows, top_ks=top_ks)


def first_rank(refs: list[str], gold_article: str) -> int | None:
    for rank, ref in enumerate(refs, start=1):
        if article_id_from_ref(ref) == gold_article:
            return rank
    return None


def summarize_split(ranks: list[int | None], rows: list[StressQuestion], *, top_ks: tuple[int, ...]) -> dict[str, object]:
    by_language: dict[str, list[int | None]] = {}
    by_style: dict[str, list[int | None]] = {}
    misses: Counter[str] = Counter()
    for rank, row in zip(ranks, rows, strict=True):
        by_language.setdefault(row.language, []).append(rank)
        by_style.setdefault(row.style, []).append(rank)
        if rank is None or rank > 10:
            misses[row.block_id] += 1
    return {
        "article_hit_at_k": summarize_ranks(ranks, top_ks=top_ks),
        "by_language": {name: summarize_ranks(values, top_ks=top_ks) for name, values in sorted(by_language.items())},
        "by_style": {name: summarize_ranks(values, top_ks=top_ks) for name, values in sorted(by_style.items())},
        "misses_by_block_top": [{"block_id": block, "misses_or_late": count} for block, count in misses.most_common(20)],
    }


def objective(summary: dict[str, object]) -> float:
    metrics = summary["article_hit_at_k"]
    hit_at_k = metrics["hit_at_k"]
    return (2.0 * float(hit_at_k.get("10", 0.0))) + float(hit_at_k.get("50", 0.0)) + float(metrics["mrr"])


if __name__ == "__main__":
    main()
