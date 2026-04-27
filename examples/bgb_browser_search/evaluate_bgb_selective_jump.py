"""Evaluate selective-jump and candidate-ceiling diagnostics for BGB search.

This report is meant to answer product questions that hit@k alone hides:

- How often is the gold article inside the candidate pool?
- If the system jumps directly only above a confidence threshold, how much
  query coverage can it support at 90/95% precision?
- When it does not jump, is the gold still present in the fallback list?

The script uses article-level BGB refs, so a hit on any region inside the gold
article counts as a hit for this coarse navigation experiment.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
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
from refmark.search_index import PortableBM25Index, SearchHit, load_search_index  # noqa: E402


@dataclass(frozen=True)
class QueryEval:
    query: str
    gold_article: str
    language: str
    style: str
    source_report: str
    rank: int | None
    top_article: str | None
    top_score: float
    second_score: float
    margin: float
    margin_ratio: float
    entropy: float
    correct_top1: bool


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BGB selective-jump diagnostics.")
    parser.add_argument("--index", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_openrouter_index.json")
    parser.add_argument("--stress-report", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-depths", default="1,3,5,10,20,50,80,100,200,500,1000")
    parser.add_argument("--precision-targets", default="0.8,0.85,0.9,0.95")
    parser.add_argument("--fallback-k", type=int, default=10)
    parser.add_argument("--top-limit", type=int, default=30)
    parser.add_argument("--split", choices=("all", "train", "eval"), default="all")
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1515)
    args = parser.parse_args()

    source_index = load_search_index(args.index)
    index = PortableBM25Index(article_regions(source_index.regions), include_source=True)
    questions = load_questions(args.stress_report, split=args.split, train_fraction=args.train_fraction, seed=args.seed)
    candidate_depths = tuple(sorted({int(part) for part in args.candidate_depths.split(",") if part.strip()}))
    precision_targets = tuple(float(part) for part in args.precision_targets.split(",") if part.strip())
    max_depth = max(candidate_depths)

    rows: list[QueryEval] = []
    for question in questions:
        hits = index.search(question["query"], top_k=max_depth)
        rows.append(evaluate_question(question, hits))

    report = {
        "schema": "refmark.bgb_selective_jump.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_index": args.index,
        "stress_reports": args.stress_report,
        "settings": vars(args),
        "question_count": len(rows),
        "overall": summarize_ranks([row.rank for row in rows], top_ks=candidate_depths),
        "candidate_ceiling": candidate_ceiling(rows, candidate_depths=candidate_depths, fallback_k=args.fallback_k),
        "selective_jump": {
            "top_score": selective_jump(rows, key="top_score", precision_targets=precision_targets, fallback_k=args.fallback_k),
            "margin": selective_jump(rows, key="margin", precision_targets=precision_targets, fallback_k=args.fallback_k),
            "margin_ratio": selective_jump(rows, key="margin_ratio", precision_targets=precision_targets, fallback_k=args.fallback_k),
            "low_entropy": selective_jump(rows, key="negative_entropy", precision_targets=precision_targets, fallback_k=args.fallback_k),
        },
        "calibration": {
            "top_score": calibration(rows, key="top_score"),
            "margin": calibration(rows, key="margin"),
            "margin_ratio": calibration(rows, key="margin_ratio"),
            "low_entropy": calibration(rows, key="negative_entropy"),
        },
        "by_language": split_reports(rows, "language", candidate_depths=candidate_depths, precision_targets=precision_targets, fallback_k=args.fallback_k),
        "by_style": split_reports(rows, "style", candidate_depths=candidate_depths, precision_targets=precision_targets, fallback_k=args.fallback_k),
        "wrong_jumps_top": wrong_jumps(rows, limit=args.top_limit),
        "sample_wrong_jumps": sample_wrong_jumps(rows, limit=args.top_limit),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def load_questions(paths: list[str], *, split: str, train_fraction: float, seed: int) -> list[dict[str, str]]:
    rows = []
    for offset, path in enumerate(paths):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        questions = stress_questions(payload)
        if split != "all":
            train, eval_rows = split_questions_by_block(questions, train_fraction=train_fraction, seed=seed + offset)
            questions = train if split == "train" else eval_rows
        for question in questions:
            rows.append(question_row(question, Path(path).name))
    return rows


def question_row(question: StressQuestion, source_report: str) -> dict[str, str]:
    return {
        "query": question.query,
        "block_id": question.block_id,
        "language": question.language,
        "style": question.style,
        "source_report": source_report,
    }


def evaluate_question(question: dict[str, str], hits: list[SearchHit]) -> QueryEval:
    gold_article = article_id_from_ref(question["block_id"])
    top_article = article_id_from_ref(hits[0].stable_ref) if hits else None
    top_score = float(hits[0].score) if hits else 0.0
    second_score = float(hits[1].score) if len(hits) > 1 else 0.0
    margin = max(top_score - second_score, 0.0)
    margin_ratio = margin / max(abs(top_score), 1e-9)
    entropy = score_entropy([float(hit.score) for hit in hits[:10]])
    rank = first_rank([hit.stable_ref for hit in hits], gold_article)
    return QueryEval(
        query=question["query"],
        gold_article=gold_article,
        language=question["language"],
        style=question["style"],
        source_report=question["source_report"],
        rank=rank,
        top_article=top_article,
        top_score=top_score,
        second_score=second_score,
        margin=margin,
        margin_ratio=margin_ratio,
        entropy=entropy,
        correct_top1=top_article == gold_article,
    )


def first_rank(stable_refs: list[str], gold_article: str) -> int | None:
    for rank, stable_ref in enumerate(stable_refs, start=1):
        if article_id_from_ref(stable_ref) == gold_article:
            return rank
    return None


def score_entropy(scores: list[float]) -> float:
    if not scores:
        return 0.0
    floor = min(scores)
    shifted = [max(score - floor, 0.0) + 1e-9 for score in scores]
    total = sum(shifted)
    probs = [score / total for score in shifted]
    return -sum(prob * math.log(prob) for prob in probs)


def value(row: QueryEval, key: str) -> float:
    if key == "negative_entropy":
        return -row.entropy
    return float(getattr(row, key))


def candidate_ceiling(rows: list[QueryEval], *, candidate_depths: tuple[int, ...], fallback_k: int) -> dict[str, object]:
    output: dict[str, object] = {}
    total = max(len(rows), 1)
    for depth in candidate_depths:
        in_pool = [row for row in rows if row.rank is not None and row.rank <= depth]
        fallback_hits = [row for row in in_pool if row.rank is not None and row.rank <= fallback_k]
        output[str(depth)] = {
            "candidate_recall": round(len(in_pool) / total, 4),
            f"hit_at_{fallback_k}_conditional_on_pool": round(len(fallback_hits) / max(len(in_pool), 1), 4),
            "pool_count": len(in_pool),
        }
    return output


def selective_jump(
    rows: list[QueryEval],
    *,
    key: str,
    precision_targets: tuple[float, ...],
    fallback_k: int,
) -> dict[str, object]:
    ordered_thresholds = sorted({value(row, key) for row in rows}, reverse=True)
    points = [threshold_point(rows, key=key, threshold=threshold, fallback_k=fallback_k) for threshold in ordered_thresholds]
    best_by_target = {}
    for target in precision_targets:
        feasible = [point for point in points if point["jump_precision"] >= target and point["jump_count"] > 0]
        if feasible:
            fallback_key = f"fallback_hit_at_{fallback_k}"
            best = max(feasible, key=lambda point: (point["jump_coverage"], point[fallback_key], -point["wrong_jumps"]))
            best_by_target[str(target)] = best
        else:
            best_by_target[str(target)] = None
    return {
        "best_by_precision_target": best_by_target,
        "points_sample": sampled_points(points),
    }


def threshold_point(rows: list[QueryEval], *, key: str, threshold: float, fallback_k: int) -> dict[str, object]:
    jumped = [row for row in rows if value(row, key) >= threshold]
    fallback = [row for row in rows if value(row, key) < threshold]
    correct = sum(1 for row in jumped if row.correct_top1)
    wrong = len(jumped) - correct
    fallback_hits = sum(1 for row in fallback if row.rank is not None and row.rank <= fallback_k)
    total = max(len(rows), 1)
    return {
        "threshold": round(threshold, 6),
        "jump_count": len(jumped),
        "jump_coverage": round(len(jumped) / total, 4),
        "jump_precision": round(correct / max(len(jumped), 1), 4),
        "wrong_jumps": wrong,
        "fallback_count": len(fallback),
        f"fallback_hit_at_{fallback_k}": round(fallback_hits / max(len(fallback), 1), 4),
        "expected_success": round((correct + fallback_hits) / total, 4),
    }


def sampled_points(points: list[dict[str, object]], *, limit: int = 12) -> list[dict[str, object]]:
    if len(points) <= limit:
        return points
    indexes = sorted({round(index * (len(points) - 1) / (limit - 1)) for index in range(limit)})
    return [points[index] for index in indexes]


def calibration(rows: list[QueryEval], *, key: str, buckets: int = 10) -> list[dict[str, object]]:
    ordered = sorted(rows, key=lambda row: value(row, key), reverse=True)
    if not ordered:
        return []
    output = []
    for bucket in range(buckets):
        start = round(bucket * len(ordered) / buckets)
        end = round((bucket + 1) * len(ordered) / buckets)
        items = ordered[start:end]
        if not items:
            continue
        output.append(
            {
                "bucket": bucket + 1,
                "count": len(items),
                "min_confidence": round(min(value(row, key) for row in items), 6),
                "max_confidence": round(max(value(row, key) for row in items), 6),
                "top1_accuracy": round(sum(1 for row in items if row.correct_top1) / len(items), 4),
                "hit_at_10": round(sum(1 for row in items if row.rank is not None and row.rank <= 10) / len(items), 4),
            }
        )
    return output


def split_reports(
    rows: list[QueryEval],
    field: str,
    *,
    candidate_depths: tuple[int, ...],
    precision_targets: tuple[float, ...],
    fallback_k: int,
) -> dict[str, object]:
    groups: dict[str, list[QueryEval]] = defaultdict(list)
    for row in rows:
        groups[str(getattr(row, field))].append(row)
    return {
        name: {
            "count": len(items),
            "overall": summarize_ranks([row.rank for row in items], top_ks=candidate_depths),
            "candidate_ceiling": candidate_ceiling(items, candidate_depths=candidate_depths, fallback_k=fallback_k),
            "selective_jump_margin_ratio": selective_jump(
                items,
                key="margin_ratio",
                precision_targets=precision_targets,
                fallback_k=fallback_k,
            )["best_by_precision_target"],
        }
        for name, items in sorted(groups.items())
    }


def wrong_jumps(rows: list[QueryEval], *, limit: int) -> list[dict[str, object]]:
    pairs: Counter[tuple[str, str]] = Counter()
    for row in rows:
        if row.top_article and not row.correct_top1:
            pairs[(row.gold_article, row.top_article)] += 1
    return [
        {"gold_article": gold, "wrong_top_article": wrong, "count": count}
        for (gold, wrong), count in pairs.most_common(limit)
    ]


def sample_wrong_jumps(rows: list[QueryEval], *, limit: int) -> list[dict[str, object]]:
    output = []
    for row in rows:
        if row.top_article and not row.correct_top1:
            output.append(
                {
                    "query": row.query,
                    "gold_article": row.gold_article,
                    "wrong_top_article": row.top_article,
                    "language": row.language,
                    "style": row.style,
                    "top_score": round(row.top_score, 4),
                    "margin_ratio": round(row.margin_ratio, 4),
                }
            )
            if len(output) >= limit:
                break
    return output


if __name__ == "__main__":
    main()
