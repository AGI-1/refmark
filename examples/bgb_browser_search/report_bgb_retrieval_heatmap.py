"""Build article-level BGB retrieval heatmaps from stress reports.

The goal is product evidence, not only model debugging: show which refs are
hard to recover, which wrong refs attract traffic, and how candidate-depth
recall changes by language/style/source report.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.bgb_browser_search.adapt_bgb_static_views import article_id_from_ref, article_regions, stress_questions, summarize_ranks  # noqa: E402
from refmark.search_index import PortableBM25Index, load_search_index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a BGB retrieval heatmap from stress reports.")
    parser.add_argument("--index", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_openrouter_index.json")
    parser.add_argument("--stress-report", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-ks", default="1,3,5,10,20,50")
    parser.add_argument("--candidate-depths", default="10,20,50,80,100,200,500,1000")
    parser.add_argument("--top-limit", type=int, default=40)
    args = parser.parse_args()

    source_index = load_search_index(args.index)
    articles = article_regions(source_index.regions)
    index = PortableBM25Index(articles, include_source=True)
    questions = load_questions(args.stress_report)
    top_ks = tuple(int(part) for part in args.top_ks.split(",") if part.strip())
    candidate_depths = tuple(int(part) for part in args.candidate_depths.split(",") if part.strip())
    max_depth = max(max(top_ks), max(candidate_depths))

    ranks: list[int | None] = []
    by_language: dict[str, list[int | None]] = defaultdict(list)
    by_style: dict[str, list[int | None]] = defaultdict(list)
    by_report: dict[str, list[int | None]] = defaultdict(list)
    article_stats: dict[str, dict[str, object]] = {}
    confusion: Counter[tuple[str, str]] = Counter()
    wrong_top_articles: Counter[str] = Counter()
    sample_misses = []

    for row in questions:
        hits = index.search(row["query"], top_k=max_depth)
        hit_refs = [hit.stable_ref for hit in hits]
        gold_article = article_id_from_ref(row["block_id"])
        rank = first_rank(hit_refs, gold_article)
        ranks.append(rank)
        by_language[row["language"]].append(rank)
        by_style[row["style"]].append(rank)
        by_report[row["source_report"]].append(rank)
        stats = article_stats.setdefault(
            gold_article,
            {
                "article_ref": gold_article,
                "queries": 0,
                "misses_at_10": 0,
                "misses_at_50": 0,
                "rank_sum": 0.0,
                "ranked": 0,
                "languages": Counter(),
                "styles": Counter(),
            },
        )
        stats["queries"] += 1
        stats["languages"][row["language"]] += 1
        stats["styles"][row["style"]] += 1
        if rank is not None:
            stats["ranked"] += 1
            stats["rank_sum"] += rank
        if rank is None or rank > 10:
            stats["misses_at_10"] += 1
            if hit_refs:
                wrong = article_id_from_ref(hit_refs[0])
                wrong_top_articles[wrong] += 1
                confusion[(gold_article, wrong)] += 1
            if len(sample_misses) < args.top_limit:
                sample_misses.append(
                    {
                        "query": row["query"],
                        "gold_article": gold_article,
                        "language": row["language"],
                        "style": row["style"],
                        "source_report": row["source_report"],
                        "top_articles": [article_id_from_ref(ref) for ref in hit_refs[:5]],
                    }
                )
        if rank is None or rank > 50:
            stats["misses_at_50"] += 1

    article_rows = normalize_article_stats(article_stats.values())
    report = {
        "schema": "refmark.bgb_retrieval_heatmap.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_index": args.index,
        "stress_reports": args.stress_report,
        "settings": vars(args),
        "question_count": len(questions),
        "article_count": len(article_rows),
        "overall": summarize_ranks(ranks, top_ks=tuple(sorted(set(top_ks + candidate_depths)))),
        "by_language": summarize_groups(by_language, top_ks=top_ks),
        "by_style": summarize_groups(by_style, top_ks=top_ks),
        "by_report": summarize_groups(by_report, top_ks=top_ks),
        "hard_articles_by_miss10": sorted(article_rows, key=lambda row: (-row["miss_rate_at_10"], -row["queries"], row["article_ref"]))[: args.top_limit],
        "hard_articles_by_miss50": sorted(article_rows, key=lambda row: (-row["miss_rate_at_50"], -row["queries"], row["article_ref"]))[: args.top_limit],
        "wrong_top_articles": [{"article_ref": ref, "count": count} for ref, count in wrong_top_articles.most_common(args.top_limit)],
        "confusion_pairs": [
            {"gold_article": gold, "wrong_top_article": wrong, "count": count}
            for (gold, wrong), count in confusion.most_common(args.top_limit)
        ],
        "sample_misses": sample_misses,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def load_questions(paths: list[str]) -> list[dict[str, str]]:
    rows = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for question in stress_questions(payload):
            rows.append(
                {
                    "query": question.query,
                    "block_id": question.block_id,
                    "language": question.language,
                    "style": question.style,
                    "source_report": Path(path).name,
                }
            )
    return rows


def first_rank(stable_refs: list[str], gold_article: str) -> int | None:
    for rank, stable_ref in enumerate(stable_refs, start=1):
        if article_id_from_ref(stable_ref) == gold_article:
            return rank
    return None


def summarize_groups(groups: dict[str, list[int | None]], *, top_ks: tuple[int, ...]) -> dict[str, object]:
    return {name: summarize_ranks(ranks, top_ks=top_ks) for name, ranks in sorted(groups.items())}


def normalize_article_stats(values) -> list[dict[str, object]]:
    rows = []
    for stats in values:
        queries = max(int(stats["queries"]), 1)
        ranked = int(stats["ranked"])
        rows.append(
            {
                "article_ref": stats["article_ref"],
                "queries": queries,
                "misses_at_10": int(stats["misses_at_10"]),
                "miss_rate_at_10": round(int(stats["misses_at_10"]) / queries, 4),
                "misses_at_50": int(stats["misses_at_50"]),
                "miss_rate_at_50": round(int(stats["misses_at_50"]) / queries, 4),
                "mean_rank_when_found": round(float(stats["rank_sum"]) / ranked, 2) if ranked else None,
                "languages": dict(stats["languages"]),
                "styles": dict(stats["styles"]),
            }
        )
    return rows


if __name__ == "__main__":
    main()
