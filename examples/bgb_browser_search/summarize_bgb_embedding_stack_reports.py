"""Summarize BGB embedding-router stack reports as a compact comparison table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize embedding-router -> BM25 stack reports.")
    parser.add_argument("reports", nargs="+")
    args = parser.parse_args()

    rows = []
    for report_path in args.reports:
        path = Path(report_path)
        report = json.loads(path.read_text(encoding="utf-8"))
        best = best_stack_row(report)
        compact = compact_stack_row(report, area_size=25, area_top_k=10) or best
        rows.append(
            {
                "report": path.name,
                "model": report["settings"]["model"],
                "view": report["settings"]["view"],
                "rows": report["eval_questions"],
                "dim": report["embedding_dim"],
                "flat_h10": hit(report["flat_bm25"]["article_hit_at_k"], 10),
                "embed_h10": hit(report["embedding_article"]["article_hit_at_k"], 10),
                "best_size": best["area_size"],
                "best_topk": best["area_top_k"],
                "best_union": best["union"],
                "best_h10": best["h10"],
                "best_h50": best["h50"],
                "compact_size": compact["area_size"],
                "compact_topk": compact["area_top_k"],
                "compact_union": compact["union"],
                "compact_h10": compact["h10"],
                "compact_h50": compact["h50"],
                "seconds": report.get("seconds"),
            }
        )

    headers = [
        "model",
        "rows",
        "dim",
        "flat_h10",
        "embed_h10",
        "best_h10",
        "best_h50",
        "best_area",
        "best_union",
        "compact_h10",
        "compact_h50",
        "compact_union",
        "seconds",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print(
            "| "
            + " | ".join(
                [
                    row["model"],
                    str(row["rows"]),
                    str(row["dim"]),
                    fmt(row["flat_h10"]),
                    fmt(row["embed_h10"]),
                    fmt(row["best_h10"]),
                    fmt(row["best_h50"]),
                    f"{row['best_size']}x{row['best_topk']}",
                    fmt(row["best_union"], digits=1),
                    fmt(row["compact_h10"]),
                    fmt(row["compact_h50"]),
                    fmt(row["compact_union"], digits=1),
                    fmt(row["seconds"], digits=1),
                ]
            )
            + " |"
        )


def best_stack_row(report: dict) -> dict[str, object]:
    rows = []
    for area in report["area_results"]:
        for area_top_k, result in area["results"].items():
            metrics = result["fusion_inside_area"]["article_hit_at_k"]
            rows.append(
                {
                    "area_size": int(area["area_size"]),
                    "area_top_k": int(area_top_k),
                    "union": float(result["mean_union_size"]),
                    "h10": hit(metrics, 10),
                    "h50": hit(metrics, 50),
                    "mrr": float(metrics["mrr"]),
                }
            )
    return max(rows, key=lambda row: (row["h10"], row["h50"], row["mrr"]))


def compact_stack_row(report: dict, *, area_size: int, area_top_k: int) -> dict[str, object] | None:
    for area in report["area_results"]:
        if int(area["area_size"]) != area_size:
            continue
        result = area["results"].get(str(area_top_k))
        if result is None:
            return None
        metrics = result["fusion_inside_area"]["article_hit_at_k"]
        return {
            "area_size": area_size,
            "area_top_k": area_top_k,
            "union": float(result["mean_union_size"]),
            "h10": hit(metrics, 10),
            "h50": hit(metrics, 50),
            "mrr": float(metrics["mrr"]),
        }
    return None


def hit(metrics: dict, k: int) -> float:
    return float(metrics["hit_at_k"][str(k)])


def fmt(value: object, *, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


if __name__ == "__main__":
    main()
