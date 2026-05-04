from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_COLUMNS = [
    "repo_url",
    "old_ref",
    "new_ref",
    "old_labels",
    "competent_silent_drift_rate",
    "competent_false_stale_rate",
    "competent_review_rate",
    "competent_preserved_rate",
    "layered_silent_drift_rate",
    "layered_review_rate",
    "layered_preserved_rate",
    "refmark_auto_rate",
    "refmark_review_rate",
    "refmark_stale_rate",
    "naive_correct_rate",
    "naive_silent_wrong_rate",
    "naive_missing_rate",
    "workload_reduction_vs_audit",
]

METHOD_COLUMNS = [
    "method",
    "total_labels",
    "silent_drift_rate",
    "false_stale_alert_rate",
    "human_review_workload_rate",
    "valid_evals_preserved_rate",
    "silent_drift",
    "false_stale_alerts",
    "human_review_workload",
    "valid_evals_preserved",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Refmark evidence lifecycle benchmark outputs.")
    parser.add_argument("inputs", nargs="+", help="Benchmark JSON files containing summary_rows.")
    parser.add_argument("--format", choices=["markdown", "json", "csv"], default="markdown")
    parser.add_argument("--output", default="", help="Optional output path. Defaults to stdout.")
    parser.add_argument("--columns", default=",".join(DEFAULT_COLUMNS), help="Comma-separated summary columns.")
    parser.add_argument(
        "--aggregate-methods",
        action="store_true",
        help="Aggregate eval_label_lifecycle.method_comparison across full benchmark payloads.",
    )
    args = parser.parse_args()

    input_paths = [Path(path) for path in args.inputs]
    if args.aggregate_methods:
        rows = load_method_comparison_rows(input_paths)
        columns = METHOD_COLUMNS
    else:
        rows = load_summary_rows(input_paths)
        columns = [column.strip() for column in args.columns.split(",") if column.strip()]
    rendered = render(rows, columns, fmt=args.format)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered)


def load_summary_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows.extend(dict(row) for row in payload)
        elif "summary_rows" in payload:
            rows.extend(dict(row) for row in payload["summary_rows"])
        else:
            raise ValueError(f"{path} does not contain summary_rows")
    return rows


def load_method_comparison_rows(paths: list[Path]) -> list[dict[str, Any]]:
    totals: dict[str, dict[str, int]] = {}
    labels_by_method: dict[str, int] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "revision_reports" not in payload:
            raise ValueError(f"{path} does not contain revision_reports")
        for report in payload["revision_reports"]:
            lifecycle = report["eval_label_lifecycle"]
            total_labels = int(lifecycle["total_labels"])
            for method, row in lifecycle["method_comparison"].items():
                bucket = totals.setdefault(
                    method,
                    {
                        "silent_drift": 0,
                        "false_stale_alerts": 0,
                        "human_review_workload": 0,
                        "valid_evals_preserved": 0,
                    },
                )
                labels_by_method[method] = labels_by_method.get(method, 0) + total_labels
                for key in bucket:
                    bucket[key] += int(row.get(key, 0))

    rows = []
    for method in [
        "chunk_id_only",
        "chunk_id_content_hash",
        "qrels_source_hash",
        "chunk_hash_quote_selector",
        "refmark_layered_selector",
        "refmark",
    ]:
        if method not in totals:
            continue
        total = max(labels_by_method[method], 1)
        bucket = totals[method]
        rows.append(
            {
                "method": method,
                "total_labels": labels_by_method[method],
                "silent_drift_rate": bucket["silent_drift"] / total,
                "false_stale_alert_rate": bucket["false_stale_alerts"] / total,
                "human_review_workload_rate": bucket["human_review_workload"] / total,
                "valid_evals_preserved_rate": bucket["valid_evals_preserved"] / total,
                **bucket,
            }
        )
    return rows


def render(rows: list[dict[str, Any]], columns: list[str], *, fmt: str) -> str:
    if fmt == "json":
        return json.dumps([{column: row.get(column, "") for column in columns} for row in rows], indent=2)
    if fmt == "csv":
        return render_csv(rows, columns)
    return render_markdown(rows, columns)


def render_markdown(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(_cell(row.get(column, ""), column) for column in columns) + " |" for row in rows]
    return "\n".join([header, divider, *body]) + "\n"


def render_csv(rows: list[dict[str, Any]], columns: list[str]) -> str:
    from io import StringIO

    out = StringIO()
    writer = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows({column: row.get(column, "") for column in columns} for row in rows)
    return out.getvalue()


def _cell(value: Any, column: str) -> str:
    if isinstance(value, float):
        return f"{value:.1%}" if ("rate" in column or "reduction" in column) else f"{value:.3f}"
    return str(value).replace("|", "\\|")


if __name__ == "__main__":
    main()
