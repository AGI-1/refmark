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
    "refmark_auto_rate",
    "refmark_review_rate",
    "refmark_stale_rate",
    "naive_correct_rate",
    "naive_silent_wrong_rate",
    "naive_missing_rate",
    "workload_reduction_vs_audit",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Refmark evidence lifecycle benchmark outputs.")
    parser.add_argument("inputs", nargs="+", help="Benchmark JSON files containing summary_rows.")
    parser.add_argument("--format", choices=["markdown", "json", "csv"], default="markdown")
    parser.add_argument("--output", default="", help="Optional output path. Defaults to stdout.")
    parser.add_argument("--columns", default=",".join(DEFAULT_COLUMNS), help="Comma-separated summary columns.")
    args = parser.parse_args()

    rows = load_summary_rows([Path(path) for path in args.inputs])
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
