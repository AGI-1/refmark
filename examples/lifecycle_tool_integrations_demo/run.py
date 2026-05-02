from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from refmark import export_lifecycle_summary_rows, write_lifecycle_tool_jsonl


HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "lifecycle_summary_rows.json"
OUTPUT = HERE / "output"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary_rows = json.loads(FIXTURE.read_text(encoding="utf-8"))
    tool_rows = export_lifecycle_summary_rows(summary_rows, tool="ragas-plus-refmark")
    write_lifecycle_tool_jsonl(OUTPUT / "lifecycle_tool_rows.jsonl", summary_rows, tool="ragas-plus-refmark")
    report = _report(summary_rows, tool_rows)
    (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTPUT / "summary.md").write_text(_render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _report(summary_rows: list[dict[str, Any]], tool_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        by_repo[str(row["repo_url"])].append(row)
    corpus_rows = []
    for repo_url, rows in sorted(by_repo.items()):
        latest = rows[-1]
        corpus_rows.append(
            {
                "repo_url": repo_url,
                "revision_count": len(rows),
                "base_ref": latest["old_ref"],
                "latest_ref": latest["new_ref"],
                "old_labels": latest["old_labels"],
                "latest_refmark_auto_rate": latest["refmark_auto_rate"],
                "latest_refmark_review_rate": latest["refmark_review_rate"],
                "latest_refmark_stale_rate": latest["refmark_stale_rate"],
                "latest_naive_silent_wrong_rate": latest["naive_silent_wrong_rate"],
                "latest_workload_reduction_vs_audit": latest["workload_reduction_vs_audit"],
            }
        )
    max_silent_wrong = max(summary_rows, key=lambda row: row["naive_silent_wrong_rate"])
    max_stale = max(summary_rows, key=lambda row: row["refmark_stale_rate"])
    avg_silent_wrong = sum(row["naive_silent_wrong_rate"] for row in summary_rows) / len(summary_rows)
    avg_review_or_stale = sum(row["refmark_review_rate"] + row["refmark_stale_rate"] for row in summary_rows) / len(summary_rows)
    return {
        "schema": "refmark.lifecycle_tool_integrations_demo.v1",
        "source": "compact fixture from five Git-backed documentation lifecycle runs",
        "rows": len(summary_rows),
        "corpora": len(by_repo),
        "tool_rows": len(tool_rows),
        "integration_surfaces": {
            "ragas_without_refmark": [
                "answer/context rows can be scored, but old evidence labels have no source-hash lifecycle state",
                "a refreshed current reference string can hide that the maintained label changed underneath",
            ],
            "ragas_with_refmark": [
                "log the same answer/context rows",
                "attach lifecycle metrics as experiment metadata",
                "gate answer metrics when stale/review rates exceed project thresholds",
            ],
            "phoenix_langfuse_or_tracker": [
                "store one lifecycle row per corpus revision pair",
                "chart naive_silent_wrong_rate, refmark_stale_rate, and workload_reduction_vs_audit",
            ],
        },
        "aggregate": {
            "avg_naive_silent_wrong_rate": round(avg_silent_wrong, 4),
            "avg_refmark_review_or_stale_rate": round(avg_review_or_stale, 4),
            "max_naive_silent_wrong": {
                "repo_url": max_silent_wrong["repo_url"],
                "old_ref": max_silent_wrong["old_ref"],
                "new_ref": max_silent_wrong["new_ref"],
                "value": max_silent_wrong["naive_silent_wrong_rate"],
            },
            "max_refmark_stale": {
                "repo_url": max_stale["repo_url"],
                "old_ref": max_stale["old_ref"],
                "new_ref": max_stale["new_ref"],
                "value": max_stale["refmark_stale_rate"],
            },
        },
        "corpus_rows": corpus_rows,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Lifecycle Tool Integration Demo",
        "",
        f"Rows: {report['rows']} revision comparisons over {report['corpora']} documentation corpora.",
        "",
        "| Corpus | Base -> Latest | Labels | Refmark auto | Refmark review | Refmark stale | Naive silent wrong | Workload reduction |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["corpus_rows"]:
        lines.append(
            "| {repo} | `{base}` -> `{latest}` | {labels} | {auto} | {review} | {stale} | {silent} | {workload} |".format(
                repo=row["repo_url"],
                base=row["base_ref"],
                latest=row["latest_ref"],
                labels=row["old_labels"],
                auto=_pct(row["latest_refmark_auto_rate"]),
                review=_pct(row["latest_refmark_review_rate"]),
                stale=_pct(row["latest_refmark_stale_rate"]),
                silent=_pct(row["latest_naive_silent_wrong_rate"]),
                workload=_pct(row["latest_workload_reduction_vs_audit"]),
            )
        )
    agg = report["aggregate"]
    lines.extend(
        [
            "",
            "## Reading It",
            "",
            f"- Average naive silent-wrong rate: {_pct(agg['avg_naive_silent_wrong_rate'])}.",
            f"- Average Refmark review-or-stale rate: {_pct(agg['avg_refmark_review_or_stale_rate'])}.",
            "- Negative workload reduction is still useful: it means the corpus changed enough that old labels should not be trusted automatically.",
            "- Ragas can score answer/context quality on the exported rows; Refmark decides whether the underlying evidence labels are still valid enough to score.",
            "",
        ]
    )
    return "\n".join(lines)


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


if __name__ == "__main__":
    main()
