"""Run the frozen no-infra documentation navigation pipeline example."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = Path(__file__).resolve().parent
CORPUS = EXAMPLE / "sample_corpus"
OUTPUT = EXAMPLE / "output"
MANIFEST = OUTPUT / "corpus.refmark.jsonl"
SECTION_MAP = OUTPUT / "sections.json"
INDEX = OUTPUT / "docs.index.json"
BROWSER_INDEX = OUTPUT / "docs.browser.json"
QUESTIONS = EXAMPLE / "eval_questions.jsonl"
SMELLS = OUTPUT / "smells.json"
ADAPTATION_PLAN = OUTPUT / "adaptation_plan.json"
COMPARISON = OUTPUT / "compare_index.json"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run_cli("map", str(CORPUS), "-o", str(MANIFEST))
    run_cli("toc", str(MANIFEST), "-o", str(SECTION_MAP))
    run_cli("build-index", str(CORPUS), "-o", str(INDEX), "--source", "local")
    run_cli("export-browser-index", str(INDEX), "-o", str(BROWSER_INDEX))

    reports = {}
    smell_summary = None
    plan_summary = None
    for strategy in ("flat", "hierarchical", "rerank"):
        report = OUTPUT / f"eval_{strategy}.json"
        extra_args = []
        if strategy == "rerank":
            extra_args = ["--smell-report-output", str(SMELLS), "--adapt-plan-output", str(ADAPTATION_PLAN)]
        run_cli(
            "eval-index",
            str(INDEX),
            str(QUESTIONS),
            "--manifest",
            str(MANIFEST),
            "--strategy",
            strategy,
            "--top-k",
            "5",
            "--expand-after",
            "1",
            *extra_args,
            "-o",
            str(report),
        )
        payload = json.loads(report.read_text(encoding="utf-8"))
        reports[strategy] = payload["metrics"]
        if strategy == "rerank":
            smell_summary = payload["data_smells"]["summary"]

    plan_summary = json.loads(ADAPTATION_PLAN.read_text(encoding="utf-8"))["summary"]
    run_cli(
        "compare-index",
        str(INDEX),
        str(QUESTIONS),
        "--manifest",
        str(MANIFEST),
        "--strategies",
        "flat,hierarchical,rerank",
        "--top-k",
        "5",
        "--expand-after",
        "1",
        "-o",
        str(COMPARISON),
    )
    comparison = json.loads(COMPARISON.read_text(encoding="utf-8"))

    query = "How do I rotate API tokens without downtime?"
    hits = capture_cli(
        "search-index",
        str(INDEX),
        query,
        "--strategy",
        "rerank",
        "--top-k",
        "3",
        "--expand-after",
        "1",
        "--json",
    )
    sample_payload = {"query": query, "hits": json.loads(hits)}
    (OUTPUT / "sample_query.json").write_text(json.dumps(sample_payload, indent=2), encoding="utf-8")

    summary = {
        "schema": "refmark.docs_navigation_example.v1",
        "corpus": str(CORPUS.relative_to(ROOT)),
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "section_map": str(SECTION_MAP.relative_to(ROOT)),
        "index": str(INDEX.relative_to(ROOT)),
        "browser_index": str(BROWSER_INDEX.relative_to(ROOT)),
        "smell_report": str(SMELLS.relative_to(ROOT)),
        "adaptation_plan": str(ADAPTATION_PLAN.relative_to(ROOT)),
        "comparison": str(COMPARISON.relative_to(ROOT)),
        "reports": {
            name: {
                "count": metrics.get("count"),
                "hit_at_1": metrics.get("hit_at_1"),
                "hit_at_k": metrics.get("hit_at_k"),
                "mrr": metrics.get("mrr"),
                "gold_coverage": metrics.get("gold_coverage"),
            }
            for name, metrics in reports.items()
        },
        "smell_summary": smell_summary,
        "adaptation_plan_summary": plan_summary,
        "best_strategy": comparison["best_by_hit_at_k"],
        "sample_top_ref": sample_payload["hits"][0]["stable_ref"] if sample_payload["hits"] else None,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def run_cli(*args: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "refmark.cli", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def capture_cli(*args: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "refmark.cli", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


if __name__ == "__main__":
    main()
