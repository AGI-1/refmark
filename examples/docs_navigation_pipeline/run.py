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


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run_cli("map", str(CORPUS), "-o", str(MANIFEST))
    run_cli("toc", str(MANIFEST), "-o", str(SECTION_MAP))
    run_cli("build-index", str(CORPUS), "-o", str(INDEX), "--source", "local")
    run_cli("export-browser-index", str(INDEX), "-o", str(BROWSER_INDEX))

    reports = {}
    for strategy in ("flat", "hierarchical", "rerank"):
        report = OUTPUT / f"eval_{strategy}.json"
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
            "-o",
            str(report),
        )
        payload = json.loads(report.read_text(encoding="utf-8"))
        reports[strategy] = payload["metrics"]

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
