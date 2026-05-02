"""Show how to attach Refmark to an existing retriever callback."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from refmark import (
    CorpusMap,
    EvalExample,
    EvalSuite,
    RegionRecord,
    build_adaptation_plan,
    build_data_smell_report,
    refmark_evidence_metrics,
)


EXAMPLE = Path(__file__).resolve().parent
OUTPUT = EXAMPLE / "output"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    corpus = CorpusMap.from_records(
        [
            RegionRecord("security", "P01", "All API requests must use TLS 1.3.", 1, 1, 1, "h-tls"),
            RegionRecord("security", "P02", "Rotate API tokens every ninety days.", 2, 2, 2, "h-token"),
            RegionRecord("billing", "P01", "Invoices are issued on the first business day.", 1, 1, 1, "h-invoice"),
        ],
        revision_id="demo-rev-a",
        metadata={"source": "library_integration_demo"},
    )
    suite = EvalSuite(
        examples=[
            EvalExample("How often should I rotate API tokens?", ["security:P02"], metadata={"query_style": "direct"}),
            EvalExample("When are invoices issued?", ["billing:P01"], metadata={"query_style": "direct"}),
        ],
        corpus=corpus,
    ).with_source_hashes()

    # This can be any existing retriever. It only has to return stable refs or
    # hit dictionaries containing stable_ref/context_refs/score.
    def existing_retriever(query: str):
        if "invoice" in query.lower():
            return [{"stable_ref": "billing:P01", "score": 0.91}]
        return [
            {"stable_ref": "security:P01", "score": 0.62},
            {"stable_ref": "security:P02", "score": 0.60},
        ]

    run = suite.evaluate(existing_retriever, name="existing_retriever", k=2)
    smell_report = build_data_smell_report(suite, run).to_dict()
    adaptation_plan = build_adaptation_plan(smell_report).to_dict()
    artifact = suite.run_artifact(
        run,
        settings={"retriever": "existing_retriever", "top_k": 2},
        artifacts={"source": "examples/library_integration_demo/run.py"},
    )
    metrics = refmark_evidence_metrics(suite, run)

    outputs = {
        "run_artifact.json": artifact,
        "smells.json": smell_report,
        "adaptation_plan.json": adaptation_plan,
        "metrics.json": metrics,
    }
    for filename, payload in outputs.items():
        (OUTPUT / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary = {
        "schema": "refmark.library_integration_demo.v1",
        "hit_at_k": run.metrics["hit_at_k"],
        "hit_at_1": run.metrics["hit_at_1"],
        "smell_count": smell_report["summary"]["smell_count"],
        "adaptation_actions": adaptation_plan["summary"]["action_count"],
        "run_artifact": str((OUTPUT / "run_artifact.json").relative_to(EXAMPLE.parent.parent)),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
