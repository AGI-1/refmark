from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from refmark import (
    CorpusMap,
    EvalExample,
    EvalRun,
    EvalSuite,
    RegionRecord,
    eval_tool_summary,
    export_deepeval_cases,
    export_ragas_rows,
    export_trace_events,
    write_deepeval_jsonl,
    write_ragas_jsonl,
    write_trace_jsonl,
)


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"


def build_demo() -> tuple[EvalSuite, EvalRun, dict[str, str]]:
    corpus = CorpusMap.from_records(
        [
            _record("P01", "Refunds are available within 30 days.", 1),
            _record("P02", "Expedited shipping is non-refundable after dispatch.", 2),
            _record("P03", "Audit logs are retained for 180 days.", 3),
        ],
        revision_id="demo-rev-a",
        metadata={"source": "eval_tool_integrations_demo"},
    )
    suite = EvalSuite(
        examples=[
            EvalExample("Which clause covers expedited shipping?", ["policy:P02"], metadata={"query_style": "direct"}),
            EvalExample("Which clauses describe refunds and shipping?", ["policy:P01-policy:P02"], metadata={"query_style": "range"}),
        ],
        corpus=corpus,
    ).with_source_hashes()
    run = suite.evaluate(
        lambda query: (
            [{"stable_ref": "policy:P02", "score": 3.0}]
            if "expedited" in query
            else [{"stable_ref": "policy:P01", "context_refs": ["policy:P01", "policy:P02"], "score": 2.0}]
        ),
        name="demo_retriever",
        k=2,
    )
    answers = {
        "Which clause covers expedited shipping?": "Expedited shipping is covered by policy:P02.",
        "Which clauses describe refunds and shipping?": "Refunds and expedited shipping are covered by policy:P01-policy:P02.",
    }
    return suite, run, answers


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    suite, run, answers = build_demo()
    write_ragas_jsonl(OUTPUT / "ragas_rows.jsonl", suite, run, answers=answers)
    write_deepeval_jsonl(OUTPUT / "deepeval_cases.jsonl", suite, run, answers=answers)
    write_trace_jsonl(OUTPUT / "trace_events.jsonl", suite, run, tool="phoenix", answers=answers)
    summary = {
        "ragas_rows": len(export_ragas_rows(suite, run, answers=answers)),
        "deepeval_cases": len(export_deepeval_cases(suite, run, answers=answers)),
        "trace_events": len(export_trace_events(suite, run, tool="phoenix", answers=answers)),
        "summary": eval_tool_summary(suite, run, tool="generic"),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"rows": summary["ragas_rows"], "hit_at_k": summary["summary"]["metrics"]["hit_at_k"]}, indent=2))


def _record(region_id: str, text: str, ordinal: int) -> RegionRecord:
    return RegionRecord(
        doc_id="policy",
        region_id=region_id,
        text=text,
        start_line=ordinal,
        end_line=ordinal,
        ordinal=ordinal,
        hash=f"h-{region_id}-{len(text)}",
    )


if __name__ == "__main__":
    main()
