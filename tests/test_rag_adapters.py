import json

from refmark.pipeline import RegionRecord
from refmark.rag_adapters import eval_tool_summary, export_ragas_rows, refmark_evidence_metrics, write_ragas_jsonl
from refmark.rag_eval import CorpusMap, EvalExample, EvalSuite


def _record(region_id: str, text: str, ordinal: int, *, doc_id: str = "policy") -> RegionRecord:
    return RegionRecord(
        doc_id=doc_id,
        region_id=region_id,
        text=text,
        start_line=ordinal,
        end_line=ordinal,
        ordinal=ordinal,
        hash=f"h-{region_id}-{len(text)}",
    )


def test_export_ragas_rows_includes_resolved_contexts_and_refmark_fields():
    corpus = CorpusMap.from_records(
        [
            _record("P01", "Refunds are available within 30 days.", 1),
            _record("P02", "Expedited shipping is non-refundable.", 2),
            _record("P03", "Audit logs are retained for 180 days.", 3),
        ]
    )
    suite = EvalSuite(
        examples=[
            EvalExample("Which clause covers expedited shipping?", ["policy:P02"], metadata={"query_style": "direct"}),
            EvalExample("Which clauses describe refunds and shipping?", ["policy:P01-policy:P02"]),
        ],
        corpus=corpus,
    ).with_source_hashes()
    run = suite.evaluate(
        lambda query: (
            [{"stable_ref": "policy:P02", "score": 3.0}]
            if "expedited" in query
            else [{"stable_ref": "policy:P01", "context_refs": ["policy:P01", "policy:P02"], "score": 2.0}]
        ),
        k=2,
    )

    rows = export_ragas_rows(suite, run, answers={"Which clause covers expedited shipping?": "P02 covers it."})

    assert rows[0]["user_input"] == "Which clause covers expedited shipping?"
    assert rows[0]["response"] == "P02 covers it."
    assert rows[0]["retrieved_contexts"] == ["[policy:P02]\nExpedited shipping is non-refundable."]
    assert "Expedited shipping is non-refundable." in rows[0]["reference"]
    assert rows[0]["gold_refs"] == ["policy:P02"]
    assert rows[0]["refmark"]["query_style"] == "direct"
    assert rows[0]["refmark"]["hit_at_k"] is True
    assert rows[0]["refmark"]["source_hashes"] == {"policy:P02": "h-P02-37"}
    assert rows[1]["context_refs"] == ["policy:P01", "policy:P02"]


def test_refmark_evidence_metrics_reports_over_under_stale_and_token_costs(tmp_path):
    original = CorpusMap.from_records(
        [
            _record("P01", "Original refund text.", 1),
            _record("P02", "Shipping text.", 2),
            _record("P03", "Audit text.", 3),
        ],
        revision_id="rev-a",
    )
    current = CorpusMap.from_records(
        [
            _record("P01", "Changed refund text.", 1),
            _record("P02", "Shipping text.", 2),
            _record("P03", "Audit text.", 3),
        ],
        revision_id="rev-b",
    )
    suite = EvalSuite(
        examples=[
            EvalExample("refund?", ["policy:P01"]).with_source_hashes(original),
            EvalExample("shipping?", ["policy:P02", "policy:P03"]).with_source_hashes(original),
        ],
        corpus=current,
    )
    run = suite.evaluate(
        lambda query: (
            [{"stable_ref": "policy:P02", "context_refs": ["policy:P02"]}]
            if query == "refund?"
            else [{"stable_ref": "policy:P02", "context_refs": ["policy:P02", "policy:P01"]}]
        ),
        k=1,
    )

    metrics = refmark_evidence_metrics(suite, run)
    summary = eval_tool_summary(suite, run, tool="ragas")
    out_path = tmp_path / "ragas.jsonl"
    write_ragas_jsonl(out_path, suite, run, answers=["", ""])
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]

    assert metrics["schema"] == "refmark.evidence_metrics.v1"
    assert metrics["stale_example_count"] == 1
    assert metrics["stale_ref_count"] == 1
    assert metrics["avg_overcitation_refs"] == 1.0
    assert metrics["avg_undercitation_refs"] == 1.0
    assert metrics["avg_support_tokens"] > 0.0
    assert summary["schema"] == "refmark.eval_tool_summary.v1"
    assert summary["tool"] == "ragas"
    assert summary["metrics"]["run_fingerprint"] == run.fingerprint
    assert rows[0]["gold_refs"] == ["policy:P01"]
