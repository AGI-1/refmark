import json

from refmark.pipeline import RegionRecord
from refmark.rag_adapters import (
    eval_tool_summary,
    export_document_metadata,
    export_deepeval_cases,
    export_lifecycle_summary_rows,
    export_qrels_rows,
    export_ragas_rows,
    export_trace_events,
    refmark_evidence_metrics,
    write_document_metadata_jsonl,
    write_deepeval_jsonl,
    write_lifecycle_tool_jsonl,
    write_qrels_jsonl,
    write_ragas_jsonl,
    write_trace_jsonl,
)
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


def test_deepeval_and_trace_exports_preserve_refmark_metadata(tmp_path):
    corpus = CorpusMap.from_records(
        [
            _record("P01", "Refund policy.", 1),
            _record("P02", "Shipping policy.", 2),
        ]
    )
    suite = EvalSuite(
        examples=[EvalExample("What covers refunds and shipping?", ["policy:P01-policy:P02"])],
        corpus=corpus,
    ).with_source_hashes()
    run = suite.evaluate(
        lambda _query: [{"stable_ref": "policy:P01", "context_refs": ["policy:P01", "policy:P02"], "score": 1.0}],
        k=1,
    )

    deepeval_rows = export_deepeval_cases(suite, run, answers=["Refund and shipping policy."])
    trace_rows = export_trace_events(suite, run, tool="phoenix")
    deepeval_path = tmp_path / "deepeval.jsonl"
    trace_path = tmp_path / "trace.jsonl"
    write_deepeval_jsonl(deepeval_path, suite, run, answers=[""])
    write_trace_jsonl(trace_path, suite, run, tool="langfuse")

    assert deepeval_rows[0]["input"] == "What covers refunds and shipping?"
    assert deepeval_rows[0]["retrieval_context"] == ["[policy:P01]\nRefund policy.", "[policy:P02]\nShipping policy."]
    assert deepeval_rows[0]["context"] == ["[policy:P01]\nRefund policy.", "[policy:P02]\nShipping policy."]
    assert deepeval_rows[0]["refmark"]["gold_refs"] == ["policy:P01", "policy:P02"]
    assert trace_rows[0]["tool"] == "phoenix"
    assert trace_rows[0]["attributes"]["refmark.context_refs"] == ["policy:P01", "policy:P02"]
    assert trace_rows[0]["attributes"]["refmark.stale"] is False
    assert "refmark.trace_event.v1" in trace_path.read_text(encoding="utf-8")
    assert "retrieval_context" in deepeval_path.read_text(encoding="utf-8")


def test_qrels_and_document_metadata_exports_preserve_address_space(tmp_path):
    corpus = CorpusMap.from_records(
        [
            _record("P01", "Refund policy.", 1),
            _record("P02", "Shipping policy.", 2),
        ],
        revision_id="rev-a",
    )
    suite = EvalSuite(
        examples=[
            EvalExample(
                "What covers refunds and shipping?",
                ["policy:P01-policy:P02"],
                metadata={"query_id": "q-refund-shipping"},
            ).with_source_hashes(corpus)
        ],
        corpus=corpus,
    )
    qrels_path = tmp_path / "qrels.jsonl"
    metadata_path = tmp_path / "metadata.jsonl"

    qrels = export_qrels_rows(suite, run_id="gold")
    metadata_rows = export_document_metadata(corpus)
    write_qrels_jsonl(qrels_path, suite, run_id="gold")
    write_document_metadata_jsonl(metadata_path, corpus)

    assert [row["document_id"] for row in qrels] == ["policy:P01", "policy:P02"]
    assert qrels[0]["query_id"] == "q-refund-shipping"
    assert qrels[0]["source_hash"] == "h-P01-14"
    assert metadata_rows[0]["metadata"]["refmark.ref"] == "policy:P01"
    assert metadata_rows[0]["metadata"]["refmark.corpus_fingerprint"] == corpus.fingerprint
    assert "refmark.qrels_row.v1" in qrels_path.read_text(encoding="utf-8")
    assert "refmark.document_metadata.v1" in metadata_path.read_text(encoding="utf-8")


def test_lifecycle_summary_rows_export_as_tracker_events(tmp_path):
    payload = {
        "summary_rows": [
            {
                "repo_url": "https://example.test/docs.git",
                "subdir": "docs",
                "old_ref": "v1",
                "new_ref": "v2",
                "old_labels": 10,
                "new_regions": 12,
                "new_tokens": 1200,
                "refmark_auto_rate": 0.4,
                "refmark_review_rate": 0.2,
                "refmark_stale_rate": 0.4,
                "naive_correct_rate": 0.5,
                "naive_silent_wrong_rate": 0.4,
                "naive_missing_rate": 0.1,
                "workload_reduction_vs_audit": 0.3,
            }
        ]
    }
    out_path = tmp_path / "lifecycle_tool.jsonl"

    rows = export_lifecycle_summary_rows(payload, tool="langfuse")
    write_lifecycle_tool_jsonl(out_path, payload["summary_rows"], tool="phoenix")

    assert rows[0]["schema"] == "refmark.lifecycle_tool_row.v1"
    assert rows[0]["tool"] == "langfuse"
    assert rows[0]["metrics"]["naive_silent_wrong_rate"] == 0.4
    assert rows[0]["refmark"]["lifecycle"]["new_ref"] == "v2"
    assert "refmark.lifecycle_tool_row.v1" in out_path.read_text(encoding="utf-8")
