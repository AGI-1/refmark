from refmark.data_smells import build_data_smell_report
from refmark.pipeline import RegionRecord
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


def test_data_smell_report_combines_stale_confusion_style_and_citation_signals():
    original = CorpusMap.from_records(
        [
            _record("P01", "Refunds are available within 30 days.", 1),
            _record("P02", "Expedited shipping is non-refundable.", 2),
            _record("P03", "Audit logs are retained for 180 days.", 3),
        ]
    )
    current = CorpusMap.from_records(
        [
            _record("P01", "Refund text changed substantially.", 1),
            _record("P02", "Expedited shipping is non-refundable.", 2),
            _record("P03", "Audit logs are retained for 180 days.", 3),
        ]
    )
    suite = EvalSuite(
        examples=[
            EvalExample("refund direct?", ["policy:P01"], metadata={"query_style": "direct"}).with_source_hashes(original),
            EvalExample("refund concern?", ["policy:P01"], metadata={"query_style": "concern"}).with_source_hashes(original),
            EvalExample("shipping?", ["policy:P02"], metadata={"query_style": "direct"}).with_source_hashes(original),
            EvalExample("audit?", ["policy:P03"], metadata={"query_style": "direct"}).with_source_hashes(original),
        ],
        corpus=current,
    )
    run = suite.evaluate(
        lambda query: (
            [{"stable_ref": "policy:P01", "score": 1.0}, {"stable_ref": "policy:P02", "score": 0.99}]
            if query == "refund direct?"
            else [{"stable_ref": "policy:P02", "score": 1.0}, {"stable_ref": "policy:P01", "score": 0.99}]
            if "refund" in query
            else [{"stable_ref": "policy:P02", "score": 1.0}]
            if query == "shipping?"
            else [{"stable_ref": "policy:P02", "context_refs": ["policy:P02", "policy:P01"], "score": 1.0}]
        ),
        name="smell-test",
        k=2,
    )

    report = build_data_smell_report(suite, run)
    payload = report.to_dict()
    smell_types = {smell["type"] for smell in payload["smells"]}

    assert payload["schema"] == "refmark.data_smells.v1"
    assert payload["summary"]["smell_count"] >= 4
    assert "stale_label" in smell_types
    assert "confusion_pair" in smell_types
    assert "query_style_gap" in smell_types
    assert "undercitation" in smell_types
    assert payload["summary"]["by_severity"]["high"] >= 1


def test_data_smell_report_can_omit_text_packets():
    corpus = CorpusMap.from_records([_record("P01", "Refunds are available.", 1), _record("P02", "Shipping.", 2)])
    suite = EvalSuite(examples=[EvalExample("refund?", ["policy:P01"])], corpus=corpus)
    run = suite.evaluate(lambda _query: [{"stable_ref": "policy:P02", "score": 1.0}], k=1)

    report = build_data_smell_report(suite, run, include_text=False)

    hard_ref = next(smell for smell in report.to_dict()["smells"] if smell["type"] == "hard_ref")
    assert "text" not in hard_ref["evidence"]
