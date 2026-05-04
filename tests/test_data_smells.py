from refmark.data_smells import build_data_smell_report, compare_data_smell_reports
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


def test_data_smell_report_flags_duplicate_contradictory_and_uncovered_regions():
    corpus = CorpusMap.from_records(
        [
            _record("P01", "Service tokens are rotated every ninety days and stored in the managed secret store.", 1),
            _record("P02", "Service tokens are rotated every ninety days and stored in the managed secret store.", 2),
            _record(
                "P03",
                "Service token rotation policy for automation jobs must use admin approval and managed secret storage.",
                3,
            ),
            _record(
                "P04",
                "Service token rotation policy for automation jobs may use admin approval and managed secret storage.",
                4,
            ),
            _record("P05", "Uncovered onboarding text with no evaluation question.", 5),
        ]
    )
    suite = EvalSuite(
        examples=[EvalExample("Where are service tokens stored?", ["policy:P01"])],
        corpus=corpus,
    )
    run = suite.evaluate(lambda _query: [{"stable_ref": "policy:P01", "score": 1.0}], k=1)

    report = build_data_smell_report(suite, run).to_dict()
    smell_types = {smell["type"] for smell in report["smells"]}

    assert "duplicate_support" in smell_types
    assert "contradictory_support" in smell_types
    assert "uncovered_region" in smell_types
    duplicate = next(smell for smell in report["smells"] if smell["type"] == "duplicate_support")
    assert duplicate["refs"] == ["policy:P01", "policy:P02"]
    uncovered = next(smell for smell in report["smells"] if smell["type"] == "uncovered_region")
    assert uncovered["evidence"]["uncovered_count"] == 4


def test_compare_data_smell_reports_tracks_resolved_and_new_smells():
    baseline = {
        "schema": "refmark.data_smells.v1",
        "summary": {
            "run_fingerprint": "run-a",
            "corpus_fingerprint": "corp",
            "smell_count": 2,
            "metric_hit_at_k": 0.5,
            "metric_gold_coverage": 0.5,
            "by_type": {"hard_ref": 1, "query_magnet": 1},
            "by_severity": {"high": 1, "medium": 1},
        },
        "smells": [
            {"type": "hard_ref", "severity": "high", "message": "miss", "refs": ["policy:P01"]},
            {"type": "query_magnet", "severity": "medium", "message": "hub", "refs": ["policy:P02"]},
        ],
    }
    current = {
        "schema": "refmark.data_smells.v1",
        "summary": {
            "run_fingerprint": "run-b",
            "corpus_fingerprint": "corp",
            "smell_count": 1,
            "metric_hit_at_k": 1.0,
            "metric_gold_coverage": 1.0,
            "by_type": {"query_magnet": 1},
            "by_severity": {"medium": 1},
        },
        "smells": [
            {"type": "query_magnet", "severity": "medium", "message": "hub", "refs": ["policy:P02"]},
        ],
    }

    compared = compare_data_smell_reports(baseline, current, baseline_name="before", current_name="after")

    assert compared["schema"] == "refmark.data_smell_comparison.v1"
    assert compared["status"] == "improved"
    assert compared["delta"]["smell_count"] == -1
    assert compared["delta"]["high_severity_count"] == -1
    assert compared["delta"]["metric_hit_at_k"] == 0.5
    assert compared["baseline"]["name"] == "before"
    assert compared["current"]["name"] == "after"
    assert compared["resolved_smells"] == ["hard_ref|policy:P01|miss"]
    assert compared["persistent_smells"] == ["query_magnet|policy:P02|hub"]
