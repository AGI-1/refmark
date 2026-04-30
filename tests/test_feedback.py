import json
import subprocess
import sys

from refmark.feedback import FeedbackEvent, analyze_feedback, read_feedback_jsonl
from refmark.pipeline import RegionRecord
from refmark.rag_eval import CorpusMap


def _record(region_id: str, text: str) -> RegionRecord:
    return RegionRecord(
        doc_id="docs",
        region_id=region_id,
        text=text,
        start_line=1,
        end_line=1,
        ordinal=int(region_id.removeprefix("P")),
        hash=f"h-{region_id}",
    )


def test_feedback_report_suggests_alias_and_confusion():
    events = [
        FeedbackEvent(query="cors browser error", top_refs=["docs:P01"], selected_ref="docs:P02", useful=True),
        FeedbackEvent(query="CORS browser error", top_refs=["docs:P01"], clicked_ref="docs:P02", useful=True),
        FeedbackEvent(query="cors browser error", top_refs=["docs:P01"], clicked_ref="docs:P01", useful=False),
    ]

    report = analyze_feedback(events, min_count=2)

    cluster = report.clusters[0]
    assert cluster.query == "cors browser error"
    assert cluster.count == 3
    assert cluster.top_refs[0] == {"ref": "docs:P01", "count": 3}
    assert cluster.target_refs[0] == {"ref": "docs:P02", "count": 2}
    assert {action["action"] for action in cluster.actions} == {
        "add_shadow_alias_or_doc2query",
        "record_confusion_pair",
    }


def test_feedback_report_flags_query_magnet_and_no_answer():
    events = [
        FeedbackEvent(query="latest change", top_refs=["docs:P99"], useful=False, no_answer=True),
        FeedbackEvent(query="latest change", top_refs=["docs:P99"], useful=False, no_answer=True),
    ]

    report = analyze_feedback(events, min_count=2)

    actions = {action["action"] for action in report.clusters[0].actions}
    assert "review_query_magnet" in actions
    assert "review_no_answer_or_missing_coverage" in actions


def test_feedback_jsonl_cli_validates_refs(tmp_path):
    feedback_path = tmp_path / "feedback.jsonl"
    manifest_path = tmp_path / "manifest.jsonl"
    output_path = tmp_path / "feedback_report.json"
    feedback_path.write_text(
        "\n".join(
            [
                json.dumps({"query": "token rotation", "top_refs": ["docs:P01"], "selected_ref": "docs:P02"}),
                json.dumps({"query": "token rotation", "top_refs": ["docs:P01"], "selected_ref": "docs:P02"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    corpus = CorpusMap.from_records([_record("P01", "Token rotation overview.")])
    manifest_path.write_text("".join(json.dumps(record.to_dict()) + "\n" for record in corpus.records), encoding="utf-8")

    events = read_feedback_jsonl(feedback_path)
    assert events[0].selected_ref == "docs:P02"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "feedback-diagnostics",
            str(feedback_path),
            "--manifest",
            str(manifest_path),
            "-o",
            str(output_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "refmark.feedback_report.v1"
    assert payload["clusters"][0]["missing_refs"] == ["docs:P02"]
