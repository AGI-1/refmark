import json
import subprocess
import sys

from refmark.pipeline import RegionRecord, write_manifest
from refmark.lifecycle import load_summary_rows, render_summary_rows
from refmark.rag_eval import CorpusMap, EvalSuite


def test_lifecycle_summary_rows_render_markdown_and_json(tmp_path):
    payload = {
        "summary_rows": [
            {
                "repo_url": "https://example.test/repo.git",
                "old_ref": "v1",
                "new_ref": "v2",
                "old_labels": 10,
                "refmark_auto_rate": 0.5,
                "refmark_review_rate": 0.2,
                "refmark_stale_rate": 0.3,
                "naive_correct_rate": 0.6,
                "naive_silent_wrong_rate": 0.4,
                "naive_missing_rate": 0.0,
                "workload_reduction_vs_audit": -0.125,
            }
        ]
    }
    path = tmp_path / "lifecycle.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    rows = load_summary_rows([path])
    markdown = render_summary_rows(rows)
    rendered_json = json.loads(render_summary_rows(rows, output_format="json"))

    assert rows[0]["new_ref"] == "v2"
    assert "40.0%" in markdown
    assert "-12.5%" in markdown
    assert rendered_json[0]["refmark_auto_rate"] == 0.5


def test_lifecycle_summarize_cli_writes_markdown(tmp_path):
    payload_path = tmp_path / "lifecycle.json"
    output_path = tmp_path / "summary.md"
    payload_path.write_text(
        json.dumps(
            {
                "summary_rows": [
                    {
                        "repo_url": "repo",
                        "old_ref": "a",
                        "new_ref": "b",
                        "old_labels": 2,
                        "refmark_auto_rate": 1.0,
                        "refmark_review_rate": 0.0,
                        "refmark_stale_rate": 0.0,
                        "naive_correct_rate": 0.5,
                        "naive_silent_wrong_rate": 0.5,
                        "naive_missing_rate": 0.0,
                        "workload_reduction_vs_audit": 1.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "lifecycle-summarize",
            str(payload_path),
            "--output",
            str(output_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "100.0%" in output_path.read_text(encoding="utf-8")


def test_lifecycle_validate_labels_cli_reports_stale_examples(tmp_path):
    previous_manifest = tmp_path / "previous.jsonl"
    current_manifest = tmp_path / "current.jsonl"
    examples = tmp_path / "examples.jsonl"
    report = tmp_path / "report.json"
    previous = [
        RegionRecord(
            doc_id="policy",
            region_id="P01",
            text="Original refund text.",
            start_line=1,
            end_line=1,
            ordinal=1,
            hash="old-hash",
        )
    ]
    current = [
        RegionRecord(
            doc_id="policy",
            region_id="P01",
            text="Changed refund text.",
            start_line=1,
            end_line=1,
            ordinal=1,
            hash="new-hash",
        )
    ]
    write_manifest(previous, previous_manifest)
    write_manifest(current, current_manifest)
    examples.write_text(
        json.dumps(
            {
                "query": "What is the refund rule?",
                "gold_refs": ["policy:P01"],
                "source_hashes": {"policy:P01": "old-hash"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "lifecycle-validate-labels",
            str(current_manifest),
            str(examples),
            "--previous-manifest",
            str(previous_manifest),
            "--previous-revision",
            "rev-a",
            "--current-revision",
            "rev-b",
            "--max-stale",
            "1",
            "--output",
            str(report),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert result.returncode == 0, result.stderr
    assert payload["stale_example_count"] == 1
    assert payload["status"] == "ok"
    assert payload["ci_status"]["counts"]["changed_refs"] == 1
    assert payload["revision_diff"]["changed_refs"] == ["policy:P01"]


def test_manifest_diff_cli_reports_revision_churn_and_affected_examples(tmp_path):
    previous_manifest = tmp_path / "previous.jsonl"
    current_manifest = tmp_path / "current.jsonl"
    examples = tmp_path / "examples.jsonl"
    report = tmp_path / "diff.json"
    previous = [
        RegionRecord("policy", "P01", "Refunds last 30 days.", 1, 1, 1, "h-old"),
        RegionRecord("policy", "P02", "Shipping is tracked.", 2, 2, 2, "h-ship"),
    ]
    current = [
        RegionRecord("policy", "P01", "Refunds last 45 days.", 1, 1, 1, "h-new"),
        RegionRecord("policy", "P03", "Support replies by email.", 3, 3, 3, "h-support"),
    ]
    write_manifest(previous, previous_manifest)
    write_manifest(current, current_manifest)
    examples.write_text(
        "\n".join(
            [
                json.dumps({"query": "How long do refunds last?", "gold_refs": ["policy:P01"]}),
                json.dumps({"query": "How is shipping tracked?", "gold_refs": ["policy:P02"]}),
                json.dumps({"query": "How do I contact support?", "gold_refs": ["policy:P03"]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "manifest-diff",
            str(previous_manifest),
            str(current_manifest),
            "--examples",
            str(examples),
            "--previous-revision",
            "rev-a",
            "--current-revision",
            "rev-b",
            "--max-stale",
            "2",
            "--output",
            str(report),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert result.returncode == 0, result.stderr
    assert payload["schema"] == "refmark.manifest_diff.v1"
    assert payload["summary"]["added_refs"] == 1
    assert payload["summary"]["removed_refs"] == 1
    assert payload["summary"]["changed_refs"] == 1
    assert payload["affected_example_count"] == 2
    assert payload["revision_diff"]["added_refs"] == ["policy:P03"]
    assert payload["revision_diff"]["removed_refs"] == ["policy:P02"]
    assert payload["revision_diff"]["changed_refs"] == ["policy:P01"]
    assert [item["changed_refs"] for item in payload["affected_examples"]] == [["policy:P01"], []]
    assert [item["missing_refs"] for item in payload["affected_examples"]] == [[], ["policy:P02"]]


def test_lifecycle_primitives_detect_changed_removed_and_stale_examples():
    previous = CorpusMap.from_records(
        [
            RegionRecord("policy", "P01", "Refunds last 30 days.", 1, 1, 1, "h-old"),
            RegionRecord("policy", "P02", "Shipping is tracked.", 2, 2, 2, "h-ship"),
        ],
        revision_id="rev-a",
    )
    current = CorpusMap.from_records(
        [RegionRecord("policy", "P01", "Refunds last 45 days.", 1, 1, 1, "h-new")],
        revision_id="rev-b",
    )
    suite = EvalSuite.from_rows(
        [
            {
                "query": "How long do refunds last?",
                "gold_refs": ["policy:P01"],
                "source_hashes": {"policy:P01": "h-old"},
            },
            {
                "query": "How is shipping tracked?",
                "gold_refs": ["policy:P02"],
                "source_hashes": {"policy:P02": "h-ship"},
            },
        ],
        corpus=current,
    )

    diff = current.diff_revision(previous)
    stale = current.stale_examples(suite.examples)

    assert diff.changed_refs == ["policy:P01"]
    assert diff.removed_refs == ["policy:P02"]
    assert [item.changed_refs for item in stale] == [["policy:P01"], []]
    assert [item.missing_refs for item in stale] == [[], ["policy:P02"]]
