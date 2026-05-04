import json
import subprocess
import sys

from refmark.pipeline import RegionRecord, write_manifest
from refmark.lifecycle import (
    Region,
    evaluate_chunk_hash_quote_selector,
    evaluate_chunk_id_content_hash,
    evaluate_eval_label_lifecycle,
    evaluate_layered_anchor_selector,
    evaluate_naive_path_ordinal,
    evaluate_qrels_source_hash,
    evaluate_stable_migration,
    FuzzyRegionIndex,
    load_summary_rows,
    render_summary_rows,
)
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


def test_layered_selector_preserves_same_ordinal_rewrite_with_quote() -> None:
    old = {
        "docs/page.md": [
            Region(
                path="docs/page.md",
                ref="docs/page.md:P001",
                ordinal=1,
                text="Refund policy overview. Customers can return items within 30 days. Use the portal.",
                fingerprint="old",
            )
        ]
    }
    new = {
        "docs/page.md": [
            Region(
                path="docs/page.md",
                ref="docs/page.md:P001",
                ordinal=1,
                text="Refund policy overview. Customers can return items within 45 days. Use the portal or email support.",
                fingerprint="new",
            )
        ]
    }
    stable = {
        "total": 1,
        "counts": {"fuzzy": 1},
        "status_by_ref": {"docs/page.md:P001": "fuzzy"},
    }

    report = evaluate_layered_anchor_selector(
        old,
        new,
        stable,
        same_ordinal_rewrite_threshold=0.50,
    )

    assert report["counts"]["preserved"] == 1
    assert report["examples"]["preserved"][0]["via"] == "same_ordinal_quote_rewrite"


def test_layered_selector_reviews_low_similarity_same_ordinal_rewrite() -> None:
    old = {
        "docs/page.md": [
            Region(
                path="docs/page.md",
                ref="docs/page.md:P001",
                ordinal=1,
                text="Refund policy overview. Customers can return items within 30 days. Use the portal.",
                fingerprint="old",
            )
        ]
    }
    new = {
        "docs/page.md": [
            Region(
                path="docs/page.md",
                ref="docs/page.md:P001",
                ordinal=1,
                text=(
                    "Refund policy overview. This page now describes fraud review, manual approvals, "
                    "warehouse exceptions, and escalation queues for enterprise accounts."
                ),
                fingerprint="new",
            )
        ]
    }
    stable = {
        "total": 1,
        "counts": {"fuzzy": 1},
        "status_by_ref": {"docs/page.md:P001": "fuzzy"},
    }

    report = evaluate_layered_anchor_selector(old, new, stable)

    assert report["counts"].get("preserved", 0) == 0
    assert report["counts"]["review_needed"] == 1
    assert report["examples"]["review"][0]["reason"] == "no_quote_selector_hit"


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


def test_lifecycle_competent_baselines_compare_moved_evidence():
    old = {
        "guide.md": [
            Region("guide.md", "guide.md:P001", 1, "Refunds are available for thirty days after purchase.", "h-refund"),
            Region("guide.md", "guide.md:P002", 2, "Shipping uses tracked parcel delivery for every order.", "h-ship"),
        ]
    }
    new = {
        "policy.md": [
            Region("policy.md", "policy.md:P001", 1, "Refunds are available for thirty days after purchase.", "h-refund"),
        ],
        "guide.md": [
            Region("guide.md", "guide.md:P001", 1, "Shipping uses tracked parcel delivery for every order.", "h-ship"),
        ],
    }

    stable = evaluate_stable_migration(old, new)
    naive = evaluate_naive_path_ordinal(old, new)
    chunk_hash = evaluate_chunk_id_content_hash(old, new, stable)
    source_hash = evaluate_qrels_source_hash(old, new, stable)
    quote_selector = evaluate_chunk_hash_quote_selector(old, new, stable)
    layered_selector = evaluate_layered_anchor_selector(old, new, stable)
    lifecycle = evaluate_eval_label_lifecycle(
        stable,
        naive,
        chunk_hash=chunk_hash,
        source_hash=source_hash,
        quote_selector=quote_selector,
        layered_selector=layered_selector,
    )

    assert stable["counts"]["moved_exact"] == 1
    assert naive["counts"]["wrong_same_id"] == 1
    assert chunk_hash["counts"]["false_stale_alert"] == 2
    assert source_hash["counts"]["false_stale_alert"] == 2
    assert quote_selector["counts"]["preserved"] == 2
    assert layered_selector["counts"]["preserved"] == 2
    assert lifecycle["method_comparison"]["chunk_id_only"]["silent_drift"] == 1
    assert lifecycle["method_comparison"]["chunk_id_content_hash"]["false_stale_alerts"] == 2
    assert lifecycle["method_comparison"]["chunk_hash_quote_selector"]["valid_evals_preserved"] == 2
    assert lifecycle["method_comparison"]["refmark_layered_selector"]["valid_evals_preserved"] == 2
    assert lifecycle["method_comparison"]["refmark"]["valid_evals_preserved"] == 2


def test_quote_selector_requires_review_for_ambiguous_quote_hits():
    old = {
        "guide.md": [
            Region("guide.md", "guide.md:P001", 1, "Refund policy overview allows returns after approval.", "h-old"),
        ]
    }
    new = {
        "policy-a.md": [
            Region("policy-a.md", "policy-a.md:P001", 1, "Refund policy overview allows returns after approval.", "h-a"),
        ],
        "policy-b.md": [
            Region("policy-b.md", "policy-b.md:P001", 1, "Refund policy overview allows returns after approval.", "h-b"),
        ]
    }
    stable = {
        "total": 1,
        "counts": {"fuzzy": 1},
        "status_by_ref": {"guide.md:P001": "fuzzy"},
    }

    quote_selector = evaluate_chunk_hash_quote_selector(old, new, stable)
    layered_selector = evaluate_layered_anchor_selector(old, new, stable)

    assert quote_selector["counts"]["review_needed"] == 1
    assert layered_selector["counts"]["review_needed"] == 1


def test_source_hash_baseline_preserves_unchanged_file_and_flags_changed_file():
    old = {
        "same.md": [Region("same.md", "same.md:P001", 1, "Stable text.", "h-same")],
        "changed.md": [Region("changed.md", "changed.md:P001", 1, "Moved text.", "h-moved")],
    }
    new = {
        "same.md": [Region("same.md", "same.md:P001", 1, "Stable text.", "h-same")],
        "changed.md": [Region("changed.md", "changed.md:P001", 1, "Different local text.", "h-other")],
        "moved.md": [Region("moved.md", "moved.md:P001", 1, "Moved text.", "h-moved")],
    }
    stable = evaluate_stable_migration(old, new)

    source_hash = evaluate_qrels_source_hash(old, new, stable)
    chunk_hash = evaluate_chunk_id_content_hash(old, new, stable)

    assert source_hash["counts"]["preserved"] == 1
    assert source_hash["counts"]["false_stale_alert"] == 1
    assert chunk_hash["counts"]["preserved"] == 1
    assert chunk_hash["counts"]["false_stale_alert"] == 1


def test_fuzzy_region_index_matches_expected_global_candidate():
    old_region = Region(
        "guide.md",
        "guide.md:P001",
        1,
        "Refund requests must include the receipt number and be filed within thirty days.",
        "h-old",
    )
    decoy = Region(
        "guide.md",
        "guide.md:P001",
        1,
        "Shipping requests must include a tracking number and delivery address.",
        "h-decoy",
    )
    moved = Region(
        "policy.md",
        "policy.md:P001",
        1,
        "Refund requests must include the receipt number and be filed within thirty calendar days.",
        "h-moved",
    )

    best = FuzzyRegionIndex([decoy, moved]).best_match(old_region)

    assert best is not None
    assert best[0] == moved
    assert best[1] >= 0.82


def test_stable_migration_prefers_global_fuzzy_over_weak_same_file_candidate():
    old = {
        "guide.md": [
            Region(
                "guide.md",
                "guide.md:P001",
                1,
                "Refund requests must include the receipt number and be filed within thirty days.",
                "h-old",
            )
        ]
    }
    new = {
        "guide.md": [
            Region(
                "guide.md",
                "guide.md:P001",
                1,
                "Shipping requests must include a tracking number and delivery address.",
                "h-decoy",
            )
        ],
        "policy.md": [
            Region(
                "policy.md",
                "policy.md:P001",
                1,
                "Refund requests must include the receipt number and be filed within thirty calendar days.",
                "h-moved",
            )
        ],
    }

    stable = evaluate_stable_migration(old, new)

    assert stable["counts"]["fuzzy"] == 1
    assert stable["examples"]["fuzzy"][0]["new_ref"] == "policy.md:P001"
