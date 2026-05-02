import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from refmark.pipeline import (
    align_region_records,
    build_region_manifest,
    build_section_map,
    evaluate_alignment_coverage,
    expand_region_context,
    read_manifest,
    render_coverage_html,
    summarize_coverage,
    write_manifest,
)


def test_build_region_manifest_and_expand_neighbors():
    _marked, records = build_region_manifest(
        "Alpha claim.\n\nBeta supporting detail.\n\nGamma conclusion.\n",
        ".txt",
        doc_id="doc",
    )

    assert [record.region_id for record in records] == ["P01", "P02", "P03"]
    assert records[1].prev_region_id == "P01"
    assert records[1].next_region_id == "P03"
    expanded = expand_region_context(records, ["P02"], before=1, after=1)
    assert [record.region_id for record in expanded] == ["P01", "P02", "P03"]


def test_build_region_manifest_assigns_markdown_heading_parent_regions():
    _marked, records = build_region_manifest(
        "# Security\n\nToken rotation details.\n\nAudit retention details.\n\n# Billing\n\nInvoice export details.\n",
        ".md",
        doc_id="doc",
    )

    assert records[0].parent_region_id is None
    assert [record.parent_region_id for record in records[1:3]] == ["P01", "P01"]
    assert records[4].parent_region_id == "P04"

    expanded = expand_region_context(records, ["P02"], same_parent=True, include_parent=True)

    assert [record.region_id for record in expanded] == ["P01", "P02", "P03"]


def test_build_section_map_returns_heading_ref_ranges():
    _marked, records = build_region_manifest(
        "# Security\n\nToken rotation details.\n\nAudit retention details.\n\n# Billing\n\nInvoice export details.\n",
        ".md",
        doc_id="guide",
    )

    sections = build_section_map(records)

    assert [section.title for section in sections] == ["Security", "Billing"]
    assert sections[0].heading_ref == "guide:P01"
    assert sections[0].range_ref == "guide:P01-guide:P03"
    assert sections[0].refs == ["guide:P01", "guide:P02", "guide:P03"]
    assert sections[1].range_ref == "guide:P04-guide:P05"


def test_manifest_jsonl_roundtrip(tmp_path):
    _marked, records = build_region_manifest("One.\n\nTwo.\n", ".txt", doc_id="doc")
    path = tmp_path / "manifest.jsonl"

    write_manifest(records, path)
    loaded = read_manifest(path)

    assert [record.to_dict() for record in loaded] == [record.to_dict() for record in records]


def test_align_region_records_finds_lexical_overlap():
    _source_marked, source = build_region_manifest(
        "Invoices include expedited shipping fees.\n\nRefund windows vary by tier.\n",
        ".txt",
        doc_id="source",
    )
    _target_marked, target = build_region_manifest(
        "The expedited shipping fee is added to invoice totals.\n\nEnterprise refunds last 45 days.\n",
        ".txt",
        doc_id="target",
    )

    alignments = align_region_records(source, target, top_k=1)

    assert alignments[0][0].target_region_id == "P01"
    assert alignments[0][0].score > 0
    assert "expedited" in alignments[0][0].shared_terms


def test_evaluate_alignment_coverage_marks_gaps_and_expansion_gain():
    _source_marked, source = build_region_manifest(
        "Expedited shipping requires service credits.\n\nEU data residency is mandatory.\n",
        ".txt",
        doc_id="source",
    )
    _target_marked, target = build_region_manifest(
        "Expedited shipping is available.\n\nService credits apply after outages.\n\nData is stored in United States regions.\n",
        ".txt",
        doc_id="target",
    )

    coverage = evaluate_alignment_coverage(source, target, threshold=0.75, expand_after=1)
    html = render_coverage_html(coverage, title="demo")

    assert coverage[0].status == "covered"
    assert coverage[0].expanded_score > coverage[0].naive_score
    assert [record.region_id for record in coverage[0].expanded_targets] == ["P01", "P02"]
    assert coverage[1].status == "gap"
    assert summarize_coverage(coverage)["items_improved_by_expansion"] == 1
    assert "coverage rate" in html
    assert "Expanded Evidence" in html
    assert "GAP" in html


def test_alignment_normalizes_basic_variants():
    _source_marked, source = build_region_manifest(
        "The supplier must encrypt European customer data.\n",
        ".txt",
        doc_id="source",
    )
    _target_marked, target = build_region_manifest(
        "Customer data is encrypted in EU regions.\n",
        ".txt",
        doc_id="target",
    )

    alignments = align_region_records(source, target, top_k=1)

    assert alignments[0][0].target_region_id == "P01"
    assert {"encryption", "europe", "customer", "data"} <= set(alignments[0][0].shared_terms)


def test_coverage_flags_numeric_conflicts_but_allows_minimums():
    _source_marked, source = build_region_manifest(
        "The battery must provide at least 500 kWh usable capacity.\n\nDelivery must be completed within 90 days.\n",
        ".txt",
        doc_id="source",
    )
    _target_marked, target = build_region_manifest(
        "The proposed battery has 520 kWh usable capacity.\n\nDelivery is planned for 120 days.\n",
        ".txt",
        doc_id="target",
    )

    coverage = evaluate_alignment_coverage(source, target, threshold=0.35, expand_after=0)

    assert coverage[0].status == "covered"
    assert coverage[0].numeric_conflict is False
    assert coverage[1].status == "gap"
    assert coverage[1].numeric_conflict is True


def test_numeric_conflict_uses_units_with_expanded_context():
    _source_marked, source = build_region_manifest(
        "Delivery must be completed within 90 days.\n",
        ".txt",
        doc_id="source",
    )
    _target_marked, target = build_region_manifest(
        "Delivery is planned for 120 days.\n\nMaintenance lasts 3 years.\n",
        ".txt",
        doc_id="target",
    )

    coverage = evaluate_alignment_coverage(source, target, threshold=0.35, expand_after=1)

    assert coverage[0].numeric_conflict is True
    assert coverage[0].status == "gap"


def test_pipeline_cli_map_expand_and_align(tmp_path):
    source = tmp_path / "source.txt"
    target = tmp_path / "target.txt"
    manifest = tmp_path / "manifest.jsonl"
    marked_source = tmp_path / "source_marked.txt"
    marked_target = tmp_path / "target_marked.txt"
    coverage_html = tmp_path / "coverage.html"
    source.write_text("Expedited shipping is charged.\n\nRefunds vary.\n", encoding="utf-8")
    target.write_text("Invoice totals include expedited shipping.\n\nRefund policy depends on tier.\n", encoding="utf-8")

    subprocess.run(
        [sys.executable, "-m", "refmark.cli", "map", str(source), "-o", str(manifest)],
        check=True,
        capture_output=True,
        text=True,
    )
    expanded = subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "expand",
            str(manifest),
            "--refs",
            "P01",
            "--after",
            "1",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    aligned = subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "align",
            str(source),
            str(target),
            "--top-k",
            "1",
            "--marked-source",
            str(marked_source),
            "--marked-target",
            str(marked_target),
            "--coverage-html",
            str(coverage_html),
            "--no-expanded-evidence",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    expanded_payload = json.loads(expanded.stdout)
    aligned_payload = json.loads(aligned.stdout)
    assert [item["region_id"] for item in expanded_payload] == ["P01", "P02"]
    assert aligned_payload[0][0]["target_region_id"] == "P01"
    assert "[@" in marked_source.read_text(encoding="utf-8")
    assert "[@" in marked_target.read_text(encoding="utf-8")
    assert "Expanded Evidence" not in coverage_html.read_text(encoding="utf-8")
    assert "Wrote marked source" in aligned.stderr


def test_pipeline_cli_expand_same_parent(tmp_path):
    source = tmp_path / "source.md"
    manifest = tmp_path / "manifest.jsonl"
    source.write_text("# Security\n\nToken rotation.\n\nAudit retention.\n\n# Billing\n\nInvoice export.\n", encoding="utf-8")

    subprocess.run(
        [sys.executable, "-m", "refmark.cli", "map", str(source), "-o", str(manifest), "--min-words", "0"],
        check=True,
        capture_output=True,
        text=True,
    )
    expanded = subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "expand",
            str(manifest),
            "--refs",
            "P02",
            "--same-parent",
            "--include-parent",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    expanded_payload = json.loads(expanded.stdout)
    assert [item["region_id"] for item in expanded_payload] == ["P01", "P02", "P03"]


def test_pipeline_cli_expand_rejects_invalid_citation_refs(tmp_path):
    source = tmp_path / "source.txt"
    manifest = tmp_path / "manifest.jsonl"
    source.write_text("Alpha.\n\nBeta.\n", encoding="utf-8")

    subprocess.run(
        [sys.executable, "-m", "refmark.cli", "map", str(source), "-o", str(manifest), "--min-words", "0"],
        check=True,
        capture_output=True,
        text=True,
    )
    expanded = subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "expand",
            str(manifest),
            "--refs",
            "not-a-ref",
        ],
        capture_output=True,
        text=True,
    )

    assert expanded.returncode == 1
    assert "Invalid citation ref" in expanded.stderr


def test_pipeline_cli_pack_context(tmp_path):
    source = tmp_path / "source.txt"
    manifest = tmp_path / "manifest.jsonl"
    source.write_text("Alpha evidence.\n\nBeta evidence.\n", encoding="utf-8")

    subprocess.run(
        [sys.executable, "-m", "refmark.cli", "map", str(source), "-o", str(manifest), "--min-words", "0"],
        check=True,
        capture_output=True,
        text=True,
    )
    packed = subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "pack-context",
            str(manifest),
            "--refs",
            "source:P01-source:P02",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(packed.stdout)
    assert payload["refs"] == ["source:P01", "source:P02"]
    assert "Alpha evidence" in payload["text"]


def test_pipeline_cli_question_prompt_supports_overridable_template(tmp_path):
    source = tmp_path / "source.txt"
    manifest = tmp_path / "manifest.jsonl"
    template = tmp_path / "template.txt"
    source.write_text("Alpha evidence.\n\nBeta evidence.\n", encoding="utf-8")
    template.write_text("refs={refs}\njson={refs_json}\ncontext={context}\n", encoding="utf-8")

    subprocess.run(
        [sys.executable, "-m", "refmark.cli", "map", str(source), "-o", str(manifest), "--min-words", "0"],
        check=True,
        capture_output=True,
        text=True,
    )
    prompt = subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "question-prompt",
            str(manifest),
            "--refs",
            "source:P01-source:P02",
            "--template",
            str(template),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "refs=source:P01, source:P02" in prompt.stdout
    assert '"source:P01"' in prompt.stdout
    assert "Beta evidence" in prompt.stdout


def test_pipeline_cli_discovery_review_and_card(tmp_path):
    source = tmp_path / "source.txt"
    manifest = tmp_path / "manifest.jsonl"
    discovery = tmp_path / "discovery.json"
    review = tmp_path / "review.json"
    question_plan = tmp_path / "question_plan.json"
    source.write_text("Table of contents\nDefinitions\n\nSDS means safety data sheet.\n", encoding="utf-8")

    subprocess.run(
        [sys.executable, "-m", "refmark.cli", "map", str(source), "-o", str(manifest), "--min-words", "0"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [sys.executable, "-m", "refmark.cli", "discover", str(manifest), "-o", str(discovery), "--mode", "windowed", "--window-tokens", "8"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [sys.executable, "-m", "refmark.cli", "review-discovery", str(discovery), "--manifest", str(manifest), "-o", str(review)],
        check=True,
        capture_output=True,
        text=True,
    )
    card = subprocess.run(
        [sys.executable, "-m", "refmark.cli", "discovery-card", str(manifest), str(discovery), "--ref", "source:P02"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "question-plan",
            str(manifest),
            str(discovery),
            "-o",
            str(question_plan),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    review_payload = json.loads(review.read_text(encoding="utf-8"))
    card_payload = json.loads(card.stdout)
    discovery_payload = json.loads(discovery.read_text(encoding="utf-8"))
    plan_payload = json.loads(question_plan.read_text(encoding="utf-8"))
    assert review_payload["schema"] == "refmark.discovery_review.v1"
    assert discovery_payload["windows"]
    assert discovery_payload["clusters"]
    assert card_payload["schema"] == "refmark.discovery_context_card.v1"
    assert card_payload["card"]["stable_ref"] == "source:P02"
    assert plan_payload["schema"] == "refmark.question_plan.v1"
    assert "direct" in plan_payload["summary"]["by_style"]


def test_pipeline_cli_eval_index(tmp_path):
    source = tmp_path / "source.txt"
    index = tmp_path / "index.json"
    manifest = tmp_path / "manifest.jsonl"
    examples = tmp_path / "examples.jsonl"
    source.write_text("Refunds are available within 30 days.\n\nShipping is non-refundable.\n", encoding="utf-8")
    examples.write_text(
        json.dumps({"query": "refunds within 30 days", "gold_refs": ["source:P01"]}) + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "build-index",
            str(source),
            "-o",
            str(index),
            "--source",
            "local",
            "--min-words",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "map",
            str(source),
            "-o",
            str(manifest),
            "--min-words",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    evaluated = subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "eval-index",
            str(index),
            str(examples),
            "--manifest",
            str(manifest),
            "--top-k",
            "2",
            "--min-hit-at-k",
            "0.9",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(evaluated.stdout)
    assert payload["schema"] == "refmark.eval_index_report.v1"
    assert payload["metrics"]["count"] == 1.0
    assert payload["metrics"]["hit_at_k"] == 1.0
    assert payload["ci_status"]["status"] == "pass"
    assert payload["validation"] == {"missing": [], "ambiguous": []}

    inspected = subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "inspect-index",
            str(index),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    smell_payload = json.loads(inspected.stdout)
    assert smell_payload["schema"] == "refmark.index_data_smells.v1"
    assert "weighted_smell_score" in smell_payload["diagnostics"]["summary"]


def test_pipeline_cli_compare_index_reports_multiple_strategies(tmp_path):
    source = tmp_path / "source.txt"
    index = tmp_path / "index.json"
    manifest = tmp_path / "manifest.jsonl"
    examples = tmp_path / "examples.jsonl"
    report = tmp_path / "compare.json"
    source.write_text(
        "Refunds are available within 30 days for damaged goods.\n\n"
        "Shipping labels can be printed from the account portal.\n",
        encoding="utf-8",
    )
    examples.write_text(
        json.dumps({"query": "refunds damaged goods", "gold_refs": ["source:P01"]}) + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "build-index",
            str(source),
            "-o",
            str(index),
            "--source",
            "local",
            "--min-words",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "map",
            str(source),
            "-o",
            str(manifest),
            "--min-words",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    compared = subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "compare-index",
            str(index),
            str(examples),
            "--manifest",
            str(manifest),
            "--strategies",
            "flat,hierarchical,rerank",
            "--top-k",
            "2",
            "--min-best-hit-at-k",
            "0.9",
            "--fail-on-regression",
            "-o",
            str(report),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert compared.returncode == 0, compared.stderr
    assert payload["schema"] == "refmark.compare_index_report.v1"
    assert set(payload["runs"]) == {"flat", "hierarchical", "rerank"}
    assert payload["best_by_hit_at_k"]["metrics"]["hit_at_k"] == 1.0
    assert payload["ci_status"]["status"] == "ok"
    assert payload["validation"] == {"missing": [], "ambiguous": []}
    assert payload["run_artifacts"]["flat"]["comparison_key"]
    assert payload["smell_summaries"]["flat"]["smell_count"] >= 0


def test_pipeline_cli_compare_runs_reports_saved_eval_deltas(tmp_path):
    source = tmp_path / "source.txt"
    index = tmp_path / "index.json"
    manifest = tmp_path / "manifest.jsonl"
    examples = tmp_path / "examples.jsonl"
    flat_report = tmp_path / "eval_flat.json"
    rerank_report = tmp_path / "eval_rerank.json"
    compare_report = tmp_path / "compare_runs.json"
    source.write_text(
        "Refunds are available within 30 days for damaged goods.\n\n"
        "Shipping labels can be printed from the account portal.\n",
        encoding="utf-8",
    )
    examples.write_text(
        json.dumps({"query": "refunds damaged goods", "gold_refs": ["source:P01"]}) + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "build-index",
            str(source),
            "-o",
            str(index),
            "--source",
            "local",
            "--min-words",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "map",
            str(source),
            "-o",
            str(manifest),
            "--min-words",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    for strategy, output in [("flat", flat_report), ("rerank", rerank_report)]:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "refmark.cli",
                "eval-index",
                str(index),
                str(examples),
                "--manifest",
                str(manifest),
                "--strategy",
                strategy,
                "--top-k",
                "2",
                "-o",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    compared = subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "compare-runs",
            str(flat_report),
            str(rerank_report),
            "--baseline",
            "flat",
            "--min-best-hit-at-k",
            "0.9",
            "--fail-on-regression",
            "-o",
            str(compare_report),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(compare_report.read_text(encoding="utf-8"))
    assert compared.returncode == 0, compared.stderr
    assert payload["schema"] == "refmark.compare_runs_report.v1"
    assert payload["compatibility"]["same_corpus_and_eval"] is True
    assert [row["name"] for row in payload["table"]] == ["flat", "rerank"]
    assert payload["baseline"] == "flat"
    assert payload["best_by_hit_at_k"]["metrics"]["hit_at_k"] == 1.0
    assert payload["table"][1]["delta_vs_baseline"]["hit_at_k"] == 0.0


def test_pipeline_cli_map_and_build_index_use_same_directory_doc_ids(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    source = docs / "security-guide.md"
    index = tmp_path / "index.json"
    manifest = tmp_path / "manifest.jsonl"
    examples = tmp_path / "examples.jsonl"
    source.write_text(
        "# Security Guide\n\n"
        "Rotate API tokens every ninety days after replacement credentials are deployed.\n",
        encoding="utf-8",
    )
    examples.write_text(
        json.dumps({"query": "rotate api tokens", "gold_refs": ["security_guide:P02"]}) + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "map",
            str(docs),
            "-o",
            str(manifest),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "build-index",
            str(docs),
            "-o",
            str(index),
            "--source",
            "local",
            "--min-words",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    evaluated = subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "eval-index",
            str(index),
            str(examples),
            "--manifest",
            str(manifest),
            "--top-k",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(evaluated.stdout)
    assert payload["validation"] == {"missing": [], "ambiguous": []}
    assert payload["metrics"]["hit_at_k"] == 1.0


def test_pipeline_cli_eval_index_fails_thresholds_and_preserves_stale_hashes(tmp_path):
    source = tmp_path / "source.txt"
    index = tmp_path / "index.json"
    manifest = tmp_path / "manifest.jsonl"
    examples = tmp_path / "examples.jsonl"
    source.write_text("Refunds are available within 30 days.\n\nShipping is non-refundable.\n", encoding="utf-8")
    examples.write_text(
        json.dumps(
            {
                "query": "refunds within 30 days",
                "gold_refs": ["source:P01"],
                "source_hashes": {"source:P01": "stale-hash"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "build-index",
            str(source),
            "-o",
            str(index),
            "--source",
            "local",
            "--min-words",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "map",
            str(source),
            "-o",
            str(manifest),
            "--min-words",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    evaluated = subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "eval-index",
            str(index),
            str(examples),
            "--manifest",
            str(manifest),
            "--top-k",
            "2",
            "--max-stale",
            "0",
            "--fail-on-regression",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert evaluated.returncode == 3
    payload = json.loads(evaluated.stdout)
    assert payload["ci_status"]["status"] == "fail"
    assert payload["stale_examples"][0]["changed_refs"] == ["source:P01"]
    assert payload["data_smells"]["summary"]["by_type"]["stale_label"] == 1


def test_pipeline_cli_eval_index_writes_data_smell_report(tmp_path):
    source = tmp_path / "source.txt"
    index = tmp_path / "index.json"
    manifest = tmp_path / "manifest.jsonl"
    examples = tmp_path / "examples.jsonl"
    retriever_results = tmp_path / "hits.jsonl"
    smells = tmp_path / "smells.json"
    adapt_plan = tmp_path / "adapt_plan.json"
    source.write_text("Refunds are available within 30 days.\n\nShipping is non-refundable.\n", encoding="utf-8")
    examples.write_text(
        json.dumps({"query": "where are refunds", "gold_refs": ["source:P01"]}) + "\n",
        encoding="utf-8",
    )
    retriever_results.write_text(
        json.dumps({"query": "where are refunds", "hits": [{"stable_ref": "source:P02", "score": 1.0}]}) + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "build-index",
            str(source),
            "-o",
            str(index),
            "--source",
            "local",
            "--min-words",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "map",
            str(source),
            "-o",
            str(manifest),
            "--min-words",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    evaluated = subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "eval-index",
            str(index),
            str(examples),
            "--manifest",
            str(manifest),
            "--retriever-results",
            str(retriever_results),
            "--smell-report-output",
            str(smells),
            "--adapt-plan-output",
            str(adapt_plan),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(evaluated.stdout)
    smell_payload = json.loads(smells.read_text(encoding="utf-8"))
    plan_payload = json.loads(adapt_plan.read_text(encoding="utf-8"))
    assert payload["data_smells"]["schema"] == "refmark.data_smells.v1"
    assert payload["adaptation_plan"]["schema"] == "refmark.adaptation_plan.v1"
    assert smell_payload["summary"]["by_type"]["hard_ref"] == 1
    assert plan_payload["summary"]["source_run_fingerprint"] == smell_payload["summary"]["run_fingerprint"]
    assert any(smell["type"] == "confusion_pair" for smell in smell_payload["smells"])


def test_pipeline_cli_eval_index_can_call_http_retriever(tmp_path):
    source = tmp_path / "source.txt"
    index = tmp_path / "index.json"
    examples = tmp_path / "examples.jsonl"
    source.write_text("Refunds are available within 30 days.\n\nShipping is non-refundable.\n", encoding="utf-8")
    examples.write_text(
        json.dumps({"query": "refunds within 30 days", "gold_refs": ["source:P01"]}) + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "build-index",
            str(source),
            "-o",
            str(index),
            "--source",
            "local",
            "--min-words",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            body = json.dumps({"hits": [{"stable_ref": "source:P01", "score": 1.0}]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        evaluated = subprocess.run(
            [
                sys.executable,
                "-m",
                "refmark.cli",
                "eval-index",
                str(index),
                str(examples),
                "--retriever-endpoint",
                f"http://127.0.0.1:{server.server_port}/retrieve",
                "--top-k",
                "1",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)

    payload = json.loads(evaluated.stdout)
    assert payload["metrics"]["hit_at_1"] == 1.0
    assert payload["settings"]["retriever_endpoint"] is True


def test_pipeline_cli_eval_index_can_score_exported_retriever_results(tmp_path):
    source = tmp_path / "source.txt"
    index = tmp_path / "index.json"
    examples = tmp_path / "examples.jsonl"
    retriever_results = tmp_path / "hits.jsonl"
    source.write_text("Refunds are available within 30 days.\n\nShipping is non-refundable.\n", encoding="utf-8")
    examples.write_text(
        json.dumps({"query": "refunds within 30 days", "gold_refs": ["source:P01"]}) + "\n",
        encoding="utf-8",
    )
    retriever_results.write_text(
        json.dumps({"query": "refunds within 30 days", "hits": [{"stable_ref": "source:P01", "score": 1.0}]}) + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "build-index",
            str(source),
            "-o",
            str(index),
            "--source",
            "local",
            "--min-words",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    evaluated = subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "eval-index",
            str(index),
            str(examples),
            "--retriever-results",
            str(retriever_results),
            "--top-k",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(evaluated.stdout)
    assert payload["metrics"]["hit_at_1"] == 1.0
    assert payload["settings"]["retriever_results"] == str(retriever_results)
