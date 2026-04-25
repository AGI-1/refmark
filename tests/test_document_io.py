from pathlib import Path
import json
import subprocess
import sys

from refmark.document_io import extract_docx_text, extract_pdf_text, text_mapping_extension


def test_text_mapping_extension_for_document_inputs():
    assert text_mapping_extension(Path("contract.docx")) == ".txt"
    assert text_mapping_extension(Path("offer.pdf")) == ".txt"
    assert text_mapping_extension(Path("notes.md")) == ".md"


def test_docx_extraction_from_generated_example(tmp_path):
    from examples.coverage_alignment.run import _write_docx

    path = tmp_path / "sample.docx"
    _write_docx(path, "Sample", ["First requirement.", "Second requirement."])

    text = extract_docx_text(path)

    assert "Sample" in text
    assert "First requirement." in text
    assert "Second requirement." in text


def test_pdf_extraction_from_generated_example(tmp_path):
    from examples.coverage_alignment.run import _write_pdf

    path = tmp_path / "sample.pdf"
    _write_pdf(path, "Sample", ["First requirement.", "Second requirement."])

    text = extract_pdf_text(path)

    assert "Sample" in text
    assert "First requirement." in text
    assert "Second requirement." in text


def test_cli_map_and_align_accept_docx_and_pdf(tmp_path):
    from examples.coverage_alignment.run import _write_docx, _write_pdf

    source = tmp_path / "request.docx"
    target = tmp_path / "offer.pdf"
    manifest = tmp_path / "manifest.jsonl"
    coverage_json = tmp_path / "coverage.json"
    coverage_html = tmp_path / "coverage.html"
    _write_docx(source, "Request", ["The service must provide encrypted data storage."])
    _write_pdf(target, "Offer", ["The platform provides encrypted data storage."])

    subprocess.run(
        [sys.executable, "-m", "refmark.cli", "map", str(source), "-o", str(manifest)],
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "align",
            str(source),
            str(target),
            "--coverage-json",
            str(coverage_json),
            "--coverage-html",
            str(coverage_html),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    manifest_rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    coverage_rows = json.loads(coverage_json.read_text(encoding="utf-8"))
    assert manifest_rows
    assert "encrypted data storage" in completed.stdout
    assert coverage_rows
    assert "Refmark Coverage Review" in coverage_html.read_text(encoding="utf-8")
