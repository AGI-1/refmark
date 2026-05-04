import json
import subprocess
import sys
from pathlib import Path

from refmark.document_io import extract_docx_text
from refmark.ephemeral import apply_ephemeral_edits, build_ephemeral_map


def test_ephemeral_map_and_apply_text_replacement(tmp_path: Path):
    source = tmp_path / "note.md"
    output = tmp_path / "patched.md"
    source.write_text("First paragraph.\n\nSecond paragraph.\n", encoding="utf-8")

    mapped = build_ephemeral_map(source, doc_id="note")
    assert "[@P01]" in mapped.document.marked_text
    assert "P01" in mapped.instructions

    result = apply_ephemeral_edits(
        source,
        [{"ref": "P02", "action": "replace", "new_text": "Second paragraph, revised."}],
        output=output,
    )

    assert result["ok"] is True
    assert output.read_text(encoding="utf-8") == "First paragraph.\n\nSecond paragraph, revised.\n"


def test_ephemeral_apply_docx_exact_paragraph_replacement(tmp_path: Path):
    from examples.coverage_alignment.run import _write_docx

    source = tmp_path / "contract.docx"
    output = tmp_path / "contract_patched.docx"
    _write_docx(source, "Contract", ["Payment is due in 30 days.", "Termination requires notice."])

    result = apply_ephemeral_edits(
        source,
        [{"ref": "P02", "action": "replace", "new_text": "Payment is due in 45 days."}],
        output=output,
    )

    assert result["ok"] is True
    text = extract_docx_text(output)
    assert "Payment is due in 45 days." in text
    assert "Termination requires notice." in text


def test_ephemeral_apply_rejects_docx_multi_paragraph_replacement(tmp_path: Path):
    from examples.coverage_alignment.run import _write_docx

    source = tmp_path / "contract.docx"
    output = tmp_path / "contract_patched.docx"
    _write_docx(source, "Contract", ["Payment is due in 30 days."])

    result = apply_ephemeral_edits(
        source,
        [{"ref": "P02", "action": "replace", "new_text": "Payment is due in 45 days.\nLate fees are waived."}],
        output=output,
    )

    assert result["ok"] is False
    assert "single paragraph" in result["errors"][0]
    assert not output.exists()


def test_ephemeral_apply_rejects_ambiguous_text_replacement(tmp_path: Path):
    source = tmp_path / "note.txt"
    output = tmp_path / "patched.txt"
    source.write_text("Same paragraph.\n\nSame paragraph.\n", encoding="utf-8")

    result = apply_ephemeral_edits(
        source,
        [{"ref": "P01", "action": "replace", "new_text": "Changed."}],
        output=output,
    )

    assert result["ok"] is False
    assert "matched 2 source spans" in result["errors"][0]
    assert not output.exists()


def test_ephemeral_apply_reports_unknown_ref_and_unsupported_action(tmp_path: Path):
    source = tmp_path / "note.md"
    output = tmp_path / "patched.md"
    source.write_text("Alpha paragraph.\n\nBeta paragraph.\n", encoding="utf-8")

    result = apply_ephemeral_edits(
        source,
        [
            {"ref": "P99", "action": "replace", "new_text": "Unknown."},
            {"ref": "P01", "action": "delete", "new_text": ""},
        ],
        output=output,
    )

    assert result["ok"] is False
    assert "Unknown ref: P99" in result["errors"]
    assert "Unsupported ephemeral action for P01: delete" in result["errors"]
    assert not output.exists()


def test_ephemeral_apply_pdf_writes_extracted_text_warning(tmp_path: Path, monkeypatch):
    source = tmp_path / "report.pdf"
    output = tmp_path / "report_patched.txt"
    source.write_bytes(b"%PDF-pretend")
    extracted = "Executive summary.\n\nThe fee is 30 EUR.\n"

    monkeypatch.setattr("refmark.documents.extract_document_text", lambda _path: extracted)
    monkeypatch.setattr("refmark.ephemeral.extract_document_text", lambda _path: extracted)

    result = apply_ephemeral_edits(
        source,
        [{"ref": "P02", "action": "replace", "new_text": "The fee is 45 EUR."}],
        output=output,
    )

    assert result["ok"] is True
    assert "PDF output is patched extracted text, not an edited PDF." in result["warnings"]
    assert output.read_text(encoding="utf-8") == "Executive summary.\n\nThe fee is 45 EUR.\n"


def test_ephemeral_cli_map_and_apply(tmp_path: Path):
    source = tmp_path / "note.md"
    mapped = tmp_path / "mapped.json"
    edits = tmp_path / "edits.json"
    output = tmp_path / "patched.md"
    source.write_text("Alpha paragraph.\n\nBeta paragraph.\n", encoding="utf-8")
    edits.write_text(
        json.dumps({"edits": [{"ref": "P01", "action": "replace", "new_text": "Alpha paragraph revised."}]}),
        encoding="utf-8",
    )

    subprocess.run(
        [sys.executable, "-m", "refmark.cli", "ephemeral-map", str(source), "--json", "-o", str(mapped)],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    payload = json.loads(mapped.read_text(encoding="utf-8"))
    assert payload["schema"] == "refmark.ephemeral_map.v1"
    assert payload["regions"][0]["region_id"] == "P01"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "ephemeral-apply",
            str(source),
            "--edits-file",
            str(edits),
            "-o",
            str(output),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert "Alpha paragraph revised." in output.read_text(encoding="utf-8")


def test_ephemeral_cli_rejects_invalid_json_without_writing_output(tmp_path: Path):
    source = tmp_path / "note.md"
    output = tmp_path / "patched.md"
    source.write_text("Alpha paragraph.\n\nBeta paragraph.\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "ephemeral-apply",
            str(source),
            "--edits-json",
            "{\"edits\":",
            "-o",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 1
    assert "Invalid edit payload" in result.stderr
    assert not output.exists()
