import subprocess
import sys

from refmark.prompt import build_reference_prompt


def test_build_reference_prompt_marks_document_and_requests_ranges():
    result = build_reference_prompt(
        "Alpha supports the first claim.\n\nBeta supports the second claim.\n",
        ".txt",
        question="Which claims are supported?",
    )

    assert result.marker_count == 2
    assert "[@P01]" in result.prompt
    assert "[@P02]" in result.prompt
    assert "[P01-P03]" in result.prompt
    assert "QUESTION:\nWhich claims are supported?" in result.prompt
    assert "Return the answer with region citations" in result.prompt


def test_enrich_prompt_cli_outputs_marked_prompt(tmp_path):
    document = tmp_path / "doc.txt"
    document.write_text("First paragraph.\n\nSecond paragraph.\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "refmark.cli",
            "enrich-prompt",
            str(document),
            "--question",
            "What is in the document?",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "MARKED DOCUMENT:" in completed.stdout
    assert "[@P01]" in completed.stdout
    assert "[@P02]" in completed.stdout
    assert "What is in the document?" in completed.stdout
