import subprocess
import sys
import json

from refmark.smoke import run_smoke as run_refmark_smoke
from refmark_train.smoke import run_smoke as run_train_smoke


def test_refmark_public_smoke_passes():
    result = run_refmark_smoke()

    assert result["ok"] is True
    assert result["citation_exact_match"] == 1.0
    assert result["edit_ok"] is True


def test_refmark_cli_smoke_passes():
    result = subprocess.run(
        [sys.executable, "-m", "refmark.cli", "smoke", "--json"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert '"ok": true' in result.stdout


def test_refmark_train_smoke_passes():
    result = run_train_smoke()

    assert result["ok"] is True
    assert result["anchors"] == 12
    assert result["train_examples"] > 0


def test_main_cli_default_inject_uses_highlightable_live_refs(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text(
        "def normalize_name(name: str) -> str:\n"
        "    return name.strip()\n\n\n"
        "def greet(name: str) -> str:\n"
        "    return f\"Hello, {normalize_name(name)}\"\n",
        encoding="utf-8",
    )

    inject_result = subprocess.run(
        [sys.executable, "-m", "refmark.cli", "inject", str(path)],
        text=True,
        capture_output=True,
        check=False,
    )
    highlight_result = subprocess.run(
        [sys.executable, "-m", "refmark.cli", "highlight", str(path), "--refs", "F01-F02", "--context-lines", "0"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert inject_result.returncode == 0, inject_result.stderr
    assert "# [@F01]" in path.read_text(encoding="utf-8")
    assert highlight_result.returncode == 0, highlight_result.stderr
    assert "refs: F01, F02" in highlight_result.stdout


def test_main_cli_apply_ref_diff_accepts_utf8_bom_json(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text(
        "# [@F01]\n"
        "def normalize_name(name: str) -> str:\n"
        "    return name.strip()\n",
        encoding="utf-8",
    )
    edits_path = tmp_path / "edits.json"
    edits_path.write_text(
        json.dumps(
            {
                "edits": [
                    {
                        "region_id": "F01",
                        "action": "patch_within",
                        "patch_format": "line_edits",
                        "patch": {
                            "edits": [
                                {
                                    "start_line": 2,
                                    "end_line": 2,
                                    "expected_text": "    return name.strip()\n",
                                    "new_content": "    return name.strip().title()\n",
                                }
                            ]
                        },
                    }
                ]
            }
        ),
        encoding="utf-8-sig",
    )

    result = subprocess.run(
        [sys.executable, "-m", "refmark.cli", "apply-ref-diff", str(path), "--edits-file", str(edits_path)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert ".title()" in path.read_text(encoding="utf-8")
