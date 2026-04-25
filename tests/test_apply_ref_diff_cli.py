import json
import os
import subprocess
import sys
from pathlib import Path

from refmark.core import inject


CLI_PATH = Path(__file__).resolve().parents[1] / "scripts" / "refmark_apply_ref_diff_cli.py"


def _make_marked_python_file(tmp_path: Path) -> Path:
    source = """def greet(name: str) -> str:
    return name.strip()
"""
    marked, _ = inject(
        source,
        ".py",
        marker_format="typed_comment_py",
        chunker="hybrid",
    )
    path = tmp_path / "sample.py"
    path.write_text(marked, encoding="utf-8")
    return path


def test_refmark_apply_ref_diff_cli_from_spec_file(tmp_path: Path):
    path = _make_marked_python_file(tmp_path)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "file_path": str(path),
                "expect_live_markers": True,
                "edits": [
                    {
                        "region_id": "F01",
                        "action": "replace",
                        "new_content": "def greet(name: str) -> str:\n    return name.strip().upper()\n",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CLI_PATH), "--spec-file", str(spec_path)],
        text=True,
        capture_output=True,
        check=False,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 0
    assert payload["ok"] is True
    assert "upper" in path.read_text(encoding="utf-8").lower()


def test_refmark_apply_ref_diff_cli_from_stdin_reports_validation_error(tmp_path: Path):
    result = subprocess.run(
        [sys.executable, str(CLI_PATH), "--stdin"],
        input=json.dumps({"edits": []}),
        text=True,
        capture_output=True,
        check=False,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 1
    assert payload["ok"] is False
    assert any("file_path" in error for error in payload["errors"])


def test_refmark_apply_ref_diff_cli_normalizes_common_new_content_aliases(tmp_path: Path):
    path = _make_marked_python_file(tmp_path)
    spec_path = tmp_path / "spec_alias.json"
    spec_path.write_text(
        json.dumps(
            {
                "file_path": str(path),
                "expect_live_markers": True,
                "edits": [
                    {
                        "region_id": "F01",
                        "replacement": "def greet(name: str) -> str:\n    return name.strip().title()\n",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CLI_PATH), "--spec-file", str(spec_path)],
        text=True,
        capture_output=True,
        check=False,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 0
    assert payload["ok"] is True
    assert ".title()" in path.read_text(encoding="utf-8")


def test_refmark_apply_ref_diff_cli_preserves_expected_text_in_patch_within_spec(tmp_path: Path):
    path = _make_marked_python_file(tmp_path)
    spec_path = tmp_path / "spec_patch.json"
    spec_path.write_text(
        json.dumps(
            {
                "file_path": str(path),
                "expect_live_markers": True,
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
                                    "new_content": "    return name.strip().upper()\n",
                                }
                            ]
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(CLI_PATH), "--spec-file", str(spec_path)],
        text=True,
        capture_output=True,
        check=False,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 0
    assert payload["ok"] is True
    assert ".upper()" in path.read_text(encoding="utf-8")


def test_refmark_apply_ref_diff_cli_writes_shell_telemetry_log(tmp_path: Path, monkeypatch):
    path = _make_marked_python_file(tmp_path)
    spec_path = tmp_path / "spec.json"
    log_path = tmp_path / "shell_log.jsonl"
    spec_path.write_text(
        json.dumps(
            {
                "file_path": str(path),
                "expect_live_markers": True,
                "edits": [
                    {
                        "region_id": "F01",
                        "action": "replace",
                        "new_content": "def greet(name: str) -> str:\n    return name.strip().upper()\n",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    env = {
        **os.environ,
        "REFMARK_SHELL_LOG_PATH": str(log_path),
        "REFMARK_AGENT_TASK_ID": "demo_task",
        "REFMARK_AGENT_VARIANT_ID": "refmark_multidiff",
        "REFMARK_AGENT_NAME": "pytest",
    }

    result = subprocess.run(
        [sys.executable, str(CLI_PATH), "--spec-file", str(spec_path)],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["tool"] == "apply_ref_diff_shell"
    assert record["ok"] is True
    assert record["task_id"] == "demo_task"
    assert record["variant_id"] == "refmark_multidiff"
    assert record["agent_name"] == "pytest"
