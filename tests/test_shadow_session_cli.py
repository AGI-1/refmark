import json
import subprocess
import sys
from pathlib import Path


CLI_PATH = Path(__file__).resolve().parents[1] / "scripts" / "refmark_shadow_session_cli.py"


def test_shadow_session_cli_persists_regions_and_apply(tmp_path: Path):
    path = tmp_path / "sample.py"
    state_dir = tmp_path / ".shadow_state"
    path.write_text(
        "def greet(name: str) -> str:\n    return name.strip()\n\n\ndef loud(name: str) -> str:\n    return greet(name).upper()\n",
        encoding="utf-8",
    )

    first = subprocess.run(
        [sys.executable, str(CLI_PATH), "list", "--file-path", str(path), "--state-dir", str(state_dir)],
        text=True,
        capture_output=True,
        check=False,
    )
    second = subprocess.run(
        [sys.executable, str(CLI_PATH), "list", "--file-path", str(path), "--state-dir", str(state_dir)],
        text=True,
        capture_output=True,
        check=False,
    )
    first_payload = json.loads(first.stdout)
    second_payload = json.loads(second.stdout)
    first_ids = [region["region_id"] for region in first_payload["regions"]]
    second_ids = [region["region_id"] for region in second_payload["regions"]]
    assert first_ids == second_ids

    greet_region = next(
        region for region in first_payload["regions"] if "greet(name: str)" in " ".join(region["preview_lines"])
    )["region_id"]
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "file_path": str(path),
                "edits": [
                    {
                        "region_id": greet_region,
                        "action": "replace",
                        "new_content": "def greet(name: str) -> str:\n    return name.strip().title()\n",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    applied = subprocess.run(
        [sys.executable, str(CLI_PATH), "apply", "--state-dir", str(state_dir), "--spec-file", str(spec_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert applied.returncode == 0
    third = subprocess.run(
        [sys.executable, str(CLI_PATH), "list", "--file-path", str(path), "--state-dir", str(state_dir)],
        text=True,
        capture_output=True,
        check=False,
    )
    third_payload = json.loads(third.stdout)
    third_ids = [region["region_id"] for region in third_payload["regions"]]
    assert third_ids == first_ids
    assert "title()" in path.read_text(encoding="utf-8")
