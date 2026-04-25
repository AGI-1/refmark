import json
import subprocess
import sys
from pathlib import Path


PUBLISH_ROOT = Path(__file__).resolve().parents[1]


def test_citation_qa_example_runs():
    result = subprocess.run(
        [sys.executable, "examples/citation_qa/run_eval.py"],
        cwd=PUBLISH_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    scores = json.loads((PUBLISH_ROOT / "examples" / "citation_qa" / "output" / "scores.json").read_text())
    assert scores["marker_count"] == 4
    assert scores["means"]["exact_match"] == 0.667
    assert scores["means"]["cover"] == 1.0
    assert (PUBLISH_ROOT / "examples" / "citation_qa" / "output" / "cited_regions.html").exists()


def test_multidiff_example_runs():
    result = subprocess.run(
        [sys.executable, "examples/multidiff_demo/run.py"],
        cwd=PUBLISH_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((PUBLISH_ROOT / "examples" / "multidiff_demo" / "output" / "result.json").read_text())
    assert payload["good_edit_ok"] is True
    assert payload["stale_edit_ok"] is False
    assert payload["stale_left_file_unchanged"] is True


def test_data_smells_example_runs():
    result = subprocess.run(
        [sys.executable, "examples/data_smells/run.py"],
        cwd=PUBLISH_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((PUBLISH_ROOT / "examples" / "data_smells" / "output" / "data_smells_report.json").read_text())
    wrong = payload["models"]["wrong_place_model"]["summary"]
    sloppy = payload["models"]["sloppy_boundary_model"]["summary"]
    assert wrong["wrong_location_rate"] == 1.0
    assert sloppy["wrong_location_rate"] == 0.0
    assert sloppy["exact_match"] == 0.0


def test_judge_free_rewards_example_runs():
    result = subprocess.run(
        [sys.executable, "examples/judge_free_rewards/run.py"],
        cwd=PUBLISH_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((PUBLISH_ROOT / "examples" / "judge_free_rewards" / "output" / "judge_free_rewards.json").read_text())
    first = payload["rows"][0]["candidates"]
    assert first["exact"]["reward"] == 1.0
    assert first["exact"]["reward"] > first["overcite_neighbor"]["reward"] > first["wrong_location"]["reward"]
