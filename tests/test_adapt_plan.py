import json
import subprocess
import sys

from refmark.adapt_plan import build_adaptation_plan


def test_adaptation_plan_maps_smells_to_reviewable_actions():
    report = {
        "schema": "refmark.data_smells.v1",
        "summary": {"run_fingerprint": "run1", "corpus_fingerprint": "corp1", "smell_count": 3},
        "smells": [
            {
                "type": "hard_ref",
                "severity": "high",
                "message": "Region missed all queries.",
                "refs": ["docs:P02"],
                "evidence": {"sample_misses": [{"query": "how do refunds work?"}]},
            },
            {
                "type": "confusion_pair",
                "severity": "medium",
                "message": "Gold confused with another ref.",
                "refs": ["docs:P02", "docs:P03"],
                "evidence": {"sample_queries": ["refund policy"]},
            },
            {
                "type": "query_magnet",
                "severity": "medium",
                "message": "Release notes attract weak hits.",
                "refs": ["docs:P03"],
            },
        ],
    }

    plan = build_adaptation_plan(report).to_dict()

    assert plan["schema"] == "refmark.adaptation_plan.v1"
    assert plan["summary"]["source_run_fingerprint"] == "run1"
    assert plan["summary"]["by_adaptation_type"]["retrieval_metadata"] == 1
    assert plan["summary"]["by_adaptation_type"]["confusion_mapping"] == 1
    assert plan["summary"]["by_adaptation_type"]["corpus_roles"] == 1
    assert plan["actions"][0]["priority"] == "high"
    assert plan["actions"][0]["review_required"] is True


def test_adapt_plan_cli_writes_json(tmp_path):
    smell_report = tmp_path / "smells.json"
    output = tmp_path / "plan.json"
    smell_report.write_text(
        json.dumps(
            {
                "schema": "refmark.data_smells.v1",
                "summary": {"smell_count": 1},
                "smells": [
                    {
                        "type": "stale_label",
                        "severity": "medium",
                        "message": "Changed source.",
                        "refs": ["docs:P01"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "-m", "refmark.cli", "adapt-plan", str(smell_report), "-o", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "refmark.adaptation_plan.v1"
    assert payload["actions"][0]["action"] == "review_or_refresh_stale_eval_label"
