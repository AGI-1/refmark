import json
import subprocess
import sys
from pathlib import Path

from refmark.search_index import PortableBM25Index, RetrievalView, SearchRegion

from examples.portable_search_index.compare_navigation_search import _read_questions
from examples.portable_search_index.improve_fastapi_questions import build_adaptation_plan, build_review_input


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


def test_fastapi_review_input_includes_metadata_and_smells():
    index = PortableBM25Index(
        [
            SearchRegion(
                doc_id="docs",
                region_id="P01",
                text="Token rotation is required every 90 days for production credentials and administrator access.",
                hash="h1",
                source_path="docs/security.md",
                ordinal=1,
                prev_region_id=None,
                next_region_id="P02",
                view=RetrievalView(summary="Token rotation", questions=[], keywords=[]),
            ),
            SearchRegion(
                doc_id="docs",
                region_id="P02",
                text="Token rotation is required every 90 days for production credentials and administrator access.",
                hash="h2",
                source_path="docs/security-copy.md",
                ordinal=2,
                prev_region_id="P01",
                next_region_id=None,
                view=RetrievalView(summary="Token rotation copy", questions=[], keywords=[]),
            ),
        ]
    )
    qrows = [
        {
            "stable_ref": "docs:P01",
            "gold_refs": ["docs:P01"],
            "section_title": "Token rotation",
            "variant": "concern",
            "query": "How often should I rotate production tokens?",
        }
    ]
    eval_payload = {
        "methods": {
            "bm25": {
                "results": [
                    {
                        "stable_ref": "docs:P01",
                        "variant": "concern",
                        "query": "How often should I rotate production tokens?",
                        "hit_at_1": False,
                        "hit_at_k": True,
                        "top_refs": ["docs:P02", "docs:P01"],
                    }
                ]
            }
        }
    }
    shadow = {
        "docs:P02": {
            "doc2query": ["credential replacement schedule"],
            "keywords": ["tokens", "credentials"],
            "disambiguators": ["copy section"],
            "confusions": [{"with": "docs:P01"}],
            "source_hashes": {"docs:P02": "h2"},
            "provenance": {"source": "test"},
        }
    }

    payload = build_review_input("docs:P01", qrows, eval_payload, index, shadow_metadata=shadow)

    assert "docs:P02" in payload["shadow_metadata"]
    assert payload["shadow_metadata"]["docs:P02"]["doc2query"] == ["credential replacement schedule"]
    assert payload["data_smells"]["exact_duplicate_groups"]
    assert payload["data_smells"]["duplicate_candidates"]


def test_fastapi_adaptation_plan_can_record_data_smells():
    plan = build_adaptation_plan(
        [
            {
                "stable_ref": "docs:P01",
                "items": [
                    {
                        "variant": "concern",
                        "diagnosis": "metadata_noise",
                        "reason": "Competing ref alias appears to attract this query.",
                    }
                ],
            }
        ]
    )

    assert plan[0]["action"] == "record_data_smell"
    assert plan[0]["adaptation_type"] == "data_smell"


def test_compare_navigation_reads_product_eval_jsonl(tmp_path):
    path = tmp_path / "eval_questions.jsonl"
    path.write_text(
        json.dumps(
            {
                "query": "How do I configure CORS?",
                "gold_refs": ["docs_cors:P02"],
                "source_hashes": {"docs_cors:P02": "abc123"},
                "metadata": {
                    "source": "refmark-full-pipeline",
                    "query_style": "concern",
                    "gold_mode": "single",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    questions = _read_questions(path)

    assert len(questions) == 1
    assert questions[0].doc_id == "docs_cors"
    assert questions[0].region_id == "P02"
    assert questions[0].stable_ref == "docs_cors:P02"
    assert questions[0].hash == "abc123"
    assert questions[0].query_style == "concern"


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
