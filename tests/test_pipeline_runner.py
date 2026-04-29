import json

from refmark.pipeline_config import load_full_pipeline_config
from refmark.pipeline_runner import run_full_pipeline


def test_run_full_pipeline_easy_mode_is_idempotent(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "security.md").write_text(
        "# Security\n\n"
        "Rotate API tokens every ninety days after replacement credentials are deployed.\n\n"
        "Audit logs are retained for one hundred eighty days by default.\n",
        encoding="utf-8",
    )
    output = tmp_path / "out"
    cache = tmp_path / "cache"
    config_path = tmp_path / "pipeline.json"
    config_path.write_text(
        json.dumps(
            {
                "corpus_path": str(docs),
                "question_generation": {"enabled": False},
                "retrieval_views": {"enabled": False},
                "loop": {"sample_size": 2, "top_k": 5},
                "artifacts": {
                    "output_dir": str(output),
                    "cache_dir": str(cache),
                    "question_cache": str(cache / "questions.jsonl"),
                    "view_cache": str(cache / "views.jsonl"),
                    "overwrite": False,
                },
            }
        ),
        encoding="utf-8",
    )

    first = run_full_pipeline(load_full_pipeline_config(config_path))
    second = run_full_pipeline(config_path)

    assert first.stats["regions"] == 2
    assert first.stats["questions"] == 6
    assert first.stats["eval"]["hit_at_k"] == 1.0
    assert (output / "corpus.refmark.jsonl").exists()
    assert (output / "sections.json").exists()
    assert (output / "docs.index.json").exists()
    assert (output / "docs.browser.json").exists()
    assert "Reused manifest" in "\n".join(second.notes)
    assert "Reused eval questions" in "\n".join(second.notes)
