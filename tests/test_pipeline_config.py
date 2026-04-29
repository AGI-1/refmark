import json

from refmark.pipeline_config import load_full_pipeline_config, write_full_pipeline_config_template


def test_write_and_load_full_pipeline_config_template(tmp_path):
    path = tmp_path / "refmark_pipeline.yaml"

    write_full_pipeline_config_template(path)
    config = load_full_pipeline_config(path)

    assert config.schema == "refmark.full_pipeline_config.v1"
    assert config.question_generation.provider == "openrouter"
    assert config.retrieval_views.model == "mistralai/mistral-nemo"
    assert config.embeddings[0].name == "qwen3-embedding-8b"
    assert config.loop.max_iterations == 3
    assert config.artifacts.question_cache.endswith("questions.jsonl")


def test_load_full_pipeline_config_overrides_nested_values(tmp_path):
    path = tmp_path / "refmark_pipeline.json"
    path.write_text(
        json.dumps(
            {
                "corpus_path": "docs/api",
                "include_embeddings": True,
                "question_generation": {"model": "qwen/qwen-turbo", "concurrency": 12},
                "judge": {"enabled": True, "model": "moonshotai/kimi-k2.6"},
                "embeddings": [
                    {
                        "name": "custom",
                        "provider": "openrouter",
                        "model": "openai/text-embedding-3-small",
                        "enabled": True,
                    }
                ],
                "loop": {"max_iterations": 5, "target_hit_at_k": 0.9},
            }
        ),
        encoding="utf-8",
    )

    config = load_full_pipeline_config(path)

    assert config.corpus_path == "docs/api"
    assert config.include_embeddings is True
    assert config.question_generation.concurrency == 12
    assert config.judge.enabled is True
    assert config.judge.model == "moonshotai/kimi-k2.6"
    assert [embedding.name for embedding in config.embeddings if embedding.enabled] == ["custom"]
    assert config.loop.max_iterations == 5
    assert config.loop.target_hit_at_k == 0.9
