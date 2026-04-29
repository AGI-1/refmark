"""Configuration for full evidence-retrieval pipeline runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from refmark.search_index import OPENROUTER_CHAT_URL, approx_tokens


@dataclass(frozen=True)
class ModelTierConfig:
    provider: str = "openrouter"
    model: str = "qwen/qwen-turbo"
    endpoint: str = OPENROUTER_CHAT_URL
    api_key_env: str = "OPENROUTER_API_KEY"
    enabled: bool = True
    concurrency: int = 8
    timeout_seconds: float = 90.0
    retries: int = 2
    temperature: float = 0.2
    max_tokens: int = 512
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EmbeddingConfig:
    name: str = "qwen3-embedding-8b"
    provider: str = "openrouter"
    model: str = "qwen/qwen3-embedding-8b"
    endpoint: str = "https://openrouter.ai/api/v1/embeddings"
    api_key_env: str = "OPENROUTER_API_KEY"
    enabled: bool = False
    batch_size: int = 64
    concurrency: int = 4
    dimensions: int | None = None
    truncate_dimensions: int | None = None
    input_price_per_million: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PipelineLoopConfig:
    max_iterations: int = 3
    target_hit_at_k: float = 0.85
    target_gold_coverage: float = 0.85
    top_k: int = 10
    sample_size: int = 200
    seed: int = 13
    stop_when_not_improving: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PipelineArtifactConfig:
    cache_dir: str = ".refmark/cache"
    output_dir: str = ".refmark/pipeline"
    view_cache: str = ".refmark/cache/retrieval_views.jsonl"
    question_cache: str = ".refmark/cache/questions.jsonl"
    judge_cache: str = ".refmark/cache/judgements.jsonl"
    embedding_cache_dir: str = ".refmark/cache/embeddings"
    overwrite: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PipelineBudgetConfig:
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_estimated_usd: float | None = None
    count_tokens: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FullPipelineConfig:
    schema: str = "refmark.full_pipeline_config.v1"
    corpus_path: str = "docs"
    revision_id: str | None = None
    density: str = "balanced"
    marker_format: str = "typed_bracket"
    min_words: int = 8
    include_embeddings: bool = False
    question_generation: ModelTierConfig = field(default_factory=ModelTierConfig)
    retrieval_views: ModelTierConfig = field(
        default_factory=lambda: ModelTierConfig(
            model="mistralai/mistral-nemo",
            concurrency=8,
            max_tokens=320,
            input_price_per_million=0.01,
            output_price_per_million=0.03,
        )
    )
    judge: ModelTierConfig = field(
        default_factory=lambda: ModelTierConfig(
            model="qwen/qwen3.6-max-preview",
            enabled=False,
            concurrency=4,
            max_tokens=768,
        )
    )
    embeddings: list[EmbeddingConfig] = field(default_factory=list)
    loop: PipelineLoopConfig = field(default_factory=PipelineLoopConfig)
    artifacts: PipelineArtifactConfig = field(default_factory=PipelineArtifactConfig)
    budget: PipelineBudgetConfig = field(default_factory=PipelineBudgetConfig)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["question_generation"] = self.question_generation.to_dict()
        payload["retrieval_views"] = self.retrieval_views.to_dict()
        payload["judge"] = self.judge.to_dict()
        payload["embeddings"] = [item.to_dict() for item in self.embeddings]
        payload["loop"] = self.loop.to_dict()
        payload["artifacts"] = self.artifacts.to_dict()
        payload["budget"] = self.budget.to_dict()
        return payload

    def estimate_generation_cost(self, *, input_text: str, output_text: str = "") -> dict[str, float | None]:
        """Return a rough token/cost estimate for the question-generation tier."""
        input_tokens = approx_tokens(input_text)
        output_tokens = approx_tokens(output_text) if output_text else 0
        input_cost = _token_cost(input_tokens, self.question_generation.input_price_per_million)
        output_cost = _token_cost(output_tokens, self.question_generation.output_price_per_million)
        return {
            "input_tokens": float(input_tokens),
            "output_tokens": float(output_tokens),
            "estimated_usd": None if input_cost is None and output_cost is None else (input_cost or 0.0) + (output_cost or 0.0),
        }


def default_full_pipeline_config() -> FullPipelineConfig:
    return FullPipelineConfig(
        embeddings=[
            EmbeddingConfig(
                name="qwen3-embedding-8b",
                model="qwen/qwen3-embedding-8b",
                input_price_per_million=0.01,
            ),
            EmbeddingConfig(
                name="text-embedding-3-small",
                model="openai/text-embedding-3-small",
                input_price_per_million=0.02,
            ),
        ]
    )


def load_full_pipeline_config(path: str | Path) -> FullPipelineConfig:
    source = Path(path)
    text = source.read_text(encoding="utf-8-sig")
    if source.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        payload = _parse_simple_yaml(text)
    if not isinstance(payload, dict):
        raise ValueError("Pipeline config must be an object.")
    return full_pipeline_config_from_dict(payload)


def full_pipeline_config_from_dict(payload: dict[str, Any]) -> FullPipelineConfig:
    defaults = default_full_pipeline_config().to_dict()
    merged = _deep_merge(defaults, payload)
    return FullPipelineConfig(
        schema=str(merged.get("schema", "refmark.full_pipeline_config.v1")),
        corpus_path=str(merged.get("corpus_path", "docs")),
        revision_id=merged.get("revision_id"),
        density=str(merged.get("density", "balanced")),
        marker_format=str(merged.get("marker_format", "typed_bracket")),
        min_words=int(merged.get("min_words", 8)),
        include_embeddings=bool(merged.get("include_embeddings", False)),
        question_generation=ModelTierConfig(**_model_payload(merged.get("question_generation", {}))),
        retrieval_views=ModelTierConfig(**_model_payload(merged.get("retrieval_views", {}))),
        judge=ModelTierConfig(**_model_payload(merged.get("judge", {}))),
        embeddings=[EmbeddingConfig(**item) for item in merged.get("embeddings", [])],
        loop=PipelineLoopConfig(**merged.get("loop", {})),
        artifacts=PipelineArtifactConfig(**merged.get("artifacts", {})),
        budget=PipelineBudgetConfig(**merged.get("budget", {})),
    )


def write_full_pipeline_config_template(path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_template_yaml(default_full_pipeline_config()), encoding="utf-8")


def _model_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key in ModelTierConfig.__dataclass_fields__}


def _token_cost(tokens: int, price_per_million: float | None) -> float | None:
    if price_per_million is None:
        return None
    return (tokens / 1_000_000.0) * price_per_million


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small nested YAML subset used by Refmark config templates."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any] | list[Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if line.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"List item without list parent: {raw_line!r}")
            item_text = line[2:].strip()
            item: dict[str, Any] = {}
            parent.append(item)
            if item_text:
                key, value = _split_yaml_pair(item_text)
                item[key] = _parse_scalar(value)
            stack.append((indent, item))
            continue
        key, value = _split_yaml_pair(line)
        if isinstance(parent, list):
            raise ValueError(f"Mapping entry under list without '-': {raw_line!r}")
        if value == "":
            container: dict[str, Any] | list[Any]
            container = [] if _next_nonempty_line_is_list(text, raw_line) else {}
            parent[key] = container
            stack.append((indent, container))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _split_yaml_pair(line: str) -> tuple[str, str]:
    if ":" not in line:
        raise ValueError(f"Unsupported config line: {line!r}")
    key, value = line.split(":", 1)
    return key.strip(), value.strip()


def _next_nonempty_line_is_list(text: str, current_line: str) -> bool:
    lines = text.splitlines()
    try:
        index = lines.index(current_line)
    except ValueError:
        return False
    current_indent = len(current_line) - len(current_line.lstrip(" "))
    for line in lines[index + 1 :]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        return indent > current_indent and line.strip().startswith("- ")
    return False


def _parse_scalar(value: str) -> Any:
    cleaned = value.strip().strip('"').strip("'")
    lowered = cleaned.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    try:
        if any(char in cleaned for char in [".", "e", "E"]):
            return float(cleaned)
        return int(cleaned)
    except ValueError:
        return cleaned


def _template_yaml(config: FullPipelineConfig) -> str:
    payload = config.to_dict()
    return """# Refmark full evidence-retrieval pipeline config.
# Runtime search can stay no-infra; remote models are build/eval-time helpers.
schema: {schema}
corpus_path: {corpus_path}
revision_id: null
density: balanced
marker_format: typed_bracket
min_words: 8
include_embeddings: false

question_generation:
  provider: openrouter
  model: qwen/qwen-turbo
  endpoint: {endpoint}
  api_key_env: OPENROUTER_API_KEY
  enabled: true
  concurrency: 8
  timeout_seconds: 90
  retries: 2
  temperature: 0.2
  max_tokens: 512
  input_price_per_million: 0.05
  output_price_per_million: 0.2

retrieval_views:
  provider: openrouter
  model: mistralai/mistral-nemo
  endpoint: {endpoint}
  api_key_env: OPENROUTER_API_KEY
  enabled: true
  concurrency: 8
  timeout_seconds: 90
  retries: 2
  temperature: 0.2
  max_tokens: 320
  input_price_per_million: 0.01
  output_price_per_million: 0.03

judge:
  provider: openrouter
  model: qwen/qwen3.6-max-preview
  endpoint: {endpoint}
  api_key_env: OPENROUTER_API_KEY
  enabled: false
  concurrency: 4
  timeout_seconds: 120
  retries: 1
  temperature: 0
  max_tokens: 768

embeddings:
  - name: qwen3-embedding-8b
    provider: openrouter
    model: qwen/qwen3-embedding-8b
    endpoint: https://openrouter.ai/api/v1/embeddings
    api_key_env: OPENROUTER_API_KEY
    enabled: false
    batch_size: 64
    concurrency: 4
    input_price_per_million: 0.01
  - name: text-embedding-3-small
    provider: openrouter
    model: openai/text-embedding-3-small
    endpoint: https://openrouter.ai/api/v1/embeddings
    api_key_env: OPENROUTER_API_KEY
    enabled: false
    batch_size: 64
    concurrency: 4
    input_price_per_million: 0.02

loop:
  max_iterations: 3
  target_hit_at_k: 0.85
  target_gold_coverage: 0.85
  top_k: 10
  sample_size: 200
  seed: 13
  stop_when_not_improving: true

artifacts:
  cache_dir: .refmark/cache
  output_dir: .refmark/pipeline
  view_cache: .refmark/cache/retrieval_views.jsonl
  question_cache: .refmark/cache/questions.jsonl
  judge_cache: .refmark/cache/judgements.jsonl
  embedding_cache_dir: .refmark/cache/embeddings
  overwrite: false

budget:
  max_input_tokens: null
  max_output_tokens: null
  max_estimated_usd: null
  count_tokens: true
""".format(schema=payload["schema"], corpus_path=payload["corpus_path"], endpoint=OPENROUTER_CHAT_URL)

