"""Configuration presets for document-oriented Refmark workflows."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WorkflowConfig:
    density: str = "balanced"
    marker_style: str = "default"
    marker_format: str = "typed_bracket"
    chunker: str = "paragraph"
    lines_per_chunk: int | None = None
    tokens_per_chunk: int | None = None
    include_headings: bool = True
    min_words: int = 0
    expand_before: int = 0
    expand_after: int = 1
    coverage_threshold: float = 0.4
    numeric_checks: bool = True
    top_k: int = 3

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_DENSITY_PRESETS: dict[str, dict[str, Any]] = {
    "dense": {"chunker": "line", "lines_per_chunk": 1, "tokens_per_chunk": None},
    "balanced": {"chunker": "paragraph", "lines_per_chunk": None, "tokens_per_chunk": None},
    "coarse": {"chunker": "token", "lines_per_chunk": None, "tokens_per_chunk": 180},
    "code": {"chunker": "hybrid"},
}

_MARKER_STYLE_PRESETS = {
    "default": "typed_bracket",
    "machine": "typed_bracket",
    "explicit": "typed_explicit",
    "compact": "typed_compact",
    "xml": "typed_xml",
}


def resolve_workflow_config(
    config: WorkflowConfig | None = None,
    **overrides: Any,
) -> WorkflowConfig:
    """Resolve density and marker-style presets into concrete settings."""
    base = config or WorkflowConfig()
    values = base.to_dict()
    values.update({key: value for key, value in overrides.items() if value is not None})

    density = str(values.get("density") or "balanced")
    if density not in _DENSITY_PRESETS:
        raise ValueError(f"Unknown density preset '{density}'.")
    marker_style = str(values.get("marker_style") or "default")
    if marker_style not in _MARKER_STYLE_PRESETS:
        raise ValueError(f"Unknown marker style '{marker_style}'.")

    preset = _DENSITY_PRESETS[density]
    if "chunker" not in overrides or overrides.get("chunker") is None:
        values["chunker"] = preset["chunker"]
    if "lines_per_chunk" not in overrides or overrides.get("lines_per_chunk") is None:
        values["lines_per_chunk"] = preset.get("lines_per_chunk")
    if "tokens_per_chunk" not in overrides or overrides.get("tokens_per_chunk") is None:
        values["tokens_per_chunk"] = preset.get("tokens_per_chunk")
    if "marker_format" not in overrides or overrides.get("marker_format") is None:
        values["marker_format"] = _MARKER_STYLE_PRESETS[marker_style]
    return WorkflowConfig(**values)


def load_workflow_config(path: str | Path) -> WorkflowConfig:
    """Load a small JSON or flat YAML workflow config file."""
    source = Path(path)
    text = source.read_text(encoding="utf-8-sig")
    if source.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        payload = _parse_flat_yaml(text)
    if not isinstance(payload, dict):
        raise ValueError("Workflow config must be an object.")
    return resolve_workflow_config(**payload)


def merge_workflow_config(config: WorkflowConfig, **overrides: Any) -> WorkflowConfig:
    """Return a config with explicit overrides applied and presets re-resolved."""
    updated = replace(config, **{key: value for key, value in overrides.items() if value is not None})
    return resolve_workflow_config(updated)


def _parse_flat_yaml(text: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Unsupported config line: {raw_line!r}")
        key, value = line.split(":", 1)
        payload[key.strip()] = _parse_scalar(value.strip())
    return payload


def _parse_scalar(value: str) -> Any:
    cleaned = value.strip().strip('"').strip("'")
    lowered = cleaned.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        if "." in cleaned:
            return float(cleaned)
        return int(cleaned)
    except ValueError:
        return cleaned
