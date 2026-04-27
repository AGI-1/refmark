"""Small provenance helpers for self-checking Refmark evaluation artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


PROVENANCE_SCHEMA = "refmark.eval_provenance.v1"


@dataclass(frozen=True)
class ProvenanceMismatch:
    path: str
    expected: Any
    actual: Any

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "expected": self.expected, "actual": self.actual}


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    """Return a deterministic fingerprint for an artifact on disk."""

    file_path = Path(path)
    data = file_path.read_bytes()
    return {
        "path": str(file_path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def stable_digest(payload: Any) -> str:
    """Hash JSON-compatible settings without timestamp or formatting noise."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_eval_provenance(
    *,
    index_path: str | Path,
    examples_path: str | Path,
    settings: dict[str, Any],
    index_metadata: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe exactly which artifacts and knobs produced an eval run."""

    payload = {
        "schema": PROVENANCE_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "index": file_fingerprint(index_path),
            "examples": file_fingerprint(examples_path),
        },
        "settings": settings,
        "settings_sha256": stable_digest(settings),
    }
    if index_metadata:
        payload["index_metadata"] = index_metadata
    if extra:
        payload["extra"] = extra
    return payload


def validate_provenance(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    fields: Iterable[str] = (
        "schema",
        "artifacts.index.sha256",
        "artifacts.examples.sha256",
        "settings_sha256",
    ),
) -> dict[str, Any]:
    """Compare provenance payloads and return self-check status."""

    mismatches = []
    for field in fields:
        expected_value = _lookup(expected, field)
        actual_value = _lookup(actual, field)
        if expected_value != actual_value:
            mismatches.append(ProvenanceMismatch(field, expected_value, actual_value).to_dict())
    return {"ok": not mismatches, "mismatches": mismatches}


def _lookup(payload: dict[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value
