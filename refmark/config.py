"""Shared configuration and path helpers for refmark."""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent


def load_local_env() -> None:
    """Load a project-local ``.env`` file into the process environment if present."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        cleaned = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), cleaned)


def refmark_home() -> Path:
    """Return the default writable home directory for runtime artifacts."""
    override = os.getenv("REFMARK_HOME")
    if override:
        return Path(override)
    return Path.home() / ".refmark"


def default_mcp_log_path() -> Path:
    """Return the default MCP call log path."""
    return refmark_home() / "logs" / "apply_ref_diff_calls.jsonl"


def default_artifact_dir() -> Path:
    """Return the default directory for generated benchmark artifacts."""
    return PROJECT_ROOT / "bench_results"


def resolve_django_query_fixture() -> Path | None:
    """Resolve the optional Django QuerySet fixture used by large rewrite benches."""
    override = os.getenv("REFMARK_DJANGO_QUERY_PATH")
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override))
    candidates.append(PROJECT_ROOT / "test_sets" / "code" / "django" / "django" / "db" / "models" / "query.py")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
