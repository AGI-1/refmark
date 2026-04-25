from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from refmark.edit import (
    _apply_edits_to_lines,
    _detect_premarked,
    _remove_rfm_header,
    _select_chunker,
    _select_marker_format,
    apply_ref_diff,
)
from refmark.regions import _parse_blocks_with_mode, validate_syntax
from refmark.core import inject, strip
from refmark.languages import get_language_spec


@dataclass
class ShadowSessionEntry:
    clean_text: str
    shadow_text: str
    marker_format: str
    chunker: str
    marker_count: int


def supports_shadow_refmarks(path: Path) -> bool:
    return get_language_spec(path.suffix) is not None


def marker_count_from_text(marked_text: str, marker_format: str) -> int:
    return len(_parse_blocks_with_mode(marked_text, marker_format, line_mode="marked"))


def shadow_session_key(path: Path) -> str:
    resolved = str(path.resolve()).encode("utf-8")
    return hashlib.sha256(resolved).hexdigest()[:16]


def shadow_state_path(path: Path, state_dir: Path) -> Path:
    return state_dir / f"{shadow_session_key(path)}.json"


def build_shadow_entry(path: Path, clean_text: str) -> ShadowSessionEntry:
    marker_format = _select_marker_format(path)
    chunker = _select_chunker(path)
    shadow_text, marker_count = inject(
        clean_text,
        path.suffix,
        marker_format=marker_format,
        chunker=chunker,
    )
    return ShadowSessionEntry(
        clean_text=clean_text,
        shadow_text=shadow_text,
        marker_format=marker_format,
        chunker=chunker,
        marker_count=marker_count,
    )


def _serialize_entry(entry: ShadowSessionEntry) -> dict[str, Any]:
    return {
        "clean_text": entry.clean_text,
        "shadow_text": entry.shadow_text,
        "marker_format": entry.marker_format,
        "chunker": entry.chunker,
        "marker_count": entry.marker_count,
    }


def _deserialize_entry(payload: dict[str, Any]) -> ShadowSessionEntry:
    return ShadowSessionEntry(
        clean_text=str(payload["clean_text"]),
        shadow_text=str(payload["shadow_text"]),
        marker_format=str(payload["marker_format"]),
        chunker=str(payload["chunker"]),
        marker_count=int(payload["marker_count"]),
    )


def load_or_build_view_state(path: Path, state_dir: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    if not supports_shadow_refmarks(path):
        return {
            "supported": False,
            "marker_format": None,
            "chunker": None,
            "source": source,
            "view_text": source,
            "marker_count": 0,
            "namespace_mode": "raw",
            "shadow_persistent": False,
            "session_reset": False,
        }

    marker_format = _select_marker_format(path)
    chunker = _select_chunker(path)
    if _detect_premarked(source, marker_format=marker_format):
        return {
            "supported": True,
            "marker_format": marker_format,
            "chunker": chunker,
            "source": source,
            "view_text": source,
            "marker_count": marker_count_from_text(source, marker_format),
            "namespace_mode": "live",
            "shadow_persistent": False,
            "session_reset": False,
        }

    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = shadow_state_path(path, state_dir)
    session_reset = False
    if state_path.exists():
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        entry = _deserialize_entry(payload)
        if entry.clean_text != source:
            entry = build_shadow_entry(path, source)
            state_path.write_text(json.dumps(_serialize_entry(entry), indent=2), encoding="utf-8")
            session_reset = True
    else:
        entry = build_shadow_entry(path, source)
        state_path.write_text(json.dumps(_serialize_entry(entry), indent=2), encoding="utf-8")
        session_reset = True

    return {
        "supported": True,
        "marker_format": entry.marker_format,
        "chunker": entry.chunker,
        "source": source,
        "view_text": entry.shadow_text,
        "marker_count": entry.marker_count,
        "namespace_mode": "shadow",
        "shadow_persistent": True,
        "session_reset": session_reset,
        "state_path": str(state_path),
    }


def apply_shadow_session_ref_diff(path: Path, edits: list[dict[str, Any]], state_dir: Path) -> dict[str, Any]:
    view_state = load_or_build_view_state(path, state_dir)
    if not view_state["supported"]:
        return {
            "ok": False,
            "file_path": str(path),
            "applied_edits": 0,
            "created_regions": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_seconds": 0.0,
            "syntax_ok": False,
            "errors": [f"Persistent shadow mode is not supported for {path.suffix or 'this file type'}."],
        }

    if view_state["namespace_mode"] != "shadow":
        return apply_ref_diff(path, edits, expect_live_markers=None)

    blocks = _parse_blocks_with_mode(
        view_state["view_text"],
        view_state["marker_format"],
        line_mode="marked",
    )
    next_shadow, apply_errors, created_regions = _apply_edits_to_lines(
        view_state["view_text"].splitlines(keepends=True),
        blocks,
        edits,
        line_mode="marked",
        marker_format=view_state["marker_format"],
        file_ext=path.suffix,
        allow_region_creation=True,
    )
    next_code = strip(next_shadow, path.suffix, marker_format=view_state["marker_format"])
    next_code = _remove_rfm_header(next_code)
    if next_code and not next_code.endswith("\n"):
        next_code += "\n"

    syntax_ok = validate_syntax(next_code, path.suffix)
    if not syntax_ok:
        apply_errors.append(f"Edited file is not syntactically valid for {path.suffix or 'this language'}.")

    if not apply_errors:
        path.write_text(next_code, encoding="utf-8")
        entry = ShadowSessionEntry(
            clean_text=next_code,
            shadow_text=next_shadow,
            marker_format=view_state["marker_format"],
            chunker=view_state["chunker"],
            marker_count=marker_count_from_text(next_shadow, view_state["marker_format"]),
        )
        state_path = shadow_state_path(path, state_dir)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(_serialize_entry(entry), indent=2), encoding="utf-8")

    return {
        "ok": not apply_errors,
        "file_path": str(path),
        "applied_edits": len([edit for edit in edits if isinstance(edit, dict)]),
        "created_regions": created_regions,
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_seconds": 0.0,
        "syntax_ok": syntax_ok,
        "errors": apply_errors,
        "namespace_mode": "shadow",
        "shadow_persistent": True,
        "state_path": str(shadow_state_path(path, state_dir)),
    }
