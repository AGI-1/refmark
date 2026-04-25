"""MCP server exposing refmark edit primitives."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from refmark.edit import (
    _apply_edits_to_lines,
    _detect_premarked,
    _remove_rfm_header,
    _select_chunker,
    _select_marker_format,
    apply_ref_diff,
)
from refmark.regions import _parse_blocks_with_mode
from refmark.core import inject, strip
from refmark.regions import validate_syntax
from refmark.languages import get_language_spec
from refmark.config import default_mcp_log_path


class RefmarkEdit(BaseModel):
    action: Literal["replace", "delete", "insert_before", "patch_within"] = Field(
        default="replace",
        description="replace swaps in new_content, delete removes the addressed body while preserving boundary refs, insert_before inserts new_content before anchor_ref, patch_within applies a bounded patch inside the addressed region.",
    )
    region_id: str | None = Field(
        default=None,
        description="Semantic region identifier such as M02 or C03. Mutually exclusive with start_ref.",
    )
    start_ref: str | None = Field(
        default=None,
        description="Boundary ref where the replacement range starts.",
    )
    end_ref: str | None = Field(
        default=None,
        description="Optional boundary ref where the replacement range stops just before it.",
    )
    anchor_ref: str | None = Field(
        default=None,
        description="Anchor ref used by insert_before. Use EOF to append at the end of the file.",
    )
    create_region: bool | None = Field(
        default=None,
        description="When true for insert_before on live-marked files or persistent shadow sessions, create a fresh stable region marker and return it in created_regions.",
    )
    patch_format: Literal["line_edits", "search_replace", "unified_diff"] | None = Field(
        default=None,
        description="Required for patch_within. line_edits applies relative line replacements inside the region body, search_replace applies exact snippet replacements inside the region body, unified_diff applies a unified diff only to the region body.",
    )
    patch: dict[str, Any] | list[dict[str, Any]] | str | None = Field(
        default=None,
        description="Required for patch_within. For line_edits/search_replace use an object like {\"edits\":[...]}; for unified_diff use either a raw diff string or {\"patch\":\"...\"}. For line_edits, each entry may include optional expected_text so the tool can reject a misaligned relative line edit before it writes broken code.",
    )
    new_content: str | None = Field(
        default=None,
        description="Replacement text for replace actions. Empty string means empty body, not ref deletion.",
    )


def _default_log_path() -> Path:
    return default_mcp_log_path()


def _log_path() -> Path:
    override = os.getenv("REFMARK_MCP_LOG_PATH")
    if override:
        return Path(override)
    return _default_log_path()


def _append_call_log(payload: dict[str, Any]) -> None:
    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _shadow_read_defaults() -> tuple[int, int]:
    max_chars = int(os.getenv("REFMARK_MCP_MAX_READ_CHARS", "12000"))
    max_lines = int(os.getenv("REFMARK_MCP_MAX_READ_LINES", "400"))
    return max_chars, max_lines


def _supports_shadow_refmarks(path: Path) -> bool:
    return get_language_spec(path.suffix) is not None


@dataclass
class ShadowSessionEntry:
    clean_text: str
    shadow_text: str
    marker_format: str
    chunker: str
    marker_count: int


_SHADOW_SESSIONS: dict[str, ShadowSessionEntry] = {}


def _marker_count_from_text(marked_text: str, marker_format: str) -> int:
    return len(_parse_blocks_with_mode(marked_text, marker_format, line_mode="marked"))


def _build_shadow_entry(path: Path, clean_text: str) -> ShadowSessionEntry:
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


def _shadow_session_key(path: Path) -> str:
    return str(path.resolve())


def _get_view_state(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    if not _supports_shadow_refmarks(path):
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
            "marker_count": _marker_count_from_text(source, marker_format),
            "namespace_mode": "live",
            "shadow_persistent": False,
            "session_reset": False,
        }

    key = _shadow_session_key(path)
    entry = _SHADOW_SESSIONS.get(key)
    session_reset = False
    if entry is None or entry.clean_text != source:
        entry = _build_shadow_entry(path, source)
        _SHADOW_SESSIONS[key] = entry
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
    }


def _slice_text_window(
    text: str,
    *,
    start_line: int | None,
    end_line: int | None,
    max_chars: int,
    max_lines: int,
) -> dict[str, Any]:
    lines = text.splitlines()
    total_lines = len(lines)
    total_chars = len(text)
    requested_start = max(1, start_line or 1)
    requested_end = min(total_lines, end_line or total_lines)
    if requested_end < requested_start:
        requested_end = requested_start - 1

    selected = lines[requested_start - 1:requested_end]
    truncated_by_lines = False
    truncated_by_chars = False

    if len(selected) > max_lines:
        selected = selected[:max_lines]
        truncated_by_lines = True

    body = "\n".join(selected)
    if total_lines and requested_end >= requested_start and text.endswith("\n"):
        body += "\n"

    if len(body) > max_chars:
        body = body[:max_chars]
        truncated_by_chars = True

    returned_end = requested_start + max(0, len(selected) - 1)
    return {
        "content": body,
        "total_lines": total_lines,
        "total_chars": total_chars,
        "returned_start_line": requested_start,
        "returned_end_line": returned_end if selected else requested_start - 1,
        "truncated": truncated_by_lines or truncated_by_chars,
        "truncated_by_lines": truncated_by_lines,
        "truncated_by_chars": truncated_by_chars,
    }


def _region_preview(text: str, *, limit: int = 160) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "..."


def _preview_lines(text: str, *, max_lines: int, max_chars_per_line: int = 200) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.rstrip()
        if not stripped.strip():
            continue
        if len(stripped) > max_chars_per_line:
            stripped = stripped[:max_chars_per_line] + "..."
        lines.append(stripped)
        if len(lines) >= max_lines:
            break
    return lines


def _preview_lines_from_shadow(
    shadow_text: str,
    *,
    start_line: int,
    end_line: int,
    max_lines: int,
    max_chars_per_line: int = 200,
) -> list[str]:
    if max_lines <= 0:
        return []
    shadow_lines = shadow_text.splitlines()
    lines: list[str] = []

    def _collect(candidate_lines: list[str]) -> None:
        nonlocal lines
        for raw_line in candidate_lines:
            stripped = raw_line.rstrip()
            if not stripped.strip():
                continue
            if stripped.lstrip().startswith(("# [@", "// [@", "[@")):
                if lines:
                    break
                continue
            if len(stripped) > max_chars_per_line:
                stripped = stripped[:max_chars_per_line] + "..."
            lines.append(stripped)
            if len(lines) >= max_lines:
                break

    body_lines = shadow_lines[start_line - 1:end_line]
    _collect(body_lines)
    if lines:
        return lines

    lookahead_lines = shadow_lines[end_line:]
    _collect(lookahead_lines)
    if lines:
        return lines[:max_lines]

    for raw_line in body_lines:
        stripped = raw_line.rstrip()
        if stripped.strip():
            lines.append(stripped[:max_chars_per_line])
            break
    return lines


def _normalize_edit_payload(edit: RefmarkEdit | dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(edit, RefmarkEdit):
        return edit.model_dump(exclude_none=True)
    if isinstance(edit, dict):
        return RefmarkEdit.model_validate(edit).model_dump(exclude_none=True)
    if isinstance(edit, str):
        try:
            decoded = json.loads(edit)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Edit string must be valid JSON: {exc.msg}") from exc
        if not isinstance(decoded, dict):
            raise ValueError("Edit string JSON must decode to an object.")
        return RefmarkEdit.model_validate(decoded).model_dump(exclude_none=True)
    raise TypeError(f"Unsupported edit payload type: {type(edit).__name__}")


server = FastMCP(
    name="Refmark Apply Ref Diff",
    instructions=(
        "Use apply_ref_diff to apply multiple refmark-scoped edits to Python or TypeScript files. "
        "Edits may target a precomputed semantic region via region_id or an explicit boundary range via "
        "start_ref/end_ref. insert_before adds content before anchor_ref or EOF. "
        "patch_within applies a bounded patch only inside the resolved region body. "
        "delete removes the region body but preserves boundary markers. "
        "For unmarked supported files, the server maintains a persistent shadow refmark session so reads, region listing, "
        "and edits share one stable marker namespace across the conversation. "
        "For already live-marked files, tools reuse the live marker namespace directly. "
        "Pass edits as structured objects, not as markdown or prose. "
        'Canonical example: {"file_path":"...","expect_live_markers":true,"edits":[{"region_id":"F04","action":"replace","new_content":"def fn():\\n    return 1\\n"}]}.'
    ),
)


@server.tool(
    name="read_refmarked_file",
    description=(
        "Return a refmarked view of a file. For live-marked files, reuses the live marker namespace. "
        "For supported unmarked files, uses a persistent shadow refmark session so repeated reads keep stable ids. "
        "For unsupported file types, returns the raw file content unchanged. Supports optional line windows and truncation guardrails."
    ),
    structured_output=True,
)
def read_refmarked_file_tool(
    file_path: str,
    start_line: int | None = None,
    end_line: int | None = None,
    max_chars: int | None = None,
    max_lines: int | None = None,
) -> dict[str, Any]:
    path = Path(file_path)
    default_chars, default_lines = _shadow_read_defaults()
    effective_max_chars = max_chars or default_chars
    effective_max_lines = max_lines or default_lines
    view_state = _get_view_state(path)
    window = _slice_text_window(
        view_state["view_text"],
        start_line=start_line,
        end_line=end_line,
        max_chars=effective_max_chars,
        max_lines=effective_max_lines,
    )
    return {
        "ok": True,
        "file_path": str(path),
        "refmark_supported": bool(view_state["supported"]),
        "marker_format": view_state.get("marker_format"),
        "chunker": view_state.get("chunker"),
        "marker_count": view_state.get("marker_count", 0),
        "namespace_mode": view_state.get("namespace_mode"),
        "shadow_persistent": view_state.get("shadow_persistent", False),
        "session_reset": view_state.get("session_reset", False),
        **window,
    }


@server.tool(
    name="list_ref_regions",
    description=(
        "List semantic refmark regions for a file. For live-marked files, reuses the live marker namespace. "
        "For supported unmarked files, uses a persistent shadow refmark session so repeated calls keep stable ids. "
        "For unsupported file types, returns refmark_supported=false and an empty region list."
    ),
    structured_output=True,
)
def list_ref_regions_tool(
    file_path: str,
    max_preview_chars: int = 160,
    preview_lines: int = 3,
    min_line_count: int = 1,
) -> dict[str, Any]:
    path = Path(file_path)
    view_state = _get_view_state(path)
    if not view_state["supported"]:
        return {
            "ok": True,
            "file_path": str(path),
            "refmark_supported": False,
            "marker_format": None,
            "chunker": None,
            "namespace_mode": "raw",
            "shadow_persistent": False,
            "session_reset": False,
            "regions": [],
        }

    blocks = _parse_blocks_with_mode(
        view_state["view_text"],
        view_state["marker_format"],
        line_mode="marked",
    )
    regions: list[dict[str, Any]] = []
    for region_id in sorted(blocks.keys()):
        block = blocks[region_id]
        line_start = int(block["line_start"])
        line_end = int(block["line_end"])
        line_count = max(0, line_end - line_start + 1)
        if line_count < min_line_count:
            continue
        text = str(block.get("text", ""))
        regions.append(
            {
                "region_id": str(region_id),
                "start_line": line_start,
                "end_line": line_end,
                "line_count": line_count,
                "preview": _region_preview(text, limit=max_preview_chars),
                "preview_lines": _preview_lines_from_shadow(
                    view_state["view_text"],
                    start_line=line_start,
                    end_line=line_end,
                    max_lines=max(0, preview_lines),
                ),
            }
        )

    return {
        "ok": True,
        "file_path": str(path),
        "refmark_supported": True,
        "marker_format": view_state["marker_format"],
        "chunker": view_state["chunker"],
        "marker_count": view_state.get("marker_count", 0),
        "namespace_mode": view_state.get("namespace_mode"),
        "shadow_persistent": view_state.get("shadow_persistent", False),
        "session_reset": view_state.get("session_reset", False),
        "regions": regions,
    }


def _apply_shadow_session_ref_diff(
    path: Path,
    edits: list[dict[str, Any]],
) -> dict[str, Any]:
    view_state = _get_view_state(path)
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
        _SHADOW_SESSIONS[_shadow_session_key(path)] = ShadowSessionEntry(
            clean_text=next_code,
            shadow_text=next_shadow,
            marker_format=view_state["marker_format"],
            chunker=view_state["chunker"],
            marker_count=_marker_count_from_text(next_shadow, view_state["marker_format"]),
        )

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
    }


@server.tool(
    name="apply_ref_diff",
    description=(
        "Apply multiple refmark-based edits to a Python or TypeScript file. "
        "Each edit must specify either region_id, start_ref, or anchor_ref depending on action. "
        "Range mode replaces the span beginning at start_ref and ending just before end_ref, or to the next ref/EOF "
        "when end_ref is omitted. insert_before inserts before anchor_ref or before EOF. "
        "patch_within applies a diff-like patch only inside the addressed region body. "
        "For supported unmarked files, the server keeps a persistent shadow marker namespace across reads and edits. "
        "action supports replace, delete, insert_before, or patch_within. "
        "Preferred edit shape example: edits=[{\"region_id\":\"F04\",\"action\":\"replace\",\"new_content\":\"def fn():\\n    return 1\\n\"}]."
    ),
    structured_output=True,
)
def apply_ref_diff_tool(
    file_path: str,
    edits: list[RefmarkEdit | dict[str, Any] | str],
    expect_live_markers: bool | None = None,
) -> dict[str, Any]:
    serialized_edits = [_normalize_edit_payload(edit) for edit in edits]
    call_record: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "tool": "apply_ref_diff",
        "file_path": str(Path(file_path)),
        "expect_live_markers": expect_live_markers,
        "edits": serialized_edits,
    }

    try:
        path = Path(file_path)
        if expect_live_markers is True:
            result = apply_ref_diff(
                path,
                serialized_edits,
                expect_live_markers=True,
            )
        else:
            result = _apply_shadow_session_ref_diff(
                path,
                serialized_edits,
            )
    except Exception as exc:
        call_record["ok"] = False
        call_record["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        _append_call_log(call_record)
        raise

    call_record["ok"] = bool(result.get("ok"))
    call_record["syntax_ok"] = result.get("syntax_ok")
    call_record["changed_regions"] = result.get("changed_regions", [])
    call_record["created_regions"] = result.get("created_regions", [])
    call_record["errors"] = result.get("errors", [])
    _append_call_log(call_record)
    return result


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
