"""Stable public editing primitives for refmark-scoped same-file rewrites."""

from __future__ import annotations

import difflib
import hashlib
import re
from pathlib import Path
from typing import Any

from refmark.regions import _parse_blocks, _parse_blocks_with_mode, _resolve_block, apply_refmark_edits, validate_syntax
from refmark.patches import apply_search_replace_edits, apply_unified_diff_patch
from refmark.core import inject, strip
from refmark.languages import choose_edit_chunker, choose_live_marker_format
from refmark.markers import MarkerRegistry


def _select_chunker(file_path: Path) -> str:
    return choose_edit_chunker(file_path.suffix)


def _select_marker_format(file_path: Path) -> str:
    return choose_live_marker_format(file_path.suffix)


def _remove_rfm_header(content: str) -> str:
    lines = [line for line in content.splitlines() if not line.lstrip().startswith(("RFM:", "# RFM:", "// RFM:"))]
    return "\n".join(lines).lstrip("\n")


def _extract_rfm_header(content: str) -> str:
    header_lines = [line for line in content.splitlines() if line.lstrip().startswith(("RFM:", "# RFM:", "// RFM:"))]
    if not header_lines:
        return ""
    return "\n".join(header_lines) + "\n"


def _apply_live_marker_edits(
    marked_content: str,
    edits: list[dict[str, Any]],
    *,
    marker_format: str,
) -> tuple[str, list[str]]:
    blocks = _parse_blocks_with_mode(marked_content, marker_format, line_mode="marked")
    result_lines = marked_content.splitlines(keepends=True)
    errors: list[str] = []

    sorted_edits = sorted(
        edits,
        key=lambda edit: (_resolve_block(blocks, edit)[1] or {}).get("line_start", 0),
        reverse=True,
    )

    for edit in sorted_edits:
        block_key, block = _resolve_block(blocks, edit)
        if block_key is None or block is None:
            errors.append(f"Block {edit.get('region_id', edit.get('block_id', 0))} not found in marked content.")
            continue

        # Preserve the marker line itself and replace only the block body that follows it.
        start_idx = block["line_start"]
        end_idx = block["line_end"]
        if start_idx < 0 or end_idx < start_idx or end_idx > len(result_lines):
            errors.append(f"Invalid block range for block {block_key}: {block['line_start']}-{block['line_end']}")
            continue

        replacement_lines = str(edit.get("new_content", "")).splitlines(keepends=True)
        if replacement_lines and not replacement_lines[-1].endswith("\n"):
            replacement_lines[-1] += "\n"

        result_lines[start_idx:end_idx] = replacement_lines

    return "".join(result_lines), errors


def _normalize_ref_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    prefix = "".join(ch for ch in text if ch.isalpha()).upper()
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return text.upper()
    return f"{prefix or 'B'}{int(digits):02d}"


def _validate_edit_contract(edit: dict[str, Any]) -> str | None:
    action = str(edit.get("action", "replace")).lower()
    region_id = _normalize_ref_id(edit.get("region_id"))
    start_ref = _normalize_ref_id(edit.get("start_ref"))
    end_ref = _normalize_ref_id(edit.get("end_ref"))
    anchor_ref = _normalize_anchor_ref(edit.get("anchor_ref"))

    if action not in {"replace", "delete", "insert_before", "patch_within"}:
        return f"Unsupported action '{action}'."
    if action == "insert_before":
        if not anchor_ref:
            return "insert_before requires anchor_ref."
        if region_id or start_ref or end_ref:
            return "insert_before uses anchor_ref only and may not include region_id/start_ref/end_ref."
        if "new_content" not in edit:
            return "insert_before requires new_content."
        if edit.get("create_region") and not str(edit.get("new_content", "")).strip():
            return "insert_before with create_region requires non-empty new_content."
        return None
    if action == "patch_within":
        if bool(region_id) == bool(start_ref):
            return "patch_within must specify either region_id or start_ref, but not both."
        if end_ref and not start_ref:
            return "end_ref requires start_ref."
        patch_format = str(edit.get("patch_format", "")).lower()
        if patch_format not in {"line_edits", "search_replace", "unified_diff"}:
            return "patch_within requires patch_format of line_edits, search_replace, or unified_diff."
        if "patch" not in edit:
            return "patch_within requires patch."
        return None

    if bool(region_id) == bool(start_ref):
        return "Each edit must specify either region_id or start_ref, but not both."
    if end_ref and not start_ref:
        return "end_ref requires start_ref."
    if action == "replace" and "new_content" not in edit:
        return "replace action requires new_content."
    return None


def _infer_patch_format(raw_patch: Any) -> str | None:
    if isinstance(raw_patch, dict):
        if isinstance(raw_patch.get("patch"), str) or isinstance(raw_patch.get("diff"), str):
            return "unified_diff"
        if "original_text" in raw_patch or "search" in raw_patch:
            return "search_replace"
        edits = raw_patch.get("edits")
    else:
        edits = raw_patch

    if not isinstance(edits, list) or not edits:
        return None
    if all(isinstance(entry, dict) and "original_text" in entry for entry in edits):
        return "search_replace"
    if all(isinstance(entry, dict) and {"start_line", "end_line", "new_content"} <= set(entry) for entry in edits):
        return "line_edits"
    return None


def _repair_edit_payload(edit: dict[str, Any]) -> dict[str, Any]:
    repaired = dict(edit)
    action = str(repaired.get("action", "replace")).lower()
    if action != "insert_before" and not repaired.get("region_id") and not repaired.get("start_ref"):
        anchor_ref = _normalize_anchor_ref(repaired.get("anchor_ref"))
        if anchor_ref and anchor_ref != "EOF":
            repaired["region_id"] = anchor_ref
            repaired.pop("anchor_ref", None)
    if action == "patch_within":
        region_id = _normalize_ref_id(repaired.get("region_id"))
        start_ref = _normalize_ref_id(repaired.get("start_ref"))
        if region_id and start_ref == region_id and not repaired.get("end_ref"):
            repaired.pop("start_ref", None)

        patch_format = str(repaired.get("patch_format", "")).lower()
        if patch_format not in {"line_edits", "search_replace", "unified_diff"}:
            inferred = _infer_patch_format(repaired.get("patch"))
            if inferred:
                repaired["patch_format"] = inferred
    return repaired


def _resolve_edit_span(
    blocks: dict[str, dict[str, Any]],
    edit: dict[str, Any],
) -> tuple[int, int] | None:
    region_id = _normalize_ref_id(edit.get("region_id"))
    start_ref = _normalize_ref_id(edit.get("start_ref"))
    end_ref = _normalize_ref_id(edit.get("end_ref"))

    if region_id:
        block_key, block = _resolve_block(blocks, {"region_id": region_id})
        if block_key is None or block is None:
            return None
        return int(block["line_start"]), int(block["line_end"])

    if start_ref is None:
        return None

    start_key, start_block = _resolve_block(blocks, {"region_id": start_ref})
    if start_key is None or start_block is None:
        return None

    if end_ref is None:
        return int(start_block["line_start"]), int(start_block["line_end"])

    end_key, end_block = _resolve_block(blocks, {"region_id": end_ref})
    if end_key is None or end_block is None:
        return None

    start_line = int(start_block["line_start"])
    end_line = int(end_block["line_start"]) - 1
    if end_line < start_line:
        return None
    return start_line, end_line


def _replacement_text_for_edit(edit: dict[str, Any]) -> str:
    action = str(edit.get("action", "replace")).lower()
    if action == "delete":
        return ""
    return str(edit.get("new_content", ""))


def _source_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _unified_preview_diff(path: Path, before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"{path}",
            tofile=f"{path} (refmark)",
        )
    )


def _changed_regions_from_edits(edits: list[dict[str, Any]], created_regions: list[dict[str, Any]]) -> list[str]:
    changed: list[str] = []
    for edit in edits:
        action = str(edit.get("action", "replace")).lower()
        if action == "insert_before":
            continue
        for key in ("region_id", "start_ref"):
            value = _normalize_ref_id(edit.get(key))
            if value and value not in changed:
                changed.append(value)
    for region in created_regions:
        value = str(region.get("region_id", "")).strip()
        if value and value not in changed:
            changed.append(value)
    return changed


def _normalize_line_patch_edits(raw_patch: Any) -> list[dict[str, Any]]:
    if isinstance(raw_patch, dict):
        edits = raw_patch.get("edits")
    else:
        edits = raw_patch
    if not isinstance(edits, list):
        raise ValueError("line_edits patch must provide an edits list.")

    normalized: list[dict[str, Any]] = []
    for index, entry in enumerate(edits, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"line_edits entry {index} must be an object.")
        start_line = entry.get("start_line")
        end_line = entry.get("end_line")
        new_content = entry.get("new_content")
        expected_text = entry.get("expected_text")
        if not isinstance(start_line, int) or not isinstance(end_line, int):
            raise ValueError(f"line_edits entry {index} requires integer start_line and end_line.")
        if not isinstance(new_content, str):
            raise ValueError(f"line_edits entry {index} requires string new_content.")
        if expected_text is not None and not isinstance(expected_text, str):
            raise ValueError(f"line_edits entry {index} requires string expected_text when provided.")
        normalized.append(
            {
                "start_line": start_line,
                "end_line": end_line,
                "new_content": new_content,
                "expected_text": expected_text,
            }
        )
    return normalized


def _normalize_search_replace_patch(raw_patch: Any) -> list[dict[str, Any]]:
    if isinstance(raw_patch, dict):
        if "edits" in raw_patch:
            edits = raw_patch.get("edits")
        elif "original_text" in raw_patch or "search" in raw_patch:
            edits = [raw_patch]
        else:
            edits = None
    else:
        edits = raw_patch
    if not isinstance(edits, list):
        raise ValueError("search_replace patch must provide an edits list.")

    normalized: list[dict[str, Any]] = []
    for index, entry in enumerate(edits, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"search_replace entry {index} must be an object.")
        original_text = entry.get("original_text", entry.get("search"))
        new_content = entry.get("new_content", entry.get("replace"))
        if not isinstance(original_text, str) or not original_text:
            raise ValueError(f"search_replace entry {index} requires non-empty original_text.")
        if not isinstance(new_content, str):
            raise ValueError(f"search_replace entry {index} requires string new_content.")
        normalized.append(
            {
                "original_text": original_text,
                "new_content": new_content,
            }
        )
    return normalized


def _extract_region_body_text(lines: list[str], start_idx: int, end_idx: int) -> str:
    return "".join(lines[start_idx:end_idx])


def _resolve_expected_window(
    *,
    lines: list[str],
    start_idx: int,
    end_idx: int,
    expected_text: str,
) -> tuple[int, int] | None:
    """Resolve an expected_text mismatch to a safer in-region line window.

    Preference order:
    1. one-line shift up/down (typical off-by-one with blank lines)
    2. unique exact expected_text match anywhere in the region
    """
    window_len = max(0, end_idx - start_idx)
    if window_len <= 0:
        return None

    # Off-by-one upward.
    if start_idx > 0:
        up_start = start_idx - 1
        up_end = up_start + window_len
        if up_end <= len(lines):
            if "".join(lines[up_start:up_end]) == expected_text:
                return up_start, up_end

    # Off-by-one downward.
    down_start = start_idx + 1
    down_end = down_start + window_len
    if down_end <= len(lines):
        if "".join(lines[down_start:down_end]) == expected_text:
            return down_start, down_end

    # Unique exact match across the region.
    candidates: list[tuple[int, int]] = []
    for idx in range(0, len(lines) - window_len + 1):
        candidate_end = idx + window_len
        if "".join(lines[idx:candidate_end]) == expected_text:
            candidates.append((idx, candidate_end))
    if len(candidates) == 1:
        return candidates[0]
    return None


def _expected_text_variants(expected_text: str) -> list[str]:
    variants = [expected_text]
    if "\\n" in expected_text or "\\t" in expected_text:
        variants.append(expected_text.replace("\\n", "\n").replace("\\t", "\t"))
    if '\\"' in expected_text or "\\'" in expected_text:
        variants.append(expected_text.replace('\\"', '"').replace("\\'", "'"))
    if not expected_text.endswith("\n"):
        variants.append(expected_text + "\n")
    for variant in list(variants):
        if not variant.endswith("\n"):
            variants.append(variant + "\n")

    deduped: list[str] = []
    for variant in variants:
        if variant not in deduped:
            deduped.append(variant)
    return deduped


def _apply_patch_within_span(
    region_text: str,
    edit: dict[str, Any],
) -> tuple[str, list[str]]:
    patch_format = str(edit.get("patch_format", "")).lower()
    raw_patch = edit.get("patch")

    if patch_format == "line_edits":
        try:
            normalized_edits = _normalize_line_patch_edits(raw_patch)
        except ValueError as exc:
            return region_text, [str(exc)]
        updated_text = region_text
        for line_edit in sorted(normalized_edits, key=lambda item: item["start_line"], reverse=True):
            lines = updated_text.splitlines(keepends=True)
            start_line = line_edit["start_line"]
            end_line = line_edit["end_line"]
            if start_line < 1 or end_line < start_line:
                return region_text, [f"Invalid line_edits range: {start_line}-{end_line}."]
            if start_line > len(lines):
                return region_text, [f"line_edits start_line {start_line} exceeds region length {len(lines)}."]
            start_idx = start_line - 1
            end_idx = min(end_line, len(lines))
            expected_text = line_edit.get("expected_text")
            actual_text = "".join(lines[start_idx:end_idx])
            if expected_text is not None and actual_text not in _expected_text_variants(expected_text):
                resolved = None
                for expected_variant in _expected_text_variants(expected_text):
                    resolved = _resolve_expected_window(
                        lines=lines,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        expected_text=expected_variant,
                    )
                    if resolved is not None:
                        break
                if resolved is None:
                    return region_text, [
                        "line_edits expected_text mismatch for "
                        f"{start_line}-{end_line}. Expected {expected_text!r}, found {actual_text!r}."
                    ]
                start_idx, end_idx = resolved
            replacement_lines = line_edit["new_content"].splitlines(keepends=True)
            if replacement_lines and not replacement_lines[-1].endswith("\n"):
                replacement_lines[-1] += "\n"
            lines[start_idx:end_idx] = replacement_lines
            updated_text = "".join(lines)
        return updated_text, []

    if patch_format == "search_replace":
        try:
            normalized_edits = _normalize_search_replace_patch(raw_patch)
        except ValueError as exc:
            return region_text, [str(exc)]
        return apply_search_replace_edits(region_text, normalized_edits)

    if patch_format == "unified_diff":
        if isinstance(raw_patch, dict):
            patch_text = raw_patch.get("patch") or raw_patch.get("diff") or ""
        else:
            patch_text = raw_patch
        if not isinstance(patch_text, str) or not patch_text.strip():
            return region_text, ["unified_diff patch requires a non-empty patch string."]
        return apply_unified_diff_patch(region_text, patch_text)

    return region_text, [f"Unsupported patch_format '{patch_format}'."]


def _normalize_anchor_ref(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.upper() == "EOF":
        return "EOF"
    return _normalize_ref_id(text)


def _available_region_ids(blocks: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(str(key) for key in blocks.keys())


def _format_target_resolution_error(target: Any, blocks: dict[str, dict[str, Any]]) -> str:
    available = ", ".join(_available_region_ids(blocks)[:20])
    if available:
        return f"Could not resolve edit target '{target}'. Available regions: {available}"
    return f"Could not resolve edit target '{target}'."


def _detect_overlapping_spans(spans: list[tuple[int, int, dict[str, Any]]]) -> list[str]:
    errors: list[str] = []
    sorted_spans = sorted(spans, key=lambda item: item[0])
    for previous, current in zip(sorted_spans, sorted_spans[1:]):
        prev_start, prev_end, prev_edit = previous
        cur_start, cur_end, cur_edit = current
        if cur_start <= prev_end:
            prev_target = prev_edit.get("region_id") or prev_edit.get("start_ref") or prev_edit.get("block_id")
            cur_target = cur_edit.get("region_id") or cur_edit.get("start_ref") or cur_edit.get("block_id")
            errors.append(
                f"Overlapping edit spans are not allowed: '{prev_target}' ({prev_start}-{prev_end}) overlaps "
                f"with '{cur_target}' ({cur_start}-{cur_end})."
            )
    return errors


def _infer_created_region_prefix(
    *,
    file_ext: str,
    anchor_ref: str | None,
    new_content: str,
) -> str:
    stripped = new_content.lstrip()
    upper_anchor = (anchor_ref or "").upper()

    if stripped.startswith(("class ", "export class ")):
        return "C"
    if stripped.startswith(("def ", "async def ", "function ", "export function ")):
        if upper_anchor.startswith("M"):
            return "M"
        return "F"
    if upper_anchor:
        return "".join(ch for ch in upper_anchor if ch.isalpha()) or "B"
    if file_ext == ".py":
        return "F"
    return "B"


def _next_region_id(
    blocks: dict[str, dict[str, Any]],
    *,
    prefix: str,
) -> str:
    highest = 0
    for key in blocks:
        if not isinstance(key, str):
            continue
        normalized = _normalize_ref_id(key) or ""
        key_prefix = "".join(ch for ch in normalized if ch.isalpha())
        digits = "".join(ch for ch in normalized if ch.isdigit())
        if key_prefix == prefix and digits:
            highest = max(highest, int(digits))
    return f"{prefix}{highest + 1:02d}"


def _render_live_marker(marker_format: str, region_id: str) -> str:
    fmt = MarkerRegistry.get(marker_format)
    return fmt.inject_template.format(id=0, label=region_id)


def _build_insertion_lines(
    *,
    marker_format: str,
    region_id: str | None,
    new_content: str,
) -> list[str]:
    insertion_lines: list[str] = []
    if region_id:
        marker_line = _render_live_marker(marker_format, region_id)
        if marker_format in {"typed_comment_py", "typed_comment_ts", "comment_py", "comment_ts"}:
            insertion_lines.append(marker_line + "\n")
        else:
            insertion_lines.append(marker_line + " ")

    payload_lines = new_content.splitlines(keepends=True)
    if payload_lines and not payload_lines[-1].endswith("\n"):
        payload_lines[-1] += "\n"
    insertion_lines.extend(payload_lines)
    return insertion_lines


def _resolve_insert_index(
    blocks: dict[str, dict[str, Any]],
    *,
    anchor_ref: str,
    line_mode: str,
    total_lines: int,
) -> int | None:
    if anchor_ref == "EOF":
        return total_lines

    block_key, block = _resolve_block(blocks, {"region_id": anchor_ref})
    if block_key is None or block is None:
        return None

    if line_mode == "marked":
        return int(block["line_start"]) - 1
    return int(block["line_start"]) - 1


def _detect_premarked(content: str, marker_format: str | None = None) -> bool:
    """Return true only when content contains parseable live refmark blocks."""
    marker_formats = [marker_format] if marker_format else [
        "typed_comment_py",
        "typed_comment_ts",
        "comment_py",
        "comment_ts",
        "typed_explicit",
        "typed_compact",
        "typed_xml",
        "xml",
        "typed_bracket",
        "bracket",
    ]
    return any(_has_live_marker_at_line_start(content, fmt) for fmt in marker_formats if fmt)


def _has_live_marker_at_line_start(content: str, marker_format: str) -> bool:
    marker_res = {
        "typed_comment_py": r"^\s*# \[@[A-Z]+\d+\]\s*$",
        "typed_comment_ts": r"^\s*// \[@[A-Z]+\d+\]\s*$",
        "comment_py": r"^\s*# @ref:\d+\s*$",
        "comment_ts": r"^\s*// @ref:\d+\s*$",
        "typed_explicit": r"^\s*\[ref:[A-Z]+\d+\]",
        "typed_compact": r"^\s*\[[A-Z]+\d+\]",
        "typed_xml": r'^\s*<block id="[A-Z]+\d+"\s*/>',
        "xml": r'^\s*<block id="[A-Z]*\d+"\s*/>',
        "typed_bracket": r"^\s*\[@[A-Z]+\d+\]",
        "bracket": r"^\s*\[@[A-Z]+\d+\]",
    }
    marker_re = marker_res.get(marker_format)
    if marker_re is None:
        return bool(_parse_blocks_with_mode(content, marker_format, line_mode="marked"))
    return bool(re.search(marker_re, content, flags=re.IGNORECASE | re.MULTILINE))


def _apply_edits_to_lines(
    lines: list[str],
    blocks: dict[str, dict[str, Any]],
    edits: list[dict[str, Any]],
    *,
    line_mode: str,
    marker_format: str,
    file_ext: str,
    allow_region_creation: bool,
) -> tuple[str, list[str], list[dict[str, Any]]]:
    errors: list[str] = []
    normalized_edits: list[tuple[int, int, dict[str, Any]]] = []
    insert_ops: list[tuple[int, dict[str, Any]]] = []
    created_regions: list[dict[str, Any]] = []

    for edit in edits:
        edit = _repair_edit_payload(edit)
        error = _validate_edit_contract(edit)
        if error:
            errors.append(error)
            continue

        action = str(edit.get("action", "replace")).lower()
        if action == "insert_before":
            anchor_ref = _normalize_anchor_ref(edit.get("anchor_ref"))
            if anchor_ref is None:
                errors.append("insert_before requires anchor_ref.")
                continue
            if edit.get("create_region") and not allow_region_creation:
                errors.append("create_region requires live markers.")
                continue
            insert_index = _resolve_insert_index(
                blocks,
                anchor_ref=anchor_ref,
                line_mode=line_mode,
                total_lines=len(lines),
            )
            if insert_index is None:
                errors.append(_format_target_resolution_error(anchor_ref, blocks))
                continue
            insert_ops.append((insert_index, edit))
            continue

        span = _resolve_edit_span(blocks, edit)
        target = edit.get("region_id") or edit.get("start_ref") or edit.get("block_id")
        if span is None:
            errors.append(_format_target_resolution_error(target, blocks))
            continue

        start_line, end_line = span
        normalized_edits.append((start_line, end_line, edit))

    if errors:
        return "".join(lines), errors, created_regions

    overlap_errors = _detect_overlapping_spans(normalized_edits)
    if overlap_errors:
        return "".join(lines), overlap_errors, created_regions

    for start_line, end_line, edit in sorted(normalized_edits, key=lambda item: item[0], reverse=True):
        start_idx = start_line if line_mode == "marked" else start_line - 1
        end_idx = end_line
        if start_idx < 0 or end_idx < start_idx or end_idx > len(lines):
            target = edit.get("region_id") or edit.get("start_ref") or edit.get("block_id")
            errors.append(f"Invalid resolved span for edit target '{target}': {start_line}-{end_line}.")
            continue

        action = str(edit.get("action", "replace")).lower()
        if action == "patch_within":
            region_text = _extract_region_body_text(lines, start_idx, end_idx)
            patched_text, patch_errors = _apply_patch_within_span(region_text, edit)
            if patch_errors:
                target = edit.get("region_id") or edit.get("start_ref") or edit.get("block_id")
                errors.extend(f"{target}: {message}" for message in patch_errors)
                continue
            replacement_lines = patched_text.splitlines(keepends=True)
        else:
            replacement_lines = _replacement_text_for_edit(edit).splitlines(keepends=True)
        if replacement_lines and not replacement_lines[-1].endswith("\n"):
            replacement_lines[-1] += "\n"
        lines[start_idx:end_idx] = replacement_lines

    for insert_index, edit in sorted(insert_ops, key=lambda item: item[0], reverse=True):
        anchor_ref = _normalize_anchor_ref(edit.get("anchor_ref"))
        create_region = bool(edit.get("create_region"))
        region_id: str | None = None
        if create_region:
            prefix = _infer_created_region_prefix(
                file_ext=file_ext,
                anchor_ref=anchor_ref,
                new_content=str(edit.get("new_content", "")),
            )
            region_id = _next_region_id(blocks, prefix=prefix)
            blocks[region_id] = {
                "text": "",
                "line_start": insert_index + 1,
                "line_end": insert_index + 1,
                "ordinal": len(blocks) + 1,
            }
            created_regions.append(
                {
                    "region_id": region_id,
                    "introduced_before": anchor_ref,
                }
            )
        insertion_lines = _build_insertion_lines(
            marker_format=marker_format,
            region_id=region_id,
            new_content=str(edit.get("new_content", "")),
        )
        lines[insert_index:insert_index] = insertion_lines

    return "".join(lines), errors, created_regions


def apply_refmark_multidiff(
    file_path: str | Path,
    edits: list[dict[str, Any]],
) -> dict[str, Any]:
    path = Path(file_path)
    marker_format = _select_marker_format(path)
    chunker = _select_chunker(path)
    current_content = path.read_text(encoding="utf-8")
    was_premarked = _detect_premarked(current_content, marker_format=marker_format)

    if was_premarked:
        marked_content = current_content
        next_marked, apply_errors = _apply_live_marker_edits(
            marked_content,
            edits,
            marker_format=marker_format,
        )
        next_code = strip(next_marked, path.suffix, marker_format=marker_format)
        next_code = _remove_rfm_header(next_code)
        if next_code and not next_code.endswith("\n"):
            next_code += "\n"
    else:
        original_code = current_content
        marked_content, _ = inject(original_code, path.suffix, marker_format=marker_format, chunker=chunker)
        next_code, results = apply_refmark_edits(
            original_code,
            marked_content,
            edits,
            marker_format=marker_format,
        )
        next_code = strip(next_code, path.suffix, marker_format=marker_format)
        next_code = _remove_rfm_header(next_code)
        if next_code and not next_code.endswith("\n"):
            next_code += "\n"
        apply_errors = [result.error for result in results if result.error]

    syntax_ok = validate_syntax(next_code, path.suffix)
    if not syntax_ok:
        apply_errors.append(f"Edited file is not syntactically valid for {path.suffix or 'this language'}.")

    if not apply_errors:
        if was_premarked:
            path.write_text(next_marked, encoding="utf-8")
        else:
            path.write_text(next_code, encoding="utf-8")

    return {
        "ok": not apply_errors,
        "file_path": str(path),
        "applied_edits": 0 if apply_errors else len([edit for edit in edits if isinstance(edit, dict)]),
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_seconds": 0.0,
        "syntax_ok": syntax_ok,
        "errors": apply_errors,
    }


def apply_ref_diff(
    file_path: str | Path,
    edits: list[dict[str, Any]],
    *,
    expect_live_markers: bool | None = None,
    dry_run: bool = False,
    base_hash: str | None = None,
    include_diff: bool = False,
) -> dict[str, Any]:
    """Apply multiple refmark-scoped edits to a single file.

    This is the stable product-facing primitive used by the CLI and MCP server.
    Each edit targets one semantic region via ``region_id`` or an explicit range
    via ``start_ref`` and optional ``end_ref``.
    """
    path = Path(file_path)
    marker_format = _select_marker_format(path)
    chunker = _select_chunker(path)
    current_content = path.read_text(encoding="utf-8")
    source_hash = _source_hash(current_content)
    if base_hash and base_hash != source_hash:
        return {
            "ok": False,
            "file_path": str(path),
            "applied_edits": 0,
            "created_regions": [],
            "changed_regions": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_seconds": 0.0,
            "syntax_ok": False,
            "dry_run": dry_run,
            "source_hash": source_hash,
            "output_hash": None,
            "errors": ["Base hash mismatch; file changed since the caller inspected it."],
        }
    was_premarked = _detect_premarked(current_content, marker_format=marker_format)

    if expect_live_markers is True and not was_premarked:
        return {
            "ok": False,
            "file_path": str(path),
            "applied_edits": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_seconds": 0.0,
            "syntax_ok": False,
            "dry_run": dry_run,
            "source_hash": source_hash,
            "output_hash": None,
            "errors": ["File does not contain live refmarks, but expect_live_markers was true."],
        }

    if not was_premarked:
        original_code = current_content
        marked_content, _ = inject(original_code, path.suffix, marker_format=marker_format, chunker=chunker)
    else:
        marked_content = current_content
        original_code = strip(marked_content, path.suffix, marker_format=marker_format)
        original_code = _remove_rfm_header(original_code)
        if original_code and not original_code.endswith("\n"):
            original_code += "\n"

    marked_blocks = _parse_blocks_with_mode(marked_content, marker_format, line_mode="marked")
    original_blocks = _parse_blocks(marked_content, marker_format)

    apply_errors: list[str] = []
    created_regions: list[dict[str, Any]] = []
    if was_premarked:
        next_marked, apply_errors, created_regions = _apply_edits_to_lines(
            marked_content.splitlines(keepends=True),
            marked_blocks,
            edits,
            line_mode="marked",
            marker_format=marker_format,
            file_ext=path.suffix,
            allow_region_creation=True,
        )
        next_code = strip(next_marked, path.suffix, marker_format=marker_format)
        next_code = _remove_rfm_header(next_code)
    else:
        next_code, apply_errors, created_regions = _apply_edits_to_lines(
            original_code.splitlines(keepends=True),
            original_blocks,
            edits,
            line_mode="original",
            marker_format=marker_format,
            file_ext=path.suffix,
            allow_region_creation=False,
        )

    if next_code and not next_code.endswith("\n"):
        next_code += "\n"

    syntax_ok = validate_syntax(next_code, path.suffix)
    if not syntax_ok:
        apply_errors.append(f"Edited file is not syntactically valid for {path.suffix or 'this language'}.")

    output_content = next_marked if was_premarked else next_code
    output_hash = _source_hash(output_content)
    changed_regions = _changed_regions_from_edits(edits, created_regions)
    preview_diff = _unified_preview_diff(path, current_content, output_content) if include_diff else None

    if not apply_errors and not dry_run:
        if was_premarked:
            path.write_text(next_marked, encoding="utf-8")
        else:
            path.write_text(next_code, encoding="utf-8")

    return {
        "ok": not apply_errors,
        "file_path": str(path),
        "applied_edits": 0 if apply_errors else len([edit for edit in edits if isinstance(edit, dict)]),
        "created_regions": created_regions,
        "changed_regions": changed_regions if not apply_errors else [],
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_seconds": 0.0,
        "syntax_ok": syntax_ok,
        "dry_run": dry_run,
        "source_hash": source_hash,
        "output_hash": output_hash if not apply_errors else None,
        "diff": preview_diff,
        "errors": apply_errors,
    }
