"""Minimal multidiff patch helpers retained for the publishable edit surface."""

from __future__ import annotations

import re
from typing import Any


HUNK_HEADER_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? \+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)


def apply_search_replace_edits(original_code: str, edits: list[dict[str, Any]]) -> tuple[str, list[str]]:
    result_code = original_code
    errors: list[str] = []
    for edit in edits:
        original_text = edit.get("original_text", "")
        new_content = edit.get("new_content", "")
        if not isinstance(original_text, str) or not original_text:
            errors.append("Edit is missing original_text.")
            continue
        if not isinstance(new_content, str):
            errors.append("Edit has non-string new_content.")
            continue
        occurrences = result_code.count(original_text)
        if occurrences == 0:
            errors.append("Original snippet not found in current file.")
            continue
        if occurrences > 1:
            errors.append("Original snippet is ambiguous in current file.")
            continue
        result_code = result_code.replace(original_text, new_content, 1)
    return result_code, errors


def _extract_diff_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines)
    return stripped


def _parse_unified_diff_hunks(lines: list[str]) -> list[dict[str, Any]]:
    hunks: list[dict[str, Any]] = []
    idx = 0

    while idx < len(lines) and not lines[idx].startswith("@@"):
        idx += 1

    if idx >= len(lines):
        raise ValueError("Unified diff patch did not contain any hunks.")

    while idx < len(lines):
        header = lines[idx]
        match = HUNK_HEADER_RE.match(header)
        if not match:
            raise ValueError(f"Invalid unified diff hunk header: {header}")
        idx += 1
        hunk_lines: list[str] = []
        while idx < len(lines) and not lines[idx].startswith("@@"):
            hunk_lines.append(lines[idx] + "\n")
            idx += 1
        hunks.append(
            {
                "old_start": int(match.group("old_start")),
                "old_count": int(match.group("old_count") or "1"),
                "new_start": int(match.group("new_start")),
                "new_count": int(match.group("new_count") or "1"),
                "lines": hunk_lines,
            }
        )

    return hunks


def apply_unified_diff_patch(original_code: str, patch_text: str) -> tuple[str, list[str]]:
    patch = _extract_diff_text(patch_text)
    if not patch.strip():
        return original_code, ["No unified diff patch was returned."]

    lines = patch.splitlines()
    try:
        hunks = _parse_unified_diff_hunks(lines)
    except ValueError as exc:
        return original_code, [str(exc)]

    original_lines = original_code.splitlines(keepends=True)
    result_lines: list[str] = []
    cursor = 0

    for hunk in hunks:
        old_start_index = hunk["old_start"] - 1
        if old_start_index < cursor:
            return original_code, ["Unified diff hunks overlap or are out of order."]
        if old_start_index > len(original_lines):
            return original_code, ["Unified diff references lines past end of file."]

        result_lines.extend(original_lines[cursor:old_start_index])
        hunk_cursor = old_start_index

        for line in hunk["lines"]:
            if not line:
                continue
            prefix = line[0]
            payload = line[1:]
            if prefix == " ":
                if hunk_cursor >= len(original_lines) or original_lines[hunk_cursor] != payload:
                    return original_code, ["Unified diff context line did not match current file."]
                result_lines.append(original_lines[hunk_cursor])
                hunk_cursor += 1
            elif prefix == "-":
                if hunk_cursor >= len(original_lines) or original_lines[hunk_cursor] != payload:
                    return original_code, ["Unified diff removal line did not match current file."]
                hunk_cursor += 1
            elif prefix == "+":
                result_lines.append(payload)
            elif prefix == "\\":
                continue
            else:
                return original_code, [f"Unsupported unified diff line prefix: {prefix!r}"]

        cursor = hunk_cursor

    result_lines.extend(original_lines[cursor:])
    return "".join(result_lines), []
