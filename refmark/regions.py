"""Region parsing, syntax validation, and edit-application helpers."""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from refmark.languages import validate_code_syntax



@dataclass
class EditInstruction:
    """A single edit instruction with its target and content."""
    edit_id: int
    target: str  # block_id (Refmark), line_range (line-number), or function_name (diff)
    description: str
    original_text: str
    replacement_text: str
    # Ground truth for validation
    expected_line_start: int = 0
    expected_line_end: int = 0


@dataclass
class EditResult:
    """Result of applying a single edit."""
    edit_id: int
    target: str
    applied: bool
    correct: bool
    error: str = ""
    applied_line_start: int = 0
    applied_line_end: int = 0


@dataclass
class BatchEditResult:
    """Result of applying a batch of edits."""
    condition: str  # "line_number", "unified_diff", "refmark", "refmark_ast"
    model_id: str
    codebase_id: str
    batch_size: int
    total_edits: int
    applied_edits: int
    correct_edits: int
    conflict_count: int
    syntax_valid: bool
    edit_results: list[EditResult] = field(default_factory=list)
    model_output: str = ""
    latency: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def application_rate(self) -> float:
        return self.applied_edits / self.total_edits if self.total_edits > 0 else 0.0

    @property
    def correctness_rate(self) -> float:
        return self.correct_edits / self.total_edits if self.total_edits > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "model_id": self.model_id,
            "codebase_id": self.codebase_id,
            "batch_size": self.batch_size,
            "total_edits": self.total_edits,
            "applied_edits": self.applied_edits,
            "correct_edits": self.correct_edits,
            "conflict_count": self.conflict_count,
            "syntax_valid": self.syntax_valid,
            "application_rate": round(self.application_rate, 4),
            "correctness_rate": round(self.correctness_rate, 4),
            "model_output": self.model_output[:1000],
            "latency": round(self.latency, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "edit_results": [
                {
                    "edit_id": r.edit_id,
                    "target": r.target,
                    "applied": r.applied,
                    "correct": r.correct,
                    "error": r.error,
                }
                for r in self.edit_results
            ],
        }



def build_line_number_prompt(
    code_content: str,
    edits: list[EditInstruction],
) -> str:
    """Build a prompt using line-number based addressing."""
    lines = code_content.splitlines()
    numbered_lines = []
    for i, line in enumerate(lines, 1):
        numbered_lines.append(f"{i:5d} | {line}")
    numbered_code = "\n".join(numbered_lines)

    edit_instructions = []
    for edit in edits:
        edit_instructions.append(
            f"Edit {edit.edit_id}: Replace lines {edit.expected_line_start}–{edit.expected_line_end} "
            f"with:\n{edit.replacement_text}"
        )

    instructions_text = "\n".join(edit_instructions)
    prompt = (
        "You are a code editor. The code below has line numbers on the left.\n"
        "Make the specified edits by returning a JSON array of edit operations.\n"
        "Each edit must specify the line range to replace and the new content.\n\n"
        "CODE (with line numbers):\n"
        f"{numbered_code}\n\n"
        "EDITS TO MAKE:\n"
        f"{instructions_text}\n\n"
        "Respond with ONLY a JSON array in this format:\n"
        '[{"start_line": N, "end_line": M, "new_content": "..."}]\n'
        "Do not include any other text."
    )
    return prompt


def build_unified_diff_prompt(
    code_content: str,
    edits: list[EditInstruction],
) -> str:
    """Build a prompt using unified-diff based addressing."""
    # Build a diff-like representation for each edit
    diff_hunks = []
    for edit in edits:
        old_lines = edit.original_text.splitlines()
        new_lines = edit.replacement_text.splitlines()
        diff_lines = list(difflib.unified_diff(
            old_lines, new_lines,
            fromfile="original", tofile="modified", lineterm="",
        ))
        diff_hunks.append(f"Edit {edit.edit_id}:\n" + "\n".join(diff_lines))

    diff_text = "\n\n".join(diff_hunks)
    prompt = (
        "You are a code editor. Apply the following unified diffs to the code.\n"
        "Return the FULL edited code file after applying all changes.\n"
        "Do not include any explanation or markdown code fences.\n\n"
        "ORIGINAL CODE:\n"
        f"{code_content}\n\n"
        "DIFFS TO APPLY:\n"
        f"{diff_text}\n\n"
        "Return the full edited code now."
    )
    return prompt


def build_refmark_prompt(
    code_content: str,
    marked_content: str,
    edits: list[EditInstruction],
    marker_format: str = "xml",
) -> str:
    """Build a prompt using Refmark marker-based addressing."""
    edit_instructions = []
    for edit in edits:
        edit_instructions.append(
            f"Edit {edit.edit_id}: Replace the content in block {edit.target} "
            f"with:\n{edit.replacement_text}"
        )

    format_hint = _get_format_hint(marker_format)
    instructions_text = "\n".join(edit_instructions)
    prompt = (
        "You are a code editor. The code below contains reference markers.\n"
        "Make the specified edits by returning a JSON array of edit operations.\n"
        "Each edit must specify the block ID to replace and the new content.\n\n"
        f"Marker format: {format_hint}\n\n"
        "MARKED CODE:\n"
        f"{marked_content}\n\n"
        "EDITS TO MAKE:\n"
        f"{instructions_text}\n\n"
        "Respond with ONLY a JSON array in this format:\n"
        '[{"block_id": "B01", "new_content": "..."}]\n'
        "Do not include any other text."
    )
    return prompt


def _get_format_hint(marker_format: str) -> str:
    """Return a human-readable description of the marker format."""
    hints = {
        "xml": '<block id="N"/> - XML-style marker',
        "bracket": "[@BNN] - Bracket-style marker",
        "curly": "{BNN} - Curly-brace marker",
    }
    return hints.get(marker_format, marker_format)



def parse_line_number_edits(model_output: str) -> list[dict]:
    """Parse line-number based edit instructions from model output.

    Expected format: JSON array with start_line, end_line, new_content.
    """
    edits = _extract_json_array(model_output)
    result = []
    for edit in edits:
        if isinstance(edit, dict) and "start_line" in edit:
            result.append({
                "start_line": int(edit.get("start_line", 0)),
                "end_line": int(edit.get("end_line", 0)),
                "new_content": edit.get("new_content", ""),
            })
    return result


def parse_unified_diff_output(model_output: str) -> str:
    """Extract full code from unified-diff model output.

    The model should return the full edited code. Strip markdown fences if present.
    """
    # Try to extract from markdown code fences
    code_blocks = re.findall(r"```(?:python)?\n(.*?)```", model_output, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    # If no fences, return the whole output (trimmed)
    return model_output.strip()


def parse_refmark_edits(model_output: str) -> list[dict]:
    """Parse Refmark-based edit instructions from model output.

    Expected format: JSON array with block_id, new_content.
    """
    edits = _extract_json_array(model_output)
    result = []
    for edit in edits:
        if isinstance(edit, dict) and "block_id" in edit:
            block_id = edit["block_id"]
            # Normalize block ID: "B01" -> 1, "F03" -> 3, "1" -> 1
            m = re.search(r'([A-Za-z]+)?(\d+)', str(block_id))
            if m:
                result.append({
                    "region_id": f"{(m.group(1) or 'B').upper()}{int(m.group(2)):02d}",
                    "block_id": int(m.group(2)),
                    "new_content": edit.get("new_content", ""),
                })
    return result


def _extract_json_array(text: str) -> list:
    """Try to extract a JSON array from model output."""
    # Try direct parse
    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find JSON array in text
    m = re.search(r'\[.*\]', text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    # Try to find JSON array in code fences
    code_blocks = re.findall(r"```(?:json)?\n(.*?)```", text, re.DOTALL)
    for block in code_blocks:
        try:
            parsed = json.loads(block.strip())
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    return []



def apply_line_number_edits(
    original_code: str,
    parsed_edits: list[dict],
) -> tuple[str, list[EditResult]]:
    """Apply line-number based edits to source code.

    Handles line drift by sorting edits by start_line (descending) and
    applying them from bottom to top.
    """
    lines = original_code.splitlines(keepends=True)
    # Normalize: ensure lines end with newline
    normalized = []
    for line in lines:
        if not line.endswith("\n"):
            normalized.append(line + "\n")
        else:
            normalized.append(line)
    # Remove trailing newline from last line for consistent handling
    if normalized and normalized[-1].endswith("\n"):
        normalized[-1] = normalized[-1][:-1]

    results = []
    # Sort by start_line descending to avoid drift
    sorted_edits = sorted(parsed_edits, key=lambda e: e.get("start_line", 0), reverse=True)

    for edit in sorted_edits:
        start = edit.get("start_line", 0)
        end = edit.get("end_line", 0)
        new_content = edit.get("new_content", "")

        if start < 1 or end < start or start > len(normalized):
            results.append(EditResult(
                edit_id=0, target=f"lines {start}-{end}",
                applied=False, correct=False,
                error=f"Invalid line range: {start}-{end} (file has {len(normalized)} lines)",
            ))
            continue

        # Convert 1-based to 0-based
        start_idx = start - 1
        end_idx = min(end, len(normalized))

        new_lines = new_content.splitlines(keepends=True)
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] = new_lines[-1] + "\n"

        normalized[start_idx:end_idx] = new_lines
        results.append(EditResult(
            edit_id=0, target=f"lines {start}-{end}",
            applied=True, correct=True,
            applied_line_start=start,
            applied_line_end=start + len(new_lines) - 1,
        ))

    result_code = "\n".join(normalized)
    return result_code, results


def apply_unified_diff_output(
    original_code: str,
    model_output: str,
    expected_edits: list[EditInstruction],
) -> tuple[str, list[EditResult]]:
    """Apply unified-diff model output (full code replacement).

    The model returns the full edited code. We compare it against expected edits.
    """
    edited_code = parse_unified_diff_output(model_output)
    results = []

    for edit in expected_edits:
        # Check if the original text is gone and replacement is present
        original_present = edit.original_text.strip() in edited_code
        replacement_present = edit.replacement_text.strip() in edited_code

        applied = replacement_present
        correct = not original_present and replacement_present

        results.append(EditResult(
            edit_id=edit.edit_id, target=edit.target,
            applied=applied, correct=correct,
        ))

    return edited_code, results


def apply_refmark_edits(
    original_code: str,
    marked_content: str,
    parsed_edits: list[dict],
    marker_format: str = "xml",
) -> tuple[str, list[EditResult]]:
    """Apply Refmark-based edits to source code.

    Uses markers to identify target regions, then applies replacements.
    Markers are immune to line drift since block IDs are stable.
    """
    # Parse the marked content to find block boundaries
    blocks = _parse_blocks(marked_content, marker_format)

    results = []
    result_lines = original_code.splitlines(keepends=True)

    sorted_edits = sorted(
        parsed_edits,
        key=lambda edit: (_resolve_block(blocks, edit)[1] or {}).get("line_start", 0),
        reverse=True,
    )

    for edit in sorted_edits:
        block_key, block = _resolve_block(blocks, edit)
        new_content = edit.get("new_content", "")

        if block_key is None or block is None:
            results.append(EditResult(
                edit_id=0, target=f"block {edit.get('region_id', edit.get('block_id', 0))}",
                applied=False, correct=False,
                error=f"Block {edit.get('region_id', edit.get('block_id', 0))} not found in marked content",
            ))
            continue

        start_idx = block["line_start"] - 1
        end_idx = block["line_end"]
        if start_idx < 0 or end_idx < start_idx or start_idx > len(result_lines):
            results.append(EditResult(
                edit_id=0, target=f"block {block_key}",
                applied=False, correct=False,
                error=f"Invalid block range for block {block_key}: {block['line_start']}-{block['line_end']}",
            ))
            continue

        replacement_lines = new_content.splitlines(keepends=True)
        if replacement_lines and not replacement_lines[-1].endswith("\n"):
            replacement_lines[-1] = replacement_lines[-1] + "\n"

        result_lines[start_idx:end_idx] = replacement_lines
        results.append(EditResult(
            edit_id=0, target=f"block {block_key}",
            applied=True, correct=True,
            applied_line_start=block["line_start"],
            applied_line_end=block["line_start"] + max(len(replacement_lines), 1) - 1,
        ))

    return "".join(result_lines), results


def _parse_blocks(marked_content: str, marker_format: str) -> dict[str, dict]:
    """Parse marked content into blocks with their text and line positions.

    Returns dict mapping block_id -> {"text": str, "line_start": int, "line_end": int}
    """
    return _parse_blocks_with_mode(marked_content, marker_format, line_mode="original")


def _parse_blocks_with_mode(
    marked_content: str,
    marker_format: str,
    *,
    line_mode: str = "original",
) -> dict[str, dict]:
    """Parse marked content into blocks using original or marked line coordinates."""
    if marker_format == "xml":
        marker_re = re.compile(r'<block id="([A-Z]*\d+)"/>', re.IGNORECASE)
    elif marker_format == "bracket":
        marker_re = re.compile(r'\[@([A-Z]+\d+)\]', re.IGNORECASE)
    elif marker_format == "typed_bracket":
        marker_re = re.compile(r'\[@([A-Z]+\d+)\]', re.IGNORECASE)
    elif marker_format == "typed_xml":
        marker_re = re.compile(r'<block id="([A-Z]+\d+)"/>', re.IGNORECASE)
    elif marker_format == "curly":
        marker_re = re.compile(r'\{([A-Z]+\d+)\}', re.IGNORECASE)
    elif marker_format == "comment_py":
        marker_re = re.compile(r'^# @ref:(\d+)\s*$', re.IGNORECASE)
    elif marker_format == "typed_comment_py":
        marker_re = re.compile(r'^# \[@([A-Z]+\d+)\]\s*$', re.IGNORECASE)
    elif marker_format == "comment_ts":
        marker_re = re.compile(r'^// @ref:(\d+)\s*$', re.IGNORECASE)
    elif marker_format == "typed_comment_ts":
        marker_re = re.compile(r'^// \[@([A-Z]+\d+)\]\s*$', re.IGNORECASE)
    else:
        marker_re = re.compile(r'<block id="([A-Z]*\d+)"/>', re.IGNORECASE)

    standalone_marker_formats = {
        "comment_py",
        "typed_comment_py",
        "comment_ts",
        "typed_comment_ts",
    }
    markers_are_standalone_lines = marker_format in standalone_marker_formats

    blocks = {}
    lines = marked_content.splitlines(keepends=True)

    # Find all marker positions.
    marker_positions = []  # (line_number, block_key)
    for i, line in enumerate(lines, 1):
        for m in marker_re.finditer(line):
            block_key = _normalize_region_key(m.group(1))
            marker_positions.append((i, block_key))

    # Build blocks: text between marker N and marker N+1.
    for idx, (line_num, block_key) in enumerate(marker_positions):
        if idx + 1 < len(marker_positions):
            next_line_num = marker_positions[idx + 1][0]
            block_lines = lines[line_num - 1:next_line_num - 1]
            if block_lines:
                block_lines[0] = marker_re.sub("", block_lines[0])
            block_text = "".join(block_lines).strip()
            line_end = next_line_num - 1
        else:
            block_lines = lines[line_num - 1:]
            if block_lines:
                block_lines[0] = marker_re.sub("", block_lines[0])
            block_text = "".join(block_lines).strip()
            line_end = len(lines)

        if markers_are_standalone_lines and line_mode == "original":
            original_line_start = line_num - idx
            original_line_end = line_end - (idx + 1)
        else:
            original_line_start = line_num
            original_line_end = line_end

        blocks[block_key] = {
            "text": block_text,
            "line_start": original_line_start,
            "line_end": original_line_end,
            "ordinal": idx + 1,
        }

    return blocks


def _normalize_region_key(value: str) -> str:
    match = re.search(r"([A-Z]+)?(\d+)", str(value).upper())
    if not match:
        return str(value).upper()
    prefix = match.group(1) or "B"
    return f"{prefix}{int(match.group(2)):02d}"


def _resolve_block(blocks: dict[str, dict], edit: dict) -> tuple[str | None, dict | None]:
    region_id = edit.get("region_id")
    if isinstance(region_id, str):
        normalized = _normalize_region_key(region_id)
        if normalized in blocks:
            return normalized, blocks[normalized]

    block_id = edit.get("block_id", 0)
    try:
        block_number = int(block_id)
    except (TypeError, ValueError):
        block_number = 0

    if block_number:
        numeric_key = f"B{block_number:02d}"
        if numeric_key in blocks:
            return numeric_key, blocks[numeric_key]
        for key, block in blocks.items():
            if block.get("ordinal") == block_number:
                return key, block

    return None, None



def validate_syntax(code: str, file_ext: str = ".py") -> bool:
    """Check if code is syntactically valid for the given language."""
    return validate_code_syntax(code, file_ext)


def validate_edit_correctness(
    original_code: str,
    edited_code: str,
    expected_edits: list[EditInstruction],
) -> list[EditResult]:
    """Validate that edits were correctly applied by checking original/replacement presence."""
    results = []
    for edit in expected_edits:
        original_present = edit.original_text.strip() in edited_code
        replacement_present = edit.replacement_text.strip() in edited_code

        # Edit is correct if original is removed and replacement is added
        correct = not original_present and replacement_present
        # Edit is applied if replacement is present (even if original still there)
        applied = replacement_present

        results.append(EditResult(
            edit_id=edit.edit_id,
            target=edit.target,
            applied=applied,
            correct=correct,
        ))
    return results


def count_conflicts(edited_code: str, expected_edits: list[EditInstruction]) -> int:
    """Count conflicting edits (where original text still present after edit)."""
    conflicts = 0
    for edit in expected_edits:
        if edit.original_text.strip() in edited_code and edit.replacement_text.strip() in edited_code:
            conflicts += 1
    return conflicts
