"""Core injection and stripping logic with pluggable marker formats and chunkers."""

from __future__ import annotations

import re
from typing import Tuple

from refmark.markers import MarkerFormat, MarkerRegistry
from refmark.chunkers import (
    Chunk,
    Chunker,
    get_chunker,
    LineChunker,
    ParagraphChunker,
    marker_label_for_chunk,
)
from refmark.languages import choose_default_chunker, choose_default_marker_format


def inject(
    content: str,
    file_ext: str,
    *,
    marker_format: str | None = None,
    chunker: str | Chunker | None = None,
    chunker_kwargs: dict | None = None,
) -> Tuple[str, int]:
    """Inject reference markers into content.

    Args:
        content: The text content to process.
        file_ext: File extension (e.g. ".py", ".md") for format selection.
        marker_format: Name of the marker format to use.
            If None, selects the default for the file extension.
        chunker: Chunker name or Chunker instance.
            If None, selects the default for the file extension.
        chunker_kwargs: Additional keyword arguments for the chunker constructor.

    Returns:
        Tuple of (processed content, number of markers injected).

    Raises:
        ValueError: If the file extension is unsupported.
    """
    # Resolve marker format
    if marker_format is not None:
        fmt = MarkerRegistry.get(marker_format)
    else:
        language_default = choose_default_marker_format(file_ext)
        fmt = MarkerRegistry.get(language_default) if language_default else MarkerRegistry.get_for_ext(file_ext)

    # Resolve chunker
    if chunker is None:
        chunker = _default_chunker_for_ext(file_ext, **(chunker_kwargs or {}))
    elif isinstance(chunker, str):
        chunker = get_chunker(chunker, **(chunker_kwargs or {}))

    # Route to format-specific injection
    if file_ext == ".py":
        return _inject_with_format(content, fmt, chunker)
    elif file_ext == ".ts":
        return _inject_code_with_format(content, fmt, chunker, comment_prefixes=("//",))
    elif file_ext == ".md":
        return _inject_md_with_format(content, fmt, chunker)
    else:
        # For unknown extensions, try generic injection
        return _inject_generic_with_format(content, fmt, chunker)


def strip(
    content: str,
    file_ext: str,
    *,
    marker_format: str | None = None,
) -> str:
    """Remove reference markers from content.

    Args:
        content: The text content to process.
        file_ext: File extension (e.g. ".py", ".md").
        marker_format: Name of the marker format to strip.
            If None, strips all known marker formats.

    Returns:
        Content with markers removed.

    Raises:
        ValueError: If the file extension is unsupported.
    """
    if marker_format is not None:
        fmt = MarkerRegistry.get(marker_format)
        return fmt.strip_regex.sub("", content)

    # Strip all known formats
    for fmt in MarkerRegistry.list_all().values():
        if fmt.file_ext == file_ext or fmt.file_ext == "any":
            content = fmt.strip_regex.sub("", content)

    return content


# ── Internal helpers ─────────────────────────────────────────────────────

def _default_chunker_for_ext(file_ext: str, **kwargs) -> Chunker:
    """Return the default chunker for a file extension."""
    chunker_name = choose_default_chunker(file_ext)
    if chunker_name == "paragraph":
        return ParagraphChunker()
    return get_chunker(chunker_name, **kwargs)


def _inject_generic_with_format(
    content: str,
    fmt: MarkerFormat,
    chunker: Chunker,
) -> Tuple[str, int]:
    """Inject markers using a generic line-based approach with a specific format."""
    chunks = chunker.chunk(content)
    if not chunks:
        return content, 0

    lines = content.splitlines(keepends=True)
    result = []
    ref_count = 0
    current_line = 0

    for chunk in chunks:
        # Add lines before this chunk
        while current_line < chunk.start_line - 1:
            result.append(lines[current_line])
            current_line += 1

        # Inject marker at chunk start
        ref_count += 1
        marker = _render_marker(fmt, ref_count, chunk)
        # Prepend marker to the first line of the chunk (no extra newline)
        if current_line < len(lines):
            result.append(marker + " " + lines[current_line])
            current_line += 1
        else:
            result.append(marker + "\n")

        # Add remaining chunk lines
        while current_line < chunk.end_line:
            result.append(lines[current_line])
            current_line += 1

    # Add remaining lines
    while current_line < len(lines):
        result.append(lines[current_line])
        current_line += 1

    return "".join(result), ref_count


def _inject_with_format(
    content: str,
    fmt: MarkerFormat,
    chunker: Chunker,
) -> Tuple[str, int]:
    """Inject markers into Python code, respecting docstrings and comments."""
    chunks = chunker.chunk(content)
    if not chunks:
        return content, 0

    lines = content.splitlines(keepends=True)
    result: list[str] = []
    ref_count = 0
    in_docstring = False
    docstring_char: str | None = None

    chunk_by_start_line = {chunk.start_line: chunk for chunk in chunks}

    for i, line in enumerate(lines):
        line_num = i + 1
        stripped = line.strip()

        # Toggle docstring state
        if not in_docstring:
            if '"""' in stripped or "'''" in stripped:
                quote = '"""' if '"""' in stripped else "'''"
                if stripped.count(quote) % 2 == 1:
                    in_docstring = True
                    docstring_char = quote
        else:
            if docstring_char and docstring_char in stripped:
                in_docstring = False
                docstring_char = None

        # Inject marker at chunk start (if not in docstring or comment)
        if line_num in chunk_by_start_line and not in_docstring and not stripped.startswith('#'):
            ref_count += 1
            marker = _render_marker(fmt, ref_count, chunk_by_start_line[line_num])
            # Comment-style markers must live on their own line to keep the code valid.
            if fmt.name in {"comment_py", "typed_comment_py"}:
                result.append(marker + "\n")
                result.append(line)
            else:
                # Prepend marker to the line (no extra newline)
                result.append(marker + " " + line)
        else:
            result.append(line)

    return "".join(result), ref_count


def _inject_md_with_format(
    content: str,
    fmt: MarkerFormat,
    chunker: Chunker,
) -> Tuple[str, int]:
    """Inject markers into Markdown, respecting code blocks and blockquotes."""
    chunks = chunker.chunk(content)
    if not chunks:
        return content, 0

    lines = content.splitlines(keepends=True)
    result: list[str] = []
    ref_count = 0

    chunk_by_start_line = {chunk.start_line: chunk for chunk in chunks}

    for i, line in enumerate(lines):
        line_num = i + 1

        # Inject marker at chunk start
        if line_num in chunk_by_start_line:
            ref_count += 1
            marker = _render_marker(fmt, ref_count, chunk_by_start_line[line_num])
            # Prepend marker to the line (no extra newline)
            result.append(marker + " " + line)
        else:
            result.append(line)

    return "".join(result), ref_count


def _inject_code_with_format(
    content: str,
    fmt: MarkerFormat,
    chunker: Chunker,
    *,
    comment_prefixes: tuple[str, ...],
) -> Tuple[str, int]:
    """Inject markers into comment-oriented code files such as TypeScript."""
    chunks = chunker.chunk(content)
    if not chunks:
        return content, 0

    lines = content.splitlines(keepends=True)
    result: list[str] = []
    ref_count = 0
    chunk_by_start_line = {chunk.start_line: chunk for chunk in chunks}

    for i, line in enumerate(lines):
        line_num = i + 1
        stripped = line.strip()
        if line_num in chunk_by_start_line and not stripped.startswith(comment_prefixes):
            ref_count += 1
            marker = _render_marker(fmt, ref_count, chunk_by_start_line[line_num])
            if fmt.inject_template.lstrip().startswith(("#", "//")):
                result.append(marker + "\n")
                result.append(line)
            else:
                result.append(marker + " " + line)
        else:
            result.append(line)

    return "".join(result), ref_count


def _render_marker(fmt: MarkerFormat, ref_count: int, chunk: Chunk) -> str:
    label_style = "typed" if fmt.name.startswith("typed_") else "numeric"
    label = marker_label_for_chunk(chunk, style=label_style)
    return fmt.inject_template.format(id=ref_count, label=label)


# ── Legacy compatibility (original API) ──────────────────────────────────
# These functions preserve the original behavior for backward compatibility.

def inject_legacy(content: str, file_ext: str) -> Tuple[str, int]:
    """Original inject implementation (backward compatible)."""
    if file_ext == ".py":
        return _inject_py_legacy(content)
    elif file_ext == ".md":
        return _inject_md_legacy(content)
    else:
        raise ValueError(f"Unsupported file extension: {file_ext}")


def strip_legacy(content: str, file_ext: str) -> str:
    """Original strip implementation (backward compatible)."""
    if file_ext == ".py":
        return _strip_py_legacy(content)
    elif file_ext == ".md":
        return _strip_md_legacy(content)
    else:
        raise ValueError(f"Unsupported file extension: {file_ext}")


def _inject_py_legacy(content: str) -> Tuple[str, int]:
    lines = content.splitlines(keepends=True)
    result = []
    ref_count = 0
    line_counter = 0
    in_docstring = False
    docstring_char = None

    for line in lines:
        stripped = line.strip()

        if not in_docstring:
            if '"""' in stripped or "'''" in stripped:
                quote = '"""' if '"""' in stripped else "'''"
                if stripped.count(quote) % 2 == 1:
                    in_docstring = True
                    docstring_char = quote
        else:
            if docstring_char and docstring_char in stripped:
                in_docstring = False
                docstring_char = None

        if in_docstring or stripped.startswith('#'):
            result.append(line)
            continue

        line_counter += 1
        if line_counter % 10 == 1:
            ref_count += 1
            result.append(f"# @ref:{ref_count:02d}\n")
        result.append(line)

    return "".join(result), ref_count


def _strip_py_legacy(content: str) -> str:
    return re.sub(r'^# @ref:\d+\n?', '', content, flags=re.MULTILINE)


def _inject_md_legacy(content: str) -> Tuple[str, int]:
    lines = content.splitlines(keepends=True)
    result = []
    ref_count = 0
    in_code_block = False
    paragraph_start = True

    for line in lines:
        stripped = line.strip()

        if stripped.startswith('```'):
            in_code_block = not in_code_block
            paragraph_start = False
            result.append(line)
            continue

        if in_code_block or stripped.startswith('> ') or stripped.startswith('<!--'):
            paragraph_start = False
            result.append(line)
            continue

        if stripped == '':
            paragraph_start = True
            result.append(line)
            continue

        if paragraph_start:
            ref_count += 1
            result.append(f"<!-- @ref:{ref_count:02d} -->\n")
            paragraph_start = False

        result.append(line)

    return "".join(result), ref_count


def _strip_md_legacy(content: str) -> str:
    return re.sub(r'<!-- @ref:\d+ -->\n?', '', content)
