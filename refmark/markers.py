"""Pluggable marker format definitions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class MarkerFormat:
    """Defines how markers are injected and stripped for a given format.

    Attributes:
        name: Unique identifier for this format (e.g. "bracket", "comment_py").
        inject_template: str.format template for injecting a marker.
            Must accept an ``id`` keyword, e.g. ``"[@B{id:02d}]"``.
        strip_regex: Compiled regular expression that matches injected markers.
        file_ext: File extension this format applies to, or "any" for universal formats.
    """

    name: str
    inject_template: str
    strip_regex: re.Pattern[str]
    file_ext: str  # ".py", ".md", or "any"


class MarkerFormatBuilder:
    """Fluent builder for MarkerFormat instances."""

    def __init__(self) -> None:
        self._name: str = ""
        self._inject_template: str = ""
        self._strip_pattern: str = ""
        self._file_ext: str = "any"

    def name(self, name: str) -> MarkerFormatBuilder:
        self._name = name
        return self

    def inject(self, template: str) -> MarkerFormatBuilder:
        self._inject_template = template
        return self

    def strip(self, pattern: str) -> MarkerFormatBuilder:
        self._strip_pattern = pattern
        return self

    def for_ext(self, ext: str) -> MarkerFormatBuilder:
        self._file_ext = ext
        return self

    def build(self) -> MarkerFormat:
        return MarkerFormat(
            name=self._name,
            inject_template=self._inject_template,
            strip_regex=re.compile(self._strip_pattern),
            file_ext=self._file_ext,
        )


class MarkerRegistry:
    """Global registry for marker formats."""

    _formats: dict[str, MarkerFormat] = {}
    _default: str | None = None

    @classmethod
    def register(cls, fmt: MarkerFormat, *, default: bool = False) -> None:
        """Register a marker format.

        Args:
            fmt: The MarkerFormat to register.
            default: If True, this format becomes the default for its file extension.
        """
        cls._formats[fmt.name] = fmt
        if default:
            cls._default = fmt.name

    @classmethod
    def get(cls, name: str) -> MarkerFormat:
        """Get a marker format by name.

        Raises:
            KeyError: If the format is not found.
        """
        return cls._formats[name]

    @classmethod
    def get_for_ext(cls, file_ext: str) -> MarkerFormat:
        """Get the best marker format for a file extension.

        Tries exact match first, then falls back to "any" formats.
        """
        # Try exact extension match
        for fmt in cls._formats.values():
            if fmt.file_ext == file_ext:
                return fmt
        # Fall back to first "any" format
        for fmt in cls._formats.values():
            if fmt.file_ext == "any":
                return fmt
        raise KeyError(f"No marker format found for extension: {file_ext}")

    @classmethod
    def list_all(cls) -> dict[str, MarkerFormat]:
        """Return all registered formats."""
        return dict(cls._formats)

    @classmethod
    def reset(cls) -> None:
        """Clear all registrations (useful for testing)."""
        cls._formats.clear()
        cls._default = None


def make_marker_format(
    name: str,
    inject_template: str,
    strip_pattern: str,
    file_ext: str = "any",
) -> MarkerFormat:
    """Convenience function to create a MarkerFormat without the builder."""
    return MarkerFormat(
        name=name,
        inject_template=inject_template,
        strip_regex=re.compile(strip_pattern),
        file_ext=file_ext,
    )


# ── Built-in marker formats ──────────────────────────────────────────────

# Universal formats (work with any file type)
BRACKET_FORMAT = make_marker_format(
    name="bracket",
    inject_template="[@B{id:02d}]",
    strip_pattern=r"\[@B\d+\] ?",
    file_ext="any",
)

TYPED_BRACKET_FORMAT = make_marker_format(
    name="typed_bracket",
    inject_template="[@{label}]",
    strip_pattern=r"\[@[A-Z]+\d+\] ?",
    file_ext="any",
)

HASH_FORMAT = make_marker_format(
    name="hash",
    inject_template="#B{id:02d}",
    strip_pattern=r"#B\d+ ?",
    file_ext="any",
)

# Phase 0 candidate formats
XML_FORMAT = make_marker_format(
    name="xml",
    inject_template='<block id="{id}"/>',
    strip_pattern=r'<block id="\d+"/> ?',
    file_ext="any",
)

TYPED_XML_FORMAT = make_marker_format(
    name="typed_xml",
    inject_template='<block id="{label}"/>',
    strip_pattern=r'<block id="[A-Z]+\d+"/> ?',
    file_ext="any",
)

CURLY_FORMAT = make_marker_format(
    name="curly",
    inject_template="{{B{id}}}",   # {{ → literal {, {id} → format field, }} → literal }
    strip_pattern=r"\{B\d+\} ?",
    file_ext="any",
)

# Language-specific formats
COMMENT_PY_FORMAT = make_marker_format(
    name="comment_py",
    inject_template="# @ref:{id:02d}",
    strip_pattern=r"(?m)^# @ref:\d+\n?",
    file_ext=".py",
)

TYPED_COMMENT_PY_FORMAT = make_marker_format(
    name="typed_comment_py",
    inject_template="# [@{label}]",
    strip_pattern=r"(?m)^# \[@[A-Z]+\d+\]\n?",
    file_ext=".py",
)

COMMENT_MD_FORMAT = make_marker_format(
    name="comment_md",
    inject_template="<!-- @ref:{id:02d} -->",
    strip_pattern=r"(?m)<!-- @ref:\d+ -->\n?",
    file_ext=".md",
)

COMMENT_TS_FORMAT = make_marker_format(
    name="comment_ts",
    inject_template="// @ref:{id:02d}",
    strip_pattern=r"(?m)^// @ref:\d+\n?",
    file_ext=".ts",
)

TYPED_COMMENT_TS_FORMAT = make_marker_format(
    name="typed_comment_ts",
    inject_template="// [@{label}]",
    strip_pattern=r"(?m)^// \[@[A-Z]+\d+\]\n?",
    file_ext=".ts",
)


def register_builtins() -> None:
    """Register all built-in marker formats."""
    MarkerRegistry.register(BRACKET_FORMAT)
    MarkerRegistry.register(TYPED_BRACKET_FORMAT)
    MarkerRegistry.register(HASH_FORMAT)
    MarkerRegistry.register(XML_FORMAT)
    MarkerRegistry.register(TYPED_XML_FORMAT)
    MarkerRegistry.register(CURLY_FORMAT)
    MarkerRegistry.register(COMMENT_PY_FORMAT, default=True)
    MarkerRegistry.register(TYPED_COMMENT_PY_FORMAT)
    MarkerRegistry.register(COMMENT_MD_FORMAT, default=True)
    MarkerRegistry.register(COMMENT_TS_FORMAT, default=True)
    MarkerRegistry.register(TYPED_COMMENT_TS_FORMAT)


# Auto-register builtins on import
register_builtins()
