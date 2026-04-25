"""Language registry and validation helpers for product-facing refmark flows."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable


def _validate_python(code: str) -> bool:
    try:
        compile(code, "<refmark>", "exec")
        return True
    except SyntaxError:
        return False


def _validate_typescript(code: str) -> bool:
    try:
        from tree_sitter import Language, Parser
        import tree_sitter_typescript as tst
    except ImportError:
        print(
            "Refmark warning: TypeScript syntax validation requested but tree-sitter "
            "dependencies are not installed. Run `pip install -e .[typescript]`.",
            file=sys.stderr,
        )
        return False

    parser = Parser(Language(tst.language_typescript()))
    tree = parser.parse(code.encode("utf-8"))
    root = tree.root_node
    return not root.has_error


@dataclass(frozen=True)
class LanguageSpec:
    name: str
    extensions: tuple[str, ...]
    default_marker_format: str
    live_marker_format: str
    default_chunker: str
    edit_chunker: str
    validator: Callable[[str], bool] | None = None


_LANGUAGES: tuple[LanguageSpec, ...] = (
    LanguageSpec(
        name="python",
        extensions=(".py",),
        default_marker_format="comment_py",
        live_marker_format="typed_comment_py",
        default_chunker="line",
        edit_chunker="hybrid",
        validator=_validate_python,
    ),
    LanguageSpec(
        name="typescript",
        extensions=(".ts",),
        default_marker_format="comment_ts",
        live_marker_format="typed_comment_ts",
        default_chunker="ts_ast",
        edit_chunker="ts_hybrid",
        validator=_validate_typescript,
    ),
)


def get_language_spec(file_ext: str) -> LanguageSpec | None:
    ext = file_ext.lower()
    for spec in _LANGUAGES:
        if ext in spec.extensions:
            return spec
    return None


def choose_default_chunker(file_ext: str) -> str:
    spec = get_language_spec(file_ext)
    if spec is not None:
        return spec.default_chunker
    if file_ext == ".md":
        return "paragraph"
    return "line"


def choose_edit_chunker(file_ext: str) -> str:
    spec = get_language_spec(file_ext)
    if spec is not None:
        return spec.edit_chunker
    if file_ext == ".md":
        return "paragraph"
    return "line"


def choose_default_marker_format(file_ext: str) -> str | None:
    spec = get_language_spec(file_ext)
    if spec is not None:
        return spec.default_marker_format
    return None


def choose_live_marker_format(file_ext: str) -> str:
    spec = get_language_spec(file_ext)
    if spec is not None:
        return spec.live_marker_format
    return "typed_bracket"


def validate_code_syntax(code: str, file_ext: str) -> bool:
    spec = get_language_spec(file_ext)
    if spec is None or spec.validator is None:
        return True
    return spec.validator(code)


def list_supported_languages() -> list[dict[str, str]]:
    return [
        {
            "name": spec.name,
            "extensions": ", ".join(spec.extensions),
            "default_chunker": spec.default_chunker,
            "edit_chunker": spec.edit_chunker,
            "marker_format": spec.live_marker_format,
        }
        for spec in _LANGUAGES
    ]
