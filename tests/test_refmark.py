import importlib.util

import pytest
from refmark.core import inject, strip, inject_legacy, strip_legacy
from refmark.edit import _detect_premarked
from refmark.markers import MarkerRegistry, BRACKET_FORMAT, HASH_FORMAT
from refmark.chunkers import LineChunker, ParagraphChunker, FixedTokenChunker, ASTChunker, get_chunker
from refmark.regions import apply_refmark_edits, validate_syntax
from refmark.languages import (
    choose_default_chunker,
    choose_default_marker_format,
    choose_edit_chunker,
    choose_live_marker_format,
    validate_code_syntax,
)


# ── Legacy API tests (backward compatibility) ────────────────────────────

def test_inject_strip_py():
    original = "def foo():\n    pass\n"
    injected, count = inject_legacy(original, ".py")
    assert count == 1
    assert "# @ref:01" in injected
    stripped = strip_legacy(injected, ".py")
    assert stripped == original

def test_inject_strip_md():
    original = "Paragraph one.\n\nParagraph two.\n"
    injected, count = inject_legacy(original, ".md")
    assert count == 2
    assert "<!-- @ref:01 -->" in injected
    assert "<!-- @ref:02 -->" in injected
    stripped = strip_legacy(injected, ".md")
    assert stripped == original

def test_inject_strip_py_exact_lines():
    lines = [f"line_{i}\n" for i in range(25)]
    original = "".join(lines)
    injected, count = inject_legacy(original, ".py")
    assert count == 3  # Injects at lines 1, 11, 21
    stripped = strip_legacy(injected, ".py")
    assert stripped == original

def test_inject_strip_md_single_para():
    original = "Just one paragraph."
    injected, count = inject_legacy(original, ".md")
    assert count == 1
    stripped = strip_legacy(injected, ".md")
    assert stripped == original

def test_inject_unsupported_ext():
    with pytest.raises(ValueError):
        inject_legacy("test", ".txt")

def test_strip_unsupported_ext():
    with pytest.raises(ValueError):
        strip_legacy("test", ".txt")

def test_skip_py_docstrings_and_comments():
    code = '''def foo():
    """This is a docstring."""
    # This is a comment
    pass
'''
    injected, count = inject_legacy(code, ".py")
    # Should only inject at the very first line
    assert count == 1
    assert "# @ref:01" in injected

def test_skip_md_code_blocks_and_blockquotes():
    md = """Text before.

```python
code here
```

> Blockquote here.

Text after.
"""
    injected, count = inject_legacy(md, ".md")
    # Only "Text before." and "Text after." should get markers
    assert count == 2
    assert "<!-- @ref:01 -->" in injected
    assert "<!-- @ref:02 -->" in injected
    # Verify markers are NOT inside code block or blockquote
    assert injected.count("<!-- @ref:") == 2

def test_roundtrip_empty_py():
    original = ""
    injected, count = inject_legacy(original, ".py")
    assert count == 0
    assert strip_legacy(injected, ".py") == original

def test_roundtrip_empty_md():
    original = ""
    injected, count = inject_legacy(original, ".md")
    assert count == 0
    assert strip_legacy(injected, ".md") == original


# ── New API tests (pluggable formats + chunkers) ─────────────────────────

class TestNewAPI:
    """Tests for the new pluggable marker format and chunker API."""

    def test_inject_bracket_format_py(self):
        """Test bracket format injection for Python files."""
        original = "def foo():\n    pass\n"
        injected, count = inject(original, ".py", marker_format="bracket")
        assert count >= 1
        assert "[@B01]" in injected
        stripped = strip(injected, ".py", marker_format="bracket")
        assert stripped == original

    def test_inject_bracket_format_md(self):
        """Test bracket format injection for Markdown files."""
        original = "Paragraph one.\n\nParagraph two.\n"
        injected, count = inject(original, ".md", marker_format="bracket")
        assert count >= 1
        assert "[@B" in injected
        stripped = strip(injected, ".md", marker_format="bracket")
        assert stripped == original

    def test_inject_hash_format(self):
        """Test hash format injection."""
        original = "def foo():\n    pass\n"
        injected, count = inject(original, ".py", marker_format="hash")
        assert count >= 1
        assert "#B01" in injected
        stripped = strip(injected, ".py", marker_format="hash")
        assert stripped == original

    def test_inject_default_format(self):
        """Test that default format is selected when marker_format is None."""
        original = "def foo():\n    pass\n"
        injected, count = inject(original, ".py")
        # Default for .py is comment_py
        assert "# @ref:01" in injected or "[@B01]" in injected or "#B01" in injected

    def test_strip_all_formats(self):
        """Test that strip without marker_format removes all known formats."""
        original = "def foo():\n    pass\n"
        injected_bracket, _ = inject(original, ".py", marker_format="bracket")
        injected_hash, _ = inject(original, ".py", marker_format="hash")
        # Strip all formats
        stripped_bracket = strip(injected_bracket, ".py")
        stripped_hash = strip(injected_hash, ".py")
        assert "[@B" not in stripped_bracket
        assert "#B" not in stripped_hash

    def test_chunker_line_chunker(self):
        """Test LineChunker produces correct chunks."""
        content = "\n".join(f"line_{i}" for i in range(25))
        chunker = LineChunker(lines_per_chunk=10)
        chunks = chunker.chunk(content)
        assert len(chunks) == 3
        assert chunks[0].id == 1
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 10
        assert chunks[1].id == 2
        assert chunks[1].start_line == 11

    def test_chunker_paragraph_chunker(self):
        """Test ParagraphChunker produces correct chunks."""
        content = "Para one.\n\nPara two.\n\nPara three.\n"
        chunker = ParagraphChunker()
        chunks = chunker.chunk(content)
        assert len(chunks) == 3
        assert "Para one" in chunks[0].text
        assert "Para two" in chunks[1].text

    def test_chunker_fixed_token(self):
        """Test FixedTokenChunker produces chunks."""
        content = " ".join(f"word{i}" for i in range(100))
        chunker = FixedTokenChunker(tokens_per_chunk=50)
        chunks = chunker.chunk(content)
        assert len(chunks) >= 1

    def test_chunker_ast_python(self):
        """Test ASTChunker for Python code."""
        code = """def foo():
    pass

class Bar:
    pass

def baz():
    pass
"""
        chunker = ASTChunker()
        chunks = chunker.chunk(code)
        assert len(chunks) >= 2  # At least foo and Bar

    def test_chunker_ast_includes_top_level_statements(self):
        """AST chunking should preserve imports, constants, and main guards."""
        code = """import math

PI = 3.14

def area(radius):
    return PI * radius * radius

if __name__ == "__main__":
    print(area(2))
"""
        chunker = ASTChunker()
        chunks = chunker.chunk(code)
        texts = [chunk.text for chunk in chunks]
        assert any("import math" in text for text in texts)
        assert any("PI = 3.14" in text for text in texts)
        assert any("if __name__ ==" in text for text in texts)

    def test_chunker_ast_fallback(self):
        """Test ASTChunker falls back to LineChunker on invalid syntax."""
        code = "def foo(:\n    pass\n"  # Invalid Python
        chunker = ASTChunker()
        chunks = chunker.chunk(code)
        assert len(chunks) >= 1  # Should not crash

    def test_get_chunker(self):
        """Test get_chunker factory function."""
        c = get_chunker("line", lines_per_chunk=5)
        assert isinstance(c, LineChunker)
        assert c.lines_per_chunk == 5

    def test_get_chunker_unknown(self):
        """Test get_chunker raises KeyError for unknown chunker."""
        with pytest.raises(KeyError):
            get_chunker("nonexistent")

    def test_inject_with_custom_chunker(self):
        """Test inject with a custom chunker instance."""
        original = "\n".join(f"line_{i}" for i in range(25))
        chunker = LineChunker(lines_per_chunk=5)
        injected, count = inject(original, ".py", chunker=chunker)
        assert count >= 4  # 25 lines / 5 = 5 chunks

    def test_inject_with_chunker_name(self):
        """Test inject with chunker specified by name."""
        original = "\n".join(f"line_{i}" for i in range(25))
        injected, count = inject(original, ".py", chunker="line", chunker_kwargs={"lines_per_chunk": 5})
        assert count >= 4

    def test_inject_with_ast_chunker_marks_module_regions(self):
        """AST chunker should mark more than just function bodies."""
        code = """import math

PI = 3.14

def area(radius):
    return PI * radius * radius

if __name__ == "__main__":
    print(area(2))
"""
        injected, count = inject(code, ".py", marker_format="bracket", chunker="ast")
        assert count == 4
        assert "[@B01] import math" in injected
        assert "[@B04] if __name__ == \"__main__\":" in injected

    def test_roundtrip_bracket_format(self):
        """Test roundtrip with bracket format."""
        original = "def foo():\n    pass\n"
        injected, _ = inject(original, ".py", marker_format="bracket")
        stripped = strip(injected, ".py", marker_format="bracket")
        assert stripped == original

    def test_roundtrip_hash_format(self):
        """Test roundtrip with hash format."""
        original = "def foo():\n    pass\n"
        injected, _ = inject(original, ".py", marker_format="hash")
        stripped = strip(injected, ".py", marker_format="hash")
        assert stripped == original

    def test_marker_registry_builtins(self):
        """Test that built-in formats are registered."""
        formats = MarkerRegistry.list_all()
        assert "bracket" in formats
        assert "typed_explicit" in formats
        assert "typed_compact" in formats
        assert "hash" in formats
        assert "comment_py" in formats
        assert "comment_md" in formats

    def test_document_marker_styles_roundtrip(self):
        original = "Alpha paragraph.\n\nBeta paragraph.\n"
        explicit, _ = inject(original, ".txt", marker_format="typed_explicit", chunker="paragraph")
        compact, _ = inject(original, ".txt", marker_format="typed_compact", chunker="paragraph")

        assert "[ref:P01]" in explicit
        assert "[P01]" in compact
        assert strip(explicit, ".txt", marker_format="typed_explicit") == original
        assert strip(compact, ".txt", marker_format="typed_compact") == original

    def test_detect_premarked_ignores_marker_examples_in_strings(self):
        code = 'EXAMPLE = "Use [@P01] to cite a paragraph in prompts."\n'

        assert not _detect_premarked(code, marker_format="typed_bracket")

    def test_marker_registry_get_for_ext(self):
        """Test MarkerRegistry.get_for_ext selects correct format."""
        fmt = MarkerRegistry.get_for_ext(".py")
        assert fmt.file_ext == ".py" or fmt.file_ext == "any"

    def test_marker_registry_reset(self):
        """Test MarkerRegistry.reset clears all formats."""
        MarkerRegistry.reset()
        assert len(MarkerRegistry.list_all()) == 0
        # Re-register builtins for other tests
        from refmark.markers import register_builtins
        register_builtins()


class TestDeterministicRefmarkEdits:
    def test_apply_refmark_edits_targets_correct_duplicate_block(self):
        """Block application should use block ranges, not first matching text."""
        code = """def alpha():
    value = "same"
    return value


def beta():
    value = "same"
    return value
"""
        marked, _ = inject(code, ".py", marker_format="bracket", chunker="ast")
        edited, results = apply_refmark_edits(
            original_code=code,
            marked_content=marked,
            parsed_edits=[
                {
                    "block_id": 2,
                    "new_content": """def beta():
    value = "changed"
    return value
""",
                }
            ],
            marker_format="bracket",
        )
        assert all(result.applied for result in results)
        assert 'def alpha():\n    value = "same"' in edited
        assert 'def beta():\n    value = "changed"' in edited
        assert edited.count('value = "same"') == 1

    def test_apply_refmark_edits_supports_multiple_ast_regions(self):
        """Multiple same-file region edits should still produce valid Python."""
        code = """import math

PI = 3.14

def area(radius):
    return PI * radius * radius

def circumference(radius):
    return 2 * PI * radius

if __name__ == "__main__":
    print(area(2))
"""
        marked, _ = inject(code, ".py", marker_format="bracket", chunker="ast")
        edited, results = apply_refmark_edits(
            original_code=code,
            marked_content=marked,
            parsed_edits=[
                {
                    "block_id": 3,
                    "new_content": """def area(radius):
    return round(PI * radius * radius, 2)
""",
                },
                {
                    "block_id": 4,
                    "new_content": """def circumference(radius):
    return round(2 * PI * radius, 2)
""",
                },
            ],
            marker_format="bracket",
        )
        assert all(result.applied for result in results)
        assert "round(PI * radius * radius, 2)" in edited
        assert "round(2 * PI * radius, 2)" in edited
        assert validate_syntax(edited)


@pytest.mark.skipif(
    importlib.util.find_spec("tree_sitter") is None
    or importlib.util.find_spec("tree_sitter_typescript") is None,
    reason="TypeScript validation tests require the typescript extra",
)
class TestTypeScriptSupport:
    def test_inject_strip_ts_roundtrip_uses_comment_markers(self):
        original = """export function greet(name: string): string {
  return `Hello ${name}`;
}
"""
        injected, count = inject(original, ".ts")
        assert count >= 1
        assert "// @ref:01" in injected
        stripped = strip(injected, ".ts", marker_format="comment_ts")
        assert stripped == original

    def test_typescript_live_markers_roundtrip(self):
        original = """export class UserService {
  format(name: string): string {
    return name.trim();
  }
}
"""
        injected, count = inject(original, ".ts", marker_format="typed_comment_ts", chunker="ts_hybrid")
        assert count >= 1
        assert "// [@" in injected
        stripped = strip(injected, ".ts", marker_format="typed_comment_ts")
        assert stripped == original

    def test_typescript_refmark_edits_apply_and_validate(self):
        code = """export class UserService {
  format(name: string): string {
    return name.trim();
  }
}
"""
        marked, _ = inject(code, ".ts", marker_format="typed_comment_ts", chunker="ts_hybrid")
        edited, results = apply_refmark_edits(
            original_code=code,
            marked_content=marked,
            parsed_edits=[
                {
                    "region_id": "M02",
                    "new_content": """  format(name: string): string {
    return name.trim().toUpperCase();
  }
""",
                }
            ],
            marker_format="typed_comment_ts",
        )
        assert all(result.applied for result in results)
        assert "toUpperCase" in edited
        assert validate_syntax(edited, ".ts")

    def test_language_defaults_are_language_specific(self):
        assert choose_default_marker_format(".py") == "comment_py"
        assert choose_default_marker_format(".ts") == "comment_ts"
        assert choose_live_marker_format(".ts") == "typed_comment_ts"
        assert choose_default_chunker(".ts") == "ts_ast"
        assert choose_edit_chunker(".ts") == "ts_hybrid"

    def test_typescript_syntax_validation_rejects_broken_code(self):
        assert validate_code_syntax("export const value = 1;\n", ".ts")
        assert not validate_code_syntax("export const value = ;\n", ".ts")
