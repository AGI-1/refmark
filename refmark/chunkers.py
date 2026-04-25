"""Chunking strategy implementations."""

from __future__ import annotations

import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass

try:
    from tree_sitter import Language, Parser
    import tree_sitter_typescript as _tree_sitter_typescript
except ImportError:  # pragma: no cover - optional dependency at runtime
    Language = None
    Parser = None
    _tree_sitter_typescript = None

_WARNED_TYPESCRIPT_DEPS = False


def _warn_missing_typescript_deps() -> None:
    global _WARNED_TYPESCRIPT_DEPS
    if _WARNED_TYPESCRIPT_DEPS:
        return
    _WARNED_TYPESCRIPT_DEPS = True
    print(
        "Refmark warning: TypeScript AST chunking requested but tree-sitter dependencies "
        "are not installed. Falling back to line chunking. Run `pip install -e .[typescript]`.",
        file=sys.stderr,
    )


@dataclass(frozen=True)
class Chunk:
    """A single chunk of content with its position information.

    Attributes:
        id: 1-based chunk identifier.
        start_line: 1-based line number where this chunk starts.
        end_line: 1-based line number where this chunk ends.
        text: The chunk text content.
    """

    id: int
    start_line: int
    end_line: int
    text: str
    kind: str = "chunk"
    name: str = ""


class Chunker(ABC):
    """Base class for chunking strategies."""

    @abstractmethod
    def chunk(self, content: str) -> list[Chunk]:
        """Split content into chunks.

        Args:
            content: The full text content to chunk.

        Returns:
            List of Chunk objects with id, position, and text.
        """
        ...


class LineChunker(Chunker):
    """Chunk by fixed number of lines.

    This is the default strategy for code files.
    """

    def __init__(self, lines_per_chunk: int = 10) -> None:
        self.lines_per_chunk = lines_per_chunk

    def chunk(self, content: str) -> list[Chunk]:
        if not content:
            return []

        lines = content.splitlines(keepends=True)
        chunks: list[Chunk] = []
        chunk_id = 0

        for i in range(0, len(lines), self.lines_per_chunk):
            chunk_lines = lines[i : i + self.lines_per_chunk]
            chunk_id += 1
            chunks.append(
                Chunk(
                    id=chunk_id,
                    start_line=i + 1,
                    end_line=i + len(chunk_lines),
                    text="".join(chunk_lines),
                )
            )

        return chunks


class ParagraphChunker(Chunker):
    """Chunk by paragraph boundaries.

    This is the default strategy for markdown/text files.
    Skips fenced code blocks, blockquotes, and existing comments.
    """

    def chunk(self, content: str) -> list[Chunk]:
        if not content:
            return []

        lines = content.splitlines(keepends=True)
        chunks: list[Chunk] = []
        chunk_id = 0
        in_code_block = False
        paragraph_lines: list[str] = []
        paragraph_start_line = 0

        for i, line in enumerate(lines, start=1):
            stripped = line.strip()

            # Toggle code block state
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                # Flush any pending paragraph
                if paragraph_lines:
                    chunk_id += 1
                    chunks.append(
                        Chunk(
                            id=chunk_id,
                            start_line=paragraph_start_line,
                            end_line=i - 1,
                            text="".join(paragraph_lines),
                        )
                    )
                    paragraph_lines.clear()
                continue

            # Skip code blocks, blockquotes, and HTML comments
            if in_code_block or stripped.startswith("> ") or stripped.startswith("<!--"):
                if paragraph_lines:
                    chunk_id += 1
                    chunks.append(
                        Chunk(
                            id=chunk_id,
                            start_line=paragraph_start_line,
                            end_line=i - 1,
                            text="".join(paragraph_lines),
                        )
                    )
                    paragraph_lines.clear()
                continue

            # Empty line = paragraph boundary
            if stripped == "":
                if paragraph_lines:
                    chunk_id += 1
                    chunks.append(
                        Chunk(
                            id=chunk_id,
                            start_line=paragraph_start_line,
                            end_line=i - 1,
                            text="".join(paragraph_lines),
                        )
                    )
                    paragraph_lines.clear()
                continue

            # Content line
            if not paragraph_lines:
                paragraph_start_line = i
            paragraph_lines.append(line)

        # Flush remaining paragraph
        if paragraph_lines:
            chunk_id += 1
            chunks.append(
                Chunk(
                    id=chunk_id,
                    start_line=paragraph_start_line,
                    end_line=len(lines),
                    text="".join(paragraph_lines),
                )
            )

        return chunks


class FixedTokenChunker(Chunker):
    """Chunk by approximate token count.

    Uses a simple word-based token estimation (1 word ≈ 1.3 tokens).
    """

    def __init__(self, tokens_per_chunk: int = 200) -> None:
        self.tokens_per_chunk = tokens_per_chunk
        # Rough estimate: 1 word ≈ 1.3 tokens for English
        self.words_per_chunk = max(1, int(tokens_per_chunk / 1.3))

    def _count_tokens(self, text: str) -> int:
        """Rough token count using word splitting."""
        words = text.split()
        return int(len(words) * 1.3)

    def chunk(self, content: str) -> list[Chunk]:
        if not content:
            return []

        lines = content.splitlines(keepends=True)
        chunks: list[Chunk] = []
        chunk_id = 0
        current_lines: list[str] = []
        current_tokens = 0
        chunk_start_line = 1

        for i, line in enumerate(lines, start=1):
            line_tokens = self._count_tokens(line)

            if current_tokens + line_tokens > self.tokens_per_chunk and current_lines:
                # Flush current chunk
                chunk_id += 1
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        start_line=chunk_start_line,
                        end_line=i - 1,
                        text="".join(current_lines),
                    )
                )
                current_lines = []
                current_tokens = 0
                chunk_start_line = i

            current_lines.append(line)
            current_tokens += line_tokens

        # Flush remaining
        if current_lines:
            chunk_id += 1
            chunks.append(
                Chunk(
                    id=chunk_id,
                    start_line=chunk_start_line,
                    end_line=len(lines),
                    text="".join(current_lines),
                )
            )

        return chunks


class ASTChunker(Chunker):
    """Chunk Python code by top-level AST statement boundaries.

    This is intentionally broader than only functions/classes because a coding
    workflow often needs stable regions for imports, constants, main guards,
    and other module-level statements as well.
    """

    def chunk(self, content: str) -> list[Chunk]:
        return _chunk_python_top_level(content)


class HybridChunker(Chunker):
    """Chunk Python into stable edit regions for multidiff workflows.

    This mode keeps top-level statements granular while splitting class bodies
    into class prelude plus whole-method regions.
    """

    def chunk(self, content: str) -> list[Chunk]:
        return _chunk_python_hybrid(content)


class TypeScriptASTChunker(Chunker):
    """Chunk TypeScript code by top-level syntax nodes."""

    def chunk(self, content: str) -> list[Chunk]:
        return _chunk_typescript_top_level(content)


class TypeScriptHybridChunker(Chunker):
    """Chunk TypeScript into stable top-level and class-method regions."""

    def chunk(self, content: str) -> list[Chunk]:
        return _chunk_typescript_hybrid(content)


def _parse_python(content: str):
    import ast

    try:
        return ast.parse(content)
    except SyntaxError:
        return None


def _parse_typescript(content: str):
    if not content:
        return None
    if Parser is None or Language is None or _tree_sitter_typescript is None:
        _warn_missing_typescript_deps()
        return None
    parser = Parser(Language(_tree_sitter_typescript.language_typescript()))
    tree = parser.parse(content.encode("utf-8"))
    if tree.root_node.has_error:
        return None
    return tree


def _chunk_python_top_level(content: str) -> list[Chunk]:
        if not content:
            return []

        tree = _parse_python(content)
        if tree is None:
            # Fall back to line chunking if code doesn't parse
            return LineChunker().chunk(content)

        lines = content.splitlines(keepends=True)
        chunks: list[Chunk] = []
        chunk_id = 0

        boundaries: list[tuple[int, int, str, str]] = []
        for node in getattr(tree, "body", []):
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None) or start
            if start is None or end is None:
                continue
            boundaries.append((start, end, _python_node_kind(node), getattr(node, "name", "")))

        # Sort by start line
        boundaries.sort(key=lambda item: item[0])

        # Create chunks for each top-level statement and preserve any
        # interstitial comments or whitespace as their own chunks.
        prev_end = 0
        for start, end, kind, name in boundaries:
            # Chunk any code between previous statement and this one.
            if start > prev_end + 1:
                inter_lines = lines[prev_end : start - 1]
                if inter_lines and any(l.strip() for l in inter_lines):
                    chunk_id += 1
                    chunks.append(
                        Chunk(
                            id=chunk_id,
                            start_line=prev_end + 1,
                            end_line=start - 1,
                            text="".join(inter_lines),
                            kind="interstitial",
                        )
                    )

            # Chunk the AST statement itself.
            def_lines = lines[start - 1 : end]
            chunk_id += 1
            chunks.append(
                Chunk(
                    id=chunk_id,
                    start_line=start,
                    end_line=end,
                    text="".join(def_lines),
                    kind=kind,
                    name=name,
                )
            )
            prev_end = end

        # Chunk remaining code after last definition
        if prev_end < len(lines):
            remaining = lines[prev_end:]
            if remaining and any(l.strip() for l in remaining):
                chunk_id += 1
                chunks.append(
                        Chunk(
                            id=chunk_id,
                            start_line=prev_end + 1,
                            end_line=len(lines),
                            text="".join(remaining),
                            kind="interstitial",
                        )
                    )

        # If no AST chunks were created, fall back to line chunking
        if not chunks:
            return LineChunker().chunk(content)

        return chunks


def _chunk_typescript_top_level(content: str) -> list[Chunk]:
    if not content:
        return []

    tree = _parse_typescript(content)
    if tree is None:
        return LineChunker().chunk(content)

    lines = content.splitlines(keepends=True)
    chunks: list[Chunk] = []
    chunk_id = 0
    prev_end = 0

    for raw_node in tree.root_node.named_children:
        node = _typescript_effective_node(raw_node)
        start = raw_node.start_point.row + 1
        end = raw_node.end_point.row + 1

        if start > prev_end + 1:
            inter_lines = lines[prev_end : start - 1]
            if inter_lines and any(line.strip() for line in inter_lines):
                chunk_id += 1
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        start_line=prev_end + 1,
                        end_line=start - 1,
                        text="".join(inter_lines),
                        kind="interstitial",
                    )
                )

        chunk_id += 1
        chunks.append(
            Chunk(
                id=chunk_id,
                start_line=start,
                end_line=end,
                text="".join(lines[start - 1 : end]),
                kind=_typescript_node_kind(node),
                name=_typescript_node_name(raw_node, lines),
            )
        )
        prev_end = end

    if prev_end < len(lines):
        remaining = lines[prev_end:]
        if remaining and any(line.strip() for line in remaining):
            chunk_id += 1
            chunks.append(
                Chunk(
                    id=chunk_id,
                    start_line=prev_end + 1,
                    end_line=len(lines),
                    text="".join(remaining),
                    kind="interstitial",
                )
            )

    if not chunks:
        return LineChunker().chunk(content)
    return chunks


def _chunk_typescript_hybrid(content: str) -> list[Chunk]:
    if not content:
        return []

    tree = _parse_typescript(content)
    if tree is None:
        return LineChunker().chunk(content)

    lines = content.splitlines(keepends=True)
    chunks: list[Chunk] = []
    chunk_id = 0
    prev_end = 0

    for raw_node in tree.root_node.named_children:
        node = _typescript_effective_node(raw_node)
        start = raw_node.start_point.row + 1
        end = raw_node.end_point.row + 1

        if start > prev_end + 1:
            inter_lines = lines[prev_end : start - 1]
            if inter_lines and any(line.strip() for line in inter_lines):
                chunk_id += 1
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        start_line=prev_end + 1,
                        end_line=start - 1,
                        text="".join(inter_lines),
                        kind="interstitial",
                    )
                )

        if node.type == "class_declaration":
            class_chunks = _chunk_typescript_class_node(raw_node, node, lines, chunk_id)
            chunks.extend(class_chunks)
            if class_chunks:
                chunk_id = class_chunks[-1].id
        else:
            chunk_id += 1
            chunks.append(
                Chunk(
                    id=chunk_id,
                    start_line=start,
                    end_line=end,
                    text="".join(lines[start - 1 : end]),
                    kind=_typescript_node_kind(node),
                    name=_typescript_node_name(raw_node, lines),
                )
            )
        prev_end = end

    if prev_end < len(lines):
        remaining = lines[prev_end:]
        if remaining and any(line.strip() for line in remaining):
            chunk_id += 1
            chunks.append(
                Chunk(
                    id=chunk_id,
                    start_line=prev_end + 1,
                    end_line=len(lines),
                    text="".join(remaining),
                    kind="interstitial",
                )
            )

    if not chunks:
        return LineChunker().chunk(content)
    return chunks


def _chunk_typescript_class_node(raw_node, class_node, lines: list[str], starting_id: int) -> list[Chunk]:
    class_body = next((child for child in class_node.named_children if child.type == "class_body"), None)
    if class_body is None:
        return [
            Chunk(
                id=starting_id + 1,
                start_line=raw_node.start_point.row + 1,
                end_line=raw_node.end_point.row + 1,
                text="".join(lines[raw_node.start_point.row : raw_node.end_point.row + 1]),
                kind="class",
                name=_typescript_node_name(raw_node, lines),
            )
        ]

    members = [
        child
        for child in class_body.named_children
        if child.type in {"method_definition", "public_field_definition"}
    ]
    if not members:
        return [
            Chunk(
                id=starting_id + 1,
                start_line=raw_node.start_point.row + 1,
                end_line=raw_node.end_point.row + 1,
                text="".join(lines[raw_node.start_point.row : raw_node.end_point.row + 1]),
                kind="class",
                name=_typescript_node_name(raw_node, lines),
            )
        ]

    chunks: list[Chunk] = []
    chunk_id = starting_id
    class_start = raw_node.start_point.row + 1
    class_end = raw_node.end_point.row + 1
    first_member_start = members[0].start_point.row + 1
    if class_start <= first_member_start - 1:
        chunk_id += 1
        chunks.append(
            Chunk(
                id=chunk_id,
                start_line=class_start,
                end_line=first_member_start - 1,
                text="".join(lines[class_start - 1 : first_member_start - 1]),
                kind="class",
                name=_typescript_node_name(raw_node, lines),
            )
        )

    for idx, member in enumerate(members):
        start = member.start_point.row + 1
        end = member.end_point.row + 1
        chunk_id += 1
        chunks.append(
            Chunk(
                id=chunk_id,
                start_line=start,
                end_line=end,
                text="".join(lines[start - 1 : end]),
                kind="method" if member.type == "method_definition" else "statement",
                name=_typescript_node_name(member, lines),
            )
        )
        next_start = members[idx + 1].start_point.row + 1 if idx + 1 < len(members) else class_end + 1
        if end + 1 <= next_start - 1:
            chunk_id += 1
            chunks.append(
                Chunk(
                    id=chunk_id,
                    start_line=end + 1,
                    end_line=next_start - 1,
                    text="".join(lines[end : next_start - 1]),
                    kind="class",
                    name=_typescript_node_name(raw_node, lines),
                )
            )

    return [chunk for chunk in chunks if any(line.strip() for line in chunk.text.splitlines())]


def _typescript_effective_node(node):
    if node.type == "export_statement" and node.named_children:
        return node.named_children[0]
    return node


def _chunk_python_hybrid(content: str) -> list[Chunk]:
    if not content:
        return []

    tree = _parse_python(content)
    if tree is None:
        return LineChunker().chunk(content)

    lines = content.splitlines(keepends=True)
    chunks: list[Chunk] = []
    chunk_id = 0
    prev_end = 0

    for node in getattr(tree, "body", []):
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None) or start
        if start is None or end is None:
            continue

        if start > prev_end + 1:
            inter_lines = lines[prev_end : start - 1]
            if inter_lines and any(line.strip() for line in inter_lines):
                chunk_id += 1
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        start_line=prev_end + 1,
                        end_line=start - 1,
                        text="".join(inter_lines),
                        kind="interstitial",
                    )
                )

        if _is_class_like(node):
            class_chunks = _chunk_class_node(node, lines, chunk_id)
            chunks.extend(class_chunks)
            if class_chunks:
                chunk_id = class_chunks[-1].id
        else:
            chunk_id += 1
            chunks.append(
                Chunk(
                    id=chunk_id,
                    start_line=start,
                    end_line=end,
                    text="".join(lines[start - 1 : end]),
                    kind=_python_node_kind(node),
                    name=getattr(node, "name", ""),
                )
            )

        prev_end = end

    if prev_end < len(lines):
        remaining = lines[prev_end:]
        if remaining and any(line.strip() for line in remaining):
            chunk_id += 1
            chunks.append(
                Chunk(
                    id=chunk_id,
                    start_line=prev_end + 1,
                    end_line=len(lines),
                    text="".join(remaining),
                    kind="interstitial",
                )
            )

    if not chunks:
        return LineChunker().chunk(content)

    return chunks


def _chunk_class_node(node, lines: list[str], starting_id: int) -> list[Chunk]:
    method_nodes = [
        child
        for child in getattr(node, "body", [])
        if _is_function_like(child)
    ]
    chunks: list[Chunk] = []
    chunk_id = starting_id
    class_start = node.lineno
    class_end = node.end_lineno or class_start

    if not method_nodes:
        chunk_id += 1
        chunks.append(
            Chunk(
                id=chunk_id,
                start_line=class_start,
                end_line=class_end,
                text="".join(lines[class_start - 1 : class_end]),
                kind="class",
                name=getattr(node, "name", ""),
            )
        )
        return chunks

    method_nodes.sort(key=lambda child: child.lineno)
    cursor = class_start

    first_method = method_nodes[0]
    if cursor <= first_method.lineno - 1:
        chunk_id += 1
        chunks.append(
            Chunk(
                id=chunk_id,
                start_line=cursor,
                end_line=first_method.lineno - 1,
                text="".join(lines[cursor - 1 : first_method.lineno - 1]),
                kind="class",
                name=getattr(node, "name", ""),
            )
        )

    for idx, method in enumerate(method_nodes):
        start = method.lineno
        end = method.end_lineno or start
        chunk_id += 1
        chunks.append(
            Chunk(
                id=chunk_id,
                start_line=start,
                end_line=end,
                text="".join(lines[start - 1 : end]),
                kind="method",
                name=getattr(method, "name", ""),
            )
        )

        next_start = method_nodes[idx + 1].lineno if idx + 1 < len(method_nodes) else class_end + 1
        if end + 1 <= next_start - 1:
            chunk_id += 1
            chunks.append(
                Chunk(
                    id=chunk_id,
                    start_line=end + 1,
                    end_line=next_start - 1,
                    text="".join(lines[end : next_start - 1]),
                    kind="class",
                    name=getattr(node, "name", ""),
                )
            )

    return [chunk for chunk in chunks if any(line.strip() for line in chunk.text.splitlines())]


def _is_function_like(node) -> bool:
    import ast

    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))


def _is_class_like(node) -> bool:
    import ast

    return isinstance(node, ast.ClassDef)


def _is_main_guard(node) -> bool:
    import ast

    if not isinstance(node, ast.If):
        return False
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def _python_node_kind(node) -> str:
    import ast

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return "function"
    if isinstance(node, ast.ClassDef):
        return "class"
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return "import"
    if _is_main_guard(node):
        return "guard"
    return "statement"


def _typescript_node_kind(node) -> str:
    type_map = {
        "import_statement": "import",
        "function_declaration": "function",
        "class_declaration": "class",
        "method_definition": "method",
        "export_statement": "function",
        "lexical_declaration": "statement",
    }
    return type_map.get(node.type, "statement")


def _typescript_node_name(node, lines: list[str]) -> str:
    text = "".join(lines[node.start_point.row : node.end_point.row + 1]).strip()
    first_line = text.splitlines()[0] if text else ""
    match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", first_line)
    return match.group(1) if match else ""


def marker_label_for_chunk(chunk: Chunk, style: str = "numeric") -> str:
    """Render a marker label for a chunk.

    `numeric` preserves the historical `B01` shape.
    `typed` uses an alphanumeric prefix based on region kind while keeping the
    numeric portion globally stable.
    """

    if style == "numeric":
        return f"B{chunk.id:02d}"

    prefix_map = {
        "import": "I",
        "function": "F",
        "class": "C",
        "method": "M",
        "guard": "G",
        "statement": "P",
        "interstitial": "P",
        "chunk": "P",
    }
    prefix = prefix_map.get(chunk.kind, "P")
    return f"{prefix}{chunk.id:02d}"


# ── Chunker Registry ─────────────────────────────────────────────────────

_CHUNKERS: dict[str, type[Chunker]] = {
    "line": LineChunker,
    "paragraph": ParagraphChunker,
    "token": FixedTokenChunker,
    "ast": ASTChunker,
    "hybrid": HybridChunker,
    "ts_ast": TypeScriptASTChunker,
    "ts_hybrid": TypeScriptHybridChunker,
}


def get_chunker(name: str, **kwargs) -> Chunker:
    """Get a chunker by name with optional constructor arguments.

    Raises:
        KeyError: If the chunker is not found.
    """
    cls = _CHUNKERS.get(name)
    if cls is None:
        raise KeyError(f"Unknown chunker: {name}. Available: {list(_CHUNKERS.keys())}")
    return cls(**kwargs)


def list_chunkers() -> list[str]:
    """Return names of all registered chunkers."""
    return list(_CHUNKERS.keys())
