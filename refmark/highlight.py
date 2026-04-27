from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from refmark.edit import _remove_rfm_header
from refmark.regions import _normalize_region_key, _parse_blocks_with_mode
from refmark.config import refmark_home
from refmark.core import strip
from refmark.citations import parse_citation_refs
from refmark.shadow_session import load_or_build_view_state


@dataclass(frozen=True)
class HighlightRegion:
    region_id: str
    start_line: int
    end_line: int
    text: str


@dataclass(frozen=True)
class HighlightResult:
    file_path: str
    namespace_mode: str
    marker_format: str | None
    refs: list[str]
    regions: list[HighlightRegion]
    context_lines: int

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["regions"] = [asdict(item) for item in self.regions]
        return payload


def _default_state_dir() -> Path:
    return refmark_home() / "shadow_state"


def _clean_source_text(path: Path, view_state: dict) -> str:
    source = str(view_state["source"])
    if view_state["namespace_mode"] != "live":
        return source
    marker_format = view_state["marker_format"]
    if marker_format:
        source = strip(source, path.suffix, marker_format=marker_format)
    source = _remove_rfm_header(source)
    if source and not source.endswith("\n"):
        source += "\n"
    return source


def _ordered_blocks(view_state: dict) -> list[tuple[str, dict]]:
    marker_format = view_state["marker_format"]
    if not marker_format:
        return []
    blocks = _parse_blocks_with_mode(
        str(view_state["view_text"]),
        marker_format,
        line_mode="original",
    )
    return sorted(blocks.items(), key=lambda item: (item[1]["line_start"], item[0]))


def _expand_refs(refs: list[str], ordered_ids: list[str]) -> list[str]:
    index = {value: idx for idx, value in enumerate(ordered_ids)}
    resolved: list[str] = []
    for citation in parse_citation_refs(refs):
        if citation.is_range:
            start = _normalize_region_key(citation.ref)
            end = _normalize_region_key(citation.end_ref or "")
            if start not in index or end not in index:
                raise ValueError(f"Unknown ref range '{citation.stable_ref}-{citation.stable_end_ref}'.")
            lo = min(index[start], index[end])
            hi = max(index[start], index[end])
            resolved.extend(ordered_ids[lo : hi + 1])
            continue
        normalized = _normalize_region_key(citation.ref)
        if normalized not in index:
            raise ValueError(f"Unknown ref '{citation.stable_ref}'.")
        resolved.append(normalized)

    unique: list[str] = []
    seen: set[str] = set()
    for value in resolved:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def highlight_refs(
    file_path: str | Path,
    refs: list[str] | str,
    *,
    context_lines: int = 2,
    state_dir: str | Path | None = None,
) -> HighlightResult:
    path = Path(file_path)
    if isinstance(refs, str):
        refs = [part.strip() for part in refs.split(",") if part.strip()]
    state_root = Path(state_dir) if state_dir else _default_state_dir()
    view_state = load_or_build_view_state(path, state_root)
    if not view_state["supported"]:
        raise ValueError(f"Refmark highlighting is not supported for {path.suffix or 'this file type'}.")

    ordered = _ordered_blocks(view_state)
    ordered_ids = [region_id for region_id, _block in ordered]
    expanded_refs = _expand_refs(refs, ordered_ids)
    clean_text = _clean_source_text(path, view_state)
    clean_lines = clean_text.splitlines()
    blocks = {region_id: block for region_id, block in ordered}

    regions: list[HighlightRegion] = []
    for region_id in expanded_refs:
        block = blocks[region_id]
        start_line = int(block["line_start"])
        end_line = int(block["line_end"])
        snippet_start = max(1, start_line - context_lines)
        snippet_end = min(len(clean_lines), end_line + context_lines)
        snippet = "\n".join(clean_lines[snippet_start - 1 : snippet_end])
        regions.append(
            HighlightRegion(
                region_id=region_id,
                start_line=start_line,
                end_line=end_line,
                text=snippet,
            )
        )

    return HighlightResult(
        file_path=str(path),
        namespace_mode=str(view_state["namespace_mode"]),
        marker_format=view_state["marker_format"],
        refs=expanded_refs,
        regions=regions,
        context_lines=context_lines,
    )


def render_highlight_text(result: HighlightResult) -> str:
    lines = [
        f"file: {result.file_path}",
        f"namespace_mode: {result.namespace_mode}",
        f"refs: {', '.join(result.refs)}",
        "",
    ]
    for region in result.regions:
        snippet_lines = region.text.splitlines()
        snippet_start = max(1, region.start_line - result.context_lines)
        lines.append(f"[{region.region_id}] lines {region.start_line}-{region.end_line}")
        for offset, content in enumerate(snippet_lines, start=snippet_start):
            marker = ">" if region.start_line <= offset <= region.end_line else " "
            lines.append(f"{marker} {offset:4d} | {content}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_highlight_html(result: HighlightResult) -> str:
    parts = [
        "<html><head><meta charset='utf-8'><style>",
        "body{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;padding:16px;}",
        "pre{background:#fafafa;border:1px solid #ddd;padding:12px;overflow:auto;}",
        ".hit{background:#fff2a8;display:block;}",
        ".ctx{display:block;color:#444;}",
        ".meta{font-family:system-ui,sans-serif;margin-bottom:16px;}",
        "</style></head><body>",
        f"<div class='meta'><strong>file:</strong> {html.escape(result.file_path)}<br><strong>refs:</strong> {html.escape(', '.join(result.refs))}</div>",
    ]
    for region in result.regions:
        parts.append(
            f"<h3>{html.escape(region.region_id)} lines {region.start_line}-{region.end_line}</h3><pre>"
        )
        snippet_lines = region.text.splitlines()
        snippet_start = max(1, region.start_line - result.context_lines)
        for offset, content in enumerate(snippet_lines, start=snippet_start):
            escaped = html.escape(content)
            cls = "hit" if region.start_line <= offset <= region.end_line else "ctx"
            parts.append(f"<span class='{cls}'>{offset:4d} | {escaped}</span>")
        parts.append("</pre>")
    parts.append("</body></html>")
    return "".join(parts)


def render_highlight_json(result: HighlightResult) -> str:
    return json.dumps(result.to_dict(), indent=2)
