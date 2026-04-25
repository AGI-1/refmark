"""User-friendly document workflow API for Refmark."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from refmark.document_io import extract_document_text, text_mapping_extension
from refmark.pipeline import (
    AlignmentCandidate,
    CoverageItem,
    RegionRecord,
    align_region_records,
    build_region_manifest,
    evaluate_alignment_coverage,
    expand_region_context,
    render_coverage_report_html,
    summarize_coverage,
    write_manifest,
)
from refmark.workflow_config import WorkflowConfig, resolve_workflow_config


@dataclass(frozen=True)
class DocumentMap:
    doc_id: str
    source_path: str
    marked_text: str
    records: list[RegionRecord]
    config: WorkflowConfig
    warnings: list[str]

    def expand(self, refs: list[str], *, before: int | None = None, after: int | None = None) -> list[RegionRecord]:
        return expand_region_context(
            self.records,
            refs,
            doc_id=self.doc_id,
            before=self.config.expand_before if before is None else before,
            after=self.config.expand_after if after is None else after,
        )

    def write_manifest(self, path: str | Path) -> None:
        write_manifest(self.records, path)

    def to_dict(self) -> dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "source_path": self.source_path,
            "records": [record.to_dict() for record in self.records],
            "config": self.config.to_dict(),
            "warnings": self.warnings,
        }


@dataclass(frozen=True)
class AlignmentReport:
    source: DocumentMap
    target: DocumentMap
    alignments: list[list[AlignmentCandidate]]
    coverage: list[CoverageItem]
    config: WorkflowConfig

    @property
    def summary(self) -> dict[str, object]:
        return summarize_coverage(self.coverage)

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "alignments": [[candidate.to_dict() for candidate in row] for row in self.alignments],
            "coverage": [item.to_dict() for item in self.coverage],
            "summary": self.summary,
            "config": self.config.to_dict(),
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def write_html(self, path: str | Path, *, layout: str = "side-by-side") -> None:
        Path(path).write_text(
            render_coverage_report_html(self.coverage, title="Refmark Coverage Review", layout=layout),
            encoding="utf-8",
        )


def map_document(
    path: str | Path,
    *,
    config: WorkflowConfig | None = None,
    doc_id: str | None = None,
    **overrides,
) -> DocumentMap:
    resolved = resolve_workflow_config(config, **overrides)
    source = Path(path)
    text = extract_document_text(source)
    warnings = _extraction_warnings(source, text)
    marked, records = build_region_manifest(
        text,
        text_mapping_extension(source),
        doc_id=doc_id or source.stem,
        source_path=str(source),
        marker_format=resolved.marker_format,
        chunker=resolved.chunker,
        chunker_kwargs=_chunker_kwargs(resolved),
        min_words=resolved.min_words,
    )
    if not resolved.include_headings:
        records = _drop_first_heading(records, doc_id or source.stem)
    return DocumentMap(
        doc_id=doc_id or source.stem,
        source_path=str(source),
        marked_text=marked,
        records=records,
        config=resolved,
        warnings=warnings,
    )


def align_documents(
    source: str | Path | DocumentMap,
    target: str | Path | DocumentMap,
    *,
    config: WorkflowConfig | None = None,
    **overrides,
) -> AlignmentReport:
    resolved = resolve_workflow_config(config, **overrides)
    source_map = source if isinstance(source, DocumentMap) else map_document(source, config=resolved)
    target_map = target if isinstance(target, DocumentMap) else map_document(target, config=resolved)
    alignments = align_region_records(source_map.records, target_map.records, top_k=resolved.top_k)
    coverage = evaluate_alignment_coverage(
        source_map.records,
        target_map.records,
        top_k=resolved.top_k,
        threshold=resolved.coverage_threshold,
        expand_before=resolved.expand_before,
        expand_after=resolved.expand_after,
        numeric_checks=resolved.numeric_checks,
    )
    return AlignmentReport(
        source=source_map,
        target=target_map,
        alignments=alignments,
        coverage=coverage,
        config=resolved,
    )


def _extraction_warnings(path: Path, text: str) -> list[str]:
    warnings: list[str] = []
    if path.suffix.lower() == ".pdf" and len(text.split()) < 20:
        warnings.append("PDF extraction returned little text; scanned or layout-heavy PDFs may need OCR.")
    if path.suffix.lower() == ".docx" and not text.strip():
        warnings.append("DOCX extraction returned no text.")
    return warnings


def _drop_first_heading(records: list[RegionRecord], doc_id: str) -> list[RegionRecord]:
    if not records:
        return records
    title_tokens = set(doc_id.replace("_", " ").lower().split())
    first_tokens = set(records[0].text.lower().split())
    if first_tokens and first_tokens <= title_tokens | {"technical", "specification", "requirements", "request", "offer", "contract"}:
        return records[1:]
    return records


def _chunker_kwargs(config: WorkflowConfig) -> dict | None:
    if config.chunker == "line" and config.lines_per_chunk:
        return {"lines_per_chunk": config.lines_per_chunk}
    if config.chunker == "token" and config.tokens_per_chunk:
        return {"tokens_per_chunk": config.tokens_per_chunk}
    return None
