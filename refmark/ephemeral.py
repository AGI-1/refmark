"""Disposable address maps for one-off document workflows."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import zipfile
from xml.etree import ElementTree as ET

from refmark.document_io import extract_document_text
from refmark.documents import DocumentMap, map_document
from refmark.pipeline import RegionRecord
from refmark.workflow_config import WorkflowConfig


@dataclass(frozen=True)
class EphemeralMap:
    document: DocumentMap
    instructions: str

    def to_dict(self, *, include_marked_text: bool = True) -> dict[str, object]:
        payload = {
            "schema": "refmark.ephemeral_map.v1",
            "source_path": self.document.source_path,
            "doc_id": self.document.doc_id,
            "instructions": self.instructions,
            "regions": [record.to_dict() for record in self.document.records],
            "warnings": self.document.warnings,
        }
        if include_marked_text:
            payload["marked_text"] = self.document.marked_text
        return payload


def build_ephemeral_map(
    path: str | Path,
    *,
    config: WorkflowConfig | None = None,
    doc_id: str | None = None,
) -> EphemeralMap:
    """Return a temporary marked view and region manifest for one document.

    The source file is not modified. Callers can send `marked_text` to a model
    and ask for edits addressed by refs, then pass those edits to
    `apply_ephemeral_edits`.
    """

    document = map_document(path, config=config, doc_id=doc_id)
    return EphemeralMap(document=document, instructions=_ephemeral_instructions(document))


def apply_ephemeral_edits(
    path: str | Path,
    edits: list[dict[str, object]],
    *,
    output: str | Path,
    config: WorkflowConfig | None = None,
    doc_id: str | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    """Apply simple ref-addressed replacement edits to a one-off document.

    Supported edit shape:

    `{"ref": "P02", "action": "replace", "new_text": "..."}`

    For text-like files, replacements are applied to the source text. For DOCX,
    only exact single-paragraph replacements are applied. For PDF, Refmark writes
    patched extracted text because PDF layout-safe editing is intentionally out
    of scope for this lightweight path.
    """

    source = Path(path)
    destination = Path(output)
    document = map_document(source, config=config, doc_id=doc_id)
    by_ref = _records_by_ref(document.records)
    normalized = [_normalize_edit(edit, by_ref) for edit in edits]
    errors = [error for item in normalized for error in item["errors"]]
    if errors:
        return _result(False, source, destination, document, [], errors, dry_run=dry_run)

    replacements = [
        {
            "ref": item["ref"],
            "old_text": item["record"].text,
            "new_text": item["new_text"],
        }
        for item in normalized
    ]
    suffix = source.suffix.lower()
    if suffix == ".docx":
        changes, apply_errors = _apply_docx_replacements(source, destination, replacements, dry_run=dry_run)
    elif suffix == ".pdf":
        text = extract_document_text(source)
        patched, changes, apply_errors = _replace_unique_text(text, replacements)
        if not dry_run and not apply_errors:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(patched, encoding="utf-8")
    else:
        text = source.read_text(encoding="utf-8-sig")
        patched, changes, apply_errors = _replace_unique_text(text, replacements)
        if not dry_run and not apply_errors:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(patched, encoding="utf-8")

    warnings = list(document.warnings)
    if suffix == ".pdf":
        warnings.append("PDF output is patched extracted text, not an edited PDF.")
    return _result(not apply_errors, source, destination, document, changes, apply_errors, warnings=warnings, dry_run=dry_run)


def _ephemeral_instructions(document: DocumentMap) -> str:
    return (
        "Use the visible refs as temporary addresses. Do not rewrite the whole document. "
        "Return JSON only, with an edits array. Each edit should use "
        '{"ref":"P01","action":"replace","new_text":"..."}. '
        "Refs are disposable for this one task and do not need durable storage. "
        f"Available refs: {', '.join(record.region_id for record in document.records)}."
    )


def _records_by_ref(records: list[RegionRecord]) -> dict[str, RegionRecord]:
    by_ref: dict[str, RegionRecord] = {}
    for record in records:
        by_ref[record.region_id] = record
        by_ref[f"{record.doc_id}:{record.region_id}"] = record
    return by_ref


def _normalize_edit(edit: dict[str, object], by_ref: dict[str, RegionRecord]) -> dict[str, object]:
    ref = str(edit.get("ref") or edit.get("region_id") or edit.get("region") or "").strip()
    action = str(edit.get("action") or "replace").strip().lower()
    new_text = edit.get("new_text", edit.get("new_content", edit.get("text", "")))
    errors: list[str] = []
    record = by_ref.get(ref)
    if not ref:
        errors.append("Edit is missing ref/region_id.")
    elif record is None:
        errors.append(f"Unknown ref: {ref}")
    if action != "replace":
        errors.append(f"Unsupported ephemeral action for {ref or '<missing>'}: {action}")
    if not isinstance(new_text, str):
        errors.append(f"new_text for {ref or '<missing>'} must be a string.")
        new_text = ""
    return {"ref": ref, "record": record, "new_text": new_text, "errors": errors}


def _replace_unique_text(text: str, replacements: list[dict[str, str]]) -> tuple[str, list[dict[str, object]], list[str]]:
    patched = text
    changes: list[dict[str, object]] = []
    errors: list[str] = []
    for replacement in replacements:
        old_text = replacement["old_text"]
        new_text = replacement["new_text"]
        count = patched.count(old_text)
        if count != 1:
            errors.append(f"Ref {replacement['ref']} matched {count} source spans; expected exactly one.")
            continue
        patched = patched.replace(old_text, new_text, 1)
        changes.append({"ref": replacement["ref"], "old_chars": len(old_text), "new_chars": len(new_text)})
    return patched, changes, errors


def _apply_docx_replacements(
    source: Path,
    destination: Path,
    replacements: list[dict[str, str]],
    *,
    dry_run: bool,
) -> tuple[list[dict[str, object]], list[str]]:
    changes: list[dict[str, object]] = []
    errors: list[str] = []
    with zipfile.ZipFile(source) as package:
        document_xml = package.read("word/document.xml")
    root = ET.fromstring(document_xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = root.findall(".//w:p", ns)
    paragraph_text_nodes = [(paragraph, paragraph.findall(".//w:t", ns)) for paragraph in paragraphs]
    paragraph_texts = ["".join(node.text or "" for node in nodes).strip() for _paragraph, nodes in paragraph_text_nodes]

    for replacement in replacements:
        old_text = replacement["old_text"].strip()
        matches = [index for index, text in enumerate(paragraph_texts) if text == old_text]
        if len(matches) != 1:
            errors.append(f"Ref {replacement['ref']} matched {len(matches)} DOCX paragraphs; expected exactly one.")
            continue
        if "\n" in replacement["new_text"].strip():
            errors.append(f"Ref {replacement['ref']} DOCX replacement must be a single paragraph.")
            continue
        index = matches[0]
        nodes = paragraph_text_nodes[index][1]
        if not nodes:
            errors.append(f"Ref {replacement['ref']} DOCX paragraph has no text nodes.")
            continue
        nodes[0].text = replacement["new_text"]
        for node in nodes[1:]:
            node.text = ""
        changes.append({"ref": replacement["ref"], "paragraph_index": index + 1})

    if errors or dry_run:
        return changes, errors

    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.register_namespace("w", ns["w"])
    updated_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    with zipfile.ZipFile(source) as src, zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            if info.filename == "word/document.xml":
                dst.writestr(info, updated_xml)
            else:
                dst.writestr(info, src.read(info.filename))
    return changes, errors


def _result(
    ok: bool,
    source: Path,
    destination: Path,
    document: DocumentMap,
    changes: list[dict[str, object]],
    errors: list[str],
    *,
    warnings: list[str] | None = None,
    dry_run: bool,
) -> dict[str, object]:
    return {
        "schema": "refmark.ephemeral_apply_result.v1",
        "ok": ok,
        "dry_run": dry_run,
        "source_path": str(source),
        "output_path": str(destination),
        "doc_id": document.doc_id,
        "changes": changes,
        "errors": errors,
        "warnings": warnings or document.warnings,
    }


def edits_from_json(value: str) -> list[dict[str, object]]:
    payload = json.loads(value)
    edits = payload.get("edits") if isinstance(payload, dict) else payload
    if not isinstance(edits, list) or not all(isinstance(item, dict) for item in edits):
        raise ValueError("Expected a JSON array of edits or an object with an edits array.")
    return edits
