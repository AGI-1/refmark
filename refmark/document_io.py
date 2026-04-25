"""Text extraction helpers for lightweight document pipeline examples."""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


def extract_document_text(path: str | Path) -> str:
    """Extract plain text from simple text, DOCX, or PDF inputs."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".docx":
        return extract_docx_text(source)
    if suffix == ".pdf":
        return extract_pdf_text(source)
    return source.read_text(encoding="utf-8-sig")


def text_mapping_extension(path: str | Path) -> str:
    """Return the extension to use after document text extraction."""
    suffix = Path(path).suffix.lower()
    if suffix in {".docx", ".pdf"}:
        return ".txt"
    return suffix or ".txt"


def extract_docx_text(path: str | Path) -> str:
    """Extract paragraphs and table text from a basic Word document."""
    with zipfile.ZipFile(path) as package:
        document_xml = package.read("word/document.xml")

    root = ET.fromstring(document_xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", ns):
        texts = [node.text or "" for node in paragraph.findall(".//w:t", ns)]
        text = "".join(texts).strip()
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs) + ("\n" if paragraphs else "")


def extract_pdf_text(path: str | Path) -> str:
    """Extract text from a PDF using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF input requires pypdf. Install refmark with the dev or train-fetch extras.") from exc

    reader = PdfReader(str(path))
    paragraphs: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        paragraphs.extend(line.strip() for line in text.splitlines() if line.strip())
    text = "\n\n".join(paragraphs)
    return text + ("\n" if text else "")
