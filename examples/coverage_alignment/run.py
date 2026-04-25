from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from refmark.documents import align_documents, map_document
from refmark.pipeline import (
    summarize_coverage,
    write_manifest,
)
from refmark.workflow_config import load_workflow_config, resolve_workflow_config


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
INPUTS = OUTPUT / "inputs"
CONFIG_PATH = ROOT / "refmark_workflow.yaml"


CUSTOMER_REQUEST = [
    "The supplier must provide 24/7 priority support with a 4 hour initial response for production incidents.",
    "The service must encrypt customer data at rest and in transit using industry standard cryptography.",
    "The contract must include a 99.9 percent monthly uptime service level with service credits.",
    "The supplier must store all personal data within the European Union.",
]

OFFER_CONTRACT = [
    "Support coverage is available during business hours. A premium support addendum may be purchased separately.",
    "Customer data is encrypted in transit with TLS and at rest with AES 256 encryption.",
    "The platform target availability is 99.9 percent monthly uptime.",
    "Service credits apply after validated outages.",
    "Data residency is available in EU regions for production tenants.",
]

TENDER = [
    "The tender requires battery storage with at least 500 kWh usable capacity.",
    "The system shall provide remote monitoring with daily performance exports.",
    "Delivery and commissioning must be completed within 90 days from contract signature.",
    "The vendor shall provide a 5 year maintenance plan including spare parts.",
]

SPECIFICATION = [
    "The proposed battery system has 520 kWh usable capacity and modular inverters.",
    "Remote monitoring is included. The portal supports alerts, dashboards, and daily CSV exports.",
    "Commissioning is planned for 120 days after contract signature subject to grid approval.",
    "Maintenance is offered for 3 years. Spare parts can be purchased as needed.",
]


def main() -> int:
    OUTPUT.mkdir(exist_ok=True)
    INPUTS.mkdir(exist_ok=True)

    customer_docx = INPUTS / "customer_request.docx"
    offer_pdf = INPUTS / "offer_contract.pdf"
    tender_docx = INPUTS / "tender.docx"
    specification_pdf = INPUTS / "technical_specification.pdf"
    _write_docx(customer_docx, "Customer Request", CUSTOMER_REQUEST)
    _write_pdf(offer_pdf, "Offer Contract", OFFER_CONTRACT)
    _write_docx(tender_docx, "Tender Requirements", TENDER)
    _write_pdf(specification_pdf, "Technical Specification", SPECIFICATION)

    customer_items = _run_pair("customer_vs_offer", customer_docx, offer_pdf)
    tender_items = _run_pair("tender_vs_specification", tender_docx, specification_pdf)

    summary = {
        "customer_vs_offer": _summarize(customer_items),
        "tender_vs_specification": _summarize(tender_items),
        "inputs": {
            "customer_request": str(customer_docx),
            "offer_contract": str(offer_pdf),
            "tender": str(tender_docx),
            "technical_specification": str(specification_pdf),
        },
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote coverage review artifacts under {OUTPUT}")
    return 0


def _run_pair(name: str, source_path: Path, target_path: Path):
    pair_dir = OUTPUT / name
    pair_dir.mkdir(exist_ok=True)
    config = load_workflow_config(CONFIG_PATH)
    source_map = map_document(source_path, config=config)
    target_map = map_document(target_path, config=config)
    write_manifest(source_map.records + target_map.records, pair_dir / "manifest.jsonl")
    (pair_dir / "source_marked.txt").write_text(source_map.marked_text, encoding="utf-8")
    (pair_dir / "target_marked.txt").write_text(target_map.marked_text, encoding="utf-8")

    naive_report = align_documents(source_map, target_map, config=resolve_workflow_config(config, expand_after=0))
    expanded_report = align_documents(source_map, target_map, config=resolve_workflow_config(config, expand_after=1))
    naive = naive_report.coverage
    expanded = expanded_report.coverage
    (pair_dir / "coverage_naive.json").write_text(
        json.dumps([item.to_dict() for item in naive], indent=2),
        encoding="utf-8",
    )
    (pair_dir / "coverage_expanded.json").write_text(
        json.dumps([item.to_dict() for item in expanded], indent=2),
        encoding="utf-8",
    )
    expanded_report.write_html(pair_dir / "coverage_review.html", layout="side-by-side")
    return expanded


def _summarize(items) -> dict[str, object]:
    return summarize_coverage(items)


def _write_docx(path: Path, title: str, paragraphs: list[str]) -> None:
    body_parts = [
        _paragraph_xml(title, style="Title"),
        *[_paragraph_xml(text) for text in paragraphs],
    ]
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        + "".join(body_parts)
        + '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr>'
        "</w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        "</Relationships>"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as package:
        package.writestr("[Content_Types].xml", content_types)
        package.writestr("_rels/.rels", rels)
        package.writestr("word/document.xml", document_xml)


def _paragraph_xml(text: str, *, style: str | None = None) -> str:
    style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{style_xml}<w:r><w:t>{_xml_escape(text)}</w:t></w:r></w:p>"


def _write_pdf(path: Path, title: str, paragraphs: list[str]) -> None:
    lines = [title, "", *paragraphs]
    stream = ["BT", "/F1 11 Tf", "72 760 Td"]
    first = True
    for line in lines:
        if first:
            first = False
        else:
            stream.append("0 -18 Td")
        stream.append(f"({_pdf_escape(line)}) Tj")
    stream.append("ET")
    content = "\n".join(stream).encode("latin-1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"\nendstream",
    ]
    payload = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(payload))
        payload.extend(f"{index} 0 obj\n".encode("ascii"))
        payload.extend(obj)
        payload.extend(b"\nendobj\n")
    xref_offset = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    payload.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    path.write_bytes(payload)


def _xml_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _pdf_escape(text: str) -> str:
    ascii_text = re.sub(r"[^\x20-\x7e]", " ", text)
    return ascii_text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


if __name__ == "__main__":
    raise SystemExit(main())
