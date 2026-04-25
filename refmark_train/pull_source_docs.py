from __future__ import annotations

import json
import hashlib
from io import BytesIO
from pathlib import Path
import re

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "source_docs"
RAW_DIR = OUT_DIR / "raw"
TEXT_DIR = OUT_DIR / "text"

WHITESPACE_RE = re.compile(r"\s+")

DOCS = [
    {
        "id": "corporate_apple_10k_2024",
        "category": "corporate",
        "title": "Apple Inc. 2024 Form 10-K",
        "url": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm",
        "format": "html",
    },
    {
        "id": "corporate_microsoft_10k_2024",
        "category": "corporate",
        "title": "Microsoft 2024 Form 10-K",
        "url": "https://www.sec.gov/Archives/edgar/data/789019/000095017024087843/msft-20240630.htm",
        "format": "html",
    },
    {
        "id": "documentation_python_sqlite3",
        "category": "documentation",
        "title": "Python sqlite3 Documentation",
        "url": "https://docs.python.org/3/library/sqlite3.html",
        "format": "html",
    },
    {
        "id": "documentation_python_pathlib",
        "category": "documentation",
        "title": "Python pathlib Documentation",
        "url": "https://docs.python.org/3/library/pathlib.html",
        "format": "html",
    },
    {
        "id": "documentation_python_argparse",
        "category": "documentation",
        "title": "Python argparse Documentation",
        "url": "https://docs.python.org/3/library/argparse.html",
        "format": "html",
    },
    {
        "id": "documentation_python_asyncio_task",
        "category": "documentation",
        "title": "Python Coroutines and Tasks Documentation",
        "url": "https://docs.python.org/3/library/asyncio-task.html",
        "format": "html",
    },
    {
        "id": "documentation_python_subprocess",
        "category": "documentation",
        "title": "Python subprocess Documentation",
        "url": "https://docs.python.org/3/library/subprocess.html",
        "format": "html",
    },
    {
        "id": "medical_fda_m14_2026",
        "category": "medical",
        "title": "FDA M14 Guidance on Non-interventional Studies Using Real-World Data",
        "url": "https://www.fda.gov/media/179795/download",
        "format": "pdf",
    },
    {
        "id": "medical_fda_rwe_drugs_2025",
        "category": "medical",
        "title": "FDA Real-World Evidence Guidance for Industry",
        "url": "https://www.fda.gov/media/177128/download",
        "format": "pdf",
    },
    {
        "id": "medical_fda_pdufa_rwe_2023",
        "category": "medical",
        "title": "FDA PDUFA VII Real-World Evidence Guidance",
        "url": "https://www.fda.gov/media/170950/download",
        "format": "pdf",
    },
    {
        "id": "medical_fda_ectd_76466",
        "category": "medical",
        "title": "FDA eCTD Guidance for Industry",
        "url": "https://www.fda.gov/media/76466/download",
        "format": "pdf",
    },
    {
        "id": "medical_fda_e6_r3_gcp",
        "category": "medical",
        "title": "FDA E6(R3) Good Clinical Practice",
        "url": "https://www.fda.gov/media/169090/download",
        "format": "pdf",
    },
    {
        "id": "medical_fda_rwe_devices_2025",
        "category": "medical",
        "title": "FDA Use of Real-World Evidence to Support Regulatory Decision-Making for Medical Devices",
        "url": "https://www.fda.gov/media/190201/download",
        "format": "pdf",
    },
    {
        "id": "legal_osha_bloodborne_pathogens",
        "category": "legal",
        "title": "OSHA 1910.1030 Bloodborne Pathogens",
        "url": "https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.1030",
        "format": "html",
    },
    {
        "id": "legal_osha_hazard_communication_standard",
        "category": "legal",
        "title": "OSHA 1910.1200 Hazard Communication",
        "url": "https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.1200",
        "format": "html",
    },
    {
        "id": "legal_osha_hazard_communication_final_rule_2024",
        "category": "legal",
        "title": "OSHA Hazard Communication Standard Final Rule 2024",
        "url": "https://www.osha.gov/laws-regs/federalregister/2024-05-20",
        "format": "html",
    },
]


def _normalize_text(text: str) -> str:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = WHITESPACE_RE.sub(" ", raw_line).strip()
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        lines.append(line)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


def _extract_html_text(payload: bytes) -> str:
    soup = BeautifulSoup(payload, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    return _normalize_text(text)


def _extract_pdf_text(payload: bytes) -> str:
    reader = PdfReader(BytesIO(payload))
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return _normalize_text(text)


def _fetch(url: str) -> bytes:
    headers = {
        "User-Agent": "Codex Refmark Train Research Fetcher/1.0 (contact: local-experiment)",
        "Accept": "*/*",
    }
    response = requests.get(url, headers=headers, timeout=120)
    response.raise_for_status()
    return response.content


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def pull_docs() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for doc in DOCS:
        payload = _fetch(doc["url"])
        raw_ext = "pdf" if doc["format"] == "pdf" else "html"
        raw_path = RAW_DIR / f"{doc['id']}.{raw_ext}"
        raw_path.write_bytes(payload)
        if doc["format"] == "pdf":
            text = _extract_pdf_text(payload)
        else:
            text = _extract_html_text(payload)
        text_path = TEXT_DIR / f"{doc['id']}.txt"
        text_path.write_text(text, encoding="utf-8")
        manifest.append(
            {
                **doc,
                "raw_path": str(raw_path),
                "text_path": str(text_path),
                "raw_sha256": _sha256_bytes(payload),
                "text_sha256": _sha256_text(text),
                "chars": len(text),
                "words": len(text.split()),
            }
        )
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_category_corpora(manifest)
    return manifest_path


def _write_category_corpora(manifest: list[dict[str, object]]) -> None:
    sets_dir = OUT_DIR / "sets"
    sets_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in manifest:
        grouped.setdefault(str(row["category"]), []).append(row)
    set_manifest: list[dict[str, object]] = []
    for category, items in grouped.items():
        parts: list[str] = []
        total_words = 0
        for item in items:
            text = Path(str(item["text_path"])).read_text(encoding="utf-8")
            parts.append(f"# Document: {item['title']}\n# Source URL: {item['url']}\n\n{text}")
            total_words += int(item["words"])
        combined = "\n\n".join(parts)
        out_path = sets_dir / f"{category}_set.txt"
        out_path.write_text(combined, encoding="utf-8")
        set_manifest.append(
            {
                "category": category,
                "text_path": str(out_path),
                "text_sha256": _sha256_text(combined),
                "documents": [item["id"] for item in items],
                "words": total_words,
                "meets_50k_words": total_words >= 50000,
            }
        )
    (sets_dir / "manifest.json").write_text(json.dumps(set_manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    path = pull_docs()
    print(path)
