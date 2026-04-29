"""Small pipeline primitives for addressable documents."""

from __future__ import annotations

import hashlib
import html
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from refmark.core import inject
from refmark.metrics import expand_refs
from refmark.regions import _parse_blocks_with_mode


@dataclass(frozen=True)
class RegionRecord:
    doc_id: str
    region_id: str
    text: str
    start_line: int
    end_line: int
    ordinal: int
    hash: str
    source_path: str | None = None
    prev_region_id: str | None = None
    next_region_id: str | None = None
    parent_region_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SectionEntry:
    doc_id: str
    title: str
    level: int
    heading_ref: str
    parent_ref: str | None
    start_ref: str
    end_ref: str
    refs: list[str]
    source_path: str | None = None
    ordinal: int = 0

    @property
    def range_ref(self) -> str:
        return self.start_ref if self.start_ref == self.end_ref else f"{self.start_ref}-{self.end_ref}"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["range_ref"] = self.range_ref
        return payload


@dataclass(frozen=True)
class AlignmentCandidate:
    source_doc_id: str
    source_region_id: str
    target_doc_id: str
    target_region_id: str
    score: float
    shared_terms: list[str]
    source_text: str
    target_text: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CoverageItem:
    source: RegionRecord
    candidates: list[AlignmentCandidate]
    status: str
    coverage_score: float
    naive_score: float
    expanded_score: float
    expanded_target_refs: list[str]
    expanded_targets: list[RegionRecord]
    numeric_conflict: bool
    missing_terms: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "status": self.status,
            "coverage_score": self.coverage_score,
            "naive_score": self.naive_score,
            "expanded_score": self.expanded_score,
            "expanded_target_refs": self.expanded_target_refs,
            "expanded_targets": [record.to_dict() for record in self.expanded_targets],
            "numeric_conflict": self.numeric_conflict,
            "missing_terms": self.missing_terms,
        }


def build_region_manifest(
    content: str,
    file_ext: str,
    *,
    doc_id: str,
    source_path: str | None = None,
    marker_format: str = "typed_bracket",
    chunker: str = "paragraph",
    chunker_kwargs: dict | None = None,
    min_words: int = 0,
) -> tuple[str, list[RegionRecord]]:
    """Inject refs into content and return marked text plus region metadata."""
    marked, _count = inject(
        content,
        file_ext,
        marker_format=marker_format,
        chunker=chunker,
        chunker_kwargs=chunker_kwargs,
    )
    blocks = _parse_blocks_with_mode(marked, marker_format, line_mode="marked")
    ordered = sorted(blocks.items(), key=lambda item: int(item[1]["ordinal"]))
    pending: list[tuple[str, dict]] = []
    for region_id, block in ordered:
        text = str(block.get("text", ""))
        if min_words and len(_token_set(text)) < min_words:
            continue
        pending.append((region_id, block))

    records: list[RegionRecord] = []
    current_heading_by_level: dict[int, str] = {}
    supports_headings = file_ext.lower() in {".md", ".markdown", ".rst", ".txt"}
    for index, (region_id, block) in enumerate(pending):
        prev_region_id = pending[index - 1][0] if index > 0 else None
        next_region_id = pending[index + 1][0] if index + 1 < len(pending) else None
        text = str(block.get("text", ""))
        parent_region_id = None
        heading_level = _heading_level(text) if supports_headings else None
        if heading_level is not None:
            for stale_level in [level for level in current_heading_by_level if level >= heading_level]:
                del current_heading_by_level[stale_level]
            parent_region_id = current_heading_by_level.get(max(current_heading_by_level) if current_heading_by_level else -1)
            current_heading_by_level[heading_level] = region_id
        else:
            parent_region_id = current_heading_by_level.get(max(current_heading_by_level) if current_heading_by_level else -1)
        records.append(
            RegionRecord(
                doc_id=doc_id,
                region_id=region_id,
                text=text,
                start_line=int(block["line_start"]),
                end_line=int(block["line_end"]),
                ordinal=int(block["ordinal"]),
                hash=_stable_text_hash(text),
                source_path=source_path,
                prev_region_id=prev_region_id,
                next_region_id=next_region_id,
                parent_region_id=parent_region_id,
            )
        )
    return marked, records


def write_manifest(records: Iterable[RegionRecord], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record.to_dict(), ensure_ascii=True) for record in records]
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_manifest(path: str | Path) -> list[RegionRecord]:
    records: list[RegionRecord] = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        records.append(
            RegionRecord(
                doc_id=str(payload["doc_id"]),
                region_id=str(payload["region_id"]),
                text=str(payload["text"]),
                start_line=int(payload["start_line"]),
                end_line=int(payload["end_line"]),
                ordinal=int(payload["ordinal"]),
                hash=str(payload["hash"]),
                source_path=payload.get("source_path"),
                prev_region_id=payload.get("prev_region_id"),
                next_region_id=payload.get("next_region_id"),
                parent_region_id=payload.get("parent_region_id"),
            )
        )
    return records


def build_section_map(records: Iterable[RegionRecord]) -> list[SectionEntry]:
    """Build a heading/title TOC layer from a region manifest.

    This keeps source documents untouched: headings become navigation metadata
    that point to concrete region ranges, while evidence scoring can still
    decide whether heading-only regions should count.
    """
    by_doc: dict[str, list[RegionRecord]] = {}
    for record in records:
        by_doc.setdefault(record.doc_id, []).append(record)
    sections: list[SectionEntry] = []
    for doc_id, doc_records in sorted(by_doc.items()):
        ordered = sorted(doc_records, key=lambda record: record.ordinal)
        heading_rows = [
            (index, record, _heading_level(record.text))
            for index, record in enumerate(ordered)
            if _heading_level(record.text) is not None
        ]
        for heading_index, heading_record, level in heading_rows:
            assert level is not None
            end_index = len(ordered) - 1
            for next_index, _next_record, next_level in heading_rows:
                if next_index <= heading_index:
                    continue
                if next_level is not None and next_level <= level:
                    end_index = next_index - 1
                    break
            section_records = ordered[heading_index : end_index + 1]
            refs = [_stable_ref(record) for record in section_records]
            parent_ref = f"{doc_id}:{heading_record.parent_region_id}" if heading_record.parent_region_id else None
            sections.append(
                SectionEntry(
                    doc_id=doc_id,
                    title=_heading_title(heading_record.text),
                    level=level,
                    heading_ref=_stable_ref(heading_record),
                    parent_ref=parent_ref,
                    start_ref=refs[0],
                    end_ref=refs[-1],
                    refs=refs,
                    source_path=heading_record.source_path,
                    ordinal=heading_record.ordinal,
                )
            )
    return sections


def expand_region_context(
    records: Iterable[RegionRecord],
    refs: Iterable[str],
    *,
    doc_id: str | None = None,
    before: int = 0,
    after: int = 0,
    same_parent: bool = False,
    include_parent: bool = False,
) -> list[RegionRecord]:
    """Return cited regions plus adjacent or same-parent context regions."""
    scoped = [record for record in records if doc_id is None or record.doc_id == doc_id]
    by_doc: dict[str, list[RegionRecord]] = {}
    for record in scoped:
        by_doc.setdefault(record.doc_id, []).append(record)
    for doc_records in by_doc.values():
        doc_records.sort(key=lambda record: record.ordinal)

    expanded: list[RegionRecord] = []
    seen: set[tuple[str, str]] = set()
    for current_doc_id, doc_records in by_doc.items():
        address_space = [record.region_id for record in doc_records]
        wanted = set(expand_refs(refs, address_space=address_space))
        if same_parent:
            parent_ids = {
                record.parent_region_id
                for record in doc_records
                if record.region_id in wanted and record.parent_region_id
            }
            for record in doc_records:
                if record.parent_region_id in parent_ids:
                    wanted.add(record.region_id)
                if include_parent and record.region_id in parent_ids:
                    wanted.add(record.region_id)
        for index, record in enumerate(doc_records):
            if record.region_id not in wanted:
                continue
            start = max(0, index - before)
            end = min(len(doc_records), index + after + 1)
            for context_record in doc_records[start:end]:
                key = (current_doc_id, context_record.region_id)
                if key in seen:
                    continue
                seen.add(key)
                expanded.append(context_record)
    return expanded


def align_region_records(
    source_records: Iterable[RegionRecord],
    target_records: Iterable[RegionRecord],
    *,
    top_k: int = 3,
) -> list[list[AlignmentCandidate]]:
    """Map each source region to the best lexical target-region candidates."""
    targets = list(target_records)
    results: list[list[AlignmentCandidate]] = []
    for source in source_records:
        source_terms = _token_set(source.text)
        candidates: list[AlignmentCandidate] = []
        for target in targets:
            target_terms = _token_set(target.text)
            shared = sorted(source_terms & target_terms)
            union = source_terms | target_terms
            score = len(shared) / len(union) if union else 0.0
            candidates.append(
                AlignmentCandidate(
                    source_doc_id=source.doc_id,
                    source_region_id=source.region_id,
                    target_doc_id=target.doc_id,
                    target_region_id=target.region_id,
                    score=round(score, 4),
                    shared_terms=shared[:20],
                    source_text=source.text,
                    target_text=target.text,
                )
            )
        candidates.sort(key=lambda item: (-item.score, item.target_doc_id, item.target_region_id))
        results.append(candidates[:top_k])
    return results


def evaluate_alignment_coverage(
    source_records: Iterable[RegionRecord],
    target_records: Iterable[RegionRecord],
    *,
    top_k: int = 3,
    threshold: float = 0.18,
    expand_before: int = 0,
    expand_after: int = 1,
    numeric_checks: bool = True,
) -> list[CoverageItem]:
    """Evaluate source-region coverage by target-region candidates."""
    targets = list(target_records)
    alignments = align_region_records(source_records, targets, top_k=top_k)
    items: list[CoverageItem] = []
    for source, candidates in zip(source_records, alignments):
        best = candidates[0] if candidates else None
        source_terms = _content_terms(source.text)
        naive_score = _term_recall(source_terms, best.target_text if best else "")
        expanded_records = (
            expand_region_context(
                targets,
                [best.target_region_id],
                doc_id=best.target_doc_id,
                before=expand_before,
                after=expand_after,
            )
            if best
            else []
        )
        expanded_text = "\n".join(record.text for record in expanded_records)
        expanded_score = _term_recall(source_terms, expanded_text)
        coverage_score = max(best.score if best else 0.0, expanded_score)
        numeric_conflict = (
            _has_numeric_conflict(source.text, expanded_text if expanded_text else (best.target_text if best else ""))
            if numeric_checks
            else False
        )
        status = "covered" if coverage_score >= threshold and not numeric_conflict else "gap"
        covered_terms = _token_set(expanded_text if expanded_text else (best.target_text if best else ""))
        missing_terms = sorted(source_terms - covered_terms)
        items.append(
            CoverageItem(
                source=source,
                candidates=candidates,
                status=status,
                coverage_score=round(coverage_score, 4),
                naive_score=round(naive_score, 4),
                expanded_score=round(expanded_score, 4),
                expanded_target_refs=[record.region_id for record in expanded_records],
                expanded_targets=expanded_records,
                numeric_conflict=numeric_conflict,
                missing_terms=missing_terms[:20],
            )
        )
    return items


def render_coverage_html(items: Iterable[CoverageItem], *, title: str = "Refmark Coverage Review") -> str:
    """Render a small HTML review report with covered and gap regions."""
    return render_coverage_report_html(items, title=title, layout="stacked")


def render_coverage_report_html(
    items: Iterable[CoverageItem],
    *,
    title: str = "Refmark Coverage Review",
    layout: str = "side-by-side",
    include_expanded_evidence: bool = True,
) -> str:
    """Render a coverage report as stacked cards or side-by-side review rows."""
    item_list = list(items)
    summary = summarize_coverage(item_list)
    rows: list[str] = []
    for item in item_list:
        badge = "covered" if item.status == "covered" else "gap"
        candidate_cards = []
        for candidate in item.candidates:
            candidate_cards.append(
                "<div class='candidate'>"
                f"<div class='ref'>{html.escape(candidate.target_doc_id)}:{html.escape(candidate.target_region_id)} "
                f"score {candidate.score:.4f}</div>"
                f"<pre>{html.escape(candidate.target_text)}</pre>"
                "</div>"
            )
        expanded_cards = []
        if include_expanded_evidence:
            for target in item.expanded_targets:
                expanded_cards.append(
                    "<div class='expanded'>"
                    f"<div class='ref'>{html.escape(target.doc_id)}:{html.escape(target.region_id)}</div>"
                    f"<pre>{html.escape(target.text)}</pre>"
                    "</div>"
                )
        details = (
            f"<p>naive: <strong>{item.naive_score:.4f}</strong> | expanded: <strong>{item.expanded_score:.4f}</strong> | "
            f"coverage: <strong>{item.coverage_score:.4f}</strong></p>"
            f"<p>expanded refs: {html.escape(', '.join(item.expanded_target_refs) or 'none')}</p>"
            f"<p>numeric conflict: <strong>{html.escape(str(item.numeric_conflict).lower())}</strong></p>"
            f"<p>missing terms: {html.escape(', '.join(item.missing_terms) or 'none')}</p>"
        )
        if layout == "side-by-side":
            rows.append(
                "<section class='item side'>"
                "<div class='left'>"
                f"<div class='status {badge}'>{html.escape(item.status.upper())}</div>"
                f"<h2>{html.escape(item.source.doc_id)}:{html.escape(item.source.region_id)}</h2>"
                f"<pre>{html.escape(item.source.text)}</pre>"
                + details
                + "</div><div class='right'>"
                + ("<h3>Expanded Evidence</h3>" + "".join(expanded_cards) if include_expanded_evidence else "")
                + "<h3>Top Candidates</h3>"
                + "".join(candidate_cards)
                + "</div></section>"
            )
        else:
            rows.append(
                "<section class='item'>"
                f"<div class='status {badge}'>{html.escape(item.status.upper())}</div>"
                f"<h2>{html.escape(item.source.doc_id)}:{html.escape(item.source.region_id)}</h2>"
                f"<pre>{html.escape(item.source.text)}</pre>"
                + details
                + ("<h3>Expanded Evidence</h3>" + "".join(expanded_cards) if include_expanded_evidence else "")
                + "<h3>Top Candidates</h3>"
                + "".join(candidate_cards)
                + "</section>"
            )

    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title>"
        "<style>"
        "body{font-family:Arial,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#1f2933}"
        "h1{font-size:28px}h2{font-size:18px;margin:4px 0 10px}"
        ".item{border:1px solid #d8dee7;border-radius:8px;padding:16px;margin:16px 0;background:#fff}"
        ".side{display:grid;grid-template-columns:minmax(0,0.9fr) minmax(0,1.1fr);gap:18px;align-items:start}"
        "@media(max-width:800px){.side{grid-template-columns:1fr}}"
        ".status{display:inline-block;padding:4px 8px;border-radius:999px;font-size:12px;font-weight:700}"
        ".covered{background:#dcfce7;color:#166534}.gap{background:#fee2e2;color:#991b1b}"
        "pre{white-space:pre-wrap;background:#f7f9fc;border-left:4px solid #9fb4d8;padding:10px;line-height:1.45}"
        ".summary{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}.metric{background:#f7f9fc;border:1px solid #d8dee7;border-radius:8px;padding:10px 12px}.metric strong{display:block;font-size:20px}"
        "h3{font-size:14px;margin:16px 0 6px;color:#334e68}"
        ".candidate,.expanded{margin-top:10px}.candidate .ref,.expanded .ref{font-weight:700;color:#334e68}"
        ".expanded pre{border-left-color:#38a169}"
        "</style></head><body>"
        f"<h1>{html.escape(title)}</h1>"
        "<div class='summary'>"
        f"<div class='metric'><strong>{summary['requirements']}</strong>requirements</div>"
        f"<div class='metric'><strong>{summary['covered']}</strong>covered</div>"
        f"<div class='metric'><strong>{summary['gaps']}</strong>gaps</div>"
        f"<div class='metric'><strong>{summary['coverage_rate']:.2f}</strong>coverage rate</div>"
        f"<div class='metric'><strong>{summary['items_improved_by_expansion']}</strong>improved by expansion</div>"
        "</div>"
        + "".join(rows)
        + "</body></html>"
    )


def summarize_coverage(items: Iterable[CoverageItem]) -> dict[str, float | int | list[str]]:
    item_list = list(items)
    total = len(item_list)
    covered = sum(1 for item in item_list if item.status == "covered")
    improved = sum(1 for item in item_list if item.expanded_score > item.naive_score)
    return {
        "requirements": total,
        "covered": covered,
        "gaps": total - covered,
        "coverage_rate": round(covered / total, 4) if total else 0.0,
        "items_improved_by_expansion": improved,
        "gap_refs": [item.source.region_id for item in item_list if item.status == "gap"],
    }


def _stable_text_hash(text: str) -> str:
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _stable_ref(record: RegionRecord) -> str:
    return f"{record.doc_id}:{record.region_id}"


def _heading_level(text: str) -> int | None:
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    markdown = re.match(r"^(#{1,6})\s+\S", first_line)
    if markdown:
        return len(markdown.group(1))
    if first_line and set(first_line) <= {"=", "-"}:
        return None
    return None


def _heading_title(text: str) -> str:
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    return re.sub(r"^#{1,6}\s+", "", first_line).strip() or first_line or "Untitled section"


def _token_set(text: str) -> set[str]:
    return {
        normalized
        for token in re.findall(r"[A-Za-z0-9_]{2,}", text.lower())
        for normalized in (_normalize_token(token),)
        if normalized and normalized not in _STOPWORDS
    }


def _content_terms(text: str) -> set[str]:
    return {token for token in _token_set(text) if not token.isdigit()}


def _term_recall(source_terms: set[str], target_text: str) -> float:
    if not source_terms:
        return 0.0
    target_terms = _token_set(target_text)
    return len(source_terms & target_terms) / len(source_terms)


def _has_numeric_conflict(source_text: str, target_text: str) -> bool:
    source_pairs = _number_unit_pairs(source_text)
    source_numbers = [number for number, _unit in source_pairs]
    if not source_pairs:
        return False
    target_pairs = _number_unit_pairs(target_text)
    target_numbers = [number for number, _unit in target_pairs]
    if not target_numbers:
        return False

    lowered = source_text.lower()
    for source_number, source_unit in source_pairs:
        comparable = [
            target_number
            for target_number, target_unit in target_pairs
            if source_unit is None or target_unit is None or source_unit == target_unit
        ]
        if not comparable:
            comparable = target_numbers
        if "at least" in lowered or "minimum" in lowered:
            if any(target_number >= source_number for target_number in comparable):
                continue
            return True
        if "within" in lowered or "no more than" in lowered or "maximum" in lowered:
            if any(target_number <= source_number for target_number in comparable):
                continue
            return True
        if not any(target_number == source_number for target_number in comparable):
            return True
    return False


def _number_unit_pairs(text: str) -> list[tuple[float, str | None]]:
    pairs: list[tuple[float, str | None]] = []
    for match in re.finditer(r"\b(\d+(?:\.\d+)?)\b(?:\s+([A-Za-z%]+))?", text.lower()):
        unit = match.group(2)
        pairs.append((float(match.group(1)), _normalize_unit(unit) if unit else None))
    return pairs


def _normalize_unit(unit: str) -> str:
    aliases = {
        "day": "day",
        "days": "day",
        "year": "year",
        "years": "year",
        "hour": "hour",
        "hours": "hour",
        "kwh": "kwh",
        "percent": "percent",
        "%": "percent",
    }
    return aliases.get(unit, unit.rstrip("s"))


_STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "this",
    "from",
    "into",
    "are",
    "was",
    "shall",
    "must",
    "provide",
    "supplier",
    "vendor",
    "include",
    "including",
    "at",
    "least",
    "require",
    "requires",
    "tender",
}

_TOKEN_ALIASES = {
    "eu": "europe",
    "european": "europe",
    "encrypt": "encryption",
    "encrypted": "encryption",
    "expedited": "expedited",
    "shipping": "shipping",
    "cryptography": "encryption",
    "residency": "residence",
    "resident": "residence",
    "regions": "region",
    "accounts": "account",
    "customers": "customer",
    "refunds": "refund",
    "credits": "credit",
    "completed": "complete",
    "outages": "outage",
    "invoices": "invoice",
    "totals": "total",
    "fees": "fee",
    "exports": "export",
    "including": "include",
}


def _normalize_token(token: str) -> str:
    if token in _TOKEN_ALIASES:
        return _TOKEN_ALIASES[token]
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token
