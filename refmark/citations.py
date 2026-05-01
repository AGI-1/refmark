"""Strict parsing helpers for Refmark citation refs and ranges."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable


REGION_ID_PATTERN = r"[A-Za-z][A-Za-z0-9_]*\d+[A-Za-z0-9_]*"
REF_TOKEN_RE = re.compile(rf"(?:(?P<doc>[A-Za-z0-9_.-]+):)?(?P<region>{REGION_ID_PATTERN})")
RANGE_RE = re.compile(
    rf"^\s*(?:(?P<start_doc>[A-Za-z0-9_.-]+):)?(?P<start>{REGION_ID_PATTERN})\s*(?P<op>-|\.\.)\s*"
    rf"(?:(?P<end_doc>[A-Za-z0-9_.-]+):)?(?P<end>{REGION_ID_PATTERN})\s*$"
)


@dataclass(frozen=True)
class CitationRef:
    """One parsed citation token.

    `end_ref` is inclusive when present. Edit APIs use different range
    semantics: `end_ref` in edit calls is an exclusive stop boundary.
    """

    ref: str
    doc_id: str | None = None
    end_ref: str | None = None
    end_doc_id: str | None = None

    @property
    def is_range(self) -> bool:
        return self.end_ref is not None

    @property
    def stable_ref(self) -> str:
        return f"{self.doc_id}:{self.ref}" if self.doc_id else self.ref

    @property
    def stable_end_ref(self) -> str | None:
        if self.end_ref is None:
            return None
        doc_id = self.end_doc_id or self.doc_id
        return f"{doc_id}:{self.end_ref}" if doc_id else self.end_ref

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def parse_citation_refs(value: str | Iterable[str]) -> list[CitationRef]:
    """Parse strict Refmark citation refs from text or a token iterable.

    Accepted token forms:

    - `P03`
    - `policy:P03`
    - `P03-P05` or `P03..P05` for inclusive same-doc ranges
    - `policy:P03-P05` or `policy:P03-policy:P05`
    - bracketed groups such as `[P03, P05-P07]`
    """

    tokens = _tokens(value)
    return [_parse_token(token) for token in tokens]


def citation_refs_to_strings(refs: Iterable[CitationRef]) -> list[str]:
    result: list[str] = []
    for ref in refs:
        if ref.is_range:
            result.append(f"{ref.stable_ref}-{ref.stable_end_ref}")
        else:
            result.append(ref.stable_ref)
    return result


def validate_citation_refs(
    value: str | Iterable[str],
    *,
    address_space: Iterable[str],
) -> dict[str, object]:
    """Validate citations against a known address space without expanding them."""

    parsed = parse_citation_refs(value)
    known = set(address_space)
    missing: list[str] = []
    for item in parsed:
        if item.stable_ref not in known:
            missing.append(item.stable_ref)
        if item.stable_end_ref and item.stable_end_ref not in known:
            missing.append(item.stable_end_ref)
    return {
        "ok": not missing,
        "refs": [item.to_dict() for item in parsed],
        "missing": missing,
    }


def _tokens(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        raw_tokens = re.split(r"\s*,\s*", text)
    else:
        raw_tokens = []
        for item in value:
            text = str(item).strip()
            if text.startswith("[") and text.endswith("]"):
                text = text[1:-1]
            raw_tokens.extend(re.split(r"\s*,\s*", text))
    return [token.strip() for token in raw_tokens if token.strip()]


def _parse_token(token: str) -> CitationRef:
    range_match = RANGE_RE.fullmatch(token)
    if range_match:
        start_doc = range_match.group("start_doc")
        end_doc = range_match.group("end_doc")
        if start_doc and end_doc and start_doc != end_doc:
            raise ValueError(f"Citation ranges cannot cross documents: {token}")
        return CitationRef(
            doc_id=start_doc,
            ref=_normalize_ref(range_match.group("start")),
            end_doc_id=end_doc or start_doc,
            end_ref=_normalize_ref(range_match.group("end")),
        )

    match = REF_TOKEN_RE.fullmatch(token)
    if not match:
        raise ValueError(f"Invalid citation ref: {token}")
    return CitationRef(doc_id=match.group("doc"), ref=_normalize_ref(match.group("region")))


def _normalize_ref(value: str) -> str:
    text = value.strip()
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", text)
    if not match:
        return text.upper()
    prefix, digits = match.groups()
    width = max(2, len(digits))
    return f"{prefix.upper()}{int(digits):0{width}d}"
