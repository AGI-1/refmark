from __future__ import annotations

from refmark.discovery import (
    discover_corpus,
    discovery_excluded_refs,
    discovery_terms_for_refs,
    load_discovery,
    write_discovery,
)
from refmark.pipeline import RegionRecord


def _record(region_id: str, text: str, ordinal: int) -> RegionRecord:
    return RegionRecord(
        doc_id="policy",
        region_id=region_id,
        text=text,
        start_line=ordinal + 1,
        end_line=ordinal + 1,
        ordinal=ordinal,
        hash=f"h{ordinal}",
    )


def test_local_discovery_finds_terms_roles_and_exclusions(tmp_path):
    records = [
        _record("P01", "Table of contents\nDefinitions\nRules", 0),
        _record("P02", "SDS means safety data sheet for hazardous chemicals.", 1),
        _record("P03", "The employer shall train employees before chemical exposure.", 2),
    ]

    discovery = discover_corpus(records)

    assert discovery.schema == "refmark.discovery.v1"
    assert discovery.regions == 3
    assert any(item.term == "chemicals" for item in discovery.terms)
    assert any(item.term == "SDS" for item in discovery.abbreviations)
    excluded = discovery_excluded_refs(discovery)
    assert "policy:P01" in excluded
    assert discovery_terms_for_refs(discovery, ["policy:P02"])

    path = tmp_path / "discovery.json"
    write_discovery(discovery, path)
    loaded = load_discovery(path)
    assert loaded.to_dict() == discovery.to_dict()


def test_hierarchical_mode_uses_same_schema_for_now():
    records = [_record("P01", "The employer shall document exposure controls.", 0)]

    discovery = discover_corpus(records, mode="hierarchical")

    assert discovery.mode == "hierarchical"
    assert discovery.region_roles[0].stable_ref == "policy:P01"
