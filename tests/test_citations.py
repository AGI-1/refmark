import pytest

from refmark.citations import citation_refs_to_strings, parse_citation_refs, validate_citation_refs


def test_parse_citation_refs_accepts_refs_ranges_and_doc_ids():
    parsed = parse_citation_refs("[P1, P03-P05, policy:P07-policy:P08]")

    assert citation_refs_to_strings(parsed) == ["P01", "P03-P05", "policy:P07-policy:P08"]
    assert parsed[1].is_range
    assert parsed[2].doc_id == "policy"
    assert parsed[2].end_doc_id == "policy"


def test_parse_citation_refs_accepts_underscore_region_ids():
    parsed = parse_citation_refs("[bgb:S_58, bgb:S_1687_A01-bgb:S_1687_A03]")

    assert citation_refs_to_strings(parsed) == ["bgb:S_58", "bgb:S_1687_A01-bgb:S_1687_A03"]
    assert parsed[1].is_range


def test_parse_citation_refs_rejects_cross_document_ranges():
    with pytest.raises(ValueError, match="cannot cross documents"):
        parse_citation_refs("policy:P01-contract:P02")


def test_validate_citation_refs_reports_missing_refs():
    result = validate_citation_refs(
        ["policy:P01", "policy:P02-P03"],
        address_space=["policy:P01", "policy:P02"],
    )

    assert result["ok"] is False
    assert result["missing"] == ["policy:P03"]
