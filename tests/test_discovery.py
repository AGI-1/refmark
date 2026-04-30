from __future__ import annotations

from refmark.discovery import (
    build_discovery_context_card,
    build_discovery_windows,
    discover_corpus,
    discover_corpus_windowed,
    discovery_excluded_refs,
    discovery_terms_for_refs,
    load_discovery,
    review_discovery,
    write_discovery,
)
from refmark.pipeline import RegionRecord


def _record(region_id: str, text: str, ordinal: int, *, doc_id: str = "policy") -> RegionRecord:
    return RegionRecord(
        doc_id=doc_id,
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


def test_discovery_context_card_guides_question_generation():
    records = [
        _record("P01", "# Safety", 0),
        _record("P02", "SDS means safety data sheet for hazardous chemicals.", 1),
        _record("P03", "The employer shall train employees before chemical exposure.", 2),
    ]
    records[1] = RegionRecord(**{**records[1].to_dict(), "prev_region_id": "P01", "next_region_id": "P03", "parent_region_id": "P01"})
    discovery = discover_corpus(records)

    card = build_discovery_context_card(discovery, records[1], records=records)

    assert card.stable_ref == "policy:P02"
    assert "definition" in card.roles
    assert "SDS" in card.abbreviations
    assert "policy:P01" in card.neighboring_refs
    assert card.parent_ref == "policy:P01"
    assert any("definition" in item for item in card.generation_guidance)


def test_review_discovery_flags_reviewable_noise():
    records = [
        _record("P01", "Table of contents\nDefinitions\nRules", 0),
        _record("P02", "SDS means safety data sheet for hazardous chemicals.", 1),
        _record("P03", "The employer shall train employees before chemical exposure.", 2),
    ]
    discovery = discover_corpus(records)

    issues = review_discovery(discovery, records=records)

    assert any(issue.kind == "excluded_region" for issue in issues)
    assert any(issue.kind == "singleton_term" for issue in issues)


def test_windowed_discovery_preserves_whole_regions_and_merges_outputs():
    records = [
        _record("P01", "Alpha controls protect the system from exposure.", 0),
        _record("P02", "Beta controls require operator training and review.", 1),
        _record("P03", "Gamma procedures document review exceptions.", 2),
        _record("P04", "Delta appendix contains navigation notes.", 3),
    ]

    windows = build_discovery_windows(records, window_tokens=8, overlap_regions=1)
    discovery = discover_corpus_windowed(records, window_tokens=8, overlap_regions=1)

    assert len(windows) > 1
    assert all(ref.startswith("policy:P") for window in windows for ref in window.refs)
    assert discovery.mode == "windowed"
    assert discovery.windows
    assert discovery.clusters
    assert {role.stable_ref for role in discovery.region_roles} == {f"policy:P0{i}" for i in range(1, 5)}
    assert any(item.term == "controls" for item in discovery.terms)


def test_discover_corpus_auto_windows_when_budget_is_smaller_than_corpus():
    records = [
        _record("P01", "Alpha control text for one region.", 0),
        _record("P02", "Beta training text for another region.", 1),
        _record("P03", "Gamma review text for final region.", 2),
    ]

    discovery = discover_corpus(records, window_tokens=6, overlap_regions=0)

    assert discovery.mode == "windowed"
    assert len(discovery.windows) >= 2


def test_discovery_excludes_release_note_query_magnets_from_default_question_plan():
    records = [
        _record(
            "P01",
            "Release Notes\n\nVersion 1.2 was released with pull request updates, contributors, and translation notes.",
            0,
            doc_id="release_notes",
        ),
        _record("P02", "Configure security settings for API tokens.", 1),
    ]

    discovery = discover_corpus(records)

    release_roles = next(role.roles for role in discovery.region_roles if role.stable_ref == "release_notes:P01")
    assert "query_magnet" in release_roles
    assert "exclude_from_default_search" in release_roles
    assert "exclude_from_training" in release_roles
    assert "release_notes:P01" in discovery_excluded_refs(discovery)
