from __future__ import annotations

import io
import json

import refmark.discovery as discovery_module
from refmark.discovery import (
    build_discovery_context_card,
    build_discovery_windows,
    discover_corpus,
    discover_corpus_windowed,
    discovery_excluded_refs,
    discovery_terms_for_refs,
    load_discovery,
    repair_discovery_clusters,
    review_discovery,
    write_discovery,
)
from refmark.discovery_heatmap import discovery_map_items, render_discovery_map_html
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


def test_discovery_tag_graph_clusters_flat_wiki_topics():
    records = [
        _record("P01", "OAuth login tokens secure API access and password bearer flows.", 0, doc_id="wiki"),
        _record("P02", "JWT token refresh and OAuth scopes protect endpoints.", 1, doc_id="wiki"),
        _record("P03", "Docker containers deploy FastAPI services with images.", 2, doc_id="wiki"),
        _record("P04", "Container deployment uses Docker Compose and image tags.", 3, doc_id="wiki"),
        _record("P05", "SQL databases use sessions, models, and migrations.", 4, doc_id="wiki"),
        _record("P06", "Database migrations update SQL schemas and models.", 5, doc_id="wiki"),
    ]

    discovery = discover_corpus(records, cluster_strategy="tag_graph", target_clusters=4)

    assert discovery.clusters
    assert {ref for cluster in discovery.clusters for ref in cluster.refs} == {f"wiki:P0{i}" for i in range(1, 7)}
    assert all(cluster.strategy == "tag_graph" for cluster in discovery.clusters)
    assert len(discovery.clusters) <= 4


def test_discovery_balanced_term_clusters_are_bounded_and_complete():
    records = [
        _record(f"P{i:02d}", f"Topic {i % 3} alpha beta service cluster region {i}", i, doc_id=f"doc{i}")
        for i in range(12)
    ]

    discovery = discover_corpus(records, cluster_strategy="balanced_terms", target_clusters=5)

    assert 1 <= len(discovery.clusters) <= 5
    assert {ref for cluster in discovery.clusters for ref in cluster.refs} == {f"doc{i}:P{i:02d}" for i in range(12)}
    assert all(cluster.strategy == "balanced_terms" for cluster in discovery.clusters)


def test_llm_cluster_strategy_has_local_fallback_for_ci():
    records = [
        _record("P01", "OAuth login tokens secure API access.", 0, doc_id="wiki"),
        _record("P02", "Docker containers deploy services.", 1, doc_id="wiki"),
    ]

    discovery = discover_corpus(records, cluster_strategy="llm_topics", target_clusters=2)

    assert discovery.clusters
    assert {ref for cluster in discovery.clusters for ref in cluster.refs} == {"wiki:P01", "wiki:P02"}
    assert any("requires source=openrouter" in note for note in discovery.notes)


def test_openrouter_llm_clusters_can_fall_back_to_query_families(monkeypatch):
    records = [
        _record("P01", "OAuth login tokens secure API access.", 0, doc_id="wiki"),
        _record("P02", "Docker containers deploy services.", 1, doc_id="wiki"),
    ]
    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "corpus_summary": "Two documentation regions.",
                            "query_families": [
                                {"name": "API security", "refs": ["[P01]"], "terms": ["OAuth"]},
                                {"name": "Deployment packaging", "refs": ["wiki:P02"], "terms": ["Docker"]},
                            ],
                        }
                    )
                }
            }
        ]
    }

    class FakeResponse:
        def __enter__(self):
            return io.BytesIO(json.dumps(payload).encode("utf-8"))

        def __exit__(self, *_args):
            return False

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(discovery_module.request, "urlopen", lambda *_args, **_kwargs: FakeResponse())

    discovery = discover_corpus(records, source="openrouter", model="fake", cluster_strategy="llm_topics", target_clusters=4)

    assert [cluster.name for cluster in discovery.clusters[:2]] == ["API security", "Deployment packaging"]
    assert {ref for cluster in discovery.clusters for ref in cluster.refs} == {"wiki:P01", "wiki:P02"}
    assert all(cluster.strategy == "llm_topics" for cluster in discovery.clusters)


def test_repair_discovery_clusters_sanitizes_agent_output(monkeypatch):
    records = [
        _record("P01", "OAuth login tokens secure API access.", 0, doc_id="wiki"),
        _record("P02", "Docker containers deploy services.", 1, doc_id="wiki"),
        _record("P03", "SQL migrations update database models.", 2, doc_id="wiki"),
    ]
    discovery = discover_corpus(records, cluster_strategy="llm_topics", target_clusters=2)
    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "clusters": [
                                {"cluster_id": "auth", "name": "Authentication", "refs": ["[wiki:P01]"], "terms": ["OAuth"]},
                                {"cluster_id": "infra", "name": "Deployment and data", "refs": ["P02", "wiki:P03", "missing:P99"], "terms": ["Docker", "SQL"]},
                            ],
                            "notes": ["Split the broad cluster into reviewable task groups."],
                        }
                    )
                }
            }
        ]
    }

    class FakeResponse:
        def __enter__(self):
            return io.BytesIO(json.dumps(payload).encode("utf-8"))

        def __exit__(self, *_args):
            return False

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(discovery_module.request, "urlopen", lambda *_args, **_kwargs: FakeResponse())

    repaired = repair_discovery_clusters(discovery, records, model="fake", cluster_strategy="llm_topics", target_clusters=3)

    assert [cluster.name for cluster in repaired.clusters] == ["Authentication", "Deployment and data"]
    assert {ref for cluster in repaired.clusters for ref in cluster.refs} == {"wiki:P01", "wiki:P02", "wiki:P03"}
    assert any("Cluster repair model: fake" in note for note in repaired.notes)


def test_repair_discovery_clusters_respects_target_count(monkeypatch):
    records = [_record(f"P0{i}", f"Topic {i} content about a specific area.", i, doc_id="wiki") for i in range(5)]
    discovery = discover_corpus(records, cluster_strategy="llm_topics", target_clusters=5)
    clusters = [
        {"cluster_id": f"c{i}", "name": f"Topic {i}", "refs": [f"P0{i}"], "terms": [f"topic-{i}"]}
        for i in range(5)
    ]
    payload = {"choices": [{"message": {"content": json.dumps({"clusters": clusters, "notes": []})}}]}

    class FakeResponse:
        def __enter__(self):
            return io.BytesIO(json.dumps(payload).encode("utf-8"))

        def __exit__(self, *_args):
            return False

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(discovery_module.request, "urlopen", lambda *_args, **_kwargs: FakeResponse())

    repaired = repair_discovery_clusters(discovery, records, model="fake", cluster_strategy="llm_topics", target_clusters=3)

    assert len(repaired.clusters) == 3
    assert {ref for cluster in repaired.clusters for ref in cluster.refs} == {f"wiki:P0{i}" for i in range(5)}


def test_discovery_cluster_map_renders_human_review_html():
    records = [
        _record("P01", "OAuth login tokens secure API access.", 0, doc_id="wiki"),
        _record("P02", "Docker containers deploy services.", 1, doc_id="wiki"),
    ]
    discovery = discover_corpus(records, cluster_strategy="balanced_terms", target_clusters=2)

    items = discovery_map_items(records, discovery)
    html = render_discovery_map_html(records, discovery, title="Wiki Map")

    assert items
    assert "Wiki Map" in html
    assert "<title>Wiki Map</title>" in html
    assert "const DATA=" in html
    assert "function drill(item)" in html
    assert "Ordered layout" in html
    assert "function orderedTreemap" in html
    assert "Contained Blocks" in html
    assert "OAuth login tokens" in html or "Docker containers" in html


def test_discovery_cluster_map_extracts_rst_titles():
    records = [
        _record("P01", "=====================\nDjango API reference\n=====================\nBody text.", 0, doc_id="docs"),
    ]
    discovery = discover_corpus(records)

    items = discovery_map_items(records, discovery)
    html = render_discovery_map_html(records, discovery, title="RST Map")

    assert items[0].blocks[0]["title"] == "Django API reference"
    assert "=====================" not in items[0].blocks[0]["title"]
    assert 'id="breadcrumbs" class="breadcrumbsbar"' in html


def test_discovery_cluster_map_keeps_hierarchical_clusters():
    records = [
        _record("P01", "OAuth login tokens secure API access.", 0, doc_id="wiki"),
        _record("P02", "JWT bearer tokens protect routes.", 1, doc_id="wiki"),
    ]
    discovery = discovery_module.DiscoveryManifest(
        schema="refmark.discovery.v1",
        created_at="2026-01-01T00:00:00Z",
        mode="test",
        source="local",
        model="test",
        corpus_summary="Test hierarchy.",
        corpus_tokens=20,
        regions=2,
        clusters=[
            discovery_module.DiscoveryCluster(
                cluster_id="security",
                name="Security",
                refs=["wiki:P01", "wiki:P02"],
                strategy="manual",
            ),
            discovery_module.DiscoveryCluster(
                cluster_id="security:tokens",
                name="Tokens",
                refs=["wiki:P01", "wiki:P02"],
                strategy="manual",
                parent_id="security",
            ),
        ],
    )

    items = discovery_map_items(records, discovery)
    html = render_discovery_map_html(records, discovery, title="Hierarchy Map")

    assert {item.cluster_id: item.parent_id for item in items}["security:tokens"] == "security"
    assert '"parent_id": "security"' in html
    assert "const CHILDREN=new Map()" in html
    assert "child clusters" in html


def test_discovery_cluster_map_escapes_script_closers():
    records = [
        _record("P01", "JavaScript example\n<script>console.log('x')</script>\nBody text.", 0, doc_id="docs"),
    ]
    discovery = discover_corpus(records)

    html = render_discovery_map_html(records, discovery, title="Script Map")
    script_body = html.split("<script>", 1)[1].split("</script>", 1)[0]

    assert "<\\/script>" in script_body
    assert "</script>" not in script_body


def test_source_tree_clusters_create_parent_child_hierarchy():
    records = [
        RegionRecord("docs_ref_models_querysets_txt", "P01", "QuerySet API reference.", 1, 1, 0, "h0", "corpus/django/docs_ref_models_querysets_txt.txt"),
        RegionRecord("docs_ref_models_fields_txt", "P02", "Model fields reference.", 2, 2, 1, "h1", "corpus/django/docs_ref_models_fields_txt.txt"),
        RegionRecord("docs_howto_auth_remote_user_txt", "P03", "Authentication how-to.", 3, 3, 2, "h2", "corpus/django/docs_howto_auth_remote_user_txt.txt"),
    ]

    discovery = discover_corpus(records, cluster_strategy="source_tree")

    by_id = {cluster.cluster_id: cluster for cluster in discovery.clusters}
    assert "source:reference" in by_id
    assert "source:how-to" in by_id
    assert by_id["source:reference:reference-models"].parent_id == "source:reference"
    assert by_id["source:how-to:how-to-auth"].parent_id == "source:how-to"


def test_source_tree_clusters_use_document_headings_for_singleton_children():
    records = [
        RegionRecord(
            "src_ch05_01_md",
            "P01",
            "## Defining and Instantiating Structs\n\nStructs let you name related values.",
            1,
            2,
            0,
            "h0",
            "corpus/rust_book/src_ch05_01_md.txt",
        ),
        RegionRecord(
            "src_ch05_02_md",
            "P01",
            "## An Example Program Using Structs\n\nThis section builds a rectangle example.",
            3,
            4,
            1,
            "h1",
            "corpus/rust_book/src_ch05_02_md.txt",
        ),
    ]

    discovery = discover_corpus(records, cluster_strategy="source_tree")

    child_names = {cluster.name for cluster in discovery.clusters if cluster.parent_id}
    assert "Defining and Instantiating Structs" in child_names
    assert "An Example Program Using Structs" in child_names
    assert "Ch05 / 01" not in child_names


def test_source_tree_clusters_use_document_heading_for_chapter_parents():
    records = [
        RegionRecord(
            "src_ch05_00_md",
            "P01",
            "# Using Structs to Structure Related Data\n\nA struct is a custom data type.",
            1,
            2,
            0,
            "h0",
            "corpus/rust_book/src_ch05_00_md.txt",
        ),
        RegionRecord(
            "src_ch05_01_md",
            "P01",
            "## Defining and Instantiating Structs\n\nStructs let you name related values.",
            3,
            4,
            1,
            "h1",
            "corpus/rust_book/src_ch05_01_md.txt",
        ),
    ]

    discovery = discover_corpus(records, cluster_strategy="source_tree")

    parent_names = {cluster.name for cluster in discovery.clusters if not cluster.parent_id}
    assert "Using Structs to Structure Related Data" in parent_names
    assert "Ch05" not in parent_names


def test_source_tree_cluster_terms_prefer_heading_phrases_over_unigrams():
    records = [
        RegionRecord(
            "src_appendix_02_operators_md",
            "P01",
            "## Appendix B: Operators and Symbols\n\nThis appendix contains a glossary of Rust syntax.",
            1,
            2,
            0,
            "h0",
            "corpus/rust_book/src_appendix_02_operators_md.txt",
        ),
        RegionRecord(
            "src_appendix_04_useful_development_tools_md",
            "P01",
            "## Appendix D: Useful Development Tools\n\nThis appendix talks about development tools.",
            3,
            4,
            1,
            "h1",
            "corpus/rust_book/src_appendix_04_useful_development_tools_md.txt",
        ),
    ]

    discovery = discover_corpus(records, cluster_strategy="source_tree")
    terms_by_name = {cluster.name: cluster.terms for cluster in discovery.clusters}

    assert "operator-symbol" in terms_by_name["Appendix B: Operators and Symbols"]
    assert "development-tool" in terms_by_name["Appendix D: Useful Development Tools"]
    assert "appendix" not in terms_by_name["Appendix B: Operators and Symbols"]
    assert "book" not in terms_by_name["Appendix B: Operators and Symbols"]
