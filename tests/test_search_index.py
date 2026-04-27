from pathlib import Path

from refmark.search_index import (
    RetrievalView,
    build_search_index,
    export_browser_search_index,
    generate_views,
    load_search_index,
    map_corpus,
    tokenize,
)


def test_map_corpus_assigns_stable_document_refs(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("First region explains alpha.\n\nSecond region explains beta.\n", encoding="utf-8")

    records = map_corpus(docs, min_words=2)

    assert [record.doc_id for record in records] == ["guide", "guide"]
    assert [record.region_id for record in records] == ["P01", "P02"]
    assert records[0].next_region_id == "P02"
    assert records[1].prev_region_id == "P01"


def test_map_corpus_can_exclude_generated_artifacts(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("Keep this source document.\n", encoding="utf-8")
    (docs / "guide_combined.txt").write_text("Skip this generated aggregate.\n", encoding="utf-8")

    records = map_corpus(docs, min_words=2, exclude_globs=["*_combined.txt"])

    assert [record.doc_id for record in records] == ["guide"]


def test_map_corpus_recomputes_neighbors_after_filtered_regions(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text(
        "Keep this alpha deployment region.\n\n"
        "Tiny.\n\n"
        "Keep this beta rollback region.\n",
        encoding="utf-8",
    )

    records = map_corpus(docs, min_words=4)

    assert [record.region_id for record in records] == ["P01", "P03"]
    assert records[0].prev_region_id is None
    assert records[0].next_region_id == "P03"
    assert records[1].prev_region_id == "P01"
    assert records[1].next_region_id is None


def test_tokenize_keeps_unicode_terms_for_multilingual_docs():
    assert tokenize("Verjährungsfrist und Geschäftsfähigkeit") == [
        "verjährungsfrist",
        "und",
        "geschäftsfähigkeit",
    ]


def test_build_and_search_local_portable_index(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "security.md").write_text(
        "Tokens rotate every 90 days after replacement credentials are deployed.\n\n"
        "Audit logs are retained for 180 days by default.\n",
        encoding="utf-8",
    )
    index_path = tmp_path / "index.json"

    payload = build_search_index(docs, index_path, source="local", min_words=3)
    index = load_search_index(index_path)
    hits = index.search("How long are audit logs retained?", top_k=1)

    assert payload["stats"]["regions"] == 2
    assert hits
    assert hits[0].stable_ref == "security:P02"
    assert "Audit logs" in hits[0].text


def test_export_browser_search_index_contains_bm25_payload(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "security.md").write_text(
        "Tokens rotate every 90 days after replacement credentials are deployed.\n\n"
        "Audit logs are retained for 180 days by default.\n",
        encoding="utf-8",
    )
    index_path = tmp_path / "index.json"
    browser_path = tmp_path / "browser-index.json"

    build_search_index(docs, index_path, source="local", min_words=3)
    payload = export_browser_search_index(index_path, browser_path, max_text_chars=32)

    assert browser_path.exists()
    assert payload["schema"] == "refmark.browser_search_index.v1"
    assert payload["stats"]["regions"] == 2
    assert payload["regions"][1]["stable_ref"] == "security:P02"
    assert len(payload["regions"][1]["text"]) <= 32
    assert "audit" in payload["postings"]
    audit_postings = payload["postings"]["audit"]["p"]
    assert audit_postings[0][0] == 1
    assert audit_postings[0][1] > 0


def test_search_hierarchical_and_rerank_return_region(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "deploy.md").write_text(
        "Production releases happen between 09:00 and 15:00 UTC.\n\n"
        "Rollback plans keep the previous image available.\n",
        encoding="utf-8",
    )
    index_path = tmp_path / "index.json"

    build_search_index(docs, index_path, source="local", min_words=3)
    index = load_search_index(index_path)

    hierarchical = index.search_hierarchical("When are production releases?", top_k=1)
    reranked = index.search_reranked("When are production releases?", top_k=1)

    assert hierarchical[0].stable_ref == "deploy:P01"
    assert reranked[0].stable_ref == "deploy:P01"


def test_openrouter_view_cache_reuses_unchanged_regions(tmp_path: Path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("Alpha deployment region.\n\nBeta rollback region.\n", encoding="utf-8")
    records = map_corpus(docs, min_words=2)
    cache_path = tmp_path / "views.jsonl"
    calls = []

    def fake_openrouter_view(record, **_kwargs):
        calls.append(record.region_id)
        return RetrievalView(summary=f"summary {record.region_id}", questions=[], keywords=[])

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr("refmark.search_index.openrouter_view", fake_openrouter_view)

    first = generate_views(
        records,
        source="openrouter",
        model="fake-model",
        endpoint="https://example.invalid",
        api_key_env="OPENROUTER_API_KEY",
        questions_per_region=1,
        keywords_per_region=1,
        concurrency=1,
        sleep=0.0,
        cache_path=cache_path,
    )
    second = generate_views(
        records,
        source="openrouter",
        model="fake-model",
        endpoint="https://example.invalid",
        api_key_env="OPENROUTER_API_KEY",
        questions_per_region=1,
        keywords_per_region=1,
        concurrency=1,
        sleep=0.0,
        cache_path=cache_path,
    )

    assert calls == ["P01", "P02"]
    assert first == second
