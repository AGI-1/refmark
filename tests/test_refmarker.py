from refmark.refmarker import Refmarker, RefmarkRegistry


def test_refmarker_live_mode_embeds_markers():
    result = Refmarker(mode="live").mark_text("First requirement.\n\nSecond requirement.\n", doc_id="policy")

    assert result.namespace_mode == "live"
    assert result.content == result.marked_view
    assert "[@P01]" in result.content
    assert [record.region_id for record in result.records] == ["P01", "P02"]


def test_refmarker_shadow_mode_preserves_content_and_persists_view(tmp_path):
    registry = RefmarkRegistry(tmp_path / "registry")
    marker = Refmarker(mode="shadow", registry=registry)
    content = "First requirement.\n\nSecond requirement.\n"

    first = marker.mark_text(content, doc_id="policy")
    second = marker.mark_text(content, doc_id="policy")

    assert first.namespace_mode == "shadow"
    assert first.content == content
    assert "[@P01]" in first.marked_view
    assert first.registry_path is not None
    assert first.session_reset is True
    assert second.session_reset is False
    assert second.registry_path == first.registry_path
    assert second.marked_view == first.marked_view


def test_refmarker_detects_existing_live_markers_without_double_marking(tmp_path):
    marker = Refmarker(mode="shadow", registry_path=tmp_path / "registry")
    marked = "[@P01] First requirement.\n\n[@P02] Second requirement.\n"

    result = marker.mark_text(marked, doc_id="policy")

    assert result.namespace_mode == "live"
    assert result.content == marked
    assert result.marked_view == marked
    assert marked.count("[@P01]") == 1
    assert [record.region_id for record in result.records] == ["P01", "P02"]


def test_refmarker_changed_shadow_source_creates_new_registry_session(tmp_path):
    marker = Refmarker(mode="shadow", registry_path=tmp_path / "registry")

    first = marker.mark_text("First requirement.\n", doc_id="policy")
    changed = marker.mark_text("First requirement.\n\nNew requirement.\n", doc_id="policy")

    assert changed.registry_path != first.registry_path
    assert changed.session_reset is True
    assert [record.region_id for record in changed.records] == ["P01", "P02"]
