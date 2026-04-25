from refmark.documents import align_documents, map_document
from refmark.workflow_config import resolve_workflow_config


def test_map_document_uses_density_and_marker_style(tmp_path):
    path = tmp_path / "policy.txt"
    path.write_text("First requirement.\n\nSecond requirement.\n", encoding="utf-8")

    document = map_document(path, density="balanced", marker_style="explicit")

    assert document.doc_id == "policy"
    assert "[ref:P01]" in document.marked_text
    assert len(document.records) == 2
    assert document.expand(["P01"], after=1)[1].region_id == "P02"


def test_map_document_dense_density_uses_smaller_regions(tmp_path):
    path = tmp_path / "policy.txt"
    path.write_text("First line.\nSecond line.\nThird line.\n", encoding="utf-8")

    document = map_document(path, density="dense")

    assert len(document.records) == 3


def test_align_documents_returns_report_and_writes_artifacts(tmp_path):
    source = tmp_path / "request.txt"
    target = tmp_path / "offer.txt"
    json_path = tmp_path / "report.json"
    html_path = tmp_path / "report.html"
    source.write_text("The service must encrypt customer data.\n", encoding="utf-8")
    target.write_text("Customer data is encrypted with AES 256.\n", encoding="utf-8")
    config = resolve_workflow_config(marker_style="compact", coverage_threshold=0.3)

    report = align_documents(source, target, config=config)
    report.write_json(json_path)
    report.write_html(html_path)

    assert report.summary["covered"] == 1
    assert json_path.exists()
    assert "side" in html_path.read_text(encoding="utf-8")
