from refmark.workflow_config import load_workflow_config, resolve_workflow_config


def test_resolve_workflow_config_density_and_marker_style():
    config = resolve_workflow_config(density="dense", marker_style="explicit", expand_after=2)

    assert config.chunker == "line"
    assert config.lines_per_chunk == 1
    assert config.marker_format == "typed_explicit"
    assert config.expand_after == 2


def test_load_workflow_config_from_flat_yaml(tmp_path):
    path = tmp_path / "refmark.yaml"
    path.write_text(
        "density: coarse\nmarker_style: compact\ncoverage_threshold: 0.6\ninclude_headings: false\n",
        encoding="utf-8",
    )

    config = load_workflow_config(path)

    assert config.chunker == "token"
    assert config.marker_format == "typed_compact"
    assert config.coverage_threshold == 0.6
    assert config.include_headings is False
