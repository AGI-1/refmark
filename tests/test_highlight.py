import json
from pathlib import Path

from refmark.highlight import highlight_refs, render_highlight_html, render_highlight_text


def test_highlight_refs_uses_shadow_session_for_unmarked_python(tmp_path: Path):
    path = tmp_path / "sample.py"
    state_dir = tmp_path / ".shadow"
    path.write_text(
        "def greet(name: str) -> str:\n    return name.strip()\n\n\ndef loud(name: str) -> str:\n    return greet(name).upper()\n",
        encoding="utf-8",
    )

    result = highlight_refs(path, "F01-F02", context_lines=0, state_dir=state_dir)

    assert result.namespace_mode == "shadow"
    assert result.refs == ["F01", "F02"]
    assert len(result.regions) == 2
    assert "greet(name: str)" in result.regions[0].text
    assert "return greet(name).upper()" in result.regions[1].text


def test_highlight_refs_json_and_html_renderers_include_regions(tmp_path: Path):
    path = tmp_path / "sample.py"
    state_dir = tmp_path / ".shadow"
    path.write_text(
        "def greet(name: str) -> str:\n    return name.strip()\n\n\ndef loud(name: str) -> str:\n    return greet(name).upper()\n",
        encoding="utf-8",
    )

    result = highlight_refs(path, "F02", context_lines=1, state_dir=state_dir)
    html = render_highlight_html(result)
    text = render_highlight_text(result)
    payload = result.to_dict()

    assert "<span class='hit'>" in html
    assert "[F02]" in text
    assert json.loads(json.dumps(payload))["refs"] == ["F02"]
