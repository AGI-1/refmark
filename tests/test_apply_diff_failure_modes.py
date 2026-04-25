from refmark.edit import apply_ref_diff
from refmark.patches import apply_search_replace_edits, apply_unified_diff_patch
from refmark.core import inject


def test_unified_diff_patch_fails_after_context_drift():
    original = """def alpha():
    return 1


def target():
    value = 1
    return value
"""
    stale_patch = """--- a/file.py
+++ b/file.py
@@ -4,3 +4,6 @@
 def target():
-    value = 1
+    helper = 2
+    value = 1
+    debug = helper + value
     return value
"""
    drifted = """def alpha():
    return 1


def target():
    value = 10
    return value
"""
    updated, errors = apply_unified_diff_patch(drifted, stale_patch)
    assert updated == drifted
    assert errors == ["Unified diff context line did not match current file."]


def test_search_replace_fails_when_target_snippet_is_ambiguous():
    code = """def prepare():
    value = normalize(name)
    return value


def render():
    value = normalize(name)
    return value
"""
    updated, errors = apply_search_replace_edits(
        code,
        [
            {
                "original_text": "value = normalize(name)",
                "new_content": "value = normalize(name).strip()",
            }
        ],
    )
    assert updated == code
    assert errors == ["Original snippet is ambiguous in current file."]


def test_refmark_insert_before_handles_new_function_creation_stably():
    source = """def alpha():
    return 1


def beta():
    return 2
"""
    marked, _ = inject(source, ".py", marker_format="typed_comment_py", chunker="hybrid")

    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "sample.py"
        path.write_text(marked, encoding="utf-8")
        result = apply_ref_diff(
            path,
            [
                {
                    "anchor_ref": "F02",
                    "action": "insert_before",
                    "create_region": True,
                    "new_content": """def helper():
    return 99

""",
                }
            ],
            expect_live_markers=True,
        )

        content = path.read_text(encoding="utf-8")
        assert result["ok"]
        assert result["created_regions"]
        created = result["created_regions"][0]["region_id"]
        assert f"# [@{created}]" in content
        assert "def helper()" in content


def test_refmark_patch_within_can_update_ambiguous_line_only_inside_target_region():
    source = """def prepare():
    value = normalize(name)
    return value


def render():
    value = normalize(name)
    return value
"""
    marked, _ = inject(source, ".py", marker_format="typed_comment_py", chunker="hybrid")

    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "sample.py"
        path.write_text(marked, encoding="utf-8")
        result = apply_ref_diff(
            path,
            [
                {
                    "region_id": "F02",
                    "action": "patch_within",
                    "patch_format": "search_replace",
                    "patch": {
                        "edits": [
                            {
                                "original_text": "value = normalize(name)",
                                "new_content": "value = normalize(name).strip()",
                            }
                        ]
                    },
                }
            ],
            expect_live_markers=True,
        )

        content = path.read_text(encoding="utf-8")
        assert result["ok"]
        assert content.count("normalize(name).strip()") == 1
        assert "def render()" in content
        assert "def prepare()" in content
