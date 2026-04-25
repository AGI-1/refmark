import json
from pathlib import Path

import pytest

pytest.importorskip("mcp")
pytest.importorskip("mcp.client.session")
pytest.importorskip("mcp.client.stdio")

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from refmark.edit import apply_ref_diff
from refmark.core import inject
from refmark.mcp_server import apply_ref_diff_tool, list_ref_regions_tool, read_refmarked_file_tool

PUBLISH_ROOT = Path(__file__).resolve().parents[1]


def _make_marked_typescript_file(tmp_path: Path) -> Path:
    source = """export class UserService {
  format(name: string): string {
    return name.trim();
  }
}
"""
    marked, _ = inject(
        source,
        ".ts",
        marker_format="typed_comment_ts",
        chunker="ts_hybrid",
    )
    path = tmp_path / "user_service.ts"
    path.write_text(marked, encoding="utf-8")
    return path


def test_apply_ref_diff_replaces_live_region_by_region_id(tmp_path: Path):
    path = _make_marked_typescript_file(tmp_path)

    result = apply_ref_diff(
        path,
        [
            {
                "region_id": "M02",
                "action": "replace",
                "new_content": """  format(name: string): string {
    return name.trim().toUpperCase();
  }
""",
            }
        ],
        expect_live_markers=True,
    )

    content = path.read_text(encoding="utf-8")
    assert result["ok"]
    assert result["syntax_ok"]
    assert "toUpperCase" in content
    assert "// [@M02]" in content


def test_apply_ref_diff_range_delete_preserves_boundary_refs(tmp_path: Path):
    path = _make_marked_typescript_file(tmp_path)

    result = apply_ref_diff(
        path,
        [
            {
                "start_ref": "M02",
                "end_ref": "C03",
                "action": "delete",
            }
        ],
        expect_live_markers=True,
    )

    content = path.read_text(encoding="utf-8")
    assert result["ok"]
    assert result["syntax_ok"]
    assert "// [@M02]" in content
    assert "// [@C03]" in content
    assert "return name.trim();" not in content


def test_apply_ref_diff_rejects_mixed_address_modes(tmp_path: Path):
    path = _make_marked_typescript_file(tmp_path)

    result = apply_ref_diff(
        path,
        [
            {
                "region_id": "M02",
                "start_ref": "M02",
                "new_content": "noop",
            }
        ],
        expect_live_markers=True,
    )

    assert not result["ok"]
    assert any("either region_id or start_ref" in error for error in result["errors"])


def test_apply_ref_diff_lists_available_regions_for_missing_target(tmp_path: Path):
    path = _make_marked_typescript_file(tmp_path)

    result = apply_ref_diff(
        path,
        [
            {
                "region_id": "M99",
                "action": "replace",
                "new_content": "noop\n",
            }
        ],
        expect_live_markers=True,
    )

    assert not result["ok"]
    assert any("Available regions:" in error for error in result["errors"])


def test_apply_ref_diff_rejects_overlapping_edit_spans(tmp_path: Path):
    path = _make_marked_typescript_file(tmp_path)

    result = apply_ref_diff(
        path,
        [
            {
                "start_ref": "C01",
                "end_ref": "C03",
                "action": "replace",
                "new_content": "export class UserService {}\n",
            },
            {
                "region_id": "M02",
                "action": "replace",
                "new_content": "  format(name: string): string {\n    return name;\n  }\n",
            },
        ],
        expect_live_markers=True,
    )

    assert not result["ok"]
    assert any("Overlapping edit spans are not allowed" in error for error in result["errors"])


def test_apply_ref_diff_tool_accepts_stringified_json_edit(tmp_path: Path):
    path = _make_marked_typescript_file(tmp_path)

    result = apply_ref_diff_tool(
        str(path),
        [
            json.dumps(
                {
                    "region_id": "M02",
                    "action": "replace",
                    "new_content": "  format(name: string): string {\n    return name.trim().toUpperCase();\n  }\n",
                }
            )
        ],
        expect_live_markers=True,
    )

    content = path.read_text(encoding="utf-8")
    assert result["ok"]
    assert result["syntax_ok"]
    assert "toUpperCase" in content


def test_apply_ref_diff_insert_before_ref_creates_live_region(tmp_path: Path):
    path = _make_marked_typescript_file(tmp_path)

    result = apply_ref_diff(
        path,
        [
            {
                "anchor_ref": "M02",
                "action": "insert_before",
                "create_region": True,
                "new_content": """  normalize(name: string): string {
    return name.trim().toLowerCase();
  }
""",
            }
        ],
        expect_live_markers=True,
    )

    content = path.read_text(encoding="utf-8")
    assert result["ok"]
    assert result["syntax_ok"]
    assert result["created_regions"]
    created = result["created_regions"][0]["region_id"]
    assert f"// [@{created}]" in content
    assert "normalize(name: string)" in content
    assert content.index(f"// [@{created}]") < content.index("// [@M02]")


def test_apply_ref_diff_insert_before_eof_creates_live_region(tmp_path: Path):
    path = _make_marked_typescript_file(tmp_path)

    result = apply_ref_diff(
        path,
        [
            {
                "anchor_ref": "EOF",
                "action": "insert_before",
                "create_region": True,
                "new_content": """
export function helper(name: string): string {
  return name.trim();
}
""",
            }
        ],
        expect_live_markers=True,
    )

    content = path.read_text(encoding="utf-8")
    assert result["ok"]
    assert result["syntax_ok"]
    created = result["created_regions"][0]["region_id"]
    assert f"// [@{created}]" in content
    assert "export function helper" in content
    assert content.rstrip().endswith("}")


def test_apply_ref_diff_patch_within_line_edits_updates_only_region_body(tmp_path: Path):
    path = _make_marked_typescript_file(tmp_path)

    result = apply_ref_diff(
        path,
        [
            {
                "region_id": "M02",
                "action": "patch_within",
                "patch_format": "line_edits",
                "patch": {
                    "edits": [
                        {
                            "start_line": 2,
                            "end_line": 2,
                            "expected_text": "    return name.trim();\n",
                            "new_content": "    return name.trim().toUpperCase();\n",
                        }
                    ]
                },
            }
        ],
        expect_live_markers=True,
    )

    content = path.read_text(encoding="utf-8")
    assert result["ok"]
    assert result["syntax_ok"]
    assert "toUpperCase" in content
    assert "// [@M02]" in content
    assert "format(name: string)" in content


def test_apply_ref_diff_patch_within_infers_line_edits_format(tmp_path: Path):
    path = _make_marked_typescript_file(tmp_path)

    result = apply_ref_diff(
        path,
        [
            {
                "region_id": "M02",
                "action": "patch_within",
                "patch": {
                    "edits": [
                        {
                            "start_line": 2,
                            "end_line": 2,
                            "expected_text": "    return name.trim();\n",
                            "new_content": "    return name.trim().toLowerCase();\n",
                        }
                    ]
                },
            }
        ],
        expect_live_markers=True,
    )

    assert result["ok"]
    assert "toLowerCase" in path.read_text(encoding="utf-8")


def test_apply_ref_diff_patch_within_tolerates_duplicate_same_anchor(tmp_path: Path):
    path = _make_marked_typescript_file(tmp_path)

    result = apply_ref_diff(
        path,
        [
            {
                "region_id": "M02",
                "start_ref": "M02",
                "action": "patch_within",
                "patch_format": "line_edits",
                "patch": {
                    "edits": [
                        {
                            "start_line": 2,
                            "end_line": 2,
                            "expected_text": "    return name.trim();\n",
                            "new_content": "    return name.trim().toLowerCase();\n",
                        }
                    ]
                },
            }
        ],
        expect_live_markers=True,
    )

    assert result["ok"]
    assert "toLowerCase" in path.read_text(encoding="utf-8")


def test_apply_ref_diff_patch_within_accepts_literal_escaped_newline_expected_text(tmp_path: Path):
    path = _make_marked_typescript_file(tmp_path)

    result = apply_ref_diff(
        path,
        [
            {
                "region_id": "M02",
                "action": "patch_within",
                "patch_format": "line_edits",
                "patch": {
                    "edits": [
                        {
                            "start_line": 2,
                            "end_line": 2,
                            "expected_text": "    return name.trim();\\n",
                            "new_content": "    return name.trim().toLowerCase();\n",
                        }
                    ]
                },
            }
        ],
        expect_live_markers=True,
    )

    assert result["ok"]
    assert "toLowerCase" in path.read_text(encoding="utf-8")


def test_apply_ref_diff_patch_within_line_edits_rejects_expected_text_mismatch(tmp_path: Path):
    path = _make_marked_typescript_file(tmp_path)

    result = apply_ref_diff(
        path,
        [
            {
                "region_id": "M02",
                "action": "patch_within",
                "patch_format": "line_edits",
                "patch": {
                    "edits": [
                        {
                            "start_line": 2,
                            "end_line": 2,
                            "expected_text": "    return name.trim().toLowerCase();\n",
                            "new_content": "    return name.trim().toUpperCase();\n",
                        }
                    ]
                },
            }
        ],
        expect_live_markers=True,
    )

    content = path.read_text(encoding="utf-8")
    assert result["ok"] is False
    assert any("expected_text mismatch" in error for error in result["errors"])
    assert "toUpperCase" not in content
    assert "return name.trim();" in content


def test_apply_ref_diff_patch_within_unified_diff_is_bounded_to_region(tmp_path: Path):
    path = _make_marked_typescript_file(tmp_path)

    result = apply_ref_diff(
        path,
        [
            {
                "region_id": "M02",
                "action": "patch_within",
                "patch_format": "unified_diff",
                "patch": {
                    "patch": """--- a/region.ts
+++ b/region.ts
@@ -1,3 +1,3 @@
   format(name: string): string {
-    return name.trim();
+    return name.trim().toLowerCase();
   }
""",
                },
            }
        ],
        expect_live_markers=True,
    )

    content = path.read_text(encoding="utf-8")
    assert result["ok"]
    assert result["syntax_ok"]
    assert "toLowerCase" in content
    assert "UserService" in content
    assert "// [@M02]" in content


def test_apply_ref_diff_rejects_create_region_without_live_markers(tmp_path: Path):
    path = tmp_path / "plain.py"
    path.write_text("def greet(name: str) -> str:\n    return name.strip()\n", encoding="utf-8")

    result = apply_ref_diff(
        path,
        [
            {
                "anchor_ref": "EOF",
                "action": "insert_before",
                "create_region": True,
                "new_content": "def helper() -> str:\n    return 'ok'\n",
            }
        ],
        expect_live_markers=False,
    )

    assert not result["ok"]
    assert any("create_region requires live markers" in error for error in result["errors"])


def test_apply_ref_diff_rejects_patch_within_without_patch_payload(tmp_path: Path):
    path = _make_marked_typescript_file(tmp_path)

    result = apply_ref_diff(
        path,
        [
            {
                "region_id": "M02",
                "action": "patch_within",
                "patch_format": "line_edits",
            }
        ],
        expect_live_markers=True,
    )

    assert not result["ok"]
    assert any("patch_within requires patch" in error for error in result["errors"])


def test_list_ref_regions_includes_preview_lines_by_default(tmp_path: Path):
    path = _make_marked_typescript_file(tmp_path)

    result = list_ref_regions_tool(str(path))

    assert result["ok"] is True
    assert result["refmark_supported"] is True
    assert result["regions"]
    method_region = next(region for region in result["regions"] if region["region_id"] == "M02")
    assert method_region["line_count"] >= 1
    assert isinstance(method_region["preview_lines"], list)
    assert method_region["preview_lines"]
    assert "format(name: string)" in method_region["preview_lines"][0]


def test_shadow_session_regions_persist_for_unmarked_file_across_reads_and_edit(tmp_path: Path):
    path = tmp_path / "plain.py"
    path.write_text(
        """def greet(name: str) -> str:
    return name.strip()


def loud(name: str) -> str:
    return greet(name).upper()
""",
        encoding="utf-8",
    )

    first_regions = list_ref_regions_tool(str(path))
    second_regions = list_ref_regions_tool(str(path))

    assert first_regions["ok"] is True
    assert first_regions["namespace_mode"] == "shadow"
    assert first_regions["shadow_persistent"] is True
    assert second_regions["namespace_mode"] == "shadow"
    first_ids = [region["region_id"] for region in first_regions["regions"]]
    second_ids = [region["region_id"] for region in second_regions["regions"]]
    assert first_ids == second_ids

    greet_region = next(
        region for region in first_regions["regions"] if "greet(name: str)" in " ".join(region["preview_lines"])
    )["region_id"]

    result = apply_ref_diff_tool(
        str(path),
        [
            {
                "region_id": greet_region,
                "action": "replace",
                "new_content": "def greet(name: str) -> str:\n    return name.strip().title()\n",
            }
        ],
    )

    assert result["ok"] is True
    third_regions = list_ref_regions_tool(str(path))
    third_ids = [region["region_id"] for region in third_regions["regions"]]
    assert third_ids == first_ids
    content = path.read_text(encoding="utf-8")
    assert "title()" in content
    assert "[@" not in content


def test_read_refmarked_file_and_list_regions_use_live_namespace_for_marked_file(tmp_path: Path):
    path = _make_marked_typescript_file(tmp_path)

    listed = list_ref_regions_tool(str(path))
    read = read_refmarked_file_tool(str(path))

    assert listed["ok"] is True
    assert listed["namespace_mode"] == "live"
    assert listed["shadow_persistent"] is False
    assert any(region["region_id"] == "M02" for region in listed["regions"])
    assert read["ok"] is True
    assert read["namespace_mode"] == "live"
    assert "// [@M02]" in read["content"]


def test_apply_ref_diff_tool_shadow_session_can_create_region_for_unmarked_file(tmp_path: Path):
    path = tmp_path / "plain.py"
    path.write_text(
        """class Formatter:
    def render(self, name: str) -> str:
        return name.strip()
""",
        encoding="utf-8",
    )

    regions = list_ref_regions_tool(str(path))
    render_region = next(
        region for region in regions["regions"] if "render(self, name: str)" in " ".join(region["preview_lines"])
    )["region_id"]

    result = apply_ref_diff_tool(
        str(path),
        [
            {
                "anchor_ref": render_region,
                "action": "insert_before",
                "create_region": True,
                "new_content": "    def normalize(self, name: str) -> str:\n        return name.strip().lower()\n\n",
            }
        ],
    )

    assert result["ok"] is True
    assert result["created_regions"]
    content = path.read_text(encoding="utf-8")
    assert "def normalize(" in content
    assert "[@" not in content
    reread = list_ref_regions_tool(str(path))
    assert reread["shadow_persistent"] is True
    assert any(region["region_id"] == result["created_regions"][0]["region_id"] for region in reread["regions"])


@pytest.mark.anyio
async def test_mcp_server_exposes_apply_ref_diff_and_applies_edit(tmp_path: Path):
    path = _make_marked_typescript_file(tmp_path)
    log_path = tmp_path / "apply_ref_diff_calls.jsonl"

    params = StdioServerParameters(
        command="python",
        args=["-m", "refmark.mcp_server"],
        cwd=str(PUBLISH_ROOT),
        env={
            "PYTHONPATH": str(PUBLISH_ROOT),
            "REFMARK_MCP_LOG_PATH": str(log_path),
        },
    )

    async with stdio_client(params) as streams:
        read, write = streams
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()

            tool = next(tool for tool in tools.tools if tool.name == "apply_ref_diff")
            edits_schema = tool.inputSchema["properties"]["edits"]["items"]
            assert "anyOf" in edits_schema
            assert any(
                isinstance(option, dict) and option.get("$ref") == "#/$defs/RefmarkEdit"
                for option in edits_schema["anyOf"]
            )
            assert any(
                isinstance(option, dict) and option.get("type") == "object"
                for option in edits_schema["anyOf"]
            )
            assert any(
                isinstance(option, dict) and option.get("type") == "string"
                for option in edits_schema["anyOf"]
            )

            result = await session.call_tool(
                "apply_ref_diff",
                {
                    "file_path": str(path),
                    "expect_live_markers": True,
                    "edits": [
                        {
                            "region_id": "M02",
                            "action": "replace",
                            "new_content": """  format(name: string): string {
    return name.trim().toUpperCase();
  }
""",
                        }
                    ],
                },
            )

    assert not result.isError
    assert result.structuredContent["ok"] is True
    assert "toUpperCase" in path.read_text(encoding="utf-8")
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["tool"] == "apply_ref_diff"
    assert record["file_path"] == str(path)
    assert record["ok"] is True
    assert record["expect_live_markers"] is True
    assert record["edits"][0]["region_id"] == "M02"


def test_plugin_manifest_and_mcp_config_exist():
    plugin_manifest = PUBLISH_ROOT / "plugins" / "refmark-apply-ref-diff" / ".codex-plugin" / "plugin.json"
    mcp_config = PUBLISH_ROOT / "plugins" / "refmark-apply-ref-diff" / ".mcp.json"

    manifest = json.loads(plugin_manifest.read_text(encoding="utf-8"))
    config = json.loads(mcp_config.read_text(encoding="utf-8"))

    assert manifest["name"] == "refmark-apply-ref-diff"
    assert manifest["mcpServers"] == "./.mcp.json"
    assert "refmark-apply-ref-diff" in config["mcpServers"]
    server_config = config["mcpServers"]["refmark-apply-ref-diff"]
    assert server_config["env"]["PYTHONPATH"] == "../.."
