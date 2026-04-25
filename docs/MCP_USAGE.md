# MCP Usage

`refmark.mcp_server` exposes the stable `apply_ref_diff` primitive over MCP stdio.

`apply_ref_diff` is intentionally a same-file primitive. To edit multiple
files, call the tool once per file. Do not pass arrays of file paths.

## Start The Server

```bash
python -m refmark.mcp_server
```

## Tool Name

- `apply_ref_diff`

Related inspection tools:

- `read_refmarked_file`
- `list_ref_regions`

## Input Shape

Top-level fields:

- `file_path`
- `edits`
- optional `expect_live_markers`

Each edit must use one of:

- `region_id`
- `start_ref` with optional `end_ref`
- `anchor_ref` for `insert_before`

Edit fields:

- `action`: `replace`, `delete`, `insert_before`, or `patch_within`
- `new_content` for `replace` and `insert_before` edits
- `create_region` for live-marker insertion that should return a fresh stable ref

`patch_within` is for small in-region changes. It requires:

- `patch_format`: currently `line_edits`
- `patch.edits`: a list of relative line edits inside the addressed region
- `start_line` and `end_line`: 1-based line numbers within the region body
- `expected_text`: optional but recommended stale-edit guard
- `new_content`: replacement text for that relative line range

Example:

```json
{
  "file_path": "/absolute/path/to/file.py",
  "expect_live_markers": true,
  "edits": [
    {
      "region_id": "F04",
      "action": "replace",
      "new_content": "def fn() -> int:\n    return 1\n"
    }
  ]
}
```

Insertion example:

```json
{
  "file_path": "/absolute/path/to/file.py",
  "expect_live_markers": true,
  "edits": [
    {
      "anchor_ref": "EOF",
      "action": "insert_before",
      "create_region": true,
      "new_content": "def helper() -> int:\n    return 1\n"
    }
  ]
}
```

## Behavior Notes

- each call edits exactly one `file_path`
- `delete` removes the addressed body while preserving boundary markers
- `insert_before` inserts before a ref or before virtual `EOF`
- Empty `new_content` means an empty body, not region deletion
- Overlapping edit spans are rejected
- `patch_within` rejects stale edits when `expected_text` does not match
- Missing targets return an error that includes the currently available region ids
- Successful live-marker insertions can return `created_regions` for the newly introduced stable refs

## Logging

Default MCP call log:

- `~/.refmark/logs/apply_ref_diff_calls.jsonl`

Override with:

- `REFMARK_MCP_LOG_PATH`

Each log entry contains:

- timestamp
- tool name
- file path
- serialized edits
- success flag
- syntax status
- errors

## Recommended Agent Prompting

When an agent already has a marked file, prompt it to:

1. read the marked file
2. choose exact `region_id` or `start_ref`/`end_ref` targets
3. send structured edits only

Avoid natural-language tool payloads. The tool expects structured edit objects, not free-text instructions.

Concrete system prompt:

```text
You are an editing agent. Work one file at a time.

First call read_refmarked_file to inspect the target file. Use the returned
refs exactly as addresses. When making changes, call apply_ref_diff with one
file_path and an edits array. Prefer patch_within for small changes and include
expected_text for each line edit. If you need to edit another file, make a
separate apply_ref_diff call for that file.
```

Example tool flow:

```json
{
  "tool": "read_refmarked_file",
  "arguments": {
    "file_path": "/absolute/path/to/service.py"
  }
}
```

```json
{
  "tool": "apply_ref_diff",
  "arguments": {
    "file_path": "/absolute/path/to/service.py",
    "edits": [
      {
        "region_id": "F02",
        "action": "patch_within",
        "patch_format": "line_edits",
        "patch": {
          "edits": [
            {
              "start_line": 2,
              "end_line": 2,
              "expected_text": "    return name.strip()\n",
              "new_content": "    return name.strip().title()\n"
            }
          ]
        }
      }
    ]
  }
}
```
