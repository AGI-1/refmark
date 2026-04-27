# MCP Usage

`refmark.mcp_server` exposes the stable `apply_ref_diff` primitive over MCP stdio.

`apply_ref_diff` is intentionally a same-file primitive. To edit multiple
files, call the tool once per file. Do not pass arrays of file paths.

## Start The Server

```bash
python -m refmark.mcp_server
```

## Codex In WSL

For WSL validation, use the Linux Codex CLI installed by
`scripts/bootstrap_codex_in_wsl.sh` through nvm. Do not test WSL Codex by trying
to execute the packaged Windows desktop binary from `C:\Program Files\WindowsApps`;
that path can fail from WSL with `Permission denied` and is not the expected
benchmark path.

Correct WSL checks:

```bash
source /home/alkon/.nvm/nvm.sh
CODEX_HOME=/home/alkon/.codex-refmark codex --version
CODEX_HOME=/home/alkon/.codex-refmark codex mcp list
CODEX_HOME=/home/alkon/.codex-refmark codex mcp get refmark_apply_ref_diff
```

The refmark Codex profile should be:

```toml
[mcp_servers.refmark_apply_ref_diff]
command = "/home/alkon/.venvs/refmark-mcp/bin/python"
args = ["-m", "refmark.mcp_server"]
enabled_tools = ["read_refmarked_file", "list_ref_regions", "apply_ref_diff"]

[mcp_servers.refmark_apply_ref_diff.env]
PYTHONPATH = "/mnt/c/aider"
REFMARK_MCP_LOG_PATH = "/mnt/c/aider/codex_bench/refmark_workspace/apply_ref_diff_calls.jsonl"
```

If the server import fails in WSL, install MCP into the dedicated venv rather
than the externally managed system Python:

```bash
python3 -m venv /home/alkon/.venvs/refmark-mcp
/home/alkon/.venvs/refmark-mcp/bin/python -m pip install mcp
```

The existing Windows-side benchmark runner,
`C:\aider\scripts\run_codex_wsl_bench.py`, already sources nvm before calling
`codex exec`.

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
- optional `dry_run`: validate and return hashes/diff without writing
- optional `base_hash`: expected SHA-256 of the current source file; mismatches
  reject the write
- optional `include_diff`: include a unified preview diff in the result

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

Dry-run / base-hash example:

```json
{
  "file_path": "/absolute/path/to/file.py",
  "base_hash": "current-source-sha256",
  "dry_run": true,
  "include_diff": true,
  "edits": [
    {
      "region_id": "F04",
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
```

## Behavior Notes

- each call edits exactly one `file_path`
- `delete` removes the addressed body while preserving boundary markers
- `insert_before` inserts before a ref or before virtual `EOF`
- `region_id` targets one semantic region
- `start_ref` / `end_ref` targets an edit boundary range where `end_ref` is
  exclusive; this is different from inclusive citation ranges such as `P03-P05`
- Empty `new_content` means an empty body, not region deletion
- Overlapping edit spans are rejected
- `patch_within` rejects stale edits when `expected_text` does not match
- `base_hash` rejects stale whole-file edits before patching starts
- `dry_run` performs resolution, syntax validation, hash calculation, and diff
  generation without writing
- Successful results include `source_hash`, `output_hash`, `changed_regions`,
  and optionally `diff`
- Missing targets return an error that includes the currently available region ids
- Successful live-marker insertions can return `created_regions` for the newly introduced stable refs
- Region listing is ordinal, so `F10` comes after `F09`, not after `F01`.
- Supported unmarked files use persistent shadow sessions, so read/list/apply
  share a marker namespace without writing markers into the source file.

## Hardening Controls

Path restriction:

- `REFMARK_MCP_ALLOWED_ROOTS`
- Optional `os.pathsep`-separated list of allowed workspace roots.
- When set, read/list/apply reject files outside those roots.

Read truncation:

- `REFMARK_MCP_MAX_READ_CHARS`
- `REFMARK_MCP_MAX_READ_LINES`

Payload logging:

- By default logs contain edit metadata only, not full replacement text.
- Set `REFMARK_MCP_LOG_PAYLOADS=1` to log full edit payloads for local debugging.

## Logging

Default MCP call log:

- `~/.refmark/logs/apply_ref_diff_calls.jsonl`

Override with:

- `REFMARK_MCP_LOG_PATH`

Each log entry contains:

- timestamp
- tool name
- file path
- edit metadata by default
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
