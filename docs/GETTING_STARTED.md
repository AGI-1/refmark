# Getting Started

This is the shortest useful path to trying Refmark locally.

## Install

Core usage:

```bash
pip install -e .
```

If you want the MCP server:

```bash
pip install -e .[mcp]
```

If you want TypeScript syntax validation:

```bash
pip install -e .[typescript]
```

For the full public-artifact test surface:

```bash
pip install -e .[dev,mcp,typescript,train]
```

If you want only the exploratory training prototype:

```bash
pip install -e .[train]
```

## Verify The Artifact

Run the deterministic core smoke check:

```bash
python -m refmark.cli smoke
```

Run the exploratory training smoke check:

```bash
python -m refmark_train.smoke
```

Verify retained training datasets and checksums:

```bash
python -m refmark_train.verify_publish_artifact
```

Run the tests:

```bash
pytest
```

## Try The Examples

Run the citation-evaluation playground:

```bash
python examples/citation_qa/run_eval.py
```

Run the data-smell playground:

```bash
python examples/data_smells/run.py
```

Run the multidiff playground:

```bash
python examples/multidiff_demo/run.py
```

Run the judge-free reward playground:

```bash
python examples/judge_free_rewards/run.py
```

These examples write inspectable artifacts under their local `output/`
directories.

## Inject Markers Into A File

```bash
python -m refmark.cli inject path/to/example.py
```

That rewrites the file in place using language-aware defaults.

To preview instead:

```bash
python -m refmark.cli inject path/to/example.py --dry-run
```

## Strip Markers Back Out

```bash
python -m refmark.cli strip path/to/example.py
```

Roundtrip expectation for supported formats is:

```text
strip(inject(text)) == text
```

## Apply A Stable Multi-Edit

Create an edits file:

```json
{
  "edits": [
    {
      "region_id": "F01",
      "action": "replace",
      "new_content": "def greet(name: str) -> str:\n    return name.strip().upper()\n"
    }
  ]
}
```

Then apply it:

```bash
python -m refmark.cli apply-ref-diff your_marked_file.py --edits-file edits.json
```

Insertion is also supported:

```json
{
  "edits": [
    {
      "anchor_ref": "EOF",
      "action": "insert_before",
      "new_content": "def helper() -> str:\n    return \"ok\"\n"
    }
  ]
}
```

## Run The MCP Server

```bash
python -m refmark.mcp_server
```

The server exposes `apply_ref_diff` as a structured tool for same-file region edits.

For a persistent shadow-session view of an unmarked file:

```bash
python scripts/refmark_shadow_session_cli.py read --file-path examples/multidiff_demo/source.py --state-dir .refmark/shadow
```

## Highlight Returned Refs For Review

Once a model returns anchor ids, render them back into source snippets:

```bash
python -m refmark.cli highlight path/to/example.py --refs F03,F05-F06
```

Write an HTML audit artifact:

```bash
python -m refmark.cli highlight path/to/example.py --refs F03,F05-F06 --format html --output cited_regions.html
```

## What To Expect

- Python and TypeScript are the current productized language set
- Markdown and generic text helpers exist in core primitives, but they are not the supported code-editing product surface yet
- Marker-based editing is strongest on same-file, multi-region changes
- `apply-ref-diff` works directly on live-marked files; use the MCP server or `scripts/refmark_shadow_session_cli.py` for persistent shadow sessions on unmarked files
- Highlighted source review is ready for HIL citation audits
- The broader benchmark harness is research infrastructure and is not included in this public PoC CLI
