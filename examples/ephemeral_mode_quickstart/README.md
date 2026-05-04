# Ephemeral Mode Quickstart

This example shows the disposable address-layer workflow for one-off document
edits. It does not create a durable corpus registry.

1. Create a marked view:

```bash
python -m refmark.cli ephemeral-map examples/ephemeral_mode_quickstart/sample_contract.md \
  --instructions \
  --manifest tmp/ephemeral_contract.refmark.jsonl \
  -o tmp/ephemeral_contract.refmarked.txt
```

2. Ask a model to return JSON edits against the temporary refs, or use the
included sample edit:

```bash
python -m refmark.cli ephemeral-apply examples/ephemeral_mode_quickstart/sample_contract.md \
  --edits-file examples/ephemeral_mode_quickstart/sample_edits.json \
  --dry-run \
  -o tmp/sample_contract.patched.md
```

3. Apply after the dry run is clean:

```bash
python -m refmark.cli ephemeral-apply examples/ephemeral_mode_quickstart/sample_contract.md \
  --edits-file examples/ephemeral_mode_quickstart/sample_edits.json \
  -o tmp/sample_contract.patched.md
```

The point is bounded application: the model edits addressed regions, not the
whole document.
