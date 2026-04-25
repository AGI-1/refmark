# Refmarker Registry Design

This is the small attachable interface for systems that already own their own
document flow and want Refmark as an addressability layer.

## Core Decision

`Refmarker` accepts content and returns:

- `content`: the content the caller should continue using.
- `marked_view`: a refmarked view for prompting, citations, review, or display.
- `records`: structured region metadata.
- `namespace_mode`: `live` or `shadow`.

Live mode embeds markers directly in `content`. Shadow mode leaves `content`
unchanged and stores the marked view plus region metadata in a registry.

## Modes

- `live`: unmarked input becomes marked output. Already marked input is passed
  through unchanged.
- `shadow`: unmarked input is not modified. The marked view and manifest are
  persisted under `.refmark/registry` by default.
- `auto`: currently chooses shadow for unmarked input and live for premarked
  input.

## Registry Key

Shadow registry entries are keyed by:

- `doc_id`
- source content hash
- resolved workflow configuration fingerprint

This means the same source/config pair reuses the same marked view, while a
changed source naturally starts a new shadow session.

## Persistence Boundary

The public interface does not require persistence from the caller. A default
filesystem registry is used for shadow mode because most internal RAG/review
flows do not want to inject markers into the source on every processing step.
Callers can provide their own `RefmarkRegistry` or use live mode when embedded
markers are desired.

## Example

```python
from refmark import Refmarker

marker = Refmarker(mode="shadow")
result = marker.mark_text(
    "The supplier must provide EU data residency.\n",
    doc_id="customer_request",
)

assert result.content.startswith("The supplier")
assert "[@C01]" in result.marked_view
assert result.records[0].region_id == "C01"
```
