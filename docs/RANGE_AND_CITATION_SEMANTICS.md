# Range And Citation Semantics

Refmark uses one address space for several workflows, but not every workflow
uses range boundaries the same way. This page is the short contract.

## Terms

- **Region id / ref**: local id such as `P03` or `F04`.
- **Stable ref**: document-qualified id such as `policy:P03`.
- **Region**: the source span addressed by one ref.
- **Citation range**: an inclusive ordered set of regions, such as `P03-P05`.
- **Edit boundary range**: a replacement span from `start_ref` up to, but not
  including, `end_ref`.

## Citation Grammar

Citation refs are strict. The parser accepts:

```text
P03
policy:P03
P03-P05
P03..P05
policy:P03-P05
policy:P03-policy:P05
[P03, P05-P07]
```

Citation ranges are inclusive. `P03-P05` means `P03`, `P04`, and `P05` when
those refs exist in the same ordered address space.

Disjoint citations should be represented as a list:

```json
["policy:P03", "policy:P07-P09"]
```

Ranges cannot cross documents. `policy:P03-contract:P05` is invalid because
there is no single ordered address space that can deterministically expand it.

Python helpers:

```python
from refmark import parse_citation_refs, validate_citation_refs

refs = parse_citation_refs("[policy:P03, policy:P07-P09]")
validated = validate_citation_refs(
    refs,
    address_space=["policy:P03", "policy:P07", "policy:P08", "policy:P09"],
)
```

## Edit Boundary Ranges

The MCP and `apply_ref_diff` edit APIs use range boundaries differently:

```json
{
  "start_ref": "F03",
  "end_ref": "F05",
  "action": "replace",
  "new_content": "..."
}
```

This replaces the span beginning at `F03` and stopping immediately before
`F05`. In other words, `end_ref` is an exclusive stop boundary for edits.

Use `region_id` when you want to replace exactly one semantic region:

```json
{"region_id": "F03", "action": "replace", "new_content": "..."}
```

Use citation ranges when scoring or highlighting evidence. Use edit boundary
ranges when applying patches.
