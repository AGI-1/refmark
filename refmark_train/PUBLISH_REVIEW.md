# Publish Review

This note reviews `publish/refmark_train` as a candidate publish artifact.

## Current Status

Status: **proof-of-concept ready for public technical review**

The artifact is now good enough for:

- internal review
- sharing with collaborators
- preserving the benchmark story and retained evidence

It is not yet ideal for:

- long-term archival with strong provenance guarantees

## What Is Already In Good Shape

- the folder surface is curated
- retained datasets and runs are small enough to inspect
- the main surface now emphasizes the extracted QA/anchor datasets instead of
  the raw source payloads
- top-level docs explain the current claim and its limits
- pipeline commands are documented
- old experiment clutter and raw source payloads are excluded from the public
  surface
- retained manifests now use local portable paths instead of the old machine
  root
- the artifact now includes a local environment spec and a smoke verification
  entry point
- retained source provenance and usage are documented

## Main Remaining Gaps

### 1. Provenance Could Be Stronger

Source URLs are retained, but the artifact does not yet include checksums,
fetch dates, or a strict provenance note for locally rebuilt source documents.

Impact:

- good enough for technical review
- weaker than ideal for long-term archival

Recommended fix:

- add checksums for locally rebuilt source documents
- add fetch timestamps and normalization notes

### 2. Refinement Pipeline Has Historical Lineage

The retained refinement datasets name historical source datasets that are not
redistributed in this public artifact. The derived retained datasets and run
artifacts are present, but the full historical rebuild path is less clean than
the single-region benchmark.

Impact:

- acceptable for current PoC publication
- slightly awkward for external readers

Recommended fix:

- rebuild the historical source datasets from public manifests, or keep this
  lineage framed as retained evidence rather than fully reproducible source prep

## Recommendation

If the goal is **public proof-of-concept**, this folder is ready.

If the goal is **long-term archival quality**, I would still do these two
follow-ups first:

1. add checksums and fetch timestamps for stronger provenance
2. rebuild and publish a clean source-fetch manifest with deterministic hashes
