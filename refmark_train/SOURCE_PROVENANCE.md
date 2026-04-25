# Source Provenance

This artifact retains source manifests and derived QA/anchor datasets so the
citation-localization benchmarks can be inspected without redistributing raw
or normalized upstream source documents.

## What Is Included

The publish-facing retained source materials live under
[source_docs](source_docs) and include:

- source manifests with canonical URLs and document ids
- category manifests describing the source sets used for retained datasets

The exact document list and canonical source URLs are recorded in
[source_docs/manifest.json](source_docs/manifest.json). Each manifest entry
includes a rights note rather than a redistribution grant; raw HTML/PDF files,
normalized text payloads, and combined category text files are intentionally
excluded from the public artifact.

The retained derived datasets under [data](data) include SHA-256 checksums for
their shipped JSONL files in each dataset `manifest.json`; those checksums are
verified by `python -m refmark_train.verify_publish_artifact`.

## How They Were Prepared

The fetch and normalization pipeline lives in
[pull_source_docs.py](pull_source_docs.py).

Preparation steps:

1. fetch raw HTML or PDF from the public source URL
2. store the raw file in a local source cache
3. extract normalized plain text into a local source cache
4. build category-level combined corpora locally under `source_docs/sets/`

The fetcher records `raw_sha256`, `text_sha256`, and category-set
`text_sha256` values when sources are rebuilt locally. These hashes are not
pre-filled in the public source manifest because the raw/text payloads are not
redistributed in this artifact.

The retained benchmark datasets are generated from the combined category sets,
not directly from the raw web payloads.

## Usage Note

The derived datasets are included here for reproducible research on anchor
localization. The upstream documents remain attributable to their original
publishers and may be subject to source-specific terms, policies, or
redistribution expectations.

The retained source set was intentionally drawn from public-facing sources such
as regulatory materials, public filings, and open documentation. This provenance
note is not a legal opinion; it is a release hygiene record for users who want
to rebuild or extend the corpora.

For public release, this artifact keeps manifests and derived examples but not
the upstream source payloads themselves. Before rebuilding or redistributing
source corpora, review the source-specific terms and your organization's
publication policy.

The shipped derived rows contain generated questions, anchor ids, answer terms,
and small metadata fields used for localization experiments. They are not a
redistribution of the raw upstream documents.

## Benchmark Scope

The current publish-facing benchmark story uses:

- `documentation_set.txt` for the strongest single-region result
- archived structure-random source datasets for the retained refinement lineage
- `corporate_set.txt` indirectly through the retained corporate smoke benchmark
