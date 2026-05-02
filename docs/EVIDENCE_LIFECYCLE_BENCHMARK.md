# Evidence Lifecycle Benchmark

Refmark is useful for retrieval, but its sharper claim is lifecycle integrity:

> A refmarked corpus can preserve, migrate, review, or stale evidence labels
> across corpus revisions instead of silently reusing broken chunk ids.

This matters when retrieval artifacts live longer than one index build:

- curated `question -> evidence` eval examples;
- training labels;
- generated retrieval metadata;
- cached answer citations;
- human review notes;
- reranker hard negatives.

## Problem

Ordinary chunking often creates ids such as `doc.md:chunk_004`. After the source
document changes, that id may still exist but point to different text.

That creates silent corruption:

```text
old eval label: "How do I configure X?" -> doc.md:chunk_004
new corpus:     doc.md:chunk_004 now describes Y
```

Rebuilding the embedding index does not repair old labels, citations, metadata,
or annotations. It only rebuilds the current search surface.

## Refmark Lifecycle View

Refmark treats evidence regions as maintained addresses with a manifest:

```text
old ref -> validate against new corpus
        -> exact/moved: keep or migrate
        -> fuzzy edited: review
        -> deleted/missing: stale
```

The value is not that evidence always survives. The value is that drift becomes
explicit.

## Natural Revision Results

Two Git-backed documentation corpora were checked from one base version to later
release tags.

| Corpus | Base -> Target | Refmark Auto | Refmark Review | Refmark Stale | Naive Correct | Naive Silent Wrong |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FastAPI | `0.100.0 -> 0.105.0` | `54.2%` | `39.1%` | `6.7%` | `71.5%` | `21.3%` |
| FastAPI | `0.100.0 -> 0.110.0` | `42.5%` | `25.5%` | `32.0%` | `65.4%` | `27.3%` |
| FastAPI | `0.100.0 -> 0.115.0` | `6.6%` | `34.0%` | `59.4%` | `29.8%` | `61.5%` |
| Django | `4.2 -> 5.0` | `52.3%` | `23.9%` | `23.9%` | `73.1%` | `26.6%` |
| Django | `4.2 -> 5.1` | `42.3%` | `28.4%` | `29.3%` | `64.3%` | `35.3%` |
| Django | `4.2 -> 5.2` | `36.3%` | `30.5%` | `33.2%` | `59.4%` | `40.2%` |

Interpretation:

- naive chunk ids can look valid while silently pointing to wrong evidence;
- silent corruption grows with revision distance;
- Refmark separates preserved, changed, and stale evidence;
- stale is not a failure when the underlying evidence really changed or
  disappeared.

These numbers should not be read as universal performance claims. The naive
baseline is deliberately simple: same path plus same ordinal chunk id after a
revision. It represents a common failure mode for persisted eval labels,
citations, metadata, and review notes, but stronger registries or semantic
chunk-migration systems may do better. The benchmark is meant to test whether
evidence addresses are auditable across change, not to prove that one regioning
strategy beats every possible chunking system.

## Reproduce

See `examples/evidence_lifecycle_benchmark/README.md` for runnable commands.

The scripts emit both full JSON and compact `summary_rows`. Use
`--summary-output` or `--csv-output` when you want a table for a paper,
spreadsheet, CI dashboard, or model-review handoff. Generated outputs are
intentionally ignored by Git.

## Limitations

The current implementation is a benchmark prototype:

- regioning is simple word-window based;
- fuzzy matching uses token Jaccard;
- no human relevance review is included;
- natural revisions are limited to documentation repositories.
- large structural rewrites can correctly produce many stale/review-needed refs;
- metric quality is bounded by manifest quality and by the chosen fuzzy-match
  threshold.

These limits are acceptable for the first claim: persistent evidence labels need
validation and stale-state handling across corpus updates. A production registry
would use stronger structural anchors, hashes, semantic matching, and human
review workflows.
