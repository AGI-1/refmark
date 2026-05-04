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

Current lifecycle artifacts now carry both coarse compatibility fields and a
more actionable `lifecycle_by_ref` map. The coarse table remains:

```text
preserved / review-needed / stale
```

The richer review-state vocabulary is meant for human review and adaptation
planning:

| State | Meaning | Typical action |
| --- | --- | --- |
| `unchanged` | same evidence hash still resolves locally | keep label |
| `moved` | exact evidence survived at a new ordinal or path | migrate ref |
| `rewritten` | high-similarity candidate exists but text changed | review rewrite or preserve |
| `split_support` | old evidence appears covered by multiple new regions | repair to a range |
| `deleted` | no exact or high-confidence candidate survived | refresh or remove label |
| `ambiguous` | multiple selector candidates compete | human disambiguation |
| `alternative_support` | another support candidate may be valid | review alternate support |
| `low_confidence` | candidate exists but is below auto-preserve gates | review manually |

Future states such as `merged`, `partial_overlap`, `semantic_drift`,
`superseded`, `deprecated`, `externalized`, `duplicate_support`,
`contradictory_support`, and `invalidated` are reserved for richer
review/adaptation workflows. They are useful categories, but the current
deterministic benchmark does not claim to detect all of them automatically.

Each lifecycle decision carries a reason, confidence, candidate ref when known,
suggested next action, and coarse priority. Confidence is selector/resolver
confidence, not semantic truth.

## Natural Revision Results

Five Git-backed documentation corpora were checked from one base version to
later release tags/branches with `python -m refmark.cli lifecycle-git`.

| Corpus | Base -> Target | Refmark Auto | Refmark Review | Refmark Stale | Naive Correct | Naive Silent Wrong |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FastAPI | `0.100.0 -> 0.105.0` | `54.2%` | `39.1%` | `6.7%` | `71.5%` | `21.3%` |
| FastAPI | `0.100.0 -> 0.110.0` | `42.5%` | `25.5%` | `32.0%` | `65.4%` | `27.3%` |
| FastAPI | `0.100.0 -> 0.115.0` | `6.6%` | `34.0%` | `59.4%` | `29.8%` | `61.5%` |
| Django | `4.2 -> 5.0` | `52.2%` | `23.8%` | `23.9%` | `73.1%` | `26.6%` |
| Django | `4.2 -> 5.1` | `42.3%` | `28.4%` | `29.3%` | `64.3%` | `35.3%` |
| Django | `4.2 -> 5.2` | `36.3%` | `30.5%` | `33.2%` | `59.4%` | `40.2%` |
| Flask | `2.2.0 -> 2.3.0` | `57.5%` | `16.7%` | `25.8%` | `72.7%` | `19.6%` |
| Flask | `2.2.0 -> 3.1.0` | `49.1%` | `19.9%` | `31.0%` | `67.1%` | `22.9%` |
| HTTPX | `0.23.0 -> 0.24.0` | `56.9%` | `5.8%` | `37.2%` | `61.3%` | `32.9%` |
| HTTPX | `0.23.0 -> 0.28.0` | `21.9%` | `23.4%` | `54.7%` | `35.0%` | `20.4%` |
| Kubernetes | `release-1.28 -> release-1.29` | `63.4%` | `18.0%` | `18.7%` | `77.6%` | `19.2%` |
| Kubernetes | `release-1.28 -> release-1.31` | `38.9%` | `27.6%` | `33.5%` | `54.4%` | `35.4%` |

Interpretation:

- naive chunk ids can look valid while silently pointing to wrong evidence;
- silent corruption grows with revision distance;
- Refmark separates preserved, changed, and stale evidence;
- stale is not a failure when the underlying evidence really changed or
  disappeared.
- large restructuring can be a negative review-workload case: for example
  HTTPX `0.23.0 -> 0.28.0` moved from a compact docs layout to many split files,
  so many old evidence labels correctly become stale or review-needed.

These numbers should not be read as universal performance claims. The naive
baseline is deliberately simple: same path plus same ordinal chunk id after a
revision. It represents a common failure mode for persisted eval labels,
citations, metadata, and review notes, but stronger registries or semantic
chunk-migration systems may do better. The benchmark is meant to test whether
evidence addresses are auditable across change, not to prove that one regioning
strategy beats every possible chunking system.

## Reproduce

See `examples/evidence_lifecycle_benchmark/README.md` for runnable commands.
The public API entry point is `refmark.lifecycle.evaluate_git_revisions(...)`;
the old example script is kept as a thin compatibility wrapper around that API.

For CI over an already mapped corpus, use `lifecycle-validate-labels` instead
of the Git benchmark:

```bash
python -m refmark.cli lifecycle-validate-labels \
  .refmark/manifest.new.jsonl \
  eval_questions.old.jsonl \
  --previous-manifest .refmark/manifest.old.jsonl \
  --max-stale 0 \
  --max-removed-refs 0 \
  --output runs/lifecycle_report.json
```

The command exits `3` when a configured lifecycle threshold fails. The report
separates eval-row staleness from corpus-map churn:

- `stale_examples`: maintained `query -> gold_refs` rows whose evidence changed
  or disappeared;
- `changed_refs`: refs still present in the new manifest with different source
  hashes;
- `removed_refs`: refs present in the previous manifest but absent now.

When a pipeline only needs the shadow-manifest delta, use `manifest-diff`:

```bash
python -m refmark.cli manifest-diff \
  .refmark/manifest.old.jsonl \
  .refmark/manifest.new.jsonl \
  --examples eval_questions.old.jsonl \
  --max-stale 0 \
  --output runs/manifest_diff.json
```

The `refmark.manifest_diff.v1` artifact is intentionally smaller than full
label validation. It reports address-space churn, corpus fingerprints, optional
affected eval examples, and the same CI-style threshold status.

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
