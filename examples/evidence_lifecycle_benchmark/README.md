# Evidence Lifecycle Benchmark

This example tests the Refmark lifecycle claim:

> Stable evidence refs reduce silent corruption in maintained retrieval,
> evaluation, citation, and training labels when a corpus changes.

It is not a retrieval leaderboard. The question is what happens to old
`query -> evidence` labels, citations, annotations, or metadata when the source
corpus is updated.

## Why This Exists

Many RAG systems treat chunks as disposable index artifacts. That is fine when
the index is rebuilt and nothing else refers to old chunks.

But production systems often keep side data:

- curated eval examples: `question -> evidence`
- training labels
- generated doc2query metadata
- cached answer citations
- human review notes
- hard negatives for rerankers

If ordinary chunk ids are reused after re-chunking, they can silently point to
different evidence. Refmark-style refs should instead be validated, migrated,
marked review-needed, or marked stale.

## Natural Git Revision Benchmark

Run from the repository root:

```bash
python examples/evidence_lifecycle_benchmark/evaluate_git_revision_stability.py \
  --repo-url https://github.com/tiangolo/fastapi.git \
  --old-ref 0.100.0 \
  --new-refs 0.105.0,0.110.0,0.115.0 \
  --subdir docs/en/docs \
  --work-dir examples/evidence_lifecycle_benchmark/output/git_revision_work/fastapi \
  --output examples/evidence_lifecycle_benchmark/output/git_revision_fastapi_curve.json \
  --summary-output examples/evidence_lifecycle_benchmark/output/git_revision_fastapi_summary.json \
  --csv-output examples/evidence_lifecycle_benchmark/output/git_revision_fastapi_summary.csv
```

```bash
python examples/evidence_lifecycle_benchmark/evaluate_git_revision_stability.py \
  --repo-url https://github.com/django/django.git \
  --old-ref 4.2 \
  --new-refs 5.0,5.1,5.2 \
  --subdir docs \
  --work-dir examples/evidence_lifecycle_benchmark/output/git_revision_work/django \
  --output examples/evidence_lifecycle_benchmark/output/git_revision_django_curve.json \
  --summary-output examples/evidence_lifecycle_benchmark/output/git_revision_django_summary.json \
  --csv-output examples/evidence_lifecycle_benchmark/output/git_revision_django_summary.csv
```

The script compares:

- stable ref migration by content fingerprint and fuzzy matching;
- naive path + ordinal chunk identity;
- estimated maintenance workload for old `query -> evidence` labels.

To combine one or more result files into a table:

```bash
python examples/evidence_lifecycle_benchmark/summarize_lifecycle_results.py \
  examples/evidence_lifecycle_benchmark/output/git_revision_fastapi_curve.json \
  examples/evidence_lifecycle_benchmark/output/git_revision_django_curve.json \
  --format markdown \
  --output examples/evidence_lifecycle_benchmark/output/lifecycle_summary.md
```

## Controlled Mutation Benchmark

For a smaller controlled sanity check:

```bash
python examples/evidence_lifecycle_benchmark/evaluate_ref_stability_mutations.py \
  --dataset beir/scifact/test \
  --output examples/evidence_lifecycle_benchmark/output/ref_stability_scifact.json \
  --summary-output examples/evidence_lifecycle_benchmark/output/ref_stability_scifact_summary.json \
  --csv-output examples/evidence_lifecycle_benchmark/output/ref_stability_scifact_summary.csv
```

This creates a synthetic `t -> t+1` corpus by inserting, deleting, and lightly
editing regions, then compares stable migration against naive same-numbered
chunks.

## Reading The Results

Key fields:

- `stable_ref_migration`: exact, moved, fuzzy, and stale region outcomes.
- `naive_path_ordinal_identity`: whether the same path/ordinal chunk still
  points to the same evidence.
- `eval_label_lifecycle`: estimated effect on maintained eval/training labels.
- `summary_rows`: compact table rows for reports, spreadsheets, and dashboards.

The most important failure mode is `naive.silent_wrong`: the old chunk address
still resolves, but to different evidence.

## Current Observed Pattern

In FastAPI and Django documentation revision curves, naive chunk ids accumulated
silent wrong-evidence rates around 20-60% depending on revision distance. Refmark
did not magically preserve all labels; instead it separated labels into:

- automatically preserved or migrated;
- fuzzy review-needed;
- stale/deleted.

That distinction is the core product value: corpus drift becomes inspectable
instead of silently poisoning evaluation, training, citations, and metadata.
