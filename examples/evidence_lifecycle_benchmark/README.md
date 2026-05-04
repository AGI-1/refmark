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
python -m refmark.cli lifecycle-git \
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
python -m refmark.cli lifecycle-git \
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
- chunk id + content hash;
- qrels/source-file hash;
- chunk id + content hash + quote selector + corpus version;
- Refmark layered selector: content hash, quote selector, and similarity
  threshold with ambiguous cases sent to review;
- estimated maintenance workload for old `query -> evidence` labels.

To combine one or more result files into a table:

```bash
python -m refmark.cli lifecycle-summarize \
  examples/evidence_lifecycle_benchmark/output/git_revision_fastapi_curve.json \
  examples/evidence_lifecycle_benchmark/output/git_revision_django_curve.json \
  --format markdown \
  --output examples/evidence_lifecycle_benchmark/output/lifecycle_summary.md
```

To aggregate the skeptical method-comparison table across full benchmark
payloads:

```bash
python examples/evidence_lifecycle_benchmark/summarize_lifecycle_results.py \
  --aggregate-methods \
  examples/evidence_lifecycle_benchmark/output/competent_full_fastapi.json \
  examples/evidence_lifecycle_benchmark/output/competent_full_flask.json \
  examples/evidence_lifecycle_benchmark/output/competent_full_httpx.json \
  examples/evidence_lifecycle_benchmark/output/competent_full_django.json \
  examples/evidence_lifecycle_benchmark/output/competent_full_kubernetes.json \
  --format markdown \
  --output examples/evidence_lifecycle_benchmark/output/competent_lifecycle_5corpus_method_comparison.md
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
- `chunk_id_content_hash_identity`: conservative same chunk id plus same content
  hash baseline. It avoids silent drift but over-alerts when evidence moved or
  changed cosmetically.
- `qrels_source_hash_identity`: qrels-style labels with source-file hashes. It
  is easy to implement, but changes to a large file can force review of many
  still-valid labels.
- `chunk_hash_quote_selector_identity`: the competent skeptical baseline:
  chunk id, content hash, quote selector, and corpus version.
- `refmark_layered_selector_identity`: a safer layered variant that borrows the
  useful quote-selector trick but keeps Refmark-style ambiguity handling. It
  accepts exact hashes or unique quote hits only when full-region similarity is
  high enough; otherwise it queues review.
- `eval_label_lifecycle`: estimated effect on maintained eval/training labels.
  Its `method_comparison` table reports silent drift, false stale alerts, human
  review workload, and valid evals preserved for each identity strategy.
- `summary_rows`: compact table rows for reports, spreadsheets, and dashboards.

The most important failure mode is `naive.silent_wrong`: the old chunk address
still resolves, but to different evidence.

## What Would Convince A Skeptical Engineer

The benchmark is intended to answer a stricter question than "are ordinal chunk
ids bad?". A competent baseline would store more than an ID:

```text
source path + corpus version + chunk id + chunk hash + quote selector
```

Refmark should be compared against that baseline, not only against naive chunk
ids. The useful table is:

| Method | Silent drift | False stale alerts | Human review workload | Valid evals preserved |
| --- | ---: | ---: | ---: | ---: |
| chunk ID only | `method_comparison.chunk_id_only.rates.silent_drift` | low by construction | low but unsafe | may be misleading |
| chunk ID + content hash | usually low | can be high | conservative | lower when evidence moves |
| qrels + source hash | usually low | can be high for large files | conservative | file-change dependent |
| chunk hash + quote selector | low if quote is unique | lower | moderate | strong baseline |
| Refmark | should be low | should be acceptable | review queue is explicit | high when refs can migrate |

The current implementation uses Refmark's exact/fuzzy migration as the first
benchmark oracle. A stronger future benchmark should add human-reviewed labels
for a sampled subset of moved, rewritten, split, and deleted evidence.

## Five-Corpus Competent-Baseline Run

On 2026-05-03, the benchmark was rerun across FastAPI, Flask, HTTPX, Django,
and Kubernetes documentation revision curves. The aggregate covered 51,624 old
region labels across 15 revision comparisons.

| Method | Silent drift | Silent drift count | False stale alerts | Human review workload | Safe preserved labels |
| --- | ---: | ---: | ---: | ---: | ---: |
| chunk ID only | 30.0% | 15,499 | 0.0% | 0.0% | 34,061 / 66.0% |
| chunk ID + content hash | 0.0% | 0 | 25.3% | 52.7% | 24,436 / 47.3% |
| qrels + source hash | 0.0% | 0 | 42.6% | 69.9% | 15,514 / 30.1% |
| chunk hash + quote selector | 0.8% | 427 | 21.0% | 48.1% | 26,389 / 51.1% |
| Refmark layered selector | 0.0% | 0 | 0.0% | 48.6% | 26,520 / 51.4% |
| Refmark exact/migration only | 0.0% | 0 | 0.0% | 52.1% | 24,742 / 47.9% |

This is a useful but not final "convince me" result. It changes the claim:
Refmark's exact/migration layer is not the best automatic-preservation
mechanism by itself. The chunk-hash-plus-quote-selector baseline preserved more
labels than exact Refmark migration, but left 427 silent wrong labels. The
layered Refmark selector borrowed the quote-selector preservation trick while
keeping unsafe cases explicit: it preserved slightly more labels than the quote
selector baseline, with no observed silent drift under the current migration
oracle and no false stale alerts.

The tradeoff is:

- naive chunk IDs preserve many labels but silently corrupt about 30%;
- conservative hash/source baselines avoid silent drift but generate large
  review queues and false stale alerts;
- quote selectors are a strong baseline and have a slightly smaller review
  queue than layered Refmark, but still leave measurable silent drift: 427
  silent wrong labels in this run;
- Refmark's value in this run is explicit lifecycle state:
  preserved/migrated, review-needed, or stale, without observed silent drift
  under the current migration oracle.

The oracle caveat is not cosmetic. The current aggregate uses Refmark's own
exact/fuzzy migration heuristic to decide which old labels are still valid, so
the table is best read as a lifecycle-method comparison under that oracle. A
human-reviewed disagreement slice is still required before treating these as
publication-grade semantic-correctness numbers.

An intermediate LLM-adjudicated review pass was run over 200 sampled
disagreement/review cards. DeepSeek v3.2 judged all 200 cards, and Qwen 3.6
Plus plus Grok 4.3 judged the top 80 escalation cards. The run produced 360
successful judgments with no provider failures in the final setup. Majority
judgments classified 187/200 cards as still valid evidence
(`valid_unchanged`, `valid_moved`, `valid_rewritten`, or `split_support`) and
13/200 as stale. This does not replace human review, but it clarifies the
shape of the problem:

- most Refmark fuzzy/review-needed cases were judged to be valid rewritten or
  moved evidence, so the review queue contains many auto-preservation
  opportunities;
- quote-selector silent-drift candidates were mixed: many remained valid, but
  a non-trivial subset was judged stale or split-support, which is the hidden
  failure mode Refmark is designed to expose;
- the next algorithmic target is to promote more review-needed cases safely
  while keeping observed silent drift at zero under the current migration
  oracle.

The next publication-grade evidence step is a human-reviewed slice of the
ambiguous and model-disagreement cases. That would test whether Refmark's
fuzzy/review queue and the quote-selector baseline are making the right calls
outside both the automated oracle and LLM adjudication.

To create that review packet from sampled cards and cached LLM judgments:

```bash
python examples/evidence_lifecycle_benchmark/review_lifecycle_disagreements.py worksheet \
  --cards examples/evidence_lifecycle_benchmark/output/lifecycle_review_sample_200.jsonl \
  --judgments examples/evidence_lifecycle_benchmark/output/lifecycle_review_combined_200_plus_escalation.jsonl \
  --limit 60 \
  --output-csv examples/evidence_lifecycle_benchmark/output/lifecycle_human_review_top60.csv \
  --output-html examples/evidence_lifecycle_benchmark/output/lifecycle_human_review_top60.html
```

The CSV is the durable review artifact. It contains blank `human_verdict`,
`human_confidence`, and `human_notes` fields, plus model votes, method
decisions, signals, and the old/candidate text. The HTML is a convenience view
for diff-highlighted inspection. The highest-priority rows are intentionally
not random: they emphasize quote-selector silent-drift candidates, model
disagreements, and Refmark review-needed cases that may be safely promotable.

To render a filled CSV and produce a calibration report:

```bash
python examples/evidence_lifecycle_benchmark/review_lifecycle_disagreements.py filled-html \
  --input-csv examples/evidence_lifecycle_benchmark/output/lifecycle_human_review_top60_codex_filled.csv \
  --output-html examples/evidence_lifecycle_benchmark/output/lifecycle_human_review_top60_codex_filled.html

python examples/evidence_lifecycle_benchmark/review_lifecycle_disagreements.py calibrate \
  --input-csv examples/evidence_lifecycle_benchmark/output/lifecycle_human_review_top60_codex_filled.csv \
  --output examples/evidence_lifecycle_benchmark/output/lifecycle_human_review_top60_calibration.md
```

A Codex-filled top-60 worksheet is useful as an internal calibration aid, not
as independent human evidence. In the current top-60 pass, 46/60 labels were
adjudicated as preservable, 7 as stale, 3 as split-support, and 4 as ambiguous.
The main actionable signal was asymmetric:

- Refmark fuzzy/layered review cases were mostly recoverable: 15/16 were
  adjudicated as valid moved or rewritten evidence;
- quote-selector silent-drift candidates were mixed: 31/44 were still valid,
  but 6 were stale, 3 needed split-support/range repair, and 4 were ambiguous.

That suggests the next deterministic selector should not choose between
Refmark and quote selectors. It should combine them: use quote/heading/context
signals to recover more moved or rewritten evidence, but keep low-similarity,
split-support, and ambiguous cases in explicit lifecycle states.

The next product/research target is also clear:

| Target | Current layered Refmark | Current quote-selector baseline | vNext goal |
| --- | ---: | ---: | ---: |
| silent drift | 0.0% | 0.8% | keep 0.0% |
| false stale alerts | 0.0% | 21.0% | keep near 0.0% |
| review workload | 48.6% | 48.1% | below quote selector |
| safe preserved labels | 51.4% | 51.1% | stay above quote selector |

Likely path: layered deterministic anchoring, such as structural path, exact
quote selector, content hash, neighboring-context hash, section-heading
lineage, and explicit ambiguity thresholds. Optional semantic checks can be
added later, but they should be cached or reviewed metadata rather than hidden
runtime magic.

## Layered Selector Update

The next iteration implemented the first deterministic layered selector:

```text
same path/ordinal + content hash
or unique quote selector hit + full-region similarity >= threshold
or same path/ordinal + short quote + very high full-region similarity
otherwise review/stale
```

The default thresholds are deliberately conservative documentation-benchmark
settings: `0.82` token-Jaccard for fuzzy/quote preservation and `0.95` for
same-ordinal rewrites. A looser `0.70` same-ordinal gate preserved more labels
in this benchmark but reintroduced silent drift, so it is not the default.

The full five-corpus run was repeated after fixing a migration-oracle bug where
same-file fuzzy candidates could hide stronger cross-file matches. The matcher
now checks both same-file and global candidates, using an indexed token matcher
so large corpora remain practical.

Corrected aggregate:

| Method | Silent drift | Silent drift count | False stale alerts | Human review workload | Safe preserved labels |
| --- | ---: | ---: | ---: | ---: | ---: |
| chunk ID only | 30.0% | 15,499 | 0.0% | 0.0% | 34,061 / 66.0% |
| chunk ID + content hash | 0.0% | 0 | 25.3% | 52.7% | 24,436 / 47.3% |
| qrels + source hash | 0.0% | 0 | 42.6% | 69.9% | 15,514 / 30.1% |
| chunk hash + quote selector | 0.8% | 427 | 21.0% | 48.1% | 26,389 / 51.1% |
| Refmark layered selector | 0.0% | 0 | 0.0% | 48.6% | 26,520 / 51.4% |
| Refmark exact/migration only | 0.0% | 0 | 0.0% | 52.1% | 24,742 / 47.9% |

This is the stronger result: Refmark can borrow quote-selector preservation
without inheriting quote-selector silent drift. The stricter layered selector
preserved slightly more labels than quote selectors while converting 427
silently wrong quote-selector labels into explicit review/stale states. A looser
same-ordinal rewrite gate improved preservation further but reintroduced silent
drift, so the default remains conservative.

That makes the near-term claim sharper:

> Refmark's durable value is not that it preserves the most labels by default.
> It lets preservation strategies be layered behind a stable evidence address
> space while turning unsafe cases into inspectable lifecycle states.

The remaining target is to reduce the extra review workload without
reintroducing silent drift. Candidate layers include neighboring-context hashes,
heading lineage, structural path similarity, split-support detection, and cached
human/LLM review decisions.

## Current Observed Pattern

In FastAPI and Django documentation revision curves, naive chunk ids accumulated
silent wrong-evidence rates around 20-60% depending on revision distance. Refmark
did not magically preserve all labels; instead it separated labels into:

- automatically preserved or migrated;
- fuzzy review-needed;
- stale/deleted.

That distinction is the core product value: corpus drift becomes inspectable
instead of silently poisoning evaluation, training, citations, and metadata.
