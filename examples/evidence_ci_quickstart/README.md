# Evidence CI Quickstart

Tiny committed fixture for the default Refmark evidence-CI loop. It is small on
purpose: the goal is to show the product path without remote models, generated
artifacts, or a large corpus.

Run from the repository root:

```bash
python -m refmark.cli ci examples/evidence_ci_quickstart/corpus examples/evidence_ci_quickstart/eval.jsonl --out-dir tmp/evidence_ci_quickstart --source local --min-hit-at-k 1.0 --min-best-hit-at-k 1.0 --fail-on-regression
```

The command writes a manifest, portable search index, eval report, comparison
report, data-smell report, and adaptation plan into the output directory.
