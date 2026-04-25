# Refmark Examples

These examples are intentionally small and inspectable. They are meant for
researchers and developers who want to poke at the core claims without running
a broad benchmark harness.

Run them from the repository root:

```bash
python examples/citation_qa/run_eval.py
python examples/data_smells/run.py
python examples/judge_free_rewards/run.py
python examples/multidiff_demo/run.py
python examples/pipeline_primitives/run.py
python examples/coverage_alignment/run.py
```

## Citation QA

`citation_qa` demonstrates the loop behind cheap citation evaluation:

1. copy a tiny corpus into `output/`
2. inject addressable refs
3. score predicted refs against gold refs with exact/overlap/cover metrics
4. render cited regions as text and HTML for audit

Edit `examples/citation_qa/predictions.json` to see the scores change.

## Data Smells

`data_smells` compares two mock models:

- one predicts completely wrong locations
- one finds the right neighborhood but overcites or undercites

This demonstrates why locate-only metrics are more informative than exact
match alone.

## Judge-Free Rewards

`judge_free_rewards` uses the retained
`refmark_train/data/documentation_full_paragraph_contextual_idf_lean2` dataset
to show deterministic continuous rewards for exact, overbroad, wrong-location,
and missing citation outputs. No LLM judge or API call is used.

## Multidiff Demo

`multidiff_demo` demonstrates bounded same-file edits:

1. copy a source file into `output/`
2. inject live refs
3. apply two successful edits in one payload
4. try one intentionally stale edit and show that the file is unchanged

Edit `examples/multidiff_demo/good_edits.json` to try your own ref-addressed
patches.

## Pipeline Primitives

`pipeline_primitives` demonstrates the plug-and-play document surface:

1. map documents into a JSONL region manifest
2. expand a retrieved region into neighboring context
3. align source document regions to target document regions
4. emit a paste-ready prompt that asks a general chat model for cited region ranges

## Coverage Alignment

`coverage_alignment` generates small `.docx` and `.pdf` input documents and
runs two review flows:

- customer request vs offer/contract
- tender requirements vs technical specification

The example writes marked text, region manifests, naive and expanded coverage
JSON, and an HTML review page that highlights covered items and gaps by stable
Refmark regions.
