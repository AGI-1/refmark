# Data-Smell Analysis

Refmark data smells are review signals, not automatic proof that the corpus is
wrong. They point reviewers and adaptation agents at places where retrieval,
labels, corpus structure, or metadata deserve attention.

## Eval-Run Smells

`build_data_smell_report(suite, run)` converts an `EvalSuite` and `EvalRun` into
a first-class report:

```python
from refmark import build_data_smell_report

run = suite.evaluate(retriever, name="hybrid", k=10)
report = build_data_smell_report(suite, run)
report.write_json("runs/smells.json")
```

The CLI also emits the same report from `eval-index`:

```bash
python -m refmark.cli eval-index index.json eval.jsonl \
  --manifest manifest.jsonl \
  --smell-report-output runs/smells.json \
  --adapt-plan-output runs/adaptation_plan.json \
  -o runs/eval.json
```

Or convert a saved smell report into a conservative adaptation plan later:

```bash
python -m refmark.cli adapt-plan runs/smells.json -o runs/adaptation_plan.json
```

The report schema is `refmark.data_smells.v1` and contains:

- `summary`: counts by type/severity plus run/corpus fingerprints;
- `smells`: reviewable items with refs, evidence, and suggested actions.

The adaptation-plan schema is `refmark.adaptation_plan.v1`. It does not mutate
the corpus. It normalizes smells into review-required actions such as stale
label refresh, shadow metadata/doc2query additions, confusion-pair mapping,
range/context tuning, confidence gating, and query-magnet role assignment.

Current smell types:

| Type | Meaning | Common action |
| --- | --- | --- |
| `stale_label` | Saved eval label points to changed or missing source text. | Refresh or regenerate the label after review. |
| `hard_ref` | A region is repeatedly missed at `@k`. | Add questions, metadata, aliases, or review boundaries. |
| `confusion_pair` | Gold ref repeatedly retrieves a competing top ref. | Add hard negatives, disambiguators, or valid alternate refs. |
| `query_style_gap` | Direct questions pass while concern/adversarial styles fail. | Generate/adapt weak query styles. |
| `gold_mode_gap` | Single/range/distributed evidence modes behave differently. | Tune range/context expansion and add coverage rows. |
| `undercitation` | Context misses required gold refs. | Expand context or increase candidate recall. |
| `overcitation` | Context includes broad non-gold evidence. | Tighten expansion or split broad regions. |
| `low_confidence` | Top scores have narrow margins. | Route to reranker, review, or multi-candidate UI. |
| `query_magnet` | A ref attracts many wrong top hits. | Downweight/exclude hub content or add disambiguation. |

Each smell should include enough information for an agent or reviewer to act:

- gold and competing refs;
- sample queries;
- score/margin evidence where available;
- source hashes and text snippets;
- suggested adaptation actions.

## Index/Corpus Smells

`inspect-index` reports deterministic corpus/index smells without an eval run:

```bash
python -m refmark.cli inspect-index index.json -o index_smells.json
```

That report includes query magnets, oversized regions, sparse retrieval views,
exact duplicates, duplicate candidates, and possible conflict wording. These
signals are useful before any questions exist.

## How To Use Smells

A typical improvement loop is:

1. Run retrieval eval and write a smell report.
2. Convert the report to an adaptation plan.
3. Sort by severity and repeated confusions.
4. For each high-value action, decide whether it is a corpus issue, label issue,
   metadata issue, or retriever issue.
5. Apply a scoped adaptation: shadow metadata, query rewrite, alternate gold,
   region boundary change, exclusion role, or reranker/hard-negative update.
6. Re-run the eval and compare the smell report against the previous run.

The point is visibility. A greener heatmap is only valuable when the smell
report shows that real weak topics, stale labels, or repeated confusions were
reduced without introducing new query magnets elsewhere.
