# Coverage Alignment Example

This example shows a small human-in-the-loop review pipeline over DOCX and PDF
inputs:

1. generate example `.docx` and `.pdf` files
2. extract text into Refmark regions
3. align source requirements to target offer/specification regions
4. compare naive best-hit coverage with best-hit-plus-neighbor expansion
5. write JSON and HTML review artifacts with stable region anchors

Run from the repository root:

```bash
python examples/coverage_alignment/run.py
```

Artifacts are written under `examples/coverage_alignment/output/`.

The generated scenarios are:

- `customer_request.docx` vs `offer_contract.pdf`
- `tender.docx` vs `technical_specification.pdf`

The example intentionally filters title regions out of coverage scoring. The
underlying CLI maps all extracted regions; a production review workflow can
choose whether headings are evidence, navigation, or ignored metadata.

The example uses `refmark_workflow.yaml` as a human-editable configuration
shape. Useful knobs are:

- `density`: `dense`, `balanced`, `coarse`, or `code`
- `marker_style`: `machine`, `explicit`, `compact`, or `xml`
- `include_headings`: whether title-like regions remain in review scoring
- `coverage_threshold`: score threshold for covered vs gap
- `expand_before` / `expand_after`: controlled neighbor expansion
- `numeric_checks`: small deterministic checks for units such as days, years,
  hours, kWh, and percent

You can also run the CLI directly:

```bash
python -m refmark.cli map examples/coverage_alignment/output/inputs -o .refmark/coverage_inputs_manifest.jsonl
python -m refmark.cli align examples/coverage_alignment/output/inputs/customer_request.docx examples/coverage_alignment/output/inputs/offer_contract.pdf --config examples/coverage_alignment/refmark_workflow.yaml --coverage-html .refmark/customer_offer_coverage.html --summary-json .refmark/customer_offer_summary.json
```
