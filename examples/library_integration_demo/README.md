# Library Integration Demo

This example shows the smallest non-CLI integration shape:

```text
existing retriever callback -> EvalSuite.evaluate(...)
                            -> evidence metrics
                            -> data-smell report
                            -> adaptation plan
                            -> comparable run artifact
```

Run it from the repository root:

```bash
python examples/library_integration_demo/run.py
```

The retriever can be anything that returns stable refs or hit dictionaries with
`stable_ref`, `score`, and optional `context_refs`. Refmark does not own the
retrieval stack; it scores whether the stack recovered the expected evidence.

The script writes ignored artifacts under `examples/library_integration_demo/output/`:

- `run_artifact.json`
- `metrics.json`
- `smells.json`
- `adaptation_plan.json`
- `summary.json`
