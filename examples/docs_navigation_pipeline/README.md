# Documentation Navigation Pipeline

This is the small frozen path for the product-shaped Refmark workflow:

```text
software docs -> refmark manifest -> portable search index
              -> evidence eval -> data smells -> adaptation plan
              -> local navigation
```

The runtime side has no vector database, no API call, no embedding service, and
no server requirement. The output is a JSON search index that can be searched
from Python, a CLI, or a static browser page.

## Run The Full Example

From the repository root:

```bash
python examples/docs_navigation_pipeline/run.py
```

Or run the configurable easy-mode full pipeline:

```bash
python -m refmark.cli run-pipeline examples/docs_navigation_pipeline/refmark_pipeline.local.yaml
```

The script writes ignored artifacts under `examples/docs_navigation_pipeline/output/`:

- `corpus.refmark.jsonl`: stable region manifest for the source docs
- `sections.json`: TOC/section map from headings to stable ref ranges
- `docs.index.json`: portable BM25 region index
- `docs.browser.json`: compact browser-search payload
- `eval_flat.json`, `eval_hierarchical.json`, `eval_rerank.json`: evidence eval reports
- `compare_index.json`: one side-by-side comparison of flat, hierarchical, and reranked modes
- `smells.json`: reviewable data-smell report from the rerank eval
- `adaptation_plan.json`: conservative review-required adaptation actions
- `sample_query.json`: example navigation hits for a free-text query

The full-pipeline config writes the same family of artifacts under
`examples/docs_navigation_pipeline/output_full/`, plus `run_summary.json`.

## Equivalent CLI Commands

```bash
python -m refmark.cli map examples/docs_navigation_pipeline/sample_corpus \
  -o examples/docs_navigation_pipeline/output/corpus.refmark.jsonl

python -m refmark.cli build-index examples/docs_navigation_pipeline/sample_corpus \
  -o examples/docs_navigation_pipeline/output/docs.index.json \
  --source local

python -m refmark.cli toc \
  examples/docs_navigation_pipeline/output/corpus.refmark.jsonl \
  -o examples/docs_navigation_pipeline/output/sections.json

python -m refmark.cli export-browser-index \
  examples/docs_navigation_pipeline/output/docs.index.json \
  -o examples/docs_navigation_pipeline/output/docs.browser.json

python -m refmark.cli eval-index \
  examples/docs_navigation_pipeline/output/docs.index.json \
  examples/docs_navigation_pipeline/eval_questions.jsonl \
  --manifest examples/docs_navigation_pipeline/output/corpus.refmark.jsonl \
  --strategy rerank \
  --top-k 5 \
  --smell-report-output examples/docs_navigation_pipeline/output/smells.json \
  --adapt-plan-output examples/docs_navigation_pipeline/output/adaptation_plan.json \
  -o examples/docs_navigation_pipeline/output/eval_rerank.json

python -m refmark.cli compare-index \
  examples/docs_navigation_pipeline/output/docs.index.json \
  examples/docs_navigation_pipeline/eval_questions.jsonl \
  --manifest examples/docs_navigation_pipeline/output/corpus.refmark.jsonl \
  --strategies flat,hierarchical,rerank \
  --top-k 5 \
  -o examples/docs_navigation_pipeline/output/compare_index.json

python -m refmark.cli search-index \
  examples/docs_navigation_pipeline/output/docs.index.json \
  "How do I rotate API tokens without downtime?" \
  --strategy rerank \
  --top-k 3 \
  --expand-after 1
```

## What This Proves

This example is deliberately modest. It proves the integration shape:

- a documentation folder can become a stable ref-addressed corpus;
- a heading/TOC layer can point to section-level ref ranges without modifying
  source docs;
- the local index returns concrete refs such as `security:P02`;
- the same refs can be evaluated against a small query suite;
- eval weaknesses become data-smell reports and adaptation-plan actions instead
  of disappearing into one average score;
- the browser export can be shipped with docs for client-side navigation.

It does not claim that local BM25 beats embeddings or vector databases. Refmark
makes that comparison easy: run the external retriever separately, export
`query -> hits`, and score it with `eval-index --retriever-results`. The runtime
tradeoff is explicit:

| Runtime path | Extra infra | What it buys |
| --- | --- | --- |
| Local Refmark BM25 | none | tiny, static, inspectable navigation with exact refs |
| LLM-enriched Refmark BM25 | build-time API only | better semantic wording while keeping runtime local |
| Embeddings/vector DB | model/service/index infra | stronger semantic recall, especially for indirect queries |

The current larger public-doc experiments in `docs/EVIDENCE_RETRIEVAL_PIPELINE.md`
show that cached LLM retrieval views improved rank-1 evidence localization on
FastAPI, Django, Kubernetes, Rust, TypeScript, and a mixed 1M-token corpus while
the final search still ran locally.
