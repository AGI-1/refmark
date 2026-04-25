![Refmark banner](refmark.png)
## Refmark
Refmark is a research toolkit for making corpora addressable to AI systems.
It injects stable, resolvable anchors into documents or code so models can
point at source regions by id instead of by fragile prose, copied snippets, or
line numbers.

This does not guarantee that a model cites the right region. It guarantees a
different and useful thing: cited regions exist, resolve back to source text,
and can be audited. For review workflows, an irrelevant citation and a
fabricated citation are not the same failure. One is inspectable; the other is
not.

Once a corpus has addresses, citation behavior becomes structured data:
exact hits, overlap, overcitation, undercitation, wrong-region hits, and
scattered citations can be measured without an LLM judge. Those failures are
often useful signals in their own right. Repeated wrong-region hits can reveal
ambiguous passages, stale labels, weak questions, or missing support. Diffuse
"all around" citations can indicate that no clean support span exists or that
the anchor granularity is wrong.

The same addressability primitive also supports stable same-file multi-region
edits. Instead of asking a model to patch drifting line numbers or copied
context, Refmark lets tools target explicit regions and apply bounded edits
through `apply_ref_diff`.

This publish layout keeps the stable surface small and explicit:

1. deterministic locate-only QA and citation evaluation with data-smell metrics
2. highlighted review of cited regions for human-in-the-loop audit workflows
3. stable same-file multi-region editing for Python and TypeScript through `apply_ref_diff`
4. exploratory corpus-local anchor prediction with retained derived datasets
5. small deterministic smoke checks that prove the public artifact works locally

## Included Here

- `refmark/`
  - the publishable package surface
- `scripts/`
  - shell-friendly wrappers for `apply_ref_diff` and persistent shadow sessions
- `tests/`
  - focused tests for the stable surface
- `docs/`
  - curated user-facing and publication-facing notes
- `examples/`
  - runnable citation-evaluation and multidiff playgrounds
- `refmark_train/`
  - exploratory corpus-local citation-localization prototype with retained derived datasets

## Not Included In This Publish Surface

- broad benchmark CLI scaffolding
- SWE-bench harnesses
- agentic experiment runners
- large result dumps
- redistributed raw/source document payloads

Those remain useful research assets, but they are intentionally outside this cleaned publish subtree.

## Quick Start

```bash
pip install -e .[dev,mcp,typescript,train]
python -m refmark.cli languages
python -m refmark.cli smoke
python -m refmark_train.verify_publish_artifact
python -m refmark_train.smoke
python examples/citation_qa/run_eval.py
python examples/data_smells/run.py
python examples/judge_free_rewards/run.py
python examples/multidiff_demo/run.py
pytest
```

To try the CLI on a real file without mutating the example source:

```bash
python -m refmark.cli inject examples/multidiff_demo/source.py --output marked.py
python -m refmark.cli highlight marked.py --refs F02,F03 --format text
```

## Core Workflow

Models can output raw citation refs such as:

```json
["F03", "F04"]
```

Refmark then makes that output auditable and measurable:

- `refmark.metrics.score_ref_range(...)` scores exactness, overlap, cover,
  precision/recall/F1, breadth, overcite, undercite, and wrong-location rates
- `python -m refmark.cli highlight file.py --refs F03,F04 --format html`
  renders clean source snippets for human review

Text audit output looks like:

```text
[F03] lines 8-11
>    8 | def compute_invoice_total(...):
>    9 |     shipping = 18.0 if expedited else 6.0
>   10 |     taxed = subtotal * (1.0 + TAX_RATE)
>   11 |     return round(taxed + shipping, 2)
```

This is the main research surface: locate-only outputs can be evaluated
deterministically, and misses can be separated into wrong-location failures,
overcitation, undercitation, or boundary mismatch.

## Judge-Free Rewards

For RLHF/DPO-style experiments, refs turn citation grading into deterministic
integer math. A model can emit `["D00284"]`, and the reward can be computed
without an LLM judge:

```python
from refmark.metrics import citation_reward, score_ref_range

score = score_ref_range(["D00283", "D00284"], ["D00284"])
reward = citation_reward(score)
```

The retained `refmark_train/data/documentation_full_paragraph_contextual_idf_lean2`
dataset provides thousands of anchored question/ref pairs for this style of
local reward experiment. Run:

```bash
python examples/judge_free_rewards/run.py
```

## Living Corpus Evaluation

Generating good questions for a fresh corpus is not free. Even with local
models, it costs time, power, and review attention. Refmark changes that cost
model by attaching evaluation examples to stable source addresses.

When the corpus changes, existing questions for unchanged anchors can remain
useful. New generation can focus on added, removed, or touched anchors, while a
cheap local model is retrained against the refreshed address space. That makes
anchored QA closer to regression testing than one-off benchmark construction:
build supervision once, preserve it across ordinary corpus mutations, and spend
generation budget where the corpus actually changed.

The retained `refmark_train` prototype explores this path. It intentionally
overfits small models to a fixed addressable corpus, treating narrow local
specialization as a feature rather than a bug. Current evidence supports this
as a promising corpus-local navigation experiment, not yet as a broad claim
about general model training.

## Current Evidence

The current evidence is strongest for:

- deterministic locate-only citation evaluation and human-auditable source
  review
- stable same-file anchored edits for bounded Python and TypeScript workflows
- data-smell diagnostics from wrong-region, broad, and scattered citations

The current evidence is more limited for:

- broad coding-agent superiority
- universal efficiency gains
- exact-minimal citation as a solved problem
- training-based localization as a proven product path

Existing general LLMs were not specifically trained to use injected anchors.
In local experiments, useful zero-shot anchor use generally appeared in larger
open models, while smaller models often struggled with the notation. Some
bounded SWE-style and multi-diff slices showed positive signals, especially
for weaker or mid-tier models, but strong coding models were often reliable
enough without Refmark.

The research hypothesis is that anchors will matter more when they are part of
the actual training or fine-tuning loop, rather than introduced only at
inference time. In particular, 4B-14B models trained to understand addressable
corpora may become better at local information navigation and bounded code
modification. This artifact is a proof of concept for that path, not proof that
the hypothesis is already solved.

## Recommended Reading Order

- [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)
- [examples/README.md](examples/README.md)
- [docs/MCP_USAGE.md](docs/MCP_USAGE.md)
- [docs/PUBLICATION_READY.md](docs/PUBLICATION_READY.md)
- [docs/CURRENT_BENCHMARK_SNAPSHOT.md](docs/CURRENT_BENCHMARK_SNAPSHOT.md)
- [docs/TRAINING_PROTOTYPE.md](docs/TRAINING_PROTOTYPE.md)

## Reproducibility Notes

The historical broad benchmark harness is not part of this PoC artifact. The
figures in the benchmark snapshot are retained as research context, while the
publicly runnable checks are:

- `python -m refmark.cli smoke`
- `python -m refmark_train.smoke`
- `python examples/citation_qa/run_eval.py`
- `python examples/data_smells/run.py`
- `python examples/judge_free_rewards/run.py`
- `pytest`

The training prototype includes derived datasets and run summaries. Raw and
normalized source documents are intentionally not redistributed; use
`refmark_train/pull_source_docs.py` and the source manifests if you need to
rebuild that corpus from canonical upstream URLs.

## Positioning

Refmark should currently be presented as:

- strong on deterministic locate-only citation evaluation and HiL review
- practical for bounded same-file anchored edits
- promising but still experimental for broader coding-agent claims
- exploratory on trainable corpus-local anchor prediction
