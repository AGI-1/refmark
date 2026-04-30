# Refmark Agent Workflow

Refmark's evaluation loop should not depend on a human saying "Codex, inspect
this heatmap and fix the weak parts." That manual loop is useful dogfooding, but
the product shape should be a set of Refmark agents that operate over the same
stable evidence address space.

The important contract is that every agent reads and writes artifacts with
stable refs/ranges, source hashes, config hashes, and model/provider provenance.
Agents may propose changes, score changes, or apply safe metadata updates, but
they should not silently mutate corpus truth.

## Agent Roles

| Agent | Input | Output | Human review point |
| --- | --- | --- | --- |
| Corpus mapper | Source files or extracted text | `corpus.refmark.jsonl`, sections, source hashes | Region granularity, excluded source families |
| Discovery agent | Corpus manifest | `discovery.json`, clusters, roles, query families, terms, abbreviations | Broad terms, noisy clusters, exclusions |
| Discovery reviewer | Discovery + manifest | `discovery_review.json` issue queue | Accept/split/merge/exclude guidance |
| Question planner | Manifest + discovery | `question_plan.json` with direct/concern/adversarial/range targets | Style mix, target coverage, cost budget |
| Question generator | Question plan + context cards | `eval_questions.jsonl` with model/cache provenance | Generated question quality and ambiguity |
| Retrieval benchmarker | Index + eval rows + retrieval config | `eval.json`, hit/mrr/coverage/style metrics | Baseline selection and regression gates |
| Heatmap diagnostician | Eval + index + discovery | Heatmap data, hard refs, confusions, query magnets, data smells | Which weak zones deserve adaptation |
| Adaptation reviewer | Weak rows + gold text + competing hits + metadata | Question rewrites, alternate refs, range edits, shadow metadata suggestions | Approve corpus-boundary and label changes |
| Mini-eval runner | Candidate adaptation + affected refs | Before/after affected-row score and blast-radius report | Promote/reject adaptation |
| Feedback agent | Production query/click/manual-selection events | Feedback clusters, confusion pairs, alias/query-magnet/no-answer candidates | Approve telemetry-derived adaptations |
| Report agent | All artifacts | Human-facing summary, claims, open risks | Publication and roadmap decisions |

This is deliberately more conservative than a single autonomous "fix the
corpus" agent. Refmark should make agent work measurable and reversible.

## Artifact Discipline

Each agent should preserve enough evidence for another agent, CI job, or human
reviewer to reproduce the decision:

- stable refs/ranges and source hashes;
- retrieval config, embedding model, and prompt/model versions;
- top competing hits, scores, and margins for weak rows;
- shadow metadata/aliases already active for the gold ref and competing refs;
- whether the row was training, held-out, curated, generated, or adapted;
- whether the agent saw gold evidence, competing evidence, or only blind query
  text.

That last point matters because some agents diagnose failures and therefore may
need gold refs, while deployed retrieval must not depend on gold refs.

## Adaptation Boundaries

Safe-by-default adaptations:

- rewrite or replace a clearly bad generated question;
- add Doc2Query-style shadow aliases to a region, with held-out checks;
- mark generated rows as ambiguous or invalid when the evidence proves it;
- record confusion pairs and hard negatives;
- add review-only data-smell flags.

Review-required adaptations:

- merge or split regions;
- change gold refs/ranges;
- add alternate valid refs;
- exclude source areas from default search/training;
- rename clusters or introduce parent/section hierarchy.

The intended loop is:

```text
evaluate -> diagnose weak refs -> propose adaptation -> mini-eval affected rows
         -> check held-out/blast-radius rows -> record or apply -> full rerun
```

## Why This Belongs To Refmark

These agents are not separate magic. They are consumers of the same primitive:
addressable evidence regions. Refmark gives every agent a common coordinate
system, so discovery, question generation, search, citation scoring, heatmaps,
metadata adaptation, corpus drift checks, and training can all refer to the same
refs/ranges.

In product terms: Refmark agents help maintain a corpus-as-test-suite. The user
should be able to run one piece or the whole chain depending on where their
system already exists.
