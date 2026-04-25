# Release Scope

This document records what the current proof-of-concept release is meant to
stand behind. It is not a product roadmap and it is not a claim that Refmark
solves grounded generation or coding-agent reliability in general.

## Supported Today

Refmark is strongest as an addressability layer for corpora. The stable public
surface supports:

- deterministic citation-location scoring over returned refs
- highlighted review of cited source regions
- data-smell metrics such as wrong-location, overcite, undercite, breadth, and
  overlap
- same-file multi-region edits for Python and TypeScript through
  `apply_ref_diff`
- retained corpus-local training artifacts for small anchor-localization
  experiments

The important guarantee is narrow and useful: citation targets are real
resolvable corpus regions. Refmark does not guarantee that the selected region
is semantically correct.

## Evidence Level

The strongest evidence in this release is for deterministic locate-only
evaluation and human-auditable review. The included examples and tests are
small, runnable checks for that surface.

The coding-agent evidence is narrower. Refmark is most plausible for bounded
same-file edits where the relevant regions are already identified. It should
not be read as a broad SWE-bench or general coding-agent superiority claim.

The training evidence is exploratory. `refmark_train` shows that small models
can be intentionally overfit to a fixed addressable corpus and evaluated
cheaply. It does not yet prove cross-domain transfer or general small-model
anchor competence.

## Current Boundaries

This release intentionally excludes:

- broad benchmark runner infrastructure
- large historical result dumps
- raw redistributed source-document payloads
- claims of universal token savings
- claims that exact-minimal citation is solved
- claims that injected anchors already help all model sizes

The public artifact should be read as a focused proof of concept: make the
corpus addressable, make model references resolvable, and make citation
behavior measurable enough to audit, score, and diagnose.
