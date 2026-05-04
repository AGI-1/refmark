# Ephemeral Mode

Ephemeral mode is the disposable version of Refmark.

Use it when the task is one-off:

- proofread this document;
- suggest edits to this contract;
- clean up this Markdown file;
- patch selected paragraphs;
- highlight citations in a single report;
- compare two drafts.

The source does not need a durable registry. Refmark builds a temporary marked
view, lets a model talk about concrete refs, applies bounded edits where safe,
and then the map can be discarded.

```text
temporary address map > blind full-document rewrite
```

## CLI

Create a temporary marked view:

```bash
python -m refmark.cli ephemeral-map contract.docx \
  --instructions \
  -o contract.refmarked.txt
```

Ask a model to return edits like:

```json
{
  "edits": [
    {
      "ref": "P03",
      "action": "replace",
      "new_text": "Payment is due within 45 days after invoice receipt."
    }
  ]
}
```

Apply the edits:

```bash
python -m refmark.cli ephemeral-apply contract.docx \
  --edits-file edits.json \
  -o contract.patched.docx
```

Inline JSON is also accepted:

```bash
python -m refmark.cli ephemeral-apply contract.md \
  --edits-json "{\"edits\":[{\"ref\":\"P03\",\"action\":\"replace\",\"new_text\":\"Payment is due within 45 days.\"}]}" \
  -o contract.patched.md
```

Use `--dry-run` to validate refs, payload shape, and unique source matches
without writing the patched output.

## Current Safety Boundaries

- Text, Markdown, RST, and similar files are patched by exact unique text
  replacement.
- DOCX supports exact single-paragraph replacement. This is intentionally
  conservative and lightweight; it does not attempt rich layout editing.
- PDF support is extracted-text oriented. `ephemeral-apply` writes patched text,
  not an edited PDF.
- If a ref's source text appears multiple times, the edit is rejected rather
  than guessing.
- The map is disposable. For long-lived corpora, use a manifest or shadow
  registry instead.

## Disposable vs Durable

Ephemeral mode is for one-off work. It gives a model temporary addresses and a
bounded apply step, then the map can be thrown away. That is enough for tasks
like proofreading a contract, patching a few paragraphs, or reviewing a report.

Use durable Refmark mode instead when any of these are true:

- the same document will be reviewed repeatedly;
- eval labels, citations, or training examples must survive revisions;
- reviewers need an audit trail of why an address moved or became stale;
- the output affects a production corpus or compliance workflow.

For important one-off work, keep the optional manifest and dry-run report as an
audit artifact. They do not turn the workflow into lifecycle mode, but they make
the model's requested edits and Refmark's resolution decisions inspectable.

## Why This Matters

Durable Refmark mode is about lifecycle: stable refs, revision diffs, stale
examples, retrieval evaluation, and human review queues.

Ephemeral mode is about immediate operational risk reduction. It gives a model
small, concrete edit targets without forcing the user to persist a registry or
adopt a full corpus pipeline.

This mode is especially useful as a middleware layer for existing systems:

```text
input document -> temporary refs -> model edits refs -> bounded apply -> output
```

## Relation To Lifecycle Anchoring

The lifecycle benchmark showed that quote selectors are a strong competitor for
automatic preservation. Refmark can borrow those ideas:

- content hash;
- exact quote selector;
- neighboring-context hash;
- section-heading lineage;
- explicit ambiguity thresholds.

In ephemeral mode, these checks are local guardrails. In durable mode, they
become part of revision-aware migration and review classification.
