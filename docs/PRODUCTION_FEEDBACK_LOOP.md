# Production Feedback Loop

Refmark's generated eval suites are the starting point, not the end state. In a
real documentation or knowledge-base search experience, user queries should feed
back into the same evidence-addressed loop.

The goal is not to let telemetry silently rewrite the corpus. The goal is to
turn repeated user behavior into reviewable, testable adaptation candidates.

```text
user queries -> feedback batch -> diagnostics -> adaptation candidates
             -> shadow metadata / review queue -> held-out eval -> promote
```

## Feedback Event Shape

Feedback rows are JSONL. Minimal rows can contain only a query and shown refs;
richer rows can include clicks, manual selections, and usefulness signals.

```json
{"query":"cors browser error","top_refs":["docs:P01","docs:P03"],"clicked_ref":"docs:P02","useful":true}
{"query":"cors browser error","top_refs":["docs:P01"],"selected_ref":"docs:P02","useful":true}
{"query":"latest change","top_refs":["docs:P99"],"useful":false,"no_answer":true}
```

Supported fields:

| Field | Meaning |
| --- | --- |
| `query` | User query text. |
| `top_refs` / `shown_refs` / `refs` | Returned refs in rank order. |
| `clicked_ref` | Ref the user opened. |
| `selected_ref` / `manual_ref` / `correct_ref` | Ref a user or reviewer identified as the desired target. |
| `useful` | Boolean usefulness signal. |
| `feedback` | Optional string alias for usefulness, such as `positive` or `negative`. |
| `no_answer` | User indicated no result answered the query. |
| `metadata` | Product-specific fields such as user segment, locale, or UI surface. |

## CLI

```bash
python -m refmark.cli feedback-diagnostics feedback.jsonl \
  --manifest corpus.refmark.jsonl \
  --min-count 2 \
  -o feedback_report.json
```

The output groups repeated normalized queries and emits reviewable actions:

- `add_shadow_alias_or_doc2query`: users repeatedly select/click a ref that was
  not the dominant top result.
- `record_confusion_pair`: the query repeatedly lands on a competing ref.
- `review_query_magnet`: a ref is repeatedly top-ranked but receives negative
  feedback and no better target.
- `review_no_answer_or_missing_coverage`: repeated no-answer signals.
- `review_ambiguous_query`: different users select different target refs.
- `review_missing_refs`: feedback points to refs absent from the current map.

These actions are inputs to the adaptation reviewer. They are not automatic
corpus changes.

## Why It Matters

Generated questions let a team bootstrap a measurable retrieval suite before
users arrive. Production feedback turns the same suite into a living corpus
maintenance system:

- real user language becomes Doc2Query/shadow metadata candidates;
- repeated wrong-top refs become hard negatives or confusion pairs;
- noisy hubs and release-note/query-magnet pages become visible;
- missing coverage becomes explicit;
- stale or deleted refs are caught against the current manifest.

## Promotion Rule

Treat feedback-derived changes like any other adaptation:

1. Build a candidate shadow metadata patch or review item.
2. Run affected-row mini-eval.
3. Run held-out/blast-radius queries that the adaptation did not see.
4. Promote only if held-out discovery improves or stays stable.

This keeps the system from merely making a board greener. The product claim is
strong only when unseen user-style queries discover the intended refs or
articles more reliably.
