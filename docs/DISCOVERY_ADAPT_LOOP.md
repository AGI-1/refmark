# Discovery And Adapt Loop

Refmark's evidence-retrieval loop should evolve from one-shot benchmarking into
an iterative corpus improvement cycle:

```text
discover -> generate questions -> train/evaluate -> heatmap -> adapt -> regenerate/train
```

The point of the heatmap is to make corpus and training failures actionable at
the ref/range level. A weak area should point to one or more concrete
adaptations, not just a lower aggregate score.

This document describes the mechanics. The intended product shape is a set of
Refmark agents that run these steps, preserve artifacts, and ask for review at
the right boundaries. See `docs/REFMARK_AGENT_WORKFLOW.md`.

## Discovery Stage

Discovery prepares the context that question generation and training need before
the first evaluation run.

Output artifact:

```json
{
  "schema": "refmark.discovery.v1",
  "corpus_summary": "...",
  "terms": [{"term": "SDS", "refs": ["manual:P12"]}],
  "abbreviations": [{"term": "GHS", "refs": ["manual:P20"]}],
  "region_roles": [
    {"stable_ref": "manual:P01", "roles": ["navigation_only", "exclude_from_training"]},
    {"stable_ref": "manual:P42", "roles": ["definition"], "terms": ["exposure"]}
  ],
  "range_candidates": [
    {"refs": ["manual:P42", "manual:P43"], "kind": "adjacent", "reason": "..."}
  ],
  "query_families": [
    {"name": "questions about exposure controls", "refs": ["manual:P42"], "terms": ["exposure"]}
  ]
}
```

Current implementation:

```bash
python -m refmark.cli discover corpus.refmark.jsonl \
  -o corpus.discovery.json \
  --source local
```

Optional model-backed whole-corpus discovery:

```bash
python -m refmark.cli discover corpus.refmark.jsonl \
  -o corpus.discovery.json \
  --source openrouter \
  --model deepseek/deepseek-v4-flash \
  --max-input-tokens 180000
```

The local source is deterministic and good for smoke tests. Model-backed
discovery should be used for real section summaries, glossary quality, and
region-role review.

## Context Modes

### 1. Whole-Corpus Discovery

Use this when the corpus fits in the model context window. For corpora below
roughly 200k tokens, this is the simplest and usually best mode.

The model sees:

- the corpus summary request;
- every refmarked region;
- instructions to identify terms, abbreviations, region roles, exclusions,
  likely query families, range candidates, and distributed-support candidates.

This mode is best for the next implementation iteration because it minimizes
summary drift.

### 2. Hierarchical Discovery

Use this when the corpus does not fit in context.

Planned shape:

```text
1. Map corpus into refs.
2. Split into region-safe overlapping section windows.
3. Discover each window with:
   - local refs;
   - prior/global summary;
   - glossary so far;
   - known exclusions;
   - unresolved conflicts.
4. Merge window discoveries while preserving every source ref.
5. Normalize terms/clusters globally.
6. Produce a discovery review queue for noisy roles, broad terms, query magnets,
   large ranges, stale refs, and heading-boundary failures.
7. Revisit weak/conflicting windows after heatmap evaluation.
```

The critical rule: summaries must remain structured and cited. Do not rely on a
free-form "summary so far" alone. Every discovered term, exclusion, range
candidate, and query family should carry refs/ranges.

Model-review notes from the current design pass converged on three guardrails:

- never cut a model prompt through the middle of a refmarked region;
- keep canonical clusters as metadata over refs, not replacement prose;
- route low-confidence discoveries to review instead of letting noisy terms
  silently steer question generation.

Current implementation now exposes deterministic review primitives:

```bash
python -m refmark.cli review-discovery corpus.discovery.json \
  --manifest corpus.refmark.jsonl \
  -o corpus.discovery_review.json
```

The review output is a HiL/LLM-judge queue. It does not mutate discovery by
itself. Typical issue kinds are `broad_term`, `singleton_term`,
`excluded_region`, `heading_detection`, `large_range_candidate`,
`broad_query_family`, `stale_or_unknown_ref`, `broad_cluster`,
`unclustered_regions`, `empty_window`, and `oversized_source_region`.

Windowed discovery is now available:

```bash
python -m refmark.cli discover corpus.refmark.jsonl \
  -o corpus.discovery.json \
  --mode windowed \
  --window-tokens 40000 \
  --overlap-regions 2
```

Windowing is region-safe: a region is either included whole in a window or moved
to the next one. Per-window discoveries are merged conservatively by stable ref.
The merged manifest includes:

- `windows`: token counts, refs, and doc ids per window;
- merged terms, abbreviations, roles, ranges, and query families;
- deterministic `clusters` seeded from document ids and discovery terms.

The first cluster layer is intentionally simple. It is a navigation/review seed,
not a final ontology. Later LLM normalization can rename, split, and merge
clusters while preserving the same ref lists.

## Discovery Context Cards

Question generation should not receive the entire discovery manifest. It should
receive a compact context card for the target ref/range:

```json
{
  "stable_ref": "docs:P42",
  "corpus_summary": "...",
  "region_summary": "...",
  "roles": ["definition"],
  "terms": ["exposure", "controls"],
  "abbreviations": ["SDS"],
  "query_families": ["questions about exposure"],
  "range_candidates": [["docs:P42", "docs:P43"]],
  "neighboring_refs": ["docs:P41", "docs:P43"],
  "parent_ref": "docs:P40",
  "generation_guidance": ["include at least one definition-style query"]
}
```

CLI:

```bash
python -m refmark.cli discovery-card corpus.refmark.jsonl corpus.discovery.json \
  --ref docs:P42
```

The full pipeline now stores this context card in the question cache and includes
it in remote question-generation prompts. The cache key includes a hash of the
card, so question rows are regenerated when discovery changes.

## Question Generation

Question generation should consume:

- target refs/ranges;
- local target text;
- neighbor refs;
- parent/section summary;
- corpus summary;
- special terms and abbreviations;
- known hard negatives/magnet refs from prior heatmaps;
- generation mode: `single`, `range`, `distributed`, or `parent`.

Before generation, the pipeline writes an explicit question plan:

```bash
python -m refmark.cli question-plan corpus.refmark.jsonl corpus.discovery.json \
  -o corpus.question_plan.json
```

Default styles:

- `direct`: normal lookup wording with source terminology;
- `concern`: user problem/goal/symptom wording;
- `adversarial`: valid paraphrase with lower lexical overlap.

The plan is a reviewable artifact, not a hidden prompt detail. Generated rows
carry `metadata.query_style`, and eval diagnostics summarize metrics by style.
That is important because a green direct-query heatmap can still hide weak
concern/adversarial retrieval.

Current local question generation can use discovery terms, but it is too
lexical. It is useful for smoke tests and fast heatmaps, not for public quality
claims. For serious evaluation, use cached LLM-generated questions and keep the
prompt version, model, provider, target refs, and source hashes in the cache key.

Current implementation state:

- local generation uses context-card terms, roles, and query families to vary
  templates;
- remote generation receives the compact context card plus the target region
  text;
- generated eval rows keep source hashes and prompt/discovery metadata;
- the next major step is windowed LLM discovery with global normalization and
  reviewed cluster naming.

## Motivation Understanding

Discovery should also capture motivation-level aliases: user intent phrases that
do not appear in source text but reliably map to evidence refs/ranges.

Example:

```json
{
  "query": "Ich habe ein Handy gekauft und es ist kaputt. Was kann ich tun?",
  "gold_refs": ["bgb:S_434", "bgb:S_437", "bgb:S_439", "bgb:S_474"],
  "aliases": ["kaputtes Handy", "defektes Smartphone", "repair replacement refund"]
}
```

This is a general adaptation type, not a BGB-specific trick:

- support tickets map layperson symptoms to troubleshooting articles;
- legal/reference corpora map concerns to provisions;
- product docs map task intent to API/tutorial pages;
- internal knowledge bases map organizational jargon to canonical pages.

The important rule is that aliases are retrieval metadata, not source evidence.
They can improve navigation and training examples, but citations and highlighted
answers must still resolve to real refs/ranges.

## Heatmap Signals

Heatmaps should split results by target type:

- `single`: one ref should be sufficient;
- `adjacent` / `range`: contiguous support should be recovered or covered by
  context expansion;
- `distributed`: non-neighbor support requires multi-hop collection;
- `parent`: coarse hit inside a larger section is acceptable.

High-value signals:

- repeated wrong top refs;
- missed gold refs;
- undercitation by mode;
- overcitation/breadth;
- low precision at high coverage;
- hard negatives that frequently beat gold refs;
- regions with weak training examples;
- excluded/navigation/boilerplate regions entering training.

## Adaptation Actions

Use the heatmap to choose one or more actions:

- change weak/generated questions for specific refs;
- add questions to uncovered refs/ranges;
- merge adjacent refs that behave as one semantic unit;
- split regions that attract unrelated queries;
- convert a single-ref target into a range target when neighbors are required;
- add additional valid refs for distributed support;
- add motivation aliases or concern-query families for refs that are correct
  but lexically unreachable from real user wording;
- mark navigation, table-of-contents, summary, boilerplate, or language-menu
  areas as `exclude_from_training`;
- mine hard negatives from repeated wrong top refs;
- add a breadth penalty when high top-k coverage comes only from too much
  context.

## Current OSHA Loop Observation

The first discovery-aware local run produced useful but biased evidence:

- local-view BM25 reached `hit@1 = 0.808` and `hit@10 = 1.000`;
- single and adjacent targets became much easier;
- disjoint targets still had `context_hit@10 = 0.000`;
- top-10 coverage was high, but precision stayed around `0.076`;
- the simple learned reranker collapsed badly, showing it needs hard negatives
  and a safer blend/fallback.

This is a good heatmap result, not a final claim. It says:

- discovery terms can remove many generic-query failures;
- local/discovery questions are too lexical and must be replaced or audited by
  LLM-generated natural questions for claim-grade eval;
- distributed support needs a collector/multi-hop objective rather than a
  single-region reranker;
- overcitation must be scored separately from recall.

## Current BGB Motivation Loop Observation

The BGB article-navigation smoke test aggregates about 4,988 BGB regions into
2,526 article-level refs. With `concern_aliases_extended.json`, it now checks
104 expected concern rows plus five ambiguous/no-answer probes. Held-out query
text is not injected into the index; only separate alias phrases are retrieval
metadata.

| Index | expected hit@1 | expected hit@3 | expected hit@5 |
| --- | ---: | ---: | ---: |
| raw article index | 0.7885 | 0.9135 | 0.9327 |
| article index + concern aliases | 0.8173 | 0.9231 | 0.9327 |

The key failure was the everyday broken-cellphone query. Raw lexical search
followed the word "found" into finder/lost-property law. The aliased article
index recovered the buyer-defect neighborhood at rank 1.

The full BGB pipeline now treats those aliases as part of the same loop:

- concern queries are appended as `gold_mode=concern` examples;
- expected article refs expand to concrete region refs when needed;
- concern alias text is injected into enriched retrieval views only, while
  held-out query text remains evaluation-only;
- raw BM25 remains unmodified, so the delta stays visible.

The latest full BGB concern run is more useful than the earlier tiny smoke:

| Method | concern rows | concern hit@1 | concern hit@10 | concern misses |
| --- | ---: | ---: | ---: | ---: |
| raw BM25 | 99 | 0.1717 | 0.2828 | 71 |
| Refmark BM25 + generated views + concern aliases | 99 | 0.8182 | 0.9495 | 5 |
| Refmark rerank | 99 | 0.8081 | 0.9596 | 4 |

The remaining misses are exactly what the adapt loop needs: concrete weak
neighborhoods around buyer defects, limitation periods, pledge expiration,
partnership exit, and finder duties.

The next pressure layer is randomized stress generation. Instead of reusing a
curated concern file, `run_bgb_stress_eval.py` samples article blocks and asks
several generator models to create direct, concern, and adversarial questions.
The generated question does not define the gold target; the sampled block does.

Current randomized stress signals:

| Run | Rows | Method | article hit@1 | article hit@10 |
| --- | ---: | --- | ---: | ---: |
| mixed direct/concern/adversarial | 953 | raw BM25 | 0.1427 | 0.2445 |
| mixed direct/concern/adversarial | 953 | Refmark BM25 | 0.3578 | 0.5824 |
| concern-heavy | 873 | raw BM25 | 0.1924 | 0.3116 |
| concern-heavy | 873 | Refmark BM25 | 0.4719 | 0.7320 |

These stress numbers should drive adaptation more than the curated smoke. Weak
blocks should get one of several treatments: better generated retrieval views,
targeted concern aliases, hard negatives against repeated wrong-top articles,
or a question-quality judge when the generator wrote a query whose best answer
is actually a neighboring article.

This supports the next loop:

```text
user concern misses -> heatmap shows wrong magnet refs -> add/learn aliases ->
rerun eval -> accept only if gold refs improve without hiding ambiguity
```
