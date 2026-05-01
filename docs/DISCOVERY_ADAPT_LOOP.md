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

## Discovery Cluster Strategies

Clusters are metadata over refs, not replacements for refs. A ref remains the
atomic evidence address; clusters are overview/evaluation units used by editor
boards, explorer maps, and agent navigation. Heatmaps, confusion pairs, query
magnets, and stale refs are metrics over a cluster layer, not clustering
strategies by themselves.

Tags are supporting evidence for clusters, not the cluster ontology. Generic
term buckets are weak navigation. The useful direction is typed metadata such as
entity/action/property tags, audience/task intents, and source structure. Those
signals can explain or repair a cluster, but they should not force the primary
map to be a bag of nouns.

Current deterministic strategies:

- `doc_id`: group refs by source document id. This is the best default when a
  corpus has meaningful files, pages, or TOC structure.
- `source_tree`: build a parent/child hierarchy from source paths or structured
  document ids. This is the best first map for documentation sites, wikis with
  path-like article ids, and corpora where navigation should mirror how humans
  browse the source.
- `tag_graph`: extract normalized high-signal tags per region, assign each ref
  to a dominant shared tag, and fold small buckets into `other topics`. This is
  useful for flat wiki-style corpora because the output is explainable and
  human-reviewable.
- `balanced_terms`: build local term vectors from rare-but-repeated terms, pick
  diverse seeds, and assign refs under a soft capacity. This is a no-embedding
  proxy for balanced semantic partitions, mainly for dashboard/explorer views
  when source structure creates too many tiny blocks. It is sensitive to noisy
  source artifacts and should prefer reviewed terminology from discovery when
  available.

LLM-backed strategies:

- `llm_topics`: ask the discovery model to propose normalized topic/editor
  clusters. This is intended for flat wiki collections, legal corpora, and
  heterogeneous documentation where file structure is too shallow or too noisy
  for review boards.
- `llm_intents`: ask the discovery model to cluster by user task/question
  intent. This is useful when the user-facing navigation question is "what
  should I search for?" rather than "which source folder is this in?"

LLM clusters are sanitized against the manifest: invalid refs are ignored,
duplicate primary assignments are removed, and unassigned refs are folded into
deterministic backfill clusters. That keeps the cluster layer reviewable without
letting the model invent addresses. LLM clustering uses compact per-region cards
instead of full raw text so broad overview passes can see more refs within the
same context budget.

Example:

```bash
python -m refmark.cli discover corpus.refmark.jsonl \
  -o corpus.discovery.json \
  --cluster-strategy source_tree \
  --target-clusters 40

python -m refmark.cli discover corpus.refmark.jsonl \
  -o corpus.discovery.llm_topics.json \
  --source openrouter \
  --model qwen/qwen-turbo \
  --cluster-strategy llm_topics \
  --target-clusters 40

python -m refmark.cli discovery-map corpus.refmark.jsonl corpus.discovery.json \
  -o corpus.discovery_map.html \
  --title "Corpus Discovery Map"

python -m refmark.cli repair-discovery-clusters corpus.refmark.jsonl \
  corpus.discovery.llm_topics.json \
  -o corpus.discovery.llm_topics.repaired.json \
  --model qwen/qwen-turbo \
  --cluster-strategy llm_topics \
  --target-clusters 40
```

`discovery-map` renders a drill-down view. The simplest case is
cluster -> block, but cluster manifests may also use `parent_id` to form
hierarchies such as corpus section -> review window -> refmarked block. This is
important for corpora with thousands of regions: a flat 40-cluster map can hide
too much structure, while a 5,000-block map is not reviewable. The right panel is
for the selected cluster/block; global run metadata should stay secondary so the
map remains usable as an explorer view.

The map defaults to an order-preserving layout. This matters for law codes,
manuals, and other sequential corpora where left-to-right/top-to-bottom should
roughly follow source order. A balanced layout remains useful for dashboard
inspection, but it may rearrange blocks by size and obscure source sequence.

For flat corpora, a practical workflow is:

```text
region refs
 -> local/LLM summaries and terms
 -> tag_graph or balanced_terms cluster manifest
 -> editor heatmap and explorer overview
 -> evaluate cluster@k, region@k, query magnets, and weak zones
 -> HiL/LLM review renames/splits/merges clusters
```

Future model-backed strategies should follow the same contract: emit a
reviewable `clusters` list with `cluster_id`, label, refs, terms, source, and
strategy. The system should then score whether the clusters are useful by
cluster-level retrieval, label coherence, query-magnet concentration, and
cross-cluster confusion.

Early feel-test guidance:

- `tag_graph` is best for explaining "why are these refs together?", but it
  needs corpus/language normalization and usually leaves an `other topics`
  bucket for HiL/LLM review.
- `source_tree` should be tried before tag/term clustering whenever source
  paths carry real navigation structure. It can then be repaired with LLM topic
  names instead of forcing tags to invent the ontology.
- `balanced_terms` is best for a first overview board because cluster sizes are
  stable and readable, but labels are weaker and should be reviewed or renamed.
  For technical corpora, it works best after filtering common language words,
  normalizing variants, and using rare domain terms rather than raw frequency.
- For flat wiki-style collections, render both. Use `balanced_terms` for the
  map shape and `tag_graph` for candidate labels, tags, and merge/split hints.
- For real product discovery, run at least one LLM-backed strategy as a review
  candidate. The deterministic maps are valuable baselines; the LLM pass is
  where normalized labels, intent clusters, and cross-document topic names can
  be tested against the same ref-level evaluation harness.
- High deterministic backfill after an LLM pass is itself a data smell: either
  the prompt/model did not understand the corpus shape, the requested strategy
  is too abstract for the source, or the sample is too sparse for coherent
  cluster labels. BGB-style legal corpora exposed this more clearly than
  technical wiki documentation.
- Broad labels with glue words such as "and" are usually a smell in overview
  clusters. They often mean the cluster is compensating for missing hierarchy.
  Prefer a parent cluster with narrower child clusters, or split the label into
  explicit alternatives if the refs do not share one coherent concept.
- For BGB-scale corpora, a deterministic hierarchy is already useful before any
  LLM naming pass: top-level legal books or source collections, second-level
  bounded section windows, and leaf-level refs. This keeps the full corpus
  navigable while the agentic discovery loop works on better names and topic
  normalization.
- For flat corpora with weak or missing hierarchy, expect deterministic
  strategies to expose normalization issues first. Source-prefix tokens,
  boilerplate, and broad buckets such as `other topics` are not just ugly map
  output; they are discovery review items. The next pass should either remove
  those terms from the candidate vocabulary or ask a discovery agent to produce
  normalized cluster labels over the same refs.
- Current flat-corpus smoke tests show a useful split in behavior:
  `tag_graph` is the better data-smell view because large `other topics`
  buckets expose missing normalization; `balanced_terms` is the better shape
  baseline because it creates evenly sized boards, but labels can become
  rare-word soup on article-level documentation. Treat balanced labels as
  provisional unless a review/LLM pass confirms them.
- Repair agents need bounded, validator-friendly output. Asking a model to
  rewrite hundreds of refs into full JSON can fail through malformed or
  truncated JSON even when the reasoning is plausible. For larger corpora,
  prefer hierarchical repair or cluster-by-cluster repair instead of one broad
  all-refs rewrite.

## Discovery Map Reviewability

The map is not just a visualization; it is part of the discovery-agent review
loop. A usable map should let a human or model answer:

- what does this broad cluster cover?
- which child topics or sample blocks explain the label?
- is this an `other topics` bucket hiding missing normalization?
- should this be split, merged, renamed, excluded, or expanded with metadata?

The renderer supports nested `parent_id` chains, so it is not inherently capped
at two levels. The current limitation is upstream: most deterministic and repair
strategies still produce one flat layer, and some demos only add one child
level. For larger or flatter corpora, agents should be allowed to create deeper
structures such as:

```text
corpus overview -> topic family -> concrete subtopic -> refs
```

Broad clusters should carry useful summaries: a normalized title, tag badges,
sample topics, child cluster previews, and review notes. Huge `other topics`
clusters are never a good final state. They are review issues that should
trigger one of:

- stopword/noise normalization;
- source boilerplate exclusion;
- split by child topics;
- LLM review of only the broad cluster;
- creation of an intermediate parent layer.

Current repair agents do not yet get a full visual feedback loop. They receive
current clusters, deterministic review issues, and compact region cards, but
they do not inspect the rendered board or run a post-repair map-quality check.
That is the next agentic shape: propose hierarchy -> render/review metrics ->
repair only weak/broad nodes -> repeat.

When a map exposes poor high-level clusters but useful drill-down blocks, use
`repair-discovery-clusters` rather than editing clusters by hand. The command
acts as a discovery-agent tool: it sends current clusters, deterministic review
issues, and compact region cards to a model, then replaces only the cluster
layer after sanitizing refs against the manifest. This keeps the loop
reproducible:

```text
discover -> map exposes weak clusters -> repair cluster layer -> remap -> review
```

The repair sanitizer should preserve the agent's semantic intent, not merely
sort by size. BGB exposed a failure mode where target-count enforcement merged
useful one-ref legal clusters into `other reviewed topics` while keeping weak
backfill clusters. The current repair path first tries semantic coalescing
(`Contract Remedies and Obligations`, `Property Rights and Claims`, etc.) before
falling back to generic overflow. This is still a review aid, not a final legal
taxonomy.

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
