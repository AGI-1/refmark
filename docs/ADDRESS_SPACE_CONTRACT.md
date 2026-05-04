# Address Space Contract

Refmark is an address-space layer for evidence-bearing text. It gives documents,
code, and corpora stable refs/ranges that other systems can cite, retrieve,
score, review, edit, or mark stale.

The useful mental model is:

```text
source corpus -> address space -> evidence obligations -> lifecycle states -> consumers
```

## Layers

### Source Corpus

The original files or extracted text being addressed. The source may be Markdown,
plain text, code, extracted DOCX/PDF text, or a corpus assembled from multiple
documents.

### Address Space

A logical registry or manifest that maps refs to source regions:

```text
policy:P13 -> docs/policy.md, ordinal 13, text span, content hash, metadata
```

Refs can be visible markers in a marked copy, or shadow refs stored externally
without mutating the source. They are logical identifiers, not memory pointers
or fixed byte offsets.

### Addressable Unit

A **region** is the source span addressed by one ref. A **range** is an ordered
set of regions, usually inclusive for citation semantics:

```text
policy:P13-P15
```

Region size is a corpus policy, not a universal truth. A region might be a
paragraph, section, function, article, or extracted text block.

### Anchor Bundle And Resolver

A robust implementation can attach multiple selectors to the same ref:

- structural path;
- ordinal;
- source/content hash;
- quote selector;
- surrounding context;
- fuzzy or semantic match candidates.

These selectors help reattach evidence after corpus edits. They are primitives
Refmark can use; they are not the whole product.

### Evidence Obligation

An evidence obligation is any maintained claim that says a task depends on a
source region:

- `query -> gold_refs`;
- model answer citation refs;
- training labels;
- review notes;
- hard negatives;
- bounded edit targets.

The obligation matters because it can become stale when the corpus changes.

### Lifecycle State

When a corpus changes, a ref or evidence obligation should be classified rather
than silently reused:

```text
unchanged / moved / rewritten / ambiguous / split / deleted / stale
```

The current public implementation mostly exposes preserved, review-needed, and
stale states, with richer lifecycle states on the roadmap.

Lifecycle state depends on the available selector and resolver evidence. A safe
implementation should expose confidence and review requirements instead of
pretending every reattachment is deterministic.

### Consumers

Once the address-space contract exists, many systems can share it:

- RAG retrieval evaluation;
- citation scoring and highlighting;
- context expansion and packing;
- data-smell reports;
- corpus CI;
- human review queues;
- bounded document/code edits;
- training and corpus-local navigation experiments.

## Prior-Art Positioning

Refmark is related to several established ideas:

- **qrels and passage-level IR judgments**: query relevance labels over document
  or passage ids;
- **QA evidence spans**: answer-support annotations in benchmark datasets;
- **Web Annotation selectors**: text position, quote, and selector models for
  anchoring annotations;
- **content hashes and source hashes**: deterministic invalidation when evidence
  changes;
- **robust anchoring / quote selectors**: reattaching annotations after document
  edits;
- **RAG lifecycle tools**: evaluation and observability systems that can carry
  reference ids.

Refmark does not replace these. It packages their signals into a versioned
evidence address space that is meant to be owned by the corpus and reused across
tools.

The core distinction:

```text
Selectors answer: can I find this text again?
Refmark asks: what lifecycle state is this evidence obligation in now?
```

That is why a quote-selector baseline can be a strong competitor for automatic
reattachment, while Refmark still adds value through explicit lifecycle state,
review packets, corpus fingerprints, and shared refs for multiple consumers.

## Claim Discipline

Safe wording:

> Refmark makes evidence refs resolvable, scoreable, and lifecycle-checkable.

Avoid:

> Refmark guarantees that a model found the right evidence.

Safe wording:

> Refmark can use quote selectors, hashes, and fuzzy matching to migrate refs.

Avoid:

> Refmark invented robust anchoring.

Safe wording:

> In the current benchmark tables, the Refmark layered selector had 0 observed
> silent-drift labels under the current migration oracle and corpus
> fingerprinting policy.

Avoid:

> Refmark proves zero semantic drift universally.
