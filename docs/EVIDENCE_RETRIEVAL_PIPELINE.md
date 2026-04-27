# Evidence Retrieval Pipeline

Refmark turns a corpus into a testable evidence space. The pipeline is:

```text
structured corpus -> regions -> stable refs/ranges -> questions -> retrieval runs -> evidence metrics
```

The important question is not only whether a final generated answer sounds
right. The lower-level CI question is:

> Did this retrieval configuration recover the source region or range required
> to answer the query?

Because the target is a ref or range, the same suite can compare BM25,
generated retrieval views, embeddings, rerankers, context expansion, and a
corpus-specific trained resolver.

The next loop is documented in `docs/DISCOVERY_ADAPT_LOOP.md`:

```text
discover -> generate questions -> train/evaluate -> heatmap -> adapt -> regenerate/train
```

## Corpus Shape

Use a single structured domain first. The current retained local corpus for this
flow is an OSHA legal set packaged as generated Markdown:

- generated file: `examples/portable_search_index/output_legal_md/osha_legal_corpus.md`
- source family: OSHA legal/regulatory documents
- regex token count: about 215k
- Refmark evaluator token estimate: about 329k
- mapped regions: 1,134

That generated Markdown artifact is ignored by git. It exists only to prove the
flow against a large `.md` corpus without spending API credits.

The stronger remote-model stress corpus is the German BGB demo:

- 4,988 regions
- about 421k source tokens
- 800 bilingual Qwen Turbo generated eval questions
- 99 held-out curated concern questions in the latest concern run
- Qwen Turbo bilingual retrieval views
- Qwen3 Embedding 8B semantic retrieval
- tiny multilingual resolver artifact: 1.31M parameters, about 5.43 MB

## Region Targets

Keep target shapes explicit. Do not mix them into one score without reporting
the distribution.

| Target shape | Example gold refs | Meaning |
| --- | --- | --- |
| Single region | `policy:P042` | One region should be enough. |
| Contiguous range | `policy:P042-policy:P044` | The support is an inclusive ordered range. |
| Distributed refs | `policy:P042,policy:P087` | The support is split across non-neighbor regions. |

Single-region targets are the cleanest baseline. Contiguous ranges test context
expansion and undercitation. Distributed refs are expected to be hard for
ordinary retrieval because one top hit rarely carries both pieces of support.
They are still useful as a pressure test: Refmark can show partial coverage
instead of hiding the failure inside an answer score.

## Product Heatmap Metrics

The evidence heatmap should become a core Refmark product surface, not only an
experiment artifact. The point is to make a corpus operationally inspectable:
which refs are easy to recover, which refs attract false positives, which
queries are safe to auto-jump, and which corpus areas need regeneration or
human review.

Core heatmap dimensions to expose:

| Metric family | Examples | Why it matters |
| --- | --- | --- |
| Candidate recall | `recall@10/@50/@200/@500/@1000` | Separates candidate-generation failures from reranker failures. |
| Conditional rerank quality | `hit@10 if gold in pool@200` | Shows whether a reranker is weak or simply candidate-limited. |
| Selective jump | `coverage@precision=0.90/0.95`, wrong jumps, fallback hit@10 | Matches the UI choice between direct jump and candidate list. |
| Confidence diagnostics | top score, top1/top2 margin, entropy, score calibration buckets | Supports learned gates and abstention. |
| Hard refs | per-ref miss rate, mean rank when found, no-recall refs | Points the adapt loop at concrete regions. |
| Confusions | gold ref -> wrong top ref pairs, repeated attractor refs | Supports negative-aware signatures and hard negatives. |
| Split metrics | language, style, generator, source document, section, target shape | Catches overfitting and domain-specific failures. |
| Range quality | exact hit, parent hit, neighbor hit, range cover, overcite, undercite | Prevents top-k success from hiding citation-quality problems. |
| Drift/staleness | stale examples, changed refs, deleted refs, hash mismatch | Turns corpus updates into incremental test maintenance. |
| Provenance hygiene | source reports, split seed, train/eval membership, generation model, cache hash | Prevents adapted indexes from being evaluated against examples that influenced their metadata. |

The generic `eval-index` CLI now emits the first portable subset of this shape:

- self-checking provenance for the search index, eval JSONL, and retrieval
  settings;
- stale-ref validation against source hashes;
- hard-ref and confusion heatmaps;
- score-margin selective-jump diagnostics;
- first-pass adaptation recommendations derived from repeated misses and
  confusions.

That keeps BGB-specific adaptation experiments from being the only place where
these signals exist. A small docs corpus can now run the same evidence loop and
produce artifacts that can be checked in CI or compared between retrieval
variants.

The desired product report should answer:

```text
Can I jump directly?
If not, is the right evidence at least in the candidate list?
Which refs repeatedly fail, and what kind of adaptation would help?
Did a corpus update invalidate any examples or signatures?
Did this index use the same split/provenance as the benchmark I am reading?
```

This also gives a clean acceptance test for future retrieval variants. A method
that improves hit@10 while increasing wrong high-confidence jumps may be worse
for a browser-navigation UI; a method that preserves high-precision jumps and
improves fallback candidate recall is more useful.

## Reproduction Commands

Generate the ignored Markdown corpus from the retained legal set:

```bash
python - <<'PY'
from pathlib import Path

src = Path("refmark_train/source_docs/sets/legal_set.txt")
out_dir = Path("examples/portable_search_index/output_legal_md")
out_dir.mkdir(parents=True, exist_ok=True)
text = src.read_text(encoding="utf-8", errors="ignore")
out = out_dir / "osha_legal_corpus.md"
out.write_text(
    "# OSHA Legal Corpus\n\n"
    "Source set: retained OSHA legal/source documents used for Refmark training experiments.\n\n"
    + text.replace("\r\n", "\n"),
    encoding="utf-8",
)
print(out)
PY
```

Run a local mixed-target retrieval evaluation:

```bash
python examples/portable_search_index/evaluate_real_corpus.py \
  examples/portable_search_index/output_legal_md/osha_legal_corpus.md \
  --output examples/portable_search_index/output_legal_md/legal_mixed_local_eval.json \
  --cache examples/portable_search_index/output_legal_md/legal_mixed_question_cache.jsonl \
  --budgets 250000 \
  --sample-size 120 \
  --question-source local \
  --index-view-source local \
  --strategies flat,hierarchical,rerank,learned-rerank \
  --expand-after 1 \
  --gold-mode mixed \
  --learned-reranker-epochs 5 \
  --learned-reranker-train-fraction 0.5 \
  --top-ks 1,3,5,10
```

The local question generator is intentionally cheap and imperfect. It is good
for exercising the evaluation harness, range metrics, and failure modes. For a
claim-grade run, replace `--question-source local` and `--index-view-source
local` with cached LLM generation and report model/provider details.

## Local Mixed-Target Result

The local OSHA Markdown run used 120 mixed gold targets:

| Gold mode | Count |
| --- | ---: |
| single | 62 |
| adjacent range | 49 |
| disjoint refs | 9 |

Selected results on the 250k-token budget:

| Method | hit@1 | hit@10 | context hit@10 | MRR | range cover@10 | precision@10 | undercite@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw BM25 flat | 0.275 | 0.742 | 0.633 | 0.412 | 0.704 | 0.052 | 0.367 |
| local-view BM25 flat | 0.325 | 0.792 | 0.692 | 0.476 | 0.754 | 0.055 | 0.308 |
| local-view BM25 rerank | 0.308 | 0.800 | 0.700 | 0.457 | 0.758 | 0.056 | 0.300 |
| local-view learned rerank | 0.233 | 0.883 | 0.650 | 0.396 | 0.783 | 0.059 | 0.350 |

Interpretation:

- Local generated views improved BM25 across hit rate and range coverage.
- The one-document hierarchy was a no-op, which is expected for this packaging.
- The simple learned reranker improved top-10 recall but hurt rank-1 precision
  and MRR. This is useful evidence that training should be treated as a
  candidate-resolution layer with a held-out gate, not as a default win.
- Range metrics are more informative than hit rate alone. At top-10, coverage
  improves, but precision is low because expanded top-k context overcites.

This is the kind of result the CI harness should preserve: it captures both a
positive gain and a failed training/reranking variant in the same address space.

## BGB Evidence Result

The BGB run is the stronger end-to-end evidence for generated views, embeddings,
hybrids, and tiny resolver training.

Full BGB fixed-question retrieval:

| Method | hit@1 | hit@10 | MRR |
| --- | ---: | ---: | ---: |
| raw BM25 | 0.364 | 0.574 | 0.434 |
| Refmark BM25 | 0.764 | 0.929 | 0.824 |
| raw Qwen3 embedding | 0.816 | 0.975 | 0.878 |
| Refmark Qwen3 embedding | 0.831 | 0.989 | 0.895 |
| Refmark hybrid w0.05 | 0.826 | 0.989 | 0.892 |

Natural query judge results over 12 fresh German/English concerns:

| Method | supported | usefulness | citation quality |
| --- | ---: | ---: | ---: |
| raw BM25 | 0.333 | 0.208 | 0.200 |
| Refmark BM25 | 0.500 | 0.479 | 0.492 |
| raw Qwen3 embedding | 0.833 | 0.650 | 0.658 |
| Refmark Qwen3 embedding | 0.917 | 0.717 | 0.750 |
| Refmark hybrid w0.05 | 0.917 | 0.800 | 0.833 |

The trained BGB resolver remains exploratory:

- 1.31M parameters
- 5.43 MB artifact
- about 21.9 minutes CPU training in the full bilingual run
- low-millisecond candidate scoring
- BM25 baseline MRR: 0.746
- best blend MRR: 0.753

The useful claim is not "training is solved." The useful claim is that Refmark
makes the training attempt cheap to evaluate, compare, reject, or keep.

### BGB Concern Navigation

Article-level navigation is a complementary BGB path for the practical
"semantic Ctrl+F" demo. The goal is coarser than exact Absatz recovery: given a
free-text concern, land on the right article neighborhood quickly.

The article builder:

```bash
python examples/bgb_browser_search/build_bgb_article_navigation.py
```

uses existing BGB regions, groups them into article refs such as `bgb:S_437`,
and injects curated concern aliases as retrieval metadata. The same query suite
can be run without aliases:

```bash
python examples/bgb_browser_search/build_bgb_article_navigation.py \
  --without-aliases \
  --output-dir examples/bgb_browser_search/output_article_nav_raw
```

Current extended article-level smoke metrics over 104 expected concern rows
plus five ambiguous/no-answer probes:

| Index | expected hit@1 | expected hit@5 |
| --- | ---: | ---: |
| raw article index | 0.7885 | 0.9327 |
| article index + concern aliases | 0.8173 | 0.9327 |

The article-level task is intentionally coarser; raw lexical search is already
strong when exact article vocabulary appears in the question. This is still
useful for finding "motivation understanding" gaps. The
query "I bought a cellphone and found it is broken" fails raw lexical search
because "found" pulls it toward lost-property law. A concern alias maps it to
buyer-defect refs, and the eval suite verifies that the resulting hit is inside
the expected evidence range.

The same alias file is wired into `run_bgb_pipeline.py`. In a 700-region
local-only smoke, the pipeline added four available concern questions:

| Method | concern hit@1 | concern hit@5 |
| --- | ---: | ---: |
| raw BM25 | 0.0000 | 0.0000 |
| Refmark BM25 + concern metadata | 1.0000 | 1.0000 |

This result is intentionally small and not a final benchmark. Its value is that
the loop now measures motivation aliases, generated questions, and normal
retrieval variants in one report.

The full cached BGB Qwen Turbo rerun with
`concern_aliases_extended.json` produced a combined eval suite:

- regions: 4,988
- eval rows: 899
- concern rows: 99
- eval suite artifact: `examples/bgb_browser_search/output_full_qwen_turbo/bgb_eval_questions.jsonl`

| Method | overall hit@1 | overall hit@10 | concern hit@1 | concern hit@10 |
| --- | ---: | ---: | ---: | ---: |
| raw BM25 | 0.3471 | 0.5384 | 0.1717 | 0.2828 |
| Refmark BM25 + generated views + concern aliases | 0.7642 | 0.9366 | 0.8182 | 0.9495 |
| Refmark rerank | 0.7608 | 0.9333 | 0.8081 | 0.9596 |

This is a much more useful concern slice than the earlier 11-row smoke. Held-out
concern queries are not injected into retrieval views; only separate alias text
is injected. The main remaining concern misses now identify concrete adapt-loop
targets: buyer-defect vocabulary around `bgb:S_434/S_437/S_439`, limitation
periods for defects around `bgb:S_438`, pledge expiration around `bgb:S_1252`,
partnership exit rights around `bgb:S_723/S_728/S_740c`, and finder duties
around `bgb:S_965/S_966/S_973`.

The resolver trainer now reports by `gold_mode` too. A bounded full-BGB pass
without index-view augmentation produced:

- train questions: 579
- held-out eval questions: 226
- held-out concern rows: 3
- parameters: 574,977
- artifact size: 2.49 MB
- train time: 22.6 seconds
- inference: about 1 ms/query

| Method | held-out hit@1 | held-out hit@10 | MRR | concern hit@1 |
| --- | ---: | ---: | ---: | ---: |
| BM25 candidate baseline | 0.7257 | 0.9336 | 0.8079 | 1.0000 |
| trained resolver | 0.7566 | 0.9292 | 0.8219 | 1.0000 |

This is a healthier training claim than the earlier broad one: the resolver can
slightly improve rank-1/MRR on held-out candidates while the report verifies it
does not break the concern slice. The concern eval count is still small, so the
next run should force a stratified split that always holds out more concern
rows.

Demo-style browser-index probes after the same rerun:

| Query | Top neighborhood |
| --- | --- |
| `Ich habe ein Handy gekauft und es ist kaputt. Was kann ich tun?` | `bgb:S_439`, `bgb:S_434` |
| `Mein Kind hat online etwas gekauft, ist der Vertrag gültig?` | `bgb:S_106`, `bgb:S_108`, `bgb:S_107`, `bgb:S_110` |
| `Ich habe online bestellt und möchte zurücktreten, welche Frist gilt?` | `bgb:S_355`, `bgb:S_312g`, `bgb:S_356` |
| `Ich habe Schulden geerbt, kann ich die Erbschaft ablehnen?` | `bgb:S_1942`, `bgb:S_1944`, `bgb:S_1943` |

The out-of-domain probe `Python package installieren` still returns package-tour
law because "package" is a valid legal term in this corpus. That is a useful
data-smell: the demo needs confidence/no-answer handling before it should be
presented as an answer system.

### Full-Corpus Weakness Heatmap

The full BGB rerun now reports weakness heatmaps: missed gold refs, wrong top
refs/articles, mode-specific misses, and near-article misses.

| Method | misses | miss rate | near-article misses | concern misses |
| --- | ---: | ---: | ---: | ---: |
| raw BM25 | 415 | 0.4616 | 14 | 71 |
| Refmark BM25 | 57 | 0.0634 | 6 | 5 |
| Refmark rerank | 60 | 0.0667 | 6 | 4 |

Raw BM25 has strong wrong-result magnets:

| Wrong top article | Count | Likely smell |
| --- | ---: | --- |
| `bgb:S_493` | 123 | generic consumer-credit/legal-cost wording |
| `bgb:S_651a` | 35 | "package" ambiguity from package-tour law |
| `bgb:S_438` | 19 | limitation/defect vocabulary |
| `bgb:S_1080` | 16 | property/right terminology magnet |
| `bgb:S_634a` | 11 | limitation/defect vocabulary |

After Refmark enrichment, misses are much more specific. Top missed refs include
`bgb:S_1431_A03`, `bgb:S_575a_A04`, `bgb:S_555d_A06`, `bgb:S_327j_A05`, and
`bgb:S_1252`. These are now candidates for the adapt loop:

- review generated questions for overbroad wording;
- add article-level/neighbor-valid targets when the top hit is the right
  article but wrong Absatz;
- add targeted motivation aliases where user wording is realistic but
  lexically far from the legal text;
- add hard-negative examples for repeated magnets such as generic withdrawal,
  digital-product, and limitation-period provisions.

## CI Framework Evolution

The retrieval CI loop should track these artifacts:

1. Corpus manifest: stable refs, region hashes, parent/neighbor refs.
2. Eval suite: query, gold refs/ranges, source hashes, target shape metadata.
3. Retrieval run: method name, settings, top-k refs, expanded context refs.
4. Metrics: hit@k, MRR, gold coverage, precision, undercite, overcite,
   breadth, stale examples, and sample misses.
5. Generated-view cache: keyed by stable ref, hash, provider, and model.
6. Training artifact: parameters, train/eval split, candidate recall, blend
   weights, latency, and the baseline it had to beat.

Suggested CI gates:

- fail on missing or ambiguous gold refs;
- fail when stale examples exceed an allowed threshold;
- fail when a protected baseline loses more than a configured delta;
- warn when range coverage improves only by overciting too much context;
- report distributed-ref examples separately from single/range examples.

This keeps the system honest: better search is accepted only when the same
evidence address space shows the improvement.

## Next Experiments

- Add separate reports by `gold_mode` for single, adjacent, and disjoint targets.
- Use discovery manifests to guide question generation while keeping local
  discovery questions separate from claim-grade LLM/natural questions.
- Generate higher-quality range/distributed questions with a cheap LLM and keep
  them cached by source hashes.
- Add parent/section targets so a coarse hit inside a long section can be scored
  differently from exact paragraph localization.
- Train the resolver only on high-recall candidate sets and evaluate it as a
  second-stage scorer, not as a replacement for retrieval.
- Add semantic embedding candidates to the same mixed-target report, then check
  whether BM25, embeddings, generated views, and training make overlapping or
  complementary errors.
