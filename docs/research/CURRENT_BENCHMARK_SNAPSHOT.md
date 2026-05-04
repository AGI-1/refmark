# Current Benchmark Snapshot

This is a historical appendix, not a runnable benchmark suite.

This note summarizes the most decision-relevant benchmark evidence behind the current public `refmark` position.

The broad benchmark runners that produced these figures were internal research
infrastructure and are not shipped in this proof-of-concept artifact. The
publicly runnable verification path is intentionally smaller:

```bash
python -m refmark.cli smoke
python -m refmark_train.smoke
pytest
```

Treat the tables below as retained research context. The exact internal runs
are not fully reproducible from this artifact unless a table explicitly points
to shipped retained data or a public example.

For checks that are reproducible from the public artifact, run
`python -m refmark.cli smoke`, `python examples/data_smells/run.py`, and
`python examples/judge_free_rewards/run.py`.

## Supported Claims

The current evidence is strongest for these claims:

1. deterministic support-region retrieval is a useful signal even when exact minimal-span selection remains difficult
2. explicit anchors make citation outputs auditable because returned ids can be resolved back into real source regions
3. on some bounded same-file coding tasks, refmark-style region edits improve reliability for weaker or mid-tier models

## Claims We Do Not Make Yet

The current evidence is not strong enough for:

- a universal "markers help every model" claim
- a universal "refmarks reduce output tokens" claim
- a universal coding-agent improvement claim
- a solved "exact minimal citation" claim
- a broad SWE-bench superiority claim

## Retrieval And Citation Read

### Main result

The strongest retrieval story today is:

- models often find the right support neighborhood even when they miss the exact minimal gold range
- many misses are overcite or undercite boundary errors rather than wrong-location failures
- this makes refmark useful for deterministic evaluation and human review even before exact-minimal citation is fully solved

### Granularity snapshot

Corrected coarse rerun summary:

- `bpm1` mean marker size: about `95.7` words
- `bpm2` mean marker size: about `190.7` words
- `bpm3` mean marker size: about `285.6` words

| Model | bpm1 F1 | bpm1 EM | bpm2 F1 | bpm2 EM | bpm3 F1 | bpm3 EM |
| --- | --- | --- | --- | --- | --- | --- |
| deepseek/deepseek-v3.2 | 0.7926 | 0.4130 | 0.7937 | 0.5870 | 0.8701 | 0.6957 |
| google/gemma-4-26b-a4b-it | 0.7412 | 0.3696 | 0.8304 | 0.5435 | 0.8333 | 0.5870 |
| openai/gpt-5.4-mini | 0.8056 | 0.3478 | 0.8469 | 0.4565 | 0.8312 | 0.4783 |
| openai/gpt-5.4-nano | 0.6471 | 0.2174 | 0.6949 | 0.3913 | 0.6493 | 0.3261 |
| qwen/qwen3.5-flash-02-23 | 0.7286 | 0.3696 | 0.8167 | 0.5217 | 0.8102 | 0.5870 |

### Repeatability snapshot

Repeatability on `golden_range_v2`, XML anchors, corrected `bpm3`, structured output, `5` repeats:

| Model | Exact Same Pred | All Runs Overlap Gold | All Runs Exact Gold | Avg Pairwise Jaccard |
| --- | --- | --- | --- | --- |
| deepseek/deepseek-v3.2 | 0.7174 | 0.8913 | 0.5217 | 0.9264 |
| openai/gpt-5.4-mini | 0.5000 | 1.0000 | 0.2826 | 0.8707 |

Interpretation:

- stronger models usually retrieve the correct support neighborhood
- many non-exact outputs are boundary errors, not hallucinated locations
- the benchmark currently supports a support-region retrieval story more strongly than a perfect minimal-span story

## Agentic Coding Read

### Clean bounded tasks

The strongest coding signal comes from bounded same-file multi-edit tasks, not broad large-file navigation tasks.

Representative Gemini Flash clean-slice result:

- baseline success `0.9667`
- refmark success `1.0`
- baseline avg steps `3.7667`
- refmark avg steps `3.0667`
- baseline output tokens `659.27`
- refmark output tokens `526.63`
- baseline latency `9.70s`
- refmark latency `8.25s`

Important follow-up:

- on a broader mixed slice, refmark was only a slight reliability win and a clear efficiency loss
- on a sharper `patchwithin_v1` slice, refmark became a strong win on both reliability and efficiency

That means the coding story is real but narrow:

- strong for bounded, correctly localized, same-file edits
- not yet proven for broader navigation-heavy coding workflows

### Official-style SWE-bench signal

The official-style `mini-swe-agent` SWE-bench path currently supports a modest but real claim:

- on a small Flash Lite slice, refmark improved task completion tendency and reduced empty-patch failures
- on a larger `0:25` slice, Flash Lite improved from `3` resolved baseline tasks to `7` resolved refmark tasks
- stronger Gemini Flash matched baseline solved count on the same slice rather than clearly exceeding it

Interpretation:

- refmark can materially help weaker agentic models reach valid patches
- stronger models do not automatically benefit unless the tool surface is ergonomic enough

## Cheap Structured-Tool Adoption Read

The local native-tools miniloop is the cleaner place to ask whether non-SWE models can use the refmark localization/editing approach at all.

Current read:

- `google/gemma-4-31b-it` is the strongest cheap non-Gemini positive signal so far
- `qwen/qwen3-235b-a22b-2507` can use the tool surface, but robustness did not survive a broader slice
- `google/gemma-4-26b-a4b-it` and `mistralai/mistral-small-2603` show partial adoption, but not enough stability yet

This supports a practical conclusion:

- the refmark localization/editing abstraction is learnable by multiple model families
- robustness depends heavily on the precision of bounded edit payloads such as `patch_within`

## Publication Read

For public release, the evidence is strong enough to support:

- deterministic locate-only QA and citation evaluation
- highlighted source review for returned refs
- stable same-file Python/TypeScript anchored edits

The evidence is not yet strong enough to support:

- broad coding-agent superiority
- universal efficiency gains
- exact-minimal citation as a solved problem
- training-based localization as a proven product path

## Where More Evidence Would Help Most

The highest-value next evidence would be:

1. more repeatability on one strong citation model family
2. a denser exact-minimal benchmark layer
3. one or two stronger external transfer checks for anchored QA
4. broader coding-agent evidence only after tool ergonomics stabilize further
