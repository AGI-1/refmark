from __future__ import annotations

import json
from pathlib import Path
import random
import re

from refmark_train.real_corpus import load_bundle_from_dir
from refmark_train.synthetic import Anchor, CorpusBundle, Example, save_bundle, tokenize


def prepare_contiguous_range_dataset(
    *,
    data_dir: Path,
    output_dir: Path,
    max_ranges: int = 120,
    min_width: int = 2,
    max_width: int = 4,
    seed: int = 13,
    train_per_range: int = 3,
    valid_per_range: int = 1,
    reform_per_range: int = 2,
    include_single_examples: int = 0,
) -> dict[str, str]:
    source = load_bundle_from_dir(data_dir)
    rng = random.Random(seed)
    anchors = sorted(source.anchors, key=lambda anchor: anchor.refmark)
    candidates = _candidate_windows(anchors, min_width=min_width, max_width=max_width)
    rng.shuffle(candidates)
    selected = sorted(candidates[:max_ranges], key=lambda item: (item[0], item[1]))

    train: list[Example] = []
    valid: list[Example] = []
    reformulated: list[Example] = []
    for start_idx, end_idx in selected:
        window = anchors[start_idx : end_idx + 1]
        center = window[len(window) // 2]
        keyphrases = _window_keyphrases(window)
        question_sets = _range_questions(keyphrases, window)
        answer_text = "\n".join(anchor.text for anchor in window)
        answer_start = int(window[0].region_start)
        answer_end = int(window[-1].region_end)
        for split, limit, target in [
            ("train", train_per_range, train),
            ("valid", valid_per_range, valid),
            ("reformulated", reform_per_range, reformulated),
        ]:
            for question in question_sets[split][:limit]:
                target.append(
                    Example(
                        split=split,
                        prompt_style="generated_contiguous_range",
                        question=question,
                        refmark=center.refmark,
                        title=center.title,
                        answer_text=answer_text,
                        answer_start=answer_start,
                        answer_end=answer_end,
                        gold_start_refmark=window[0].refmark,
                        gold_end_refmark=window[-1].refmark,
                    )
                )

    if include_single_examples > 0:
        single_candidates = [
            anchor
            for anchor in anchors
            if anchor.region_start is not None
            and anchor.region_end is not None
            and 12 <= len(anchor.text.split()) <= 120
        ]
        rng.shuffle(single_candidates)
        for anchor in sorted(single_candidates[:include_single_examples], key=lambda item: item.refmark):
            keyphrases = _window_keyphrases([anchor])
            question_sets = _single_questions(keyphrases, anchor)
            for split, limit, target in [
                ("train", train_per_range, train),
                ("valid", valid_per_range, valid),
                ("reformulated", reform_per_range, reformulated),
            ]:
                for question in question_sets[split][:limit]:
                    target.append(
                        Example(
                            split=split,
                            prompt_style="generated_single_range",
                            question=question,
                            refmark=anchor.refmark,
                            title=anchor.title,
                            answer_text=anchor.text,
                            answer_start=anchor.region_start,
                            answer_end=anchor.region_end,
                            gold_start_refmark=anchor.refmark,
                            gold_end_refmark=anchor.refmark,
                        )
                    )

    bundle = CorpusBundle(anchors=anchors, train=train, valid=valid, reformulated=reformulated)
    files = save_bundle(bundle, output_dir)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "source_data_dir": str(data_dir),
            "range_dataset": {
                "max_ranges": max_ranges,
                "min_width": min_width,
                "max_width": max_width,
                "seed": seed,
                "train_per_range": train_per_range,
                "valid_per_range": valid_per_range,
                "reform_per_range": reform_per_range,
                "include_single_examples": include_single_examples,
                "selected_ranges": len(selected),
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return files


def prepare_refinement_dataset(
    *,
    data_dir: Path,
    output_dir: Path,
    max_examples: int = 160,
    precise_min_width: int = 1,
    precise_max_width: int = 2,
    broad_margin_min: int = 1,
    broad_margin_max: int = 3,
    seed: int = 13,
) -> dict[str, str]:
    source = load_bundle_from_dir(data_dir)
    rng = random.Random(seed)
    anchors = sorted(source.anchors, key=lambda anchor: anchor.refmark)
    precise_candidates = _candidate_windows(anchors, min_width=precise_min_width, max_width=precise_max_width)
    rng.shuffle(precise_candidates)

    train: list[Example] = []
    valid: list[Example] = []
    reformulated: list[Example] = []
    selected: list[tuple[int, int, int, int]] = []
    for precise_start, precise_end in precise_candidates:
        margin = rng.randint(broad_margin_min, broad_margin_max)
        broad_start = max(precise_start - margin, 0)
        broad_end = min(precise_end + margin, len(anchors) - 1)
        if broad_start == precise_start and broad_end == precise_end:
            continue
        broad_window = anchors[broad_start : broad_end + 1]
        precise_window = anchors[precise_start : precise_end + 1]
        if len({anchor.title for anchor in broad_window}) != 1:
            continue
        selected.append((precise_start, precise_end, broad_start, broad_end))
        _append_refinement_examples(
            train,
            valid,
            reformulated,
            precise_window=precise_window,
            broad_window=broad_window,
        )
        if len(selected) >= max_examples:
            break

    bundle = CorpusBundle(anchors=anchors, train=train, valid=valid, reformulated=reformulated)
    files = save_bundle(bundle, output_dir)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "source_data_dir": str(data_dir),
            "refinement_dataset": {
                "max_examples": max_examples,
                "precise_min_width": precise_min_width,
                "precise_max_width": precise_max_width,
                "broad_margin_min": broad_margin_min,
                "broad_margin_max": broad_margin_max,
                "seed": seed,
                "selected_examples": len(selected),
                "pattern": "request -> broad range, then request+broad range -> precise range",
                "generation_rules": [
                    "Broad examples ask for surrounding context and intentionally target a wider covering contiguous range.",
                    "Precise examples target a narrower contiguous sub-range inside the broad range.",
                    "Refinement prompts include a summary of the broad range and ask to narrow to only the needed anchors.",
                    "The same recipe is intended to be reusable across anchored corpora.",
                ],
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return files


def _candidate_windows(anchors: list[Anchor], *, min_width: int, max_width: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for start_idx in range(len(anchors)):
        for width in range(min_width, max_width + 1):
            end_idx = start_idx + width - 1
            if end_idx >= len(anchors):
                continue
            window = anchors[start_idx : end_idx + 1]
            if len({anchor.title for anchor in window}) != 1:
                continue
            if any(anchor.region_start is None or anchor.region_end is None for anchor in window):
                continue
            if sum(len(anchor.text.split()) for anchor in window) < 24:
                continue
            candidates.append((start_idx, end_idx))
    return candidates


def _window_keyphrases(window: list[Anchor]) -> list[str]:
    text = " ".join(anchor.text for anchor in window)
    stopwords = {
        "documentation_set",
        "medical_set",
        "corporate_set",
        "legal_set",
        "documentation",
        "anchor",
        "refmark",
        "marked",
        "range",
        "section",
        "source",
        "using",
        "which",
        "these",
        "there",
        "their",
        "about",
    }
    tokens = [token for token in tokenize(text) if len(token) >= 5 and token not in stopwords]
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts, key=lambda token: (-counts[token], token))
    phrases = ranked[:4]
    caps = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", text)
    for phrase in caps:
        normalized = phrase.strip()
        if len(normalized) >= 5 and normalized.lower() not in {item.lower() for item in phrases}:
            phrases.append(normalized)
        if len(phrases) >= 6:
            break
    return phrases or ["this topic"]


def _range_questions(keyphrases: list[str], window: list[Anchor]) -> dict[str, list[str]]:
    first = keyphrases[0]
    second = keyphrases[1] if len(keyphrases) > 1 else keyphrases[0]
    width = len(window)
    start_hint = _short_preview(window[0].text)
    end_hint = _short_preview(window[-1].text)
    return {
        "train": [
            f"Which contiguous range covers {first} and {second}?",
            f"Find the marked section spanning the discussion from {start_hint} to {end_hint}.",
            f"Which {width}-anchor passage should be cited for {first} together with {second}?",
            f"Locate the continuous citation range that includes both {first} and {second}.",
            f"What adjacent passage gives the complete context for {first} and {second}?",
            f"Which nearby anchors should be cited together for the topic around {first}?",
        ],
        "valid": [
            f"What range should I cite for the connected discussion of {first} and {second}?",
            f"Where is the continuous passage that covers {first} through {second}?",
            f"Which adjacent citation span gives the full context for {first}?",
        ],
        "reformulated": [
            f"I need the whole nearby section, not just one anchor, for {first} and {second}.",
            f"Which adjacent anchors jointly support the topic around {first}?",
            f"Give me the citation span that starts near {start_hint} and ends near {end_hint}.",
            f"Find the complete local passage for {first}, including the surrounding support.",
        ],
    }


def _single_questions(keyphrases: list[str], anchor: Anchor) -> dict[str, list[str]]:
    first = keyphrases[0]
    preview = _short_preview(anchor.text)
    return {
        "train": [
            f"Which single anchor discusses {first}?",
            f"Find the marked passage for {first}.",
            f"Which citation anchor contains {preview}?",
            f"Locate the narrow citation for {first}.",
        ],
        "valid": [
            f"What single citation should I use for {first}?",
            f"Where is {first} covered directly?",
        ],
        "reformulated": [
            f"I only need the focused anchor about {first}.",
            f"Give me the narrow passage centered on {first}.",
        ],
    }


def _append_refinement_examples(
    train: list[Example],
    valid: list[Example],
    reformulated: list[Example],
    *,
    precise_window: list[Anchor],
    broad_window: list[Anchor],
) -> None:
    keyphrases = _window_keyphrases(precise_window)
    first = keyphrases[0]
    second = keyphrases[1] if len(keyphrases) > 1 else keyphrases[0]
    broad_ref = f"{broad_window[0].refmark}-{broad_window[-1].refmark}"
    precise_center = precise_window[len(precise_window) // 2]
    broad_center = broad_window[len(broad_window) // 2]
    broad_text = "\n".join(anchor.text for anchor in broad_window)
    precise_text = "\n".join(anchor.text for anchor in precise_window)
    broad_question = f"Where is the surrounding context for {first} and {second}?"
    broad_reform = f"Find a broad local citation span that covers the area around {first}."
    refine_prefix = f"Previous broad range {broad_ref}: {_short_preview(broad_text, max_words=18)}."
    refine_question = f"{refine_prefix} Narrow this to only the anchors needed for {first} and {second}."
    refine_reform = f"{refine_prefix} Reduce the citation to the focused passage about {first}."

    broad_train_questions = [
        broad_question,
        f"Find a broad local citation span that covers the area around {first}.",
        f"Give me surrounding context, not the narrow answer yet, for {first} and {second}.",
        f"Which nearby section should I open first before narrowing down {first}?",
        f"Show me the wider local passage for {first}, including useful surrounding anchors.",
        f"I need the broader context around {first} and {second} before selecting a final citation.",
        f"What adjacent region gives the setup and follow-up around {first}?",
        f"Locate the local context window around {first}, not just the exact line.",
        f"Which broader citation span should I inspect first for {first}?",
        f"Find the surrounding explanation that contains the topic around {first} and {second}.",
    ]
    train.extend(
        [_range_example("train", "refinement_broad_train", question, broad_window, broad_center, broad_text) for question in broad_train_questions]
        + [
            _range_example("train", "refinement_precise_train", refine_question, precise_window, precise_center, precise_text),
            _range_example(
                "train",
                "refinement_precise_train",
                f"Given broad range {broad_ref}, cite only the narrow part for {first}.",
                precise_window,
                precise_center,
                precise_text,
            ),
            _range_example(
                "train",
                "refinement_precise_train",
                f"{refine_prefix} Keep only the minimal contiguous anchors for {first}.",
                precise_window,
                precise_center,
                precise_text,
            ),
        ]
    )
    valid.extend(
        [
            _range_example("valid", "refinement_broad_valid", broad_question, broad_window, broad_center, broad_text),
            _range_example("valid", "refinement_precise_valid", refine_question, precise_window, precise_center, precise_text),
        ]
    )
    reformulated.extend(
        [
            _range_example(
                "reformulated",
                "refinement_broad_reformulated",
                broad_reform,
                broad_window,
                broad_center,
                broad_text,
            ),
            _range_example(
                "reformulated",
                "refinement_precise_reformulated",
                refine_reform,
                precise_window,
                precise_center,
                precise_text,
            ),
        ]
    )


def _range_example(
    split: str,
    prompt_style: str,
    question: str,
    window: list[Anchor],
    center: Anchor,
    answer_text: str,
) -> Example:
    return Example(
        split=split,
        prompt_style=prompt_style,
        question=question,
        refmark=center.refmark,
        title=center.title,
        answer_text=answer_text,
        answer_start=window[0].region_start,
        answer_end=window[-1].region_end,
        gold_start_refmark=window[0].refmark,
        gold_end_refmark=window[-1].refmark,
    )


def _short_preview(text: str, max_words: int = 8) -> str:
    words = " ".join(text.split()).split()
    return " ".join(words[:max_words])
