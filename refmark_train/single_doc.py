from __future__ import annotations

import json
from pathlib import Path
import random
import re
import math

from refmark_train.synthetic import Anchor, CorpusBundle, Example, save_bundle


PARAGRAPH_SPLIT_RE = re.compile(r"\r?\n\s*\r?\n+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
CAP_PHRASE_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")
LONG_WORD_RE = re.compile(r"\b[a-z]{7,}\b")
TERM_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_-]{3,}\b")
LIST_ITEM_RE = re.compile(r"^(?:[-*•]|\d+[.)]|[A-Z][.)])\s+")
TABLEISH_RE = re.compile(r"\s{2,}|\|")
STOPWORDS = {
    "about",
    "after",
    "also",
    "because",
    "before",
    "between",
    "could",
    "during",
    "example",
    "following",
    "however",
    "include",
    "includes",
    "including",
    "information",
    "other",
    "should",
    "source",
    "their",
    "there",
    "these",
    "those",
    "through",
    "which",
    "while",
    "would",
}


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def split_paragraphs(text: str) -> list[tuple[int, int, str]]:
    paragraphs: list[tuple[int, int, str]] = []
    cursor = 0
    for part in PARAGRAPH_SPLIT_RE.split(text):
        chunk = part.strip()
        if len(chunk) < 80:
            idx = text.find(part, cursor)
            if idx >= 0:
                cursor = idx + len(part)
            continue
        idx = text.find(part, cursor)
        if idx < 0:
            idx = text.find(chunk, cursor)
        if idx < 0:
            continue
        start = text.find(chunk, idx)
        end = start + len(chunk)
        paragraphs.append((start, end, chunk))
        cursor = end
    return paragraphs


def split_sentences(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    start = 0
    for match in SENTENCE_SPLIT_RE.finditer(text):
        end = match.start()
        sentence = text[start:end].strip()
        if sentence and len(sentence) >= 30:
            sentence_start = text.find(sentence, start, end + 1)
            spans.append((sentence_start, sentence_start + len(sentence), sentence))
        start = match.end()
    tail = text[start:].strip()
    if tail and len(tail) >= 30:
        sentence_start = text.find(tail, start)
        spans.append((sentence_start, sentence_start + len(tail), tail))
    return spans


def window_paragraphs(
    paragraphs: list[tuple[int, int, str]],
    *,
    window: int = 1,
    stride: int = 1,
) -> list[tuple[int, int, str]]:
    regions: list[tuple[int, int, str]] = []
    for i in range(0, max(len(paragraphs) - window + 1, 0), stride):
        chunk = paragraphs[i : i + window]
        if not chunk:
            continue
        start = chunk[0][0]
        end = chunk[-1][1]
        text = "\n\n".join(part[2] for part in chunk)
        regions.append((start, end, text))
    return regions


def _token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9_]+", text))


def _is_heading(block: str) -> bool:
    line = " ".join(block.split())
    if not line or len(line) > 120:
        return False
    if line.endswith((".", ";", ":")):
        return False
    words = line.split()
    if not words or len(words) > 14:
        return False
    alpha_words = [word for word in words if any(ch.isalpha() for ch in word)]
    if not alpha_words:
        return False
    title_like = sum(1 for word in alpha_words if word[:1].isupper() or word.isupper())
    return title_like / len(alpha_words) >= 0.7


def _classify_block(block: str) -> str:
    stripped = block.strip()
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        return "empty"
    if len(lines) == 1 and _is_heading(lines[0]):
        return "heading"
    if all(LIST_ITEM_RE.match(line) for line in lines):
        return "list"
    if len(lines) >= 2 and sum(1 for line in lines if TABLEISH_RE.search(line)) >= max(2, len(lines) // 2):
        return "table"
    return "paragraph"


def _find_block_offsets(text: str, blocks: list[str]) -> list[tuple[int, int, str]]:
    output: list[tuple[int, int, str]] = []
    cursor = 0
    for block in blocks:
        idx = text.find(block, cursor)
        if idx < 0:
            idx = text.find(block.strip(), cursor)
            if idx < 0:
                continue
            block = block.strip()
        start = idx
        end = start + len(block)
        output.append((start, end, block.strip()))
        cursor = end
    return output


def split_structural_blocks(
    text: str,
    *,
    min_tokens: int = 12,
    max_tokens: int = 80,
    sentence_window: int = 2,
) -> list[tuple[int, int, str, str]]:
    raw_blocks = [part for part in PARAGRAPH_SPLIT_RE.split(text) if part.strip()]
    spans = _find_block_offsets(text, raw_blocks)
    output: list[tuple[int, int, str, str]] = []
    pending_small: tuple[int, int, str, str] | None = None
    for start, end, block in spans:
        block_type = _classify_block(block)
        normalized = block.strip()
        if block_type == "paragraph" and _token_count(normalized) > max_tokens:
            sentences = split_sentences(normalized)
            if sentences:
                for i in range(0, len(sentences), sentence_window):
                    chunk = sentences[i : i + sentence_window]
                    chunk_text = " ".join(part[2] for part in chunk).strip()
                    if not chunk_text:
                        continue
                    local_start = normalized.find(chunk[0][2], 0)
                    if local_start < 0:
                        continue
                    local_end = local_start + len(chunk_text)
                    output.append((start + local_start, start + local_end, chunk_text, "sentence_window"))
                continue
        token_count = _token_count(normalized)
        current = (start, end, normalized, block_type)
        if token_count < min_tokens:
            if pending_small is None:
                pending_small = current
            else:
                merged = (
                    pending_small[0],
                    end,
                    pending_small[2] + "\n\n" + normalized,
                    "merged_small",
                )
                output.append(merged)
                pending_small = None
            continue
        if pending_small is not None:
            merged = (
                pending_small[0],
                end,
                pending_small[2] + "\n\n" + normalized,
                "merged_prefix",
            )
            output.append(merged)
            pending_small = None
        else:
            output.append(current)
    if pending_small is not None:
        output.append(pending_small)
    return output


def extract_keyphrases(text: str) -> list[tuple[str, int, int]]:
    candidates: list[tuple[str, int, int]] = []
    seen: set[str] = set()
    for match in CAP_PHRASE_RE.finditer(text):
        phrase = match.group(0).strip()
        if len(phrase) < 4 or phrase.lower() in seen:
            continue
        seen.add(phrase.lower())
        candidates.append((phrase, match.start(), match.end()))
    for match in LONG_WORD_RE.finditer(text):
        phrase = match.group(0).strip()
        if len(phrase) < 8 or phrase.lower() in seen:
            continue
        seen.add(phrase.lower())
        candidates.append((phrase, match.start(), match.end()))
    return candidates[:8]


def _region_terms(text: str) -> set[str]:
    return {
        match.group(0).lower()
        for match in TERM_RE.finditer(text)
        if match.group(0).lower() not in STOPWORDS
    }


def _region_document_frequencies(regions) -> dict[str, int]:
    frequencies: dict[str, int] = {}
    for region in regions:
        if len(region) == 4:
            _, _, region_text, _ = region
        else:
            _, _, region_text = region
        for term in _region_terms(region_text):
            frequencies[term] = frequencies.get(term, 0) + 1
    return frequencies


def _keyphrase_score(phrase: str, doc_freq: dict[str, int], region_count: int) -> float:
    terms = [term for term in _region_terms(phrase) if len(term) >= 4]
    if not terms:
        return -100.0
    idf_sum = sum(math.log((region_count + 1) / (doc_freq.get(term, region_count) + 1)) for term in terms)
    generic_penalty = 1.0 if any(doc_freq.get(term, region_count) > region_count * 0.08 for term in terms) else 0.0
    length_bonus = min(len(" ".join(terms)) / 40.0, 0.5)
    return idf_sum + length_bonus - generic_penalty


def _distinctive_terms(text: str, *, limit: int = 5) -> list[str]:
    counts: dict[str, int] = {}
    first_seen: dict[str, str] = {}
    for match in re.finditer(r"\b[A-Za-z][A-Za-z0-9_-]{4,}\b", text):
        token = match.group(0)
        key = token.lower()
        if key in STOPWORDS:
            continue
        counts[key] = counts.get(key, 0) + 1
        first_seen.setdefault(key, token)
    ranked = sorted(counts, key=lambda key: (-counts[key], -len(key), key))
    return [first_seen[key] for key in ranked[:limit]]


def _region_summary_terms(region_text: str) -> tuple[str, str, str]:
    compact = " ".join(region_text.split())
    lead = " ".join(compact.split()[:18])
    terms = _distinctive_terms(region_text, limit=5)
    primary = terms[0] if terms else (lead.split()[0] if lead else "this passage")
    secondary = terms[1] if len(terms) > 1 else primary
    term_list = ", ".join(terms[:4]) if terms else primary
    return primary, secondary, term_list


def _contextual_anchor_question_sets(
    phrase: str,
    clue: str | None,
    doc_id: str,
    region_text: str,
    rng: random.Random,
    *,
    train_count: int = 5,
    valid_count: int = 2,
    reform_count: int = 4,
) -> tuple[list[str], list[str], list[str]]:
    safe = phrase.strip()
    primary, secondary, term_list = _region_summary_terms(region_text)
    clue_text = clue or secondary
    train_pool = [
        f"Which local citation area explains {safe} in the context of {primary}?",
        f"Find the anchor passage where {safe} is discussed with {clue_text}.",
        f"Which marked region contains evidence about {safe} and related terms {term_list}?",
        f"Where should I cite for the part about {safe} near {primary}?",
        f"Locate the paragraph anchor that connects {safe} with {clue_text}.",
        f"Which refmark has the local discussion involving {safe}, {primary}, and {secondary}?",
    ]
    valid_pool = [
        f"What anchor best supports a claim about {safe} in the same passage as {primary}?",
        f"Which citation region is most directly about {safe} and {clue_text}?",
        f"Point to the local marked paragraph for {safe} around {term_list}.",
    ]
    reform_pool = [
        f"In {doc_id}, identify the local support span for a question involving {safe} and {primary}.",
        f"Which paragraph-level refmark would verify a statement about {safe} alongside {clue_text}?",
        f"Find the evidence area in {doc_id} where the discussion combines {safe} with {term_list}.",
        f"If answering about {safe}, which nearby citation should be used when {primary} is also mentioned?",
        f"Which local anchor in {doc_id} is the best match for the topic cluster {safe}, {primary}, {secondary}?",
    ]
    rng.shuffle(train_pool)
    rng.shuffle(valid_pool)
    rng.shuffle(reform_pool)
    return (
        train_pool[: min(train_count, len(train_pool))],
        valid_pool[: min(valid_count, len(valid_pool))],
        reform_pool[: min(reform_count, len(reform_pool))],
    )


def _question_templates(phrase: str) -> list[str]:
    safe = phrase.strip()
    return [
        f"Which marker range discusses {safe}?",
        f"Where in the document is {safe} described?",
        f"Find the refmark range most relevant to {safe}.",
        f"Which anchor region mentions {safe}?",
        f"Locate the passage centered on {safe}.",
    ]


def _randomized_train_templates(phrase: str, clue: str | None) -> list[str]:
    safe = phrase.strip()
    templates = [
        f"Which anchor explains {safe}?",
        f"Where is {safe} introduced in the source?",
        f"Find the citation that mentions {safe}.",
        f"Which marked passage contains the term {safe}?",
        f"Locate the source passage for {safe}.",
    ]
    if clue:
        templates.extend(
            [
                f"Which anchor connects {safe} with {clue}?",
                f"Where are {safe} and {clue} discussed together?",
                f"Find the passage that mentions both {safe} and {clue}.",
            ]
        )
    return templates


def _mutation_templates(question: str) -> list[str]:
    base = question.strip().rstrip("?").rstrip(".")
    variants = [
        base + "?",
        base.replace("Which anchor", "Which marked passage") + "?",
        base.replace("Where is", "Where can I find") + "?",
        base.replace("Find the citation", "Locate the citation") + "?",
        base.replace("mentions", "refers to") + "?",
        base.replace("contains", "includes") + "?",
    ]
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in variants:
        text = re.sub(r"\s+", " ", item).strip()
        if text and text.lower() not in seen:
            cleaned.append(text)
            seen.add(text.lower())
    return cleaned


def _randomized_valid_templates(phrase: str, clue: str | None) -> list[str]:
    safe = phrase.strip()
    templates = [
        f"What citation should support a statement about {safe}?",
        f"Which anchored passage is the best evidence for {safe}?",
        f"Point to the marked source text relevant to {safe}.",
        f"Where would you cite for information on {safe}?",
        f"Which refmark is most directly about {safe}?",
    ]
    if clue:
        templates.extend(
            [
                f"What citation supports the relationship between {safe} and {clue}?",
                f"Which marked passage provides evidence involving {safe} and {clue}?",
                f"Where is {safe} discussed in the same context as {clue}?",
            ]
        )
    return templates


def _randomized_reformulation_templates(phrase: str, clue: str | None, doc_id: str) -> list[str]:
    safe = phrase.strip()
    templates = [
        f"In {doc_id}, identify the citation most relevant to the concept {safe}.",
        f"From {doc_id}, which anchored text would verify a claim about {safe}?",
        f"If someone asks about {safe}, which marked passage in {doc_id} should be cited?",
        f"Find evidence in {doc_id} for the topic {safe}.",
        f"Which refmark in {doc_id} would help answer a question involving {safe}?",
    ]
    if clue:
        templates.extend(
            [
                f"In {doc_id}, locate the citation where {safe} appears with supporting context about {clue}.",
                f"Which anchor in {doc_id} supports a claim connecting {safe} and {clue}?",
                f"Find the marked evidence in {doc_id} that includes both {safe} and {clue}.",
            ]
        )
    return templates


def _reformulation_templates(phrase: str) -> list[str]:
    safe = phrase.strip()
    return [
        f"Which marked region is about the topic '{safe}'?",
        f"Identify the anchor span that best matches {safe}.",
        f"What citation range should be used for {safe}?",
    ]


def _descriptor_for_phrase(phrase: str) -> str:
    tokens = phrase.split()
    initials = "".join(token[0].upper() for token in tokens if token)
    if len(tokens) >= 2 and all(token[:1].isupper() for token in tokens):
        return f"the {len(tokens)}-word capitalized expression with initials {initials}"
    if phrase[:1].isupper():
        return f"the capitalized expression beginning with {phrase[0].upper()} and length {len(phrase)}"
    return f"the lowercase keyword beginning with {phrase[0].lower()} and length {len(phrase)}"


def _clue_templates(phrase: str, clue: str) -> list[str]:
    descriptor = _descriptor_for_phrase(phrase)
    return [
        f"Which marker range discusses {clue} together with {descriptor}?",
        f"Locate the anchor region where {clue} appears alongside {descriptor}.",
        f"Find the marked passage combining {clue} and {descriptor}.",
        f"Which refmark range is about {clue} and also mentions {descriptor}?",
    ]


def _hard_reformulation_templates(phrase: str, clue: str, doc_id: str) -> list[str]:
    descriptor = _descriptor_for_phrase(phrase)
    return [
        f"In {doc_id}, which marked region matches {descriptor} in the same discussion as {clue}?",
        f"Which citation span in {doc_id} should be used for {descriptor}, using {clue} as the supporting clue?",
        f"Identify the anchored range in {doc_id} where {clue} co-occurs with {descriptor}.",
    ]


def _anchor_question_sets(phrase: str, clue: str | None, doc_id: str) -> tuple[list[str], list[str], list[str]]:
    train_questions = list(_question_templates(phrase)[:2])
    valid_questions: list[str] = []
    reform_questions: list[str] = []
    if clue:
        clue_templates = _clue_templates(phrase, clue)
        train_questions.extend(clue_templates[:2])
        valid_questions.append(clue_templates[2])
        reform_questions.extend(_hard_reformulation_templates(phrase, clue, doc_id))
    else:
        train_questions.append(_question_templates(phrase)[2])
        valid_questions.append(_question_templates(phrase)[3])
        reform_questions.extend(_reformulation_templates(phrase))
    return train_questions, valid_questions, reform_questions


def _randomized_anchor_question_sets(
    phrase: str,
    clue: str | None,
    doc_id: str,
    rng: random.Random,
    *,
    train_count: int = 4,
    valid_count: int = 2,
    reform_count: int = 4,
) -> tuple[list[str], list[str], list[str]]:
    train_pool = _randomized_train_templates(phrase, clue)
    valid_pool = _randomized_valid_templates(phrase, clue)
    reform_pool = _randomized_reformulation_templates(phrase, clue, doc_id)
    rng.shuffle(train_pool)
    rng.shuffle(valid_pool)
    rng.shuffle(reform_pool)
    return (
        train_pool[: min(train_count, len(train_pool))],
        valid_pool[: min(valid_count, len(valid_pool))],
        reform_pool[: min(reform_count, len(reform_pool))],
    )


def _augment_training_questions(questions: list[str], rng: random.Random, max_extra: int) -> list[str]:
    if max_extra <= 0:
        return questions
    augmented = list(questions)
    candidates: list[str] = []
    for question in questions:
        candidates.extend(_mutation_templates(question))
    rng.shuffle(candidates)
    seen = {question.lower() for question in augmented}
    for candidate in candidates:
        if candidate.lower() in seen:
            continue
        augmented.append(candidate)
        seen.add(candidate.lower())
        if len(augmented) >= len(questions) + max_extra:
            break
    return augmented


def build_single_doc_bundle(
    text: str,
    *,
    doc_id: str,
    unit: str = "paragraph",
    paragraph_window: int = 1,
    stride: int = 1,
    anchor_limit: int = 500,
    seed: int = 13,
    question_mode: str = "legacy",
    train_mutations: int = 0,
    train_questions_per_phrase: int | None = None,
    valid_questions_per_phrase: int | None = None,
    reform_questions_per_phrase: int | None = None,
) -> CorpusBundle:
    rng = random.Random(seed)
    if unit == "sentence":
        base_spans = split_sentences(text)
        regions = [(start, end, content) for start, end, content in window_paragraphs(base_spans, window=paragraph_window, stride=stride)]
        anchor_type = "sentence_window"
    elif unit == "structure":
        structured = split_structural_blocks(text)
        base_spans = [(start, end, content) for start, end, content, _ in structured]
        regions = [(start, end, content, block_type) for start, end, content, block_type in structured]
        anchor_type = None
    else:
        base_spans = split_paragraphs(text)
        regions = [(start, end, content) for start, end, content in window_paragraphs(base_spans, window=paragraph_window, stride=stride)]
        anchor_type = "paragraph_window"
    anchors: list[Anchor] = []
    train: list[Example] = []
    valid: list[Example] = []
    reformulated: list[Example] = []
    doc_freq = _region_document_frequencies(regions)
    region_count = max(len(regions), 1)

    for idx, region in enumerate(regions[:anchor_limit], start=1):
        if unit == "structure":
            start, end, region_text, region_anchor_type = region
        else:
            start, end, region_text = region
            region_anchor_type = anchor_type or "paragraph_window"
        refmark = f"D{idx:05d}"
        preview = region_text.replace("\n", " ")
        preview = preview[:320] + ("..." if len(preview) > 320 else "")
        anchor = Anchor(
            refmark=refmark,
            domain="document",
            subject=doc_id,
            action="discuss",
            object="topic",
            condition="from source text",
            anchor_type=region_anchor_type,
            text=f"[{refmark}] {doc_id}: {preview}",
            title=doc_id,
            region_start=start,
            region_end=end,
        )
        keyphrases = extract_keyphrases(region_text)
        if len(keyphrases) < 2:
            continue
        anchors.append(anchor)
        keyphrases = sorted(
            keyphrases,
            key=lambda item: _keyphrase_score(item[0], doc_freq, region_count),
            reverse=True,
        )
        phrase_pairs = [item for item in keyphrases if _keyphrase_score(item[0], doc_freq, region_count) > -0.25][:3]
        if len(phrase_pairs) < 2:
            phrase_pairs = keyphrases[:3]
        for phrase_idx, (phrase, local_start, local_end) in enumerate(phrase_pairs[:2]):
            global_start = start + local_start
            global_end = start + local_end
            clue = None
            for other_idx, (other_phrase, _, _) in enumerate(phrase_pairs):
                if other_idx != phrase_idx and other_phrase.lower() != phrase.lower():
                    clue = other_phrase
                    break
            if question_mode == "contextual":
                train_questions, valid_questions, reform_questions = _contextual_anchor_question_sets(
                    phrase,
                    clue,
                    doc_id,
                    region_text,
                    rng,
                    train_count=train_questions_per_phrase or 5,
                    valid_count=valid_questions_per_phrase or 2,
                    reform_count=reform_questions_per_phrase or 4,
                )
            elif question_mode == "randomized":
                train_questions, valid_questions, reform_questions = _randomized_anchor_question_sets(
                    phrase,
                    clue,
                    doc_id,
                    rng,
                )
            else:
                train_questions, valid_questions, reform_questions = _anchor_question_sets(
                    phrase,
                    clue,
                    doc_id,
                )
                if train_questions_per_phrase is not None:
                    train_questions = train_questions[:train_questions_per_phrase]
                if valid_questions_per_phrase is not None:
                    valid_questions = valid_questions[:valid_questions_per_phrase]
                if reform_questions_per_phrase is not None:
                    reform_questions = reform_questions[:reform_questions_per_phrase]
            train_questions = _augment_training_questions(train_questions, rng, train_mutations)
            for template in train_questions:
                train.append(
                    Example(
                        split="train",
                        prompt_style="generated_train",
                        question=template,
                        refmark=refmark,
                        title=doc_id,
                        answer_text=phrase,
                        answer_start=global_start,
                        answer_end=global_end,
                    )
                )
            for question in valid_questions:
                valid.append(
                    Example(
                        split="valid",
                        prompt_style="generated_valid",
                        question=question,
                        refmark=refmark,
                        title=doc_id,
                        answer_text=phrase,
                        answer_start=global_start,
                        answer_end=global_end,
                    )
                )
            for reform in reform_questions:
                reformulated.append(
                    Example(
                        split="reformulated",
                        prompt_style="generated_reformulation",
                        question=reform,
                        refmark=refmark,
                        title=doc_id,
                        answer_text=phrase,
                        answer_start=global_start,
                        answer_end=global_end,
                    )
                )
    rng.shuffle(train)
    rng.shuffle(valid)
    rng.shuffle(reformulated)
    return CorpusBundle(
        anchors=anchors,
        train=train,
        valid=valid,
        reformulated=reformulated,
    )


def prepare_single_doc_dataset(
    input_path: Path,
    output_dir: Path,
    *,
    unit: str = "paragraph",
    paragraph_window: int = 1,
    stride: int = 1,
    anchor_limit: int = 500,
    seed: int = 13,
    question_mode: str = "legacy",
    train_mutations: int = 0,
    train_questions_per_phrase: int | None = None,
    valid_questions_per_phrase: int | None = None,
    reform_questions_per_phrase: int | None = None,
) -> dict[str, str]:
    text = load_text(input_path)
    bundle = build_single_doc_bundle(
        text,
        doc_id=input_path.stem,
        unit=unit,
        paragraph_window=paragraph_window,
        stride=stride,
        anchor_limit=anchor_limit,
        seed=seed,
        question_mode=question_mode,
        train_mutations=train_mutations,
        train_questions_per_phrase=train_questions_per_phrase,
        valid_questions_per_phrase=valid_questions_per_phrase,
        reform_questions_per_phrase=reform_questions_per_phrase,
    )
    files = save_bundle(bundle, output_dir)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "source": str(input_path),
            "unit": unit,
            "paragraph_window": paragraph_window,
            "stride": stride,
            "seed": seed,
            "question_mode": question_mode,
            "train_mutations": train_mutations,
            "train_questions_per_phrase": train_questions_per_phrase,
            "valid_questions_per_phrase": valid_questions_per_phrase,
            "reform_questions_per_phrase": reform_questions_per_phrase,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return files
