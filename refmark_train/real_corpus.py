from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import re
from typing import Iterable
from urllib.request import urlopen

from refmark_train.synthetic import Anchor, CorpusBundle, Example, save_bundle, tokenize


SQUAD_TRAIN_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
SQUAD_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
STOPWORDS = {
    "a", "an", "the", "in", "on", "of", "to", "for", "and", "or", "by", "with",
    "is", "was", "were", "are", "be", "from", "that", "this", "which", "what",
    "who", "when", "where", "why", "how", "did", "does", "do", "it", "as", "at",
}


def download_json(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=60) as response:
        payload = response.read()
    destination.write_bytes(payload)
    return destination


def split_sentences(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    start = 0
    for match in SENTENCE_RE.finditer(text):
        end = match.start()
        sentence = text[start:end].strip()
        if sentence:
            sentence_start = text.find(sentence, start, end + 1)
            spans.append((sentence_start, sentence_start + len(sentence), sentence))
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentence_start = text.find(tail, start)
        spans.append((sentence_start, sentence_start + len(tail), tail))
    return spans


def _pick_sentence(context: str, answer_start: int, sentence_window: int = 0) -> tuple[int, int, str]:
    sentences = split_sentences(context)
    if not sentences:
        stripped = context.strip()
        return (0, len(stripped), stripped)
    target_idx = 0
    for idx, span in enumerate(sentences):
        start, end, _ = span
        if start <= answer_start < end:
            target_idx = idx
            break
    start_idx = max(0, target_idx - sentence_window)
    end_idx = min(len(sentences) - 1, target_idx + sentence_window)
    region_start = sentences[start_idx][0]
    region_end = sentences[end_idx][1]
    region_text = context[region_start:region_end].strip()
    return (region_start, region_end, region_text)


def _normalize_question(question: str) -> str:
    text = question.strip().rstrip("?").strip()
    lowered = text.lower()
    replacements = [
        ("how many", "the count of"),
        ("how much", "the amount of"),
        ("how old", "the age of"),
        ("when did", "the time when"),
        ("where did", "the place where"),
        ("who was", "the person who was"),
        ("who is", "the person who is"),
        ("what was", "the fact about"),
        ("what is", "the fact about"),
        ("which", "the specific"),
        ("who", "the person who"),
        ("when", "the time when"),
        ("where", "the place where"),
        ("why", "the reason why"),
        ("what", "the fact about"),
    ]
    for source, target in replacements:
        if lowered.startswith(source + " "):
            return target + " " + text[len(source):].strip()
    return text


def _reformulate_question(question: str, title: str, *, hard: bool = False) -> str:
    base = question.strip().rstrip("?")
    lowered = base.lower()
    if not hard:
        prefixes = [
            f"In the Wikipedia article '{title}', {lowered}?",
            f"From the article '{title}', identify where this is answered: {lowered}.",
            f"Using the page '{title}', which anchored sentence answers: {lowered}?",
        ]
    else:
        normalized = _normalize_question(question)
        prefixes = [
            f"Identify the anchored passage in '{title}' needed to resolve this prompt: {normalized}.",
            f"Locate the citation from '{title}' that establishes {normalized}.",
            f"Within '{title}', find the anchor that supports the query about {normalized}.",
            f"Choose the anchored region in '{title}' that directly answers this request: {normalized}.",
        ]
    return prefixes[hash((question, title, hard)) % len(prefixes)]


def _content_tokens(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in STOPWORDS and len(token) > 2}


def _anchor_difficulty_scores(anchors: Iterable[Anchor]) -> dict[str, float]:
    anchor_list = list(anchors)
    token_sets = {anchor.refmark: _content_tokens(anchor.text) for anchor in anchor_list}
    scores: dict[str, float] = {}
    for anchor in anchor_list:
        base = token_sets[anchor.refmark]
        best_jaccard = 0.0
        best_overlap = 0
        same_title_bonus = 0.0
        for other in anchor_list:
            if other.refmark == anchor.refmark:
                continue
            other_tokens = token_sets[other.refmark]
            intersection = len(base & other_tokens)
            if intersection <= 0:
                continue
            union = len(base | other_tokens)
            jaccard = intersection / max(union, 1)
            best_jaccard = max(best_jaccard, jaccard)
            best_overlap = max(best_overlap, intersection)
            if anchor.title and anchor.title == other.title:
                same_title_bonus = max(same_title_bonus, jaccard)
        scores[anchor.refmark] = best_jaccard * 10.0 + best_overlap + same_title_bonus * 5.0
    return scores


def _question_difficulty_scores(
    candidate_refmarks: list[str],
    grouped_questions: dict[str, list[tuple[str, str, int, int, str]]],
    anchor_by_refmark: dict[str, Anchor],
) -> dict[str, float]:
    candidate_set = set(candidate_refmarks)
    candidate_anchors = [anchor for refmark, anchor in anchor_by_refmark.items() if refmark in candidate_set]
    anchor_tokens = {anchor.refmark: _content_tokens(anchor.text) for anchor in candidate_anchors}
    anchor_conf = _anchor_difficulty_scores(candidate_anchors)
    scores: dict[str, float] = {}
    for refmark in candidate_refmarks:
        gold_tokens = anchor_tokens.get(refmark, set())
        question_rows = grouped_questions.get(refmark, [])
        if not question_rows:
            scores[refmark] = anchor_conf.get(refmark, 0.0)
            continue
        question_score = 0.0
        used = 0
        for question, _, _, _, _ in question_rows[: min(4, len(question_rows))]:
            qtokens = _content_tokens(question)
            if not qtokens:
                continue
            gold_overlap = len(qtokens & gold_tokens)
            best_other_overlap = 0
            for other_refmark, other_tokens in anchor_tokens.items():
                if other_refmark == refmark:
                    continue
                best_other_overlap = max(best_other_overlap, len(qtokens & other_tokens))
            gap = gold_overlap - best_other_overlap
            question_score += best_other_overlap * 2.0 + max(0.0, 2.0 - gap)
            if best_other_overlap >= gold_overlap:
                question_score += 2.0
            used += 1
        if used > 0:
            question_score /= used
        scores[refmark] = anchor_conf.get(refmark, 0.0) + question_score
    return scores


def build_squad_bundle(
    train_payload: dict,
    dev_payload: dict,
    *,
    train_limit: int = 3000,
    dev_limit: int = 600,
    min_questions_per_anchor: int = 3,
    anchor_limit: int | None = None,
    sentence_window: int = 0,
    rank_mode: str = "frequency",
    hard_reformulated: bool = False,
) -> CorpusBundle:
    anchor_map: dict[tuple[str, int, int], Anchor] = {}
    grouped_questions: dict[str, list[tuple[str, str, int, int, str]]] = {}
    anchor_counter = 1

    def ensure_anchor(title: str, context: str, answer_start: int) -> Anchor:
        nonlocal anchor_counter
        sent_start, sent_end, sentence = _pick_sentence(
            context,
            answer_start,
            sentence_window=sentence_window,
        )
        key = (title, sent_start, sent_end)
        if key not in anchor_map:
            refmark = f"W{anchor_counter:05d}"
            anchor_counter += 1
            anchor_map[key] = Anchor(
                refmark=refmark,
                domain="wikipedia",
                subject=title,
                action="answer",
                object="question",
                condition="from squad",
                anchor_type="sentence",
                text=f"[{refmark}] {title}: {sentence}",
                title=title,
                region_start=sent_start,
                region_end=sent_end,
            )
        return anchor_map[key]

    def collect(payload: dict) -> None:
        for article in payload["data"]:
            title = article["title"].replace("_", " ")
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                for qa in paragraph["qas"]:
                    if not qa.get("answers"):
                        continue
                    answer = qa["answers"][0]
                    answer_start = int(answer["answer_start"])
                    answer_text = answer["text"]
                    answer_end = answer_start + len(answer_text)
                    anchor = ensure_anchor(title, context, answer_start)
                    question = qa["question"].strip()
                    grouped_questions.setdefault(anchor.refmark, []).append(
                        (question, title, answer_start, answer_end, answer_text)
                    )

    collect(train_payload)
    collect(dev_payload)

    train_examples: list[Example] = []
    valid_examples: list[Example] = []
    reformulated_examples: list[Example] = []

    train_count = 0
    dev_count = 0
    selected_refmarks: set[str] = set()
    if rank_mode == "confusable":
        candidate_refmarks = sorted(
            grouped_questions,
            key=lambda refmark: (-len(grouped_questions[refmark]), refmark),
        )
        if anchor_limit is not None:
            candidate_refmarks = candidate_refmarks[: max(anchor_limit * 4, anchor_limit)]
        difficulty = _question_difficulty_scores(
            candidate_refmarks,
            grouped_questions,
            {anchor.refmark: anchor for anchor in anchor_map.values()},
        )
        ranked_refmarks = sorted(
            candidate_refmarks,
            key=lambda refmark: (-difficulty.get(refmark, 0.0), -len(grouped_questions[refmark]), refmark),
        )
    else:
        ranked_refmarks = sorted(
            grouped_questions,
            key=lambda refmark: (-len(grouped_questions[refmark]), refmark),
        )
    if anchor_limit is not None:
        ranked_refmarks = ranked_refmarks[:anchor_limit]

    for refmark in ranked_refmarks:
        items = grouped_questions[refmark]
        if len(items) < min_questions_per_anchor:
            continue

        first_question, title, answer_start, answer_end, answer_text = items[0]
        if train_count < train_limit:
            train_examples.append(
                Example(
                    split="train",
                    prompt_style="squad_original",
                    question=first_question,
                    refmark=refmark,
                    title=title,
                    answer_text=answer_text,
                    answer_start=answer_start,
                    answer_end=answer_end,
                )
            )
            train_count += 1
            selected_refmarks.add(refmark)

        for question, title, answer_start, answer_end, answer_text in items[1:-1]:
            if train_count >= train_limit:
                break
            train_examples.append(
                Example(
                    split="train",
                    prompt_style="squad_original",
                    question=question,
                    refmark=refmark,
                    title=title,
                    answer_text=answer_text,
                    answer_start=answer_start,
                    answer_end=answer_end,
                )
            )
            train_count += 1
            selected_refmarks.add(refmark)

        if dev_count < dev_limit:
            valid_question, title, answer_start, answer_end, answer_text = items[-1]
            valid_examples.append(
                Example(
                    split="valid",
                    prompt_style="squad_original",
                    question=valid_question,
                    refmark=refmark,
                    title=title,
                    answer_text=answer_text,
                    answer_start=answer_start,
                    answer_end=answer_end,
                )
            )
            reformulated_examples.append(
                Example(
                    split="reformulated",
                    prompt_style="squad_reformulated_hard" if hard_reformulated else "squad_reformulated",
                    question=_reformulate_question(valid_question, title, hard=hard_reformulated),
                    refmark=refmark,
                    title=title,
                    answer_text=answer_text,
                    answer_start=answer_start,
                    answer_end=answer_end,
                )
            )
            dev_count += 1
            selected_refmarks.add(refmark)

        if train_count >= train_limit and dev_count >= dev_limit:
            break

    anchors = sorted(
        (anchor for anchor in anchor_map.values() if anchor.refmark in selected_refmarks),
        key=lambda anchor: anchor.refmark,
    )
    return CorpusBundle(
        anchors=anchors,
        train=train_examples,
        valid=valid_examples,
        reformulated=reformulated_examples,
    )


def prepare_squad_dataset(
    output_dir: Path,
    *,
    train_limit: int = 3000,
    dev_limit: int = 600,
    min_questions_per_anchor: int = 3,
    anchor_limit: int | None = None,
    sentence_window: int = 0,
    rank_mode: str = "frequency",
    hard_reformulated: bool = False,
) -> dict[str, str]:
    raw_dir = output_dir / "raw"
    train_path = download_json(SQUAD_TRAIN_URL, raw_dir / "train-v1.1.json")
    dev_path = download_json(SQUAD_DEV_URL, raw_dir / "dev-v1.1.json")
    train_payload = json.loads(train_path.read_text(encoding="utf-8"))
    dev_payload = json.loads(dev_path.read_text(encoding="utf-8"))
    bundle = build_squad_bundle(
        train_payload,
        dev_payload,
        train_limit=train_limit,
        dev_limit=dev_limit,
        min_questions_per_anchor=min_questions_per_anchor,
        anchor_limit=anchor_limit,
        sentence_window=sentence_window,
        rank_mode=rank_mode,
        hard_reformulated=hard_reformulated,
    )
    files = save_bundle(bundle, output_dir)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "source": "SQuAD v1.1",
            "train_limit": train_limit,
            "dev_limit": dev_limit,
            "min_questions_per_anchor": min_questions_per_anchor,
            "anchor_limit": anchor_limit,
            "sentence_window": sentence_window,
            "rank_mode": rank_mode,
            "hard_reformulated": hard_reformulated,
            "raw_files": {
                "train": str(train_path),
                "dev": str(dev_path),
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return files


def load_bundle_from_dir(data_dir: Path) -> CorpusBundle:
    def load_jsonl(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    anchors = [Anchor(**row) for row in load_jsonl(data_dir / "anchors.jsonl")]
    train = [Example(**row) for row in load_jsonl(data_dir / "train.jsonl")]
    valid = [Example(**row) for row in load_jsonl(data_dir / "valid.jsonl")]
    reformulated = [Example(**row) for row in load_jsonl(data_dir / "reformulated.jsonl")]
    return CorpusBundle(
        anchors=anchors,
        train=train,
        valid=valid,
        reformulated=reformulated,
    )
