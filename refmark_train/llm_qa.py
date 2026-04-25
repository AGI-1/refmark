from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
import re
from typing import Any

import requests

from refmark_train.real_corpus import load_bundle_from_dir
from refmark_train.synthetic import Anchor, CorpusBundle, Example, save_bundle


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass(frozen=True)
class LlmQaConfig:
    data_dir: Path
    output_dir: Path
    model: str = "google/gemma-4-31b-it"
    reviewer_model: str | None = None
    endpoint: str = OPENROUTER_URL
    max_anchors: int = 24
    train_per_anchor: int = 3
    valid_per_anchor: int = 1
    reform_per_anchor: int = 2
    seed: int = 13
    temperature: float = 0.4
    max_tokens: int = 900
    api_key_env: str = "OPENROUTER_API_KEY"
    resume: bool = False
    question_style: str = "natural"
def generate_llm_qa_dataset(config: LlmQaConfig) -> tuple[dict[str, str], Path]:
    bundle = load_bundle_from_dir(config.data_dir)
    rng = random.Random(config.seed)
    anchors = _select_anchors(bundle.anchors, config.max_anchors, rng)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = config.output_dir / "llm_generation.jsonl"
    raw_rows = _load_raw_rows(raw_path) if config.resume else []
    completed = _completed_refmarks(raw_rows, config)

    with raw_path.open("a" if config.resume else "w", encoding="utf-8") as raw_file:
        for idx, anchor in enumerate(anchors):
            if anchor.refmark in completed:
                continue
            distractors = _nearby_distractors(anchors, idx)
            generated_rows = _generate_anchor_questions(config, anchor, distractors)
            anchor_rows = _review_generated_rows(config, anchor, distractors, generated_rows)
            anchor_rows.append({"anchor": anchor.refmark, "complete": True})
            for raw_row in anchor_rows:
                raw_file.write(json.dumps(raw_row, ensure_ascii=False) + "\n")
            raw_file.flush()
            raw_rows.extend(anchor_rows)

    train, valid, reformulated = _examples_from_raw_rows(config, anchors, raw_rows)
    output_bundle = CorpusBundle(anchors=anchors, train=train, valid=valid, reformulated=reformulated)
    files = save_bundle(output_bundle, config.output_dir)
    manifest_path = config.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "source_data_dir": str(config.data_dir),
            "llm_qa_config": {
                **asdict(config),
                "data_dir": str(config.data_dir),
                "output_dir": str(config.output_dir),
            },
            "raw_generation": str(raw_path),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    files["raw_generation"] = str(raw_path)
    return files, config.output_dir
def _select_anchors(anchors: list[Anchor], max_anchors: int, rng: random.Random) -> list[Anchor]:
    candidates = [
        anchor
        for anchor in anchors
        if anchor.region_start is not None
        and anchor.region_end is not None
        and 12 <= len(anchor.text.split()) <= 120
        and not _looks_like_boilerplate(anchor.text)
    ]
    rng.shuffle(candidates)
    selected = sorted(candidates[:max_anchors], key=lambda anchor: anchor.refmark)
    return selected


def _looks_like_boilerplate(text: str) -> bool:
    lowered = text.lower()
    boilerplate_terms = ["skip to main content", "table of contents", "source url:", ".gov means", "theme auto"]
    if any(term in lowered for term in boilerplate_terms):
        return True
def _load_raw_rows(raw_path: Path) -> list[dict[str, Any]]:
    if not raw_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _completed_refmarks(raw_rows: list[dict[str, Any]], config: LlmQaConfig) -> set[str]:
    completed = {str(row.get("anchor")) for row in raw_rows if row.get("complete") is True}
    counts: dict[str, dict[str, int]] = {}
    for raw_row in raw_rows:
        if not raw_row.get("accepted"):
            continue
        refmark = str(raw_row.get("anchor", ""))
        row = raw_row.get("row", {})
        if not refmark or not isinstance(row, dict):
            continue
        split = str(row.get("split", "")).strip().lower()
        if split not in {"train", "valid", "reformulated"}:
            continue
        counts.setdefault(refmark, {"train": 0, "valid": 0, "reformulated": 0})[split] += 1
    for refmark, split_counts in counts.items():
        if (
            split_counts["train"] >= config.train_per_anchor
            and split_counts["valid"] >= config.valid_per_anchor
            and split_counts["reformulated"] >= config.reform_per_anchor
        ):
            completed.add(refmark)
    return completed


def _review_generated_rows(
    config: LlmQaConfig,
    anchor: Anchor,
    distractors: list[Anchor],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    raw_rows: list[dict[str, Any]] = []
    for row in rows:
        split = str(row.get("split", "")).strip().lower()
        question = str(row.get("question", "")).strip()
        if split not in {"train", "valid", "reformulated"} or not question:
            continue
        review = None
        if config.reviewer_model:
            review = _review_question(config, anchor, distractors, question)
            if not review.get("accept", False):
                raw_rows.append({"anchor": anchor.refmark, "row": row, "review": review, "accepted": False})
                continue
        raw_rows.append({"anchor": anchor.refmark, "row": row, "review": review, "accepted": True})
    return raw_rows


def _examples_from_raw_rows(
    config: LlmQaConfig,
    anchors: list[Anchor],
    raw_rows: list[dict[str, Any]],
) -> tuple[list[Example], list[Example], list[Example]]:
    anchor_by_refmark = {anchor.refmark: anchor for anchor in anchors}
    grouped: dict[str, list[dict[str, str]]] = {anchor.refmark: [] for anchor in anchors}
    for raw_row in raw_rows:
        if not raw_row.get("accepted"):
            continue
        refmark = str(raw_row.get("anchor", ""))
        row = raw_row.get("row", {})
        if refmark not in grouped or not isinstance(row, dict):
            continue
        split = str(row.get("split", "")).strip().lower()
        question = str(row.get("question", "")).strip()
        if split in {"train", "valid", "reformulated"} and question:
            grouped[refmark].append({"split": split, "question": question})

    train: list[Example] = []
    valid: list[Example] = []
    reformulated: list[Example] = []
    for refmark, rows in grouped.items():
        anchor = anchor_by_refmark[refmark]
        for split, limit, target in [
            ("train", config.train_per_anchor, train),
            ("valid", config.valid_per_anchor, valid),
            ("reformulated", config.reform_per_anchor, reformulated),
        ]:
            split_rows = [row for row in rows if row["split"] == split][:limit]
            for row in split_rows:
                target.append(_example_for_anchor(anchor, row["question"], split, config.model))
    return train, valid, reformulated

    url_count = lowered.count("http://") + lowered.count("https://")
    return url_count >= 2


def _nearby_distractors(anchors: list[Anchor], idx: int, radius: int = 2) -> list[Anchor]:
    out: list[Anchor] = []
    for pos in range(max(0, idx - radius), min(len(anchors), idx + radius + 1)):
        if pos != idx:
            out.append(anchors[pos])
    return out


def _example_for_anchor(anchor: Anchor, question: str, split: str, model: str) -> Example:
    return Example(
        split=split,
        prompt_style=f"llm_{model}",
        question=question,
        refmark=anchor.refmark,
        title=anchor.title,
        answer_text=anchor.text,
        answer_start=anchor.region_start,
        answer_end=anchor.region_end,
    )


def _generate_anchor_questions(config: LlmQaConfig, anchor: Anchor, distractors: list[Anchor]) -> list[dict[str, Any]]:
    total = config.train_per_anchor + config.valid_per_anchor + config.reform_per_anchor
    style_rules = _question_style_rules(config.question_style)
    prompt = f"""
You create citation-localization benchmark questions.

Target anchor:
{_anchor_block(anchor)}

Nearby distractor anchors:
{_anchors_block(distractors)}

Create {total} natural user questions that should cite the target anchor, not the distractors.
Use these exact split counts:
- train: {config.train_per_anchor}
- valid: {config.valid_per_anchor}
- reformulated: {config.reform_per_anchor}

Rules:
- Questions must be answerable from the target anchor alone.
- Do not mention refmark ids, anchors, markers, citation spans, or the word "document".
- Avoid template repetition.
- Prefer realistic wording a user might ask.
{style_rules}
- Output only JSON with this schema:
{{"questions":[{{"split":"train|valid|reformulated","question":"..."}}]}}
""".strip()
    payload = _chat(config, config.model, prompt)
    data = _parse_json_object(payload)
    questions = data.get("questions", [])
    if not isinstance(questions, list):
        return []
    return [row for row in questions if isinstance(row, dict)]


def _question_style_rules(question_style: str) -> str:
    if question_style == "hard":
        return "\n".join(
            [
                "- Make at least half of the questions lexically indirect: use synonyms, scenarios, or user intent instead of copying exact target wording.",
                "- Avoid rare proper nouns unless they are essential to identify the answer.",
                "- Include some questions that are short and underspecified but still uniquely answerable from the target anchor.",
                "- Do not make the question impossible; the target anchor must remain the best citation.",
            ]
        )
    return "- Keep the questions natural and direct."


def _review_question(config: LlmQaConfig, anchor: Anchor, distractors: list[Anchor], question: str) -> dict[str, Any]:
    prompt = f"""
Review whether a generated question has the correct citation target.

Question:
{question}

Target anchor:
{_anchor_block(anchor)}

Nearby distractor anchors:
{_anchors_block(distractors)}

Return JSON only:
{{"accept":true|false,"reason":"short reason"}}

Accept only if the target anchor is a clearly sufficient citation and the distractors are not better.
""".strip()
    assert config.reviewer_model is not None
    payload = _chat(config, config.reviewer_model, prompt)
    data = _parse_json_object(payload)
    return data if isinstance(data, dict) else {"accept": False, "reason": "invalid reviewer json"}


def _anchor_block(anchor: Anchor) -> str:
    text = re.sub(r"\s+", " ", anchor.text).strip()
    return f"[{anchor.refmark}] {text[:1000]}"


def _anchors_block(anchors: list[Anchor]) -> str:
    if not anchors:
        return "(none)"
    return "\n".join(_anchor_block(anchor) for anchor in anchors)


def _chat(config: LlmQaConfig, model: str, prompt: str) -> str:
    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RuntimeError(f"missing API key environment variable: {config.api_key_env}")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://local.refmark-train",
        "X-Title": "refmark_train",
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You produce strict JSON for citation benchmark generation."},
            {"role": "user", "content": prompt},
        ],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "response_format": {"type": "json_object"},
    }
    response = requests.post(config.endpoint, headers=headers, json=body, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
