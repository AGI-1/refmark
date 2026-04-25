"""Shared retrieval helpers for the RAG benchmark examples."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from refmark_train.synthetic import tokenize


ROOT = Path(__file__).resolve().parent
PUBLISH_ROOT = ROOT.parents[1]
DEFAULT_DATA_DIR = PUBLISH_ROOT / "refmark_train" / "data" / "documentation_full_paragraph_contextual_idf_lean2"
OUTPUT_DIR = ROOT / "output"


@dataclass(frozen=True)
class RetrievalUnit:
    unit_id: str
    text: str
    refs: tuple[str, ...]
    token_count: int


class BM25Index:
    def __init__(self, units: list[RetrievalUnit], *, k1: float = 1.5, b: float = 0.75):
        self.units = units
        self.k1 = k1
        self.b = b
        self.unit_tokens = [tokenize(unit.text) for unit in units]
        self.unit_counts = [Counter(tokens) for tokens in self.unit_tokens]
        self.unit_lengths = [len(tokens) for tokens in self.unit_tokens]
        self.avg_len = sum(self.unit_lengths) / max(len(self.unit_lengths), 1)
        doc_freq: Counter[str] = Counter()
        for counts in self.unit_counts:
            doc_freq.update(counts.keys())
        count = max(len(units), 1)
        self.idf = {
            token: math.log(((count - freq + 0.5) / (freq + 0.5)) + 1.0)
            for token, freq in doc_freq.items()
        }

    def search(self, query: str, *, top_k: int) -> list[tuple[RetrievalUnit, float]]:
        query_terms = set(tokenize(query))
        scored: list[tuple[RetrievalUnit, float]] = []
        for unit, counts, length in zip(self.units, self.unit_counts, self.unit_lengths, strict=True):
            score = 0.0
            for token in query_terms:
                tf = counts.get(token, 0)
                if tf <= 0:
                    continue
                norm = self.k1 * (1.0 - self.b + self.b * (length / max(self.avg_len, 1e-6)))
                score += self.idf.get(token, 0.0) * ((tf * (self.k1 + 1.0)) / (tf + norm))
            if score > 0.0:
                scored.append((unit, score))
        scored.sort(key=lambda item: (-item[1], item[0].unit_id))
        return scored[:top_k]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def anchor_units(anchors: list[dict]) -> list[RetrievalUnit]:
    return [
        RetrievalUnit(
            unit_id=str(anchor["refmark"]),
            text=str(anchor["text"]),
            refs=(str(anchor["refmark"]),),
            token_count=len(tokenize(str(anchor["text"]))),
        )
        for anchor in anchors
    ]


def enriched_anchor_units(anchors: list[dict], train: list[dict]) -> list[RetrievalUnit]:
    questions_by_ref: dict[str, list[str]] = defaultdict(list)
    for row in train:
        questions_by_ref[str(row["refmark"])].append(str(row["question"]))
    units: list[RetrievalUnit] = []
    for anchor in anchors:
        ref = str(anchor["refmark"])
        question_view = "\n".join(questions_by_ref.get(ref, [])[:8])
        text = f"{anchor['text']}\n{question_view}" if question_view else str(anchor["text"])
        units.append(
            RetrievalUnit(
                unit_id=ref,
                text=text,
                refs=(ref,),
                token_count=len(tokenize(str(anchor["text"]))),
            )
        )
    return units


def view_anchor_units(anchors: list[dict], views: list[dict], *, include_source: bool = True) -> list[RetrievalUnit]:
    view_by_ref = {str(row["refmark"]): row for row in views}
    units: list[RetrievalUnit] = []
    for anchor in anchors:
        ref = str(anchor["refmark"])
        row = view_by_ref.get(ref, {})
        parts = []
        if include_source:
            parts.append(str(anchor["text"]))
        parts.extend(str(item) for item in row.get("questions", []) if str(item).strip())
        if row.get("summary"):
            parts.append(str(row["summary"]))
        parts.extend(str(item) for item in row.get("keywords", []) if str(item).strip())
        text = "\n".join(parts) if parts else str(anchor["text"])
        units.append(
            RetrievalUnit(
                unit_id=ref,
                text=text,
                refs=(ref,),
                token_count=len(tokenize(str(anchor["text"]))),
            )
        )
    return units


def expanded_anchor_units(anchors: list[dict], *, margin: int) -> list[RetrievalUnit]:
    units: list[RetrievalUnit] = []
    for index, anchor in enumerate(anchors):
        start = max(0, index - margin)
        end = min(len(anchors), index + margin + 1)
        window = anchors[start:end]
        refs = tuple(str(item["refmark"]) for item in window)
        text = "\n".join(str(item["text"]) for item in window)
        units.append(
            RetrievalUnit(
                unit_id=str(anchor["refmark"]),
                text=str(anchor["text"]),
                refs=refs,
                token_count=sum(len(tokenize(str(item["text"]))) for item in window),
            )
        )
    return units


def fixed_window_units(anchors: list[dict], *, chunk_tokens: int, stride: int) -> list[RetrievalUnit]:
    units: list[RetrievalUnit] = []
    token_lengths = [len(tokenize(str(anchor["text"]))) for anchor in anchors]
    start = 0
    chunk_id = 1
    while start < len(anchors):
        total = 0
        end = start
        while end < len(anchors) and (total < chunk_tokens or end == start):
            total += token_lengths[end]
            end += 1
        window = anchors[start:end]
        units.append(
            RetrievalUnit(
                unit_id=f"C{chunk_id:04d}",
                text="\n".join(str(anchor["text"]) for anchor in window),
                refs=tuple(str(anchor["refmark"]) for anchor in window),
                token_count=total,
            )
        )
        chunk_id += 1
        step_tokens = 0
        next_start = start
        while next_start < len(anchors) and step_tokens < stride:
            step_tokens += token_lengths[next_start]
            next_start += 1
        start = max(next_start, start + 1)
    return units


def add_distractor_copies(anchors: list[dict], *, copies: int) -> list[dict]:
    if copies <= 0:
        return anchors
    expanded = list(anchors)
    for copy_idx in range(1, copies + 1):
        for anchor in anchors:
            clone = dict(anchor)
            clone["refmark"] = f"X{copy_idx}_{anchor['refmark']}"
            clone["text"] = f"{anchor['text']}\nDistractor copy {copy_idx}."
            expanded.append(clone)
    return expanded


def evaluate(index: BM25Index, examples: list[dict], *, top_ks: tuple[int, ...]) -> dict[str, object]:
    max_k = max(top_ks)
    hits = {k: 0 for k in top_ks}
    reciprocal_sum = 0.0
    token_cost = {k: 0 for k in top_ks}
    refs_returned = {k: 0 for k in top_ks}
    misses: list[dict[str, object]] = []
    for row in examples:
        gold = str(row["refmark"])
        results = index.search(str(row["question"]), top_k=max_k)
        rank = None
        for idx, (unit, _score) in enumerate(results, start=1):
            if gold in unit.refs:
                rank = idx
                break
        if rank is not None:
            reciprocal_sum += 1.0 / rank
        else:
            misses.append(
                {
                    "question": row["question"],
                    "gold": gold,
                    "top_units": [unit.unit_id for unit, _score in results[:3]],
                    "top_refs": [list(unit.refs)[:6] for unit, _score in results[:3]],
                }
            )
        for k in top_ks:
            selected = [unit for unit, _score in results[:k]]
            returned_refs = {ref for unit in selected for ref in unit.refs}
            if gold in returned_refs:
                hits[k] += 1
            token_cost[k] += sum(unit.token_count for unit in selected)
            refs_returned[k] += len(returned_refs)
    total = max(len(examples), 1)
    return {
        "unit_count": len(index.units),
        "avg_unit_tokens": round(sum(unit.token_count for unit in index.units) / max(len(index.units), 1), 2),
        "mrr": round(reciprocal_sum / total, 4),
        "hit_at_k": {str(k): round(hits[k] / total, 4) for k in top_ks},
        "avg_token_cost_at_k": {str(k): round(token_cost[k] / total, 2) for k in top_ks},
        "avg_refs_returned_at_k": {str(k): round(refs_returned[k] / total, 2) for k in top_ks},
        "sample_misses": misses[:8],
    }


def keywords_for(text: str, *, limit: int = 10) -> list[str]:
    stopwords = {
        "the", "and", "for", "that", "with", "this", "from", "into", "are", "was", "were",
        "has", "have", "not", "you", "can", "will", "may", "when", "where", "which", "what",
        "using", "used", "use", "all", "any", "but", "its", "their", "there", "then",
    }
    counts = Counter(token for token in tokenize(text) if len(token) > 2 and token not in stopwords)
    return [token for token, _count in counts.most_common(limit)]


def extractive_summary(text: str, *, max_words: int = 45) -> str:
    clean = " ".join(str(text).split())
    words = clean.split()
    if len(words) <= max_words:
        return clean
    return " ".join(words[:max_words])
