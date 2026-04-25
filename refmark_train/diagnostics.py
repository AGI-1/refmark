from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from refmark_train.models import BM25AnchorRetriever, Vocab
from refmark_train.real_corpus import load_bundle_from_dir
from refmark_train.synthetic import Example, tokenize


@dataclass(frozen=True)
class Bm25DiagnosticsConfig:
    data_dir: Path
    output_path: Path | None = None
    top_ks: tuple[int, ...] = (1, 3, 5, 10)
    window_widths: tuple[int, ...] = (1, 3, 5, 9)
    neighborhood_top_k: int = 3
    neighborhood_margins: tuple[int, ...] = (0, 1, 2, 4)
    distributed_top_k: int = 3
    distributed_min_gap: int = 10
    distributed_pairs: int = 20


def run_bm25_diagnostics(config: Bm25DiagnosticsConfig) -> tuple[dict, Path | None]:
    bundle = load_bundle_from_dir(config.data_dir)
    sorted_anchors = sorted(bundle.anchors, key=lambda anchor: anchor.refmark)
    anchor_to_index = {anchor.refmark: idx for idx, anchor in enumerate(sorted_anchors)}
    vocab = Vocab.build(
        [anchor.text for anchor in sorted_anchors]
        + [example.question for example in bundle.train]
        + [example.question for example in bundle.valid]
        + [example.question for example in bundle.reformulated]
    )
    label_index = dict(anchor_to_index)
    retriever = BM25AnchorRetriever(vocab)
    retriever.fit([anchor.text for anchor in sorted_anchors], [anchor.refmark for anchor in sorted_anchors])
    enriched_retriever = BM25AnchorRetriever(vocab)
    enriched_retriever.fit(
        _enriched_anchor_texts(sorted_anchors, bundle.train),
        [anchor.refmark for anchor in sorted_anchors],
    )

    report = {
        "dataset": {
            "data_dir": str(config.data_dir),
            "anchors": len(sorted_anchors),
            "train_examples": len(bundle.train),
            "valid_examples": len(bundle.valid),
            "reformulated_examples": len(bundle.reformulated),
        },
        "valid": _split_diagnostics(bundle.valid, retriever, label_index, sorted_anchors, config),
        "reformulated": _split_diagnostics(bundle.reformulated, retriever, label_index, sorted_anchors, config),
        "enriched_bm25": {
            "description": "BM25 over anchor text plus train questions with the same gold refmark.",
            "valid": _split_diagnostics(bundle.valid, enriched_retriever, label_index, sorted_anchors, config),
            "reformulated": _split_diagnostics(bundle.reformulated, enriched_retriever, label_index, sorted_anchors, config),
        },
        "distributed_anchor_candidates": _distributed_anchor_candidates(
            sorted_anchors,
            vocab,
            min_gap=config.distributed_min_gap,
            limit=config.distributed_pairs,
        ),
    }
    if config.output_path:
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        config.output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report, config.output_path


def format_bm25_diagnostics(report: dict, output_path: Path | None = None) -> str:
    lines = [
        "BM25 Diagnostics",
        "================",
        "",
        f"Dataset: {report['dataset']['data_dir']}",
        f"Anchors: {report['dataset']['anchors']}",
        f"Train/valid/reform: {report['dataset']['train_examples']} / {report['dataset']['valid_examples']} / {report['dataset']['reformulated_examples']}",
        "",
    ]
    for split in ["valid", "reformulated"]:
        metrics = report[split]
        lines.extend(
            [
                split.capitalize(),
                f"  MRR / mean rank:     {metrics['mrr']:.3f} / {metrics['mean_gold_rank']:.2f}",
                f"  lexical overlap:     {metrics['mean_query_gold_token_overlap']:.3f}",
                f"  score gap top1-top2: {metrics['mean_top1_top2_gap']:.3f}",
            ]
        )
        for key, value in metrics["top_k_hit"].items():
            lines.append(f"  top-{key} hit:          {value:.3f}")
        for width, value in metrics["center_window_cover"].items():
            breadth = metrics["center_window_anchor_count"][width]
            lines.append(f"  center window {width}:  {value:.3f} cover ({breadth:.2f} anchors)")
        for margin, value in metrics["neighborhood_cover"].items():
            breadth = metrics["neighborhood_anchor_count"][margin]
            lines.append(
                f"  top-{metrics['neighborhood_top_k']} +/-{margin}:      {value:.3f} cover ({breadth:.2f} anchors)"
            )
        lines.append(
            f"  distributed top-{metrics['distributed_top_k']}: {metrics['distributed_top_k_hit']:.3f} cover ({metrics['distributed_top_k']:.2f} anchors)"
        )
        lines.append("")
    enriched = report.get("enriched_bm25", {})
    if enriched:
        lines.append("Enriched BM25")
        lines.append("  indexes anchor text plus train questions for each refmark")
        for split in ["valid", "reformulated"]:
            metrics = enriched[split]
            lines.append(
                f"  {split}: top-1 {metrics['top_k_hit']['1']:.3f}, top-3 {metrics['top_k_hit']['3']:.3f}, MRR {metrics['mrr']:.3f}"
            )
        lines.append("")
    pairs = report.get("distributed_anchor_candidates", [])
    if pairs:
        lines.append("Far Similar Anchor Candidates")
        for pair in pairs[:8]:
            lines.append(
                f"  {pair['left_refmark']} <-> {pair['right_refmark']} gap={pair['gap']} sim={pair['similarity']:.3f}"
            )
        lines.append("")
    if output_path:
        lines.append(f"Artifact: {output_path}")
    return "\n".join(lines)


def _split_diagnostics(
    examples: list[Example],
    retriever: BM25AnchorRetriever,
    label_index: dict[str, int],
    sorted_anchors,
    config: Bm25DiagnosticsConfig,
) -> dict:
    if not examples:
        return {}
    logits = retriever.logits([example.question for example in examples], label_index)
    labels = np.array([label_index[example.refmark] for example in examples], dtype=np.int64)
    descending = np.argsort(logits, axis=1)[:, ::-1]
    ranks = np.zeros(len(labels), dtype=np.int64)
    for row, label in enumerate(labels):
        ranks[row] = int(np.where(descending[row] == label)[0][0]) + 1
    top_k_hit = {
        str(k): float(np.mean(ranks <= min(k, logits.shape[1])))
        for k in config.top_ks
    }
    center_window_cover: dict[str, float] = {}
    center_window_anchor_count: dict[str, float] = {}
    top1 = descending[:, 0]
    for width in config.window_widths:
        covers = []
        counts = []
        for center, label in zip(top1, labels, strict=True):
            start, end = _centered_window(int(center), width, len(sorted_anchors))
            covers.append(start <= int(label) <= end)
            counts.append(end - start + 1)
        center_window_cover[str(width)] = float(np.mean(covers))
        center_window_anchor_count[str(width)] = float(np.mean(counts))

    neighborhood_cover: dict[str, float] = {}
    neighborhood_anchor_count: dict[str, float] = {}
    top_n = min(config.neighborhood_top_k, logits.shape[1])
    for margin in config.neighborhood_margins:
        covers = []
        counts = []
        for row, label in enumerate(labels):
            selected: set[int] = set()
            for idx in descending[row, :top_n]:
                start = max(int(idx) - margin, 0)
                end = min(int(idx) + margin, len(sorted_anchors) - 1)
                selected.update(range(start, end + 1))
            covers.append(int(label) in selected)
            counts.append(len(selected))
        neighborhood_cover[str(margin)] = float(np.mean(covers))
        neighborhood_anchor_count[str(margin)] = float(np.mean(counts))

    query_gold_overlap = [
        _token_overlap(example.question, sorted_anchors[int(label)].text)
        for example, label in zip(examples, labels, strict=True)
    ]
    top1_top2_gap = logits[np.arange(len(examples)), descending[:, 0]] - logits[np.arange(len(examples)), descending[:, 1]]
    return {
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_gold_rank": float(np.mean(ranks)),
        "top_k_hit": top_k_hit,
        "center_window_cover": center_window_cover,
        "center_window_anchor_count": center_window_anchor_count,
        "neighborhood_top_k": top_n,
        "neighborhood_cover": neighborhood_cover,
        "neighborhood_anchor_count": neighborhood_anchor_count,
        "distributed_top_k": min(config.distributed_top_k, logits.shape[1]),
        "distributed_top_k_hit": float(np.mean(ranks <= min(config.distributed_top_k, logits.shape[1]))),
        "mean_query_gold_token_overlap": float(np.mean(query_gold_overlap)),
        "mean_top1_top2_gap": float(np.mean(top1_top2_gap)),
    }


def _centered_window(center: int, width: int, count: int) -> tuple[int, int]:
    width = max(width, 1)
    left = (width - 1) // 2
    right = width // 2
    return max(center - left, 0), min(center + right, count - 1)


def _token_overlap(query: str, text: str) -> float:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    text_tokens = set(tokenize(text))
    return len(query_tokens & text_tokens) / len(query_tokens)


def _distributed_anchor_candidates(anchors, vocab: Vocab, *, min_gap: int, limit: int) -> list[dict]:
    matrix = _tfidf_matrix([anchor.text for anchor in anchors], vocab)
    similarity = matrix @ matrix.T
    pairs: list[dict] = []
    for left in range(len(anchors)):
        for right in range(left + min_gap, len(anchors)):
            sim = float(similarity[left, right])
            if sim <= 0.0:
                continue
            pairs.append(
                {
                    "left_refmark": anchors[left].refmark,
                    "right_refmark": anchors[right].refmark,
                    "gap": right - left,
                    "similarity": sim,
                    "left_preview": _preview(anchors[left].text),
                    "right_preview": _preview(anchors[right].text),
                }
            )
    pairs.sort(key=lambda row: row["similarity"], reverse=True)
    return pairs[:limit]


def _enriched_anchor_texts(anchors, train_examples: list[Example]) -> list[str]:
    questions_by_refmark: dict[str, list[str]] = {anchor.refmark: [] for anchor in anchors}
    for example in train_examples:
        questions_by_refmark.setdefault(example.refmark, []).append(example.question)
    return [
        anchor.text + "\n" + "\n".join(questions_by_refmark.get(anchor.refmark, []))
        for anchor in anchors
    ]


def _tfidf_matrix(texts: Iterable[str], vocab: Vocab) -> np.ndarray:
    encoded = [vocab.encode(text) for text in texts]
    counts = np.zeros((len(encoded), len(vocab.id_to_token)), dtype=np.float32)
    for row, token_ids in enumerate(encoded):
        for token_id in token_ids:
            counts[row, token_id] += 1.0
    doc_freq = (counts > 0).sum(axis=0)
    idf = np.log((1.0 + len(encoded)) / (1.0 + doc_freq)) + 1.0
    weighted = counts * idf
    norms = np.linalg.norm(weighted, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return weighted / norms


def _preview(text: str, max_chars: int = 180) -> str:
    compact = " ".join(text.split())
    return compact[:max_chars]
