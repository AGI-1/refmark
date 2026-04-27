"""Train a tiny candidate reranker and measure repeated-refinement uncertainty."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.rag_retrieval_benchmark.common import (  # noqa: E402
    BM25Index,
    DEFAULT_DATA_DIR,
    OUTPUT_DIR,
    RetrievalUnit,
    anchor_units,
    enriched_anchor_units,
    load_jsonl,
    view_anchor_units,
)
from refmark_train.synthetic import tokenize  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark candidate refinement and no-answer uncertainty.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--negative-data-dir", action="append", default=[])
    parser.add_argument("--unit-mode", choices=["anchors", "enriched", "views"], default="enriched")
    parser.add_argument("--views-jsonl")
    parser.add_argument("--train-limit", type=int, default=8000)
    parser.add_argument("--eval-limit", type=int, default=3000)
    parser.add_argument("--negative-limit", type=int, default=3000)
    parser.add_argument("--sample-mode", choices=["first", "even", "random"], default="even")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--noise", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--output", default=str(OUTPUT_DIR / "refinement_uncertainty.json"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    data_dir = Path(args.data_dir)
    train_rows = _select_examples(load_jsonl(data_dir / "train.jsonl"), limit=args.train_limit, mode=args.sample_mode, seed=args.seed)
    valid_rows = _select_examples(load_jsonl(data_dir / "valid.jsonl"), limit=args.eval_limit, mode=args.sample_mode, seed=args.seed)
    anchors = load_jsonl(data_dir / "anchors.jsonl")
    train_for_units = load_jsonl(data_dir / "train.jsonl")
    units = _units_for_mode(args, anchors, train_for_units)
    index = BM25Index(units)

    train_examples = _candidate_examples(index, train_rows, candidate_k=args.candidate_k)
    model = _train_logistic(train_examples.features, train_examples.labels, epochs=args.epochs, lr=args.learning_rate, l2=args.l2)
    positives = _evaluate_answerable(
        index,
        valid_rows,
        model,
        candidate_k=args.candidate_k,
        samples=args.samples,
        temperature=args.temperature,
        noise=args.noise,
        rng=np_rng,
    )
    negatives = _evaluate_absence(
        index,
        _load_negative_rows(args, seed=args.seed),
        model,
        candidate_k=args.candidate_k,
        samples=args.samples,
        temperature=args.temperature,
        noise=args.noise,
        rng=np_rng,
    )

    payload = {
        "data_dir": str(data_dir),
        "unit_mode": args.unit_mode,
        "train_examples": len(train_rows),
        "eval_examples": len(valid_rows),
        "negative_examples": len(negatives["rows"]),
        "settings": {
            "candidate_k": args.candidate_k,
            "samples": args.samples,
            "temperature": args.temperature,
            "noise": args.noise,
            "sample_mode": args.sample_mode,
            "seed": args.seed,
        },
        "candidate_recall": positives["candidate_recall"],
        "single_refinement": positives["single"],
        "vote_refinement": positives["vote"],
        "citation_shape": positives["citation_shape"],
        "uncertainty_answerable": positives["uncertainty"],
        "uncertainty_negative": negatives["uncertainty"],
        "absence_detection": _absence_report(positives["rows"], negatives["rows"]),
        "interpretation": [
            "candidate_recall is the ceiling: if the gold ref is absent from top-k, refinement cannot recover it.",
            "single_refinement uses the reranker's top candidate once; vote_refinement samples repeated predictions and majority-votes.",
            "High vote agreement means stable citation; high entropy and low max probability are useful no-answer signals.",
            "negative examples are cross-domain questions with no gold ref in this corpus, so they approximate absence-of-citation behavior.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nWrote refinement uncertainty benchmark to {output}")


class CandidateExamples:
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = features
        self.labels = labels


class LogisticModel:
    def __init__(self, weights: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
        self.weights = weights
        self.mean = mean
        self.std = std

    def logits(self, features: np.ndarray) -> np.ndarray:
        normalized = (features - self.mean) / self.std
        with_bias = np.concatenate([normalized, np.ones((normalized.shape[0], 1), dtype=np.float32)], axis=1)
        return with_bias @ self.weights


def _units_for_mode(args, anchors: list[dict], train: list[dict]) -> list[RetrievalUnit]:
    if args.unit_mode == "anchors":
        return anchor_units(anchors)
    if args.unit_mode == "enriched":
        return enriched_anchor_units(anchors, train)
    if not args.views_jsonl:
        raise RuntimeError("--views-jsonl is required with --unit-mode views.")
    return view_anchor_units(anchors, load_jsonl(Path(args.views_jsonl)))


def _candidate_examples(index: BM25Index, rows: list[dict], *, candidate_k: int) -> CandidateExamples:
    features: list[list[float]] = []
    labels: list[float] = []
    for row in rows:
        query = str(row["question"])
        gold = str(row["refmark"])
        results = index.search(query, top_k=candidate_k)
        max_score = max((score for _unit, score in results), default=1.0)
        for rank, (unit, score) in enumerate(results, start=1):
            features.append(_features(query, unit, score=score, max_score=max_score, rank=rank, candidate_k=candidate_k))
            labels.append(1.0 if gold in unit.refs else 0.0)
    return CandidateExamples(np.array(features, dtype=np.float32), np.array(labels, dtype=np.float32))


def _features(query: str, unit: RetrievalUnit, *, score: float, max_score: float, rank: int, candidate_k: int) -> list[float]:
    query_tokens = tokenize(query)
    unit_tokens = tokenize(unit.text)
    q = Counter(query_tokens)
    u = Counter(unit_tokens)
    overlap = sum(min(q[token], u[token]) for token in q)
    unique_overlap = len(set(q) & set(u))
    query_len = max(len(query_tokens), 1)
    unit_len = max(len(unit_tokens), 1)
    return [
        score,
        score / max(max_score, 1e-6),
        1.0 / rank,
        (candidate_k - rank + 1) / candidate_k,
        overlap / query_len,
        unique_overlap / max(len(set(q)), 1),
        overlap / unit_len,
        math.log1p(unit.token_count),
        len(unit.refs),
    ]


def _train_logistic(features: np.ndarray, labels: np.ndarray, *, epochs: int, lr: float, l2: float) -> LogisticModel:
    mean = features.mean(axis=0, keepdims=True)
    std = np.maximum(features.std(axis=0, keepdims=True), 1e-6)
    x = (features - mean) / std
    x = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
    weights = np.zeros(x.shape[1], dtype=np.float32)
    pos_weight = float((labels == 0).sum() / max((labels == 1).sum(), 1))
    weights_per_row = np.where(labels > 0, pos_weight, 1.0).astype(np.float32)
    for _epoch in range(epochs):
        logits = x @ weights
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        grad = x.T @ ((probs - labels) * weights_per_row) / max(len(labels), 1)
        grad[:-1] += l2 * weights[:-1]
        weights -= lr * grad
    return LogisticModel(weights, mean, std)


def _evaluate_answerable(
    index: BM25Index,
    rows: list[dict],
    model: LogisticModel,
    *,
    candidate_k: int,
    samples: int,
    temperature: float,
    noise: float,
    rng: np.random.Generator,
) -> dict[str, object]:
    candidate_hits = 0
    single_hits = 0
    vote_hits = 0
    token_single = 0
    token_vote = 0
    eval_rows: list[dict] = []
    for row in rows:
        result = _refine_one(index, str(row["question"]), model, candidate_k=candidate_k, samples=samples, temperature=temperature, noise=noise, rng=rng)
        gold = str(row["refmark"])
        candidate_hit = gold in {ref for item in result["candidates"] for ref in item["refs"]}
        single_hit = gold in result["single_refs"]
        vote_hit = gold in result["vote_refs"]
        candidate_hits += int(candidate_hit)
        single_hits += int(single_hit)
        vote_hits += int(vote_hit)
        token_single += result["single_tokens"]
        token_vote += result["vote_tokens"]
        eval_rows.append({**result, "gold": gold, "candidate_hit": candidate_hit, "single_hit": single_hit, "vote_hit": vote_hit, "answerable": True})
    total = max(len(rows), 1)
    return {
        "candidate_recall": round(candidate_hits / total, 4),
        "single": {
            "hit@1": round(single_hits / total, 4),
            "avg_token_cost": round(token_single / total, 2),
            "undercitation_rate": round(1.0 - (single_hits / total), 4),
        },
        "vote": {
            "hit": round(vote_hits / total, 4),
            "avg_token_cost": round(token_vote / total, 2),
            "undercitation_rate": round(1.0 - (vote_hits / total), 4),
        },
        "citation_shape": _citation_shape(eval_rows),
        "uncertainty": _uncertainty_summary(eval_rows),
        "rows": eval_rows,
    }


def _evaluate_absence(
    index: BM25Index,
    rows: list[dict],
    model: LogisticModel,
    *,
    candidate_k: int,
    samples: int,
    temperature: float,
    noise: float,
    rng: np.random.Generator,
) -> dict[str, object]:
    eval_rows: list[dict] = []
    for row in rows:
        result = _refine_one(index, str(row["question"]), model, candidate_k=candidate_k, samples=samples, temperature=temperature, noise=noise, rng=rng)
        eval_rows.append({**result, "gold": None, "answerable": False})
    return {"uncertainty": _uncertainty_summary(eval_rows), "rows": eval_rows}


def _refine_one(
    index: BM25Index,
    query: str,
    model: LogisticModel,
    *,
    candidate_k: int,
    samples: int,
    temperature: float,
    noise: float,
    rng: np.random.Generator,
) -> dict:
    results = index.search(query, top_k=candidate_k)
    if not results:
        return _empty_refinement(query)
    max_score = max(score for _unit, score in results)
    features = np.array(
        [_features(query, unit, score=score, max_score=max_score, rank=rank, candidate_k=candidate_k) for rank, (unit, score) in enumerate(results, start=1)],
        dtype=np.float32,
    )
    logits = model.logits(features)
    probs = _softmax(logits)
    single_idx = int(np.argmax(logits))
    sampled = []
    for _ in range(samples):
        sample_logits = logits + rng.normal(0.0, noise, size=logits.shape)
        sample_probs = _softmax(sample_logits / max(temperature, 1e-6))
        sampled.append(int(rng.choice(len(results), p=sample_probs)))
    counts = Counter(sampled)
    vote_idx = min(counts, key=lambda idx: (-counts[idx], idx))
    vote_share = counts[vote_idx] / max(samples, 1)
    entropy = _entropy([count / max(samples, 1) for count in counts.values()])
    single_unit = results[single_idx][0]
    vote_unit = results[vote_idx][0]
    return {
        "question": query,
        "candidates": [
            {
                "unit_id": unit.unit_id,
                "refs": list(unit.refs),
                "bm25": round(score, 4),
                "rerank_prob": round(float(probs[idx]), 4),
                "tokens": unit.token_count,
            }
            for idx, (unit, score) in enumerate(results)
        ],
        "single_unit": single_unit.unit_id,
        "single_refs": list(single_unit.refs),
        "single_tokens": single_unit.token_count,
        "vote_unit": vote_unit.unit_id,
        "vote_refs": list(vote_unit.refs),
        "vote_tokens": vote_unit.token_count,
        "vote_share": round(vote_share, 4),
        "vote_entropy": round(entropy, 4),
        "max_prob": round(float(probs.max()), 4),
        "score_margin": round(float(probs.max() - np.partition(probs, -2)[-2]) if len(probs) > 1 else float(probs.max()), 4),
        "sampled_unique": len(counts),
    }


def _empty_refinement(query: str) -> dict:
    return {
        "question": query,
        "candidates": [],
        "single_unit": None,
        "single_refs": [],
        "single_tokens": 0,
        "vote_unit": None,
        "vote_refs": [],
        "vote_tokens": 0,
        "vote_share": 0.0,
        "vote_entropy": 1.0,
        "max_prob": 0.0,
        "score_margin": 0.0,
        "sampled_unique": 0,
    }


def _absence_report(answerable: list[dict], negative: list[dict]) -> dict[str, object]:
    rows = [(float(row["max_prob"]), 1) for row in answerable if row.get("candidate_hit")]
    rows.extend((float(row["max_prob"]), 0) for row in negative)
    entropy_rows = [(-float(row["vote_entropy"]), 1) for row in answerable if row.get("candidate_hit")]
    entropy_rows.extend((-float(row["vote_entropy"]), 0) for row in negative)
    share_rows = [(float(row["vote_share"]), 1) for row in answerable if row.get("candidate_hit")]
    share_rows.extend((float(row["vote_share"]), 0) for row in negative)
    return {
        "max_prob_auc": round(_auc(rows), 4),
        "low_entropy_auc": round(_auc(entropy_rows), 4),
        "vote_share_auc": round(_auc(share_rows), 4),
        "note": "AUC compares answerable positives whose gold was in candidates against cross-domain negatives.",
    }


def _uncertainty_summary(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {"avg_max_prob": 0.0, "avg_vote_share": 0.0, "avg_vote_entropy": 0.0, "avg_sampled_unique": 0.0}
    return {
        "avg_max_prob": round(sum(float(row["max_prob"]) for row in rows) / len(rows), 4),
        "avg_vote_share": round(sum(float(row["vote_share"]) for row in rows) / len(rows), 4),
        "avg_vote_entropy": round(sum(float(row["vote_entropy"]) for row in rows) / len(rows), 4),
        "avg_sampled_unique": round(sum(float(row["sampled_unique"]) for row in rows) / len(rows), 4),
    }


def _citation_shape(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {
            "avg_candidate_refs": 0.0,
            "avg_candidate_tokens": 0.0,
            "avg_single_refs": 0.0,
            "avg_vote_refs": 0.0,
        }
    candidate_refs = []
    candidate_tokens = []
    for row in rows:
        refs = {ref for candidate in row["candidates"] for ref in candidate["refs"]}
        candidate_refs.append(len(refs))
        candidate_tokens.append(sum(candidate.get("tokens", 0) for candidate in row["candidates"]))
    return {
        "avg_candidate_refs": round(sum(candidate_refs) / len(rows), 2),
        "avg_candidate_tokens": round(sum(candidate_tokens) / len(rows), 2),
        "avg_single_refs": round(sum(len(row["single_refs"]) for row in rows) / len(rows), 2),
        "avg_vote_refs": round(sum(len(row["vote_refs"]) for row in rows) / len(rows), 2),
    }


def _load_negative_rows(args, *, seed: int) -> list[dict]:
    rows: list[dict] = []
    for value in args.negative_data_dir:
        path = Path(value)
        if not path.exists():
            continue
        rows.extend(load_jsonl(path / "valid.jsonl"))
    return _select_examples(rows, limit=args.negative_limit, mode=args.sample_mode, seed=seed + 101)


def _select_examples(examples: list[dict], *, limit: int, mode: str, seed: int) -> list[dict]:
    if limit <= 0 or limit >= len(examples):
        return examples
    if mode == "first":
        return examples[:limit]
    if mode == "random":
        rng = random.Random(seed)
        indexes = sorted(rng.sample(range(len(examples)), limit))
        return [examples[index] for index in indexes]
    if limit == 1:
        return [examples[0]]
    indexes = sorted({round(index * (len(examples) - 1) / (limit - 1)) for index in range(limit)})
    return [examples[index] for index in indexes[:limit]]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(np.clip(shifted, -30, 30))
    return exp / max(float(exp.sum()), 1e-12)


def _entropy(probs: list[float]) -> float:
    if not probs:
        return 0.0
    value = -sum(prob * math.log(prob + 1e-12) for prob in probs)
    return value / math.log(max(len(probs), 2))


def _auc(scored_labels: list[tuple[float, int]]) -> float:
    positives = [score for score, label in scored_labels if label == 1]
    negatives = [score for score, label in scored_labels if label == 0]
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


if __name__ == "__main__":
    main()
