"""Train a small Refmark evidence resolver over BM25 candidate regions."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.portable_search_index.evaluate_real_corpus import _read_question_cache  # noqa: E402
from refmark.search_index import PortableBM25Index, load_search_index, tokenize  # noqa: E402

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover - exercised only in missing optional dep envs
    raise SystemExit("This training example requires torch.") from exc


@dataclass(frozen=True)
class CandidateExample:
    question_index: int
    query_ids: list[int]
    region_ids: list[int]
    features: list[float]
    label: float
    stable_ref: str
    bm25_score: float


@dataclass(frozen=True)
class ResolverQuestion:
    query: str
    doc_id: str
    gold_refs: list[str]
    source: str = "index_view"
    gold_mode: str = "index_view"


class PairResolver(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, layers: int, feature_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        input_dim = (embed_dim * 4) + feature_dim
        modules: list[nn.Module] = []
        current = input_dim
        for _ in range(layers):
            modules.append(nn.Linear(current, hidden_dim))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(0.08))
            current = hidden_dim
        modules.append(nn.Linear(current, 1))
        self.scorer = nn.Sequential(*modules)

    def forward(self, query_ids: torch.Tensor, region_ids: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        query_vec = _mean_pool(self.embedding(query_ids), query_ids.ne(0))
        region_vec = _mean_pool(self.embedding(region_ids), region_ids.ne(0))
        pair = torch.cat([query_vec, region_vec, torch.abs(query_vec - region_vec), query_vec * region_vec, features], dim=1)
        return self.scorer(pair).squeeze(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny Refmark region resolver on cached eval questions.")
    parser.add_argument("index", help="Portable search index JSON.")
    parser.add_argument("--question-cache", required=True, help="JSONL question cache produced by evaluate_real_corpus.py.")
    parser.add_argument("--output", default="examples/portable_search_index/output/refmark_resolver.pt")
    parser.add_argument("--report", default="examples/portable_search_index/output/refmark_resolver_report.json")
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--top-ks", default="1,3,5,10")
    parser.add_argument("--train-fraction", type=float, default=0.72)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--max-query-tokens", type=int, default=48)
    parser.add_argument("--max-region-tokens", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0015)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--pos-weight", type=float, default=8.0)
    parser.add_argument("--loss", choices=["bce", "pairwise", "hybrid"], default="hybrid")
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument(
        "--train-from-index-views",
        action="store_true",
        help="Add generated retrieval-view questions as training supervision for every region.",
    )
    parser.add_argument("--view-questions-per-region", type=int, default=4)
    parser.add_argument(
        "--vector-features",
        action="store_true",
        help="Add sparse TF-IDF cosine features for query vs. region text/metadata vectors.",
    )
    parser.add_argument(
        "--coarse-mode",
        choices=["anchor", "article"],
        default="anchor",
        help="Training/evaluation target. article treats any candidate in the gold document as positive.",
    )
    parser.add_argument(
        "--blend-alphas",
        default="0,0.15,0.25,0.35,0.5,0.65,0.8,1",
        help="Comma-separated neural-score weights to evaluate against normalized BM25. 1 is pure resolver.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    top_ks = tuple(int(part) for part in args.top_ks.split(",") if part.strip())
    blend_alphas = tuple(float(part) for part in args.blend_alphas.split(",") if part.strip())
    index = load_search_index(args.index)
    questions = _questions_for_index(index, Path(args.question_cache))
    train_questions, eval_questions = _split_questions(questions, train_fraction=args.train_fraction, seed=args.seed)
    if args.train_from_index_views:
        train_questions = list(train_questions) + _view_questions(index, limit_per_region=args.view_questions_per_region)
    vocab = _build_vocab(index, questions, vocab_size=args.vocab_size)
    idf = _build_idf(index)
    train_rows = _build_examples(index, train_questions, vocab=vocab, idf=idf, args=args)
    eval_rows = _build_examples(index, eval_questions, vocab=vocab, idf=idf, args=args)

    if not train_rows or not eval_rows:
        raise SystemExit("Not enough train/eval rows. Check question cache and candidate recall.")

    model = PairResolver(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        feature_dim=len(train_rows[0].features),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight))
    train_groups = _group_rows(train_rows)
    train_start = time.perf_counter()
    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        if args.loss == "bce":
            random.shuffle(train_rows)
            for batch in _batches(train_rows, args.batch_size):
                query_batch, region_batch, feature_batch, label_batch = _tensorize(batch)
                optimizer.zero_grad(set_to_none=True)
                logits = model(query_batch, region_batch, feature_batch)
                loss = loss_fn(logits, label_batch)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
        else:
            shuffled_groups = list(train_groups.values())
            random.shuffle(shuffled_groups)
            for group in shuffled_groups:
                query_batch, region_batch, feature_batch, label_batch = _tensorize(group)
                optimizer.zero_grad(set_to_none=True)
                logits = model(query_batch, region_batch, feature_batch)
                loss = _ranking_loss(logits, label_batch, margin=args.margin)
                if args.loss == "hybrid":
                    loss = loss + (0.25 * loss_fn(logits, label_batch))
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
        eval_metrics = evaluate_model(
            index,
            model,
            eval_rows,
            eval_questions,
            top_ks=top_ks,
            candidate_k=args.candidate_k,
            blend_alpha=1.0,
            coarse_mode=args.coarse_mode,
        )
        eval_score = eval_metrics["mrr"] + eval_metrics["resolver_hit_at_k"]["1"]
        if eval_score > best_score:
            best_score = eval_score
            best_state = copy.deepcopy(model.state_dict())
        history.append(
            {
                "epoch": epoch,
                "loss": round(sum(losses) / max(len(losses), 1), 6),
                "eval_hit_at_1": eval_metrics["resolver_hit_at_k"]["1"],
                "eval_mrr": eval_metrics["mrr"],
            }
        )
        print(json.dumps(history[-1]))

    train_seconds = time.perf_counter() - train_start
    model.load_state_dict(best_state)
    baseline = evaluate_baseline(index, eval_questions, top_ks=top_ks, candidate_k=args.candidate_k, coarse_mode=args.coarse_mode)
    resolver = evaluate_model(
        index,
        model,
        eval_rows,
        eval_questions,
        top_ks=top_ks,
        candidate_k=args.candidate_k,
        blend_alpha=1.0,
        coarse_mode=args.coarse_mode,
    )
    blends = {
        str(alpha): evaluate_model(
            index,
            model,
            eval_rows,
            eval_questions,
            top_ks=top_ks,
            candidate_k=args.candidate_k,
            blend_alpha=alpha,
            coarse_mode=args.coarse_mode,
        )
        for alpha in blend_alphas
    }
    best_blend_alpha, best_blend = max(blends.items(), key=lambda item: (item[1]["resolver_hit_at_k"]["1"], item[1]["mrr"]))
    parameter_count = sum(param.numel() for param in model.parameters())
    payload = {
        "schema": "refmark.resolver.v1",
        "settings": vars(args),
        "vocab": vocab,
        "model_state": model.state_dict(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    artifact_bytes = output.stat().st_size
    report = {
        "index": args.index,
        "question_cache": args.question_cache,
        "train_questions": len(train_questions),
        "eval_questions": len(eval_questions),
        "train_candidate_rows": len(train_rows),
        "eval_candidate_rows": len(eval_rows),
        "parameters": parameter_count,
        "artifact_bytes": artifact_bytes,
        "artifact_megabytes": round(artifact_bytes / 1_000_000, 4),
        "train_seconds": round(train_seconds, 3),
        "history": history,
        "best_epoch": max(history, key=lambda row: row["eval_mrr"] + row["eval_hit_at_1"])["epoch"],
        "baseline": baseline,
        "resolver": resolver,
        "blends": blends,
        "best_blend_alpha": best_blend_alpha,
        "best_blend": best_blend,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _questions_for_index(index: PortableBM25Index, cache_path: Path):
    region_refs = {region.stable_ref for region in index.regions}
    questions = [
        question
        for question in _read_question_cache(cache_path).values()
        if set(question.gold_refs).issubset(region_refs)
    ]
    questions.sort(key=lambda item: (item.stable_ref, item.query))
    return questions


def _split_questions(questions, *, train_fraction: float, seed: int):
    rng = random.Random(seed)
    shuffled = list(questions)
    rng.shuffle(shuffled)
    split = max(1, min(len(shuffled) - 1, int(len(shuffled) * train_fraction)))
    return shuffled[:split], shuffled[split:]


def _view_questions(index: PortableBM25Index, *, limit_per_region: int) -> list[ResolverQuestion]:
    questions: list[ResolverQuestion] = []
    for region in index.regions:
        for query in region.view.questions[:limit_per_region]:
            if query.strip():
                questions.append(
                    ResolverQuestion(
                        query=query,
                        doc_id=region.doc_id,
                        gold_refs=[region.stable_ref],
                        gold_mode="index_view",
                    )
                )
    return questions


def _build_vocab(index: PortableBM25Index, questions, *, vocab_size: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for region in index.regions:
        counts.update(tokenize(region.index_text(include_source=index.include_source)))
    for question in questions:
        counts.update(tokenize(question.query))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, _count in counts.most_common(max(vocab_size - len(vocab), 0)):
        vocab.setdefault(token, len(vocab))
    return vocab


def _build_examples(index: PortableBM25Index, questions, *, vocab: dict[str, int], idf: dict[str, float], args) -> list[CandidateExample]:
    rows: list[CandidateExample] = []
    for question_index, question in enumerate(questions):
        candidates = index._rank_regions(question.query, top_k=args.candidate_k)
        max_score = max((score for _region_index, score in candidates), default=1.0) or 1.0
        query_ids = _encode(question.query, vocab, args.max_query_tokens)
        for rank, (region_index, bm25_score) in enumerate(candidates, start=1):
            region = index.regions[region_index]
            metadata = "\n".join([region.view.summary, *region.view.questions, *region.view.keywords])
            rows.append(
                CandidateExample(
                    question_index=question_index,
                    query_ids=query_ids,
                    region_ids=_encode(region.index_text(include_source=index.include_source), vocab, args.max_region_tokens),
                    features=_features(
                        question.query,
                        region.text,
                        metadata,
                        bm25_score=bm25_score,
                        max_score=max_score,
                        rank=rank,
                        idf=idf,
                        vector_features=args.vector_features,
                    ),
                    label=1.0 if _is_positive(region, question, coarse_mode=args.coarse_mode) else 0.0,
                    stable_ref=region.stable_ref,
                    bm25_score=bm25_score,
                )
            )
    return rows


def _is_positive(region, question, *, coarse_mode: str) -> bool:
    if coarse_mode == "article":
        return region.doc_id == question.doc_id
    return region.stable_ref in set(question.gold_refs)


def _group_rows(rows: list[CandidateExample]) -> dict[int, list[CandidateExample]]:
    groups: dict[int, list[CandidateExample]] = {}
    for row in rows:
        groups.setdefault(row.question_index, []).append(row)
    return groups


def _encode(text: str, vocab: dict[str, int], max_tokens: int) -> list[int]:
    ids = [vocab.get(token, 1) for token in tokenize(text)[:max_tokens]]
    return ids or [1]


def _features(
    query: str,
    text: str,
    metadata: str,
    *,
    bm25_score: float,
    max_score: float,
    rank: int,
    idf: dict[str, float],
    vector_features: bool,
) -> list[float]:
    query_terms = set(tokenize(query))
    text_terms = set(tokenize(text))
    metadata_terms = set(tokenize(metadata))
    features = [
        bm25_score / max(max_score, 1e-6),
        1.0 / max(rank, 1),
        len(query_terms & text_terms) / max(len(query_terms), 1),
        len(query_terms & metadata_terms) / max(len(query_terms), 1),
        _bigram_overlap(query, text),
        _bigram_overlap(query, metadata),
    ]
    if vector_features:
        query_vector = _tfidf_vector(query, idf)
        text_vector = _tfidf_vector(text, idf)
        metadata_vector = _tfidf_vector(metadata, idf)
        combined_vector = _merge_vectors(text_vector, metadata_vector)
        features.extend(
            [
                _cosine(query_vector, text_vector),
                _cosine(query_vector, metadata_vector),
                _cosine(query_vector, combined_vector),
            ]
        )
    return features


def _build_idf(index: PortableBM25Index) -> dict[str, float]:
    doc_freq: Counter[str] = Counter()
    for region in index.regions:
        doc_freq.update(set(tokenize(region.index_text(include_source=index.include_source))))
    count = max(len(index.regions), 1)
    return {token: math.log(((count - freq + 0.5) / (freq + 0.5)) + 1.0) for token, freq in doc_freq.items()}


def _tfidf_vector(text: str, idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(tokenize(text))
    return {token: count * idf.get(token, 0.0) for token, count in counts.items()}


def _merge_vectors(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    merged = dict(left)
    for token, value in right.items():
        merged[token] = merged.get(token, 0.0) + value
    return merged


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(token, 0.0) for token, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def evaluate_baseline(index: PortableBM25Index, questions, *, top_ks: tuple[int, ...], candidate_k: int, coarse_mode: str) -> dict[str, object]:
    ranks = []
    doc_ranks = []
    mode_ranks: dict[str, list[int | None]] = {}
    mode_doc_ranks: dict[str, list[int | None]] = {}
    misses: list[dict[str, object]] = []
    for question in questions:
        candidates = index._rank_regions(question.query, top_k=candidate_k)
        rank = _gold_rank(index, candidates, question, coarse_mode=coarse_mode)
        doc_rank = _doc_rank(index, candidates, question.doc_id)
        ranks.append(rank)
        doc_ranks.append(doc_rank)
        mode = str(getattr(question, "gold_mode", "single"))
        mode_ranks.setdefault(mode, []).append(rank)
        mode_doc_ranks.setdefault(mode, []).append(doc_rank)
        if rank is None and len(misses) < 12:
            misses.append(
                {
                    "query": question.query,
                    "gold_refs": question.gold_refs,
                    "gold_mode": mode,
                    "top_refs": [index.regions[region_index].stable_ref for region_index, _score in candidates[:5]],
                }
            )
    summary = _summarize_ranks(ranks, doc_ranks, top_ks=top_ks)
    summary["by_gold_mode"] = _summarize_by_mode(mode_ranks, mode_doc_ranks, top_ks=top_ks)
    summary["sample_misses"] = misses
    return summary


def evaluate_model(
    index: PortableBM25Index,
    model: PairResolver,
    rows: list[CandidateExample],
    questions,
    *,
    top_ks: tuple[int, ...],
    candidate_k: int,
    blend_alpha: float,
    coarse_mode: str,
) -> dict[str, object]:
    model.eval()
    rows_by_question: dict[int, list[CandidateExample]] = {}
    for row in rows:
        rows_by_question.setdefault(row.question_index, []).append(row)
    ranks = []
    doc_ranks = []
    mode_ranks: dict[str, list[int | None]] = {}
    mode_doc_ranks: dict[str, list[int | None]] = {}
    inference_times = []
    misses: list[dict[str, object]] = []
    with torch.no_grad():
        for question_index, question in enumerate(questions):
            group = rows_by_question.get(question_index, [])
            if not group:
                ranks.append(None)
                doc_ranks.append(None)
                mode = str(getattr(question, "gold_mode", "single"))
                mode_ranks.setdefault(mode, []).append(None)
                mode_doc_ranks.setdefault(mode, []).append(None)
                continue
            start = time.perf_counter()
            query_batch, region_batch, feature_batch, _label_batch = _tensorize(group)
            logits = model(query_batch, region_batch, feature_batch)
            inference_times.append(time.perf_counter() - start)
            neural_scores = _minmax(logits.tolist())
            bm25_scores = _minmax([row.bm25_score for row in group])
            combined_scores = [
                (blend_alpha * neural_score) + ((1.0 - blend_alpha) * bm25_score)
                for neural_score, bm25_score in zip(neural_scores, bm25_scores, strict=True)
            ]
            scored = sorted(zip(group, combined_scores, strict=True), key=lambda item: (-item[1], item[0].stable_ref))
            stable_refs = [row.stable_ref for row, _score in scored]
            rank = _first_scored_rank(stable_refs, question, coarse_mode=coarse_mode)
            ranks.append(rank)
            doc_ids = [ref.split(":", 1)[0] for ref in stable_refs]
            try:
                doc_rank = doc_ids.index(question.doc_id) + 1
            except ValueError:
                doc_rank = None
            doc_ranks.append(doc_rank)
            mode = str(getattr(question, "gold_mode", "single"))
            mode_ranks.setdefault(mode, []).append(rank)
            mode_doc_ranks.setdefault(mode, []).append(doc_rank)
            if rank is None and len(misses) < 12:
                misses.append(
                    {
                        "query": question.query,
                        "gold_refs": question.gold_refs,
                        "gold_mode": mode,
                        "top_refs": stable_refs[:5],
                    }
                )
    summary = _summarize_ranks(ranks, doc_ranks, top_ks=top_ks)
    summary["avg_inference_ms_per_query"] = round((sum(inference_times) / max(len(inference_times), 1)) * 1000, 4)
    summary["blend_alpha"] = blend_alpha
    summary["by_gold_mode"] = _summarize_by_mode(mode_ranks, mode_doc_ranks, top_ks=top_ks)
    summary["sample_misses"] = misses
    return summary


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high <= low:
        return [0.0 for _value in values]
    return [(value - low) / (high - low) for value in values]


def _summarize_ranks(ranks: list[int | None], doc_ranks: list[int | None], *, top_ks: tuple[int, ...]) -> dict[str, object]:
    total = max(len(ranks), 1)
    reciprocal = sum((1.0 / rank) for rank in ranks if rank is not None)
    candidate_recall = sum(1 for rank in ranks if rank is not None) / total
    return {
        "resolver_hit_at_k": {str(k): round(sum(1 for rank in ranks if rank is not None and rank <= k) / total, 4) for k in top_ks},
        "doc_hit_at_k": {str(k): round(sum(1 for rank in doc_ranks if rank is not None and rank <= k) / total, 4) for k in top_ks},
        "mrr": round(reciprocal / total, 4),
        "candidate_recall_at_k": round(candidate_recall, 4),
    }


def _summarize_by_mode(
    mode_ranks: dict[str, list[int | None]],
    mode_doc_ranks: dict[str, list[int | None]],
    *,
    top_ks: tuple[int, ...],
) -> dict[str, object]:
    return {
        mode: {
            "count": len(ranks),
            **_summarize_ranks(ranks, mode_doc_ranks.get(mode, []), top_ks=top_ks),
        }
        for mode, ranks in sorted(mode_ranks.items())
    }


def _gold_rank(index: PortableBM25Index, candidates: list[tuple[int, float]], question, *, coarse_mode: str) -> int | None:
    for rank, (region_index, _score) in enumerate(candidates, start=1):
        if _is_positive(index.regions[region_index], question, coarse_mode=coarse_mode):
            return rank
    return None


def _first_rank(stable_refs: list[str], gold_refs: set[str]) -> int | None:
    for rank, stable_ref in enumerate(stable_refs, start=1):
        if stable_ref in gold_refs:
            return rank
    return None


def _first_scored_rank(stable_refs: list[str], question, *, coarse_mode: str) -> int | None:
    if coarse_mode == "article":
        for rank, stable_ref in enumerate(stable_refs, start=1):
            if stable_ref.split(":", 1)[0] == question.doc_id:
                return rank
        return None
    return _first_rank(stable_refs, set(question.gold_refs))


def _doc_rank(index: PortableBM25Index, candidates: list[tuple[int, float]], doc_id: str) -> int | None:
    for rank, (region_index, _score) in enumerate(candidates, start=1):
        if index.regions[region_index].doc_id == doc_id:
            return rank
    return None


def _tensorize(rows: list[CandidateExample]):
    max_query = max(len(row.query_ids) for row in rows)
    max_region = max(len(row.region_ids) for row in rows)
    query_batch = torch.tensor([_pad(row.query_ids, max_query) for row in rows], dtype=torch.long)
    region_batch = torch.tensor([_pad(row.region_ids, max_region) for row in rows], dtype=torch.long)
    feature_batch = torch.tensor([row.features for row in rows], dtype=torch.float32)
    label_batch = torch.tensor([row.label for row in rows], dtype=torch.float32)
    return query_batch, region_batch, feature_batch, label_batch


def _ranking_loss(logits: torch.Tensor, labels: torch.Tensor, *, margin: float) -> torch.Tensor:
    positive = logits[labels > 0.5]
    negative = logits[labels <= 0.5]
    if positive.numel() == 0 or negative.numel() == 0:
        return F.binary_cross_entropy_with_logits(logits, labels)
    best_positive = positive.max()
    return F.softplus(margin - best_positive + negative).mean()


def _pad(ids: list[int], length: int) -> list[int]:
    return ids + ([0] * (length - len(ids)))


def _mean_pool(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).float()
    summed = (values * weights).sum(dim=1)
    counts = weights.sum(dim=1).clamp_min(1.0)
    return summed / counts


def _batches(rows: list[CandidateExample], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def _bigram_overlap(query: str, text: str) -> float:
    query_tokens = tokenize(query)
    text_tokens = tokenize(text)
    if len(query_tokens) < 2 or len(text_tokens) < 2:
        return 0.0
    query_bigrams = set(zip(query_tokens, query_tokens[1:], strict=False))
    text_bigrams = set(zip(text_tokens, text_tokens[1:], strict=False))
    return len(query_bigrams & text_bigrams) / max(len(query_bigrams), 1)


if __name__ == "__main__":
    main()
