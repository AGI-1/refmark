"""Train a text-only BGB reranker from embedding-teacher scores.

This experiment asks whether the strong Qwen3 embedding signal can be used only
as offline supervision. At runtime the model receives a query, a lexical
candidate article, and cheap lexical/BM25 features. It does not receive query
embeddings. During training, cached query and article embeddings define soft
targets so the small reranker can try to imitate the embedding teacher inside a
BM25 candidate pool while still boosting the gold Refmark article.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.bgb_browser_search.adapt_bgb_static_views import article_id_from_ref, article_regions, summarize_ranks  # noqa: E402
from examples.bgb_browser_search.train_bgb_article_candidate_generator import CandidateQuestion, load_split_questions, summarize_groups  # noqa: E402
from refmark.search_index import PortableBM25Index, SearchHit, SearchRegion, load_search_index, tokenize  # noqa: E402

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This training example requires torch.") from exc


@dataclass(frozen=True)
class PairRow:
    question_index: int
    query_ids: list[int]
    article_ids: list[int]
    features: list[float]
    target: float
    stable_ref: str
    bm25_score: float


class TeacherReranker(nn.Module):
    def __init__(self, vocab_size: int, *, embed_dim: int, hidden_dim: int, layers: int, feature_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        input_dim = (embed_dim * 4) + feature_dim
        modules: list[nn.Module] = []
        current = input_dim
        for _ in range(layers):
            modules.extend([nn.Linear(current, hidden_dim), nn.ReLU(), nn.Dropout(0.08)])
            current = hidden_dim
        modules.append(nn.Linear(current, 1))
        self.scorer = nn.Sequential(*modules)

    def forward(self, query_ids: torch.Tensor, article_ids: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        query_vec = mean_pool(self.embedding(query_ids), query_ids.ne(0))
        article_vec = mean_pool(self.embedding(article_ids), article_ids.ne(0))
        pair = torch.cat(
            [query_vec, article_vec, torch.abs(query_vec - article_vec), query_vec * article_vec, features],
            dim=1,
        )
        return self.scorer(pair).squeeze(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a BGB reranker from Qwen3 embedding-teacher scores.")
    parser.add_argument("--index", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_intent_signatures_3cycle_index.json")
    parser.add_argument("--source-index", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_openrouter_index.json")
    parser.add_argument("--region-embedding-cache", default="examples/bgb_browser_search/output_scratch_multi_full/embedding_cache_qwen3_8b.jsonl")
    parser.add_argument("--query-cache", action="append", required=True)
    parser.add_argument("--stress-report", action="append", required=True)
    parser.add_argument("--embedding-model", default="qwen/qwen3-embedding-8b")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--candidate-k", type=int, default=120)
    parser.add_argument("--top-ks", default="1,3,5,10,20,50")
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=3535)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--max-query-tokens", type=int, default=72)
    parser.add_argument("--max-article-tokens", type=int, default=420)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--teacher-weight", type=float, default=0.65)
    parser.add_argument("--gold-boost", type=float, default=0.35)
    parser.add_argument("--margin", type=float, default=0.18)
    parser.add_argument("--blend-alphas", default="0,0.1,0.2,0.35,0.5,0.65,0.8,1")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    top_ks = tuple(int(part) for part in args.top_ks.split(",") if part.strip())
    blend_alphas = tuple(float(part) for part in args.blend_alphas.split(",") if part.strip())

    index = load_search_index(args.index)
    candidate_index = PortableBM25Index(index.regions, include_source=True)
    source_regions = load_search_index(args.source_index).regions
    article_vectors = load_article_vectors(
        Path(args.region_embedding_cache),
        source_regions,
        model=args.embedding_model,
    )
    query_vectors = load_query_vectors([Path(path) for path in args.query_cache], model=args.embedding_model)

    train_rows, eval_rows = load_split_questions(args.stress_report, train_fraction=args.train_fraction, seed=args.seed)
    train_rows = [row for row in train_rows if row.query in query_vectors]
    eval_rows = [row for row in eval_rows if row.query in query_vectors]
    vocab = build_vocab(index.regions, train_rows, vocab_size=args.vocab_size)

    train_pairs = build_pairs(
        candidate_index,
        train_rows,
        query_vectors,
        article_vectors,
        vocab,
        max_query_tokens=args.max_query_tokens,
        max_article_tokens=args.max_article_tokens,
        candidate_k=args.candidate_k,
        teacher_weight=args.teacher_weight,
        gold_boost=args.gold_boost,
        include_gold=True,
    )
    eval_pairs = build_pairs(
        candidate_index,
        eval_rows,
        query_vectors,
        article_vectors,
        vocab,
        max_query_tokens=args.max_query_tokens,
        max_article_tokens=args.max_article_tokens,
        candidate_k=args.candidate_k,
        teacher_weight=args.teacher_weight,
        gold_boost=args.gold_boost,
        include_gold=False,
    )
    if not train_pairs or not eval_pairs:
        raise SystemExit("Not enough pair rows. Check caches and candidate recall.")

    model = TeacherReranker(
        len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        feature_dim=len(train_pairs[0].features),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_groups = group_pairs(train_pairs)
    history: list[dict[str, object]] = []
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_score = -1.0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        groups = list(train_groups.values())
        random.shuffle(groups)
        for group in groups:
            query_batch, article_batch, feature_batch, target_batch = tensorize(group)
            optimizer.zero_grad(set_to_none=True)
            logits = model(query_batch, article_batch, feature_batch)
            loss = F.mse_loss(torch.sigmoid(logits), target_batch)
            loss = loss + ranking_loss(logits, target_batch, margin=args.margin)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = evaluate_model(
            candidate_index,
            model,
            eval_pairs,
            eval_rows,
            top_ks=top_ks,
            candidate_k=args.candidate_k,
            blend_alpha=1.0,
        )
        score = float(metrics["article_hit_at_k"]["hit_at_k"].get("10", 0.0)) + float(metrics["article_hit_at_k"]["mrr"])
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        row = {
            "epoch": epoch,
            "loss": round(sum(losses) / max(len(losses), 1), 6),
            "hit_at_1": metrics["article_hit_at_k"]["hit_at_k"].get("1", 0.0),
            "hit_at_10": metrics["article_hit_at_k"]["hit_at_k"].get("10", 0.0),
            "mrr": metrics["article_hit_at_k"]["mrr"],
        }
        history.append(row)
        print(json.dumps(row))

    train_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    baseline = evaluate_baseline(candidate_index, eval_rows, top_ks=top_ks, candidate_k=args.candidate_k)
    resolver = evaluate_model(candidate_index, model, eval_pairs, eval_rows, top_ks=top_ks, candidate_k=args.candidate_k, blend_alpha=1.0)
    blends = {
        str(alpha): evaluate_model(
            candidate_index,
            model,
            eval_pairs,
            eval_rows,
            top_ks=top_ks,
            candidate_k=args.candidate_k,
            blend_alpha=alpha,
        )
        for alpha in blend_alphas
    }
    best_blend_alpha, best_blend = max(
        blends.items(),
        key=lambda item: (
            float(item[1]["article_hit_at_k"]["hit_at_k"].get("10", 0.0)),
            float(item[1]["article_hit_at_k"]["mrr"]),
        ),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "refmark.bgb_embedding_teacher_reranker.v1",
        "settings": vars(args),
        "vocab": vocab,
        "model_state": model.state_dict(),
    }
    torch.save(payload, output)
    artifact_bytes = output.stat().st_size
    report = {
        "schema": "refmark.bgb_embedding_teacher_reranker_report.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": vars(args),
        "train_questions": len(train_rows),
        "eval_questions": len(eval_rows),
        "train_candidate_rows": len(train_pairs),
        "eval_candidate_rows": len(eval_pairs),
        "parameters": sum(param.numel() for param in model.parameters()),
        "artifact_bytes": artifact_bytes,
        "artifact_megabytes": round(artifact_bytes / 1_000_000, 4),
        "train_seconds": round(train_seconds, 3),
        "history": history,
        "best_epoch": max(history, key=lambda row: float(row["hit_at_10"]) + float(row["mrr"]))["epoch"],
        "baseline": baseline,
        "teacher_reranker": resolver,
        "blends": blends,
        "best_blend_alpha": best_blend_alpha,
        "best_blend": best_blend,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def load_query_vectors(paths: list[Path], *, model: str) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for path in paths:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("model") != model:
                    continue
                vector = torch.tensor([float(value) for value in row["embedding"]], dtype=torch.float32)
                output[str(row["query"])] = F.normalize(vector, dim=0)
    return output


def load_article_vectors(path: Path, regions: list[SearchRegion], *, model: str) -> dict[str, torch.Tensor]:
    wanted: dict[str, str] = {region.stable_ref: article_id_from_ref(region.stable_ref) for region in regions}
    buckets: dict[str, list[torch.Tensor]] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            stable_ref = str(row.get("stable_ref", ""))
            if row.get("model") != model or row.get("input_type") != "search_document" or stable_ref not in wanted:
                continue
            vector = torch.tensor([float(value) for value in row["embedding"]], dtype=torch.float32)
            buckets.setdefault(wanted[stable_ref], []).append(F.normalize(vector, dim=0))
    return {article_ref: F.normalize(torch.stack(vectors).mean(dim=0), dim=0) for article_ref, vectors in buckets.items()}


def build_vocab(regions: list[SearchRegion], rows: list[CandidateQuestion], *, vocab_size: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(tokenize(row.query))
    for region in regions:
        counts.update(tokenize(article_text(region)))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, _count in counts.most_common(max(vocab_size - len(vocab), 0)):
        vocab.setdefault(token, len(vocab))
    return vocab


def build_pairs(
    index: PortableBM25Index,
    rows: list[CandidateQuestion],
    query_vectors: dict[str, torch.Tensor],
    article_vectors: dict[str, torch.Tensor],
    vocab: dict[str, int],
    *,
    max_query_tokens: int,
    max_article_tokens: int,
    candidate_k: int,
    teacher_weight: float,
    gold_boost: float,
    include_gold: bool,
) -> list[PairRow]:
    region_by_ref = {region.stable_ref: region for region in index.regions}
    output: list[PairRow] = []
    for question_index, row in enumerate(rows):
        query_vector = query_vectors[row.query]
        hits = index.search(row.query, top_k=candidate_k)
        hit_refs = [hit.stable_ref for hit in hits]
        if include_gold and row.article_ref not in hit_refs and row.article_ref in region_by_ref:
            doc_id, region_id = row.article_ref.split(":", 1)
            region = region_by_ref[row.article_ref]
            hits.append(
                SearchHit(
                    rank=len(hits) + 1,
                    score=0.0,
                    doc_id=doc_id,
                    region_id=region_id,
                    stable_ref=row.article_ref,
                    text=region.text,
                    summary=region.view.summary,
                    source_path=region.source_path,
                    context_refs=[],
                )
            )
        max_score = max((hit.score for hit in hits), default=1.0) or 1.0
        query_tokens = set(tokenize(row.query))
        for rank, hit in enumerate(hits, start=1):
            region = region_by_ref.get(hit.stable_ref)
            if region is None:
                continue
            article_ref = article_id_from_ref(hit.stable_ref)
            article_vector = article_vectors.get(article_ref)
            teacher_score = float(torch.dot(query_vector, article_vector)) if article_vector is not None else 0.0
            teacher_target = max(0.0, min(1.0, (teacher_score + 1.0) / 2.0))
            is_gold = article_ref == article_id_from_ref(row.article_ref)
            target = 1.0 if is_gold else min(0.95, teacher_weight * teacher_target)
            article_tokens = set(tokenize(article_text(region)))
            overlap = len(query_tokens & article_tokens)
            union = len(query_tokens | article_tokens) or 1
            features = [
                hit.score / max_score,
                1.0 / rank,
                overlap / max(len(query_tokens), 1),
                overlap / union,
                min(len(query_tokens), 100) / 100.0,
                min(len(article_tokens), 800) / 800.0,
            ]
            output.append(
                PairRow(
                    question_index=question_index,
                    query_ids=encode(row.query, vocab, max_query_tokens),
                    article_ids=encode(article_text(region), vocab, max_article_tokens),
                    features=features,
                    target=target,
                    stable_ref=hit.stable_ref,
                    bm25_score=hit.score,
                )
            )
    return output


def evaluate_baseline(index: PortableBM25Index, rows: list[CandidateQuestion], *, top_ks: tuple[int, ...], candidate_k: int) -> dict[str, object]:
    ranks: list[int | None] = []
    for row in rows:
        refs = [hit.stable_ref for hit in index.search(row.query, top_k=candidate_k)]
        ranks.append(first_rank(refs, row.article_ref))
    return {"article_hit_at_k": summarize_ranks(ranks, top_ks=top_ks)}


def evaluate_model(
    index: PortableBM25Index,
    model: TeacherReranker,
    pairs: list[PairRow],
    rows: list[CandidateQuestion],
    *,
    top_ks: tuple[int, ...],
    candidate_k: int,
    blend_alpha: float,
) -> dict[str, object]:
    max_k = max(top_ks)
    groups = group_pairs(pairs)
    ranks: list[int | None] = []
    by_language: dict[str, list[int | None]] = {}
    by_style: dict[str, list[int | None]] = {}
    inference_times: list[float] = []
    model.eval()
    with torch.no_grad():
        for question_index, row in enumerate(rows):
            group = groups.get(question_index)
            if not group:
                refs = [hit.stable_ref for hit in index.search(row.query, top_k=candidate_k)[:max_k]]
            else:
                started = time.perf_counter()
                query_batch, article_batch, feature_batch, _target_batch = tensorize(group)
                logits = model(query_batch, article_batch, feature_batch)
                inference_times.append(time.perf_counter() - started)
                bm25_scores = torch.tensor([pair.bm25_score for pair in group], dtype=torch.float32)
                bm25_scores = bm25_scores / bm25_scores.max().clamp_min(1e-6)
                neural_scores = torch.sigmoid(logits)
                scores = (blend_alpha * neural_scores) + ((1.0 - blend_alpha) * bm25_scores)
                ordered = sorted(zip(group, scores.tolist(), strict=True), key=lambda item: (-item[1], item[0].stable_ref))
                refs = [pair.stable_ref for pair, _score in ordered[:max_k]]
            rank = first_rank(refs, row.article_ref)
            ranks.append(rank)
            by_language.setdefault(row.language, []).append(rank)
            by_style.setdefault(row.style, []).append(rank)
    return {
        "article_hit_at_k": summarize_ranks(ranks, top_ks=top_ks),
        "by_language": summarize_groups(by_language, top_ks=top_ks),
        "by_style": summarize_groups(by_style, top_ks=top_ks),
        "avg_inference_ms_per_query": round((sum(inference_times) / max(len(inference_times), 1)) * 1000, 4),
    }


def ranking_loss(logits: torch.Tensor, targets: torch.Tensor, *, margin: float) -> torch.Tensor:
    positive = logits[targets >= 0.95]
    negative = logits[targets < 0.95]
    if positive.numel() == 0 or negative.numel() == 0:
        return torch.zeros((), dtype=logits.dtype)
    return F.relu(margin - positive.mean() + negative).mean()


def tensorize(rows: list[PairRow]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    query_width = max(len(row.query_ids) for row in rows)
    article_width = max(len(row.article_ids) for row in rows)
    query_batch = torch.tensor([row.query_ids + [0] * (query_width - len(row.query_ids)) for row in rows], dtype=torch.long)
    article_batch = torch.tensor([row.article_ids + [0] * (article_width - len(row.article_ids)) for row in rows], dtype=torch.long)
    feature_batch = torch.tensor([row.features for row in rows], dtype=torch.float32)
    target_batch = torch.tensor([row.target for row in rows], dtype=torch.float32)
    return query_batch, article_batch, feature_batch, target_batch


def group_pairs(rows: list[PairRow]) -> dict[int, list[PairRow]]:
    groups: dict[int, list[PairRow]] = {}
    for row in rows:
        groups.setdefault(row.question_index, []).append(row)
    return groups


def mean_pool(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.unsqueeze(-1).float()
    return (values * expanded).sum(dim=1) / expanded.sum(dim=1).clamp_min(1.0)


def encode(text: str, vocab: dict[str, int], max_tokens: int) -> list[int]:
    ids = [vocab.get(token, 1) for token in tokenize(text)[:max_tokens]]
    return ids or [1]


def article_text(region: SearchRegion) -> str:
    return "\n".join([region.text, region.view.summary, *region.view.questions, *region.view.keywords])


def first_rank(stable_refs: list[str], gold_ref: str) -> int | None:
    gold_article = article_id_from_ref(gold_ref)
    for rank, stable_ref in enumerate(stable_refs, start=1):
        if article_id_from_ref(stable_ref) == gold_article:
            return rank
    return None


if __name__ == "__main__":
    main()
