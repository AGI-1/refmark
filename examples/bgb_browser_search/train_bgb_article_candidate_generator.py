"""Train a tiny BGB article candidate generator.

This is the direct distillation/local-navigation experiment: given only a query,
the model emits likely article refs. It does not need BM25 candidates at runtime,
though the report also evaluates a BM25+model fusion to show whether the model
adds complementary recall.
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

from examples.bgb_browser_search.adapt_bgb_static_views import (  # noqa: E402
    article_id_from_ref,
    article_regions,
    split_questions_by_block,
    stress_questions,
    summarize_ranks,
)
from examples.bgb_browser_search.run_bgb_stress_eval import StressQuestion  # noqa: E402
from refmark.search_index import PortableBM25Index, SearchHit, load_search_index, tokenize  # noqa: E402

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This training example requires torch.") from exc


@dataclass(frozen=True)
class CandidateQuestion:
    query: str
    article_ref: str
    language: str
    style: str
    source_report: str


class ArticleCandidateGenerator(nn.Module):
    def __init__(self, vocab_size: int, class_count: int, *, embed_dim: int, hidden_dim: int, layers: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        modules: list[nn.Module] = [nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.08)]
        for _ in range(max(layers - 1, 0)):
            modules.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.08)])
        modules.append(nn.Linear(hidden_dim, class_count))
        self.classifier = nn.Sequential(*modules)

    def forward(self, query_ids: torch.Tensor) -> torch.Tensor:
        mask = query_ids.ne(0).unsqueeze(-1).float()
        embedded = self.embedding(query_ids)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.classifier(pooled)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny direct BGB article candidate generator.")
    parser.add_argument("--index", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_openrouter_index.json")
    parser.add_argument("--stress-report", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=919)
    parser.add_argument("--top-ks", default="1,3,5,10,20,50")
    parser.add_argument("--vocab-size", type=int, default=24000)
    parser.add_argument("--max-query-tokens", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--label-smoothing", type=float, default=0.04)
    parser.add_argument("--fusion-candidate-k", type=int, default=80)
    parser.add_argument("--model-fusion-weight", type=float, default=0.7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    top_ks = tuple(int(part) for part in args.top_ks.split(",") if part.strip())

    source_index = load_search_index(args.index)
    articles = article_regions(source_index.regions)
    article_refs = [article.stable_ref for article in articles]
    class_by_ref = {stable_ref: index for index, stable_ref in enumerate(article_refs)}
    bm25_index = PortableBM25Index(articles, include_source=True)

    train_rows, eval_rows = load_split_questions(args.stress_report, train_fraction=args.train_fraction, seed=args.seed)
    train_rows = [row for row in train_rows if row.article_ref in class_by_ref]
    eval_rows = [row for row in eval_rows if row.article_ref in class_by_ref]
    vocab = build_vocab(articles, train_rows, vocab_size=args.vocab_size)

    train_batches = make_batches(train_rows, class_by_ref, vocab, args.max_query_tokens, args.batch_size, shuffle=True)
    model = ArticleCandidateGenerator(
        len(vocab),
        len(article_refs),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    started = time.perf_counter()
    history: list[dict[str, object]] = []
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for query_batch, label_batch in train_batches:
            optimizer.zero_grad(set_to_none=True)
            logits = model(query_batch)
            loss = F.cross_entropy(logits, label_batch, label_smoothing=args.label_smoothing)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = evaluate_model(model, eval_rows, class_by_ref, article_refs, vocab, args.max_query_tokens, top_ks=top_ks)
        score = float(metrics["article_hit_at_k"]["mrr"]) + float(metrics["article_hit_at_k"]["hit_at_k"].get("10", 0.0))
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
        train_batches = make_batches(train_rows, class_by_ref, vocab, args.max_query_tokens, args.batch_size, shuffle=True)

    train_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    model_metrics = evaluate_model(model, eval_rows, class_by_ref, article_refs, vocab, args.max_query_tokens, top_ks=top_ks)
    bm25_metrics = evaluate_bm25(bm25_index, eval_rows, top_ks=top_ks)
    fusion_metrics = evaluate_fusion(
        model,
        bm25_index,
        eval_rows,
        class_by_ref,
        article_refs,
        vocab,
        args.max_query_tokens,
        top_ks=top_ks,
        candidate_k=args.fusion_candidate_k,
        model_weight=args.model_fusion_weight,
    )
    payload = {
        "schema": "refmark.bgb_article_candidate_generator.v1",
        "settings": vars(args),
        "vocab": vocab,
        "article_refs": article_refs,
        "model_state": model.state_dict(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    artifact_bytes = output.stat().st_size
    report = {
        "schema": "refmark.bgb_article_candidate_generator_report.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_index": args.index,
        "stress_reports": args.stress_report,
        "settings": vars(args),
        "article_count": len(article_refs),
        "train_questions": len(train_rows),
        "eval_questions": len(eval_rows),
        "train_article_count": len({row.article_ref for row in train_rows}),
        "eval_article_count": len({row.article_ref for row in eval_rows}),
        "vocab_size": len(vocab),
        "parameters": sum(param.numel() for param in model.parameters()),
        "artifact_bytes": artifact_bytes,
        "artifact_megabytes": round(artifact_bytes / 1_000_000, 4),
        "train_seconds": round(train_seconds, 3),
        "history": history,
        "best_epoch": max(history, key=lambda row: float(row["mrr"]) + float(row["hit_at_10"]))["epoch"],
        "bm25_article_baseline": bm25_metrics,
        "candidate_generator": model_metrics,
        "bm25_plus_candidate_generator": fusion_metrics,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def load_split_questions(paths: list[str], *, train_fraction: float, seed: int) -> tuple[list[CandidateQuestion], list[CandidateQuestion]]:
    train: list[CandidateQuestion] = []
    eval_rows: list[CandidateQuestion] = []
    for offset, path in enumerate(paths):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        train_stress, eval_stress = split_questions_by_block(
            stress_questions(payload),
            train_fraction=train_fraction,
            seed=seed + offset,
        )
        train.extend(to_candidate_questions(train_stress, source_report=path))
        eval_rows.extend(to_candidate_questions(eval_stress, source_report=path))
    return train, eval_rows


def to_candidate_questions(rows: list[StressQuestion], *, source_report: str) -> list[CandidateQuestion]:
    output: list[CandidateQuestion] = []
    for row in rows:
        output.append(
            CandidateQuestion(
                query=row.query,
                article_ref=row.block_id,
                language=row.language,
                style=row.style,
                source_report=Path(source_report).name,
            )
        )
    return output


def build_vocab(articles, train_rows: list[CandidateQuestion], *, vocab_size: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in train_rows:
        counts.update(tokenize(row.query))
    for article in articles:
        counts.update(tokenize(article.index_text(include_source=True)))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, _count in counts.most_common(max(vocab_size - len(vocab), 0)):
        vocab.setdefault(token, len(vocab))
    return vocab


def make_batches(
    rows: list[CandidateQuestion],
    class_by_ref: dict[str, int],
    vocab: dict[str, int],
    max_query_tokens: int,
    batch_size: int,
    *,
    shuffle: bool,
):
    ordered = list(rows)
    if shuffle:
        random.shuffle(ordered)
    batches = []
    for start in range(0, len(ordered), batch_size):
        batch = ordered[start : start + batch_size]
        encoded = [encode(row.query, vocab, max_query_tokens) for row in batch]
        max_len = max(len(ids) for ids in encoded)
        query_batch = torch.tensor([ids + [0] * (max_len - len(ids)) for ids in encoded], dtype=torch.long)
        label_batch = torch.tensor([class_by_ref[row.article_ref] for row in batch], dtype=torch.long)
        batches.append((query_batch, label_batch))
    return batches


def encode(text: str, vocab: dict[str, int], max_tokens: int) -> list[int]:
    ids = [vocab.get(token, 1) for token in tokenize(text)[:max_tokens]]
    return ids or [1]


def evaluate_model(
    model: ArticleCandidateGenerator,
    rows: list[CandidateQuestion],
    class_by_ref: dict[str, int],
    article_refs: list[str],
    vocab: dict[str, int],
    max_query_tokens: int,
    *,
    top_ks: tuple[int, ...],
) -> dict[str, object]:
    max_k = max(top_ks)
    ranks: list[int | None] = []
    by_language: dict[str, list[int | None]] = {}
    by_style: dict[str, list[int | None]] = {}
    by_report: dict[str, list[int | None]] = {}
    inference_times: list[float] = []
    misses = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            started = time.perf_counter()
            ids = encode(row.query, vocab, max_query_tokens)
            query_batch = torch.tensor([ids], dtype=torch.long)
            logits = model(query_batch)[0]
            inference_times.append(time.perf_counter() - started)
            top_indices = torch.topk(logits, k=min(max_k, len(article_refs))).indices.tolist()
            stable_refs = [article_refs[index] for index in top_indices]
            rank = first_rank(stable_refs, row.article_ref)
            ranks.append(rank)
            by_language.setdefault(row.language, []).append(rank)
            by_style.setdefault(row.style, []).append(rank)
            by_report.setdefault(row.source_report, []).append(rank)
            if rank is None and len(misses) < 20:
                misses.append({"query": row.query, "gold_ref": row.article_ref, "top_refs": stable_refs[:5]})
    summary = {
        "article_hit_at_k": summarize_ranks(ranks, top_ks=top_ks),
        "by_language": summarize_groups(by_language, top_ks=top_ks),
        "by_style": summarize_groups(by_style, top_ks=top_ks),
        "by_report": summarize_groups(by_report, top_ks=top_ks),
        "avg_inference_ms_per_query": round((sum(inference_times) / max(len(inference_times), 1)) * 1000, 4),
        "sample_misses": misses,
    }
    return summary


def evaluate_bm25(index: PortableBM25Index, rows: list[CandidateQuestion], *, top_ks: tuple[int, ...]) -> dict[str, object]:
    max_k = max(top_ks)
    ranks = []
    for row in rows:
        hits = index.search(row.query, top_k=max_k)
        ranks.append(first_rank([hit.stable_ref for hit in hits], row.article_ref))
    return {"article_hit_at_k": summarize_ranks(ranks, top_ks=top_ks)}


def evaluate_fusion(
    model: ArticleCandidateGenerator,
    bm25_index: PortableBM25Index,
    rows: list[CandidateQuestion],
    class_by_ref: dict[str, int],
    article_refs: list[str],
    vocab: dict[str, int],
    max_query_tokens: int,
    *,
    top_ks: tuple[int, ...],
    candidate_k: int,
    model_weight: float,
) -> dict[str, object]:
    max_k = max(top_ks)
    ranks = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            ids = encode(row.query, vocab, max_query_tokens)
            logits = model(torch.tensor([ids], dtype=torch.long))[0]
            model_indices = torch.topk(logits, k=min(candidate_k, len(article_refs))).indices.tolist()
            model_refs = [article_refs[index] for index in model_indices]
            bm25_refs = [hit.stable_ref for hit in bm25_index.search(row.query, top_k=candidate_k)]
            fused_refs = rrf_fuse(model_refs, bm25_refs, first_weight=model_weight)[:max_k]
            ranks.append(first_rank(fused_refs, row.article_ref))
    return {"article_hit_at_k": summarize_ranks(ranks, top_ks=top_ks)}


def rrf_fuse(first: list[str], second: list[str], *, first_weight: float) -> list[str]:
    scores: dict[str, float] = {}
    for weight, refs in ((first_weight, first), (1.0 - first_weight, second)):
        for rank, stable_ref in enumerate(refs, start=1):
            scores[stable_ref] = scores.get(stable_ref, 0.0) + weight / (60.0 + rank)
    return sorted(scores, key=lambda stable_ref: (-scores[stable_ref], stable_ref))


def first_rank(stable_refs: list[str], gold_ref: str) -> int | None:
    gold_article = article_id_from_ref(gold_ref)
    for rank, stable_ref in enumerate(stable_refs, start=1):
        if article_id_from_ref(stable_ref) == gold_article:
            return rank
    return None


def summarize_groups(groups: dict[str, list[int | None]], *, top_ks: tuple[int, ...]) -> dict[str, object]:
    return {name: summarize_ranks(ranks, top_ks=top_ks) for name, ranks in sorted(groups.items())}


if __name__ == "__main__":
    main()
