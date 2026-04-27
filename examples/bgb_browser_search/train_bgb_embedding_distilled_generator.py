"""Distill Qwen3 article embeddings into a tiny local query encoder.

This is the closest current test of the "offline embedding teacher, local
runtime model" idea. Qwen3 document vectors are built offline and averaged to
article vectors. A small query encoder is trained so its output lands near the
gold article vector; runtime search is then a local dot product against stored
article vectors, without OpenRouter or a vector DB.
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
from examples.bgb_browser_search.build_bgb_article_navigation import article_id_for  # noqa: E402
from examples.bgb_browser_search.evaluate_bgb_stress_embeddings import read_region_vectors  # noqa: E402
from examples.bgb_browser_search.train_bgb_article_candidate_generator import (  # noqa: E402
    CandidateQuestion,
    build_vocab,
    encode,
    load_split_questions,
    summarize_groups,
)
from refmark.search_index import PortableBM25Index, SearchRegion, load_search_index  # noqa: E402

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This training example requires torch.") from exc


class QueryVectorModel(nn.Module):
    def __init__(self, vocab_size: int, output_dim: int, *, embed_dim: int, hidden_dim: int, layers: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        modules: list[nn.Module] = [nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.08)]
        for _ in range(max(layers - 1, 0)):
            modules.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.08)])
        modules.append(nn.Linear(hidden_dim, output_dim))
        self.projector = nn.Sequential(*modules)

    def forward(self, query_ids: torch.Tensor) -> torch.Tensor:
        mask = query_ids.ne(0).unsqueeze(-1).float()
        embedded = self.embedding(query_ids)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return F.normalize(self.projector(pooled), dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny query encoder against offline Qwen3 article vectors.")
    parser.add_argument("--index", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_openrouter_index.json")
    parser.add_argument("--embedding-cache", default="examples/bgb_browser_search/output_scratch_multi_full/embedding_cache_qwen3_8b.jsonl")
    parser.add_argument("--embedding-model", default="qwen/qwen3-embedding-8b")
    parser.add_argument("--query-cache", action="append", default=[])
    parser.add_argument("--target-mode", choices=["gold-article", "query-embedding"], default="gold-article")
    parser.add_argument("--stress-report", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1219)
    parser.add_argument("--top-ks", default="1,3,5,10,20,50")
    parser.add_argument("--vocab-size", type=int, default=24000)
    parser.add_argument("--max-query-tokens", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--cosine-weight", type=float, default=0.15)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    top_ks = tuple(int(part) for part in args.top_ks.split(",") if part.strip())

    source_index = load_search_index(args.index)
    articles = article_regions(source_index.regions)
    article_refs, article_matrix = build_article_matrix(
        source_index.regions,
        embedding_cache=Path(args.embedding_cache),
        model=args.embedding_model,
    )
    class_by_ref = {stable_ref: index for index, stable_ref in enumerate(article_refs)}
    bm25_index = PortableBM25Index(articles, include_source=True)

    train_rows, eval_rows = load_split_questions(args.stress_report, train_fraction=args.train_fraction, seed=args.seed)
    train_rows = [row for row in train_rows if row.article_ref in class_by_ref]
    eval_rows = [row for row in eval_rows if row.article_ref in class_by_ref]
    query_vectors_by_text = {}
    if args.target_mode == "query-embedding":
        query_vectors_by_text = read_query_vectors([Path(path) for path in args.query_cache], model=args.embedding_model)
        train_rows = [row for row in train_rows if row.query in query_vectors_by_text]
        eval_rows = [row for row in eval_rows if row.query in query_vectors_by_text]
    vocab = build_vocab(articles, train_rows, vocab_size=args.vocab_size)

    model = QueryVectorModel(
        len(vocab),
        article_matrix.shape[1],
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_batches = make_batches(
        train_rows,
        class_by_ref,
        vocab,
        args.max_query_tokens,
        args.batch_size,
        shuffle=True,
        query_vectors_by_text=query_vectors_by_text,
    )
    started = time.perf_counter()
    history: list[dict[str, object]] = []
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for query_batch, label_batch, query_target_batch in train_batches:
            optimizer.zero_grad(set_to_none=True)
            query_vectors = model(query_batch)
            logits = query_vectors @ article_matrix.T / args.temperature
            ce = F.cross_entropy(logits, label_batch)
            target_vectors = query_target_batch if query_target_batch is not None else article_matrix[label_batch]
            cosine_loss = 1.0 - (query_vectors * target_vectors).sum(dim=1).mean()
            loss = ce + (args.cosine_weight * cosine_loss)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = evaluate_model(model, eval_rows, class_by_ref, article_refs, article_matrix, vocab, args.max_query_tokens, top_ks=top_ks)
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
        train_batches = make_batches(
            train_rows,
            class_by_ref,
            vocab,
            args.max_query_tokens,
            args.batch_size,
            shuffle=True,
            query_vectors_by_text=query_vectors_by_text,
        )

    train_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    distilled = evaluate_model(model, eval_rows, class_by_ref, article_refs, article_matrix, vocab, args.max_query_tokens, top_ks=top_ks)
    bm25 = evaluate_bm25(bm25_index, eval_rows, top_ks=top_ks)
    payload = {
        "schema": "refmark.bgb_embedding_distilled_generator.v1",
        "settings": vars(args),
        "vocab": vocab,
        "article_refs": article_refs,
        "article_matrix": article_matrix.half(),
        "model_state": model.state_dict(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    artifact_bytes = output.stat().st_size
    report = {
        "schema": "refmark.bgb_embedding_distilled_generator_report.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_index": args.index,
        "embedding_cache": args.embedding_cache,
        "query_caches": args.query_cache,
        "embedding_model": args.embedding_model,
        "stress_reports": args.stress_report,
        "settings": vars(args),
        "article_count": len(article_refs),
        "article_vector_dim": int(article_matrix.shape[1]),
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
        "bm25_article_baseline": bm25,
        "distilled_query_encoder": distilled,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def build_article_matrix(
    regions: list[SearchRegion],
    *,
    embedding_cache: Path,
    model: str,
) -> tuple[list[str], torch.Tensor]:
    vectors_by_ref = read_region_vectors(embedding_cache, regions, model=model)
    missing = [region.stable_ref for region in regions if region.stable_ref not in vectors_by_ref]
    if missing:
        sample = ", ".join(missing[:5])
        raise RuntimeError(f"Embedding cache is missing {len(missing)} region vectors, e.g. {sample}")
    grouped: dict[str, list[torch.Tensor]] = {}
    for region in regions:
        article_ref = f"{region.doc_id}:{article_id_for(region.region_id)}"
        grouped.setdefault(article_ref, []).append(torch.tensor(vectors_by_ref[region.stable_ref], dtype=torch.float32))
    article_refs = sorted(grouped, key=lambda ref: min(region.ordinal for region in regions if f"{region.doc_id}:{article_id_for(region.region_id)}" == ref))
    vectors = []
    for article_ref in article_refs:
        stacked = torch.stack(grouped[article_ref])
        vectors.append(F.normalize(stacked.mean(dim=0), dim=0))
    return article_refs, torch.stack(vectors)


def make_batches(
    rows: list[CandidateQuestion],
    class_by_ref: dict[str, int],
    vocab: dict[str, int],
    max_query_tokens: int,
    batch_size: int,
    *,
    shuffle: bool,
    query_vectors_by_text: dict[str, torch.Tensor] | None = None,
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
        query_target_batch = None
        if query_vectors_by_text:
            query_target_batch = torch.stack([query_vectors_by_text[row.query] for row in batch])
        batches.append((query_batch, label_batch, query_target_batch))
    return batches


def read_query_vectors(paths: list[Path], *, model: str) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for path in paths:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("model") != model:
                    continue
                output[str(row["query"])] = F.normalize(torch.tensor(row["embedding"], dtype=torch.float32), dim=0)
    return output


def evaluate_model(
    model: QueryVectorModel,
    rows: list[CandidateQuestion],
    class_by_ref: dict[str, int],
    article_refs: list[str],
    article_matrix: torch.Tensor,
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
            query_vector = model(torch.tensor([ids], dtype=torch.long))[0]
            scores = query_vector @ article_matrix.T
            inference_times.append(time.perf_counter() - started)
            top_indices = torch.topk(scores, k=min(max_k, len(article_refs))).indices.tolist()
            stable_refs = [article_refs[index] for index in top_indices]
            rank = first_rank(stable_refs, row.article_ref)
            ranks.append(rank)
            by_language.setdefault(row.language, []).append(rank)
            by_style.setdefault(row.style, []).append(rank)
            by_report.setdefault(row.source_report, []).append(rank)
            if rank is None and len(misses) < 20:
                misses.append({"query": row.query, "gold_ref": row.article_ref, "top_refs": stable_refs[:5]})
    return {
        "article_hit_at_k": summarize_ranks(ranks, top_ks=top_ks),
        "by_language": summarize_groups(by_language, top_ks=top_ks),
        "by_style": summarize_groups(by_style, top_ks=top_ks),
        "by_report": summarize_groups(by_report, top_ks=top_ks),
        "avg_inference_ms_per_query": round((sum(inference_times) / max(len(inference_times), 1)) * 1000, 4),
        "sample_misses": misses,
    }


def evaluate_bm25(index: PortableBM25Index, rows: list[CandidateQuestion], *, top_ks: tuple[int, ...]) -> dict[str, object]:
    max_k = max(top_ks)
    ranks = []
    for row in rows:
        refs = [hit.stable_ref for hit in index.search(row.query, top_k=max_k)]
        ranks.append(first_rank(refs, row.article_ref))
    return {"article_hit_at_k": summarize_ranks(ranks, top_ks=top_ks)}


def first_rank(stable_refs: list[str], gold_ref: str) -> int | None:
    gold_article = article_id_from_ref(gold_ref)
    for rank, stable_ref in enumerate(stable_refs, start=1):
        if article_id_from_ref(stable_ref) == gold_article:
            return rank
    return None


if __name__ == "__main__":
    main()
