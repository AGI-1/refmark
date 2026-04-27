"""Train a small classifier from cached query embeddings to BGB article refs.

This is a focused detour from the no-runtime-embedding goal. Earlier runs
trained tiny text models to imitate embedding behavior. Here the input is the
cached Qwen3 query embedding itself, so the question is narrower:

    If an embedding provider is already available, can a small supervised head
    map query vectors directly to Refmark/article targets?

That is not browser-offline by itself, but it tells us whether embeddings carry
enough corpus-local address signal to be learned cheaply.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.bgb_browser_search.adapt_bgb_static_views import article_regions, summarize_ranks  # noqa: E402
from examples.bgb_browser_search.train_bgb_article_candidate_generator import CandidateQuestion, load_split_questions, summarize_groups  # noqa: E402
from refmark.search_index import PortableBM25Index, load_search_index  # noqa: E402

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This training example requires torch.") from exc


class EmbeddingClassifier(nn.Module):
    def __init__(self, input_dim: int, class_count: int, *, hidden_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        current = input_dim
        for _ in range(max(layers, 0)):
            modules.extend([nn.Linear(current, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            current = hidden_dim
        modules.append(nn.Linear(current, class_count))
        self.classifier = nn.Sequential(*modules)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train cached query-embedding -> BGB article classifier.")
    parser.add_argument("--index", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_openrouter_index.json")
    parser.add_argument("--stress-report", action="append", required=True)
    parser.add_argument("--query-cache", action="append", required=True)
    parser.add_argument("--embedding-model", default="qwen/qwen3-embedding-8b")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2424)
    parser.add_argument("--top-ks", default="1,3,5,10,20,50")
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0008)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--fusion-candidate-k", type=int, default=80)
    parser.add_argument("--model-fusion-weight", type=float, default=0.65)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    top_ks = tuple(int(part) for part in args.top_ks.split(",") if part.strip())

    source_index = load_search_index(args.index)
    articles = article_regions(source_index.regions)
    article_refs = [article.stable_ref for article in articles]
    class_by_ref = {stable_ref: index for index, stable_ref in enumerate(article_refs)}
    bm25_index = PortableBM25Index(articles, include_source=True)

    query_vectors = read_query_vectors([Path(path) for path in args.query_cache], model=args.embedding_model)
    train_rows, eval_rows = load_split_questions(args.stress_report, train_fraction=args.train_fraction, seed=args.seed)
    train_rows = [row for row in train_rows if row.article_ref in class_by_ref and row.query in query_vectors]
    eval_rows = [row for row in eval_rows if row.article_ref in class_by_ref and row.query in query_vectors]
    if not train_rows or not eval_rows:
        raise SystemExit("No train/eval rows matched the query embedding cache.")

    input_dim = int(next(iter(query_vectors.values())).shape[0])
    model = EmbeddingClassifier(
        input_dim,
        len(article_refs),
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        dropout=args.dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_batches = make_batches(train_rows, class_by_ref, query_vectors, args.batch_size, shuffle=True)
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_score = -1.0
    history: list[dict[str, object]] = []
    started = time.perf_counter()

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
        metrics = evaluate_model(model, eval_rows, class_by_ref, article_refs, query_vectors, top_ks=top_ks)
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
        train_batches = make_batches(train_rows, class_by_ref, query_vectors, args.batch_size, shuffle=True)

    train_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    classifier_metrics = evaluate_model(model, eval_rows, class_by_ref, article_refs, query_vectors, top_ks=top_ks)
    bm25_metrics = evaluate_bm25(bm25_index, eval_rows, top_ks=top_ks)
    fusion_metrics = evaluate_fusion(
        model,
        bm25_index,
        eval_rows,
        class_by_ref,
        article_refs,
        query_vectors,
        top_ks=top_ks,
        candidate_k=args.fusion_candidate_k,
        model_weight=args.model_fusion_weight,
    )

    payload = {
        "schema": "refmark.bgb_query_embedding_classifier.v1",
        "settings": vars(args),
        "embedding_model": args.embedding_model,
        "article_refs": article_refs,
        "model_state": model.state_dict(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    artifact_bytes = output.stat().st_size
    report = {
        "schema": "refmark.bgb_query_embedding_classifier_report.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_index": args.index,
        "stress_reports": args.stress_report,
        "query_caches": args.query_cache,
        "embedding_model": args.embedding_model,
        "settings": vars(args),
        "article_count": len(article_refs),
        "embedding_dim": input_dim,
        "train_questions": len(train_rows),
        "eval_questions": len(eval_rows),
        "train_article_count": len({row.article_ref for row in train_rows}),
        "eval_article_count": len({row.article_ref for row in eval_rows}),
        "parameters": sum(param.numel() for param in model.parameters()),
        "artifact_bytes": artifact_bytes,
        "artifact_megabytes": round(artifact_bytes / 1_000_000, 4),
        "train_seconds": round(train_seconds, 3),
        "history": history,
        "best_epoch": max(history, key=lambda row: float(row["mrr"]) + float(row["hit_at_10"]))["epoch"],
        "bm25_article_baseline": bm25_metrics,
        "query_embedding_classifier": classifier_metrics,
        "bm25_plus_query_embedding_classifier": fusion_metrics,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


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


def make_batches(
    rows: list[CandidateQuestion],
    class_by_ref: dict[str, int],
    query_vectors: dict[str, torch.Tensor],
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
        query_batch = torch.stack([query_vectors[row.query] for row in batch])
        label_batch = torch.tensor([class_by_ref[row.article_ref] for row in batch], dtype=torch.long)
        batches.append((query_batch, label_batch))
    return batches


def evaluate_model(
    model: EmbeddingClassifier,
    rows: list[CandidateQuestion],
    class_by_ref: dict[str, int],
    article_refs: list[str],
    query_vectors: dict[str, torch.Tensor],
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
            logits = model(query_vectors[row.query].unsqueeze(0))[0]
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


def evaluate_fusion(
    model: EmbeddingClassifier,
    bm25_index: PortableBM25Index,
    rows: list[CandidateQuestion],
    class_by_ref: dict[str, int],
    article_refs: list[str],
    query_vectors: dict[str, torch.Tensor],
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
            logits = model(query_vectors[row.query].unsqueeze(0))[0]
            model_indices = torch.topk(logits, k=min(candidate_k, len(article_refs))).indices.tolist()
            model_refs = [article_refs[index] for index in model_indices]
            bm25_refs = [hit.stable_ref for hit in bm25_index.search(row.query, top_k=candidate_k)]
            fused_refs = rrf_fuse(model_refs, bm25_refs, first_weight=model_weight)[:max_k]
            ranks.append(first_rank(fused_refs, row.article_ref))
    return {"article_hit_at_k": summarize_ranks(ranks, top_ks=top_ks)}


def first_rank(stable_refs: list[str], gold_ref: str) -> int | None:
    for rank, stable_ref in enumerate(stable_refs, start=1):
        if stable_ref == gold_ref:
            return rank
    return None


def rrf_fuse(first: list[str], second: list[str], *, first_weight: float) -> list[str]:
    scores: dict[str, float] = {}
    for weight, refs in ((first_weight, first), (1.0 - first_weight, second)):
        for rank, ref in enumerate(refs, start=1):
            scores[ref] = scores.get(ref, 0.0) + (weight / (rank + 60.0))
    return [ref for ref, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


if __name__ == "__main__":
    main()
