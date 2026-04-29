"""Train a tiny BGB query-to-area router.

This is the coarse stage for a possible hierarchy:

    query -> one of ~50 article areas -> article/ref resolver inside that area

The first version intentionally uses boring equal-sized article windows. The
report then exposes stable misroutes so the next loop can adapt boundaries or
add overlapping/merged areas with evidence instead of guesswork.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
)
from examples.bgb_browser_search.run_bgb_stress_eval import StressQuestion  # noqa: E402
from refmark.search_index import load_search_index, tokenize  # noqa: E402

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This training example requires torch.") from exc


@dataclass(frozen=True)
class Area:
    area_id: str
    start_index: int
    end_index: int
    start_ref: str
    end_ref: str
    refs: tuple[str, ...]


@dataclass(frozen=True)
class AreaQuestion:
    query: str
    article_ref: str
    area_index: int
    language: str
    style: str
    source_report: str


class AreaRouter(nn.Module):
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
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Train a BGB query-to-area router.")
    parser.add_argument("--index", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_confusion_signatures_gemma_top300_index.json")
    parser.add_argument("--stress-report", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--area-size", type=int, default=50)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2929)
    parser.add_argument("--top-ks", default="1,2,3,5,8")
    parser.add_argument("--vocab-size", type=int, default=24000)
    parser.add_argument("--max-query-tokens", type=int, default=72)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    top_ks = tuple(int(part) for part in args.top_ks.split(",") if part.strip())
    started = time.perf_counter()

    articles = article_regions(load_search_index(args.index).regions)
    areas = equal_areas([article.stable_ref for article in articles], area_size=args.area_size)
    area_by_ref = {ref: area_index for area_index, area in enumerate(areas) for ref in area.refs}
    train_rows, eval_rows = load_split_questions(
        args.stress_report,
        area_by_ref=area_by_ref,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    vocab = build_vocab(train_rows, vocab_size=args.vocab_size)
    model = AreaRouter(
        len(vocab),
        len(areas),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history: list[dict[str, object]] = []
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for query_batch, label_batch in make_batches(
            train_rows,
            vocab,
            max_query_tokens=args.max_query_tokens,
            batch_size=args.batch_size,
            shuffle=True,
        ):
            optimizer.zero_grad(set_to_none=True)
            logits = model(query_batch)
            loss = F.cross_entropy(logits, label_batch, label_smoothing=args.label_smoothing)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = evaluate_model(model, eval_rows, vocab, max_query_tokens=args.max_query_tokens, top_ks=top_ks)
        score = float(metrics["hit_at_k"].get("1", 0.0)) + (0.25 * float(metrics["hit_at_k"].get("3", 0.0)))
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        row = {
            "epoch": epoch,
            "loss": round(sum(losses) / max(len(losses), 1), 6),
            "hit_at_1": metrics["hit_at_k"].get("1", 0.0),
            "hit_at_3": metrics["hit_at_k"].get("3", 0.0),
            "hit_at_5": metrics["hit_at_k"].get("5", 0.0),
            "mrr": metrics["mrr"],
        }
        history.append(row)
        print(json.dumps(row))

    model.load_state_dict(best_state)
    train_metrics = evaluate_model(model, train_rows, vocab, max_query_tokens=args.max_query_tokens, top_ks=top_ks)
    eval_metrics = evaluate_model(model, eval_rows, vocab, max_query_tokens=args.max_query_tokens, top_ks=top_ks)
    eval_predictions = predict_rows(model, eval_rows, vocab, max_query_tokens=args.max_query_tokens, top_k=max(top_ks))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "refmark.bgb_area_router.v1",
            "settings": vars(args),
            "areas": [area.__dict__ for area in areas],
            "vocab": vocab,
            "model_state": model.state_dict(),
        },
        output,
    )
    report = {
        "schema": "refmark.bgb_area_router_train_report.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_index": args.index,
        "stress_reports": args.stress_report,
        "settings": vars(args),
        "article_count": len(articles),
        "area_count": len(areas),
        "areas": [area.__dict__ for area in areas],
        "train_questions": len(train_rows),
        "eval_questions": len(eval_rows),
        "parameters": sum(param.numel() for param in model.parameters()),
        "artifact_bytes": output.stat().st_size,
        "artifact_megabytes": round(output.stat().st_size / 1_000_000, 4),
        "seconds": round(time.perf_counter() - started, 3),
        "history": history,
        "best_epoch": max(history, key=lambda row: float(row["hit_at_1"]) + (0.25 * float(row["hit_at_3"])))["epoch"],
        "train": train_metrics,
        "eval": eval_metrics,
        "misroute_heatmap": misroute_heatmap(eval_rows, eval_predictions, areas),
        "boundary_adaptation_candidates": boundary_adaptation_candidates(eval_rows, eval_predictions, areas),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def equal_areas(article_refs: list[str], *, area_size: int) -> list[Area]:
    if area_size <= 0:
        raise SystemExit("--area-size must be positive.")
    areas = []
    for area_index, start in enumerate(range(0, len(article_refs), area_size)):
        refs = tuple(article_refs[start : start + area_size])
        areas.append(
            Area(
                area_id=f"A{area_index:03d}",
                start_index=start,
                end_index=start + len(refs) - 1,
                start_ref=refs[0],
                end_ref=refs[-1],
                refs=refs,
            )
        )
    return areas


def load_split_questions(
    paths: list[str],
    *,
    area_by_ref: dict[str, int],
    train_fraction: float,
    seed: int,
) -> tuple[list[AreaQuestion], list[AreaQuestion]]:
    train: list[AreaQuestion] = []
    eval_rows: list[AreaQuestion] = []
    for offset, path in enumerate(paths):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        train_stress, eval_stress = split_questions_by_block(
            stress_questions(payload),
            train_fraction=train_fraction,
            seed=seed + offset,
        )
        train.extend(to_area_questions(train_stress, area_by_ref=area_by_ref, source_report=path))
        eval_rows.extend(to_area_questions(eval_stress, area_by_ref=area_by_ref, source_report=path))
    return train, eval_rows


def to_area_questions(rows: list[StressQuestion], *, area_by_ref: dict[str, int], source_report: str) -> list[AreaQuestion]:
    output = []
    for row in rows:
        article_ref = article_id_from_ref(row.block_id)
        if article_ref not in area_by_ref:
            continue
        output.append(
            AreaQuestion(
                query=row.query,
                article_ref=article_ref,
                area_index=area_by_ref[article_ref],
                language=row.language,
                style=row.style,
                source_report=Path(source_report).name,
            )
        )
    return output


def build_vocab(rows: list[AreaQuestion], *, vocab_size: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(tokenize(row.query))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, _count in counts.most_common(max(vocab_size - len(vocab), 0)):
        vocab.setdefault(token, len(vocab))
    return vocab


def make_batches(
    rows: list[AreaQuestion],
    vocab: dict[str, int],
    *,
    max_query_tokens: int,
    batch_size: int,
    shuffle: bool,
):
    ordered = list(rows)
    if shuffle:
        random.shuffle(ordered)
    for start in range(0, len(ordered), batch_size):
        batch = ordered[start : start + batch_size]
        encoded = [encode(row.query, vocab, max_query_tokens) for row in batch]
        max_len = max(len(ids) for ids in encoded)
        query_batch = torch.tensor([ids + [0] * (max_len - len(ids)) for ids in encoded], dtype=torch.long)
        label_batch = torch.tensor([row.area_index for row in batch], dtype=torch.long)
        yield query_batch, label_batch


def encode(text: str, vocab: dict[str, int], max_tokens: int) -> list[int]:
    ids = [vocab.get(token, 1) for token in tokenize(text)[:max_tokens]]
    return ids or [1]


def evaluate_model(
    model: AreaRouter,
    rows: list[AreaQuestion],
    vocab: dict[str, int],
    *,
    max_query_tokens: int,
    top_ks: tuple[int, ...],
) -> dict[str, object]:
    predictions = predict_rows(model, rows, vocab, max_query_tokens=max_query_tokens, top_k=max(top_ks))
    ranks = []
    by_style: dict[str, list[int | None]] = defaultdict(list)
    by_language: dict[str, list[int | None]] = defaultdict(list)
    for row, predicted in zip(rows, predictions, strict=True):
        rank = first_rank(predicted, row.area_index)
        ranks.append(rank)
        by_style[row.style].append(rank)
        by_language[row.language].append(rank)
    return {
        **summarize_ranks(ranks, top_ks=top_ks),
        "by_style": {name: summarize_ranks(values, top_ks=top_ks) for name, values in sorted(by_style.items())},
        "by_language": {name: summarize_ranks(values, top_ks=top_ks) for name, values in sorted(by_language.items())},
    }


def predict_rows(
    model: AreaRouter,
    rows: list[AreaQuestion],
    vocab: dict[str, int],
    *,
    max_query_tokens: int,
    top_k: int,
) -> list[list[int]]:
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(rows), 256):
            batch = rows[start : start + 256]
            encoded = [encode(row.query, vocab, max_query_tokens) for row in batch]
            max_len = max(len(ids) for ids in encoded)
            query_batch = torch.tensor([ids + [0] * (max_len - len(ids)) for ids in encoded], dtype=torch.long)
            logits = model(query_batch)
            top = torch.topk(logits, k=min(top_k, logits.shape[1]), dim=1).indices.tolist()
            output.extend([[int(value) for value in row] for row in top])
    return output


def first_rank(predicted: list[int], gold: int) -> int | None:
    for rank, value in enumerate(predicted, start=1):
        if value == gold:
            return rank
    return None


def summarize_ranks(ranks: list[int | None], *, top_ks: tuple[int, ...]) -> dict[str, object]:
    total = max(len(ranks), 1)
    return {
        "count": len(ranks),
        "hit_at_k": {
            str(k): round(sum(1 for rank in ranks if rank is not None and rank <= k) / total, 4)
            for k in top_ks
        },
        "mrr": round(sum((1.0 / rank) for rank in ranks if rank is not None) / total, 4),
    }


def misroute_heatmap(rows: list[AreaQuestion], predictions: list[list[int]], areas: list[Area]) -> list[dict[str, object]]:
    counter: Counter[tuple[int, int]] = Counter()
    for row, predicted in zip(rows, predictions, strict=True):
        top = predicted[0] if predicted else None
        if top is not None and top != row.area_index:
            counter[(row.area_index, top)] += 1
    return [
        {
            "gold_area": areas[gold].area_id,
            "predicted_area": areas[predicted].area_id,
            "distance": predicted - gold,
            "count": count,
        }
        for (gold, predicted), count in counter.most_common(30)
    ]


def boundary_adaptation_candidates(rows: list[AreaQuestion], predictions: list[list[int]], areas: list[Area]) -> list[dict[str, object]]:
    by_article: Counter[tuple[str, int, int]] = Counter()
    for row, predicted in zip(rows, predictions, strict=True):
        top = predicted[0] if predicted else None
        if top is not None and top != row.area_index:
            by_article[(row.article_ref, row.area_index, top)] += 1
    candidates = []
    for (article_ref, gold, predicted), count in by_article.most_common(50):
        near_boundary = min(
            abs(index - areas[gold].start_index)
            for index, ref in enumerate([ref for area in areas for ref in area.refs])
            if ref == article_ref
        )
        candidates.append(
            {
                "article_ref": article_ref,
                "gold_area": areas[gold].area_id,
                "predicted_area": areas[predicted].area_id,
                "distance": predicted - gold,
                "count": count,
                "note": "adjacent-boundary-candidate" if abs(predicted - gold) == 1 and near_boundary <= 8 else "semantic-misroute",
            }
        )
    return candidates[:30]


if __name__ == "__main__":
    main()
