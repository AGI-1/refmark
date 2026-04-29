"""Evaluate compressed embedding signals for BGB coarse area routing.

The question is whether a high-quality embedding router can be reduced enough
to act as a cheap coarse navigation layer:

    query embedding -> ~17 areas of ~150 articles -> finer resolver

This script compares full embeddings, random projections, PCA/SVD projections,
prototype routing, and simple quantization using cached query embeddings.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.bgb_browser_search.adapt_bgb_static_views import article_regions  # noqa: E402
from examples.bgb_browser_search.train_bgb_article_candidate_generator import load_split_questions  # noqa: E402
from refmark.search_index import load_search_index  # noqa: E402

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This experiment requires torch.") from exc


@dataclass(frozen=True)
class Area:
    area_id: str
    start_index: int
    end_index: int
    start_ref: str
    end_ref: str
    refs: tuple[str, ...]


class EmbeddingHead(nn.Module):
    def __init__(self, input_dim: int, class_count: int, *, hidden_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        current = input_dim
        for _ in range(max(layers, 0)):
            modules.extend([nn.Linear(current, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            current = hidden_dim
        modules.append(nn.Linear(current, class_count))
        self.net = nn.Sequential(*modules)

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        return self.net(vectors)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Evaluate compressed BGB embedding area routing.")
    parser.add_argument("--index", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_confusion_signatures_gemma_top300_index.json")
    parser.add_argument("--stress-report", action="append", required=True)
    parser.add_argument("--query-cache", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--embedding-model", default="qwen/qwen3-embedding-8b")
    parser.add_argument("--area-size", type=int, default=150)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=4141)
    parser.add_argument("--dims", default="4096,1024,512,256,128,64,32,16")
    parser.add_argument("--top-ks", default="1,2,3,5")
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--learning-rate", type=float, default=0.0008)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--skip-pca", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dims = tuple(int(part) for part in args.dims.split(",") if part.strip())
    top_ks = tuple(int(part) for part in args.top_ks.split(",") if part.strip())
    started = time.perf_counter()

    articles = article_regions(load_search_index(args.index).regions)
    areas = equal_areas([article.stable_ref for article in articles], area_size=args.area_size)
    area_by_ref = {ref: area_index for area_index, area in enumerate(areas) for ref in area.refs}
    query_vectors = read_query_vectors([Path(path) for path in args.query_cache], model=args.embedding_model)
    train_rows, eval_rows = load_split_questions(args.stress_report, train_fraction=args.train_fraction, seed=args.seed)
    train_rows = [row for row in train_rows if row.query in query_vectors and row.article_ref in area_by_ref]
    eval_rows = [row for row in eval_rows if row.query in query_vectors and row.article_ref in area_by_ref]
    if not train_rows or not eval_rows:
        raise SystemExit("No train/eval rows matched the query embedding cache.")

    train_x = torch.stack([query_vectors[row.query] for row in train_rows])
    eval_x = torch.stack([query_vectors[row.query] for row in eval_rows])
    train_y = torch.tensor([area_by_ref[row.article_ref] for row in train_rows], dtype=torch.long)
    eval_y = torch.tensor([area_by_ref[row.article_ref] for row in eval_rows], dtype=torch.long)
    input_dim = int(train_x.shape[1])

    results = []
    for dim in dims:
        if dim > input_dim:
            continue
        if dim == input_dim:
            train_proj = train_x
            eval_proj = eval_x
            projection_meta = {
                "kind": "identity",
                "dimension": dim,
                "projection_matrix_size_bytes": 0,
                "projection_storage_note": "No projection matrix; uses cached embedding directly.",
            }
            results.extend(evaluate_projection(train_proj, eval_proj, train_y, eval_y, projection_meta, args=args, top_ks=top_ks))
            continue
        random_matrix = random_projection(input_dim, dim, seed=args.seed + dim)
        train_proj = F.normalize(train_x @ random_matrix, dim=1)
        eval_proj = F.normalize(eval_x @ random_matrix, dim=1)
        projection_meta = {
            "kind": "random_projection",
            "dimension": dim,
            "projection_matrix_size_bytes": 0,
            "projection_storage_note": "Matrix can be regenerated from the seed; compute still needs the full embedding.",
        }
        results.extend(evaluate_projection(train_proj, eval_proj, train_y, eval_y, projection_meta, args=args, top_ks=top_ks))
        if not args.skip_pca:
            pca_matrix = pca_projection(train_x, dim)
            train_pca = F.normalize(train_x @ pca_matrix, dim=1)
            eval_pca = F.normalize(eval_x @ pca_matrix, dim=1)
            results.extend(
                evaluate_projection(
                    train_pca,
                    eval_pca,
                    train_y,
                    eval_y,
                    {
                        "kind": "pca_projection",
                        "dimension": dim,
                        "projection_matrix_size_bytes": int(input_dim * dim * 4),
                        "projection_storage_note": "FP32 PCA matrix size; can be quantized separately.",
                    },
                    args=args,
                    top_ks=top_ks,
                )
            )

    report = {
        "schema": "refmark.bgb_embedding_area_reduction.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_index": args.index,
        "stress_reports": args.stress_report,
        "query_caches": args.query_cache,
        "embedding_model": args.embedding_model,
        "settings": vars(args),
        "article_count": len(articles),
        "area_size": args.area_size,
        "area_count": len(areas),
        "areas": [area.__dict__ for area in areas],
        "embedding_dim": input_dim,
        "train_questions": len(train_rows),
        "eval_questions": len(eval_rows),
        "seconds": round(time.perf_counter() - started, 3),
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def evaluate_projection(
    train_x: torch.Tensor,
    eval_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_y: torch.Tensor,
    projection_meta: dict[str, object],
    *,
    args: argparse.Namespace,
    top_ks: tuple[int, ...],
) -> list[dict[str, object]]:
    rows = []
    rows.append(
        {
            **projection_meta,
            "router": "nearest_centroid",
            "quantization": "fp32",
            "metrics": prototype_metrics(train_x, eval_x, train_y, eval_y, top_ks=top_ks),
            "size_estimate_bytes": int(train_x.shape[1] * (int(train_y.max()) + 1) * 4),
        }
    )
    rows.append(
        {
            **projection_meta,
            "router": "nearest_centroid",
            "quantization": "int8",
            "metrics": prototype_metrics(quantize_int8(train_x), quantize_int8(eval_x), train_y, eval_y, top_ks=top_ks),
            "size_estimate_bytes": int(train_x.shape[1] * (int(train_y.max()) + 1)),
        }
    )
    head_metrics, head_info = train_head(
        train_x,
        eval_x,
        train_y,
        eval_y,
        class_count=int(train_y.max()) + 1,
        args=args,
        top_ks=top_ks,
    )
    rows.append(
        {
            **projection_meta,
            "router": "mlp_head",
            "quantization": "fp32",
            "metrics": head_metrics,
            **head_info,
        }
    )
    return rows


def train_head(
    train_x: torch.Tensor,
    eval_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_y: torch.Tensor,
    *,
    class_count: int,
    args: argparse.Namespace,
    top_ks: tuple[int, ...],
) -> tuple[dict[str, object], dict[str, object]]:
    torch.manual_seed(args.seed + int(train_x.shape[1]))
    model = EmbeddingHead(
        int(train_x.shape[1]),
        class_count,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        dropout=args.dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_score = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x_batch, y_batch in batches(train_x, train_y, batch_size=args.batch_size, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x_batch), y_batch, label_smoothing=args.label_smoothing)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = model_metrics(model, eval_x, eval_y, top_ks=top_ks)
        score = float(metrics["hit_at_k"].get("1", 0.0)) + (0.25 * float(metrics["hit_at_k"].get("2", 0.0)))
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        history.append(
            {
                "epoch": epoch,
                "loss": round(sum(losses) / max(len(losses), 1), 6),
                "hit_at_1": metrics["hit_at_k"].get("1", 0.0),
                "hit_at_2": metrics["hit_at_k"].get("2", 0.0),
                "hit_at_3": metrics["hit_at_k"].get("3", 0.0),
            }
        )
    model.load_state_dict(best_state)
    parameters = sum(param.numel() for param in model.parameters())
    return model_metrics(model, eval_x, eval_y, top_ks=top_ks), {
        "parameters": parameters,
        "size_estimate_bytes": int(parameters * 4),
        "best_epoch": max(history, key=lambda row: float(row["hit_at_1"]) + (0.25 * float(row["hit_at_2"])))["epoch"],
        "history_tail": history[-5:],
    }


def prototype_metrics(
    train_x: torch.Tensor,
    eval_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_y: torch.Tensor,
    *,
    top_ks: tuple[int, ...],
) -> dict[str, object]:
    class_count = int(train_y.max()) + 1
    centroids = []
    for class_index in range(class_count):
        mask = train_y == class_index
        if bool(mask.any()):
            centroids.append(F.normalize(train_x[mask].mean(dim=0), dim=0))
        else:
            centroids.append(torch.zeros(train_x.shape[1], dtype=train_x.dtype))
    centroid_matrix = torch.stack(centroids)
    scores = eval_x @ centroid_matrix.T
    return summarize_predictions(scores, eval_y, top_ks=top_ks)


def model_metrics(model: nn.Module, eval_x: torch.Tensor, eval_y: torch.Tensor, *, top_ks: tuple[int, ...]) -> dict[str, object]:
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(eval_x), 512):
            chunks.append(model(eval_x[start : start + 512]))
    return summarize_predictions(torch.cat(chunks, dim=0), eval_y, top_ks=top_ks)


def summarize_predictions(scores: torch.Tensor, labels: torch.Tensor, *, top_ks: tuple[int, ...]) -> dict[str, object]:
    max_k = min(max(top_ks), scores.shape[1])
    top = torch.topk(scores, k=max_k, dim=1).indices.tolist()
    ranks = []
    for predicted, gold in zip(top, labels.tolist(), strict=True):
        rank = None
        for index, value in enumerate(predicted, start=1):
            if int(value) == int(gold):
                rank = index
                break
        ranks.append(rank)
    total = max(len(ranks), 1)
    return {
        "hit_at_k": {
            str(k): round(sum(1 for rank in ranks if rank is not None and rank <= k) / total, 4)
            for k in top_ks
        },
        "mrr": round(sum(1.0 / rank for rank in ranks if rank is not None) / total, 4),
    }


def equal_areas(article_refs: list[str], *, area_size: int) -> list[Area]:
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


def random_projection(input_dim: int, output_dim: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(input_dim, output_dim, generator=generator)
    return F.normalize(matrix, dim=0)


def pca_projection(train_x: torch.Tensor, output_dim: int) -> torch.Tensor:
    centered = train_x - train_x.mean(dim=0, keepdim=True)
    _u, _s, v = torch.pca_lowrank(centered, q=output_dim, center=False)
    return v[:, :output_dim]


def quantize_int8(values: torch.Tensor) -> torch.Tensor:
    scale = values.abs().amax(dim=1, keepdim=True).clamp_min(1e-9) / 127.0
    return F.normalize((values / scale).round().clamp(-127, 127) * scale, dim=1)


def batches(x: torch.Tensor, y: torch.Tensor, *, batch_size: int, shuffle: bool):
    indices = list(range(len(x)))
    if shuffle:
        random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        yield x[batch], y[batch]


if __name__ == "__main__":
    main()
