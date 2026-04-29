"""Sweep compact embedding reducers for coarse BGB area routing.

This is a focused follow-up to ``evaluate_bgb_embedding_area_reduction.py``.
It fixes a coarse article area size, then asks whether the embedding dimensions
can be reduced, sliced, shuffled, or quantized while preserving area signal.
The script deliberately separates cheap centroid probes from the slower MLP
head training step so random reducer variants can be explored quickly.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.bgb_browser_search.adapt_bgb_static_views import article_regions  # noqa: E402
from examples.bgb_browser_search.evaluate_bgb_embedding_area_reduction import (  # noqa: E402
    Area,
    evaluate_projection,
    pca_projection,
    quantize_int8,
    random_projection,
    read_query_vectors,
    summarize_predictions,
)
from examples.bgb_browser_search.train_bgb_article_candidate_generator import load_split_questions  # noqa: E402
from refmark.search_index import load_search_index  # noqa: E402

try:
    import torch
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This experiment requires torch.") from exc


@dataclass(frozen=True)
class ProjectionCandidate:
    kind: str
    dimension: int
    matrix: torch.Tensor | None
    columns: torch.Tensor | None
    seed: int | None
    meta: dict[str, object]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Sweep BGB coarse-area embedding reducers.")
    parser.add_argument("--index", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_confusion_signatures_gemma_top300_index.json")
    parser.add_argument("--stress-report", action="append", required=True)
    parser.add_argument("--query-cache", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--embedding-model", default="qwen/qwen3-embedding-8b")
    parser.add_argument("--area-size", type=int, default=300)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=5151)
    parser.add_argument("--dims", default="512,256,128,64,32,16,8")
    parser.add_argument("--random-trials", type=int, default=16)
    parser.add_argument("--mlp-random-top", type=int, default=8)
    parser.add_argument("--mlp-random-top-per-dim", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--learning-rate", type=float, default=0.0008)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dims = tuple(int(part) for part in args.dims.split(",") if part.strip())
    top_ks = (1, 2, 3, 5)
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

    centroid_rows: list[dict[str, object]] = []
    mlp_candidates: list[ProjectionCandidate] = []

    full_candidate = ProjectionCandidate(
        kind="identity",
        dimension=input_dim,
        matrix=None,
        columns=None,
        seed=None,
        meta={
            "kind": "identity",
            "dimension": input_dim,
            "projection_matrix_size_bytes": 0,
            "projection_storage_note": "No reducer; uses cached embedding directly.",
        },
    )
    centroid_rows.extend(evaluate_centroids(full_candidate, train_x, eval_x, train_y, eval_y, top_ks=top_ks))
    mlp_candidates.append(full_candidate)

    deterministic: list[ProjectionCandidate] = []
    for dim in dims:
        if dim >= input_dim:
            continue
        deterministic.extend(
            [
                ProjectionCandidate(
                    kind="pca_projection",
                    dimension=dim,
                    matrix=pca_projection(train_x, dim),
                    columns=None,
                    seed=None,
                    meta={
                        "kind": "pca_projection",
                        "dimension": dim,
                        "projection_matrix_size_bytes": int(input_dim * dim * 4),
                        "projection_storage_note": "FP32 PCA matrix size; can be quantized separately.",
                    },
                ),
                column_candidate("variance_select", dim, top_variance_columns(train_x, dim)),
                column_candidate("fisher_select", dim, fisher_columns(train_x, train_y, dim)),
                column_candidate("contiguous_head_slice", dim, torch.arange(dim)),
                column_candidate("contiguous_tail_slice", dim, torch.arange(input_dim - dim, input_dim)),
            ]
        )

    for candidate in deterministic:
        centroid_rows.extend(evaluate_centroids(candidate, train_x, eval_x, train_y, eval_y, top_ks=top_ks))
        mlp_candidates.append(candidate)

    random_candidates: list[ProjectionCandidate] = []
    for dim in dims:
        if dim >= input_dim:
            continue
        for trial in range(args.random_trials):
            seed = args.seed + (dim * 1000) + trial
            random_candidates.extend(
                [
                    ProjectionCandidate(
                        kind="random_projection",
                        dimension=dim,
                        matrix=random_projection(input_dim, dim, seed=seed),
                        columns=None,
                        seed=seed,
                        meta={
                            "kind": "random_projection",
                            "dimension": dim,
                            "seed": seed,
                            "projection_matrix_size_bytes": 0,
                            "projection_storage_note": "Matrix can be regenerated from the seed; compute still needs the full embedding.",
                        },
                    ),
                    column_candidate(
                        "random_coordinate_sample",
                        dim,
                        random_columns(input_dim, dim, seed=seed),
                        seed=seed,
                    ),
                ]
            )

    random_centroid_groups: list[tuple[float, ProjectionCandidate, list[dict[str, object]]]] = []
    for candidate in random_candidates:
        rows = evaluate_centroids(candidate, train_x, eval_x, train_y, eval_y, top_ks=top_ks)
        centroid_rows.extend(rows)
        best_score = max(score_row(row) for row in rows)
        random_centroid_groups.append((best_score, candidate, rows))
    selected_random: list[tuple[float, ProjectionCandidate, list[dict[str, object]]]] = []
    selected_random.extend(sorted(random_centroid_groups, key=lambda item: item[0], reverse=True)[: args.mlp_random_top])
    by_dim_kind: dict[tuple[int, str], list[tuple[float, ProjectionCandidate, list[dict[str, object]]]]] = {}
    for item in random_centroid_groups:
        _score, candidate, _rows = item
        by_dim_kind.setdefault((candidate.dimension, candidate.kind), []).append(item)
    for items in by_dim_kind.values():
        selected_random.extend(sorted(items, key=lambda item: item[0], reverse=True)[: args.mlp_random_top_per_dim])
    for _score, candidate, _rows in selected_random:
        mlp_candidates.append(candidate)

    mlp_rows: list[dict[str, object]] = []
    seen_mlp_keys: set[tuple[object, ...]] = set()
    for candidate in mlp_candidates:
        key = (candidate.kind, candidate.dimension, candidate.seed, tuple(candidate.columns.tolist()) if candidate.columns is not None else None)
        if key in seen_mlp_keys:
            continue
        seen_mlp_keys.add(key)
        train_proj, eval_proj = apply_candidate(candidate, train_x, eval_x)
        mlp_row = evaluate_projection(
            train_proj,
            eval_proj,
            train_y,
            eval_y,
            candidate.meta,
            args=args,
            top_ks=top_ks,
        )[-1]
        mlp_rows.append(mlp_row)

    report = {
        "schema": "refmark.bgb_area_reducer_sweep.v1",
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
        "centroid_results": sorted(centroid_rows, key=score_row, reverse=True),
        "mlp_results": sorted(mlp_rows, key=score_row, reverse=True),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def evaluate_centroids(
    candidate: ProjectionCandidate,
    train_x: torch.Tensor,
    eval_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_y: torch.Tensor,
    *,
    top_ks: tuple[int, ...],
) -> list[dict[str, object]]:
    train_proj, eval_proj = apply_candidate(candidate, train_x, eval_x)
    centroid_matrix = centroids(train_proj, train_y)
    rows = []
    for quantization, train_values, eval_values in (
        ("fp32", train_proj, eval_proj),
        ("int8", quantize_int8(train_proj), quantize_int8(eval_proj)),
    ):
        matrix = centroids(train_values, train_y)
        rows.append(
            {
                **candidate.meta,
                "router": "nearest_centroid",
                "quantization": quantization,
                "metrics": summarize_predictions(eval_values @ matrix.T, eval_y, top_ks=top_ks),
                "size_estimate_bytes": int(centroid_matrix.numel() * (1 if quantization == "int8" else 4)),
            }
        )
    return rows


def apply_candidate(candidate: ProjectionCandidate, train_x: torch.Tensor, eval_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if candidate.matrix is not None:
        return F.normalize(train_x @ candidate.matrix, dim=1), F.normalize(eval_x @ candidate.matrix, dim=1)
    if candidate.columns is not None:
        return F.normalize(train_x[:, candidate.columns], dim=1), F.normalize(eval_x[:, candidate.columns], dim=1)
    return train_x, eval_x


def centroids(values: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    output = []
    for class_index in range(int(labels.max()) + 1):
        mask = labels == class_index
        if bool(mask.any()):
            output.append(F.normalize(values[mask].mean(dim=0), dim=0))
        else:
            output.append(torch.zeros(values.shape[1], dtype=values.dtype))
    return torch.stack(output)


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


def column_candidate(kind: str, dim: int, columns: torch.Tensor, *, seed: int | None = None) -> ProjectionCandidate:
    return ProjectionCandidate(
        kind=kind,
        dimension=dim,
        matrix=None,
        columns=columns.long(),
        seed=seed,
        meta={
            "kind": kind,
            "dimension": dim,
            "seed": seed,
            "projection_matrix_size_bytes": 0,
            "projection_storage_note": "Stores selected coordinate ids, not a dense projection matrix.",
        },
    )


def top_variance_columns(values: torch.Tensor, dim: int) -> torch.Tensor:
    scores = values.var(dim=0, unbiased=False)
    return torch.topk(scores, k=dim).indices.sort().values


def fisher_columns(values: torch.Tensor, labels: torch.Tensor, dim: int) -> torch.Tensor:
    global_mean = values.mean(dim=0)
    between = torch.zeros(values.shape[1])
    within = torch.zeros(values.shape[1])
    for class_index in range(int(labels.max()) + 1):
        class_values = values[labels == class_index]
        if len(class_values) == 0:
            continue
        class_mean = class_values.mean(dim=0)
        between += len(class_values) * (class_mean - global_mean).pow(2)
        within += ((class_values - class_mean).pow(2)).sum(dim=0)
    scores = between / within.clamp_min(1e-9)
    return torch.topk(scores, k=dim).indices.sort().values


def random_columns(input_dim: int, dim: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(input_dim, generator=generator)[:dim].sort().values


def score_row(row: dict[str, object]) -> float:
    hit_at_k = row["metrics"]["hit_at_k"]  # type: ignore[index]
    return (
        float(hit_at_k.get("1", 0.0))  # type: ignore[union-attr]
        + 0.25 * float(hit_at_k.get("2", 0.0))  # type: ignore[union-attr]
        + 0.05 * float(hit_at_k.get("5", 0.0))  # type: ignore[union-attr]
    )


if __name__ == "__main__":
    main()
