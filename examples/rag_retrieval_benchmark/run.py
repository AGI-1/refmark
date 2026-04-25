"""Compare naive chunk retrieval with refmark-region retrieval on retained data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.rag_retrieval_benchmark.common import (
    BM25Index,
    DEFAULT_DATA_DIR,
    OUTPUT_DIR,
    add_distractor_copies,
    anchor_units,
    enriched_anchor_units,
    evaluate,
    expanded_anchor_units,
    fixed_window_units,
    load_jsonl,
    view_anchor_units,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark refmark retrieval against naive chunk retrieval.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--split", choices=["valid", "reformulated"], default="valid")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--top-ks", default="1,3,5,10")
    parser.add_argument("--chunk-tokens", type=int, default=220)
    parser.add_argument("--chunk-stride", type=int, default=110)
    parser.add_argument("--expand", type=int, default=1)
    parser.add_argument("--views-jsonl", help="Optional generated retrieval views keyed by refmark.")
    parser.add_argument("--view-name", default="generated_views")
    parser.add_argument("--view-source-only", action="store_true", help="Index views without original anchor text.")
    parser.add_argument("--distractor-copies", type=int, default=0, help="Append synthetic distractor copies of all anchors.")
    parser.add_argument("--output", default=str(OUTPUT_DIR / "retrieval_benchmark.json"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    source_anchors = load_jsonl(data_dir / "anchors.jsonl")
    anchors = add_distractor_copies(source_anchors, copies=args.distractor_copies)
    train = load_jsonl(data_dir / "train.jsonl")
    examples = load_jsonl(data_dir / f"{args.split}.jsonl")[: args.limit]
    top_ks = tuple(int(part) for part in args.top_ks.split(",") if part.strip())

    reports = {
        "naive_fixed_chunks": evaluate(
            BM25Index(fixed_window_units(anchors, chunk_tokens=args.chunk_tokens, stride=args.chunk_stride)),
            examples,
            top_ks=top_ks,
        ),
        "refmark_regions": evaluate(BM25Index(anchor_units(anchors)), examples, top_ks=top_ks),
        "refmark_regions_plus_neighbor_expansion": evaluate(
            BM25Index(expanded_anchor_units(anchors, margin=args.expand)),
            examples,
            top_ks=top_ks,
        ),
        "refmark_regions_enriched_with_train_questions": evaluate(
            BM25Index(enriched_anchor_units(anchors, train)),
            examples,
            top_ks=top_ks,
        ),
    }
    if args.views_jsonl:
        views = load_jsonl(Path(args.views_jsonl))
        reports[args.view_name] = evaluate(
            BM25Index(view_anchor_units(anchors, views, include_source=not args.view_source_only)),
            examples,
            top_ks=top_ks,
        )

    payload = {
        "data_dir": str(data_dir),
        "split": args.split,
        "examples": len(examples),
        "anchors": len(source_anchors),
        "indexed_anchors": len(anchors),
        "settings": {
            "top_ks": top_ks,
            "chunk_tokens": args.chunk_tokens,
            "chunk_stride": args.chunk_stride,
            "expand": args.expand,
            "distractor_copies": args.distractor_copies,
            "views_jsonl": args.views_jsonl,
        },
        "reports": reports,
        "interpretation": [
            "naive_fixed_chunks returns larger anonymous windows, so hit rates should be read with token_cost.",
            "refmark_regions returns precise auditable ids.",
            "neighbor expansion tests whether deterministic child-to-parent context improves support coverage.",
            "train-question enrichment tests whether retained supervision improves retrieval without neural training.",
            "views-jsonl lets generated summaries/questions/keywords become retrieval metadata.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nWrote retrieval benchmark to {output}")


if __name__ == "__main__":
    main()
