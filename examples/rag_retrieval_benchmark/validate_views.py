"""Validate generated retrieval views by retrieving each view question."""

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
    anchor_units,
    evaluate,
    load_jsonl,
    view_anchor_units,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated refmark retrieval views.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--views-jsonl", required=True)
    parser.add_argument("--top-ks", default="1,3,5")
    parser.add_argument("--index", choices=["raw", "views"], default="views")
    parser.add_argument("--output", default=str(OUTPUT_DIR / "view_validation.json"))
    args = parser.parse_args()

    anchors = load_jsonl(Path(args.data_dir) / "anchors.jsonl")
    views = load_jsonl(Path(args.views_jsonl))
    top_ks = tuple(int(part) for part in args.top_ks.split(",") if part.strip())
    examples = [
        {
            "question": question,
            "refmark": row["refmark"],
        }
        for row in views
        for question in row.get("questions", [])
        if str(question).strip()
    ]
    units = anchor_units(anchors) if args.index == "raw" else view_anchor_units(anchors, views)
    report = evaluate(BM25Index(units), examples, top_ks=top_ks)
    payload = {
        "data_dir": args.data_dir,
        "views_jsonl": args.views_jsonl,
        "index": args.index,
        "generated_questions": len(examples),
        "report": report,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nWrote view validation to {output}")


if __name__ == "__main__":
    main()
