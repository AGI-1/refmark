"""Evaluate adjacent-article tolerance for BGB navigation.

Exact article hits are the strict retrieval target. For a browser/navigation UI,
however, landing one or two articles away is still useful if the result opens a
small local cluster. This report measures that local-neighborhood behavior
without changing the underlying retrieval ranking.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
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
from refmark.search_index import PortableBM25Index, load_search_index  # noqa: E402


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Evaluate BGB adjacent-article hit windows.")
    parser.add_argument("--index", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_confusion_signatures_gemma_top300_index.json")
    parser.add_argument("--stress-report", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("all", "train", "eval"), default="eval")
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1515)
    parser.add_argument("--top-ks", default="1,3,5,10,20,50,100")
    parser.add_argument("--windows", default="0,1,2,3,5")
    parser.add_argument("--candidate-k", type=int, default=100)
    args = parser.parse_args()

    source_index = load_search_index(args.index)
    articles = article_regions(source_index.regions)
    search = PortableBM25Index(articles, include_source=True)
    article_positions = {region.stable_ref: index for index, region in enumerate(articles)}
    questions = load_questions(
        args.stress_report,
        split=args.split,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    top_ks = tuple(int(part) for part in args.top_ks.split(",") if part.strip())
    windows = tuple(int(part) for part in args.windows.split(",") if part.strip())

    started = time.perf_counter()
    hit_positions: list[list[int | None]] = []
    for question in questions:
        hits = search.search(question.query, top_k=args.candidate_k)
        hit_positions.append([article_positions.get(article_id_from_ref(hit.stable_ref)) for hit in hits])

    summaries = []
    by_style = {window: {} for window in windows}
    by_language = {window: {} for window in windows}
    for window in windows:
        ranks = [neighbor_rank(question, positions, article_positions, window=window) for question, positions in zip(questions, hit_positions, strict=True)]
        summaries.append({"window": window, "summary": summarize_ranks(ranks, top_ks=top_ks)})
        for key, grouped in (("style", by_style[window]), ("language", by_language[window])):
            values: dict[str, list[int | None]] = {}
            for question, positions in zip(questions, hit_positions, strict=True):
                values.setdefault(getattr(question, key), []).append(
                    neighbor_rank(question, positions, article_positions, window=window)
                )
            grouped.update({name: summarize_ranks(ranks, top_ks=top_ks) for name, ranks in sorted(values.items())})

    report = {
        "schema": "refmark.bgb_neighbor_window_eval.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_index": args.index,
        "stress_reports": args.stress_report,
        "settings": vars(args),
        "question_count": len(questions),
        "article_count": len(articles),
        "seconds": round(time.perf_counter() - started, 3),
        "windows": summaries,
        "by_style": {str(window): rows for window, rows in by_style.items()},
        "by_language": {str(window): rows for window, rows in by_language.items()},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def load_questions(paths: list[str], *, split: str, train_fraction: float, seed: int) -> list[StressQuestion]:
    rows: list[StressQuestion] = []
    for offset, path in enumerate(paths):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        questions = stress_questions(payload)
        if split != "all":
            train, eval_rows = split_questions_by_block(questions, train_fraction=train_fraction, seed=seed + offset)
            questions = train if split == "train" else eval_rows
        rows.extend(questions)
    return rows


def neighbor_rank(
    question: StressQuestion,
    hit_positions: list[int | None],
    article_positions: dict[str, int],
    *,
    window: int,
) -> int | None:
    gold_position = article_positions.get(article_id_from_ref(question.block_id))
    if gold_position is None:
        return None
    for rank, hit_position in enumerate(hit_positions, start=1):
        if hit_position is not None and abs(hit_position - gold_position) <= window:
            return rank
    return None


if __name__ == "__main__":
    main()
