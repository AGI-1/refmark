"""Evaluate fielded static BGB retrieval on stress questions.

This tests a no-vector-runtime adaptation path. Instead of pasting generated
questions into the same BM25 text as the source, it scores source text,
generated summaries/questions/keywords, and held-out-safe training aliases as
separate fields, then combines ranks with weighted reciprocal-rank fusion.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.bgb_browser_search.adapt_bgb_static_views import (  # noqa: E402
    alias_text,
    article_id_from_ref,
    article_regions,
    keyword_terms,
    split_questions_by_block,
    stress_questions,
    summarize_ranks,
    unique,
)
from examples.bgb_browser_search.run_bgb_stress_eval import StressQuestion  # noqa: E402
from refmark.search_index import PortableBM25Index, RetrievalView, SearchHit, SearchRegion, load_search_index  # noqa: E402


FieldSearchFn = Callable[[str, int], list[SearchHit]]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Evaluate fielded static BGB retrieval on generated stress questions.")
    parser.add_argument("--index", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_openrouter_index.json")
    parser.add_argument("--stress-report", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=515)
    parser.add_argument("--top-ks", default="1,3,5,10,20,50")
    parser.add_argument("--candidate-k", type=int, default=80)
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--max-aliases-per-article", type=int, default=16)
    args = parser.parse_args()

    source_index = load_search_index(args.index)
    stress = json.loads(Path(args.stress_report).read_text(encoding="utf-8"))
    train_questions, eval_questions = split_questions_by_block(
        stress_questions(stress),
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    base_regions = article_regions(source_index.regions)
    top_ks = tuple(int(part) for part in args.top_ks.split(",") if part.strip())

    fields = build_field_indexes(
        base_regions,
        train_questions,
        max_aliases_per_article=args.max_aliases_per_article,
    )
    plans = retrieval_plans(fields, candidate_k=args.candidate_k, rrf_k=args.rrf_k)
    results = {
        name: evaluate_fielded(search_fn, eval_questions, top_ks=top_ks)
        for name, search_fn in plans.items()
    }
    report = {
        "schema": "refmark.bgb_fielded_static_eval.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_index": args.index,
        "stress_report": args.stress_report,
        "settings": vars(args),
        "article_count": len(base_regions),
        "train_questions": len(train_questions),
        "eval_questions": len(eval_questions),
        "field_document_lengths": {
            name: {
                "regions": len(index.regions),
                "avg_tokens": round(index.avg_len, 2),
            }
            for name, index in fields.items()
        },
        "results": results,
        "deltas_vs_baseline": deltas(results, baseline="baseline_source_plus_views"),
        "best_by_hit_at_10": best_plan(results, k="10"),
    }
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def build_field_indexes(
    base_regions: list[SearchRegion],
    train_questions: list[StressQuestion],
    *,
    max_aliases_per_article: int,
) -> dict[str, PortableBM25Index]:
    aliases_by_ref: dict[str, list[str]] = {}
    for question in train_questions:
        aliases_by_ref.setdefault(question.block_id, []).append(alias_text(question))

    return {
        "baseline_source_plus_views": PortableBM25Index(base_regions, include_source=True),
        "source": PortableBM25Index([source_region(region) for region in base_regions], include_source=True),
        "summary": PortableBM25Index([summary_region(region) for region in base_regions], include_source=False),
        "questions": PortableBM25Index([questions_region(region) for region in base_regions], include_source=False),
        "keywords": PortableBM25Index([keywords_region(region) for region in base_regions], include_source=False),
        "train_aliases": PortableBM25Index(
            [
                alias_region(
                    region,
                    unique(aliases_by_ref.get(region.stable_ref, []))[:max_aliases_per_article],
                )
                for region in base_regions
            ],
            include_source=False,
        ),
    }


def retrieval_plans(
    fields: dict[str, PortableBM25Index],
    *,
    candidate_k: int,
    rrf_k: float,
) -> dict[str, FieldSearchFn]:
    return {
        "baseline_source_plus_views": lambda query, top_k: fields["baseline_source_plus_views"].search(query, top_k=top_k),
        "source_only": lambda query, top_k: fields["source"].search(query, top_k=top_k),
        "summary_only": lambda query, top_k: fields["summary"].search(query, top_k=top_k),
        "questions_only": lambda query, top_k: fields["questions"].search(query, top_k=top_k),
        "keywords_only": lambda query, top_k: fields["keywords"].search(query, top_k=top_k),
        "train_aliases_only": lambda query, top_k: fields["train_aliases"].search(query, top_k=top_k),
        "fielded_original_rrf": lambda query, top_k: rrf_search(
            query,
            top_k=top_k,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            fields={
                "source": (fields["source"], 1.0),
                "summary": (fields["summary"], 0.45),
                "questions": (fields["questions"], 0.75),
                "keywords": (fields["keywords"], 0.35),
            },
        ),
        "fielded_with_train_aliases_rrf": lambda query, top_k: rrf_search(
            query,
            top_k=top_k,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            fields={
                "source": (fields["source"], 1.0),
                "summary": (fields["summary"], 0.4),
                "questions": (fields["questions"], 0.65),
                "keywords": (fields["keywords"], 0.3),
                "train_aliases": (fields["train_aliases"], 0.65),
            },
        ),
        "source_plus_train_aliases_rrf": lambda query, top_k: rrf_search(
            query,
            top_k=top_k,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            fields={
                "source": (fields["source"], 1.0),
                "train_aliases": (fields["train_aliases"], 0.8),
            },
        ),
    }


def source_region(region: SearchRegion) -> SearchRegion:
    return replace(region, view=empty_view())


def summary_region(region: SearchRegion) -> SearchRegion:
    return replace(region, text="", view=RetrievalView(summary=region.view.summary, questions=[], keywords=[]))


def questions_region(region: SearchRegion) -> SearchRegion:
    return replace(region, text="", view=RetrievalView(summary="", questions=region.view.questions, keywords=[]))


def keywords_region(region: SearchRegion) -> SearchRegion:
    return replace(region, text="", view=RetrievalView(summary="", questions=[], keywords=region.view.keywords))


def alias_region(region: SearchRegion, aliases: list[str]) -> SearchRegion:
    return replace(
        region,
        text="",
        view=RetrievalView(summary="", questions=aliases, keywords=keyword_terms(aliases)),
    )


def empty_view() -> RetrievalView:
    return RetrievalView(summary="", questions=[], keywords=[])


def rrf_search(
    query: str,
    *,
    top_k: int,
    candidate_k: int,
    rrf_k: float,
    fields: dict[str, tuple[PortableBM25Index, float]],
) -> list[SearchHit]:
    scores: dict[str, float] = {}
    refs: dict[str, SearchHit] = {}
    for _name, (index, weight) in fields.items():
        for rank, hit in enumerate(index.search(query, top_k=max(top_k, candidate_k)), start=1):
            scores[hit.stable_ref] = scores.get(hit.stable_ref, 0.0) + (weight / (rrf_k + rank))
            refs.setdefault(hit.stable_ref, hit)
    ordered_refs = sorted(scores, key=lambda ref: (-scores[ref], ref))[:top_k]
    return [replace(refs[ref], rank=rank, score=round(scores[ref], 6)) for rank, ref in enumerate(ordered_refs, start=1)]


def evaluate_fielded(search_fn: FieldSearchFn, questions: list[StressQuestion], *, top_ks: tuple[int, ...]) -> dict[str, object]:
    max_k = max(top_ks)
    ranks: list[int | None] = []
    by_language: dict[str, list[int | None]] = {}
    by_style: dict[str, list[int | None]] = {}
    misses_by_block: dict[str, int] = {}
    for question in questions:
        hits = search_fn(question.query, max_k)
        gold_articles = {article_id_from_ref(ref) for ref in question.gold_refs}
        rank = None
        for index, hit in enumerate(hits, start=1):
            if article_id_from_ref(hit.stable_ref) in gold_articles:
                rank = index
                break
        ranks.append(rank)
        by_language.setdefault(question.language, []).append(rank)
        by_style.setdefault(question.style, []).append(rank)
        if rank is None:
            misses_by_block[question.block_id] = misses_by_block.get(question.block_id, 0) + 1
    return {
        "article_hit_at_k": summarize_ranks(ranks, top_ks=top_ks),
        "by_language": {name: summarize_ranks(values, top_ks=top_ks) for name, values in sorted(by_language.items())},
        "by_style": {name: summarize_ranks(values, top_ks=top_ks) for name, values in sorted(by_style.items())},
        "misses_by_block_top": [
            {"block_id": block_id, "misses": misses}
            for block_id, misses in sorted(misses_by_block.items(), key=lambda item: (-item[1], item[0]))[:20]
        ],
    }


def deltas(results: dict[str, dict[str, object]], *, baseline: str) -> dict[str, dict[str, float]]:
    baseline_summary = results[baseline]["article_hit_at_k"]
    baseline_hits = baseline_summary["hit_at_k"]
    output: dict[str, dict[str, float]] = {}
    for name, result in results.items():
        summary = result["article_hit_at_k"]
        hits = summary["hit_at_k"]
        output[name] = {
            **{f"hit@{k}": round(float(value) - float(baseline_hits.get(k, 0.0)), 4) for k, value in hits.items()},
            "mrr": round(float(summary["mrr"]) - float(baseline_summary["mrr"]), 4),
        }
    return output


def best_plan(results: dict[str, dict[str, object]], *, k: str) -> dict[str, object]:
    name, result = max(
        results.items(),
        key=lambda item: (float(item[1]["article_hit_at_k"]["hit_at_k"][k]), float(item[1]["article_hit_at_k"]["mrr"])),
    )
    summary = result["article_hit_at_k"]
    return {"name": name, "hit_at_k": k, "hit": summary["hit_at_k"][k], "mrr": summary["mrr"]}


if __name__ == "__main__":
    main()
