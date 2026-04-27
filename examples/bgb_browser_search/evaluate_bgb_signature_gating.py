"""Evaluate confidence-gated LLM signature rescue for BGB article search.

LLM-generated signatures improved known hard articles but can add global noise
when mixed into every query path. This script tests a deployable gating shape:
use deterministic intent signatures as the default static index, then switch to
the LLM-repaired index only when an LLM-signature side index shows enough
confidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.bgb_browser_search.adapt_bgb_static_views import article_id_from_ref, article_regions, summarize_ranks  # noqa: E402
from examples.bgb_browser_search.evaluate_bgb_intent_signatures import apply_signatures, build_signatures, load_split, signature_only_regions  # noqa: E402
from examples.bgb_browser_search.evaluate_bgb_llm_intent_signatures import clean_signatures, select_hard_articles, signature_examples  # noqa: E402
from examples.bgb_browser_search.run_bgb_stress_eval import StressQuestion  # noqa: E402
from refmark.search_index import PortableBM25Index, SearchHit, SearchRegion, load_search_index  # noqa: E402


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Evaluate gated BGB LLM intent-signature rescue.")
    parser.add_argument("--index", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_openrouter_index.json")
    parser.add_argument("--stress-report", action="append", required=True)
    parser.add_argument("--heatmap", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_retrieval_heatmap_3cycle.json")
    parser.add_argument("--signature-cache", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_llm_intent_signatures.jsonl")
    parser.add_argument("--report", required=True)
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--max-hard-articles", type=int, default=40)
    parser.add_argument("--signature-limit", type=int, default=24)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1515)
    parser.add_argument("--top-ks", default="1,3,5,10,20,50")
    parser.add_argument("--candidate-k", type=int, default=80)
    parser.add_argument("--thresholds", default="0.5,1,1.5,2,3,4,5,7.5,10,15,20,30")
    parser.add_argument("--default-score-ceilings", default="2,4,6,8,10,12,15,20")
    args = parser.parse_args()

    source_index = load_search_index(args.index)
    base_regions = article_regions(source_index.regions)
    regions_by_ref = {region.stable_ref: region for region in base_regions}
    selected_refs = select_hard_articles(args.heatmap, regions_by_ref, limit=args.max_hard_articles)
    train_questions, eval_questions = load_split(args.stress_report, train_fraction=args.train_fraction, seed=args.seed)
    selected_eval = [row for row in eval_questions if row.block_id in selected_refs]
    top_ks = tuple(int(part) for part in args.top_ks.split(",") if part.strip())
    thresholds = tuple(float(part) for part in args.thresholds.split(",") if part.strip())
    default_score_ceilings = tuple(float(part) for part in args.default_score_ceilings.split(",") if part.strip())

    deterministic_signatures = build_signatures(train_questions, signature_limit=args.signature_limit, min_df=1, max_phrase_tokens=3)
    llm_signatures = load_llm_signatures(
        args.signature_cache,
        selected_refs=selected_refs,
        regions_by_ref=regions_by_ref,
        model=args.model,
        signature_limit=args.signature_limit,
    )
    deterministic_regions = apply_signatures(base_regions, deterministic_signatures)
    llm_regions = apply_signatures(base_regions, llm_signatures)
    combined_regions = apply_signatures(base_regions, merge_signatures(deterministic_signatures, llm_signatures, limit=args.signature_limit))
    side_regions = signature_only_regions(base_regions, llm_signatures)

    indexes = {
        "baseline_source_plus_views": PortableBM25Index(base_regions, include_source=True),
        "deterministic_signatures": PortableBM25Index(deterministic_regions, include_source=True),
        "llm_hard_article_signatures": PortableBM25Index(llm_regions, include_source=True),
        "deterministic_plus_llm": PortableBM25Index(combined_regions, include_source=True),
        "llm_signature_side": PortableBM25Index(side_regions, include_source=False),
    }
    all_runs = precompute(eval_questions, indexes=indexes, candidate_k=args.candidate_k)
    selected_runs = [run for run in all_runs if run.question.block_id in selected_refs]

    report = {
        "schema": "refmark.bgb_signature_gating_eval.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_index": args.index,
        "stress_reports": args.stress_report,
        "heatmap": args.heatmap,
        "settings": vars(args),
        "selected_articles": len(selected_refs),
        "cached_articles": len(llm_signatures),
        "train_questions": len(train_questions),
        "eval_questions": len(eval_questions),
        "selected_eval_questions": len(selected_eval),
        "signature_examples": signature_examples(llm_signatures),
        "all_eval": evaluate_scope(
            all_runs,
            top_ks=top_ks,
            thresholds=thresholds,
            default_score_ceilings=default_score_ceilings,
        ),
        "selected_eval": evaluate_scope(
            selected_runs,
            top_ks=top_ks,
            thresholds=thresholds,
            default_score_ceilings=default_score_ceilings,
        ),
    }
    report["best_all_gate"] = best_gate(report["all_eval"]["gated"], primary_k="10", require_nonnegative_mrr=True)
    report["best_selected_gate"] = best_gate(report["selected_eval"]["gated"], primary_k="10", require_nonnegative_mrr=False)
    report["best_constrained_low_confidence_gate"] = best_constrained_gate(
        all_rows=report["all_eval"]["gated_low_default_confidence"],
        selected_rows=report["selected_eval"]["gated_low_default_confidence"],
        primary_k="10",
        min_global_delta=-0.002,
    )
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


class QueryRun:
    def __init__(self, question: StressQuestion, hits_by_name: dict[str, list[SearchHit]]) -> None:
        self.question = question
        self.hits_by_name = hits_by_name


def precompute(
    questions: list[StressQuestion],
    *,
    indexes: dict[str, PortableBM25Index],
    candidate_k: int,
) -> list[QueryRun]:
    runs: list[QueryRun] = []
    for question in questions:
        runs.append(
            QueryRun(
                question,
                {
                    name: index.search(question.query, top_k=candidate_k)
                    for name, index in indexes.items()
                },
            )
        )
    return runs


def evaluate_scope(
    runs: list[QueryRun],
    *,
    top_ks: tuple[int, ...],
    thresholds: tuple[float, ...],
    default_score_ceilings: tuple[float, ...],
) -> dict[str, object]:
    static = {
        name: summarize_hits([run.question for run in runs], [run.hits_by_name[name] for run in runs], top_ks=top_ks)
        for name in (
            "baseline_source_plus_views",
            "deterministic_signatures",
            "llm_hard_article_signatures",
            "deterministic_plus_llm",
        )
    }
    gated = []
    for threshold in thresholds:
        summary = summarize_gated(runs, top_ks=top_ks, static=static, threshold=threshold)
        summary["threshold"] = threshold
        gated.append(summary)
    gated_low_default = []
    for threshold in thresholds:
        for ceiling in default_score_ceilings:
            summary = summarize_gated(
                runs,
                top_ks=top_ks,
                static=static,
                threshold=threshold,
                default_score_ceiling=ceiling,
            )
            summary["threshold"] = threshold
            summary["default_score_ceiling"] = ceiling
            gated_low_default.append(summary)
    return {"static": static, "gated": gated, "gated_low_default_confidence": gated_low_default}


def summarize_gated(
    runs: list[QueryRun],
    *,
    top_ks: tuple[int, ...],
    static: dict[str, object],
    threshold: float,
    default_score_ceiling: float | None = None,
) -> dict[str, object]:
    gated_hits = []
    switches = 0
    for run in runs:
        if should_switch(run, threshold=threshold, default_score_ceiling=default_score_ceiling):
            gated_hits.append(run.hits_by_name["llm_hard_article_signatures"])
            switches += 1
        else:
            gated_hits.append(run.hits_by_name["deterministic_signatures"])
    summary = summarize_hits([run.question for run in runs], gated_hits, top_ks=top_ks)
    summary["switches"] = switches
    summary["switch_rate"] = round(switches / max(len(runs), 1), 4)
    summary["delta_vs_deterministic"] = delta_summary(static["deterministic_signatures"], summary)
    return summary


def should_switch(run: QueryRun, *, threshold: float, default_score_ceiling: float | None = None) -> bool:
    side_hits = run.hits_by_name["llm_signature_side"]
    if not side_hits:
        return False
    top = side_hits[0]
    if top.score < threshold:
        return False
    if default_score_ceiling is not None:
        default_hits = run.hits_by_name["deterministic_signatures"]
        default_score = default_hits[0].score if default_hits else 0.0
        if default_score > default_score_ceiling:
            return False
    # Only use the rescue path when the side index clearly points to a repaired
    # hard article. The side index contains only LLM signatures, but this guard
    # makes the behavior explicit for future multi-source side indexes.
    return bool(top.summary or top.text or top.stable_ref)


def summarize_hits(
    questions: list[StressQuestion],
    hits_by_query: list[list[SearchHit]],
    *,
    top_ks: tuple[int, ...],
) -> dict[str, object]:
    ranks = []
    by_language: dict[str, list[int | None]] = {}
    by_style: dict[str, list[int | None]] = {}
    for question, hits in zip(questions, hits_by_query, strict=True):
        gold_article = article_id_from_ref(question.block_id)
        rank = first_rank([hit.stable_ref for hit in hits], gold_article)
        ranks.append(rank)
        by_language.setdefault(question.language, []).append(rank)
        by_style.setdefault(question.style, []).append(rank)
    return {
        "article_hit_at_k": summarize_ranks(ranks, top_ks=top_ks),
        "by_language": {name: summarize_ranks(values, top_ks=top_ks) for name, values in sorted(by_language.items())},
        "by_style": {name: summarize_ranks(values, top_ks=top_ks) for name, values in sorted(by_style.items())},
    }


def first_rank(stable_refs: list[str], gold_article: str) -> int | None:
    for rank, stable_ref in enumerate(stable_refs, start=1):
        if article_id_from_ref(stable_ref) == gold_article:
            return rank
    return None


def delta_summary(baseline: dict[str, object], candidate: dict[str, object]) -> dict[str, float]:
    base = baseline["article_hit_at_k"]
    cand = candidate["article_hit_at_k"]
    return {
        **{
            f"hit@{key}": round(float(value) - float(base["hit_at_k"].get(key, 0.0)), 4)
            for key, value in cand["hit_at_k"].items()
        },
        "mrr": round(float(cand["mrr"]) - float(base["mrr"]), 4),
    }


def best_gate(rows: list[dict[str, object]], *, primary_k: str, require_nonnegative_mrr: bool) -> dict[str, object] | None:
    candidates = []
    for row in rows:
        delta = row["delta_vs_deterministic"]
        if require_nonnegative_mrr and float(delta["mrr"]) < 0:
            continue
        candidates.append(row)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            row["delta_vs_deterministic"].get(f"hit@{primary_k}", 0.0),
            row["delta_vs_deterministic"].get("mrr", 0.0),
            -float(row["threshold"]),
        ),
    )


def best_constrained_gate(
    *,
    all_rows: list[dict[str, object]],
    selected_rows: list[dict[str, object]],
    primary_k: str,
    min_global_delta: float,
) -> dict[str, object] | None:
    selected_by_key = {
        (row["threshold"], row["default_score_ceiling"]): row
        for row in selected_rows
    }
    candidates = []
    for row in all_rows:
        delta = row["delta_vs_deterministic"]
        if float(delta.get(f"hit@{primary_k}", 0.0)) < min_global_delta:
            continue
        selected = selected_by_key.get((row["threshold"], row["default_score_ceiling"]))
        if selected is None:
            continue
        candidates.append({"all": row, "selected": selected})
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            item["selected"]["delta_vs_deterministic"].get(f"hit@{primary_k}", 0.0),
            item["selected"]["delta_vs_deterministic"].get("mrr", 0.0),
            item["all"]["delta_vs_deterministic"].get(f"hit@{primary_k}", 0.0),
        ),
    )


def load_llm_signatures(
    path: str,
    *,
    selected_refs: list[str],
    regions_by_ref: dict[str, SearchRegion],
    model: str,
    signature_limit: int,
) -> dict[str, list[str]]:
    cache = {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("model")) == model:
            cache[str(row.get("article_ref"))] = row
    output = {}
    for ref in selected_refs:
        row = cache.get(ref)
        region = regions_by_ref.get(ref)
        if not row or not region or row.get("article_hash") != region.hash:
            continue
        output[ref] = clean_signatures(list(row.get("signatures", [])), limit=signature_limit)
    return output


def merge_signatures(first: dict[str, list[str]], second: dict[str, list[str]], *, limit: int) -> dict[str, list[str]]:
    refs = set(first) | set(second)
    merged = {}
    for ref in refs:
        values = [*second.get(ref, []), *first.get(ref, [])]
        seen = set()
        clean = []
        for value in values:
            key = value.lower()
            if value and key not in seen:
                clean.append(value)
                seen.add(key)
        merged[ref] = clean[:limit]
    return merged


if __name__ == "__main__":
    main()
