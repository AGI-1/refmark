"""Adapt static BGB Refmark retrieval views from stress questions.

This is the no-runtime-infra adaptation path: use generated stress questions as
offline metadata, rebuild a static BM25 index, and evaluate on held-out
questions from the same article blocks.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.bgb_browser_search.build_bgb_article_navigation import article_id_for  # noqa: E402
from examples.bgb_browser_search.run_bgb_stress_eval import evaluate as evaluate_stress  # noqa: E402
from examples.bgb_browser_search.run_bgb_stress_eval import StressQuestion  # noqa: E402
from refmark.search_index import PortableBM25Index, RetrievalView, SearchRegion, load_search_index  # noqa: E402


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Adapt static BGB retrieval views from generated stress questions.")
    parser.add_argument("--index", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_openrouter_index.json")
    parser.add_argument("--stress-report", required=True)
    parser.add_argument("--output-index", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=515)
    parser.add_argument("--max-aliases-per-region", type=int, default=24)
    parser.add_argument("--unit", choices=["region", "article"], default="region")
    parser.add_argument("--top-ks", default="1,3,5,10,20,50")
    args = parser.parse_args()

    source_payload = json.loads(Path(args.index).read_text(encoding="utf-8-sig"))
    source_index = load_search_index(args.index)
    stress = json.loads(Path(args.stress_report).read_text(encoding="utf-8"))
    train_questions, eval_questions = split_questions_by_block(
        stress_questions(stress),
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    base_regions = source_index.regions if args.unit == "region" else article_regions(source_index.regions)
    adapted_regions, alias_counts = adapt_regions(
        base_regions,
        train_questions,
        max_aliases_per_region=args.max_aliases_per_region,
        unit=args.unit,
    )
    adapted_index = PortableBM25Index(adapted_regions, include_source=source_index.include_source)
    top_ks = tuple(int(part) for part in args.top_ks.split(",") if part.strip())
    baseline_index = source_index if args.unit == "region" else PortableBM25Index(base_regions, include_source=source_index.include_source)
    baseline = evaluate_stress(baseline_index, eval_questions, top_ks=top_ks)
    adapted = evaluate_stress(adapted_index, eval_questions, top_ks=top_ks)
    alias_index = PortableBM25Index(alias_only_regions(base_regions, train_questions, max_aliases_per_region=args.max_aliases_per_region), include_source=False)
    alias_only = evaluate_custom(lambda query, top_k: alias_index.search(query, top_k=top_k), eval_questions, top_ks=top_ks)
    source_alias_hybrid = evaluate_custom(
        lambda query, top_k: hybrid_hits(
            baseline_index.search(query, top_k=top_k),
            alias_index.search(query, top_k=top_k),
            first_weight=0.65,
        ),
        eval_questions,
        top_ks=top_ks,
    )

    write_index(source_payload, adapted_regions, Path(args.output_index), source_stress=args.stress_report, settings=vars(args))
    report = {
        "schema": "refmark.bgb_static_view_adapt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_index": args.index,
        "stress_report": args.stress_report,
        "output_index": args.output_index,
        "settings": vars(args),
        "train_questions": len(train_questions),
        "eval_questions": len(eval_questions),
        "adapted_regions": len(alias_counts),
        "aliases_added": sum(alias_counts.values()),
        "alias_counts_top": [{"ref": ref, "count": count} for ref, count in Counter(alias_counts).most_common(20)],
        "baseline": baseline,
        "adapted": adapted,
        "alias_only": alias_only,
        "source_alias_hybrid": source_alias_hybrid,
        "delta": delta_report(baseline, adapted),
        "alias_hybrid_delta": delta_report_custom(baseline, source_alias_hybrid),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def stress_questions(report: dict[str, object]) -> list[StressQuestion]:
    output: list[StressQuestion] = []
    for section in report.get("sections", []):
        for row in section.get("questions", []):
            output.append(
                StressQuestion(
                    query=str(row["query"]),
                    block_id=str(row["block_id"]),
                    gold_refs=[str(ref) for ref in row["gold_refs"]],
                    block_hash=str(row["block_hash"]),
                    generator_model=str(row["generator_model"]),
                    language=str(row["language"]),
                    style=str(row["style"]),
                    source=str(row.get("source", "openrouter")),
                )
            )
    return output


def split_questions_by_block(
    questions: list[StressQuestion],
    *,
    train_fraction: float,
    seed: int,
) -> tuple[list[StressQuestion], list[StressQuestion]]:
    rng = random.Random(seed)
    by_block: dict[str, list[StressQuestion]] = {}
    for question in questions:
        by_block.setdefault(question.block_id, []).append(question)
    train: list[StressQuestion] = []
    eval_rows: list[StressQuestion] = []
    for block_id, rows in sorted(by_block.items()):
        shuffled = list(rows)
        rng.shuffle(shuffled)
        split = max(1, min(len(shuffled) - 1, round(len(shuffled) * train_fraction))) if len(shuffled) > 1 else len(shuffled)
        train.extend(shuffled[:split])
        eval_rows.extend(shuffled[split:])
    return train, eval_rows


def adapt_regions(
    regions: list[SearchRegion],
    questions: list[StressQuestion],
    *,
    max_aliases_per_region: int,
    unit: str,
) -> tuple[list[SearchRegion], dict[str, int]]:
    aliases_by_ref: dict[str, list[str]] = {}
    for question in questions:
        alias = alias_text(question)
        target_refs = [question.block_id] if unit == "article" else question.gold_refs
        for stable_ref in target_refs:
            aliases_by_ref.setdefault(stable_ref, []).append(alias)

    adapted: list[SearchRegion] = []
    counts: dict[str, int] = {}
    for region in regions:
        aliases = unique(aliases_by_ref.get(region.stable_ref, []))[:max_aliases_per_region]
        if not aliases:
            adapted.append(region)
            continue
        view = region.view
        adapted_view = RetrievalView(
            summary=view.summary,
            questions=unique([*view.questions, *aliases])[: max(len(view.questions), 0) + max_aliases_per_region],
            keywords=unique([*view.keywords, *keyword_terms(aliases)])[: max(len(view.keywords), 0) + max_aliases_per_region],
        )
        adapted.append(replace(region, view=adapted_view))
        counts[region.stable_ref] = len(aliases)
    return adapted, counts


def article_regions(regions: list[SearchRegion]) -> list[SearchRegion]:
    grouped: dict[tuple[str, str], list[SearchRegion]] = {}
    for region in regions:
        grouped.setdefault((region.doc_id, article_id_for(region.region_id)), []).append(region)
    output: list[SearchRegion] = []
    for ordinal, ((doc_id, article_id), items) in enumerate(
        sorted(grouped.items(), key=lambda item: min(region.ordinal for region in item[1]))
    ):
        ordered = sorted(items, key=lambda region: region.ordinal)
        text = "\n\n".join(region.text for region in ordered)
        view = RetrievalView(
            summary=first_nonempty([region.view.summary for region in ordered]),
            questions=unique([question for region in ordered for question in region.view.questions])[:120],
            keywords=unique([keyword for region in ordered for keyword in region.view.keywords])[:120],
        )
        output.append(
            SearchRegion(
                doc_id=doc_id,
                region_id=article_id,
                text=text,
                hash=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
                source_path=ordered[0].source_path,
                ordinal=ordinal,
                prev_region_id=None,
                next_region_id=None,
                view=view,
            )
        )
    return [
        replace(
            region,
            prev_region_id=output[index - 1].region_id if index > 0 else None,
            next_region_id=output[index + 1].region_id if index + 1 < len(output) else None,
        )
        for index, region in enumerate(output)
    ]


def alias_only_regions(
    base_regions: list[SearchRegion],
    questions: list[StressQuestion],
    *,
    max_aliases_per_region: int,
) -> list[SearchRegion]:
    aliases_by_ref: dict[str, list[str]] = {}
    for question in questions:
        aliases_by_ref.setdefault(question.block_id, []).append(alias_text(question))
    output = []
    for region in base_regions:
        aliases = unique(aliases_by_ref.get(region.stable_ref, []))[:max_aliases_per_region]
        output.append(
            replace(
                region,
                text="",
                view=RetrievalView(
                    summary=region.stable_ref,
                    questions=aliases,
                    keywords=keyword_terms(aliases),
                ),
            )
        )
    return output


def alias_text(question: StressQuestion) -> str:
    return f"[{question.language}/{question.style}] {question.query}"


def keyword_terms(values: list[str]) -> list[str]:
    terms: list[str] = []
    for value in values:
        for token in value.replace("/", " ").replace("[", " ").replace("]", " ").split():
            clean = token.strip(".,;:!?()\"'").lower()
            if len(clean) > 2:
                terms.append(clean)
    return unique(terms)


def write_index(
    source_payload: dict[str, object],
    regions: list[SearchRegion],
    path: Path,
    *,
    source_stress: str,
    settings: dict[str, object],
) -> None:
    payload = dict(source_payload)
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    payload["settings"] = {
        **dict(source_payload.get("settings", {})),
        "adaptation": "stress-question-static-view-aliases",
        "adaptation_stress_report": source_stress,
        "adaptation_settings_hash": hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()[:16],
    }
    payload["regions"] = [region.to_dict() for region in regions]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def delta_report(baseline: dict[str, object], adapted: dict[str, object]) -> dict[str, object]:
    output = {}
    base_hits = baseline["article_hit_at_k"]["hit_at_k"]
    adapted_hits = adapted["article_hit_at_k"]["hit_at_k"]
    for key, value in adapted_hits.items():
        output[f"article_hit@{key}"] = round(float(value) - float(base_hits.get(key, 0.0)), 4)
    output["article_mrr"] = round(float(adapted["article_mrr"]) - float(baseline["article_mrr"]), 4)
    return output


def delta_report_custom(baseline: dict[str, object], adapted: dict[str, object]) -> dict[str, object]:
    output = {}
    base_hits = baseline["article_hit_at_k"]["hit_at_k"]
    adapted_hits = adapted["article_hit_at_k"]["hit_at_k"]
    for key, value in adapted_hits.items():
        output[f"article_hit@{key}"] = round(float(value) - float(base_hits.get(key, 0.0)), 4)
    output["article_mrr"] = round(float(adapted["article_hit_at_k"]["mrr"]) - float(baseline["article_mrr"]), 4)
    return output


def evaluate_custom(search_fn, questions: list[StressQuestion], *, top_ks: tuple[int, ...]) -> dict[str, object]:
    max_k = max(top_ks)
    ranks = []
    by_style: dict[str, list[int | None]] = {}
    for question in questions:
        hits = search_fn(question.query, max_k)
        gold_articles = {article_id_from_ref(ref) for ref in question.gold_refs}
        rank = None
        for index, hit in enumerate(hits, start=1):
            if article_id_from_ref(hit.stable_ref) in gold_articles:
                rank = index
                break
        ranks.append(rank)
        by_style.setdefault(question.style, []).append(rank)
    return {
        "article_hit_at_k": summarize_ranks(ranks, top_ks=top_ks),
        "by_style": {style: summarize_ranks(values, top_ks=top_ks) for style, values in sorted(by_style.items())},
        "misses": sum(1 for rank in ranks if rank is None),
    }


def hybrid_hits(first, second, *, first_weight: float):
    scores = {}
    refs = {}
    for weight, hits in ((first_weight, first), (1.0 - first_weight, second)):
        for rank, hit in enumerate(hits, start=1):
            scores[hit.stable_ref] = scores.get(hit.stable_ref, 0.0) + (weight / (rank + 60.0))
            refs[hit.stable_ref] = hit
    ordered_refs = sorted(scores, key=lambda ref: (-scores[ref], ref))
    return [replace(refs[ref], rank=index, score=round(scores[ref], 6)) for index, ref in enumerate(ordered_refs, start=1)]


def summarize_ranks(ranks: list[int | None], *, top_ks: tuple[int, ...]) -> dict[str, object]:
    total = max(len(ranks), 1)
    return {
        "count": len(ranks),
        "hit_at_k": {str(k): round(sum(1 for rank in ranks if rank is not None and rank <= k) / total, 4) for k in top_ks},
        "mrr": round(sum(1.0 / rank for rank in ranks if rank is not None) / total, 4),
    }


def article_id_from_ref(stable_ref: str) -> str:
    doc_id, region_id = stable_ref.split(":", 1)
    return f"{doc_id}:{article_id_for(region_id)}"


def unique(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = " ".join(str(value).split())
        key = clean.lower()
        if clean and key not in seen:
            output.append(clean)
            seen.add(key)
    return output


def first_nonempty(values: list[str]) -> str:
    for value in values:
        if value:
            return value
    return ""


if __name__ == "__main__":
    main()
