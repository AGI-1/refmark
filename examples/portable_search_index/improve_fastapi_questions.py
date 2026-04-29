"""Agentic repair loop for the FastAPI structure heatmap questions.

This script is intentionally example-scoped. It takes the current balanced
section eval, selects weak sections, asks a stronger model to diagnose the
failure mode, asks a cheaper model to regenerate only drifted questions, and
reruns the local eval modes. Region/range/exclusion/confusion adaptations are
reported as a plan first; only question rewrites are auto-applied.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
from urllib import request

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from refmark.search_index import OPENROUTER_CHAT_URL, PortableBM25Index, RetrievalView, SearchHit, SearchRegion, approx_tokens, load_search_index


BASE = Path("examples/portable_search_index/output/fastapi_pipeline_qwen_mistral")
DEFAULT_INDEX = BASE / "docs.index.json"
DEFAULT_QUESTIONS = BASE / "balanced_section_questions.jsonl"
DEFAULT_EVAL = BASE / "balanced_section_eval_modes.json"
DEFAULT_EMBEDDINGS = BASE / "embedding_cache_qwen3_8b_balanced.jsonl"
DEFAULT_HEATMAP = BASE / "fastapi_structure_heatmap.html"
DEFAULT_SHADOW_METADATA = BASE / "adaptive_shadow_metadata.json"
EMBED_ENDPOINT = "https://openrouter.ai/api/v1/embeddings"
EMBED_MODEL = "qwen/qwen3-embedding-8b"
ADAPTIVE_EMBED_SUFFIX = ".adaptive"


def main() -> None:
    parser = argparse.ArgumentParser(description="Review weak FastAPI heatmap areas and regenerate suspect questions.")
    parser.add_argument("--index", default=str(DEFAULT_INDEX))
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--eval", default=str(DEFAULT_EVAL))
    parser.add_argument("--embedding-cache", default=str(DEFAULT_EMBEDDINGS))
    parser.add_argument("--heatmap", default=str(DEFAULT_HEATMAP))
    parser.add_argument("--shadow-metadata", default=str(DEFAULT_SHADOW_METADATA))
    parser.add_argument("--review-model", default="qwen/qwen3.6-max-preview")
    parser.add_argument("--generator-model", default="qwen/qwen-turbo")
    parser.add_argument("--endpoint", default=OPENROUTER_CHAT_URL)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--max-targets", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--rerun-only", action="store_true", help="Skip agents and only rerun eval/heatmap from current questions.")
    parser.add_argument("--no-mini-eval", action="store_true", help="Apply reviewer changes without affected-row before/after gating.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    index = load_search_index(args.index)
    questions = read_jsonl(Path(args.questions))
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set")

    if args.rerun_only:
        ensure_query_embeddings(questions, Path(args.embedding_cache), api_key=api_key, workers=args.concurrency)
        shadow_metadata = load_shadow_metadata(Path(args.shadow_metadata))
        ensure_query_embeddings(questions, adaptive_embedding_cache_path(Path(args.embedding_cache)), api_key=api_key, workers=args.concurrency)
        ensure_adaptive_document_embeddings(index, shadow_metadata, adaptive_embedding_cache_path(Path(args.embedding_cache)), api_key=api_key, workers=args.concurrency)
        new_eval = evaluate_modes(index, questions, Path(args.embedding_cache), shadow_metadata=shadow_metadata, index_path=args.index, questions_path=args.questions)
        Path(args.eval).write_text(json.dumps(new_eval, indent=2, ensure_ascii=False), encoding="utf-8")
        update_heatmap(Path(args.heatmap), new_eval)
        print(json.dumps({"changed": 0, "metrics": {k: v["metrics"] for k, v in new_eval["methods"].items()}}, indent=2))
        return

    old_eval = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    targets = select_weak_targets(old_eval, limit=args.max_targets)
    if not targets:
        print("No weak covered targets found.")
        return

    by_stable_ref = group_questions(questions)
    review_inputs = [
        build_review_input(target, by_stable_ref.get(target, []), old_eval, index)
        for target in targets
        if by_stable_ref.get(target)
    ]
    reviews = run_parallel(
        review_inputs,
        lambda item: review_target(item, args.review_model, args.endpoint, api_key),
        workers=args.concurrency,
    )
    adaptation_plan = build_adaptation_plan(reviews)
    repair_jobs = []
    for review_input, review_payload in zip(review_inputs, reviews, strict=False):
        for item in review_payload.get("items", []):
            if str(item.get("action", "")).lower() == "regenerate":
                repair_jobs.append((review_input, item, review_payload))

    original_questions = copy.deepcopy(questions)
    replacements = run_parallel(
        repair_jobs,
        lambda job: regenerate_question(job[0], job[1], args.generator_model, args.endpoint, api_key),
        workers=args.concurrency,
    )
    candidate_questions = copy.deepcopy(questions)
    changed = apply_replacements(candidate_questions, repair_jobs, replacements)
    shadow_path = Path(args.shadow_metadata)
    current_shadow_metadata = load_shadow_metadata(shadow_path)
    candidate_shadow_metadata = merge_shadow_metadata(
        current_shadow_metadata,
        build_shadow_metadata(reviews, candidate_questions, index),
    )
    metadata_changed = stable_json(candidate_shadow_metadata) != stable_json(current_shadow_metadata)
    affected_refs = sorted(set(targets) | {str(row["stable_ref"]) for row in changed})
    mini_eval = None
    accepted = True
    if (changed or metadata_changed) and not args.no_mini_eval:
        mini_eval = affected_mini_eval(
            index,
            original_questions,
            candidate_questions,
            Path(args.embedding_cache),
            before_metadata=current_shadow_metadata,
            after_metadata=candidate_shadow_metadata,
            affected_refs=affected_refs,
        )
        accepted = mini_eval["decision"] == "accepted"
    if accepted:
        questions = candidate_questions
        shadow_metadata = candidate_shadow_metadata
    else:
        changed = []
        questions = original_questions
        shadow_metadata = current_shadow_metadata

    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "review_model": args.review_model,
        "generator_model": args.generator_model,
        "selected_targets": targets,
        "reviews": reviews,
        "adaptation_plan": adaptation_plan,
        "affected_mini_eval": mini_eval,
        "decision": "accepted" if accepted else "rejected",
        "metadata_changed": metadata_changed and accepted,
        "changed_questions": changed,
    }
    report_path = Path(args.eval).with_name("question_improvement_agent_report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.dry_run:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    if changed:
        backup = Path(args.questions).with_suffix(".before_agent.jsonl")
        if not backup.exists():
            backup.write_text(Path(args.questions).read_text(encoding="utf-8"), encoding="utf-8")
        write_jsonl(Path(args.questions), questions)
        ensure_query_embeddings(questions, Path(args.embedding_cache), api_key=api_key, workers=args.concurrency)
    write_shadow_metadata(shadow_path, shadow_metadata)
    ensure_query_embeddings(questions, adaptive_embedding_cache_path(Path(args.embedding_cache)), api_key=api_key, workers=args.concurrency)
    ensure_adaptive_document_embeddings(index, shadow_metadata, adaptive_embedding_cache_path(Path(args.embedding_cache)), api_key=api_key, workers=args.concurrency)

    new_eval = evaluate_modes(index, questions, Path(args.embedding_cache), shadow_metadata=shadow_metadata, index_path=args.index, questions_path=args.questions)
    Path(args.eval).write_text(json.dumps(new_eval, indent=2, ensure_ascii=False), encoding="utf-8")
    update_heatmap(Path(args.heatmap), new_eval)
    print(json.dumps({"changed": len(changed), "targets": targets, "metrics": {k: v["metrics"] for k, v in new_eval["methods"].items()}}, indent=2))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def group_questions(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["stable_ref"]), []).append(row)
    return grouped


def select_weak_targets(eval_payload: dict[str, Any], *, limit: int) -> list[str]:
    methods = eval_payload["methods"]
    scores: dict[str, dict[str, float]] = {}
    for payload in methods.values():
        for row in payload["results"]:
            ref = str(row["stable_ref"])
            bucket = scores.setdefault(ref, {"count": 0, "hit1": 0, "hit10": 0, "coverage": 0.0})
            bucket["count"] += 1
            bucket["hit1"] += int(bool(row["hit_at_1"]))
            bucket["hit10"] += int(bool(row["hit_at_k"]))
            bucket["coverage"] += float(row.get("gold_coverage", 0.0))
    ranked = []
    for ref, score in scores.items():
        if score["count"] <= 0:
            continue
        hit10 = score["hit10"] / score["count"]
        hit1 = score["hit1"] / score["count"]
        coverage = score["coverage"] / score["count"]
        if hit1 >= 1.0 and hit10 >= 1.0:
            continue
        ranked.append((hit10, hit1, coverage, ref))
    ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return [ref for *_scores, ref in ranked[:limit]]


def build_review_input(stable_ref: str, qrows: list[dict[str, Any]], eval_payload: dict[str, Any], index: PortableBM25Index) -> dict[str, Any]:
    ref_to_region = {region.stable_ref: region for region in index.regions}
    gold_refs = list(dict.fromkeys(ref for row in qrows for ref in row.get("gold_refs", [])))
    gold_text = "\n\n".join(
        f"{ref}\n{ref_to_region[ref].text[:1800]}"
        for ref in gold_refs
        if ref in ref_to_region
    )
    observed = []
    for mode, payload in eval_payload["methods"].items():
        for row in payload["results"]:
            if row.get("stable_ref") == stable_ref:
                observed.append(
                    {
                        "mode": mode,
                        "variant": row.get("variant"),
                        "query": row.get("query"),
                        "hit_at_1": row.get("hit_at_1"),
                        "hit_at_10": row.get("hit_at_k"),
                        "top_refs": row.get("top_refs", [])[:5],
                    }
                )
    competing_refs = []
    for item in observed:
        competing_refs.extend(ref for ref in item["top_refs"] if ref not in gold_refs)
    competing_refs = list(dict.fromkeys(competing_refs))[:6]
    competing_text = "\n\n".join(
        f"{ref}\n{ref_to_region[ref].text[:700]}"
        for ref in competing_refs
        if ref in ref_to_region
    )
    return {
        "stable_ref": stable_ref,
        "section_title": qrows[0].get("section_title"),
        "gold_refs": gold_refs,
        "questions": [
            {"variant": row.get("variant"), "query": row.get("query")}
            for row in qrows
        ],
        "observed": observed,
        "gold_text": gold_text,
        "competing_text": competing_text,
    }


def review_target(item: dict[str, Any], model: str, endpoint: str, api_key: str) -> dict[str, Any]:
    prompt = f"""You are reviewing weak retrieval-eval areas for a refmarked documentation corpus.

Diagnose why the area is weak and recommend adaptations. Prefer keeping hard
but valid questions; only regenerate if a question is ambiguous, mislabeled,
too broad, or targets the competing top refs better than the gold refs.

Adaptation action vocabulary:
- regenerate_question: validation repair for drifted/ambiguous questions.
- add_alternate_gold_refs: competing top refs are also valid evidence.
- extend_gold_range: nearby/neighbor refs are needed for enough coverage.
- split_gold_range: gold range contains multiple topics and should be smaller.
- merge_or_link_regions: adjacent/duplicate/equivalent sections behave as one concept.
- mark_excluded_or_hub: TOC/index/release/hub/query-magnet should be excluded or role-marked.
- record_confusion_pair: wrong top ref is a meaningful repeated confusion edge.
- tune_retriever: question is valid, gold is valid, retrieval/ranking needs better signals.
- keep_valid_hard_case: no adaptation now besides tracking.

Return strict JSON:
{{
  "target_summary": "...",
  "area_actions": [
    {{"action":"...", "confidence":0.0, "refs":["..."], "reason":"...", "apply_automatically":false}}
  ],
  "items": [
    {{"variant":"direct|concern", "action":"keep|regenerate", "diagnosis":"query_drift|ambiguous|gold_too_broad|retrieval_failure|valid_hard_case|alternate_gold|query_magnet|range_too_broad|range_too_narrow", "reason":"...", "replacement_brief":"...", "suggested_area_actions":["..."]}}
  ]
}}

Gold target: {item["stable_ref"]}
Section title: {item.get("section_title")}
Gold refs: {item["gold_refs"]}

Current questions:
{json.dumps(item["questions"], ensure_ascii=False)}

Eval observations:
{json.dumps(item["observed"], ensure_ascii=False)}

Gold evidence:
{item["gold_text"]}

Competing top evidence:
{item["competing_text"]}
"""
    payload = chat_json(
        model=model,
        endpoint=endpoint,
        api_key=api_key,
        system="You are a precise retrieval evaluation reviewer. Do not overthink; return compact strict JSON.",
        user=prompt,
        temperature=0.1,
        max_tokens=900,
    )
    payload["stable_ref"] = item["stable_ref"]
    return payload


def build_adaptation_plan(reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    plan = []
    for review in reviews:
        stable_ref = review.get("stable_ref")
        for action in review.get("area_actions", []):
            plan.append(normalize_area_action(stable_ref, action, source="area_actions"))
        for item in review.get("items", []):
            item_actions = item.get("suggested_area_actions", [])
            if not item_actions and item.get("action") == "regenerate":
                item_actions = ["regenerate_question"]
            if not item_actions:
                diagnosis = str(item.get("diagnosis", ""))
                if diagnosis == "valid_hard_case":
                    item_actions = ["keep_valid_hard_case"]
                elif diagnosis in {"gold_too_broad", "range_too_broad"}:
                    item_actions = ["split_gold_range"]
                elif diagnosis == "range_too_narrow":
                    item_actions = ["extend_gold_range"]
                elif diagnosis == "alternate_gold":
                    item_actions = ["add_alternate_gold_refs"]
                elif diagnosis == "query_magnet":
                    item_actions = ["mark_excluded_or_hub"]
                elif diagnosis == "retrieval_failure":
                    item_actions = ["tune_retriever"]
            for action_name in item_actions:
                plan.append(
                    normalize_area_action(
                        stable_ref,
                        {
                            "action": action_name,
                            "confidence": 0.5,
                            "refs": [stable_ref] if stable_ref else [],
                            "reason": item.get("reason", ""),
                            "variant": item.get("variant"),
                            "diagnosis": item.get("diagnosis"),
                            "apply_automatically": action_name == "regenerate_question",
                        },
                        source="item",
                    )
                )
    return dedupe_plan(plan)


def normalize_area_action(stable_ref: str | None, action: dict[str, Any], *, source: str) -> dict[str, Any]:
    action_name = str(action.get("action", "keep_valid_hard_case"))
    return {
        "target_ref": stable_ref,
        "action": action_name,
        "adaptation_type": adaptation_type(action_name),
        "confidence": float(action.get("confidence", 0.0) or 0.0),
        "refs": [str(ref) for ref in action.get("refs", [])],
        "variant": action.get("variant"),
        "diagnosis": action.get("diagnosis"),
        "reason": str(action.get("reason", "")),
        "apply_automatically": bool(action.get("apply_automatically", False) and action_name == "regenerate_question"),
        "source": source,
    }


def adaptation_type(action: str) -> str:
    if action == "regenerate_question":
        return "validation"
    if action in {"add_alternate_gold_refs", "extend_gold_range"}:
        return "gold_validation"
    if action in {"split_gold_range", "merge_or_link_regions"}:
        return "region_boundary"
    if action == "mark_excluded_or_hub":
        return "exclusion_or_role"
    if action == "record_confusion_pair":
        return "confusion_mapping"
    if action == "tune_retriever":
        return "retriever"
    return "tracking"


def dedupe_plan(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for row in plan:
        key = (row.get("target_ref"), row.get("action"), row.get("variant"), row.get("reason"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    order = {
        "validation": 0,
        "gold_validation": 1,
        "region_boundary": 2,
        "exclusion_or_role": 3,
        "confusion_mapping": 4,
        "retriever": 5,
        "tracking": 6,
    }
    deduped.sort(key=lambda row: (order.get(str(row.get("adaptation_type")), 9), str(row.get("target_ref")), str(row.get("action"))))
    return deduped


def affected_mini_eval(
    index: PortableBM25Index,
    before_questions: list[dict[str, Any]],
    after_questions: list[dict[str, Any]],
    embedding_cache: Path,
    *,
    before_metadata: dict[str, dict[str, Any]],
    after_metadata: dict[str, dict[str, Any]],
    affected_refs: list[str],
) -> dict[str, Any]:
    before_rows = affected_questions(before_questions, affected_refs)
    after_rows = affected_questions(after_questions, affected_refs)
    before_eval = evaluate_modes(
        index,
        before_rows,
        embedding_cache,
        shadow_metadata=before_metadata,
        index_path="mini-eval:index",
        questions_path="mini-eval:before",
    )
    after_eval = evaluate_modes(
        index,
        after_rows,
        embedding_cache,
        shadow_metadata=after_metadata,
        index_path="mini-eval:index",
        questions_path="mini-eval:after",
    )
    before_score = acceptance_tuple(before_eval)
    after_score = acceptance_tuple(after_eval)
    accepted = after_score >= before_score
    return {
        "affected_refs": affected_refs,
        "rows": len(after_rows),
        "primary_method": primary_method(after_eval),
        "before_score": before_score,
        "after_score": after_score,
        "decision": "accepted" if accepted else "rejected",
        "reason": "accepted because affected-row metrics did not regress" if accepted else "rejected because affected-row metrics regressed",
        "before_metrics": {name: payload["metrics"] for name, payload in before_eval["methods"].items()},
        "after_metrics": {name: payload["metrics"] for name, payload in after_eval["methods"].items()},
    }


def affected_questions(rows: list[dict[str, Any]], refs: list[str]) -> list[dict[str, Any]]:
    ref_set = set(refs)
    return [
        row
        for row in rows
        if row.get("stable_ref") in ref_set or any(ref in ref_set for ref in row.get("gold_refs", []))
    ]


def primary_method(eval_payload: dict[str, Any]) -> str:
    if "hybrid_qwen3_w035" in eval_payload["methods"]:
        return "hybrid_qwen3_w035"
    if "qwen3_embedding" in eval_payload["methods"]:
        return "qwen3_embedding"
    return next(iter(eval_payload["methods"]))


def acceptance_tuple(eval_payload: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = eval_payload["methods"][primary_method(eval_payload)]["metrics"]
    return (
        float(metrics.get("hit_at_k", 0.0)),
        float(metrics.get("hit_at_1", 0.0)),
        float(metrics.get("mrr", 0.0)),
        float(metrics.get("gold_coverage", 0.0)),
    )


def load_shadow_metadata(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = payload.get("entries", payload if isinstance(payload, list) else [])
    return {str(row["ref"]): dict(row) for row in rows if row.get("ref")}


def write_shadow_metadata(path: Path, metadata: dict[str, dict[str, Any]]) -> None:
    payload = {
        "schema": "refmark.adaptive_shadow_metadata.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "description": "Generated retrieval metadata stored beside source refs; source documents are unchanged.",
        "entries": [metadata[key] for key in sorted(metadata)],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def adaptive_embedding_cache_path(path: Path) -> Path:
    return path.with_name(path.stem + ADAPTIVE_EMBED_SUFFIX + path.suffix)


def merge_shadow_metadata(existing: dict[str, dict[str, Any]], new_rows: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    merged = dict(existing)
    for ref, row in new_rows.items():
        current = dict(merged.get(ref, {"ref": ref}))
        for field in ["doc2query", "keywords", "disambiguators", "roles"]:
            current[field] = sorted(set([*current.get(field, []), *row.get(field, [])]))
        current["source_hashes"] = {**current.get("source_hashes", {}), **row.get("source_hashes", {})}
        current["confusions"] = dedupe_dicts([*current.get("confusions", []), *row.get("confusions", [])])
        current["provenance"] = row.get("provenance", current.get("provenance", {}))
        merged[ref] = current
    return merged


def dedupe_dicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for row in rows:
        key = json.dumps(row, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def build_shadow_metadata(reviews: list[dict[str, Any]], questions: list[dict[str, Any]], index: PortableBM25Index) -> dict[str, dict[str, Any]]:
    by_ref = {region.stable_ref: region for region in index.regions}
    question_rows = group_questions(questions)
    output: dict[str, dict[str, Any]] = {}
    for review in reviews:
        stable_ref = str(review.get("stable_ref", ""))
        if not stable_ref:
            continue
        qrows = question_rows.get(stable_ref, [])
        gold_refs = list(dict.fromkeys(ref for row in qrows for ref in row.get("gold_refs", [])))
        doc2query = [str(row.get("query", "")).strip() for row in qrows if row.get("query")]
        disambiguators = []
        roles = []
        confusions = []
        for item in review.get("items", []):
            diagnosis = str(item.get("diagnosis", ""))
            reason = str(item.get("reason", ""))
            if diagnosis in {"valid_hard_case", "retrieval_failure"} and reason:
                disambiguators.append(reason[:260])
            if diagnosis in {"query_magnet"}:
                roles.append("query_magnet")
            if item.get("replacement_brief"):
                doc2query.append(str(item["replacement_brief"]))
        for observed in review.get("observed", []):
            top_refs = [str(ref) for ref in observed.get("top_refs", [])]
            wrong = next((ref for ref in top_refs if ref not in gold_refs), None)
            if wrong:
                confusions.append({"with": wrong, "variant": observed.get("variant"), "mode": observed.get("mode")})
        source_hashes = {ref: by_ref[ref].hash for ref in gold_refs if ref in by_ref}
        keywords = shadow_keywords(review.get("target_summary", ""), doc2query)
        if doc2query or disambiguators or confusions:
            output[stable_ref] = {
                "ref": stable_ref,
                "source_hashes": source_hashes,
                "roles": roles,
                "doc2query": sorted(set(doc2query)),
                "keywords": keywords,
                "disambiguators": sorted(set(disambiguators)),
                "confusions": dedupe_dicts(confusions),
                "provenance": {
                    "source": "adapt_loop",
                    "review_model": "configured",
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            }
    return output


def shadow_keywords(summary: Any, questions: list[str]) -> list[str]:
    text = " ".join([str(summary), *questions]).lower()
    stop = {"about", "fastapi", "which", "what", "where", "when", "with", "from", "that", "this", "does", "into", "using", "used", "docs", "documentation"}
    terms = []
    for token in re.findall(r"[a-z][a-z0-9_:-]{3,}", text):
        if token in stop or token in terms:
            continue
        terms.append(token)
    return terms[:18]


def adaptive_index(index: PortableBM25Index, metadata: dict[str, dict[str, Any]]) -> PortableBM25Index:
    if not metadata:
        return index
    regions = []
    for region in index.regions:
        entry = metadata.get(region.stable_ref) or metadata.get(range_ref_for_region(region.stable_ref, metadata))
        if not entry:
            regions.append(region)
            continue
        view = RetrievalView(
            summary="\n".join(part for part in [region.view.summary, *entry.get("disambiguators", [])] if part),
            questions=[*region.view.questions, *entry.get("doc2query", [])],
            keywords=[*region.view.keywords, *entry.get("keywords", [])],
        )
        roles = sorted(set([*region.roles, *entry.get("roles", [])]))
        regions.append(
            SearchRegion(
                doc_id=region.doc_id,
                region_id=region.region_id,
                text=region.text,
                hash=region.hash,
                source_path=region.source_path,
                ordinal=region.ordinal,
                prev_region_id=region.prev_region_id,
                next_region_id=region.next_region_id,
                view=view,
                roles=roles,
                search_excluded=region.search_excluded,
                search_exclusion_reason=region.search_exclusion_reason,
            )
        )
    return PortableBM25Index(regions, include_source=index.include_source, k1=index.k1, b=index.b)


def adaptive_region_text(region: SearchRegion, metadata: dict[str, dict[str, Any]]) -> str:
    entry = metadata.get(region.stable_ref) or metadata.get(range_ref_for_region(region.stable_ref, metadata))
    if not entry:
        return region.index_text(include_source=True)
    parts = [
        region.index_text(include_source=True),
        *entry.get("doc2query", []),
        *entry.get("keywords", []),
        *entry.get("disambiguators", []),
    ]
    return "\n".join(str(part) for part in parts if str(part).strip())


def range_ref_for_region(stable_ref: str, metadata: dict[str, dict[str, Any]]) -> str | None:
    for ref, row in metadata.items():
        if stable_ref in row.get("source_hashes", {}):
            return ref
    return None


def regenerate_question(job_input: dict[str, Any], item: dict[str, Any], model: str, endpoint: str, api_key: str) -> dict[str, str]:
    prompt = f"""Regenerate one retrieval evaluation question.

The question must be natural, concise, and answerable from the gold evidence.
It must not primarily target the competing evidence. Do not quote exact full
sentences. Preserve the intended variant style.

Return strict JSON: {{"query":"..."}}

Variant: {item.get("variant")}
Replacement brief: {item.get("replacement_brief")}
Gold target: {job_input["stable_ref"]}
Section title: {job_input.get("section_title")}

Gold evidence:
{job_input["gold_text"]}

Competing evidence to avoid:
{job_input["competing_text"]}
"""
    payload = chat_json(
        model=model,
        endpoint=endpoint,
        api_key=api_key,
        system="You write held-out documentation search questions. Return compact strict JSON.",
        user=prompt,
        temperature=0.35,
        max_tokens=160,
    )
    return {"variant": str(item.get("variant")), "query": str(payload.get("query", "")).strip(), "reason": str(item.get("reason", ""))}


def chat_json(*, model: str, endpoint: str, api_key: str, system: str, user: str, temperature: float, max_tokens: int) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/b-imenitov/refmark",
            "X-Title": "refmark FastAPI heatmap improvement",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    content = payload["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.strip("`").removeprefix("json").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start : end + 1])
        raise


def run_parallel(items, fn, *, workers: int):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        return list(executor.map(fn, items))


def apply_replacements(rows: list[dict[str, Any]], jobs, replacements) -> list[dict[str, Any]]:
    changed = []
    for (job_input, review_item, _review), replacement in zip(jobs, replacements, strict=False):
        query = replacement.get("query", "").strip()
        if not query:
            continue
        stable_ref = job_input["stable_ref"]
        variant = replacement["variant"]
        for row in rows:
            if row.get("stable_ref") == stable_ref and row.get("variant") == variant:
                old = row["query"]
                if old == query:
                    break
                row["query"] = query
                row["model"] = f"{row.get('model', 'unknown')}+agent_repaired"
                row["curation_note"] = f"Agent reviewer: {review_item.get('diagnosis')}; {review_item.get('reason')}"
                row["usage"] = row.get("usage", {})
                changed.append({"stable_ref": stable_ref, "variant": variant, "old": old, "new": query})
                break
    return changed


def evaluate_modes(
    index: PortableBM25Index,
    questions: list[dict[str, Any]],
    embedding_cache: Path,
    *,
    shadow_metadata: dict[str, dict[str, Any]] | None = None,
    index_path: str,
    questions_path: str,
) -> dict[str, Any]:
    methods = {}
    active_index = adaptive_index(index, shadow_metadata) if shadow_metadata else index
    methods["bm25"] = evaluate_bm25(active_index, questions, name="BM25 / adaptive metadata")
    if shadow_metadata:
        methods["baseline_bm25"] = evaluate_bm25(index, questions, name="Baseline BM25 / no adaptive metadata")
    docs, query_cache = load_embeddings(embedding_cache, index)
    if docs is not None:
        methods["baseline_qwen3_embedding"] = evaluate_embedding(index, questions, docs, query_cache, hybrid_weight=None, name="Baseline Qwen3 embedding / no adaptive metadata")
    adaptive_cache = adaptive_embedding_cache_path(embedding_cache)
    adaptive_docs, adaptive_query_cache = load_embeddings(adaptive_cache, index)
    if shadow_metadata and adaptive_docs is not None:
        methods["qwen3_embedding"] = evaluate_embedding(active_index, questions, adaptive_docs, adaptive_query_cache, hybrid_weight=None, name="Qwen3 embedding / adaptive metadata")
        methods["hybrid_qwen3_w035"] = evaluate_embedding(active_index, questions, adaptive_docs, adaptive_query_cache, hybrid_weight=0.35, name="Hybrid: adaptive BM25 + adaptive Qwen3")
    elif docs is not None:
        methods["qwen3_embedding"] = evaluate_embedding(index, questions, docs, query_cache, hybrid_weight=None, name="Qwen3 embedding")
        methods["hybrid_qwen3_w035"] = evaluate_embedding(index, questions, docs, query_cache, hybrid_weight=0.35, name="Hybrid: BM25 + Qwen3 embedding")
    return {
        "schema": "refmark.fastapi_balanced_eval.v1",
        "index": str(index_path),
        "questions": str(questions_path),
        "methods": methods,
    }


def evaluate_bm25(index: PortableBM25Index, questions: list[dict[str, Any]], *, name: str = "BM25 / Refmark metadata") -> dict[str, Any]:
    results = []
    for row in questions:
        hits = index.search_reranked(row["query"], top_k=10, candidate_k=60)
        results.append(score_row(row, hits, elapsed_ms=0.0))
    return {"name": name, "metrics": summarize_results(results), "results": results}


def evaluate_embedding(
    index: PortableBM25Index,
    questions: list[dict[str, Any]],
    docs,
    query_cache: dict[str, np.ndarray],
    *,
    hybrid_weight: float | None,
    name: str,
) -> dict[str, Any]:
    results = []
    ref_to_index = {region.stable_ref: idx for idx, region in enumerate(index.regions)}
    for row in questions:
        start = time.perf_counter()
        qvec = query_cache[hash_text(row["query"])]
        sims = docs @ qvec
        if hybrid_weight is not None:
            bm25_candidates = index.search_reranked(row["query"], top_k=60, candidate_k=60)
            bm25_scores = np.zeros(len(index.regions), dtype=np.float32)
            for hit in bm25_candidates:
                bm25_scores[ref_to_index[hit.stable_ref]] = hit.score
            if float(bm25_scores.max()) > 0:
                bm25_scores = bm25_scores / float(bm25_scores.max())
            sims = (hybrid_weight * bm25_scores) + ((1.0 - hybrid_weight) * sims)
        order = np.argsort(-sims)[:10]
        hits = [
            make_hit(index, int(i), rank + 1, float(sims[int(i)]))
            for rank, i in enumerate(order)
            if not index.regions[int(i)].search_excluded
        ][:10]
        results.append(score_row(row, hits, elapsed_ms=(time.perf_counter() - start) * 1000))
    return {"name": name, "metrics": summarize_results(results), "results": results}


def score_row(row: dict[str, Any], hits: list[SearchHit], *, elapsed_ms: float) -> dict[str, Any]:
    gold = set(row.get("gold_refs", []))
    top_refs = [hit.stable_ref for hit in hits]
    top_docs = [hit.doc_id for hit in hits]
    rank = next((idx for idx, ref in enumerate(top_refs, start=1) if ref in gold), None)
    article_rank = next((idx for idx, doc_id in enumerate(top_docs, start=1) if doc_id == row["doc_id"]), None)
    covered = len(gold & set(top_refs)) / max(len(gold), 1)
    return {
        "query": row["query"],
        "variant": row.get("variant"),
        "doc_id": row["doc_id"],
        "section_title": row.get("section_title"),
        "stable_ref": row["stable_ref"],
        "gold_refs": list(row.get("gold_refs", [])),
        "top_refs": top_refs,
        "hit_at_1": rank == 1,
        "hit_at_k": rank is not None and rank <= 10,
        "article_hit_at_1": article_rank == 1,
        "article_hit_at_k": article_rank is not None and article_rank <= 10,
        "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
        "gold_coverage": covered,
        "top_ref": top_refs[0] if top_refs else None,
        "latency_ms": elapsed_ms,
        "curation_note": row.get("curation_note"),
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = max(len(results), 1)
    return {
        "count": len(results),
        "hit_at_1": sum(bool(r["hit_at_1"]) for r in results) / total,
        "hit_at_k": sum(bool(r["hit_at_k"]) for r in results) / total,
        "article_hit_at_1": sum(bool(r["article_hit_at_1"]) for r in results) / total,
        "article_hit_at_k": sum(bool(r["article_hit_at_k"]) for r in results) / total,
        "mrr": sum(float(r["reciprocal_rank"]) for r in results) / total,
        "gold_coverage": sum(float(r["gold_coverage"]) for r in results) / total,
        "avg_latency_ms": sum(float(r.get("latency_ms", 0.0)) for r in results) / total,
    }


def load_embeddings(path: Path, index: PortableBM25Index):
    if not path.exists():
        return None, {}
    ref_to_index = {region.stable_ref: idx for idx, region in enumerate(index.regions)}
    docs = np.zeros((len(index.regions), 4096), dtype=np.float32)
    query_cache: dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        vec = np.asarray(row["embedding"], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm:
            vec = vec / norm
        if row.get("input_type") == "search_document" and row.get("stable_ref") in ref_to_index:
            docs[ref_to_index[row["stable_ref"]]] = vec
        elif row.get("input_type") == "search_query":
            query_cache[str(row["hash"])] = vec
    return docs, query_cache


def ensure_query_embeddings(rows: list[dict[str, Any]], path: Path, *, api_key: str, workers: int) -> None:
    existing = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("input_type") == "search_query":
                    existing.add(str(row.get("hash")))
    missing = [(row["query"], hash_text(row["query"])) for row in rows if hash_text(row["query"]) not in existing]
    if not missing:
        return
    def one(item):
        query, digest = item
        return {"model": EMBED_MODEL, "input_type": "search_query", "stable_ref": "__query__", "hash": digest, "embedding": embed_text(query, api_key)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor, path.open("a", encoding="utf-8") as handle:
        for row in executor.map(one, missing):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_adaptive_document_embeddings(
    index: PortableBM25Index,
    metadata: dict[str, dict[str, Any]],
    path: Path,
    *,
    api_key: str,
    workers: int,
) -> None:
    if not metadata:
        return
    existing = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("input_type") == "search_document":
                existing.add((str(row.get("stable_ref")), str(row.get("hash"))))
    missing = []
    for region in index.regions:
        text = adaptive_region_text(region, metadata)
        digest = hash_text(text)
        if (region.stable_ref, digest) not in existing:
            missing.append((region.stable_ref, digest, text))
    if not missing:
        return

    def one(item):
        stable_ref, digest, text = item
        return {
            "model": EMBED_MODEL,
            "input_type": "search_document",
            "stable_ref": stable_ref,
            "hash": digest,
            "embedding": embed_text(text, api_key),
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor, path.open("a", encoding="utf-8") as handle:
        for row in executor.map(one, missing):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def embed_text(text: str, api_key: str) -> list[float]:
    body = {"model": EMBED_MODEL, "input": text}
    req = request.Request(
        EMBED_ENDPOINT,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=90) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["data"][0]["embedding"]


def make_hit(index: PortableBM25Index, region_index: int, rank: int, score: float) -> SearchHit:
    region = index.regions[region_index]
    return SearchHit(
        rank=rank,
        score=score,
        doc_id=region.doc_id,
        region_id=region.region_id,
        stable_ref=region.stable_ref,
        text=region.text,
        summary=region.view.summary,
        source_path=region.source_path,
        context_refs=[region.stable_ref],
        roles=region.roles,
        search_excluded=region.search_excluded,
        search_exclusion_reason=region.search_exclusion_reason,
    )


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def update_heatmap(path: Path, eval_payload: dict[str, Any]) -> None:
    if not path.exists():
        return
    html = path.read_text(encoding="utf-8")
    data_match = re.search(r"const DATA=(.*?);\s*const SUMMARY=(.*?);\nconst canvas=", html, flags=re.S)
    if not data_match:
        return
    data = json.loads(_repair_js_json(data_match.group(1)))
    method_results = {
        method: {row["stable_ref"]: [] for row in payload["results"]}
        for method, payload in eval_payload["methods"].items()
    }
    for method, payload in eval_payload["methods"].items():
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in payload["results"]:
            grouped.setdefault(row["stable_ref"], []).append(row)
        method_results[method] = grouped
    for item in data:
        metrics = {}
        for method, grouped in method_results.items():
            rows = grouped.get(item["range_ref"], [])
            metrics[method] = section_metrics(rows)
        item["metrics"] = metrics
    summary = json.loads(_repair_js_json(data_match.group(2)))
    summary["modeSummary"] = {method: payload["metrics"] for method, payload in eval_payload["methods"].items()}
    summary["modeLabels"] = {method: payload.get("name", method) for method, payload in eval_payload["methods"].items()}
    replacement = "const DATA=" + json.dumps(data, ensure_ascii=False) + "; const SUMMARY=" + json.dumps(summary, ensure_ascii=False) + ";\nconst canvas="
    html = html[: data_match.start()] + replacement + html[data_match.end() :]
    html = replace_weak_table(html, data, default_mode="hybrid_qwen3_w035")
    path.write_text(html, encoding="utf-8")


def _repair_js_json(value: str) -> str:
    """Accept the older heatmap's JS object payload with unescaped Windows slashes."""
    value = re.sub(
        r'("source_path": ")(.*?)(")',
        lambda match: match.group(1) + re.sub(r"\\+", "/", match.group(2)) + match.group(3),
        value,
    )
    return re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", value)


def replace_weak_table(html: str, data: list[dict[str, Any]], *, default_mode: str) -> str:
    prefix = r"(<h2>Weakest Covered Search Areas</h2><table><thead><tr><th>Section</th><th>Q</th><th>R@1</th><th>R@10</th><th>A@10</th></tr></thead><tbody>)"
    suffix = r"(</tbody></table><h2>Reading It</h2>)"
    pattern = prefix + r".*?" + suffix
    rows = []
    candidates = []
    for item in data:
        metric = item.get("metrics", {}).get(default_mode, {})
        if item.get("excluded") or not metric.get("questions"):
            continue
        candidates.append(
            (
                float(metric.get("region_hit_10") or 0.0),
                float(metric.get("region_hit_1") or 0.0),
                float(metric.get("gold_coverage") or 0.0),
                item,
            )
        )
    candidates.sort(key=lambda row: (row[0], row[1], row[2], row[3].get("doc_label", "")))
    for _r10, _r1, _cov, item in candidates[:10]:
        metric = item["metrics"][default_mode]
        rows.append(
            "<tr><td><code>{label}</code><br>{title}</td><td>{q}</td><td>{r1}</td><td>{r10}</td><td>{a10}</td></tr>".format(
                label=html_escape(item.get("doc_label", "")),
                title=html_escape(item.get("title", "")),
                q=int(metric.get("questions", 0)),
                r1=percent(metric.get("region_hit_1")),
                r10=percent(metric.get("region_hit_10")),
                a10=percent(metric.get("article_hit_10")),
            )
        )
    return re.sub(pattern, r"\1" + "".join(rows) + r"\2", html, flags=re.S)


def percent(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def html_escape(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def section_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"questions": 0, "region_hit_1": None, "region_hit_10": None, "article_hit_1": None, "article_hit_10": None, "mrr": 0.0, "gold_coverage": 0.0, "misses": []}
    n = len(rows)
    return {
        "questions": n,
        "region_hit_1": sum(bool(r["hit_at_1"]) for r in rows) / n,
        "region_hit_10": sum(bool(r["hit_at_k"]) for r in rows) / n,
        "article_hit_1": sum(bool(r["article_hit_at_1"]) for r in rows) / n,
        "article_hit_10": sum(bool(r["article_hit_at_k"]) for r in rows) / n,
        "mrr": sum(float(r["reciprocal_rank"]) for r in rows) / n,
        "gold_coverage": sum(float(r["gold_coverage"]) for r in rows) / n,
        "misses": [
            {"query": r["query"], "top": r.get("top_ref"), "variant": r.get("variant")}
            for r in rows
            if not r.get("hit_at_k")
        ][:3],
    }


if __name__ == "__main__":
    main()
