"""Generate and evaluate cached LLM intent signatures for BGB articles.

The deterministic signature experiment showed that compact intent phrases help
while raw generated questions hurt. This script tests the next step: ask a
cheap LLM to produce short bilingual search signatures for heatmap-hard
articles, cache them per article hash, and evaluate whether those signatures
improve held-out retrieval.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib import error, request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.bgb_browser_search.adapt_bgb_static_views import article_regions, split_questions_by_block, stress_questions, summarize_ranks, unique  # noqa: E402
from examples.bgb_browser_search.evaluate_bgb_intent_signatures import apply_signatures, build_signatures, delta, evaluate, load_split, rrf_hits, signature_only_regions  # noqa: E402
from examples.bgb_browser_search.run_bgb_stress_eval import StressQuestion, parse_json_object  # noqa: E402
from refmark.search_index import OPENROUTER_CHAT_URL, PortableBM25Index, SearchRegion, load_search_index  # noqa: E402


PROMPT_VERSION = "bgb-llm-intent-signatures-v1"
DEFAULT_MODEL = "qwen/qwen-turbo"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Evaluate cached LLM-generated BGB intent signatures.")
    parser.add_argument("--index", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_openrouter_index.json")
    parser.add_argument("--stress-report", action="append", required=True)
    parser.add_argument("--heatmap", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_retrieval_heatmap_3cycle.json")
    parser.add_argument("--cache", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_llm_intent_signatures.jsonl")
    parser.add_argument("--report", required=True)
    parser.add_argument("--output-index", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=OPENROUTER_CHAT_URL)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-hard-articles", type=int, default=40)
    parser.add_argument("--signature-limit", type=int, default=24)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1515)
    parser.add_argument("--top-ks", default="1,3,5,10,20,50")
    parser.add_argument("--max-article-chars", type=int, default=5500)
    parser.add_argument("--max-train-queries", type=int, default=16)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--no-generate", action="store_true", help="Evaluate only cached signatures.")
    args = parser.parse_args()

    source_payload = json.loads(Path(args.index).read_text(encoding="utf-8-sig"))
    source_index = load_search_index(args.index)
    base_regions = article_regions(source_index.regions)
    regions_by_ref = {region.stable_ref: region for region in base_regions}
    train_questions, eval_questions = load_split(args.stress_report, train_fraction=args.train_fraction, seed=args.seed)
    selected_refs = select_hard_articles(args.heatmap, regions_by_ref, limit=args.max_hard_articles)
    train_by_ref = questions_by_article(train_questions)
    top_ks = tuple(int(part) for part in args.top_ks.split(",") if part.strip())

    cache_path = Path(args.cache)
    cache = read_signature_cache(cache_path)
    if args.preflight and not args.no_generate:
        preflight(args.model, endpoint=args.endpoint, api_key=api_key(args.api_key_env))
    if not args.no_generate:
        missing = [
            region
            for ref in selected_refs
            if (region := regions_by_ref.get(ref)) is not None
            and cache_key(
                region,
                train_by_ref.get(ref, []),
                model=args.model,
                signature_limit=args.signature_limit,
                max_train_queries=args.max_train_queries,
                max_article_chars=args.max_article_chars,
            )
            not in cache
        ]
        generated = generate_missing(
            missing,
            train_by_ref=train_by_ref,
            args=args,
            api_key=api_key(args.api_key_env),
        )
        append_signature_cache(cache_path, generated)
        for row in generated:
            cache[row["cache_key"]] = row

    llm_signatures = signatures_from_cache(
        cache,
        selected_refs,
        regions_by_ref,
        train_by_ref=train_by_ref,
        model=args.model,
        signature_limit=args.signature_limit,
        max_train_queries=args.max_train_queries,
        max_article_chars=args.max_article_chars,
    )
    deterministic_signatures = build_signatures(
        train_questions,
        signature_limit=args.signature_limit,
        min_df=1,
        max_phrase_tokens=3,
    )
    combined_signatures = merge_signatures(deterministic_signatures, llm_signatures, limit=args.signature_limit)

    baseline_regions = base_regions
    deterministic_regions = apply_signatures(base_regions, deterministic_signatures)
    llm_regions = apply_signatures(base_regions, llm_signatures)
    combined_regions = apply_signatures(base_regions, combined_signatures)

    baseline_index = PortableBM25Index(baseline_regions, include_source=True)
    deterministic_index = PortableBM25Index(deterministic_regions, include_source=True)
    llm_index = PortableBM25Index(llm_regions, include_source=True)
    combined_index = PortableBM25Index(combined_regions, include_source=True)
    llm_side_index = PortableBM25Index(signature_only_regions(base_regions, llm_signatures), include_source=False)

    selected_eval = [row for row in eval_questions if row.block_id in selected_refs]
    all_searchers = searchers(
        baseline_index=baseline_index,
        deterministic_index=deterministic_index,
        llm_index=llm_index,
        combined_index=combined_index,
        llm_side_index=llm_side_index,
    )
    report = {
        "schema": "refmark.bgb_llm_intent_signature_eval.v1",
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
        "all_eval": compare_searchers(all_searchers, eval_questions, top_ks=top_ks),
        "selected_eval": compare_searchers(all_searchers, selected_eval, top_ks=top_ks),
    }
    add_deltas(report["all_eval"])
    add_deltas(report["selected_eval"])
    if args.output_index:
        write_index(source_payload, combined_regions, Path(args.output_index), signatures=combined_signatures, settings=vars(args))
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def select_hard_articles(heatmap_path: str, regions_by_ref: dict[str, SearchRegion], *, limit: int) -> list[str]:
    if not heatmap_path or not Path(heatmap_path).exists():
        return list(sorted(regions_by_ref))[:limit]
    payload = json.loads(Path(heatmap_path).read_text(encoding="utf-8-sig"))
    refs: list[str] = []
    for section in ("hard_articles_by_miss10", "hard_articles_by_miss50"):
        for row in payload.get(section, []):
            ref = str(row.get("article_ref", ""))
            if ref in regions_by_ref and ref not in refs:
                refs.append(ref)
            if len(refs) >= limit:
                return refs
    return refs[:limit]


def questions_by_article(questions: list[StressQuestion]) -> dict[str, list[StressQuestion]]:
    output: dict[str, list[StressQuestion]] = defaultdict(list)
    for question in questions:
        output[question.block_id].append(question)
    return output


def generate_missing(
    regions: list[SearchRegion],
    *,
    train_by_ref: dict[str, list[StressQuestion]],
    args: argparse.Namespace,
    api_key: str,
) -> list[dict[str, object]]:
    if not regions:
        return []

    def one(region: SearchRegion) -> dict[str, object] | None:
        try:
            signatures = generate_signatures(region, train_by_ref.get(region.stable_ref, []), args=args, api_key=api_key)
            return cache_row(
                region,
                train_by_ref.get(region.stable_ref, []),
                signatures,
                model=args.model,
                signature_limit=args.signature_limit,
                max_train_queries=args.max_train_queries,
                max_article_chars=args.max_article_chars,
            )
        except Exception as exc:
            print(f"signature generation failed: {region.stable_ref}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return None

    rows: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        for row in executor.map(one, regions):
            if row:
                rows.append(row)
    return rows


def generate_signatures(region: SearchRegion, train_questions: list[StressQuestion], *, args: argparse.Namespace, api_key: str) -> list[str]:
    body = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You create compact bilingual retrieval metadata for a legal documentation search index. "
                    "Return strict JSON only. Do not provide legal advice."
                ),
            },
            {"role": "user", "content": signature_prompt(region, train_questions, args=args)},
        ],
        "temperature": 0.35,
        "max_tokens": 700,
    }
    payload = openrouter_json(body, endpoint=args.endpoint, api_key=api_key, timeout=90)
    parsed = parse_json_object(payload["choices"][0]["message"]["content"])
    raw = parsed.get("signatures", [])
    if not isinstance(raw, list):
        raw = []
    return clean_signatures(raw, limit=args.signature_limit)


def signature_prompt(region: SearchRegion, train_questions: list[StressQuestion], *, args: argparse.Namespace) -> str:
    question_lines = "\n".join(f"- [{row.language}/{row.style}] {row.query}" for row in train_questions[: args.max_train_queries])
    text = region.text[: args.max_article_chars]
    return f"""Create up to {args.signature_limit} compact search intent signatures for this BGB article.

Goal: improve article-level retrieval for realistic German and English user
queries. The signatures will be mixed into a BM25 index, so each item should be
a short phrase, synonym set, layperson formulation, or hard-negative distinction.

Rules:
- Mix German and English.
- Prefer phrases under 8 words.
- Include layperson wording, legal terms, synonyms, and situations.
- Do not include article numbers, refs, paragraph numbers, or "BGB".
- Do not copy long source sentences.
- Do not include generic words like "law", "right", "claim" alone.
- Return strict JSON only: {{"signatures":["..."]}}

Held-out-safe training queries for this same article:
{question_lines or "- none"}

Article ref for cache only: {region.stable_ref}
Article text:
{text}
"""


def clean_signatures(values: list[object], *, limit: int) -> list[str]:
    output: list[str] = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("text", "")
        text = " ".join(str(value).strip().split())
        text = re.sub(r"\b(?:bgb|§+\s*\d+[a-zA-Z]*)\b", "", text, flags=re.IGNORECASE)
        text = " ".join(text.strip(" -:;,.").split())
        if not text or len(text) > 90:
            continue
        if len(text.split()) > 10:
            continue
        output.append(text)
    return unique(output)[:limit]


def openrouter_json(body: dict[str, object], *, endpoint: str, api_key: str, timeout: int) -> dict[str, object]:
    last_exc: Exception | None = None
    for attempt in range(4):
        try:
            req = request.Request(
                endpoint,
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/b-imenitov/refmark",
                    "X-Title": "refmark BGB intent signatures",
                },
                method="POST",
            )
            with request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (error.HTTPError, error.URLError, TimeoutError) as exc:
            last_exc = exc
            time.sleep(min(2**attempt, 8))
    assert last_exc is not None
    raise last_exc


def preflight(model: str, *, endpoint: str, api_key: str) -> None:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": 'Return {"signatures":["test"]}.'},
        ],
        "temperature": 0.0,
        "max_tokens": 20,
    }
    parsed = openrouter_json(body, endpoint=endpoint, api_key=api_key, timeout=30)
    parse_json_object(parsed["choices"][0]["message"]["content"])
    print(f"preflight ok: {model}", file=sys.stderr)


def api_key(env_name: str) -> str:
    value = os.environ.get(env_name)
    if not value:
        raise SystemExit(f"{env_name} is not set")
    return value


def cache_key(
    region: SearchRegion,
    train_questions: list[StressQuestion],
    *,
    model: str,
    signature_limit: int,
    max_train_queries: int,
    max_article_chars: int,
) -> str:
    raw = json.dumps(
        {
            "prompt_version": PROMPT_VERSION,
            "article_ref": region.stable_ref,
            "article_hash": region.hash,
            "article_text_prefix_hash": hashlib.sha256(region.text[:max_article_chars].encode("utf-8")).hexdigest()[:16],
            "train_query_context": [
                {
                    "language": row.language,
                    "style": row.style,
                    "query": row.query,
                }
                for row in train_questions[:max_train_queries]
            ],
            "model": model,
            "signature_limit": signature_limit,
            "max_train_queries": max_train_queries,
            "max_article_chars": max_article_chars,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def cache_row(
    region: SearchRegion,
    train_questions: list[StressQuestion],
    signatures: list[str],
    *,
    model: str,
    signature_limit: int,
    max_train_queries: int,
    max_article_chars: int,
) -> dict[str, object]:
    key = cache_key(
        region,
        train_questions,
        model=model,
        signature_limit=signature_limit,
        max_train_queries=max_train_queries,
        max_article_chars=max_article_chars,
    )
    return {
        "schema": "refmark.bgb_llm_intent_signature_cache.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "cache_key": key,
        "article_ref": region.stable_ref,
        "article_hash": region.hash,
        "model": model,
        "signature_limit": signature_limit,
        "max_train_queries": max_train_queries,
        "max_article_chars": max_article_chars,
        "signatures": signatures,
    }


def read_signature_cache(path: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[str(row["cache_key"])] = row
    return rows


def append_signature_cache(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def signatures_from_cache(
    cache: dict[str, dict[str, object]],
    selected_refs: list[str],
    regions_by_ref: dict[str, SearchRegion],
    *,
    train_by_ref: dict[str, list[StressQuestion]],
    model: str,
    signature_limit: int,
    max_train_queries: int,
    max_article_chars: int,
) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for ref in selected_refs:
        region = regions_by_ref.get(ref)
        if not region:
            continue
        key = cache_key(
            region,
            train_by_ref.get(ref, []),
            model=model,
            signature_limit=signature_limit,
            max_train_queries=max_train_queries,
            max_article_chars=max_article_chars,
        )
        row = cache.get(key)
        if not row:
            continue
        output[ref] = clean_signatures(list(row.get("signatures", [])), limit=signature_limit)
    return output


def merge_signatures(first: dict[str, list[str]], second: dict[str, list[str]], *, limit: int) -> dict[str, list[str]]:
    refs = set(first) | set(second)
    return {ref: unique([*second.get(ref, []), *first.get(ref, [])])[:limit] for ref in refs}


def searchers(
    *,
    baseline_index: PortableBM25Index,
    deterministic_index: PortableBM25Index,
    llm_index: PortableBM25Index,
    combined_index: PortableBM25Index,
    llm_side_index: PortableBM25Index,
):
    return {
        "baseline_source_plus_views": lambda query, top_k: baseline_index.search(query, top_k=top_k),
        "deterministic_signatures": lambda query, top_k: deterministic_index.search(query, top_k=top_k),
        "llm_hard_article_signatures": lambda query, top_k: llm_index.search(query, top_k=top_k),
        "deterministic_plus_llm": lambda query, top_k: combined_index.search(query, top_k=top_k),
        "baseline_llm_side_rrf_w0.90": lambda query, top_k: rrf_hits(
            baseline_index.search(query, top_k=max(top_k, 80)),
            llm_side_index.search(query, top_k=max(top_k, 80)),
            first_weight=0.90,
            rrf_k=60.0,
        )[:top_k],
        "baseline_llm_side_rrf_w0.97": lambda query, top_k: rrf_hits(
            baseline_index.search(query, top_k=max(top_k, 80)),
            llm_side_index.search(query, top_k=max(top_k, 80)),
            first_weight=0.97,
            rrf_k=60.0,
        )[:top_k],
    }


def compare_searchers(searchers_by_name, questions: list[StressQuestion], *, top_ks: tuple[int, ...]) -> dict[str, object]:
    return {name: evaluate(fn, questions, top_ks=top_ks) for name, fn in searchers_by_name.items()}


def add_deltas(section: dict[str, object]) -> None:
    baseline = section["baseline_source_plus_views"]
    section["deltas_vs_baseline"] = {
        name: delta(baseline, value)
        for name, value in section.items()
        if name not in {"baseline_source_plus_views", "deltas_vs_baseline"}
    }


def signature_examples(signatures: dict[str, list[str]]) -> list[dict[str, object]]:
    return [{"article_ref": ref, "signatures": values[:12]} for ref, values in sorted(signatures.items())[:20]]


def write_index(source_payload: dict[str, object], regions: list[SearchRegion], path: Path, *, signatures: dict[str, list[str]], settings: dict[str, object]) -> None:
    payload = dict(source_payload)
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    payload["settings"] = {
        **dict(source_payload.get("settings", {})),
        "adaptation": "bgb-llm-intent-signatures",
        "intent_signature_settings": settings,
        "intent_signature_articles": len(signatures),
    }
    payload["regions"] = [region.to_dict() for region in regions]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
