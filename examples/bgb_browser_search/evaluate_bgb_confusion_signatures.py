"""Generate and evaluate confusion-conditioned BGB retrieval signatures.

Generic generated aliases were noisy, while targeted LLM signatures repaired
hard rows. This experiment tightens the prompt further: for each hard gold
article, include the articles that BM25 repeatedly retrieves instead and ask
for short bilingual phrases that distinguish the gold article from those
confusions.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from collections import Counter, defaultdict
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

from examples.bgb_browser_search.adapt_bgb_static_views import article_regions, summarize_ranks, unique  # noqa: E402
from examples.bgb_browser_search.evaluate_bgb_intent_signatures import (  # noqa: E402
    apply_signatures,
    build_signatures,
    delta,
    evaluate,
    load_split,
    rrf_hits,
    signature_only_regions,
)
from examples.bgb_browser_search.run_bgb_stress_eval import StressQuestion, parse_json_object  # noqa: E402
from refmark.search_index import OPENROUTER_CHAT_URL, PortableBM25Index, SearchRegion, load_search_index, tokenize  # noqa: E402


PROMPT_VERSION = "bgb-confusion-signatures-v1"
DEFAULT_MODEL = "google/gemma-3-27b-it"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Evaluate confusion-conditioned BGB signatures.")
    parser.add_argument("--index", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_openrouter_index.json")
    parser.add_argument("--stress-report", action="append", required=True)
    parser.add_argument("--heatmap", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_retrieval_heatmap_3cycle.json")
    parser.add_argument("--cache", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_confusion_signatures.jsonl")
    parser.add_argument("--report", required=True)
    parser.add_argument("--output-index", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=OPENROUTER_CHAT_URL)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-articles", type=int, default=80)
    parser.add_argument("--max-confusions", type=int, default=4)
    parser.add_argument("--signature-limit", type=int, default=18)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1515)
    parser.add_argument("--top-ks", default="1,3,5,10,20,50")
    parser.add_argument("--max-article-chars", type=int, default=4200)
    parser.add_argument("--max-confusion-chars", type=int, default=1200)
    parser.add_argument("--max-train-queries", type=int, default=10)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--no-generate", action="store_true")
    args = parser.parse_args()

    source_payload = json.loads(Path(args.index).read_text(encoding="utf-8-sig"))
    source_index = load_search_index(args.index)
    base_regions = article_regions(source_index.regions)
    regions_by_ref = {region.stable_ref: region for region in base_regions}
    train_questions, eval_questions = load_split(args.stress_report, train_fraction=args.train_fraction, seed=args.seed)
    train_by_ref = questions_by_article(train_questions)
    selected = select_confusion_targets(args.heatmap, regions_by_ref, limit=args.max_articles, max_confusions=args.max_confusions)
    top_ks = tuple(int(part) for part in args.top_ks.split(",") if part.strip())

    cache_path = Path(args.cache)
    cache = read_cache(cache_path)
    if args.preflight and not args.no_generate:
        preflight(args.model, endpoint=args.endpoint, api_key=api_key(args.api_key_env))
    if not args.no_generate:
        missing = [
            target
            for target in selected
            if cache_key(target, regions_by_ref, model=args.model, signature_limit=args.signature_limit) not in cache
        ]
        generated = generate_missing(missing, regions_by_ref=regions_by_ref, train_by_ref=train_by_ref, args=args, api_key=api_key(args.api_key_env))
        append_cache(cache_path, generated)
        for row in generated:
            cache[str(row["cache_key"])] = row

    confusion_signatures = signatures_from_cache(cache, selected, regions_by_ref, model=args.model, signature_limit=args.signature_limit)
    deterministic_signatures = build_signatures(train_questions, signature_limit=24, min_df=1, max_phrase_tokens=3)
    combined_signatures = merge_signatures(deterministic_signatures, confusion_signatures, limit=30)

    baseline_index = PortableBM25Index(base_regions, include_source=True)
    deterministic_index = PortableBM25Index(apply_signatures(base_regions, deterministic_signatures), include_source=True)
    confusion_index = PortableBM25Index(apply_signatures(base_regions, confusion_signatures), include_source=True)
    combined_index = PortableBM25Index(apply_signatures(base_regions, combined_signatures), include_source=True)
    confusion_side_index = PortableBM25Index(signature_only_regions(base_regions, confusion_signatures), include_source=False)
    deterministic_side_index = PortableBM25Index(signature_only_regions(base_regions, deterministic_signatures), include_source=False)

    selected_refs = {target["gold_article"] for target in selected}
    selected_eval = [row for row in eval_questions if row.block_id in selected_refs]
    hard40_refs = set(selected_refs_from_heatmap(args.heatmap, regions_by_ref, limit=40))
    hard40_eval = [row for row in eval_questions if row.block_id in hard40_refs]

    searchers = {
        "baseline_source_plus_views": lambda query, top_k: baseline_index.search(query, top_k=top_k),
        "deterministic_signatures": lambda query, top_k: deterministic_index.search(query, top_k=top_k),
        "confusion_signatures": lambda query, top_k: confusion_index.search(query, top_k=top_k),
        "deterministic_plus_confusion": lambda query, top_k: combined_index.search(query, top_k=top_k),
        "baseline_confusion_side_rrf_w0.90": lambda query, top_k: rrf_hits(
            baseline_index.search(query, top_k=max(top_k, 120)),
            confusion_side_index.search(query, top_k=max(top_k, 120)),
            first_weight=0.90,
            rrf_k=60.0,
        )[:top_k],
        "deterministic_confusion_side_rrf_w0.90": lambda query, top_k: rrf_hits(
            deterministic_index.search(query, top_k=max(top_k, 120)),
            confusion_side_index.search(query, top_k=max(top_k, 120)),
            first_weight=0.90,
            rrf_k=60.0,
        )[:top_k],
        "deterministic_side_confusion_side_rrf_w0.70": lambda query, top_k: rrf_hits(
            deterministic_side_index.search(query, top_k=max(top_k, 120)),
            confusion_side_index.search(query, top_k=max(top_k, 120)),
            first_weight=0.70,
            rrf_k=60.0,
        )[:top_k],
    }
    report = {
        "schema": "refmark.bgb_confusion_signature_eval.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_index": args.index,
        "stress_reports": args.stress_report,
        "heatmap": args.heatmap,
        "settings": vars(args),
        "selected_articles": len(selected),
        "cached_articles": len(confusion_signatures),
        "train_questions": len(train_questions),
        "eval_questions": len(eval_questions),
        "selected_eval_questions": len(selected_eval),
        "hard40_eval_questions": len(hard40_eval),
        "selected_examples": selected[:20],
        "signature_examples": signature_examples(confusion_signatures),
        "all_eval": compare(searchers, eval_questions, top_ks=top_ks),
        "selected_eval": compare(searchers, selected_eval, top_ks=top_ks),
        "hard40_eval": compare(searchers, hard40_eval, top_ks=top_ks),
    }
    add_deltas(report["all_eval"])
    add_deltas(report["selected_eval"])
    add_deltas(report["hard40_eval"])
    if args.output_index:
        write_index(source_payload, apply_signatures(base_regions, combined_signatures), Path(args.output_index), signatures=combined_signatures, settings=vars(args))
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def select_confusion_targets(path: str, regions_by_ref: dict[str, SearchRegion], *, limit: int, max_confusions: int) -> list[dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    confusions: dict[str, Counter[str]] = defaultdict(Counter)
    for row in payload.get("confusion_pairs", []):
        gold = str(row.get("gold_article", ""))
        wrong = str(row.get("wrong_top_article", ""))
        if gold in regions_by_ref and wrong in regions_by_ref:
            confusions[gold][wrong] += int(row.get("count", 1))
    targets: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in payload.get("hard_articles_by_miss10", []):
        gold = str(row.get("article_ref", ""))
        if gold not in regions_by_ref:
            continue
        wrongs = [ref for ref, _count in confusions.get(gold, Counter()).most_common(max_confusions)]
        if not wrongs:
            continue
        seen.add(gold)
        targets.append(
            {
                "gold_article": gold,
                "miss_rate_at_10": row.get("miss_rate_at_10"),
                "queries": row.get("queries"),
                "confusions": wrongs,
            }
        )
        if len(targets) >= limit:
            break
    if len(targets) < limit:
        hard_by_ref = {str(row.get("article_ref", "")): row for row in payload.get("hard_articles_by_miss10", [])}
        for gold, wrong_counter in sorted(confusions.items(), key=lambda item: (-sum(item[1].values()), item[0])):
            if gold in seen or gold not in regions_by_ref:
                continue
            row = hard_by_ref.get(gold, {})
            targets.append(
                {
                    "gold_article": gold,
                    "miss_rate_at_10": row.get("miss_rate_at_10"),
                    "queries": row.get("queries"),
                    "confusions": [ref for ref, _count in wrong_counter.most_common(max_confusions)],
                }
            )
            seen.add(gold)
            if len(targets) >= limit:
                break
    return targets


def selected_refs_from_heatmap(path: str, regions_by_ref: dict[str, SearchRegion], *, limit: int) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    refs = []
    for row in payload.get("hard_articles_by_miss10", []):
        ref = str(row.get("article_ref", ""))
        if ref in regions_by_ref:
            refs.append(ref)
        if len(refs) >= limit:
            break
    return refs


def questions_by_article(questions: list[StressQuestion]) -> dict[str, list[StressQuestion]]:
    output: dict[str, list[StressQuestion]] = defaultdict(list)
    for question in questions:
        output[question.block_id].append(question)
    return output


def generate_missing(
    targets: list[dict[str, object]],
    *,
    regions_by_ref: dict[str, SearchRegion],
    train_by_ref: dict[str, list[StressQuestion]],
    args: argparse.Namespace,
    api_key: str,
) -> list[dict[str, object]]:
    if not targets:
        return []

    def one(target: dict[str, object]) -> dict[str, object] | None:
        try:
            signatures = generate_signatures(target, regions_by_ref, train_by_ref, args=args, api_key=api_key)
            return cache_row(target, regions_by_ref, signatures, model=args.model, signature_limit=args.signature_limit)
        except Exception as exc:
            print(f"confusion signature generation failed: {target.get('gold_article')}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return None

    rows: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        for row in executor.map(one, targets):
            if row:
                rows.append(row)
    return rows


def generate_signatures(
    target: dict[str, object],
    regions_by_ref: dict[str, SearchRegion],
    train_by_ref: dict[str, list[StressQuestion]],
    *,
    args: argparse.Namespace,
    api_key: str,
) -> list[str]:
    body = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You create compact bilingual retrieval metadata for a legal search index. "
                    "Return strict JSON only. Do not provide legal advice."
                ),
            },
            {"role": "user", "content": prompt(target, regions_by_ref, train_by_ref, args=args)},
        ],
        "temperature": 0.25,
        "max_tokens": 800,
    }
    payload = openrouter_json(body, endpoint=args.endpoint, api_key=api_key, timeout=90)
    parsed = parse_json_object(payload["choices"][0]["message"]["content"])
    raw = parsed.get("signatures", [])
    return clean_signatures(raw if isinstance(raw, list) else [], limit=args.signature_limit)


def prompt(
    target: dict[str, object],
    regions_by_ref: dict[str, SearchRegion],
    train_by_ref: dict[str, list[StressQuestion]],
    *,
    args: argparse.Namespace,
) -> str:
    gold_ref = str(target["gold_article"])
    gold = regions_by_ref[gold_ref]
    confusions = [regions_by_ref[str(ref)] for ref in target.get("confusions", []) if str(ref) in regions_by_ref]
    query_lines = "\n".join(f"- [{row.language}/{row.style}] {row.query}" for row in train_by_ref.get(gold_ref, [])[: args.max_train_queries])
    confusion_text = "\n\n".join(
        f"CONFUSED ARTICLE {region.stable_ref}\n{region.text[: args.max_confusion_chars]}"
        for region in confusions
    )
    return f"""Create up to {args.signature_limit} minimal discriminative search signatures for the GOLD article.

The current BM25 system often retrieves the CONFUSED articles instead. Your
signatures will be added to the GOLD article only. They must help a search
engine find GOLD and avoid the confused articles.

Rules:
- Mix German and English.
- Use short phrases under 8 words.
- Prefer distinctive legal/user-problem vocabulary.
- Include negative distinctions only as natural phrases, e.g. "defect remedies not limitation period".
- Avoid generic phrases that could match many BGB articles.
- Do not include article numbers, refs, paragraph signs, or "BGB".
- Do not copy long source sentences.
- Return strict JSON only: {{"signatures":["..."]}}

Held-out-safe training queries for GOLD:
{query_lines or "- none"}

GOLD ARTICLE {gold.stable_ref}
{gold.text[: args.max_article_chars]}

{confusion_text}
"""


def clean_signatures(values: list[object], *, limit: int) -> list[str]:
    output: list[str] = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("text", "")
        text = " ".join(str(value).strip().split())
        text = re.sub(r"\b(?:bgb|Â§+\s*\d+[a-zA-Z]*)\b", "", text, flags=re.IGNORECASE)
        text = " ".join(text.strip(" -:;,.").split())
        if not text or len(text) > 90 or len(text.split()) > 10:
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
                    "X-Title": "refmark BGB confusion signatures",
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


def cache_key(target: dict[str, object], regions_by_ref: dict[str, SearchRegion], *, model: str, signature_limit: int) -> str:
    gold_ref = str(target["gold_article"])
    gold = regions_by_ref[gold_ref]
    confusions = [str(ref) for ref in target.get("confusions", [])]
    raw = json.dumps(
        {
            "prompt_version": PROMPT_VERSION,
            "gold_article": gold_ref,
            "gold_hash": gold.hash,
            "confusions": confusions,
            "confusion_hashes": [regions_by_ref[ref].hash for ref in confusions if ref in regions_by_ref],
            "model": model,
            "signature_limit": signature_limit,
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def cache_row(target: dict[str, object], regions_by_ref: dict[str, SearchRegion], signatures: list[str], *, model: str, signature_limit: int) -> dict[str, object]:
    return {
        "schema": "refmark.bgb_confusion_signature_cache.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "cache_key": cache_key(target, regions_by_ref, model=model, signature_limit=signature_limit),
        "gold_article": target["gold_article"],
        "gold_hash": regions_by_ref[str(target["gold_article"])].hash,
        "confusions": target.get("confusions", []),
        "model": model,
        "signature_limit": signature_limit,
        "signatures": signatures,
    }


def read_cache(path: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[str(row["cache_key"])] = row
    return rows


def append_cache(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def signatures_from_cache(
    cache: dict[str, dict[str, object]],
    selected: list[dict[str, object]],
    regions_by_ref: dict[str, SearchRegion],
    *,
    model: str,
    signature_limit: int,
) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for target in selected:
        key = cache_key(target, regions_by_ref, model=model, signature_limit=signature_limit)
        row = cache.get(key)
        if row:
            output[str(target["gold_article"])] = clean_signatures(list(row.get("signatures", [])), limit=signature_limit)
    return output


def merge_signatures(first: dict[str, list[str]], second: dict[str, list[str]], *, limit: int) -> dict[str, list[str]]:
    refs = set(first) | set(second)
    return {ref: unique([*second.get(ref, []), *first.get(ref, [])])[:limit] for ref in refs}


def compare(searchers: dict[str, object], questions: list[StressQuestion], *, top_ks: tuple[int, ...]) -> dict[str, object]:
    return {name: evaluate(fn, questions, top_ks=top_ks) for name, fn in searchers.items()}


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
        "adaptation": "bgb-confusion-signatures",
        "confusion_signature_settings": settings,
        "confusion_signature_articles": len(signatures),
    }
    payload["regions"] = [region.to_dict() for region in regions]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
