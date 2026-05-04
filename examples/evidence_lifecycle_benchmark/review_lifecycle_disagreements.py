from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import html
import json
import os
from pathlib import Path
import random
from difflib import SequenceMatcher
import re
import sys
import time
from typing import Any
from urllib import request, error

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from refmark.lifecycle import (
    FuzzyRegionIndex,
    Region,
    _file_fingerprints,
    _quote_index,
    load_regions,
    quote_selector,
    safe_name,
    token_jaccard,
)


OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODELS = [
    "moonshotai/kimi-k2.6",
    "deepseek/deepseek-v4-pro",
    "x-ai/grok-4.3",
]
DEFAULT_CATEGORIES = [
    "quote_selector_silent_drift",
    "quote_preserved_layered_review",
    "layered_review_valid",
    "refmark_fuzzy_review",
    "hash_false_stale_refmark_valid",
    "qrels_false_stale_refmark_valid",
]
VALID_VERDICTS = {
    "valid_unchanged",
    "valid_moved",
    "valid_rewritten",
    "split_support",
    "stale",
    "deleted",
    "ambiguous",
    "alternative_valid",
    "invalid_original_label",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample and LLM-review lifecycle benchmark disagreement cases.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser("sample", help="Create review cards from lifecycle benchmark artifacts.")
    sample_parser.add_argument("inputs", nargs="+", help="layered lifecycle benchmark JSON files.")
    sample_parser.add_argument("--output", required=True, help="Review cards JSONL output.")
    sample_parser.add_argument("--html", default="", help="Optional human-review HTML output.")
    sample_parser.add_argument("--sample-size", type=int, default=120)
    sample_parser.add_argument("--per-category", type=int, default=25)
    sample_parser.add_argument("--seed", type=int, default=7)
    sample_parser.add_argument("--text-chars", type=int, default=1600)
    sample_parser.add_argument("--scan-per-revision", type=int, default=700)

    judge_parser = subparsers.add_parser("judge", help="Run OpenRouter LLM judges over review cards.")
    judge_parser.add_argument("--cards", required=True)
    judge_parser.add_argument("--output", required=True, help="Judgments JSONL output.")
    judge_parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    judge_parser.add_argument("--endpoint", default=OPENROUTER_CHAT_URL)
    judge_parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    judge_parser.add_argument("--limit", type=int, default=60)
    judge_parser.add_argument("--parallel", type=int, default=3)
    judge_parser.add_argument("--timeout", type=int, default=90)
    judge_parser.add_argument("--text-chars", type=int, default=1400)
    judge_parser.add_argument("--max-output-tokens", type=int, default=900)
    judge_parser.add_argument("--preflight", action="store_true")

    summarize_parser = subparsers.add_parser("summarize", help="Summarize LLM review outcomes.")
    summarize_parser.add_argument("--cards", required=True)
    summarize_parser.add_argument("--judgments", required=True)
    summarize_parser.add_argument("--output", default="")
    summarize_parser.add_argument("--format", choices=["markdown", "json"], default="markdown")

    worksheet_parser = subparsers.add_parser("worksheet", help="Create a human-review worksheet from cards and judgments.")
    worksheet_parser.add_argument("--cards", required=True)
    worksheet_parser.add_argument("--judgments", required=True)
    worksheet_parser.add_argument("--output-csv", required=True)
    worksheet_parser.add_argument("--output-html", default="")
    worksheet_parser.add_argument("--limit", type=int, default=50)
    worksheet_parser.add_argument("--text-chars", type=int, default=2200)

    filled_parser = subparsers.add_parser("filled-html", help="Render a filled human-review CSV as HTML.")
    filled_parser.add_argument("--input-csv", required=True)
    filled_parser.add_argument("--output-html", required=True)

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Summarize filled review labels into lifecycle-rule calibration evidence.",
    )
    calibrate_parser.add_argument("--input-csv", required=True)
    calibrate_parser.add_argument("--output", default="")
    calibrate_parser.add_argument("--format", choices=["markdown", "json"], default="markdown")

    args = parser.parse_args()
    if args.command == "sample":
        cards = sample_cards(
            [Path(path) for path in args.inputs],
            sample_size=args.sample_size,
            per_category=args.per_category,
            seed=args.seed,
            text_chars=args.text_chars,
            scan_per_revision=args.scan_per_revision,
        )
        write_jsonl(Path(args.output), cards)
        if args.html:
            Path(args.html).parent.mkdir(parents=True, exist_ok=True)
            Path(args.html).write_text(render_cards_html(cards), encoding="utf-8")
        print(json.dumps({"cards": len(cards), "output": args.output, "html": args.html or None}, indent=2))
    elif args.command == "judge":
        cards = read_jsonl(Path(args.cards))
        results = judge_cards(
            cards,
            output=Path(args.output),
            models=[model.strip() for model in args.models.split(",") if model.strip()],
            endpoint=args.endpoint,
            api_key=os.environ.get(args.api_key_env, ""),
            limit=args.limit,
            parallel=args.parallel,
            timeout=args.timeout,
            text_chars=args.text_chars,
            max_output_tokens=args.max_output_tokens,
            preflight=args.preflight,
        )
        print(json.dumps(results, indent=2))
    elif args.command == "summarize":
        cards = read_jsonl(Path(args.cards))
        judgments = read_jsonl(Path(args.judgments))
        summary = summarize(cards, judgments)
        rendered = json.dumps(summary, indent=2) if args.format == "json" else render_summary_markdown(summary)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(rendered, encoding="utf-8")
        print(rendered)
    elif args.command == "worksheet":
        cards = read_jsonl(Path(args.cards))
        judgments = read_jsonl(Path(args.judgments))
        rows = human_review_rows(cards, judgments, limit=args.limit, text_chars=args.text_chars)
        write_worksheet_csv(Path(args.output_csv), rows)
        if args.output_html:
            Path(args.output_html).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output_html).write_text(render_worksheet_html(rows), encoding="utf-8")
        print(json.dumps({"rows": len(rows), "csv": args.output_csv, "html": args.output_html or None}, indent=2))
    elif args.command == "filled-html":
        rows = read_csv_dicts(Path(args.input_csv))
        Path(args.output_html).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_html).write_text(render_filled_worksheet_html(rows), encoding="utf-8")
        print(json.dumps({"rows": len(rows), "html": args.output_html}, indent=2))
    elif args.command == "calibrate":
        rows = read_csv_dicts(Path(args.input_csv))
        calibration = calibrate_filled_review(rows)
        rendered = (
            json.dumps(calibration, indent=2, ensure_ascii=False)
            if args.format == "json"
            else render_calibration_markdown(calibration)
        )
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(rendered, encoding="utf-8")
        print(rendered)


def sample_cards(
    inputs: list[Path],
    *,
    sample_size: int,
    per_category: int,
    seed: int,
    text_chars: int,
    scan_per_revision: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        settings = payload["settings"]
        work_dir = Path(settings["work_dir"])
        old_regions = load_regions(
            work_dir / "old",
            region_tokens=int(settings.get("region_tokens", 110)),
            region_stride=int(settings.get("region_stride", 110)),
            max_files=int(settings.get("max_files", 0)),
        )
        old_by_ref = {artifact_ref(region): region for regions in old_regions.values() for region in regions}
        for report in payload["revision_reports"]:
            new_ref = str(report["new_ref"])
            new_regions = load_regions(
                work_dir / f"new_{safe_name(new_ref)}",
                region_tokens=int(settings.get("region_tokens", 110)),
                region_stride=int(settings.get("region_stride", 110)),
                max_files=int(settings.get("max_files", 0)),
            )
            for card in cards_for_revision(
                path,
                payload,
                report,
                old_regions,
                new_regions,
                old_by_ref,
                text_chars=text_chars,
                scan_per_revision=scan_per_revision,
                rng=rng,
            ):
                for category in card["categories"]:
                    if len(buckets[category]) < per_category * 8:
                        buckets[category].append(card)

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for category in DEFAULT_CATEGORIES:
        rows = list(buckets.get(category, []))
        rng.shuffle(rows)
        for card in diverse_take(rows, per_category):
            if card["card_id"] not in seen:
                selected.append(card)
                seen.add(card["card_id"])
    if len(selected) < sample_size:
        remainder = [card for rows in buckets.values() for card in rows if card["card_id"] not in seen]
        rng.shuffle(remainder)
        for card in remainder:
            if len(selected) >= sample_size:
                break
            selected.append(card)
            seen.add(card["card_id"])
    rng.shuffle(selected)
    return selected[:sample_size]


def diverse_take(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    by_artifact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_artifact[str(row.get("artifact", ""))].append(row)
    selected: list[dict[str, Any]] = []
    while len(selected) < limit and by_artifact:
        for artifact in sorted(list(by_artifact)):
            if not by_artifact[artifact]:
                by_artifact.pop(artifact, None)
                continue
            selected.append(by_artifact[artifact].pop())
            if len(selected) >= limit:
                break
    return selected


def cards_for_revision(
    artifact_path: Path,
    payload: dict[str, Any],
    report: dict[str, Any],
    old_regions: dict[str, list[Region]],
    new_regions: dict[str, list[Region]],
    old_by_ref: dict[str, Region],
    *,
    text_chars: int,
    scan_per_revision: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    stable_status = report["stable_ref_migration"].get("status_by_ref", {})
    new_all = [region for regions in new_regions.values() for region in regions]
    new_by_ref = {artifact_ref(region): region for region in new_all}
    new_by_path_ord = {path: {region.ordinal: region for region in regions} for path, regions in new_regions.items()}
    old_file_hashes = _file_fingerprints(old_regions)
    new_file_hashes = _file_fingerprints(new_regions)
    quote_index = _quote_index(new_all)
    fuzzy_index = FuzzyRegionIndex(new_all)
    cards: list[dict[str, Any]] = []
    old_items = list(old_by_ref.items())
    rng.shuffle(old_items)
    if scan_per_revision > 0:
        old_items = old_items[:scan_per_revision]
    for old_ref, old_region in old_items:
        stable = stable_status.get(old_ref, "stale")
        valid = stable != "stale"
        same_candidate = new_by_path_ord.get(old_region.path, {}).get(old_region.ordinal)
        quote_hits = quote_index.get(quote_selector(old_region.text), [])
        quote_candidate = quote_hits[0] if len(quote_hits) == 1 else None
        stable_candidate = stable_candidate_for(old_region, stable, same_candidate, new_by_ref, fuzzy_index)

        quote_decision = quote_selector_decision(old_region, same_candidate, quote_hits, valid)
        layered_decision = layered_decision_for(old_region, same_candidate, quote_hits, valid)
        hash_decision = "preserved" if same_candidate and same_candidate.fingerprint == old_region.fingerprint else ("false_stale_alert" if valid else "true_stale_alert")
        qrels_decision = "preserved" if old_file_hashes.get(old_region.path) == new_file_hashes.get(old_region.path) else ("false_stale_alert" if valid else "true_stale_alert")

        categories: list[str] = []
        review_candidate = stable_candidate
        review_focus = "stable_candidate"
        if quote_decision == "silent_wrong":
            categories.append("quote_selector_silent_drift")
            review_candidate = quote_candidate
            review_focus = "quote_candidate"
        if quote_decision == "preserved" and layered_decision != "preserved":
            categories.append("quote_preserved_layered_review")
            review_candidate = quote_candidate
            review_focus = "quote_candidate"
        if layered_decision == "review_needed" and valid:
            categories.append("layered_review_valid")
            review_candidate = stable_candidate or quote_candidate
            review_focus = "layered_review_candidate"
        if stable == "fuzzy":
            categories.append("refmark_fuzzy_review")
            review_candidate = stable_candidate
            review_focus = "stable_fuzzy_candidate"
        if hash_decision == "false_stale_alert":
            categories.append("hash_false_stale_refmark_valid")
        if qrels_decision == "false_stale_alert":
            categories.append("qrels_false_stale_refmark_valid")
        if not categories:
            continue

        candidate = review_candidate or quote_candidate or same_candidate
        if candidate is None:
            continue
        candidate_ref = artifact_ref(candidate)
        card_id = stable_card_id(artifact_path, str(report["new_ref"]), old_ref, review_focus, candidate_ref)
        cards.append(
            {
                "schema": "refmark.lifecycle_review_card.v1",
                "card_id": card_id,
                "artifact": str(artifact_path),
                "repo_url": payload.get("repo_url"),
                "subdir": payload.get("subdir"),
                "old_ref_version": payload.get("old_ref"),
                "new_ref_version": report.get("new_ref"),
                "old_ref": old_ref,
                "old_path": old_region.path,
                "candidate_ref": candidate_ref,
                "candidate_path": candidate.path,
                "review_focus": review_focus,
                "categories": sorted(set(categories)),
                "stable_status": stable,
                "method_decisions": {
                    "chunk_id_content_hash": hash_decision,
                    "qrels_source_hash": qrels_decision,
                    "chunk_hash_quote_selector": quote_decision,
                    "refmark_layered_selector": layered_decision,
                    "refmark_exact_migration": "preserved" if stable in {"same_file_exact", "moved_exact"} else ("review_needed" if stable == "fuzzy" else "true_stale_alert"),
                },
                "signals": {
                    "quote_hits": len(quote_hits),
                    "candidate_similarity": round(token_jaccard(old_region.text, candidate.text), 4),
                    "same_path_ordinal_ref": artifact_ref(same_candidate) if same_candidate else None,
                    "same_path_ordinal_similarity": round(token_jaccard(old_region.text, same_candidate.text), 4) if same_candidate else None,
                    "quote_candidate_ref": artifact_ref(quote_candidate) if quote_candidate else None,
                    "stable_candidate_ref": artifact_ref(stable_candidate) if stable_candidate else None,
                },
                "old_text": trim(old_region.text, text_chars),
                "candidate_text": trim(candidate.text, text_chars),
            }
        )
    return cards


def artifact_ref(region: Region) -> str:
    """Return the benchmark artifact ref namespace for copied docs trees."""

    if region.ref.startswith("docs/"):
        return region.ref[len("docs/") :]
    return region.ref


def stable_candidate_for(
    old_region: Region,
    stable: str,
    same_candidate: Region | None,
    new_by_ref: dict[str, Region],
    fuzzy_index: FuzzyRegionIndex,
) -> Region | None:
    if stable == "same_file_exact" and same_candidate:
        return same_candidate
    if stable == "moved_exact":
        matches = [region for region in new_by_ref.values() if region.fingerprint == old_region.fingerprint and region.path != old_region.path]
        return matches[0] if matches else None
    if stable == "fuzzy":
        best = fuzzy_index.best_match(old_region)
        return best[0] if best else None
    return None


def quote_selector_decision(old_region: Region, same_candidate: Region | None, quote_hits: list[Region], valid: bool) -> str:
    if same_candidate and same_candidate.fingerprint == old_region.fingerprint:
        return "preserved"
    if len(quote_hits) == 1:
        return "preserved" if valid else "silent_wrong"
    return "false_stale_alert" if valid else "true_stale_alert"


def layered_decision_for(old_region: Region, same_candidate: Region | None, quote_hits: list[Region], valid: bool, threshold: float = 0.82) -> str:
    if same_candidate and same_candidate.fingerprint == old_region.fingerprint:
        return "preserved"
    if len(quote_hits) == 1:
        similarity = token_jaccard(old_region.text, quote_hits[0].text)
        if similarity >= threshold:
            return "preserved" if valid else "silent_wrong"
        return "review_needed" if valid else "true_stale_alert"
    return "review_needed" if valid else "true_stale_alert"


def judge_cards(
    cards: list[dict[str, Any]],
    *,
    output: Path,
    models: list[str],
    endpoint: str,
    api_key: str,
    limit: int,
    parallel: int,
    timeout: int,
    text_chars: int,
    max_output_tokens: int,
    preflight: bool,
) -> dict[str, Any]:
    if not api_key:
        raise RuntimeError("OpenRouter API key is not set.")
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = {(row["card_id"], row["model"]) for row in read_jsonl(output) if output.exists() and row.get("ok")}
    active_models = preflight_models(models, endpoint=endpoint, api_key=api_key, timeout=timeout, max_output_tokens=max_output_tokens) if preflight else models
    work: list[tuple[dict[str, Any], str]] = []
    for card in cards[:limit]:
        for model in active_models:
            if (card["card_id"], model) not in existing:
                work.append((card, model))
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        futures = [
            pool.submit(
                judge_one,
                card,
                model,
                endpoint=endpoint,
                api_key=api_key,
                timeout=timeout,
                text_chars=text_chars,
                max_output_tokens=max_output_tokens,
            )
            for card, model in work
        ]
        for future in as_completed(futures):
            row = future.result()
            append_jsonl(output, row)
            results.append(row)
    return {
        "cards_requested": min(limit, len(cards)),
        "models_requested": models,
        "models_used": active_models,
        "judgments_written": len(results),
        "ok": sum(1 for row in results if row.get("ok")),
        "failed": sum(1 for row in results if not row.get("ok")),
        "output": str(output),
    }


def preflight_models(models: list[str], *, endpoint: str, api_key: str, timeout: int, max_output_tokens: int) -> list[str]:
    active: list[str] = []
    probe_card = {
        "card_id": "preflight",
        "categories": ["preflight"],
        "old_text": "The refund period is thirty days.",
        "candidate_text": "Refunds are available within thirty days.",
        "method_decisions": {},
        "signals": {},
    }
    for model in models:
        row = judge_one(probe_card, model, endpoint=endpoint, api_key=api_key, timeout=timeout, text_chars=400, max_output_tokens=max_output_tokens)
        if row.get("ok"):
            active.append(model)
        else:
            print(f"[WARN] preflight failed for {model}: {row.get('error')}")
    return active


def judge_one(
    card: dict[str, Any],
    model: str,
    *,
    endpoint: str,
    api_key: str,
    timeout: int,
    text_chars: int,
    max_output_tokens: int,
) -> dict[str, Any]:
    prompt = judge_prompt(card, text_chars=text_chars)
    started = time.time()
    try:
        response = openrouter_chat(endpoint, api_key, model, prompt, timeout=timeout, max_output_tokens=max_output_tokens)
        parsed = parse_json_response(response)
        verdict = str(parsed.get("verdict", "")).strip()
        if verdict not in VALID_VERDICTS:
            raise ValueError(f"invalid verdict {verdict!r}")
        return {
            "schema": "refmark.lifecycle_llm_judgment.v1",
            "card_id": card["card_id"],
            "model": model,
            "ok": True,
            "latency_seconds": round(time.time() - started, 3),
            "verdict": verdict,
            "confidence": parsed.get("confidence"),
            "rationale": parsed.get("rationale", ""),
            "candidate_satisfies_old_obligation": parsed.get("candidate_satisfies_old_obligation"),
            "raw": response,
        }
    except Exception as exc:
        return {
            "schema": "refmark.lifecycle_llm_judgment.v1",
            "card_id": card["card_id"],
            "model": model,
            "ok": False,
            "latency_seconds": round(time.time() - started, 3),
            "error": str(exc),
        }


def judge_prompt(card: dict[str, Any], *, text_chars: int) -> str:
    old_text = trim(str(card.get("old_text", "")), text_chars)
    candidate_text = trim(str(card.get("candidate_text", "")), text_chars)
    return f"""You are reviewing corpus evidence lifecycle labels.

Task: decide whether the NEW CANDIDATE TEXT still satisfies the same evidence obligation as the OLD TEXT.

Important:
- Do not overthink. Prefer the smallest JSON answer that fits the evidence.
- Judge evidence equivalence, not writing style.
- Valid rewritten/moved evidence is acceptable if the same claim/obligation remains supported.
- If the new text is related but no longer supports the same obligation, mark stale.
- If only part of the support remains and other regions are needed, mark split_support.
- If uncertain between valid and stale, mark ambiguous.

Return JSON only with:
{{
  "verdict": "valid_unchanged|valid_moved|valid_rewritten|split_support|stale|deleted|ambiguous|alternative_valid|invalid_original_label",
  "candidate_satisfies_old_obligation": true|false|null,
  "confidence": 0.0-1.0,
  "rationale": "short reason"
}}

Metadata:
categories: {card.get("categories")}
stable_status: {card.get("stable_status")}
method_decisions: {json.dumps(card.get("method_decisions", {}), ensure_ascii=False)}
signals: {json.dumps(card.get("signals", {}), ensure_ascii=False)}

OLD TEXT:
{old_text}

NEW CANDIDATE TEXT:
{candidate_text}
"""


def openrouter_chat(endpoint: str, api_key: str, model: str, prompt: str, *, timeout: int, max_output_tokens: int) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_output_tokens,
        }
    ).encode("utf-8")
    req = request.Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/b-imenitov/refmark",
            "X-Title": "Refmark lifecycle review",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:500]}") from exc
    message = payload["choices"][0]["message"]
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
    if not content:
        raise RuntimeError(f"empty model content: {json.dumps(payload.get('choices', [{}])[0], ensure_ascii=False)[:500]}")
    return str(content)


def parse_json_response(value: str) -> dict[str, Any]:
    value = value.strip()
    if value.startswith("```"):
        value = value.strip("`")
        if value.startswith("json"):
            value = value[4:].strip()
    return json.loads(value)


def summarize(cards: list[dict[str, Any]], judgments: list[dict[str, Any]]) -> dict[str, Any]:
    cards_by_id = {card["card_id"]: card for card in cards}
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for row in judgments:
        card_id = row.get("card_id")
        model = row.get("model")
        if card_id not in cards_by_id or not model:
            continue
        key = (str(card_id), str(model))
        if key not in by_pair or (row.get("ok") and not by_pair[key].get("ok")):
            by_pair[key] = row
    relevant = list(by_pair.values())
    ok = [row for row in relevant if row.get("ok")]
    by_card: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ok:
        by_card[row["card_id"]].append(row)
    verdict_counts = Counter(row["verdict"] for row in ok)
    category_verdicts: dict[str, Counter] = defaultdict(Counter)
    disagreements: list[dict[str, Any]] = []
    method_review: dict[str, Counter] = defaultdict(Counter)
    for card_id, rows in by_card.items():
        card = cards_by_id.get(card_id)
        if not card:
            continue
        majority = Counter(row["verdict"] for row in rows).most_common(1)[0][0]
        candidate_valid = majority in {"valid_unchanged", "valid_moved", "valid_rewritten", "split_support", "alternative_valid"}
        for category in card["categories"]:
            category_verdicts[category][majority] += 1
        for method, decision in card.get("method_decisions", {}).items():
            if decision == "preserved" and not candidate_valid:
                method_review[method]["judge_says_silent_drift"] += 1
            elif decision == "preserved" and candidate_valid:
                method_review[method]["judge_says_safe_preserved"] += 1
            elif decision in {"review_needed", "false_stale_alert"} and candidate_valid:
                method_review[method]["judge_says_conservative_review"] += 1
            elif decision in {"true_stale_alert", "review_needed", "false_stale_alert"} and not candidate_valid:
                method_review[method]["judge_says_correct_not_preserved"] += 1
        if len({row["verdict"] for row in rows}) > 1:
            disagreements.append({"card_id": card_id, "verdicts": Counter(row["verdict"] for row in rows), "categories": card["categories"]})
    return {
        "cards": len(cards),
        "judgments": len(relevant),
        "ok_judgments": len(ok),
        "failed_judgments": len(relevant) - len(ok),
        "reviewed_cards": len(by_card),
        "models": sorted(set(row.get("model") for row in relevant)),
        "verdict_counts": dict(verdict_counts),
        "category_majority_verdicts": {category: dict(counter) for category, counter in sorted(category_verdicts.items())},
        "method_implications_on_reviewed_cards": {method: dict(counter) for method, counter in sorted(method_review.items())},
        "model_disagreements": disagreements[:30],
        "model_disagreement_count": len(disagreements),
    }


def render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Lifecycle LLM Review Summary",
        "",
        f"- cards: {summary['cards']}",
        f"- reviewed cards: {summary['reviewed_cards']}",
        f"- judgments: {summary['ok_judgments']} ok / {summary['failed_judgments']} failed",
        f"- models: {', '.join(str(model) for model in summary['models'])}",
        f"- model disagreement cards: {summary['model_disagreement_count']}",
        "",
        "## Verdicts",
        "",
        "| verdict | count |",
        "| --- | ---: |",
    ]
    for verdict, count in sorted(summary["verdict_counts"].items()):
        lines.append(f"| {verdict} | {count} |")
    lines.extend(["", "## Category Majority Verdicts", "", "| category | verdict counts |", "| --- | --- |"])
    for category, counts in summary["category_majority_verdicts"].items():
        lines.append(f"| {category} | {json.dumps(counts, ensure_ascii=False)} |")
    lines.extend(["", "## Method Implications On Reviewed Cards", "", "| method | implications |", "| --- | --- |"])
    for method, counts in summary["method_implications_on_reviewed_cards"].items():
        lines.append(f"| {method} | {json.dumps(counts, ensure_ascii=False)} |")
    return "\n".join(lines) + "\n"


def human_review_rows(
    cards: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    *,
    limit: int,
    text_chars: int,
) -> list[dict[str, Any]]:
    cards_by_id = {card["card_id"]: card for card in cards}
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for row in judgments:
        card_id = row.get("card_id")
        model = row.get("model")
        if card_id not in cards_by_id or not model:
            continue
        key = (str(card_id), str(model))
        if key not in by_pair or (row.get("ok") and not by_pair[key].get("ok")):
            by_pair[key] = row
    by_card: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in by_pair.values():
        if row.get("ok"):
            by_card[row["card_id"]].append(row)

    rows: list[dict[str, Any]] = []
    for card_id, model_rows in by_card.items():
        card = cards_by_id[card_id]
        verdict_counts = Counter(row["verdict"] for row in model_rows)
        majority, majority_votes = verdict_counts.most_common(1)[0]
        vote_total = len(model_rows)
        vote_detail = "; ".join(
            f"{row['model']}={row['verdict']}({row.get('confidence')})" for row in sorted(model_rows, key=lambda item: item["model"])
        )
        rationale_detail = "\n".join(
            f"{row['model']}: {row.get('rationale', '')}" for row in sorted(model_rows, key=lambda item: item["model"])
        )
        score, reasons = review_priority(card, majority, majority_votes, vote_total, verdict_counts)
        rows.append(
            {
                "priority": score,
                "priority_reasons": "; ".join(reasons),
                "card_id": card_id,
                "human_verdict": "",
                "human_confidence": "",
                "human_notes": "",
                "llm_majority": majority,
                "llm_votes": f"{majority_votes}/{vote_total}",
                "llm_vote_detail": vote_detail,
                "categories": ", ".join(card.get("categories", [])),
                "corpus": Path(str(card.get("artifact", ""))).stem.replace("layered2_full_", ""),
                "old_ref": card.get("old_ref", ""),
                "candidate_ref": card.get("candidate_ref", ""),
                "old_path": card.get("old_path", ""),
                "candidate_path": card.get("candidate_path", ""),
                "stable_status": card.get("stable_status", ""),
                "method_decisions": json.dumps(card.get("method_decisions", {}), ensure_ascii=False),
                "signals": json.dumps(card.get("signals", {}), ensure_ascii=False),
                "llm_rationales": rationale_detail,
                "old_text": trim(str(card.get("old_text", "")), text_chars),
                "candidate_text": trim(str(card.get("candidate_text", "")), text_chars),
            }
        )
    rows.sort(key=lambda row: (-int(row["priority"]), str(row["corpus"]), str(row["card_id"])))
    return rows[:limit]


def review_priority(
    card: dict[str, Any],
    majority: str,
    majority_votes: int,
    vote_total: int,
    verdict_counts: Counter,
) -> tuple[int, list[str]]:
    categories = set(card.get("categories", []))
    score = 0
    reasons: list[str] = []
    if "quote_selector_silent_drift" in categories:
        score += 10
        reasons.append("quote-selector silent-drift candidate")
    if len(verdict_counts) > 1:
        score += 5
        reasons.append("LLM disagreement")
    if majority in {"stale", "deleted", "ambiguous"}:
        score += 4
        reasons.append(f"majority {majority}")
    if "layered_review_valid" in categories and majority in {
        "valid_unchanged",
        "valid_moved",
        "valid_rewritten",
        "split_support",
        "alternative_valid",
    }:
        score += 3
        reasons.append("Refmark review likely preservable")
    if "quote_preserved_layered_review" in categories:
        score += 2
        reasons.append("quote selector vs layered selector")
    if vote_total >= 3 and majority_votes == 1:
        score += 2
        reasons.append("no majority among three judges")
    return score, reasons


def write_worksheet_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "priority",
        "priority_reasons",
        "card_id",
        "human_verdict",
        "human_confidence",
        "human_notes",
        "llm_majority",
        "llm_votes",
        "llm_vote_detail",
        "categories",
        "corpus",
        "old_ref",
        "candidate_ref",
        "old_path",
        "candidate_path",
        "stable_status",
        "method_decisions",
        "signals",
        "llm_rationales",
        "old_text",
        "candidate_text",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_worksheet_html(rows: list[dict[str, Any]]) -> str:
    rendered_rows = []
    verdict_options = [
        "",
        "valid_unchanged",
        "valid_moved",
        "valid_rewritten",
        "split_support",
        "stale",
        "deleted",
        "ambiguous",
        "alternative_valid",
        "invalid_original_label",
    ]
    options_html = "".join(f"<option>{html.escape(option)}</option>" for option in verdict_options)
    for index, row in enumerate(rows, start=1):
        rendered_rows.append(
            f"""
<section class="card">
  <div class="meta">
    <h2>{index}. {html.escape(row['card_id'])}</h2>
    <p><b>Priority:</b> {html.escape(str(row['priority']))} - {html.escape(row['priority_reasons'])}</p>
    <p><b>LLM:</b> {html.escape(row['llm_majority'])} ({html.escape(row['llm_votes'])}) - {html.escape(row['llm_vote_detail'])}</p>
    <p><b>Corpus:</b> {html.escape(row['corpus'])}</p>
    <p><b>Old:</b> {html.escape(row['old_ref'])}<br><b>Candidate:</b> {html.escape(row['candidate_ref'])}</p>
    <p><b>Categories:</b> {html.escape(row['categories'])}</p>
    <details><summary>Signals and method decisions</summary><pre>{html.escape(row['method_decisions'])}
{html.escape(row['signals'])}</pre></details>
    <details><summary>LLM rationales</summary><pre>{html.escape(row['llm_rationales'])}</pre></details>
  </div>
  <div class="human">
    <label>Human verdict <select>{options_html}</select></label>
    <label>Confidence <input type="text" placeholder="0.0-1.0"></label>
    <label>Notes <textarea rows="4"></textarea></label>
  </div>
  <h3>Side-by-side Evidence Diff</h3>
  {render_side_by_side_diff(str(row['old_text']), str(row['candidate_text']))}
  <details>
    <summary>Raw side-by-side text</summary>
  <div class="grid">
    <div><h3>Old Text</h3><pre>{html.escape(row['old_text'])}</pre></div>
    <div><h3>Candidate Text</h3><pre>{html.escape(row['candidate_text'])}</pre></div>
  </div>
  </details>
</section>
"""
        )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Refmark Human Review Worksheet</title>
<style>{review_css()}</style></head><body>
<h1>Refmark Human Review Worksheet</h1>
<p>Use the CSV for durable labels. This HTML is for inspection and note-taking.</p>
{''.join(rendered_rows)}
</body></html>"""


def render_filled_worksheet_html(rows: list[dict[str, Any]]) -> str:
    rendered_rows = []
    for index, row in enumerate(rows, start=1):
        rendered_rows.append(
            f"""
<section class="card {html.escape(str(row.get('human_verdict', '')))}">
  <h2>{index}. {html.escape(row['card_id'])} - {html.escape(row.get('human_verdict', ''))} ({html.escape(row.get('human_confidence', ''))})</h2>
  <p><b>Notes:</b> {html.escape(row.get('human_notes', ''))}</p>
  <p><b>LLM:</b> {html.escape(row.get('llm_majority', ''))} {html.escape(row.get('llm_votes', ''))} - {html.escape(row.get('llm_vote_detail', ''))}</p>
  <p><b>Categories:</b> {html.escape(row.get('categories', ''))}</p>
  <p><b>Old:</b> {html.escape(row.get('old_ref', ''))}<br><b>Candidate:</b> {html.escape(row.get('candidate_ref', ''))}</p>
  <details><summary>Signals / decisions</summary><pre>{html.escape(row.get('method_decisions', ''))}
{html.escape(row.get('signals', ''))}</pre></details>
  <details><summary>LLM rationales</summary><pre>{html.escape(row.get('llm_rationales', ''))}</pre></details>
  <h3>Side-by-side Evidence Diff</h3>
  {render_side_by_side_diff(str(row.get('old_text', '')), str(row.get('candidate_text', '')))}
  <details>
    <summary>Raw side-by-side text</summary>
    <div class="grid">
      <div><h3>Old Text</h3><pre>{html.escape(row.get('old_text', ''))}</pre></div>
      <div><h3>Candidate Text</h3><pre>{html.escape(row.get('candidate_text', ''))}</pre></div>
    </div>
  </details>
</section>
"""
        )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Codex-filled Lifecycle Review</title>
<style>{review_css()}</style></head><body>
<h1>Codex-filled Lifecycle Review</h1>
<p>This is an adjudication aid, not independent human review. The primary view is side-by-side cleaned evidence text with changed spans highlighted; raw text is collapsed below each case.</p>
{''.join(rendered_rows)}
</body></html>"""


def review_css() -> str:
    return """
body { font-family: system-ui, sans-serif; margin: 24px; color: #17202a; background: #f6f8fa; }
.card { background: white; border: 1px solid #ccd3dc; padding: 16px; margin: 18px 0; box-shadow: 0 1px 3px rgba(0,0,0,.05); }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.human { display: grid; grid-template-columns: 220px 160px 1fr; gap: 12px; margin: 12px 0; align-items: start; }
label { display: grid; gap: 4px; font-weight: 600; }
select, input, textarea { font: inherit; border: 1px solid #b7c0cc; padding: 6px; }
pre { white-space: pre-wrap; background: #f6f8fa; padding: 12px; border: 1px solid #d8dee4; }
h1, h2, h3 { margin-top: 0; }
.diff-pair { display: grid; grid-template-columns: 1fr; gap: 14px; align-items: start; }
.diff-pane { border: 1px solid #d8dee4; background: #fff; min-width: 0; }
.diff-pane h4 { margin: 0; padding: 8px 10px; border-bottom: 1px solid #d8dee4; background: #f6f8fa; }
.diff-table { display: grid; grid-template-columns: 1fr; }
.diff-row { display: grid; grid-template-columns: 1fr; border-top: 1px solid #eef1f4; }
.diff-cell { min-height: 22px; padding: 0 12px; overflow-wrap: break-word; }
.diff-cell.old.changed { background: #fff1f1; }
.diff-cell.new.changed { background: #ecfdf0; }
.md-line { margin: 8px 0; line-height: 1.45; font-size: 14px; }
.md-blank { min-height: 12px; }
.md-heading { margin: 14px 0 8px; font-weight: 700; font-size: 15px; }
.md-tab-heading { margin: 12px 0 8px; font-weight: 700; color: #3d4652; }
.md-admonition { margin: 10px 0; padding: 8px 10px; border-left: 4px solid #6e7781; background: #f6f8fa; }
.md-list { margin: 4px 0 4px 18px; }
.md-code-line, .md-fence { margin: 0; padding: 2px 8px; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 13px; background: #f6f8fa; white-space: pre-wrap; }
.md-fence { color: #57606a; border-top: 1px solid #d8dee4; }
.code-ish { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; background: #eef1f4; padding: 1px 3px; border-radius: 3px; }
.diff-old { background: #ffd7d5; color: #82071e; text-decoration: line-through; }
.diff-new { background: #aceebb; color: #116329; }
.stale { border-left: 6px solid #d73a49; }
.ambiguous, .split_support { border-left: 6px solid #d29922; }
.valid_rewritten, .valid_unchanged, .valid_moved { border-left: 6px solid #2da44e; }
@media (min-width: 1200px) { .diff-pair { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); } }
"""


def render_side_by_side_diff(old: str, new: str) -> str:
    old_clean = clean_review_text(old)
    new_clean = clean_review_text(new)
    rows = aligned_diff_rows(old_clean.splitlines(), new_clean.splitlines())
    old_rows = render_markdown_review_rows(rows, side="old")
    new_rows = render_markdown_review_rows(rows, side="new")
    return f"""
<div class="diff-pair">
  <div class="diff-pane">
    <h4>Old evidence</h4>
    <div class="diff-table">{old_rows}</div>
  </div>
  <div class="diff-pane">
    <h4>Candidate evidence</h4>
    <div class="diff-table">{new_rows}</div>
  </div>
</div>
"""


def clean_review_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r'<a\s+href="[^"]*"[^>]*>(.*?)</a>', r"\1", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+(```[A-Za-z0-9_+-]*)", r"\n\n\1", text)
    text = re.sub(r"(```)\s+(===|####|!!!|\* |- |\d+\. )", r"\1\n\n\2", text)
    text = re.sub(r"\s+(===\s+\"[^\"]+\")", r"\n\1", text)
    text = re.sub(r"\s+(!!!\s+\w+)", r"\n\n\1", text)
    text = re.sub(r"\s+(#{2,6}\s+)", r"\n\n\1", text)
    text = re.sub(r"([.!?])\s+(?=[A-Z][A-Za-z])", r"\1\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def aligned_diff_rows(old_lines: list[str], new_lines: list[str]) -> list[dict[str, object]]:
    matcher = SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    rows: list[dict[str, object]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        old_part = old_lines[i1:i2]
        new_part = new_lines[j1:j2]
        row_count = max(len(old_part), len(new_part))
        for offset in range(row_count):
            rows.append(
                {
                    "old": old_part[offset] if offset < len(old_part) else "",
                    "new": new_part[offset] if offset < len(new_part) else "",
                    "old_changed": tag in {"delete", "replace"} and offset < len(old_part),
                    "new_changed": tag in {"insert", "replace"} and offset < len(new_part),
                }
            )
    return rows


def render_markdown_review_rows(rows: list[dict[str, object]], *, side: str) -> str:
    in_code = False
    rendered: list[str] = []
    changed_key = f"{side}_changed"
    for row in rows:
        line = str(row[side])
        changed = bool(row[changed_key])
        line_html, in_code = render_review_markdown_line(line, in_code=in_code, changed=changed)
        classes = [side]
        if changed:
            classes.append("changed")
        rendered.append(f'<div class="diff-row"><div class="diff-cell {" ".join(classes)}">{line_html}</div></div>')
    return "".join(rendered)


def render_review_markdown_line(line: str, *, in_code: bool, changed: bool) -> tuple[str, bool]:
    stripped = line.strip()
    change_class = "diff-new" if changed else ""
    if not stripped:
        return '<div class="md-blank"></div>', in_code
    if stripped.startswith("```"):
        next_in_code = not in_code
        return f'<div class="md-fence">{html.escape(stripped)}</div>', next_in_code
    if in_code:
        return f'<div class="md-code-line">{html.escape(line)}</div>', in_code
    if stripped.startswith("!!!"):
        return f'<div class="md-admonition">{render_inline_markdown(stripped)}</div>', in_code
    heading = re.match(r"^(#{2,6})\s+(.*)$", stripped)
    if heading:
        return f'<div class="md-heading">{render_inline_markdown(heading.group(2))}</div>', in_code
    tab_heading = re.match(r'^===\s+"?([^"]+)"?$', stripped)
    if tab_heading:
        return f'<div class="md-tab-heading">{render_inline_markdown(tab_heading.group(1))}</div>', in_code
    if stripped.startswith(("* ", "- ")):
        return f'<div class="md-list">{render_inline_markdown(stripped[2:])}</div>', in_code
    numbered = re.match(r"^\d+\.\s+(.*)$", stripped)
    if numbered:
        return f'<div class="md-list">{render_inline_markdown(numbered.group(1))}</div>', in_code
    if change_class:
        return f'<div class="md-line"><span class="{change_class}">{render_inline_markdown(stripped)}</span></div>', in_code
    return f'<div class="md-line">{render_inline_markdown(stripped)}</div>', in_code


def render_inline_markdown(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`\n]+)`", r'<span class="code-ish">\1</span>', escaped)
    escaped = re.sub(r"\*\*([^*\n]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"\*([^*\n]+)\*", r"<em>\1</em>", escaped)
    return escaped


def render_text_diff(old: str, new: str) -> str:
    old_tokens = diff_tokens(old)
    new_tokens = diff_tokens(new)
    matcher = SequenceMatcher(a=old_tokens, b=new_tokens, autojunk=False)
    chunks: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        old_part = "".join(old_tokens[i1:i2])
        new_part = "".join(new_tokens[j1:j2])
        if tag == "equal":
            chunks.append(f'<span class="eq">{html.escape(old_part)}</span>')
        elif tag == "delete":
            chunks.append(f'<span class="del">{html.escape(old_part)}</span>')
        elif tag == "insert":
            chunks.append(f'<span class="ins">{html.escape(new_part)}</span>')
        elif tag == "replace":
            chunks.append(f'<span class="rep-old">{html.escape(old_part)}</span><span class="rep-new">{html.escape(new_part)}</span>')
    return "".join(chunks)


def diff_tokens(text: str) -> list[str]:
    return re.findall(r"\s+|[^\s]+", text)


def calibrate_filled_review(rows: list[dict[str, Any]]) -> dict[str, Any]:
    judged = [row for row in rows if row.get("human_verdict")]
    verdict_counts = Counter(row.get("human_verdict", "") for row in judged)
    category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    threshold_rows: list[dict[str, Any]] = []
    for row in judged:
        verdict = row.get("human_verdict", "")
        for category in split_list_field(row.get("categories", "")):
            category_counts[category][verdict] += 1
        status_counts[row.get("stable_status", "")][verdict] += 1

    for threshold in (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95):
        eligible = [
            row
            for row in judged
            if numeric_signal(row, "candidate_similarity") >= threshold
            or numeric_signal(row, "same_path_ordinal_similarity") >= threshold
        ]
        threshold_rows.append(rule_metrics(f"similarity >= {threshold:.2f}", eligible))

    refmark_review = [
        row
        for row in judged
        if any(
            category in split_list_field(row.get("categories", ""))
            for category in ("refmark_fuzzy_review", "layered_review_valid", "quote_preserved_layered_review")
        )
    ]
    quote_drift = [
        row
        for row in judged
        if "quote_selector_silent_drift" in split_list_field(row.get("categories", ""))
    ]
    high_conf_valid_review = [
        row
        for row in refmark_review
        if is_valid_preservation(row.get("human_verdict", ""))
        and float_or_zero(row.get("human_confidence", "")) >= 0.75
    ]
    high_conf_bad_quote = [
        row
        for row in quote_drift
        if row.get("human_verdict") in {"stale", "deleted", "split_support", "ambiguous"}
        and float_or_zero(row.get("human_confidence", "")) >= 0.70
    ]

    return {
        "rows": len(rows),
        "judged_rows": len(judged),
        "verdict_counts": dict(verdict_counts),
        "category_counts": {key: dict(value) for key, value in sorted(category_counts.items())},
        "stable_status_counts": {key: dict(value) for key, value in sorted(status_counts.items())},
        "similarity_thresholds": threshold_rows,
        "refmark_review": rule_metrics("Refmark/layered review queue", refmark_review),
        "quote_selector_silent_drift": rule_metrics("Quote-selector silent-drift candidates", quote_drift),
        "candidate_rule_signals": {
            "high_conf_valid_refmark_review_cards": len(high_conf_valid_review),
            "high_conf_bad_quote_selector_cards": len(high_conf_bad_quote),
            "split_support_cards": verdict_counts.get("split_support", 0),
            "ambiguous_cards": verdict_counts.get("ambiguous", 0),
        },
        "candidate_rules": [
            {
                "name": "promote_review_valid_exact_or_high_similarity",
                "intent": "Move high-confidence Refmark/layered review cases into automatic preservation when same-path/fuzzy similarity is high and no split/ambiguity signal is present.",
                "guardrails": [
                    "candidate verdict class must be valid_unchanged, valid_moved, valid_rewritten, or alternative_valid in calibration",
                    "do not promote split_support, stale, deleted, ambiguous, or invalid_original_label",
                    "cache the rule decision with corpus/eval fingerprints",
                ],
            },
            {
                "name": "quote_selector_needs_refmark_crosscheck",
                "intent": "Treat quote-selector preservation as unsafe when Refmark/layered evidence marks the same case stale/review and similarity is low or support appears split.",
                "guardrails": [
                    "quote hit alone is not enough",
                    "low similarity quote matches stay in review",
                    "split_support should become an explicit multi-ref/range lifecycle state",
                ],
            },
            {
                "name": "split_support_as_range_migration",
                "intent": "Do not count split-support cases as ordinary stale; route them to range/multi-ref repair.",
                "guardrails": [
                    "requires follow-up range candidate generation",
                    "requires reviewer or deterministic acceptance before preserving",
                ],
            },
        ],
    }


def rule_metrics(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    verdicts = Counter(row.get("human_verdict", "") for row in rows)
    safe = sum(verdicts.get(verdict, 0) for verdict in safe_preservation_verdicts())
    unsafe = sum(verdicts.get(verdict, 0) for verdict in unsafe_preservation_verdicts())
    review = sum(verdicts.get(verdict, 0) for verdict in review_needed_verdicts())
    total = len(rows)
    return {
        "name": name,
        "rows": total,
        "safe_preservation": safe,
        "unsafe_if_auto_preserved": unsafe,
        "review_needed": review,
        "safe_rate": round(safe / total, 4) if total else 0.0,
        "unsafe_rate": round(unsafe / total, 4) if total else 0.0,
        "verdict_counts": dict(verdicts),
    }


def render_calibration_markdown(calibration: dict[str, Any]) -> str:
    lines = [
        "# Lifecycle Review Calibration",
        "",
        "This report turns the filled review worksheet into rule-design evidence. "
        "It is an adjudication aid, not independent human review.",
        "",
        f"- worksheet rows: {calibration['rows']}",
        f"- judged rows: {calibration['judged_rows']}",
        "",
        "## Verdicts",
        "",
        "| verdict | rows |",
        "|---|---:|",
    ]
    for verdict, count in sorted(calibration["verdict_counts"].items()):
        lines.append(f"| {verdict} | {count} |")
    lines.extend(["", "## Category Outcomes", "", "| category | rows | safe | unsafe if auto-preserved | review needed | verdicts |", "|---|---:|---:|---:|---:|---|"])
    for category, counts in calibration["category_counts"].items():
        rows = sum(counts.values())
        safe = sum(counts.get(verdict, 0) for verdict in safe_preservation_verdicts())
        unsafe = sum(counts.get(verdict, 0) for verdict in unsafe_preservation_verdicts())
        review = sum(counts.get(verdict, 0) for verdict in review_needed_verdicts())
        verdicts = ", ".join(f"{key}:{value}" for key, value in sorted(counts.items()))
        lines.append(f"| {category} | {rows} | {safe} | {unsafe} | {review} | {verdicts} |")
    lines.extend(["", "## Similarity Threshold Probe", "", "| rule | rows | safe | unsafe | review | safe rate | unsafe rate |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in calibration["similarity_thresholds"]:
        lines.append(
            f"| {row['name']} | {row['rows']} | {row['safe_preservation']} | "
            f"{row['unsafe_if_auto_preserved']} | {row['review_needed']} | "
            f"{row['safe_rate']:.1%} | {row['unsafe_rate']:.1%} |"
        )
    lines.extend(["", "## Focus Sets", ""])
    for key in ("refmark_review", "quote_selector_silent_drift"):
        row = calibration[key]
        lines.extend(
            [
                f"### {row['name']}",
                "",
                f"- rows: {row['rows']}",
                f"- safe preservation: {row['safe_preservation']} ({row['safe_rate']:.1%})",
                f"- unsafe if auto-preserved: {row['unsafe_if_auto_preserved']} ({row['unsafe_rate']:.1%})",
                f"- review needed: {row['review_needed']}",
                f"- verdicts: {', '.join(f'{k}:{v}' for k, v in sorted(row['verdict_counts'].items()))}",
                "",
            ]
        )
    lines.extend(["## Candidate Rule Signals", ""])
    for key, value in calibration["candidate_rule_signals"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Candidate Rules", ""])
    for rule in calibration["candidate_rules"]:
        lines.append(f"### {rule['name']}")
        lines.append("")
        lines.append(rule["intent"])
        lines.append("")
        for guardrail in rule["guardrails"]:
            lines.append(f"- {guardrail}")
        lines.append("")
    return "\n".join(lines)


def split_list_field(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def numeric_signal(row: dict[str, Any], key: str) -> float:
    try:
        signals = json.loads(row.get("signals", "{}") or "{}")
    except json.JSONDecodeError:
        return 0.0
    return float_or_zero(signals.get(key, 0.0))


def float_or_zero(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def is_valid_preservation(verdict: str) -> bool:
    return verdict in safe_preservation_verdicts()


def safe_preservation_verdicts() -> set[str]:
    return {"valid_unchanged", "valid_moved", "valid_rewritten", "alternative_valid"}


def unsafe_preservation_verdicts() -> set[str]:
    return {"stale", "deleted", "invalid_original_label"}


def review_needed_verdicts() -> set[str]:
    return {"split_support", "ambiguous"}


def render_cards_html(cards: list[dict[str, Any]]) -> str:
    rows = []
    for card in cards:
        rows.append(
            f"""
<section class="card">
  <h2>{html.escape(card['card_id'])}</h2>
  <p><b>Categories:</b> {html.escape(', '.join(card['categories']))}</p>
  <p><b>Revision:</b> {html.escape(str(card['old_ref_version']))} -> {html.escape(str(card['new_ref_version']))}</p>
  <p><b>Old:</b> {html.escape(card['old_ref'])}<br><b>Candidate:</b> {html.escape(card['candidate_ref'])}</p>
  <p><b>Decisions:</b> {html.escape(json.dumps(card['method_decisions'], ensure_ascii=False))}</p>
  <p><b>Signals:</b> {html.escape(json.dumps(card['signals'], ensure_ascii=False))}</p>
  <div class="grid">
    <div><h3>Old Text</h3><pre>{html.escape(card['old_text'])}</pre></div>
    <div><h3>Candidate Text</h3><pre>{html.escape(card['candidate_text'])}</pre></div>
  </div>
</section>
"""
        )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Refmark Lifecycle Review Cards</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; color: #17202a; }}
.card {{ border: 1px solid #ccd3dc; padding: 16px; margin: 18px 0; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
pre {{ white-space: pre-wrap; background: #f6f8fa; padding: 12px; border: 1px solid #d8dee4; }}
</style></head><body>
<h1>Refmark Lifecycle Review Cards</h1>
{''.join(rows)}
</body></html>"""


def stable_card_id(*parts: object) -> str:
    return hashlib.sha1("||".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:16]


def trim(text: str, chars: int) -> str:
    text = text.strip()
    return text if len(text) <= chars else text[:chars] + "\n...[truncated]"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv_dicts(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
