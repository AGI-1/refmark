"""Generate cached retrieval views for refmark anchors."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import random
import time
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.rag_retrieval_benchmark.common import (
    DEFAULT_DATA_DIR,
    OUTPUT_DIR,
    extractive_summary,
    keywords_for,
    load_jsonl,
    write_jsonl,
)


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate retrieval views for refmark anchors.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output", default=str(OUTPUT_DIR / "views.jsonl"))
    parser.add_argument("--source", choices=["local", "openrouter"], default="local")
    parser.add_argument("--limit", type=int, default=0, help="0 means all anchors.")
    parser.add_argument("--sample-mode", choices=["first", "even", "random"], default="first")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--questions-per-anchor", type=int, default=4)
    parser.add_argument("--keywords", type=int, default=10)
    parser.add_argument("--model", default="google/gemini-3.1-flash-lite-preview")
    parser.add_argument("--endpoint", default=OPENROUTER_URL)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=450)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    anchors = load_jsonl(data_dir / "anchors.jsonl")
    train = load_jsonl(data_dir / "train.jsonl")
    questions_by_ref = _questions_by_ref(train)
    anchors = _select_anchors(anchors, limit=args.limit, mode=args.sample_mode, seed=args.seed)

    output = Path(args.output)
    existing = _load_existing(output) if args.resume else {}
    row_by_ref = {ref: row for ref, row in existing.items() if ref in {str(anchor["refmark"]) for anchor in anchors}}
    pending = [anchor for anchor in anchors if str(anchor["refmark"]) not in row_by_ref]

    if args.source == "openrouter" and args.concurrency > 1 and pending:
        _generate_openrouter_concurrent(args, pending, anchors, row_by_ref, output)
    else:
        for idx, anchor in enumerate(anchors, start=1):
            ref = str(anchor["refmark"])
            if ref in row_by_ref:
                continue
            if args.source == "openrouter":
                row = _openrouter_view_with_retries(args, anchor)
            else:
                row = _local_view(
                    anchor,
                    questions_by_ref,
                    questions_per_anchor=args.questions_per_anchor,
                    keyword_limit=args.keywords,
                )
            row_by_ref[ref] = row
            _write_ordered(output, anchors, row_by_ref)
            if args.sleep:
                time.sleep(args.sleep)
            print(f"{idx}/{len(anchors)} {ref} {args.source}")

    _write_ordered(output, anchors, row_by_ref)
    print(f"Wrote {len(row_by_ref)} retrieval views to {output}")


def _load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {str(row["refmark"]): row for row in load_jsonl(path)}


def _questions_by_ref(train: list[dict]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for row in train:
        output.setdefault(str(row["refmark"]), []).append(str(row["question"]))
    return output


def _select_anchors(anchors: list[dict], *, limit: int, mode: str, seed: int) -> list[dict]:
    if limit <= 0 or limit >= len(anchors):
        return anchors
    if mode == "first":
        return anchors[:limit]
    if mode == "random":
        rng = random.Random(seed)
        indexes = sorted(rng.sample(range(len(anchors)), limit))
        return [anchors[index] for index in indexes]
    if limit == 1:
        return [anchors[0]]
    indexes = sorted({round(index * (len(anchors) - 1) / (limit - 1)) for index in range(limit)})
    return [anchors[index] for index in indexes[:limit]]


def _write_ordered(output: Path, anchors: list[dict], row_by_ref: dict[str, dict]) -> None:
    rows = [row_by_ref[str(anchor["refmark"])] for anchor in anchors if str(anchor["refmark"]) in row_by_ref]
    write_jsonl(output, rows)


def _generate_openrouter_concurrent(
    args,
    pending: list[dict],
    anchors: list[dict],
    row_by_ref: dict[str, dict],
    output: Path,
) -> None:
    refs_to_index = {str(anchor["refmark"]): idx for idx, anchor in enumerate(anchors, start=1)}
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(_openrouter_view_with_retries, args, anchor): anchor for anchor in pending}
        for future in as_completed(futures):
            anchor = futures[future]
            ref = str(anchor["refmark"])
            row_by_ref[ref] = future.result()
            _write_ordered(output, anchors, row_by_ref)
            if args.sleep:
                time.sleep(args.sleep)
            print(f"{refs_to_index[ref]}/{len(anchors)} {ref} {args.source}")


def _local_view(anchor: dict, questions_by_ref: dict[str, list[str]], *, questions_per_anchor: int, keyword_limit: int) -> dict:
    ref = str(anchor["refmark"])
    questions = questions_by_ref.get(ref, [])[:questions_per_anchor]
    text = str(anchor["text"])
    return {
        "refmark": ref,
        "source_hash": _stable_hash(text),
        "summary": extractive_summary(text),
        "questions": questions,
        "keywords": keywords_for(text, limit=keyword_limit),
        "generator": "local",
    }


def _openrouter_view(args, anchor: dict) -> dict:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set.")
    prompt = f"""
Create retrieval metadata for this anchored source region.

Refmark: {anchor['refmark']}
Text:
{anchor['text']}

Return JSON only:
{{
  "summary": "one short factual summary",
  "questions": ["{args.questions_per_anchor} realistic user questions answerable by this region"],
  "keywords": ["up to {args.keywords} useful retrieval keywords"]
}}

Rules:
- Do not mention refmark ids.
- Questions should distinguish this region from nearby generic documentation.
- Keep the summary factual and grounded in the text.
""".strip()
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "response_format": {"type": "json_object"},
    }
    request = Request(
        args.endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=90) as response:
        data = json.loads(response.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]
    view = json.loads(content)
    return {
        "refmark": str(anchor["refmark"]),
        "source_hash": _stable_hash(str(anchor["text"])),
        "summary": str(view.get("summary", "")),
        "questions": [str(item) for item in view.get("questions", [])][: args.questions_per_anchor],
        "keywords": [str(item) for item in view.get("keywords", [])][: args.keywords],
        "generator": args.model,
    }


def _openrouter_view_with_retries(args, anchor: dict) -> dict:
    delay = max(args.sleep, 0.25)
    for attempt in range(args.retries + 1):
        try:
            return _openrouter_view(args, anchor)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt >= args.retries:
                raise
            wait = delay * (2 ** attempt)
            print(f"retry {attempt + 1}/{args.retries} {anchor['refmark']}: {exc}; sleeping {wait:.2f}s")
            time.sleep(wait)
    raise AssertionError("unreachable")


def _stable_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


if __name__ == "__main__":
    main()
