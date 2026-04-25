"""Generate cached retrieval views for refmark anchors."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
import sys
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
    parser.add_argument("--questions-per-anchor", type=int, default=4)
    parser.add_argument("--keywords", type=int, default=10)
    parser.add_argument("--model", default="google/gemini-3.1-flash-lite-preview")
    parser.add_argument("--endpoint", default=OPENROUTER_URL)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=450)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    anchors = load_jsonl(data_dir / "anchors.jsonl")
    train = load_jsonl(data_dir / "train.jsonl")
    if args.limit > 0:
        anchors = anchors[: args.limit]

    output = Path(args.output)
    existing = _load_existing(output) if args.resume else {}
    rows: list[dict] = []
    for idx, anchor in enumerate(anchors, start=1):
        ref = str(anchor["refmark"])
        if ref in existing:
            rows.append(existing[ref])
            continue
        if args.source == "openrouter":
            row = _openrouter_view(args, anchor)
        else:
            row = _local_view(anchor, train, questions_per_anchor=args.questions_per_anchor, keyword_limit=args.keywords)
        rows.append(row)
        write_jsonl(output, rows)
        if args.sleep:
            time.sleep(args.sleep)
        print(f"{idx}/{len(anchors)} {ref} {args.source}")

    write_jsonl(output, rows)
    print(f"Wrote {len(rows)} retrieval views to {output}")


def _load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {str(row["refmark"]): row for row in load_jsonl(path)}


def _local_view(anchor: dict, train: list[dict], *, questions_per_anchor: int, keyword_limit: int) -> dict:
    ref = str(anchor["refmark"])
    questions = [str(row["question"]) for row in train if str(row["refmark"]) == ref][:questions_per_anchor]
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


def _stable_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


if __name__ == "__main__":
    main()
