"""LLM-judge retrieved evidence for cached Refmark eval queries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
from urllib import request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.portable_search_index.evaluate_real_corpus import (  # noqa: E402
    EvalQuestion,
    _read_question_cache,
)
from refmark.search_index import OPENROUTER_CHAT_URL, PortableBM25Index, load_search_index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Use an LLM judge on random Refmark search results.")
    parser.add_argument("index", help="Portable index JSON.")
    parser.add_argument("--question-cache", required=True)
    parser.add_argument("--output", default="examples/portable_search_index/output/llm_judge_report.json")
    parser.add_argument("--model", default="mistralai/mistral-nemo")
    parser.add_argument("--endpoint", default=OPENROUTER_CHAT_URL)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--expand-after", type=int, default=1)
    args = parser.parse_args()

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set.")
    index = load_search_index(args.index)
    region_refs = {region.stable_ref for region in index.regions}
    questions = [
        question
        for question in _read_question_cache(Path(args.question_cache)).values()
        if set(question.gold_refs).issubset(region_refs)
    ]
    rng = random.Random(args.seed)
    rng.shuffle(questions)
    questions = questions[: args.limit]
    rows = []
    for question in questions:
        hits = index.search(question.query, top_k=args.top_k, expand_after=args.expand_after)
        row = judge_one(
            index,
            question,
            hits,
            model=args.model,
            endpoint=args.endpoint,
            api_key=api_key,
        )
        rows.append(row)
        print(json.dumps(row, indent=2))
    report = {
        "index": args.index,
        "question_cache": args.question_cache,
        "model": args.model,
        "limit": len(rows),
        "summary": summarize(rows),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote LLM judge report to {output}")


def judge_one(
    index: PortableBM25Index,
    question: EvalQuestion,
    hits,
    *,
    model: str,
    endpoint: str,
    api_key: str,
) -> dict[str, object]:
    region_by_ref = {region.stable_ref: region for region in index.regions}
    gold_text = "\n\n".join(region_by_ref[ref].text for ref in question.gold_refs if ref in region_by_ref)
    retrieved_text = "\n\n".join(f"{hit.stable_ref}:\n{hit.text}" for hit in hits)
    prompt = f"""Judge whether the retrieved evidence answers the query compared to the gold evidence.

Return strict JSON:
{{
  "answer_supported": true/false,
  "evidence_covers_gold": 0.0-1.0,
  "retrieved_has_irrelevant_extra": true/false,
  "notes": "short"
}}

Query:
{question.query}

Gold evidence:
{gold_text}

Retrieved evidence:
{retrieved_text}
"""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict retrieval-evidence judge. Return only JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 220,
    }
    req = request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/b-imenitov/refmark",
            "X-Title": "refmark retrieval judge",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    content = payload["choices"][0]["message"]["content"]
    parsed = parse_json(content)
    return {
        "query": question.query,
        "gold_refs": question.gold_refs,
        "top_refs": [hit.stable_ref for hit in hits],
        "judge": parsed,
    }


def parse_json(text: str) -> dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").removeprefix("json").strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end < start:
            return {
                "answer_supported": False,
                "evidence_covers_gold": 0.0,
                "retrieved_has_irrelevant_extra": True,
                "invalid_json": True,
                "notes": "judge returned invalid JSON",
            }
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return {
                "answer_supported": False,
                "evidence_covers_gold": 0.0,
                "retrieved_has_irrelevant_extra": True,
                "invalid_json": True,
                "notes": "judge returned invalid JSON",
            }


def summarize(rows: list[dict[str, object]]) -> dict[str, float]:
    if not rows:
        return {"answer_supported_rate": 0.0, "avg_evidence_covers_gold": 0.0, "irrelevant_extra_rate": 0.0}
    supported = 0
    coverage = 0.0
    irrelevant = 0
    invalid = 0
    for row in rows:
        judge = row["judge"]
        supported += 1 if judge.get("answer_supported") else 0
        coverage += float(judge.get("evidence_covers_gold", 0.0))
        irrelevant += 1 if judge.get("retrieved_has_irrelevant_extra") else 0
        invalid += 1 if judge.get("invalid_json") or judge.get("notes") == "judge returned invalid JSON" else 0
    total = len(rows)
    return {
        "answer_supported_rate": round(supported / total, 4),
        "avg_evidence_covers_gold": round(coverage / total, 4),
        "irrelevant_extra_rate": round(irrelevant / total, 4),
        "invalid_json_rate": round(invalid / total, 4),
    }


if __name__ == "__main__":
    main()
