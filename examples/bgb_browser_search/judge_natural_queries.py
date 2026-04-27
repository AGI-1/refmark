"""Judge natural BGB search concerns across retrieval methods.

This is intentionally not a gold-anchor benchmark. It asks a stronger model
whether the retrieved evidence would be useful for fresh user queries.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
from urllib import request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.portable_search_index.compare_navigation_search import (  # noqa: E402
    OpenRouterEmbeddingIndex,
    _bm25_hits,
    _hybrid,
)
from refmark.search_index import OPENROUTER_CHAT_URL, SearchRegion, load_search_index  # noqa: E402


DEFAULT_QUERIES = [
    ("de", "Mein Vermieter will wegen Eigenbedarf kündigen. Welche Regeln und Fristen gelten?"),
    ("de", "Ich habe online etwas gekauft und will widerrufen. Was muss ich beachten?"),
    ("de", "Eine gelieferte Software funktioniert nicht richtig und bekommt keine Updates. Welche Rechte habe ich?"),
    ("de", "Mein minderjähriges Kind hat ohne meine Zustimmung einen Vertrag abgeschlossen. Ist das wirksam?"),
    ("de", "Ich möchte eine Erbschaft ausschlagen oder anfechten. Wo steht die Frist?"),
    ("de", "Kann ich Schadensersatz verlangen, wenn jemand meine Sache beschädigt hat?"),
    ("en", "My landlord wants to terminate the lease for personal use. What BGB rules apply?"),
    ("en", "I ordered something online and want to revoke the contract. Which sections are relevant?"),
    ("en", "A digital product is defective and the provider does not deliver updates. What are my rights?"),
    ("en", "A minor signed a contract without parental approval. Is it valid under German law?"),
    ("en", "How can someone disclaim or challenge an inheritance, and what deadlines apply?"),
    ("en", "Someone damaged my property. Where does the BGB describe compensation or restoration?"),
]


@dataclass(frozen=True)
class MethodResult:
    method: str
    hits: list[dict[str, object]]


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-judge natural BGB queries across retrieval methods.")
    parser.add_argument("--raw-index", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_raw_index.json")
    parser.add_argument("--enriched-index", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_openrouter_index.json")
    parser.add_argument("--output", default="examples/bgb_browser_search/output_full_qwen_turbo/deepseek_natural_judge.json")
    parser.add_argument("--judge-model", default="deepseek/deepseek-v4-pro")
    parser.add_argument("--endpoint", default=OPENROUTER_CHAT_URL)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--embedding-model", default="qwen/qwen3-embedding-8b")
    parser.add_argument("--embedding-endpoint", default="https://openrouter.ai/api/v1/embeddings")
    parser.add_argument(
        "--embedding-cache",
        default="examples/bgb_browser_search/output_scratch_multi_full/embedding_cache_qwen3_8b.jsonl",
    )
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-hit-chars", type=int, default=1100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set.")

    raw = load_search_index(args.raw_index)
    enriched = load_search_index(args.enriched_index)
    raw_embed = OpenRouterEmbeddingIndex(
        raw.regions,
        model=args.embedding_model,
        endpoint=args.embedding_endpoint,
        api_key=api_key,
        cache_path=Path(args.embedding_cache),
        batch_size=32,
        max_chars=8000,
        include_view=False,
    )
    enriched_embed = OpenRouterEmbeddingIndex(
        enriched.regions,
        model=args.embedding_model,
        endpoint=args.embedding_endpoint,
        api_key=api_key,
        cache_path=Path(args.embedding_cache),
        batch_size=32,
        max_chars=8000,
        include_view=True,
    )

    selected_queries = DEFAULT_QUERIES[args.offset :]
    if args.limit is not None:
        selected_queries = selected_queries[: args.limit]

    rows = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for language, query in selected_queries:
        method_results = _retrieve_methods(
            query,
            raw=raw,
            enriched=enriched,
            raw_embed=raw_embed,
            enriched_embed=enriched_embed,
            candidate_k=args.candidate_k,
            top_k=args.top_k,
            max_hit_chars=args.max_hit_chars,
        )
        try:
            judge = _judge_query(
                query,
                language=language,
                method_results=method_results,
                model=args.judge_model,
                endpoint=args.endpoint,
                api_key=api_key,
            )
        except Exception as exc:
            judge = {"judge_error": type(exc).__name__, "message": str(exc)[:2000]}
        row = {
            "language": language,
            "query": query,
            "methods": [asdict(result) for result in method_results],
            "judge": judge,
        }
        rows.append(row)
        _write_report(output, args=args, rows=rows)
        print(json.dumps({"query": query, "judge": judge}, ensure_ascii=False, indent=2))

    _write_report(output, args=args, rows=rows)
    print(f"\nWrote natural-query judge report to {output}")


def _write_report(output: Path, *, args: argparse.Namespace, rows: list[dict[str, object]]) -> None:
    report = {
        "schema": "refmark.bgb_natural_judge.v1",
        "raw_index": args.raw_index,
        "enriched_index": args.enriched_index,
        "judge_model": args.judge_model,
        "embedding_model": args.embedding_model,
        "offset": args.offset,
        "queries": len(rows),
        "summary": _summarize(rows),
        "rows": rows,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _retrieve_methods(
    query: str,
    *,
    raw,
    enriched,
    raw_embed: OpenRouterEmbeddingIndex,
    enriched_embed: OpenRouterEmbeddingIndex,
    candidate_k: int,
    top_k: int,
    max_hit_chars: int,
) -> list[MethodResult]:
    region_by_ref = {region.stable_ref: region for region in enriched.regions}
    methods = {
        "raw_bm25": _bm25_hits(raw, query, candidate_k),
        "refmark_bm25": _bm25_hits(enriched, query, candidate_k),
        "raw_qwen3_embedding": raw_embed.search(query, top_k=candidate_k),
        "refmark_qwen3_embedding": enriched_embed.search(query, top_k=candidate_k),
    }
    methods["refmark_hybrid_w0.05"] = _hybrid(
        _bm25_hits(enriched, query, candidate_k),
        enriched_embed.search(query, top_k=candidate_k),
        first_weight=0.05,
    )
    return [
        MethodResult(
            method=name,
            hits=[
                _hit_payload(rank, hit.stable_ref, hit.score, region_by_ref, max_hit_chars=max_hit_chars)
                for rank, hit in enumerate(hits[:top_k], start=1)
            ],
        )
        for name, hits in methods.items()
    ]


def _hit_payload(
    rank: int,
    stable_ref: str,
    score: float,
    region_by_ref: dict[str, SearchRegion],
    *,
    max_hit_chars: int,
) -> dict[str, object]:
    region = region_by_ref[stable_ref]
    text = region.text
    if len(text) > max_hit_chars:
        text = text[:max_hit_chars].rstrip() + "..."
    return {
        "rank": rank,
        "stable_ref": stable_ref,
        "score": round(float(score), 6),
        "summary": region.view.summary,
        "text": text,
    }


def _judge_query(
    query: str,
    *,
    language: str,
    method_results: list[MethodResult],
    model: str,
    endpoint: str,
    api_key: str,
) -> dict[str, object]:
    payload = [
        {
            "method": result.method,
            "hits": result.hits,
        }
        for result in method_results
    ]
    prompt = f"""Judge retrieval evidence for a natural BGB/legal search query.

There is no gold answer. Decide whether each method returns a useful jump target for answering the user's concern.

Return strict JSON with this shape:
{{
  "method_scores": {{
    "method_name": {{
      "answer_supported": true,
      "usefulness": 0.0,
      "citation_quality": 0.0,
      "overbroad": false,
      "underbroad": false,
      "notes": "short"
    }}
  }},
  "best_method": "method_name",
  "best_refs": ["stable refs"],
  "overall_notes": "short"
}}

Scoring:
- usefulness: 1.0 means the top hits are enough to orient and likely answer the concern.
- citation_quality: 1.0 means the retrieved regions are specific, stable, and not mostly irrelevant.
- overbroad means many retrieved regions are generic or tangential.
- underbroad means the result misses an obviously needed neighboring/sibling rule.

Query language: {language}
Query: {query}

Retrieved method results:
{json.dumps(payload, ensure_ascii=False)}
"""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict retrieval quality judge. Return only JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 1400,
    }
    req = request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/b-imenitov/refmark",
            "X-Title": "refmark BGB natural retrieval judge",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=180) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
    if "choices" not in response_payload:
        return {"invalid_response": True, "raw_response": response_payload}
    content = response_payload["choices"][0]["message"]["content"]
    return _parse_json(content)


def _parse_json(text: str) -> dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").removeprefix("json").strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end < start:
            return {"invalid_json": True, "raw": text[:2000]}
        try:
            parsed = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return {"invalid_json": True, "raw": text[:2000]}
    return parsed if isinstance(parsed, dict) else {"invalid_json": True, "raw": text[:2000]}


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    totals: dict[str, dict[str, float]] = {}
    wins: dict[str, int] = {}
    invalid = 0
    for row in rows:
        judge = row["judge"]
        if judge.get("invalid_json") or judge.get("invalid_response") or judge.get("judge_error"):
            invalid += 1
            continue
        best = str(judge.get("best_method", ""))
        if best:
            wins[best] = wins.get(best, 0) + 1
        scores = judge.get("method_scores", {})
        if not isinstance(scores, dict):
            continue
        for method, score in scores.items():
            if not isinstance(score, dict):
                continue
            bucket = totals.setdefault(str(method), {"count": 0.0, "supported": 0.0, "usefulness": 0.0, "citation_quality": 0.0})
            bucket["count"] += 1.0
            bucket["supported"] += 1.0 if score.get("answer_supported") else 0.0
            bucket["usefulness"] += float(score.get("usefulness", 0.0))
            bucket["citation_quality"] += float(score.get("citation_quality", 0.0))
    methods = {}
    for method, bucket in totals.items():
        count = max(bucket["count"], 1.0)
        methods[method] = {
            "answer_supported_rate": round(bucket["supported"] / count, 4),
            "avg_usefulness": round(bucket["usefulness"] / count, 4),
            "avg_citation_quality": round(bucket["citation_quality"] / count, 4),
            "judged": int(bucket["count"]),
        }
    return {"invalid_json": invalid, "wins": wins, "methods": methods}


if __name__ == "__main__":
    main()
