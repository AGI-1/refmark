"""Evaluate coarse Refmark ranges such as articles or section windows."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from urllib import request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from refmark.metrics import score_ref_range, summarize_scores  # noqa: E402
from refmark.search_index import OPENROUTER_CHAT_URL, PortableBM25Index, SearchRegion, load_search_index, local_view  # noqa: E402


@dataclass(frozen=True)
class CoarseRange:
    range_id: str
    doc_id: str
    title: str
    stable_refs: list[str]
    text: str
    hash: str
    range_type: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RangeQuestion:
    query: str
    range_id: str
    doc_id: str
    gold_refs: list[str]
    hash: str
    source: str
    model: str
    range_type: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate coarse article/section ranges with Refmark anchors.")
    parser.add_argument("raw_index")
    parser.add_argument("enriched_index")
    parser.add_argument("--output", default="examples/portable_search_index/output/coarse_range_eval.json")
    parser.add_argument("--cache", default="examples/portable_search_index/output/coarse_range_questions.jsonl")
    parser.add_argument("--range-type", choices=["article", "window"], default="window")
    parser.add_argument("--window-size", type=int, default=6)
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--question-source", choices=["local", "openrouter"], default="openrouter")
    parser.add_argument("--model", default="mistralai/mistral-nemo")
    parser.add_argument("--endpoint", default=OPENROUTER_CHAT_URL)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--top-ks", default="1,3,5,10")
    parser.add_argument("--expand-after", type=int, default=2)
    args = parser.parse_args()

    raw = load_search_index(args.raw_index)
    enriched = load_search_index(args.enriched_index)
    ranges = build_ranges(
        enriched.regions,
        range_type=args.range_type,
        window_size=args.window_size,
    )
    ranges = sample_ranges(ranges, limit=args.sample_size, seed=args.seed)
    questions = load_or_generate_questions(
        ranges,
        cache_path=Path(args.cache),
        source=args.question_source,
        model=args.model,
        endpoint=args.endpoint,
        api_key_env=args.api_key_env,
        concurrency=args.concurrency,
    )
    top_ks = tuple(int(part) for part in args.top_ks.split(",") if part.strip())
    report = {
        "settings": vars(args),
        "ranges": len(ranges),
        "questions": len(questions),
        "range_stats": summarize_ranges(ranges),
        "reports": {
            "raw_bm25": evaluate(raw, questions, top_ks=top_ks, expand_after=args.expand_after),
            "refmark_bm25": evaluate(enriched, questions, top_ks=top_ks, expand_after=args.expand_after),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def build_ranges(regions: list[SearchRegion], *, range_type: str, window_size: int) -> list[CoarseRange]:
    by_doc: dict[str, list[SearchRegion]] = {}
    for region in regions:
        by_doc.setdefault(region.doc_id, []).append(region)
    for doc_regions in by_doc.values():
        doc_regions.sort(key=lambda item: item.ordinal)
    ranges: list[CoarseRange] = []
    for doc_id, doc_regions in sorted(by_doc.items()):
        if range_type == "article":
            ranges.append(_range_from_regions(doc_id, "article", doc_regions, title=doc_id, suffix="all"))
            continue
        for start in range(0, len(doc_regions), window_size):
            chunk = doc_regions[start : start + window_size]
            if not chunk:
                continue
            suffix = f"{chunk[0].region_id}-{chunk[-1].region_id}"
            title = f"{doc_id} {suffix}"
            ranges.append(_range_from_regions(doc_id, "window", chunk, title=title, suffix=suffix))
    return ranges


def _range_from_regions(doc_id: str, range_type: str, regions: list[SearchRegion], *, title: str, suffix: str) -> CoarseRange:
    text = "\n\n".join(region.text for region in regions)
    stable_refs = [region.stable_ref for region in regions]
    digest = hashlib.sha256(("\n".join(stable_refs) + "\n" + text).encode("utf-8")).hexdigest()[:16]
    return CoarseRange(
        range_id=f"{doc_id}:{range_type}:{suffix}",
        doc_id=doc_id,
        title=title,
        stable_refs=stable_refs,
        text=text,
        hash=digest,
        range_type=range_type,
    )


def sample_ranges(ranges: list[CoarseRange], *, limit: int, seed: int) -> list[CoarseRange]:
    if len(ranges) <= limit:
        return ranges
    rng = random.Random(seed)
    return [ranges[index] for index in sorted(rng.sample(range(len(ranges)), limit))]


def load_or_generate_questions(
    ranges: list[CoarseRange],
    *,
    cache_path: Path,
    source: str,
    model: str,
    endpoint: str,
    api_key_env: str,
    concurrency: int,
) -> list[RangeQuestion]:
    cache = read_cache(cache_path)
    missing = [item for item in ranges if cache_key(item, source=source, model=model) not in cache]
    if missing:
        generated = generate_questions(
            missing,
            source=source,
            model=model,
            endpoint=endpoint,
            api_key_env=api_key_env,
            concurrency=concurrency,
        )
        append_cache(cache_path, generated)
        for question in generated:
            cache[(question.range_id, question.hash, question.source, question.model)] = question
    return [cache[cache_key(item, source=source, model=model)] for item in ranges if cache_key(item, source=source, model=model) in cache]


def generate_questions(
    ranges: list[CoarseRange],
    *,
    source: str,
    model: str,
    endpoint: str,
    api_key_env: str,
    concurrency: int,
) -> list[RangeQuestion]:
    if source == "local":
        return [question_from_range(item, query=local_question(item), source=source, model=source) for item in ranges]
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set.")

    def one(item: CoarseRange) -> RangeQuestion:
        return question_from_range(
            item,
            query=openrouter_question(item, model=model, endpoint=endpoint, api_key=api_key),
            source=source,
            model=model,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        return list(executor.map(one, ranges))


def question_from_range(item: CoarseRange, *, query: str, source: str, model: str) -> RangeQuestion:
    return RangeQuestion(
        query=query,
        range_id=item.range_id,
        doc_id=item.doc_id,
        gold_refs=item.stable_refs,
        hash=item.hash,
        source=source,
        model=model,
        range_type=item.range_type,
    )


def local_question(item: CoarseRange) -> str:
    view = local_view(item.text, questions_per_region=1)
    topic = ", ".join(view.keywords[:4]) if view.keywords else item.title
    return f"Which section explains {topic}?"


def openrouter_question(item: CoarseRange, *, model: str, endpoint: str, api_key: str) -> str:
    text = item.text[:9000]
    prompt = f"""Write one broad documentation search query for the supplied section/range.

The query should target the whole range, not a single exact sentence.
Prefer realistic user navigation wording, for example "how do I configure X" or "where are Y options explained".
Return strict JSON: {{"query": "..."}}

Range title: {item.title}
Range refs: {item.stable_refs[0]}..{item.stable_refs[-1]}
Text:
{text}
"""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You write broad documentation navigation queries for search evaluation. Return only JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.55,
        "max_tokens": 120,
    }
    req = request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/b-imenitov/refmark",
            "X-Title": "refmark coarse range eval",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=90) as response:
        payload = json.loads(response.read().decode("utf-8"))
    content = payload["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.strip("`").removeprefix("json").strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end < start:
            return local_question(item)
        try:
            parsed = json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            return local_question(item)
    return str(parsed.get("query", "")).strip() or local_question(item)


def read_cache(path: Path) -> dict[tuple[str, str, str, str], RangeQuestion]:
    if not path.exists():
        return {}
    cache = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        question = RangeQuestion(
            query=str(row["query"]),
            range_id=str(row["range_id"]),
            doc_id=str(row["doc_id"]),
            gold_refs=[str(item) for item in row["gold_refs"]],
            hash=str(row["hash"]),
            source=str(row["source"]),
            model=str(row["model"]),
            range_type=str(row.get("range_type", "window")),
        )
        cache[(question.range_id, question.hash, question.source, question.model)] = question
    return cache


def append_cache(path: Path, questions: list[RangeQuestion]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for question in questions:
            handle.write(json.dumps(question.to_dict(), ensure_ascii=False) + "\n")


def cache_key(item: CoarseRange, *, source: str, model: str) -> tuple[str, str, str, str]:
    return (item.range_id, item.hash, source, model if source == "openrouter" else source)


def evaluate(index: PortableBM25Index, questions: list[RangeQuestion], *, top_ks: tuple[int, ...], expand_after: int) -> dict[str, object]:
    max_k = max(top_ks)
    any_hits = Counter({k: 0 for k in top_ks})
    doc_hits = Counter({k: 0 for k in top_ks})
    full_context_hits = Counter({k: 0 for k in top_ks})
    range_scores = {k: [] for k in top_ks}
    reciprocal = 0.0
    misses = []
    for question in questions:
        hits = index.search(question.query, top_k=max_k, expand_after=expand_after)
        gold = set(question.gold_refs)
        any_rank = None
        doc_rank = None
        full_context_rank = None
        for rank, hit in enumerate(hits, start=1):
            if any_rank is None and hit.stable_ref in gold:
                any_rank = rank
            if doc_rank is None and hit.doc_id == question.doc_id:
                doc_rank = rank
            if full_context_rank is None and gold.issubset(set(hit.context_refs)):
                full_context_rank = rank
        if any_rank is not None:
            reciprocal += 1.0 / any_rank
        else:
            misses.append({"query": question.query, "gold_range": [question.gold_refs[0], question.gold_refs[-1]], "top_refs": [hit.stable_ref for hit in hits[:3]]})
        for k in top_ks:
            selected_refs = sorted({ref for hit in hits[:k] for ref in hit.context_refs})
            range_scores[k].append(score_ref_range(selected_refs, question.gold_refs))
            if any_rank is not None and any_rank <= k:
                any_hits[k] += 1
            if doc_rank is not None and doc_rank <= k:
                doc_hits[k] += 1
            if full_context_rank is not None and full_context_rank <= k:
                full_context_hits[k] += 1
    total = max(len(questions), 1)
    return {
        "any_anchor_in_range_at_k": {str(k): round(any_hits[k] / total, 4) for k in top_ks},
        "article_hit_at_k": {str(k): round(doc_hits[k] / total, 4) for k in top_ks},
        "full_range_context_at_k": {str(k): round(full_context_hits[k] / total, 4) for k in top_ks},
        "range_score_at_k": {str(k): summarize_scores(range_scores[k]) for k in top_ks},
        "mrr": round(reciprocal / total, 4),
        "sample_misses": misses[:8],
    }


def summarize_ranges(ranges: list[CoarseRange]) -> dict[str, object]:
    sizes = [len(item.stable_refs) for item in ranges]
    if not sizes:
        return {"count": 0}
    return {
        "count": len(sizes),
        "min_refs": min(sizes),
        "max_refs": max(sizes),
        "avg_refs": round(sum(sizes) / len(sizes), 2),
    }


if __name__ == "__main__":
    main()
