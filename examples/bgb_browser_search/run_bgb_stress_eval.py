"""Generate and evaluate adversarial BGB navigation questions.

This script samples BGB article blocks, asks one or more remote models to create
fresh realistic navigation questions for each block, and evaluates whether raw
or Refmark-enriched search lands back inside the source block.

The generator output is cached. Gold refs come from the sampled article block,
not from the model response.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from urllib import request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.bgb_browser_search.build_bgb_article_navigation import article_id_for  # noqa: E402
from refmark.search_index import OPENROUTER_CHAT_URL, PortableBM25Index, SearchHit, SearchRegion, load_search_index  # noqa: E402


PROMPT_VERSION = "bgb-stress-v2"
DEFAULT_MODELS = "qwen/qwen-turbo,mistralai/mistral-nemo,mistralai/mistral-small-3.2-24b-instruct,google/gemma-3-27b-it"


@dataclass(frozen=True)
class ArticleBlock:
    block_id: str
    title: str
    stable_refs: list[str]
    text: str
    hash: str
    ordinal: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StressQuestion:
    query: str
    block_id: str
    gold_refs: list[str]
    block_hash: str
    generator_model: str
    language: str
    style: str
    source: str = "openrouter"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run randomized BGB stress-question retrieval evaluation.")
    parser.add_argument("--raw-index", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_raw_index.json")
    parser.add_argument("--enriched-index", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_openrouter_index.json")
    parser.add_argument("--output", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_stress_eval.json")
    parser.add_argument("--cache", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_stress_questions.jsonl")
    parser.add_argument("--models", default=DEFAULT_MODELS, help="Comma-separated OpenRouter chat models.")
    parser.add_argument("--languages", default="de,en")
    parser.add_argument("--styles", default="direct,concern,adversarial")
    parser.add_argument("--questions-per-style", type=int, default=1)
    parser.add_argument("--sample-size", type=int, default=32)
    parser.add_argument(
        "--token-budget",
        type=int,
        default=None,
        help="Sample article blocks until this approximate source-token budget is reached. Overrides --sample-size.",
    )
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--min-words", type=int, default=35)
    parser.add_argument("--max-block-chars", type=int, default=6500)
    parser.add_argument("--top-ks", default="1,3,5,10")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--endpoint", default=OPENROUTER_CHAT_URL)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--preflight", action="store_true", help="Probe each generator model before the full run.")
    args = parser.parse_args()

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set.")

    raw = load_search_index(args.raw_index)
    enriched = load_search_index(args.enriched_index)
    blocks = sample_blocks(
        build_article_blocks(enriched.regions),
        limit=args.sample_size,
        token_budget=args.token_budget,
        seed=args.seed,
        min_words=args.min_words,
    )
    models = _split_arg(args.models)
    languages = _split_arg(args.languages)
    styles = _split_arg(args.styles)
    if args.preflight:
        models = [model for model in models if preflight_model(model, endpoint=args.endpoint, api_key=api_key)]
        if not models:
            raise RuntimeError("No generator models passed preflight.")

    questions = load_or_generate_questions(
        blocks,
        models=models,
        languages=languages,
        styles=styles,
        questions_per_style=args.questions_per_style,
        cache_path=Path(args.cache),
        endpoint=args.endpoint,
        api_key=api_key,
        concurrency=args.concurrency,
        max_block_chars=args.max_block_chars,
    )
    top_ks = tuple(int(part) for part in args.top_ks.split(",") if part.strip())
    report = {
        "schema": "refmark.bgb_stress_eval.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": vars(args),
        "prompt_version": PROMPT_VERSION,
        "blocks": [block.to_dict() for block in blocks],
        "block_count": len(blocks),
        "slice_approx_tokens": sum(approx_tokens(block.text) for block in blocks),
        "question_count": len(questions),
        "models": models,
        "languages": languages,
        "styles": styles,
        "reports": {
            "raw_bm25": evaluate(raw, questions, top_ks=top_ks),
            "refmark_bm25": evaluate(enriched, questions, top_ks=top_ks),
            "refmark_rerank": evaluate(enriched, questions, top_ks=top_ks, strategy="rerank"),
        },
        "sections": section_question_rows(blocks, questions),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def build_article_blocks(regions: list[SearchRegion]) -> list[ArticleBlock]:
    groups: dict[tuple[str, str], list[SearchRegion]] = {}
    for region in regions:
        groups.setdefault((region.doc_id, article_id_for(region.region_id)), []).append(region)
    blocks: list[ArticleBlock] = []
    for (doc_id, article_id), items in sorted(groups.items(), key=lambda item: min(region.ordinal for region in item[1])):
        ordered = sorted(items, key=lambda item: item.ordinal)
        stable_refs = [region.stable_ref for region in ordered]
        title = f"{doc_id}:{article_id}"
        text = "\n\n".join(region.text for region in ordered)
        digest = hashlib.sha256(("\n".join(stable_refs) + "\n" + text).encode("utf-8")).hexdigest()[:16]
        blocks.append(
            ArticleBlock(
                block_id=f"{doc_id}:{article_id}",
                title=title,
                stable_refs=stable_refs,
                text=text,
                hash=digest,
                ordinal=min(region.ordinal for region in ordered),
            )
        )
    return blocks


def sample_blocks(
    blocks: list[ArticleBlock],
    *,
    limit: int,
    token_budget: int | None = None,
    seed: int,
    min_words: int,
) -> list[ArticleBlock]:
    candidates = [block for block in blocks if len(block.text.split()) >= min_words and "(weggefallen)" not in block.text]
    if token_budget is not None:
        rng = random.Random(seed)
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        selected = []
        used_tokens = 0
        for block in shuffled:
            selected.append(block)
            used_tokens += approx_tokens(block.text)
            if used_tokens >= token_budget:
                break
        return sorted(selected, key=lambda block: block.ordinal)
    if len(candidates) <= limit:
        return candidates
    rng = random.Random(seed)
    return [candidates[index] for index in sorted(rng.sample(range(len(candidates)), limit))]


def load_or_generate_questions(
    blocks: list[ArticleBlock],
    *,
    models: list[str],
    languages: list[str],
    styles: list[str],
    questions_per_style: int,
    cache_path: Path,
    endpoint: str,
    api_key: str,
    concurrency: int,
    max_block_chars: int,
) -> list[StressQuestion]:
    cache = read_cache(cache_path)
    jobs = [
        (block, model, language)
        for block in blocks
        for model in models
        for language in languages
        if cache_key(block, model=model, language=language, styles=styles, questions_per_style=questions_per_style) not in cache
    ]
    generated: list[StressQuestion] = []
    if jobs:
        def one(job: tuple[ArticleBlock, str, str]) -> list[StressQuestion]:
            block, model, language = job
            try:
                return generate_questions(
                    block,
                    model=model,
                    language=language,
                    styles=styles,
                    questions_per_style=questions_per_style,
                    endpoint=endpoint,
                    api_key=api_key,
                    max_block_chars=max_block_chars,
                )
            except Exception as exc:
                print(
                    f"generation failed: block={block.block_id} model={model} language={language}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                return []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
            for rows in executor.map(one, jobs):
                generated.extend(rows)
        append_cache(cache_path, generated, requested_styles=styles, questions_per_style=questions_per_style)
        for question in generated:
            cache.setdefault(cache_key_for_question(question, styles=styles, questions_per_style=questions_per_style), []).append(question)

    questions: list[StressQuestion] = []
    for block in blocks:
        for model in models:
            for language in languages:
                questions.extend(cache.get(cache_key(block, model=model, language=language, styles=styles, questions_per_style=questions_per_style), []))
    return questions


def generate_questions(
    block: ArticleBlock,
    *,
    model: str,
    language: str,
    styles: list[str],
    questions_per_style: int,
    endpoint: str,
    api_key: str,
    max_block_chars: int,
) -> list[StressQuestion]:
    prompt = stress_prompt(block, language=language, styles=styles, questions_per_style=questions_per_style, max_block_chars=max_block_chars)
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You create difficult but fair search-evaluation questions. "
                    "Return strict JSON only. Do not cite refs in the questions."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.85,
        "max_tokens": 900,
    }
    payload = _openrouter_json(body, endpoint=endpoint, api_key=api_key, timeout=120)
    parsed = parse_json_object(payload["choices"][0]["message"]["content"])
    rows = parsed.get("questions", [])
    output: list[StressQuestion] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        query = str(row.get("query", "")).strip()
        style = str(row.get("style", "")).strip() or "unknown"
        if style not in styles:
            style = styles[index % len(styles)]
        if not query:
            continue
        output.append(
            StressQuestion(
                query=query,
                block_id=block.block_id,
                gold_refs=block.stable_refs,
                block_hash=block.hash,
                generator_model=model,
                language=language,
                style=style,
            )
        )
    return dedupe_questions(output)


def stress_prompt(
    block: ArticleBlock,
    *,
    language: str,
    styles: list[str],
    questions_per_style: int,
    max_block_chars: int,
) -> str:
    style_notes = {
        "direct": "A normal documentation/legal search query that should land in this article.",
        "concern": "A layperson concern or situation. Avoid section numbers and avoid copying legal headings.",
        "adversarial": "A difficult but fair query with synonyms, vague wording, or misleading everyday words.",
    }
    style_lines = "\n".join(f"- {style}: {style_notes.get(style, 'A realistic search query.')}" for style in styles)
    example_rows = ",".join(f'{{"style":"{style}","query":"..."}}' for style in styles)
    total = len(styles) * questions_per_style
    text = block.text[:max_block_chars]
    return f"""Generate {total} fresh BGB search questions for the article block below.

Language: {language}
For each style, generate exactly {questions_per_style} question(s).

Styles:
{style_lines}

Rules:
- The answer should be in this article block, not in a different BGB article.
- Do not mention refs, paragraph numbers, article numbers, or "BGB".
- Do not copy a full sentence from the source.
- Prefer realistic user wording; stress retrieval with paraphrase and motivation.
- Return strict JSON only:
  {{"questions":[{example_rows}]}}

Article block: {block.title}
Refs: {block.stable_refs[0]}..{block.stable_refs[-1]}
Text:
{text}
"""


def evaluate(index: PortableBM25Index, questions: list[StressQuestion], *, top_ks: tuple[int, ...], strategy: str = "flat") -> dict[str, object]:
    max_k = max(top_ks)
    any_ranks: list[int | None] = []
    article_ranks: list[int | None] = []
    by_model: dict[str, list[int | None]] = {}
    by_style: dict[str, list[int | None]] = {}
    by_language: dict[str, list[int | None]] = {}
    misses = []
    wrong_top_articles: Counter[str] = Counter()
    missed_blocks: Counter[str] = Counter()
    for question in questions:
        hits = search(index, question.query, top_k=max_k, strategy=strategy)
        gold = set(question.gold_refs)
        gold_articles = {article_ref(ref) for ref in question.gold_refs}
        any_rank = None
        article_rank = None
        for hit in hits:
            if any_rank is None and hit.stable_ref in gold:
                any_rank = hit.rank
            if article_rank is None and article_ref(hit.stable_ref) in gold_articles:
                article_rank = hit.rank
        any_ranks.append(any_rank)
        article_ranks.append(article_rank)
        by_model.setdefault(question.generator_model, []).append(article_rank)
        by_style.setdefault(question.style, []).append(article_rank)
        by_language.setdefault(question.language, []).append(article_rank)
        if article_rank is None:
            missed_blocks[question.block_id] += 1
            if hits:
                wrong_top_articles[article_ref(hits[0].stable_ref)] += 1
            if len(misses) < 40:
                misses.append(
                    {
                        "query": question.query,
                        "block_id": question.block_id,
                        "language": question.language,
                        "style": question.style,
                        "generator_model": question.generator_model,
                        "gold_range": [question.gold_refs[0], question.gold_refs[-1]],
                        "top_refs": [hit.stable_ref for hit in hits[:5]],
                        "top_articles": [article_ref(hit.stable_ref) for hit in hits[:5]],
                    }
                )
    return {
        "anchor_hit_at_k": summarize_ranks(any_ranks, top_ks=top_ks),
        "article_hit_at_k": summarize_ranks(article_ranks, top_ks=top_ks),
        "article_mrr": round(sum(1.0 / rank for rank in article_ranks if rank is not None) / max(len(article_ranks), 1), 4),
        "by_generator_model": {model: summarize_ranks(ranks, top_ks=top_ks) for model, ranks in sorted(by_model.items())},
        "by_style": {style: summarize_ranks(ranks, top_ks=top_ks) for style, ranks in sorted(by_style.items())},
        "by_language": {language: summarize_ranks(ranks, top_ks=top_ks) for language, ranks in sorted(by_language.items())},
        "weakness_heatmap": {
            "misses": sum(1 for rank in article_ranks if rank is None),
            "miss_rate": round(sum(1 for rank in article_ranks if rank is None) / max(len(article_ranks), 1), 4),
            "missed_blocks_top": counter_rows(missed_blocks, limit=20),
            "wrong_top_articles_top": counter_rows(wrong_top_articles, limit=20),
        },
        "misses": misses,
    }


def search(index: PortableBM25Index, query: str, *, top_k: int, strategy: str) -> list[SearchHit]:
    if strategy == "rerank":
        return index.search_reranked(query, top_k=top_k, candidate_k=max(50, top_k))
    return index.search(query, top_k=top_k)


def summarize_ranks(ranks: list[int | None], *, top_ks: tuple[int, ...]) -> dict[str, object]:
    total = max(len(ranks), 1)
    return {
        "count": len(ranks),
        "hit_at_k": {str(k): round(sum(1 for rank in ranks if rank is not None and rank <= k) / total, 4) for k in top_ks},
        "mrr": round(sum(1.0 / rank for rank in ranks if rank is not None) / total, 4),
    }


def section_question_rows(blocks: list[ArticleBlock], questions: list[StressQuestion]) -> list[dict[str, object]]:
    by_block: dict[str, list[StressQuestion]] = {}
    for question in questions:
        by_block.setdefault(question.block_id, []).append(question)
    return [
        {
            "block_id": block.block_id,
            "gold_range": [block.stable_refs[0], block.stable_refs[-1]],
            "questions": [question.to_dict() for question in by_block.get(block.block_id, [])],
        }
        for block in blocks
    ]


def preflight_model(model: str, *, endpoint: str, api_key: str) -> bool:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": 'Return {"ok": true}.'},
        ],
        "temperature": 0.0,
        "max_tokens": 20,
    }
    try:
        payload = _openrouter_json(body, endpoint=endpoint, api_key=api_key, timeout=30)
        parse_json_object(payload["choices"][0]["message"]["content"])
        print(f"preflight ok: {model}", file=sys.stderr)
        return True
    except Exception as exc:
        print(f"preflight failed: {model}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return False


def _openrouter_json(body: dict[str, object], *, endpoint: str, api_key: str, timeout: int) -> dict[str, object]:
    req = request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/b-imenitov/refmark",
            "X-Title": "refmark BGB stress eval",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_json_object(text: str) -> dict[str, object]:
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.strip("`").removeprefix("json").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        start = clean.find("{")
        end = clean.rfind("}")
        if start < 0 or end < start:
            raise
        return json.loads(clean[start : end + 1])


def read_cache(path: Path) -> dict[tuple[str, str, str, str, str, int, str], list[StressQuestion]]:
    cache: dict[tuple[str, str, str, str, str, int, str], list[StressQuestion]] = {}
    if not path.exists():
        return cache
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        question = StressQuestion(
            query=str(row["query"]),
            block_id=str(row["block_id"]),
            gold_refs=[str(ref) for ref in row["gold_refs"]],
            block_hash=str(row["block_hash"]),
            generator_model=str(row["generator_model"]),
            language=str(row["language"]),
            style=str(row["style"]),
            source=str(row.get("source", "openrouter")),
        )
        key = cache_key_for_question(
            question,
            styles=[str(item) for item in row.get("requested_styles", [])],
            questions_per_style=int(row.get("questions_per_style", 1)),
            prompt_version=str(row.get("prompt_version", PROMPT_VERSION)),
        )
        cache.setdefault(key, []).append(question)
    return cache


def append_cache(path: Path, questions: list[StressQuestion], *, requested_styles: list[str], questions_per_style: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for question in questions:
            row = question.to_dict()
            row["prompt_version"] = PROMPT_VERSION
            row["requested_styles"] = requested_styles
            row["questions_per_style"] = questions_per_style
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def cache_key(
    block: ArticleBlock,
    *,
    model: str,
    language: str,
    styles: list[str],
    questions_per_style: int,
    prompt_version: str = PROMPT_VERSION,
) -> tuple[str, str, str, str, str, int, str]:
    return (block.block_id, block.hash, model, language, ",".join(sorted(styles)), questions_per_style, prompt_version)


def cache_key_for_question(
    question: StressQuestion,
    *,
    styles: list[str],
    questions_per_style: int,
    prompt_version: str = PROMPT_VERSION,
) -> tuple[str, str, str, str, str, int, str]:
    return (
        question.block_id,
        question.block_hash,
        question.generator_model,
        question.language,
        ",".join(sorted(styles)),
        questions_per_style,
        prompt_version,
    )


def dedupe_questions(questions: list[StressQuestion]) -> list[StressQuestion]:
    output: list[StressQuestion] = []
    seen: set[tuple[str, str]] = set()
    for question in questions:
        key = (question.style, " ".join(question.query.lower().split()))
        if key in seen:
            continue
        seen.add(key)
        output.append(question)
    return output


def article_ref(stable_ref: str) -> str:
    doc_id, region_id = stable_ref.split(":", 1)
    return f"{doc_id}:{article_id_for(region_id)}"


def counter_rows(counter: Counter[str], *, limit: int) -> list[dict[str, object]]:
    return [{"ref": ref, "count": count} for ref, count in counter.most_common(limit)]


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _split_arg(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


if __name__ == "__main__":
    main()
