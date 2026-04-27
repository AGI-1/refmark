"""Run the BGB Refmark search pipeline end to end.

This script is intentionally example-shaped: it starts with the official BGB
HTML, builds portable/browser indexes, generates held-out user questions, and
then runs concern-style searches against the resulting index.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
from urllib import request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.bgb_browser_search.build_bgb_demo import (  # noqa: E402
    BGB_URL,
    BgbRegion,
    fetch_source,
    hash_text,
    local_legal_view,
    parse_bgb,
    render_demo_html,
)
from examples.bgb_browser_search.build_bgb_article_navigation import (  # noqa: E402
    ConcernAlias,
    load_aliases,
)
from examples.portable_search_index.evaluate_real_corpus import (  # noqa: E402
    EvalQuestion,
    EvalTarget,
    _sample_targets,
)
from refmark.pipeline import RegionRecord  # noqa: E402
from refmark.search_index import (  # noqa: E402
    OPENROUTER_CHAT_URL,
    PortableBM25Index,
    RetrievalView,
    SearchRegion,
    approx_tokens,
    export_browser_search_index,
    generate_views,
    load_search_index,
)


CONCERNS = [
    {
        "query": "I rent an apartment and need to know the notice period for ending the lease.",
        "expected_contains": ["S_573c"],
    },
    {
        "query": "The apartment has defects. Can the tenant reduce rent because of the problem?",
        "expected_contains": ["S_536"],
    },
    {
        "query": "A minor wants to sign a contract. When is the contract valid?",
        "expected_contains": ["S_106", "S_107"],
    },
    {
        "query": "How can someone reject or disclaim an inheritance?",
        "expected_contains": ["S_1942"],
    },
    {
        "query": "A consumer wants to revoke a contract after ordering online.",
        "expected_contains": ["S_355"],
    },
    {
        "query": "Software or a digital product is defective and the seller must provide updates.",
        "expected_contains": ["S_327", "S_327f", "S_327i", "S_475b"],
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and evaluate a BGB Refmark search pipeline.")
    parser.add_argument("--output-dir", default="examples/bgb_browser_search/output_scratch")
    parser.add_argument("--source-url", default=BGB_URL)
    parser.add_argument("--force-fetch", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional region limit for quick probes.")
    parser.add_argument("--view-source", choices=["local", "openrouter"], default="openrouter")
    parser.add_argument("--question-source", choices=["local", "openrouter"], default="openrouter")
    parser.add_argument("--model", default="mistralai/mistral-nemo")
    parser.add_argument("--language", choices=["de", "en"], default="de")
    parser.add_argument(
        "--languages",
        default=None,
        help="Comma-separated retrieval/eval languages. Overrides --language, e.g. de,en.",
    )
    parser.add_argument("--endpoint", default=OPENROUTER_CHAT_URL)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--view-batch-size", type=int, default=250)
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--top-ks", default="1,3,5,10")
    parser.add_argument("--questions-per-region", type=int, default=4)
    parser.add_argument("--keywords-per-region", type=int, default=8)
    parser.add_argument(
        "--concern-aliases",
        default="examples/bgb_browser_search/concern_aliases.json",
        help="Optional motivation/concern alias JSON to add as gold_mode=concern eval questions.",
    )
    parser.add_argument(
        "--skip-concern-questions",
        action="store_true",
        help="Do not add concern alias questions to the eval/training question set.",
    )
    args = parser.parse_args()
    languages = tuple(part.strip() for part in (args.languages or args.language).split(",") if part.strip())
    if not languages or any(language not in {"de", "en"} for language in languages):
        raise SystemExit("--languages may only contain de,en.")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_path = output / "bgb_official.html"
    source_html = fetch_source(args.source_url, source_path, force=args.force_fetch)
    bgb_regions = parse_bgb(source_html)
    if args.limit is not None:
        bgb_regions = bgb_regions[: args.limit]
    records = [_to_record(region, bgb_regions) for region in bgb_regions]
    concern_aliases = [] if args.skip_concern_questions else _load_concern_aliases(args.concern_aliases)

    raw_index_path = output / "bgb_raw_index.json"
    local_index_path = output / "bgb_local_index.json"
    enriched_index_path = output / f"bgb_{args.view_source}_index.json"
    browser_index_path = output / f"bgb_{args.view_source}_browser_index.json"
    data_js_path = output / "bgb_demo_data.js"
    demo_path = output / "index.html"
    question_cache = output / "bgb_question_cache.jsonl"
    eval_suite_path = output / "bgb_eval_questions.jsonl"
    view_cache = output / f"bgb_view_cache_{_safe_model(args.model)}.jsonl"

    raw_views = {(record.doc_id, record.region_id): RetrievalView(summary="", questions=[], keywords=[]) for record in records}
    local_views = {(record.doc_id, record.region_id): local_legal_view(bgb_regions[index]) for index, record in enumerate(records)}
    _write_portable_index(records, raw_views, raw_index_path, source=args.source_url, settings={"view_source": "raw", "model": "raw"})
    _write_portable_index(
        records,
        local_views,
        local_index_path,
        source=args.source_url,
        settings={"view_source": "local-bgb-demo", "model": "local-bgb-demo"},
    )

    if args.view_source == "openrouter":
        views_by_language = {
            language: _generate_views_batched(records, args=args, cache_path=view_cache, language=language)
            for language in languages
        }
        views = _combine_language_views(records, views_by_language, languages=languages)
    else:
        views = local_views
    if concern_aliases:
        views = _apply_concern_aliases_to_views(records, views, concern_aliases)
    _write_portable_index(
        records,
        views,
        enriched_index_path,
        source=args.source_url,
        settings={
            "view_source": args.view_source,
            "model": args.model if args.view_source == "openrouter" else args.view_source,
            "languages": list(languages),
        },
    )

    browser_payload = export_browser_search_index(enriched_index_path, browser_index_path, max_text_chars=1000)
    data_js_path.write_text(
        "window.BGB_REFMARK_INDEX = "
        + json.dumps(browser_payload, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    demo_path.write_text(render_demo_html(bgb_regions), encoding="utf-8")

    rng = random.Random(args.seed)
    candidates = [record for record in records if len(record.text.split()) >= 8 and "(weggefallen)" not in record.text]
    targets = _sample_targets(candidates, limit=args.sample_size, seed=args.seed, gold_mode="single")
    rng.shuffle(targets)
    questions = [
        question
        for language in languages
        for question in _load_or_generate_questions_bgb(
            targets,
            source=args.question_source,
            model=args.model,
            endpoint=args.endpoint,
            api_key_env=args.api_key_env,
            cache_path=question_cache,
            concurrency=args.concurrency,
            language=language,
        )
    ]
    concern_questions = _concern_questions(
        records,
        concern_aliases,
        source="curated",
        model="concern-aliases",
    )
    questions = [*questions, *concern_questions]
    _write_jsonl(eval_suite_path, [question.to_dict() for question in questions])

    raw_index = load_search_index(raw_index_path)
    enriched_index = load_search_index(enriched_index_path)
    top_ks = tuple(int(part) for part in args.top_ks.split(",") if part.strip())
    report = {
        "schema": "refmark.bgb_pipeline_report.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_url": args.source_url,
        "regions": len(records),
        "approx_tokens": sum(approx_tokens(record.text) for record in records),
        "artifacts": {
            "raw_index": str(raw_index_path),
            "local_index": str(local_index_path),
            "enriched_index": str(enriched_index_path),
            "browser_index": str(browser_index_path),
            "demo": str(demo_path),
            "question_cache": str(question_cache),
            "eval_suite": str(eval_suite_path),
            "view_cache": str(view_cache),
        },
        "settings": vars(args),
        "languages": list(languages),
        "eval_questions": len(questions),
        "concern_questions": len(concern_questions),
        "concern_aliases": args.concern_aliases if concern_aliases else None,
        "metrics": {
            "raw_bm25": _evaluate(raw_index, questions, top_ks=top_ks),
            f"{args.view_source}_refmark_bm25": _evaluate(enriched_index, questions, top_ks=top_ks),
            f"{args.view_source}_refmark_rerank": _evaluate(enriched_index, questions, top_ks=top_ks, strategy="rerank"),
        },
        "metrics_by_language": {
            language: {
                "raw_bm25": _evaluate(raw_index, [question for question in questions if str(question.model).endswith(f"@{language}")], top_ks=top_ks),
                f"{args.view_source}_refmark_bm25": _evaluate(
                    enriched_index,
                    [question for question in questions if str(question.model).endswith(f"@{language}")],
                    top_ks=top_ks,
                ),
                f"{args.view_source}_refmark_rerank": _evaluate(
                    enriched_index,
                    [question for question in questions if str(question.model).endswith(f"@{language}")],
                    top_ks=top_ks,
                    strategy="rerank",
                ),
            }
            for language in languages
        },
        "concerns": _run_concerns(enriched_index),
    }
    report_path = output / "bgb_pipeline_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def _to_record(region: BgbRegion, regions: list[BgbRegion]) -> RegionRecord:
    prev_region_id = regions[region.ordinal - 1].region_id if region.ordinal > 0 else None
    next_region_id = regions[region.ordinal + 1].region_id if region.ordinal + 1 < len(regions) else None
    return RegionRecord(
        doc_id=region.doc_id,
        region_id=region.region_id,
        text=region.text,
        start_line=region.ordinal + 1,
        end_line=region.ordinal + 1,
        ordinal=region.ordinal,
        hash=hash_text(region.text),
        source_path=region.source_anchor,
        prev_region_id=prev_region_id,
        next_region_id=next_region_id,
    )


def _generate_views_batched(
    records: list[RegionRecord],
    *,
    args: argparse.Namespace,
    cache_path: Path,
    language: str,
) -> dict[tuple[str, str], RetrievalView]:
    return _generate_language_views_batched(records, args=args, cache_path=cache_path, language=language)


def _generate_language_views_batched(
    records: list[RegionRecord],
    *,
    args: argparse.Namespace,
    cache_path: Path,
    language: str,
) -> dict[tuple[str, str], RetrievalView]:
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set.")
    cache = _read_language_view_cache(cache_path, language=language, model=args.model)
    output: dict[tuple[str, str], RetrievalView] = {}
    missing: list[RegionRecord] = []
    for record in records:
        key = (f"{record.doc_id}:{record.region_id}", record.hash, language, args.model)
        if key in cache:
            output[(record.doc_id, record.region_id)] = cache[key]
        else:
            missing.append(record)

    batch_size = max(1, args.view_batch_size)
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        rows = []
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
            for record, view in executor.map(
                lambda item: (item, _openrouter_language_view(item, args=args, api_key=api_key, language=language)),
                batch,
            ):
                output[(record.doc_id, record.region_id)] = view
                rows.append(
                    {
                        "stable_ref": f"{record.doc_id}:{record.region_id}",
                        "doc_id": record.doc_id,
                        "region_id": record.region_id,
                        "hash": record.hash,
                        "language": language,
                        "model": args.model,
                        "view": view.to_dict(),
                    }
                )
        _append_jsonl(cache_path, rows)
        print(json.dumps({"view_batch_done": len(output), "regions": len(records), "language": language}, ensure_ascii=False), file=sys.stderr)
    return output


def _combine_language_views(
    records: list[RegionRecord],
    views_by_language: dict[str, dict[tuple[str, str], RetrievalView]],
    *,
    languages: tuple[str, ...],
) -> dict[tuple[str, str], RetrievalView]:
    combined: dict[tuple[str, str], RetrievalView] = {}
    for record in records:
        key = (record.doc_id, record.region_id)
        summaries = []
        questions = []
        keywords = []
        for language in languages:
            view = views_by_language[language][key]
            if view.summary:
                summaries.append(f"[{language}] {view.summary}")
            questions.extend(f"[{language}] {question}" for question in view.questions if question)
            keywords.extend(f"[{language}] {keyword}" for keyword in view.keywords if keyword)
        combined[key] = RetrievalView(summary=" ".join(summaries), questions=questions, keywords=keywords)
    return combined


def _openrouter_language_view(record: RegionRecord, *, args: argparse.Namespace, api_key: str, language: str) -> RetrievalView:
    if language == "de":
        prompt = f"""Erstelle Such-Metadaten für diesen Abschnitt des deutschen Bürgerlichen Gesetzbuchs.

Antworte ausschließlich auf Deutsch.
Gib striktes JSON zurück mit diesen Schlüsseln:
- summary: ein kurzer deutscher Satz
- questions: {args.questions_per_region} natürliche deutschsprachige Nutzerfragen, die dieser Abschnitt beantworten kann
- keywords: {args.keywords_per_region} kurze deutsche Suchbegriffe oder Phrasen

Keine Markdown-Fences.

Dokument: {record.doc_id}
Region: {record.region_id}
Text:
{record.text}
"""
        system = "Du erstellst präzise deutsche Such-Metadaten für juristische Dokumente."
    else:
        prompt = f"""Create search metadata for this section of the German Civil Code.

Answer only in English.
Return strict JSON with these keys:
- summary: one short English sentence
- questions: {args.questions_per_region} natural English user questions this section could answer
- keywords: {args.keywords_per_region} short English search terms or phrases

Do not include markdown fences.

Document: {record.doc_id}
Region: {record.region_id}
Text:
{record.text}
"""
        system = "You create precise English search metadata for German legal documents."
    body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 360,
    }
    parsed = _openrouter_json(body, endpoint=args.endpoint, api_key=api_key)
    if not parsed:
        return local_legal_view(_record_to_bgb_region(record))
    return RetrievalView(
        summary=str(parsed.get("summary", "")).strip(),
        questions=[str(item).strip() for item in parsed.get("questions", []) if str(item).strip()][: args.questions_per_region],
        keywords=[str(item).strip() for item in parsed.get("keywords", []) if str(item).strip()][: args.keywords_per_region],
    )


def _load_or_generate_questions_bgb(
    targets: list[EvalTarget],
    *,
    source: str,
    model: str,
    endpoint: str,
    api_key_env: str,
    cache_path: Path,
    concurrency: int,
    language: str,
) -> list[EvalQuestion]:
    cache = _read_language_question_cache(cache_path)
    api_model = model
    cache_model = f"{model}@{language}" if source == "openrouter" else f"{source}@{language}"
    missing = [target for target in targets if _question_cache_key(target, source=source, model=cache_model, language=language) not in cache]
    if missing:
        if source == "local":
            generated = [_local_bgb_question(target, language=language, model=cache_model) for target in missing]
        else:
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise RuntimeError(f"{api_key_env} is not set.")
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
                generated = list(
                    executor.map(
                        lambda target: _openrouter_bgb_question(
                            target,
                            model=api_model,
                            cache_model=cache_model,
                            endpoint=endpoint,
                            api_key=api_key,
                            language=language,
                        ),
                        missing,
                    )
                )
        _append_jsonl(cache_path, [{**question.to_dict(), "language": language} for question in generated])
        for question in generated:
            cache[(question.stable_ref, question.hash, question.source, question.model, question.gold_mode, language)] = question
    return [
        cache[_question_cache_key(target, source=source, model=cache_model, language=language)]
        for target in targets
        if _question_cache_key(target, source=source, model=cache_model, language=language) in cache
    ]


def _openrouter_bgb_question(
    target: EvalTarget,
    *,
    model: str,
    cache_model: str,
    endpoint: str,
    api_key: str,
    language: str,
) -> EvalQuestion:
    region_text = "\n\n".join(f"Region {idx + 1} ({target.stable_refs[idx]}):\n{record.text}" for idx, record in enumerate(target.records))
    if language == "de":
        prompt = f"""Schreibe genau eine natürliche deutschsprachige Suchfrage, die ein Nutzer stellen könnte, wenn die Antwort in der angegebenen BGB-Region steht.

Nicht wortwörtlich aus dem Text abschreiben.
Die Frage soll wie eine echte Sorge oder Recherchefrage klingen.
Antworte mit strengem JSON: {{"query": "..."}}

Gold-Modus: {target.gold_mode}
Text:
{region_text}
"""
        system = "Du schreibst deutsche Held-out-Suchfragen für juristische Retrieval-Evaluation."
    else:
        prompt = f"""Write one natural English search query answerable from the supplied BGB region.

Do not quote an exact sentence.
Return strict JSON: {{"query": "..."}}

Gold mode: {target.gold_mode}
Text:
{region_text}
"""
        system = "You write held-out retrieval evaluation queries for legal search."
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "temperature": 0.45,
        "max_tokens": 140,
    }
    parsed = _openrouter_json(body, endpoint=endpoint, api_key=api_key)
    query = str(parsed.get("query", "")).strip() if parsed else ""
    if not query:
        return _local_bgb_question(target, language=language, model=model)
    return EvalQuestion(
        query=query,
        doc_id=target.primary.doc_id,
        region_id=target.primary.region_id,
        stable_ref=target.stable_refs[0],
        gold_refs=target.stable_refs,
        hash=target.hash,
        source="openrouter",
        model=cache_model,
        gold_mode=target.gold_mode,
    )


def _local_bgb_question(target: EvalTarget, *, language: str, model: str) -> EvalQuestion:
    primary = target.primary
    topic = primary.text.splitlines()[0].replace("§", "Paragraph")
    query = f"Welche Regel gilt für {topic}?" if language == "de" else f"What rule applies to {topic}?"
    return EvalQuestion(
        query=query,
        doc_id=primary.doc_id,
        region_id=primary.region_id,
        stable_ref=target.stable_refs[0],
        gold_refs=target.stable_refs,
        hash=target.hash,
        source="local",
        model=model,
        gold_mode=target.gold_mode,
    )


def _openrouter_json(body: dict[str, object], *, endpoint: str, api_key: str) -> dict[str, object] | None:
    req = request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/b-imenitov/refmark",
            "X-Title": "refmark BGB search pipeline",
        },
        method="POST",
    )
    try:
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
                return None
            parsed = json.loads(content[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _read_language_view_cache(path: Path, *, language: str, model: str) -> dict[tuple[str, str, str, str], RetrievalView]:
    cache: dict[tuple[str, str, str, str], RetrievalView] = {}
    if not path.exists():
        return cache
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row_language = str(row.get("language", language))
        row_model = str(row.get("model", model))
        view = row.get("view", {})
        cache[(str(row["stable_ref"]), str(row["hash"]), row_language, row_model)] = RetrievalView(
            summary=str(view.get("summary", "")),
            questions=[str(item) for item in view.get("questions", [])],
            keywords=[str(item) for item in view.get("keywords", [])],
        )
    return cache


def _read_language_question_cache(path: Path) -> dict[tuple[str, str, str, str, str, str], EvalQuestion]:
    cache: dict[tuple[str, str, str, str, str, str], EvalQuestion] = {}
    if not path.exists():
        return cache
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        language = str(row.get("language", "de"))
        question = EvalQuestion(
            query=str(row["query"]),
            doc_id=str(row["doc_id"]),
            region_id=str(row["region_id"]),
            stable_ref=str(row["stable_ref"]),
            gold_refs=[str(item) for item in row.get("gold_refs", [row["stable_ref"]])],
            hash=str(row["hash"]),
            source=str(row["source"]),
            model=str(row.get("model", row["source"])),
            gold_mode=str(row.get("gold_mode", "single")),
        )
        cache[(question.stable_ref, question.hash, question.source, question.model, question.gold_mode, language)] = question
    return cache


def _question_cache_key(target: EvalTarget, *, source: str, model: str, language: str) -> tuple[str, str, str, str, str, str]:
    return (target.stable_refs[0], target.hash, source, model, target.gold_mode, language)


def _append_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_concern_aliases(path: str | Path | None) -> list[ConcernAlias]:
    if not path:
        return []
    source = Path(path)
    if not source.exists():
        return []
    return load_aliases(source)


def _concern_questions(
    records: list[RegionRecord],
    aliases: list[ConcernAlias],
    *,
    source: str,
    model: str,
) -> list[EvalQuestion]:
    if not aliases:
        return []
    refs_by_prefix = _refs_by_expected_prefix(records)
    output: list[EvalQuestion] = []
    for alias in aliases:
        gold_refs = _gold_refs_for_alias(alias, refs_by_prefix)
        if not gold_refs:
            continue
        gold_hash = _hash_for_refs(records, gold_refs)
        primary_ref = gold_refs[0]
        primary_doc, primary_region = primary_ref.split(":", 1)
        for query in alias.queries:
            output.append(
                EvalQuestion(
                    query=query,
                    doc_id=primary_doc,
                    region_id=primary_region,
                    stable_ref=primary_ref,
                    gold_refs=gold_refs,
                    hash=gold_hash,
                    source=source,
                    model=model,
                    gold_mode="concern",
                )
            )
    return output


def _apply_concern_aliases_to_views(
    records: list[RegionRecord],
    views: dict[tuple[str, str], RetrievalView],
    aliases: list[ConcernAlias],
) -> dict[tuple[str, str], RetrievalView]:
    refs_by_prefix = _refs_by_expected_prefix(records)
    ref_to_terms: dict[str, list[str]] = {}
    for alias in aliases:
        gold_refs = _gold_refs_for_alias(alias, refs_by_prefix)
        # Queries are held-out eval rows. Only aliases are retrieval metadata;
        # injecting eval queries would make concern metrics too easy to game.
        terms = [*alias.aliases]
        for gold_ref in gold_refs:
            ref_to_terms.setdefault(gold_ref, []).extend(terms)

    output = dict(views)
    for record in records:
        stable_ref = f"{record.doc_id}:{record.region_id}"
        terms = _unique(ref_to_terms.get(stable_ref, []))
        if not terms:
            continue
        view = output[(record.doc_id, record.region_id)]
        output[(record.doc_id, record.region_id)] = RetrievalView(
            summary=view.summary,
            questions=_unique([*view.questions, *terms])[:120],
            keywords=_unique([*view.keywords, *_keyword_terms(terms)])[:120],
        )
    return output


def _refs_by_expected_prefix(records: list[RegionRecord]) -> dict[str, list[str]]:
    stable_refs = [f"{record.doc_id}:{record.region_id}" for record in records]
    prefixes: dict[str, list[str]] = {}
    for stable_ref in stable_refs:
        prefixes[stable_ref] = [stable_ref]
        if "_A" in stable_ref:
            article_ref = stable_ref.rsplit("_A", 1)[0]
            prefixes.setdefault(article_ref, []).append(stable_ref)
    return {key: sorted(set(value)) for key, value in prefixes.items()}


def _gold_refs_for_alias(alias: ConcernAlias, refs_by_prefix: dict[str, list[str]]) -> list[str]:
    gold_refs: list[str] = []
    for expected_prefix in alias.expected_prefixes:
        if expected_prefix in refs_by_prefix:
            gold_refs.extend(refs_by_prefix[expected_prefix])
            continue
        gold_refs.extend(
            stable_ref
            for prefix, refs in refs_by_prefix.items()
            if prefix.startswith(expected_prefix)
            for stable_ref in refs
        )
    return sorted(set(gold_refs))


def _hash_for_refs(records: list[RegionRecord], gold_refs: list[str]) -> str:
    hash_by_ref = {f"{record.doc_id}:{record.region_id}": record.hash for record in records}
    return "|".join(hash_by_ref[ref] for ref in gold_refs if ref in hash_by_ref)


def _keyword_terms(values: list[str]) -> list[str]:
    terms: list[str] = []
    for value in values:
        terms.extend(token for token in value.replace("ü", "ue").split() if len(token) > 2)
        terms.extend(token for token in value.split() if len(token) > 2)
    return _unique(terms)


def _unique(values) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = str(value).strip()
        if clean and clean not in seen:
            output.append(clean)
            seen.add(clean)
    return output


def _record_to_bgb_region(record: RegionRecord) -> BgbRegion:
    heading = record.text.splitlines()[0] if record.text else record.region_id
    parts = heading.split(maxsplit=2)
    para = " ".join(parts[:2]) if len(parts) >= 2 else heading
    title = parts[2] if len(parts) >= 3 else ""
    return BgbRegion(
        doc_id=record.doc_id,
        region_id=record.region_id,
        para=para,
        title=title,
        text=record.text,
        source_anchor=str(record.source_path or ""),
        ordinal=record.ordinal,
    )


def _write_portable_index(
    records: list[RegionRecord],
    views: dict[tuple[str, str], RetrievalView],
    path: Path,
    *,
    source: str,
    settings: dict[str, object],
) -> None:
    regions = [
        SearchRegion(
            doc_id=record.doc_id,
            region_id=record.region_id,
            text=record.text,
            hash=record.hash,
            source_path=source + (f"#{record.source_path}" if record.source_path else ""),
            ordinal=record.ordinal,
            prev_region_id=record.prev_region_id,
            next_region_id=record.next_region_id,
            view=views[(record.doc_id, record.region_id)],
        )
        for record in records
    ]
    approx_input_tokens = sum(approx_tokens(record.text) for record in records)
    approx_output_tokens = sum(
        approx_tokens(region.view.summary)
        + sum(approx_tokens(question) for question in region.view.questions)
        + sum(approx_tokens(keyword) for keyword in region.view.keywords)
        for region in regions
    )
    payload = {
        "schema": "refmark.portable_search_index.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_corpus": source,
        "settings": {
            "marker_format": "browser_data_ref",
            "chunker": "bgb-paragraph",
            "include_source_in_index": True,
            **settings,
        },
        "stats": {
            "documents": len({record.doc_id for record in records}),
            "regions": len(records),
            "approx_input_tokens": approx_input_tokens,
            "approx_output_tokens": approx_output_tokens,
            "approx_openrouter_cost_usd_at_mistral_nemo": round(
                (approx_input_tokens / 1_000_000 * 0.01) + (approx_output_tokens / 1_000_000 * 0.03),
                4,
            ),
        },
        "regions": [region.to_dict() for region in regions],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _evaluate(index: PortableBM25Index, questions, *, top_ks: tuple[int, ...], strategy: str = "flat") -> dict[str, object]:
    ranks: list[int | None] = []
    mode_ranks: dict[str, list[int | None]] = {}
    max_k = max(top_ks)
    misses = []
    misses_by_mode: dict[str, list[dict[str, object]]] = {}
    missed_gold_refs: Counter[str] = Counter()
    missed_gold_refs_by_mode: dict[str, Counter[str]] = {}
    wrong_top_refs: Counter[str] = Counter()
    wrong_top_refs_by_mode: dict[str, Counter[str]] = {}
    wrong_top_articles: Counter[str] = Counter()
    wrong_top_articles_by_mode: dict[str, Counter[str]] = {}
    near_article_misses = 0
    near_article_misses_by_mode: Counter[str] = Counter()
    mode_misses: Counter[str] = Counter()
    for question in questions:
        mode = str(question.gold_mode)
        if strategy == "rerank":
            hits = index.search_reranked(question.query, top_k=max_k, candidate_k=50)
        else:
            hits = index.search(question.query, top_k=max_k)
        gold = set(question.gold_refs)
        rank = None
        for hit in hits:
            if hit.stable_ref in gold:
                rank = hit.rank
                break
        ranks.append(rank)
        mode_ranks.setdefault(mode, []).append(rank)
        if rank is None:
            mode_misses[mode] += 1
            missed_gold_refs.update(question.gold_refs)
            missed_gold_refs_by_mode.setdefault(mode, Counter()).update(question.gold_refs)
            if hits:
                wrong_top_refs[hits[0].stable_ref] += 1
                wrong_top_refs_by_mode.setdefault(mode, Counter())[hits[0].stable_ref] += 1
                wrong_top_articles[_article_ref(hits[0].stable_ref)] += 1
                wrong_top_articles_by_mode.setdefault(mode, Counter())[_article_ref(hits[0].stable_ref)] += 1
                gold_articles = {_article_ref(ref) for ref in question.gold_refs}
                if _article_ref(hits[0].stable_ref) in gold_articles:
                    near_article_misses += 1
                    near_article_misses_by_mode[mode] += 1
            miss_row = {
                "query": question.query,
                "gold": question.gold_refs,
                "gold_articles": sorted({_article_ref(ref) for ref in question.gold_refs}),
                "gold_mode": question.gold_mode,
                "top_refs": [hit.stable_ref for hit in hits[:5]],
                "top_articles": [_article_ref(hit.stable_ref) for hit in hits[:5]],
            }
            if len(misses) < 20:
                misses.append(miss_row)
            mode_misses_sample = misses_by_mode.setdefault(mode, [])
            if len(mode_misses_sample) < 10:
                mode_misses_sample.append(miss_row)
    total = max(len(questions), 1)
    reciprocal = sum(1.0 / rank for rank in ranks if rank is not None)
    miss_count = sum(1 for rank in ranks if rank is None)
    return {
        "hit_at_k": {str(k): round(sum(1 for rank in ranks if rank is not None and rank <= k) / total, 4) for k in top_ks},
        "mrr": round(reciprocal / total, 4),
        "candidate_recall_at_max_k": round(sum(1 for rank in ranks if rank is not None) / total, 4),
        "by_gold_mode": {
            mode: _summarize_ranks_by_k(mode_values, top_ks=top_ks)
            for mode, mode_values in sorted(mode_ranks.items())
        },
        "weakness_heatmap": {
            "misses": miss_count,
            "miss_rate": round(miss_count / total, 4),
            "near_article_misses": near_article_misses,
            "near_article_misses_by_mode": dict(sorted(near_article_misses_by_mode.items())),
            "mode_misses": dict(sorted(mode_misses.items())),
            "missed_gold_refs_top": _counter_rows(missed_gold_refs, limit=20),
            "missed_gold_refs_by_mode": {
                mode: _counter_rows(counter, limit=12)
                for mode, counter in sorted(missed_gold_refs_by_mode.items())
            },
            "wrong_top_refs_top": _counter_rows(wrong_top_refs, limit=20),
            "wrong_top_refs_by_mode": {
                mode: _counter_rows(counter, limit=12)
                for mode, counter in sorted(wrong_top_refs_by_mode.items())
            },
            "wrong_top_articles_top": _counter_rows(wrong_top_articles, limit=20),
            "wrong_top_articles_by_mode": {
                mode: _counter_rows(counter, limit=12)
                for mode, counter in sorted(wrong_top_articles_by_mode.items())
            },
        },
        "misses": misses,
        "misses_by_mode": misses_by_mode,
    }


def _summarize_ranks_by_k(ranks: list[int | None], *, top_ks: tuple[int, ...]) -> dict[str, object]:
    total = max(len(ranks), 1)
    reciprocal = sum(1.0 / rank for rank in ranks if rank is not None)
    return {
        "count": len(ranks),
        "hit_at_k": {str(k): round(sum(1 for rank in ranks if rank is not None and rank <= k) / total, 4) for k in top_ks},
        "mrr": round(reciprocal / total, 4),
        "candidate_recall_at_max_k": round(sum(1 for rank in ranks if rank is not None) / total, 4),
    }


def _counter_rows(counter: Counter[str], *, limit: int) -> list[dict[str, object]]:
    return [{"ref": ref, "count": count} for ref, count in counter.most_common(limit)]


def _article_ref(stable_ref: str) -> str:
    doc_id, region_id = stable_ref.split(":", 1)
    if "_A" in region_id:
        region_id = region_id.rsplit("_A", 1)[0]
    return f"{doc_id}:{region_id}"


def _run_concerns(index: PortableBM25Index) -> list[dict[str, object]]:
    rows = []
    for concern in CONCERNS:
        hits = index.search_reranked(concern["query"], top_k=5, candidate_k=50, expand_after=1)
        expected = concern["expected_contains"]
        rows.append(
            {
                "query": concern["query"],
                "expected_contains": expected,
                "pass": any(any(fragment in hit.stable_ref for fragment in expected) for hit in hits),
                "hits": [hit.to_dict() for hit in hits[:5]],
            }
        )
    return rows


def _safe_model(model: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in model).strip("_")


if __name__ == "__main__":
    main()
