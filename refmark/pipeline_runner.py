"""Easy-mode full evidence-retrieval pipeline runner."""

from __future__ import annotations

import concurrent.futures
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable
from urllib import request

from refmark.discovery import (
    DiscoveryManifest,
    build_discovery_context_card,
    discover_corpus,
    load_discovery,
    review_discovery,
    write_discovery,
)
from refmark.pipeline import build_section_map, write_manifest
from refmark.pipeline_config import FullPipelineConfig, ModelTierConfig, load_full_pipeline_config
from refmark.provenance import file_fingerprint
from refmark.question_plan import QuestionPlanItem, build_question_plan, question_plan_to_dict
from refmark.rag_eval import CorpusMap, EvalSuite
from refmark.search_index import (
    OPENROUTER_CHAT_URL,
    approx_tokens,
    build_search_index,
    export_browser_search_index,
    load_search_index,
    local_view,
    map_corpus,
)


QUESTION_PROMPT_VERSION = "refmark.question_generation.v1"


@dataclass(frozen=True)
class PipelineRunSummary:
    schema: str
    created_at: str
    config_hash: str
    corpus_fingerprint: str
    artifacts: dict[str, str]
    stats: dict[str, Any]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_full_pipeline(config_or_path: FullPipelineConfig | str | Path) -> PipelineRunSummary:
    """Run the easy-mode corpus -> searchable evidence index pipeline.

    The runner is intentionally artifact-first and idempotent: unless
    ``artifacts.overwrite`` is true, existing manifest/question/index/report
    files are reused.
    """
    config = load_full_pipeline_config(config_or_path) if isinstance(config_or_path, (str, Path)) else config_or_path
    output_dir = Path(config.artifacts.output_dir)
    cache_dir = Path(config.artifacts.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    artifacts = _artifact_paths(config)
    notes: list[str] = []
    config_hash = _json_hash(config.to_dict())

    records = _load_or_map_records(config, artifacts["manifest"], notes)
    corpus = CorpusMap.from_records(records, revision_id=config.revision_id, metadata={"source_path": config.corpus_path})
    corpus_fingerprint = corpus.fingerprint

    _write_if_needed(
        artifacts["sections"],
        lambda: json.dumps(
            {
                "schema": "refmark.section_map.v1",
                "manifest": str(artifacts["manifest"]),
                "sections": [section.to_dict() for section in build_section_map(records)],
            },
            indent=2,
        ),
        overwrite=config.artifacts.overwrite,
    )

    discovery_stats = _write_discovery_if_needed(config, records, artifacts["discovery"], artifacts["discovery_review"], notes)
    plan_stats = _write_question_plan_if_needed(config, records, artifacts["question_plan"], notes)
    question_stats = _write_questions_if_needed(config, records, artifacts["questions"], notes)
    index_stats = _write_index_if_needed(config, artifacts["index"], notes)
    browser_stats = _write_browser_if_needed(config, artifacts["index"], artifacts["browser_index"])
    eval_stats = _write_eval_if_needed(config, corpus, artifacts["index"], artifacts["questions"], artifacts["eval_report"])

    stats = {
        "regions": len(records),
        "questions": question_stats["questions"],
        "discovery": discovery_stats,
        "question_plan": plan_stats,
        "question_generation": question_stats,
        "index": index_stats,
        "browser_index": browser_stats,
        "eval": eval_stats,
    }
    summary = PipelineRunSummary(
        schema="refmark.full_pipeline_run.v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        config_hash=config_hash,
        corpus_fingerprint=corpus_fingerprint,
        artifacts={name: str(path) for name, path in artifacts.items()},
        stats=stats,
        notes=notes,
    )
    artifacts["summary"].write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    return summary


def _artifact_paths(config: FullPipelineConfig) -> dict[str, Path]:
    output = Path(config.artifacts.output_dir)
    return {
        "manifest": output / "corpus.refmark.jsonl",
        "sections": output / "sections.json",
        "discovery": output / "discovery.json",
        "discovery_review": output / "discovery_review.json",
        "question_plan": output / "question_plan.json",
        "questions": output / "eval_questions.jsonl",
        "index": output / "docs.index.json",
        "browser_index": output / "docs.browser.json",
        "eval_report": output / "eval.json",
        "summary": output / "run_summary.json",
    }


def _load_or_map_records(config: FullPipelineConfig, manifest_path: Path, notes: list[str]):
    if manifest_path.exists() and not config.artifacts.overwrite:
        notes.append(f"Reused manifest: {manifest_path}")
        return CorpusMap.from_manifest(manifest_path).records
    records = map_corpus(
        config.corpus_path,
        marker_format=config.marker_format,
        chunker=_chunker_for_density(config.density),
        tokens_per_chunk=180 if config.density == "coarse" else None,
        lines_per_chunk=1 if config.density == "dense" else None,
        min_words=config.min_words,
    )
    write_manifest(records, manifest_path)
    return records


def _write_discovery_if_needed(config: FullPipelineConfig, records, path: Path, review_path: Path, notes: list[str]) -> dict[str, Any]:
    if path.exists() and not config.artifacts.overwrite:
        notes.append(f"Reused discovery: {path}")
        discovery = load_discovery(path)
    else:
        discovery = discover_corpus(
            records,
            mode=config.discovery.mode,
            source=config.discovery.source,
            model=config.discovery.model,
            endpoint=config.discovery.endpoint,
            api_key_env=config.discovery.api_key_env,
            max_input_tokens=config.discovery.max_input_tokens,
            window_tokens=config.discovery.window_tokens,
            overlap_regions=config.discovery.overlap_regions,
        )
        write_discovery(discovery, path)
    review_issues = _write_discovery_review_if_needed(config, discovery, records, review_path, notes)
    return {
        "mode": discovery.mode,
        "source": discovery.source,
        "model": discovery.model,
        "regions": discovery.regions,
        "windows": len(discovery.windows),
        "clusters": len(discovery.clusters),
        "review_issues": len(review_issues),
    }


def _write_discovery_review_if_needed(
    config: FullPipelineConfig,
    discovery: DiscoveryManifest,
    records,
    path: Path,
    notes: list[str],
) -> list[dict[str, Any]]:
    if not config.discovery.review_enabled:
        return []
    if path.exists() and not config.artifacts.overwrite:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        notes.append(f"Reused discovery review: {path}")
        return list(payload.get("issues", []))
    issues = review_discovery(discovery, records=records, max_issues=config.discovery.max_review_issues)
    payload = {
        "schema": "refmark.discovery_review.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "discovery": str(_artifact_paths(config)["discovery"]),
        "issues": [issue.to_dict() for issue in issues],
        "provenance": {
            "config_hash": _json_hash(config.to_dict()),
            "corpus_path": config.corpus_path,
            "discovery_hash": _json_hash(discovery.to_dict()),
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload["issues"]


def _write_question_plan_if_needed(config: FullPipelineConfig, records, path: Path, notes: list[str]) -> dict[str, Any]:
    if path.exists() and not config.artifacts.overwrite:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        notes.append(f"Reused question plan: {path}")
        return dict(payload.get("summary", {}))
    selected = _sample_records(records, sample_size=config.loop.sample_size, seed=config.loop.seed)
    discovery = load_discovery(_artifact_paths(config)["discovery"])
    plan = build_question_plan(
        discovery,
        selected,
        direct_per_region=config.question_plan.direct_per_region,
        concern_per_region=config.question_plan.concern_per_region,
        adversarial_per_region=config.question_plan.adversarial_per_region,
        include_excluded=config.question_plan.include_excluded,
    )
    payload = question_plan_to_dict(plan)
    payload["provenance"] = {
        "config_hash": _json_hash(config.to_dict()),
        "discovery": str(_artifact_paths(config)["discovery"]),
        "sample_size": config.loop.sample_size,
        "seed": config.loop.seed,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dict(payload["summary"])


def _write_questions_if_needed(config: FullPipelineConfig, records, path: Path, notes: list[str]) -> dict[str, Any]:
    if path.exists() and not config.artifacts.overwrite:
        rows = _read_jsonl(path)
        notes.append(f"Reused eval questions: {path}")
        return {"questions": len(rows), "reused": True, "generated": 0, "input_tokens": 0, "output_tokens": 0, "estimated_usd": 0.0}

    discovery = load_discovery(_artifact_paths(config)["discovery"])
    plan_by_ref = _read_question_plan(_artifact_paths(config)["question_plan"])
    selected = [
        record
        for record in _sample_records(records, sample_size=config.loop.sample_size, seed=config.loop.seed)
        if _stable_ref(record) in plan_by_ref
    ]
    cache_path = Path(config.artifacts.question_cache)
    cache = _read_generation_cache(cache_path)
    generated_rows: list[dict[str, Any]] = []
    cache_updates: list[dict[str, Any]] = []
    stats = {"questions": 0, "reused": 0, "generated": 0, "input_tokens": 0, "output_tokens": 0, "estimated_usd": 0.0}
    provider_ready = _provider_ready(config.question_generation, notes)

    def one(record) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        card = build_discovery_context_card(discovery, record, records=records)
        plan = plan_by_ref[_stable_ref(record)]
        key = _question_cache_key(record, config.question_generation, card=card.to_dict(), plan=[item.to_dict() for item in plan])
        if key in cache:
            return _normalise_cached_questions(cache[key]["questions"], fallback_plan=plan), None
        if provider_ready:
            questions, usage, error = _remote_questions(
                record,
                config.question_generation,
                discovery=discovery,
                context_card=card.to_dict(),
                plan=plan,
            )
            if error:
                questions = _local_questions(record, context_card=card.to_dict(), plan=plan)
                usage["error"] = error
        else:
            questions = _local_questions(record, context_card=card.to_dict(), plan=plan)
            usage = {"input_tokens": 0, "output_tokens": 0, "estimated_usd": 0.0, "source": "local"}
        update = {
            "cache_key": key,
            "stable_ref": _stable_ref(record),
            "hash": record.hash,
            "prompt_version": QUESTION_PROMPT_VERSION,
            "model": config.question_generation.model if provider_ready else "local",
            "context_card": card.to_dict(),
            "question_plan": [item.to_dict() for item in plan],
            "questions": questions,
            "usage": usage,
        }
        return questions, update

    workers = max(1, config.question_generation.concurrency if provider_ready else 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for record, (questions, update) in zip(selected, executor.map(one, selected), strict=False):
            if update is None:
                stats["reused"] += 1
            else:
                stats["generated"] += 1
                cache_updates.append(update)
                usage = update["usage"]
                stats["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
                stats["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
                stats["estimated_usd"] += float(usage.get("estimated_usd", 0.0) or 0.0)
            for question in questions:
                query_text = str(question.get("query", "")).strip()
                if not query_text:
                    continue
                query_style = str(question.get("query_style") or "unspecified")
                generated_rows.append(
                    {
                        "query": query_text,
                        "gold_refs": [_stable_ref(record)],
                        "source_hashes": {_stable_ref(record): record.hash},
                        "metadata": {
                            "source": "refmark-full-pipeline",
                            "style": query_style,
                            "query_style": query_style,
                            "question_prompt_version": QUESTION_PROMPT_VERSION,
                            "discovery_context_ref": _stable_ref(record),
                            "question_plan": question.get("plan"),
                        },
                    }
                )
    _append_jsonl(cache_path, cache_updates)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in generated_rows) + "\n", encoding="utf-8")
    stats["questions"] = len(generated_rows)
    _check_budget(config, stats)
    return stats


def _write_index_if_needed(config: FullPipelineConfig, index_path: Path, notes: list[str]) -> dict[str, Any]:
    if index_path.exists() and not config.artifacts.overwrite:
        payload = json.loads(index_path.read_text(encoding="utf-8-sig"))
        notes.append(f"Reused index: {index_path}")
        return payload.get("stats", {})
    source = "openrouter" if config.retrieval_views.enabled and _provider_ready(config.retrieval_views, notes) else "local"
    payload = build_search_index(
        config.corpus_path,
        index_path,
        source=source,
        model=config.retrieval_views.model,
        endpoint=config.retrieval_views.endpoint,
        api_key_env=config.retrieval_views.api_key_env,
        marker_format=config.marker_format,
        chunker=_chunker_for_density(config.density),
        min_words=config.min_words,
        concurrency=config.retrieval_views.concurrency,
        view_cache_path=config.artifacts.view_cache,
    )
    return payload.get("stats", {})


def _write_browser_if_needed(config: FullPipelineConfig, index_path: Path, browser_path: Path) -> dict[str, Any]:
    if browser_path.exists() and not config.artifacts.overwrite:
        return json.loads(browser_path.read_text(encoding="utf-8-sig")).get("stats", {})
    payload = export_browser_search_index(index_path, browser_path)
    return payload.get("stats", {})


def _write_eval_if_needed(config: FullPipelineConfig, corpus: CorpusMap, index_path: Path, questions_path: Path, report_path: Path) -> dict[str, Any]:
    if report_path.exists() and not config.artifacts.overwrite:
        return json.loads(report_path.read_text(encoding="utf-8-sig")).get("metrics", {})
    index = load_search_index(index_path)
    rows = _read_jsonl(questions_path)
    suite = EvalSuite.from_rows(rows, corpus=corpus).with_source_hashes()
    run = suite.evaluate(lambda query: index.search_reranked(query, top_k=config.loop.top_k), name="rerank", k=config.loop.top_k)
    payload = {
        "schema": "refmark.full_pipeline_eval.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "index": str(index_path),
        "examples": str(questions_path),
        "metrics": run.metrics,
        "diagnostics": run.diagnostics,
        "validation": suite.validate_refs(),
        "stale_examples": [item.to_dict() for item in suite.stale_examples()],
        "results": [item.to_dict() for item in run.examples],
        "provenance": {
            "index": file_fingerprint(index_path),
            "questions": file_fingerprint(questions_path),
            "corpus_fingerprint": corpus.fingerprint,
            "config_hash": _json_hash(config.to_dict()),
            "top_k": config.loop.top_k,
            "seed": config.loop.seed,
        },
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary_metrics = dict(run.metrics)
    summary_metrics["by_query_style"] = run.diagnostics.get("by_query_style", {})
    summary_metrics["query_style_gap"] = run.diagnostics.get("query_style_gap", {})
    return summary_metrics


def _remote_questions(
    record,
    tier: ModelTierConfig,
    *,
    discovery: DiscoveryManifest,
    context_card: dict[str, Any],
    plan: list[QuestionPlanItem],
) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
    prompt = f"""Generate natural search queries for this evidence region.

Return strict JSON only:
{{"questions":[{{"query":"...","query_style":"direct"}},{{"query":"...","query_style":"concern"}},{{"query":"...","query_style":"adversarial"}}]}}

Use the discovery context to vary question style and avoid generic phrasing.
Prefer questions that a real target user of this corpus might ask. Do not cite
the ref id in the question text. Keep every question answerable from the target
region; if the context card says the region is navigation/boilerplate, generate
only navigation-specific questions.

Generate exactly the requested plan styles and counts. Style meanings:
- direct: straightforward lookup using ordinary source terminology.
- concern: user problem, goal, or symptom wording.
- adversarial: valid paraphrase with lower lexical overlap, still unambiguous.

Ref: {_stable_ref(record)}
Discovery context:
{json.dumps(context_card, ensure_ascii=False)}

Question plan:
{json.dumps([item.to_dict() for item in plan], ensure_ascii=False)}

Text:
{record.text}
"""
    input_tokens = approx_tokens(prompt)
    body = {
        "model": tier.model,
        "messages": [
            {"role": "system", "content": "You generate concise retrieval evaluation questions."},
            {"role": "user", "content": prompt},
        ],
        "temperature": tier.temperature,
        "max_tokens": tier.max_tokens,
    }
    payload = json.dumps(body).encode("utf-8")
    error: str | None = None
    for attempt in range(max(1, tier.retries + 1)):
        try:
            req = request.Request(
                tier.endpoint or OPENROUTER_CHAT_URL,
                data=payload,
                headers={
                    "Authorization": f"Bearer {os.environ[tier.api_key_env]}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/b-imenitov/refmark",
                    "X-Title": "refmark full pipeline",
                },
                method="POST",
            )
            with request.urlopen(req, timeout=tier.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            content = response_payload["choices"][0]["message"]["content"]
            parsed = _parse_json_object(content)
            questions = _normalise_cached_questions(parsed.get("questions", []), fallback_plan=plan)
            if questions:
                output_tokens = approx_tokens(content)
                return questions[: sum(item.count for item in plan)], _usage(input_tokens, output_tokens, tier, source="remote"), None
        except Exception as exc:
            error = str(exc)
            time.sleep(min(2 ** attempt, 8))
    output_tokens = 0
    return [], _usage(input_tokens, output_tokens, tier, source="remote"), error or "empty response"


def _local_questions(
    record,
    *,
    context_card: dict[str, Any] | None = None,
    plan: list[QuestionPlanItem] | None = None,
) -> list[dict[str, Any]]:
    view = local_view(record.text, questions_per_region=3, keywords_per_region=6)
    if not context_card:
        queries = view.questions or [f"What does {_stable_ref(record)} explain?"]
        return [{"query": query, "query_style": "direct", "plan": None} for query in queries]
    terms = [str(item) for item in context_card.get("terms", []) if str(item).strip()]
    families = [str(item) for item in context_card.get("query_families", []) if str(item).strip()]
    roles = set(str(item) for item in context_card.get("roles", []))
    topic = ", ".join(terms[:3]) if terms else (context_card.get("region_summary") or _stable_ref(record))
    planned = plan or [
        QuestionPlanItem(stable_ref=_stable_ref(record), query_style="direct", terms=terms, roles=list(roles)),
        QuestionPlanItem(stable_ref=_stable_ref(record), query_style="concern", terms=terms, roles=list(roles)),
        QuestionPlanItem(stable_ref=_stable_ref(record), query_style="adversarial", terms=terms, roles=list(roles)),
    ]
    output: list[dict[str, Any]] = []
    for item in planned:
        for query in _local_queries_for_style(item.query_style, topic, terms, roles, families, view.questions)[: item.count]:
            output.append({"query": query, "query_style": item.query_style, "plan": item.to_dict()})
    return output or [{"query": f"What does {_stable_ref(record)} explain?", "query_style": "direct", "plan": None}]


def _local_queries_for_style(
    style: str,
    topic: str,
    terms: list[str],
    roles: set[str],
    families: list[str],
    fallback_questions: list[str],
) -> list[str]:
    if style == "direct":
        questions = [f"What does this section explain about {topic}?"]
        if "definition" in roles:
            questions.append(f"What does {terms[0] if terms else 'this term'} mean in this corpus?")
        elif "obligation" in roles:
            questions.append(f"What is required for {topic}?")
        else:
            questions.extend(fallback_questions[:1])
        return _dedupe(questions)
    if style == "concern":
        if families:
            concern = families[0].replace("questions about ", "")
            return [f"Where should I look for guidance when I need help with {concern}?"]
        return [f"Where should I look if I have a practical problem involving {topic}?"]
    if style == "adversarial":
        return ["Which part of the docs covers the underlying rule or behavior for this topic?"]
    return [f"Where should I look for guidance on {topic}?"]


def _usage(input_tokens: int, output_tokens: int, tier: ModelTierConfig, *, source: str) -> dict[str, Any]:
    input_cost = _cost(input_tokens, tier.input_price_per_million)
    output_cost = _cost(output_tokens, tier.output_price_per_million)
    return {
        "source": source,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_usd": (input_cost or 0.0) + (output_cost or 0.0),
    }


def _provider_ready(tier: ModelTierConfig, notes: list[str]) -> bool:
    if not tier.enabled or tier.provider != "openrouter":
        return False
    if not os.getenv(tier.api_key_env):
        notes.append(f"{tier.api_key_env} is not set; using local fallback for {tier.model}.")
        return False
    return True


def _question_cache_key(
    record,
    tier: ModelTierConfig,
    *,
    card: dict[str, Any] | None = None,
    plan: list[dict[str, Any]] | None = None,
) -> str:
    payload = {
        "prompt_version": QUESTION_PROMPT_VERSION,
        "stable_ref": _stable_ref(record),
        "hash": record.hash,
        "discovery_context_hash": _json_hash(card or {}),
        "question_plan_hash": _json_hash(plan or []),
        "provider": tier.provider,
        "model": tier.model,
        "temperature": tier.temperature,
        "max_tokens": tier.max_tokens,
    }
    return _json_hash(payload)


def _read_question_plan(path: Path) -> dict[str, list[QuestionPlanItem]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    grouped: dict[str, list[QuestionPlanItem]] = {}
    for item in payload.get("items", []):
        plan_item = QuestionPlanItem(
            stable_ref=str(item["stable_ref"]),
            query_style=str(item["query_style"]),
            count=int(item.get("count", 1)),
            gold_refs=[str(ref) for ref in item.get("gold_refs", [])],
            roles=[str(role) for role in item.get("roles", [])],
            terms=[str(term) for term in item.get("terms", [])],
            guidance=[str(value) for value in item.get("guidance", [])],
        )
        grouped.setdefault(plan_item.stable_ref, []).append(plan_item)
    return grouped


def _normalise_cached_questions(
    raw_questions: Any,
    *,
    fallback_plan: list[QuestionPlanItem] | None = None,
) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    if not isinstance(raw_questions, list):
        return questions
    fallback_styles = _expanded_plan_styles(fallback_plan)
    for index, item in enumerate(raw_questions):
        if isinstance(item, dict):
            query = str(item.get("query", "")).strip()
            if query:
                style = str(item.get("query_style") or item.get("style") or "").strip()
                fallback_style = fallback_styles[min(index, len(fallback_styles) - 1)] if fallback_styles else "unspecified"
                questions.append(
                    {
                        "query": query,
                        "query_style": style or fallback_style,
                        "plan": item.get("plan"),
                    }
                )
        else:
            query = str(item).strip()
            if query:
                style = fallback_styles[min(index, len(fallback_styles) - 1)] if fallback_styles else ("direct", "concern", "adversarial")[min(index, 2)]
                questions.append({"query": query, "query_style": style, "plan": None})
    return questions


def _expanded_plan_styles(plan: list[QuestionPlanItem] | None) -> list[str]:
    styles: list[str] = []
    for item in plan or []:
        styles.extend([item.query_style] * max(item.count, 1))
    return styles


def _sample_records(records, *, sample_size: int, seed: int):
    candidates = [record for record in records if not record.text.strip().startswith("#")]
    rng = random.Random(seed)
    if sample_size and len(candidates) > sample_size:
        candidates = rng.sample(candidates, sample_size)
    return sorted(candidates, key=lambda record: (record.doc_id, record.ordinal))


def _read_generation_cache(path: Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path) if path.exists() else []
    return {str(row["cache_key"]): row for row in rows if "cache_key" in row}


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    rows = []
    for line in source.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    items = list(rows)
    if not items:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in items:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_if_needed(path: Path, build_text, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_text(), encoding="utf-8")


def _check_budget(config: FullPipelineConfig, stats: dict[str, Any]) -> None:
    budget = config.budget
    if budget.max_input_tokens is not None and stats["input_tokens"] > budget.max_input_tokens:
        raise RuntimeError("Question generation input token budget exceeded.")
    if budget.max_output_tokens is not None and stats["output_tokens"] > budget.max_output_tokens:
        raise RuntimeError("Question generation output token budget exceeded.")
    if budget.max_estimated_usd is not None and stats["estimated_usd"] > budget.max_estimated_usd:
        raise RuntimeError("Question generation estimated cost budget exceeded.")


def _cost(tokens: int, price_per_million: float | None) -> float | None:
    if price_per_million is None:
        return None
    return tokens / 1_000_000 * price_per_million


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _stable_ref(record) -> str:
    return f"{record.doc_id}:{record.region_id}"


def _chunker_for_density(density: str) -> str:
    return {"dense": "line", "coarse": "token", "code": "hybrid"}.get(density, "paragraph")


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.removeprefix("json").strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end >= start:
        stripped = stripped[start : end + 1]
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object.")
    return parsed


def _dedupe(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output
