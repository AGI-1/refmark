"""Evaluate portable Refmark search indexes on real document corpora."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Iterable
from urllib import request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from refmark.pipeline import RegionRecord  # noqa: E402
from refmark.discovery import DiscoveryManifest, discovery_excluded_refs, discovery_terms_for_refs, load_discovery  # noqa: E402
from refmark.metrics import score_ref_range, summarize_scores  # noqa: E402
from refmark.search_index import (  # noqa: E402
    OPENROUTER_CHAT_URL,
    PortableBM25Index,
    RetrievalView,
    SearchRegion,
    approx_tokens,
    generate_views,
    local_view,
    map_corpus,
    tokenize,
)


@dataclass(frozen=True)
class EvalQuestion:
    query: str
    doc_id: str
    region_id: str
    stable_ref: str
    gold_refs: list[str]
    hash: str
    source: str
    model: str
    gold_mode: str = "single"

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "doc_id": self.doc_id,
            "region_id": self.region_id,
            "stable_ref": self.stable_ref,
            "gold_refs": self.gold_refs,
            "hash": self.hash,
            "source": self.source,
            "model": self.model,
            "gold_mode": self.gold_mode,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate real-corpus Refmark search quality across corpus sizes.")
    parser.add_argument("corpus", help="Input file or directory.")
    parser.add_argument("--output", default="examples/portable_search_index/output/real_corpus_eval.json")
    parser.add_argument("--cache", default="examples/portable_search_index/output/eval_question_cache.jsonl")
    parser.add_argument("--budgets", default="50000,100000,250000,500000,1000000")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--question-source", choices=["local", "openrouter"], default="local")
    parser.add_argument("--index-view-source", choices=["raw", "local", "openrouter"], default="local")
    parser.add_argument("--model", default="mistralai/mistral-nemo")
    parser.add_argument("--endpoint", default=OPENROUTER_CHAT_URL)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--questions-per-region", type=int, default=4)
    parser.add_argument("--keywords-per-region", type=int, default=8)
    parser.add_argument("--chunker", default="paragraph")
    parser.add_argument("--tokens-per-chunk", type=int, default=None)
    parser.add_argument("--min-words", type=int, default=8)
    parser.add_argument(
        "--exclude-glob",
        action="append",
        default=[],
        help="Glob pattern for source paths to skip. May be passed multiple times.",
    )
    parser.add_argument("--top-ks", default="1,3,5,10")
    parser.add_argument("--expand-after", type=int, default=0)
    parser.add_argument("--view-cache", default="examples/portable_search_index/output/view_cache.jsonl")
    parser.add_argument("--strategies", default="flat,hierarchical,rerank")
    parser.add_argument("--doc-top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--gold-mode", choices=["single", "adjacent", "disjoint", "mixed"], default="single")
    parser.add_argument("--discovery", default=None, help="Optional discovery JSON to guide question generation and exclusions.")
    parser.add_argument("--exclude-discovery-roles", action="store_true", help="Drop discovery refs marked exclude_from_training/navigation/boilerplate.")
    parser.add_argument("--learned-reranker-train-fraction", type=float, default=0.5)
    parser.add_argument("--learned-reranker-epochs", type=int, default=8)
    args = parser.parse_args()

    budgets = [int(part) for part in args.budgets.split(",") if part.strip()]
    top_ks = tuple(int(part) for part in args.top_ks.split(",") if part.strip())
    strategies = tuple(part.strip() for part in args.strategies.split(",") if part.strip())
    records = map_corpus(
        args.corpus,
        chunker=args.chunker,
        tokens_per_chunk=args.tokens_per_chunk,
        min_words=args.min_words,
        exclude_globs=args.exclude_glob,
    )
    discovery = load_discovery(args.discovery) if args.discovery else None
    if discovery is not None and args.exclude_discovery_roles:
        excluded = discovery_excluded_refs(discovery)
        records = [record for record in records if f"{record.doc_id}:{record.region_id}" not in excluded]
    report = {
        "corpus": args.corpus,
        "settings": {
            "budgets": budgets,
            "sample_size": args.sample_size,
            "seed": args.seed,
            "question_source": args.question_source,
            "index_view_source": args.index_view_source,
            "model": args.model,
            "chunker": args.chunker,
            "tokens_per_chunk": args.tokens_per_chunk,
            "min_words": args.min_words,
            "exclude_glob": args.exclude_glob,
            "top_ks": top_ks,
            "expand_after": args.expand_after,
            "strategies": strategies,
            "doc_top_k": args.doc_top_k,
            "candidate_k": args.candidate_k,
            "gold_mode": args.gold_mode,
            "discovery": args.discovery,
            "exclude_discovery_roles": args.exclude_discovery_roles,
            "learned_reranker_train_fraction": args.learned_reranker_train_fraction,
        },
        "total_regions": len(records),
        "total_approx_tokens": sum(approx_tokens(record.text) for record in records),
        "evidence": {
            "question_cache": args.cache,
            "question_cache_sha256_before": _sha256_file(Path(args.cache)),
            "view_cache": args.view_cache,
            "view_cache_sha256_before": _sha256_file(Path(args.view_cache)),
        },
        "budget_reports": [],
    }

    for budget in budgets:
        budget_records = _records_for_budget(records, budget)
        if not budget_records:
            continue
        eval_targets = _sample_targets(budget_records, limit=args.sample_size, seed=args.seed + budget, gold_mode=args.gold_mode)
        questions = _load_or_generate_questions(
            eval_targets,
            source=args.question_source,
            model=args.model,
            endpoint=args.endpoint,
            api_key_env=args.api_key_env,
            cache_path=Path(args.cache),
            concurrency=args.concurrency,
            discovery=discovery,
        )
        variants = {
            "raw_region_bm25": _build_index(budget_records, view_source="raw", args=args),
        }
        if args.index_view_source != "raw":
            variants[f"{args.index_view_source}_enriched_bm25"] = _build_index(
                budget_records,
                view_source=args.index_view_source,
                args=args,
            )
        variant_reports = {}
        for name, index in variants.items():
            learned_model = None
            eval_questions = questions
            if "learned-rerank" in strategies:
                train_questions, eval_questions = _split_questions(
                    questions,
                    fraction=args.learned_reranker_train_fraction,
                    seed=args.seed + budget + 101,
                )
                learned_model = _train_learned_reranker(
                    index,
                    train_questions,
                    candidate_k=args.candidate_k,
                    epochs=args.learned_reranker_epochs,
                )
            for strategy in strategies:
                variant_reports[f"{name}_{strategy}"] = _evaluate(
                    index,
                    eval_questions if strategy == "learned-rerank" else questions,
                    top_ks=top_ks,
                    expand_after=args.expand_after,
                    strategy=strategy,
                    doc_top_k=args.doc_top_k,
                    candidate_k=args.candidate_k,
                    learned_model=learned_model,
                )
        budget_report = {
            "token_budget": budget,
            "actual_tokens": sum(approx_tokens(record.text) for record in budget_records),
            "documents": len({record.doc_id for record in budget_records}),
            "regions": len(budget_records),
            "eval_questions": len(questions),
            "reports": variant_reports,
        }
        report["budget_reports"].append(budget_report)
        print(json.dumps(budget_report, indent=2))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report["evidence"]["question_cache_sha256_after"] = _sha256_file(Path(args.cache))
    report["evidence"]["view_cache_sha256_after"] = _sha256_file(Path(args.view_cache))
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote real-corpus evaluation to {output}")


def _records_for_budget(records: list[RegionRecord], budget: int) -> list[RegionRecord]:
    selected: list[RegionRecord] = []
    total = 0
    for record in records:
        cost = approx_tokens(record.text)
        if selected and total + cost > budget:
            break
        selected.append(record)
        total += cost
    return selected


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class EvalTarget:
    records: list[RegionRecord]
    gold_mode: str

    @property
    def primary(self) -> RegionRecord:
        return self.records[0]

    @property
    def stable_refs(self) -> list[str]:
        return [f"{record.doc_id}:{record.region_id}" for record in self.records]

    @property
    def hash(self) -> str:
        return "|".join(record.hash for record in self.records)


def _sample_targets(records: list[RegionRecord], *, limit: int, seed: int, gold_mode: str) -> list[EvalTarget]:
    candidates = _candidate_targets(records, gold_mode=gold_mode)
    if len(candidates) <= limit:
        return candidates
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(candidates)), limit))
    return [candidates[index] for index in indices]


def _candidate_targets(records: list[RegionRecord], *, gold_mode: str) -> list[EvalTarget]:
    by_doc: dict[str, list[RegionRecord]] = {}
    for record in records:
        by_doc.setdefault(record.doc_id, []).append(record)
    for doc_records in by_doc.values():
        doc_records.sort(key=lambda item: item.ordinal)

    targets: list[EvalTarget] = []
    if gold_mode in {"single", "mixed"}:
        targets.extend(EvalTarget([record], "single") for record in records)
    if gold_mode in {"adjacent", "mixed"}:
        for doc_records in by_doc.values():
            for index in range(len(doc_records) - 1):
                targets.append(EvalTarget([doc_records[index], doc_records[index + 1]], "adjacent"))
    if gold_mode in {"disjoint", "mixed"}:
        for doc_records in by_doc.values():
            for index in range(0, max(len(doc_records) - 4, 0), 5):
                targets.append(EvalTarget([doc_records[index], doc_records[index + 3]], "disjoint"))
    return targets


def _build_index(records: list[RegionRecord], *, view_source: str, args: argparse.Namespace) -> PortableBM25Index:
    if view_source == "raw":
        views = {
            (record.doc_id, record.region_id): RetrievalView(summary="", questions=[], keywords=[])
            for record in records
        }
    elif view_source == "local":
        views = {
            (record.doc_id, record.region_id): local_view(
                record.text,
                questions_per_region=args.questions_per_region,
                keywords_per_region=args.keywords_per_region,
            )
            for record in records
        }
    else:
        views = generate_views(
            records,
            source="openrouter",
            model=args.model,
            endpoint=args.endpoint,
            api_key_env=args.api_key_env,
            questions_per_region=args.questions_per_region,
            keywords_per_region=args.keywords_per_region,
            concurrency=args.concurrency,
            sleep=0.0,
            cache_path=args.view_cache,
        )
    regions = [
        SearchRegion(
            doc_id=record.doc_id,
            region_id=record.region_id,
            text=record.text,
            hash=record.hash,
            source_path=record.source_path,
            ordinal=record.ordinal,
            prev_region_id=record.prev_region_id,
            next_region_id=record.next_region_id,
            view=views[(record.doc_id, record.region_id)],
        )
        for record in records
    ]
    return PortableBM25Index(regions)


def _load_or_generate_questions(
    targets: list[EvalTarget],
    *,
    source: str,
    model: str,
    endpoint: str,
    api_key_env: str,
    cache_path: Path,
    concurrency: int,
    discovery: DiscoveryManifest | None = None,
) -> list[EvalQuestion]:
    cache = _read_question_cache(cache_path)
    missing = [target for target in targets if _cache_key(target, source=source, model=model) not in cache]
    if missing:
        generated = _generate_questions(
            missing,
            source=source,
            model=model,
            endpoint=endpoint,
            api_key_env=api_key_env,
            concurrency=concurrency,
            discovery=discovery,
        )
        _append_question_cache(cache_path, generated)
        for question in generated:
            cache[(question.stable_ref, question.hash, question.source, question.model, question.gold_mode)] = question
    return [cache[_cache_key(target, source=source, model=model)] for target in targets if _cache_key(target, source=source, model=model) in cache]


def _generate_questions(
    targets: list[EvalTarget],
    *,
    source: str,
    model: str,
    endpoint: str,
    api_key_env: str,
    concurrency: int,
    discovery: DiscoveryManifest | None = None,
) -> list[EvalQuestion]:
    if source == "local":
        return [
            EvalQuestion(
                query=_local_eval_question(target, discovery=discovery),
                doc_id=target.primary.doc_id,
                region_id=target.primary.region_id,
                stable_ref=target.stable_refs[0],
                gold_refs=target.stable_refs,
                hash=target.hash,
                source=source,
                model=source,
                gold_mode=target.gold_mode,
            )
            for target in targets
        ]

    import os
    import concurrent.futures

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set.")

    def one(target: EvalTarget) -> EvalQuestion:
        query = _openrouter_eval_question(
            target,
            model=model,
            endpoint=endpoint,
            api_key=api_key,
        )
        return EvalQuestion(
            query=query,
            doc_id=target.primary.doc_id,
            region_id=target.primary.region_id,
            stable_ref=target.stable_refs[0],
            gold_refs=target.stable_refs,
            hash=target.hash,
            source=source,
            model=model,
            gold_mode=target.gold_mode,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        return list(executor.map(one, targets))


def _read_question_cache(path: Path) -> dict[tuple[str, str, str, str, str], EvalQuestion]:
    if not path.exists():
        return {}
    cache: dict[tuple[str, str, str, str, str], EvalQuestion] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
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
        cache[(question.stable_ref, question.hash, question.source, question.model, question.gold_mode)] = question
    return cache


def _append_question_cache(path: Path, questions: list[EvalQuestion]) -> None:
    if not questions:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for question in questions:
            handle.write(json.dumps(question.to_dict(), ensure_ascii=False) + "\n")


def _cache_key(target: EvalTarget, *, source: str, model: str) -> tuple[str, str, str, str, str]:
    return (target.stable_refs[0], target.hash, source, model if source == "openrouter" else source, target.gold_mode)


def _local_eval_question(target: EvalTarget, *, discovery: DiscoveryManifest | None = None) -> str:
    combined = "\n".join(record.text for record in target.records)
    view = local_view(combined, questions_per_region=1)
    discovery_terms = discovery_terms_for_refs(discovery, target.stable_refs, limit=5) if discovery else []
    keywords = _question_terms([*discovery_terms, *view.keywords])
    topic = ", ".join(keywords[:3]) if keywords else target.primary.region_id
    if target.gold_mode == "adjacent":
        return f"Which adjacent evidence regions together define or qualify {topic}?"
    if target.gold_mode == "disjoint":
        return f"Which separate evidence regions must be combined to resolve {topic}?"
    role = _question_role(target.primary.text)
    return f"Which evidence region gives the {role} for {topic}?"


def _question_terms(values: Iterable[str]) -> list[str]:
    stop = {
        "about",
        "also",
        "does",
        "find",
        "from",
        "have",
        "include",
        "other",
        "osha",
        "passage",
        "rule",
        "section",
        "shall",
        "standard",
        "support",
        "that",
        "this",
        "under",
        "with",
    }
    terms: list[str] = []
    seen: set[str] = set()
    for value in values:
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", value.lower()):
            if len(token) <= 3 or token in stop or token in seen:
                continue
            seen.add(token)
            terms.append(token)
    return terms


def _question_role(text: str) -> str:
    lower = text.lower()
    if " means " in lower:
        return "definition"
    if any(word in lower for word in ["shall", "must", "required"]):
        return "requirement"
    if any(word in lower for word in ["except", "unless", "exempt"]):
        return "exception"
    return "supporting source"


def _openrouter_eval_question(target: EvalTarget, *, model: str, endpoint: str, api_key: str) -> str:
    region_text = "\n\n".join(
        f"Region {idx + 1} ({target.stable_refs[idx]}):\n{record.text}"
        for idx, record in enumerate(target.records)
    )
    if target.gold_mode == "single":
        instruction = "The query should be answerable from this one region."
    elif target.gold_mode == "adjacent":
        instruction = "The query should require the adjacent regions together, not just one isolated sentence."
    else:
        instruction = "The query should require both separate regions together, even if this is difficult for search."
    prompt = f"""Write one natural search query that a user might ask if the answer is in the supplied source region(s).

{instruction}
Do not quote an exact sentence.
Return strict JSON: {{"query": "..."}}

Gold mode: {target.gold_mode}
Text:
{region_text}
"""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You write held-out retrieval evaluation queries for documentation search."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.5,
        "max_tokens": 120,
    }
    req = request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/b-imenitov/refmark",
            "X-Title": "refmark real-corpus eval",
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
            return _local_eval_question(target)
        try:
            parsed = json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            return _local_eval_question(target)
    return str(parsed.get("query", "")).strip() or _local_eval_question(target)


def _evaluate(
    index: PortableBM25Index,
    questions: list[EvalQuestion],
    *,
    top_ks: tuple[int, ...],
    expand_after: int,
    strategy: str,
    doc_top_k: int,
    candidate_k: int,
    learned_model: list[float] | None = None,
) -> dict[str, object]:
    max_k = max(top_ks)
    hits = Counter({k: 0 for k in top_ks})
    doc_hits = Counter({k: 0 for k in top_ks})
    context_hits = Counter({k: 0 for k in top_ks})
    parent_hits = Counter({k: 0 for k in top_ks})
    undercite = Counter({k: 0 for k in top_ks})
    overcite_refs = Counter({k: 0 for k in top_ks})
    range_scores = {k: [] for k in top_ks}
    mode_stats: dict[str, dict[str, object]] = {}
    reciprocal_sum = 0.0
    misses: list[dict[str, object]] = []
    for question in questions:
        results = _search(
            index,
            question.query,
            strategy=strategy,
            top_k=max_k,
            expand_after=expand_after,
            doc_top_k=doc_top_k,
            candidate_k=candidate_k,
            learned_model=learned_model,
        )
        rank = None
        doc_rank = None
        context_rank = None
        gold_refs = set(question.gold_refs)
        for idx, hit in enumerate(results, start=1):
            if hit.stable_ref in gold_refs and rank is None:
                rank = idx
            if hit.doc_id == question.doc_id and doc_rank is None:
                doc_rank = idx
            if gold_refs.issubset(set(hit.context_refs)) and context_rank is None:
                context_rank = idx
        if rank is not None:
            reciprocal_sum += 1.0 / rank
        else:
            misses.append(
                {
                    "query": question.query,
                    "gold": question.gold_refs,
                    "top_refs": [hit.stable_ref for hit in results[:3]],
                }
            )
        for k in top_ks:
            selected_refs = [hit.stable_ref for hit in results[:k]]
            selected_context_refs = sorted({ref for hit in results[:k] for ref in hit.context_refs})
            range_score = score_ref_range(selected_context_refs, question.gold_refs)
            range_scores[k].append(range_score)
            if rank is not None and rank <= k:
                hits[k] += 1
            if doc_rank is not None and doc_rank <= k:
                doc_hits[k] += 1
                parent_hits[k] += 1
            if context_rank is not None and context_rank <= k:
                context_hits[k] += 1
            else:
                undercite[k] += 1
            overcite_refs[k] += max(0, len(selected_context_refs) - len(set(question.gold_refs)))
            _update_mode_stats(
                mode_stats,
                question.gold_mode,
                k=k,
                hit=rank is not None and rank <= k,
                context_hit=context_rank is not None and context_rank <= k,
                undercite=context_rank is None or context_rank > k,
                range_score=range_score,
                extra_context_refs=max(0, len(selected_context_refs) - len(set(question.gold_refs))),
            )
    total = max(len(questions), 1)
    return {
        "hit_at_k": {str(k): round(hits[k] / total, 4) for k in top_ks},
        "context_hit_at_k": {str(k): round(context_hits[k] / total, 4) for k in top_ks},
        "parent_hit_at_k": {str(k): round(parent_hits[k] / total, 4) for k in top_ks},
        "doc_hit_at_k": {str(k): round(doc_hits[k] / total, 4) for k in top_ks},
        "undercite_at_k": {str(k): round(undercite[k] / total, 4) for k in top_ks},
        "avg_extra_context_refs_at_k": {str(k): round(overcite_refs[k] / total, 2) for k in top_ks},
        "range_score_at_k": {str(k): summarize_scores(range_scores[k]) for k in top_ks},
        "by_gold_mode": _summarize_mode_stats(mode_stats),
        "mrr": round(reciprocal_sum / total, 4),
        "sample_misses": misses[:8],
    }


def _update_mode_stats(
    mode_stats: dict[str, dict[str, object]],
    mode: str,
    *,
    k: int,
    hit: bool,
    context_hit: bool,
    undercite: bool,
    range_score,
    extra_context_refs: int,
) -> None:
    stats = mode_stats.setdefault(
        mode,
        {
            "counts": Counter(),
            "hit": Counter(),
            "context_hit": Counter(),
            "undercite": Counter(),
            "extra_context_refs": Counter(),
            "range_scores": {},
        },
    )
    stats["counts"][k] += 1
    stats["hit"][k] += int(hit)
    stats["context_hit"][k] += int(context_hit)
    stats["undercite"][k] += int(undercite)
    stats["extra_context_refs"][k] += extra_context_refs
    stats["range_scores"].setdefault(k, []).append(range_score)


def _summarize_mode_stats(mode_stats: dict[str, dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for mode, stats in sorted(mode_stats.items()):
        counts: Counter = stats["counts"]
        summary[mode] = {
            "count": max(counts.values(), default=0),
            "hit_at_k": {
                str(k): round(stats["hit"][k] / max(count, 1), 4)
                for k, count in sorted(counts.items())
            },
            "context_hit_at_k": {
                str(k): round(stats["context_hit"][k] / max(count, 1), 4)
                for k, count in sorted(counts.items())
            },
            "undercite_at_k": {
                str(k): round(stats["undercite"][k] / max(count, 1), 4)
                for k, count in sorted(counts.items())
            },
            "avg_extra_context_refs_at_k": {
                str(k): round(stats["extra_context_refs"][k] / max(count, 1), 2)
                for k, count in sorted(counts.items())
            },
            "range_score_at_k": {
                str(k): summarize_scores(stats["range_scores"].get(k, []))
                for k in sorted(counts)
            },
        }
    return summary


def _search(
    index: PortableBM25Index,
    query: str,
    *,
    strategy: str,
    top_k: int,
    expand_after: int,
    doc_top_k: int,
    candidate_k: int,
    learned_model: list[float] | None,
):
    if strategy == "hierarchical":
        return index.search_hierarchical(
            query,
            top_k=top_k,
            doc_top_k=doc_top_k,
            candidate_k=candidate_k,
            expand_after=expand_after,
        )
    if strategy == "rerank":
        return index.search_reranked(
            query,
            top_k=top_k,
            candidate_k=candidate_k,
            expand_after=expand_after,
        )
    if strategy == "learned-rerank":
        return _search_learned_rerank(
            index,
            query,
            top_k=top_k,
            candidate_k=candidate_k,
            expand_after=expand_after,
            weights=learned_model,
        )
    return index.search(query, top_k=top_k, expand_after=expand_after)


def _split_questions(questions: list[EvalQuestion], *, fraction: float, seed: int) -> tuple[list[EvalQuestion], list[EvalQuestion]]:
    if len(questions) < 4:
        return questions, questions
    rng = random.Random(seed)
    shuffled = list(questions)
    rng.shuffle(shuffled)
    split = max(1, min(len(shuffled) - 1, int(len(shuffled) * fraction)))
    return shuffled[:split], shuffled[split:]


def _train_learned_reranker(
    index: PortableBM25Index,
    questions: list[EvalQuestion],
    *,
    candidate_k: int,
    epochs: int,
) -> list[float]:
    weights = [0.0, 0.0, 0.0, 0.0, 0.0]
    learning_rate = 0.12
    rows: list[tuple[list[float], float]] = []
    for question in questions:
        candidates = index._rank_regions(question.query, top_k=candidate_k)
        max_score = max((score for _idx, score in candidates), default=1.0)
        for region_index, bm25_score in candidates:
            features = _candidate_features(index, question.query, region_index, bm25_score, max_score)
            label = 1.0 if index.regions[region_index].stable_ref in set(question.gold_refs) else 0.0
            rows.append((features, label))
    if not rows:
        return weights
    for _epoch in range(epochs):
        for features, label in rows:
            prediction = _sigmoid(sum(weight * value for weight, value in zip(weights, features, strict=True)))
            error = label - prediction
            for index_i, value in enumerate(features):
                weights[index_i] += learning_rate * error * value
        learning_rate *= 0.85
    return weights


def _search_learned_rerank(
    index: PortableBM25Index,
    query: str,
    *,
    top_k: int,
    candidate_k: int,
    expand_after: int,
    weights: list[float] | None,
):
    candidates = index._rank_regions(query, top_k=candidate_k)
    if not candidates:
        return []
    weights = weights or [1.0, 0.0, 0.0, 0.0, 0.0]
    max_score = max(score for _idx, score in candidates) or 1.0
    reranked = []
    for region_index, bm25_score in candidates:
        features = _candidate_features(index, query, region_index, bm25_score, max_score)
        score = sum(weight * value for weight, value in zip(weights, features, strict=True))
        reranked.append((region_index, score))
    reranked.sort(key=lambda item: (-item[1], index.regions[item[0]].stable_ref))
    return [
        index._hit(rank, region_index, score, expand_before=0, expand_after=expand_after, doc_score=None)
        for rank, (region_index, score) in enumerate(reranked[:top_k], start=1)
    ]


def _candidate_features(
    index: PortableBM25Index,
    query: str,
    region_index: int,
    bm25_score: float,
    max_score: float,
) -> list[float]:
    region = index.regions[region_index]
    query_terms = set(tokenize(query))
    text_terms = set(tokenize(region.text))
    metadata = "\n".join([region.view.summary, *region.view.questions, *region.view.keywords])
    metadata_terms = set(tokenize(metadata))
    return [
        bm25_score / max(max_score, 1e-6),
        len(query_terms & text_terms) / max(len(query_terms), 1),
        len(query_terms & metadata_terms) / max(len(query_terms), 1),
        _bigram_overlap(query, region.text),
        _bigram_overlap(query, metadata),
    ]


def _bigram_overlap(query: str, text: str) -> float:
    query_tokens = tokenize(query)
    text_tokens = tokenize(text)
    if len(query_tokens) < 2 or len(text_tokens) < 2:
        return 0.0
    query_bigrams = set(zip(query_tokens, query_tokens[1:], strict=False))
    text_bigrams = set(zip(text_tokens, text_tokens[1:], strict=False))
    return len(query_bigrams & text_bigrams) / max(len(query_bigrams), 1)


def _sigmoid(value: float) -> float:
    if value < -40:
        return 0.0
    if value > 40:
        return 1.0
    return 1.0 / (1.0 + math.exp(-value))


if __name__ == "__main__":
    main()
