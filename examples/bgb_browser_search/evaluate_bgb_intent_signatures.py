"""Build and evaluate compressed BGB article intent signatures.

Raw generated questions were too noisy when pasted into BM25 metadata. This
script compresses held-out-safe train questions into short article-level intent
phrases, injects those phrases as retrieval metadata, and evaluates against the
held-out questions from the same stress reports.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.bgb_browser_search.adapt_bgb_static_views import (  # noqa: E402
    article_id_from_ref,
    article_regions,
    split_questions_by_block,
    stress_questions,
    summarize_ranks,
    unique,
)
from examples.bgb_browser_search.run_bgb_stress_eval import StressQuestion  # noqa: E402
from refmark.search_index import PortableBM25Index, RetrievalView, SearchHit, SearchRegion, load_search_index, tokenize  # noqa: E402


STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "am",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "have",
    "how",
    "if",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "someone",
    "that",
    "the",
    "their",
    "then",
    "there",
    "this",
    "to",
    "under",
    "was",
    "what",
    "when",
    "who",
    "with",
    "zu",
    "der",
    "die",
    "das",
    "und",
    "oder",
    "wenn",
    "wie",
    "was",
    "wer",
    "wann",
    "wo",
    "unter",
    "mit",
    "von",
    "bei",
    "auf",
    "aus",
    "für",
    "eine",
    "einer",
    "einem",
    "einen",
    "ein",
    "nicht",
    "noch",
    "auch",
    "sich",
    "ich",
    "mir",
    "mein",
    "meine",
    "meinem",
    "meinen",
    "jemand",
    "jemandem",
    "etwas",
    "werden",
    "wird",
    "kann",
    "muss",
    "müssen",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate compressed BGB intent signatures.")
    parser.add_argument("--index", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_openrouter_index.json")
    parser.add_argument("--stress-report", action="append", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output-index", default=None)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1515)
    parser.add_argument("--top-ks", default="1,3,5,10,20,50")
    parser.add_argument("--signature-limit", type=int, default=24)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--max-phrase-tokens", type=int, default=3)
    parser.add_argument("--rrf-k", type=float, default=60.0)
    args = parser.parse_args()

    source_payload = json.loads(Path(args.index).read_text(encoding="utf-8-sig"))
    source_index = load_search_index(args.index)
    base_regions = article_regions(source_index.regions)
    train_questions, eval_questions = load_split(args.stress_report, train_fraction=args.train_fraction, seed=args.seed)
    signatures = build_signatures(
        train_questions,
        signature_limit=args.signature_limit,
        min_df=args.min_df,
        max_phrase_tokens=args.max_phrase_tokens,
    )
    signature_regions = apply_signatures(base_regions, signatures)
    top_ks = tuple(int(part) for part in args.top_ks.split(",") if part.strip())

    baseline_index = PortableBM25Index(base_regions, include_source=True)
    signature_index = PortableBM25Index(signature_regions, include_source=True)
    signature_only_index = PortableBM25Index(signature_only_regions(base_regions, signatures), include_source=False)
    baseline = evaluate(lambda query, top_k: baseline_index.search(query, top_k=top_k), eval_questions, top_ks=top_ks)
    signature_mixed = evaluate(lambda query, top_k: signature_index.search(query, top_k=top_k), eval_questions, top_ks=top_ks)
    signature_only = evaluate(lambda query, top_k: signature_only_index.search(query, top_k=top_k), eval_questions, top_ks=top_ks)
    source_signature_rrf = evaluate(
        lambda query, top_k: rrf_hits(
            baseline_index.search(query, top_k=max(top_k, 80)),
            signature_only_index.search(query, top_k=max(top_k, 80)),
            first_weight=0.7,
            rrf_k=args.rrf_k,
        )[:top_k],
        eval_questions,
        top_ks=top_ks,
    )

    if args.output_index:
        write_index(source_payload, signature_regions, Path(args.output_index), signatures=signatures, settings=vars(args))
    report = {
        "schema": "refmark.bgb_intent_signature_eval.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_index": args.index,
        "stress_reports": args.stress_report,
        "settings": vars(args),
        "train_questions": len(train_questions),
        "eval_questions": len(eval_questions),
        "signature_articles": len(signatures),
        "signature_count": sum(len(items) for items in signatures.values()),
        "signature_examples": signature_examples(signatures),
        "results": {
            "baseline_source_plus_views": baseline,
            "signature_mixed": signature_mixed,
            "signature_only": signature_only,
            "source_signature_rrf": source_signature_rrf,
        },
        "deltas_vs_baseline": {
            "signature_mixed": delta(baseline, signature_mixed),
            "signature_only": delta(baseline, signature_only),
            "source_signature_rrf": delta(baseline, source_signature_rrf),
        },
    }
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def load_split(paths: list[str], *, train_fraction: float, seed: int) -> tuple[list[StressQuestion], list[StressQuestion]]:
    train: list[StressQuestion] = []
    eval_rows: list[StressQuestion] = []
    for offset, path in enumerate(paths):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        left, right = split_questions_by_block(stress_questions(payload), train_fraction=train_fraction, seed=seed + offset)
        train.extend(left)
        eval_rows.extend(right)
    return train, eval_rows


def build_signatures(
    questions: list[StressQuestion],
    *,
    signature_limit: int,
    min_df: int,
    max_phrase_tokens: int,
) -> dict[str, list[str]]:
    article_phrases: dict[str, Counter[str]] = defaultdict(Counter)
    phrase_df: Counter[str] = Counter()
    for question in questions:
        phrases = phrase_candidates(question.query, max_phrase_tokens=max_phrase_tokens)
        article = question.block_id
        article_phrases[article].update(phrases)
        phrase_df.update(set(phrases))
    article_count = max(len(article_phrases), 1)
    signatures: dict[str, list[str]] = {}
    for article, counts in article_phrases.items():
        scored = []
        for phrase, count in counts.items():
            df = phrase_df[phrase]
            if df < min_df:
                continue
            idf = math.log((article_count + 1) / (df + 0.5))
            length_bonus = 1.0 + (0.12 * (len(phrase.split()) - 1))
            scored.append((count * idf * length_bonus, phrase))
        signatures[article] = [phrase for _score, phrase in sorted(scored, key=lambda item: (-item[0], item[1]))[:signature_limit]]
    return signatures


def phrase_candidates(text: str, *, max_phrase_tokens: int) -> list[str]:
    tokens = [clean_token(token) for token in tokenize(text)]
    tokens = [token for token in tokens if token and token not in STOPWORDS and not token.isdigit() and len(token) > 2]
    phrases: list[str] = []
    for width in range(1, max_phrase_tokens + 1):
        for start in range(0, max(len(tokens) - width + 1, 0)):
            chunk = tokens[start : start + width]
            if len(chunk) != width:
                continue
            if width > 1 and len(set(chunk)) == 1:
                continue
            phrases.append(" ".join(chunk))
    return phrases


def clean_token(token: str) -> str:
    return re.sub(r"(^\W+|\W+$)", "", token.lower())


def apply_signatures(regions: list[SearchRegion], signatures: dict[str, list[str]]) -> list[SearchRegion]:
    output = []
    for region in regions:
        phrases = signatures.get(region.stable_ref, [])
        view = region.view
        output.append(
            replace(
                region,
                view=RetrievalView(
                    summary=view.summary,
                    questions=view.questions,
                    keywords=unique([*phrases, *view.keywords])[: max(len(view.keywords), 0) + len(phrases)],
                ),
            )
        )
    return output


def signature_only_regions(regions: list[SearchRegion], signatures: dict[str, list[str]]) -> list[SearchRegion]:
    output = []
    for region in regions:
        phrases = signatures.get(region.stable_ref, [])
        output.append(replace(region, text="", view=RetrievalView(summary="", questions=[], keywords=phrases)))
    return output


def evaluate(search_fn, questions: list[StressQuestion], *, top_ks: tuple[int, ...]) -> dict[str, object]:
    max_k = max(top_ks)
    ranks = []
    by_language: dict[str, list[int | None]] = defaultdict(list)
    by_style: dict[str, list[int | None]] = defaultdict(list)
    misses_by_block: Counter[str] = Counter()
    for question in questions:
        hits = search_fn(question.query, max_k)
        gold_article = article_id_from_ref(question.block_id)
        rank = first_rank([hit.stable_ref for hit in hits], gold_article)
        ranks.append(rank)
        by_language[question.language].append(rank)
        by_style[question.style].append(rank)
        if rank is None or rank > 10:
            misses_by_block[question.block_id] += 1
    return {
        "article_hit_at_k": summarize_ranks(ranks, top_ks=top_ks),
        "by_language": {name: summarize_ranks(values, top_ks=top_ks) for name, values in sorted(by_language.items())},
        "by_style": {name: summarize_ranks(values, top_ks=top_ks) for name, values in sorted(by_style.items())},
        "misses_by_block_top": [{"block_id": block, "misses": count} for block, count in misses_by_block.most_common(20)],
    }


def first_rank(stable_refs: list[str], gold_article: str) -> int | None:
    for rank, stable_ref in enumerate(stable_refs, start=1):
        if article_id_from_ref(stable_ref) == gold_article:
            return rank
    return None


def rrf_hits(first: list[SearchHit], second: list[SearchHit], *, first_weight: float, rrf_k: float) -> list[SearchHit]:
    scores: dict[str, float] = {}
    hits_by_ref: dict[str, SearchHit] = {}
    for weight, hits in ((first_weight, first), (1.0 - first_weight, second)):
        for rank, hit in enumerate(hits, start=1):
            scores[hit.stable_ref] = scores.get(hit.stable_ref, 0.0) + (weight / (rrf_k + rank))
            hits_by_ref.setdefault(hit.stable_ref, hit)
    ordered = sorted(scores, key=lambda ref: (-scores[ref], ref))
    return [replace(hits_by_ref[ref], rank=rank, score=round(scores[ref], 6)) for rank, ref in enumerate(ordered, start=1)]


def delta(baseline: dict[str, object], candidate: dict[str, object]) -> dict[str, float]:
    base = baseline["article_hit_at_k"]
    cand = candidate["article_hit_at_k"]
    return {
        **{
            f"hit@{key}": round(float(value) - float(base["hit_at_k"].get(key, 0.0)), 4)
            for key, value in cand["hit_at_k"].items()
        },
        "mrr": round(float(cand["mrr"]) - float(base["mrr"]), 4),
    }


def signature_examples(signatures: dict[str, list[str]]) -> list[dict[str, object]]:
    return [{"article_ref": ref, "signatures": phrases[:12]} for ref, phrases in sorted(signatures.items())[:20]]


def write_index(source_payload: dict[str, object], regions: list[SearchRegion], path: Path, *, signatures: dict[str, list[str]], settings: dict[str, object]) -> None:
    payload = dict(source_payload)
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    payload["settings"] = {
        **dict(source_payload.get("settings", {})),
        "adaptation": "bgb-intent-signatures",
        "intent_signature_settings": settings,
        "intent_signature_articles": len(signatures),
    }
    payload["regions"] = [region.to_dict() for region in regions]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
