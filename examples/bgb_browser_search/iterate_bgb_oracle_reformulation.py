"""Iterate oracle query-reformulation on a tiny BGB article slice.

This is a fast diagnostic for the reformulation idea. It samples a small set
of articles, discovers expansion terms that actually improve BM25 rank on
training questions, and evaluates whether those learned article-term banks
would help held-out questions if a reformulator predicted them correctly.

The script intentionally reports both:

- learned-bank evaluation: append terms learned from train questions for the
  gold article; this tests whether the term bank is useful;
- per-query oracle: greedily choose terms from the gold article for each eval
  query; this estimates the upper bound for a predictor.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.bgb_browser_search.adapt_bgb_static_views import article_regions, summarize_ranks  # noqa: E402
from examples.bgb_browser_search.train_bgb_article_candidate_generator import CandidateQuestion, load_split_questions, summarize_groups  # noqa: E402
from refmark.search_index import PortableBM25Index, SearchRegion, load_search_index, tokenize  # noqa: E402


STOP = {
    "und",
    "oder",
    "der",
    "die",
    "das",
    "den",
    "dem",
    "ein",
    "eine",
    "einer",
    "eines",
    "mit",
    "von",
    "für",
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
    "from",
    "what",
    "when",
    "where",
    "which",
}


@dataclass(frozen=True)
class QueryResult:
    query: str
    article_ref: str
    language: str
    style: str
    raw_rank: int | None
    learned_rank: int | None
    oracle_rank: int | None
    learned_terms: list[str]
    oracle_terms: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny oracle loop for BGB query reformulation.")
    parser.add_argument("--index", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_openrouter_index.json")
    parser.add_argument("--stress-report", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--article-count", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--top-ks", default="1,3,5,10,20,50")
    parser.add_argument("--candidate-terms", type=int, default=80)
    parser.add_argument("--terms-per-query", type=int, default=3)
    parser.add_argument("--terms-per-article", type=int, default=8)
    parser.add_argument("--max-doc-freq-ratio", type=float, default=0.06)
    parser.add_argument("--fusion-alpha", type=float, default=0.15)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    top_ks = tuple(int(part) for part in args.top_ks.split(",") if part.strip())
    index = load_search_index(args.index)
    articles = article_regions(index.regions)
    article_by_ref = {article.stable_ref: article for article in articles}
    bm25 = PortableBM25Index(articles, include_source=True)
    train_rows, eval_rows = load_split_questions(args.stress_report, train_fraction=args.train_fraction, seed=args.seed)
    train_rows = [row for row in train_rows if row.article_ref in article_by_ref]
    eval_rows = [row for row in eval_rows if row.article_ref in article_by_ref]
    sampled_refs = sample_article_refs(train_rows, eval_rows, count=args.article_count, rng=rng)
    train_rows = [row for row in train_rows if row.article_ref in sampled_refs]
    eval_rows = [row for row in eval_rows if row.article_ref in sampled_refs]
    doc_freq = document_frequency(articles)
    candidates_by_ref = {
        ref: candidate_terms(
            article_by_ref[ref],
            doc_freq=doc_freq,
            article_count=len(articles),
            limit=args.candidate_terms,
            max_doc_freq_ratio=args.max_doc_freq_ratio,
        )
        for ref in sampled_refs
    }

    term_bank: dict[str, Counter[str]] = {ref: Counter() for ref in sampled_refs}
    iteration_reports = []
    for iteration in range(1, args.iterations + 1):
        hard_train = select_hard_train_rows(
            bm25,
            train_rows,
            term_bank,
            terms_per_article=args.terms_per_article,
            top_k=max(top_ks),
            fusion_alpha=args.fusion_alpha,
        )
        for row in hard_train:
            terms = greedy_oracle_terms(
                bm25,
                row.query,
                row.article_ref,
                candidates_by_ref[row.article_ref],
                max_terms=args.terms_per_query,
                top_k=max(top_ks),
            )
            for term in terms:
                term_bank[row.article_ref][term] += 1
        learned_results = evaluate_rows(
            bm25,
            eval_rows,
            term_bank,
            candidates_by_ref,
            terms_per_article=args.terms_per_article,
            oracle_terms_per_query=args.terms_per_query,
            top_ks=top_ks,
            fusion_alpha=args.fusion_alpha,
        )
        iteration_reports.append(
            {
                "iteration": iteration,
                "adapted_train_questions": len(hard_train),
                "learned_terms": {
                    ref: [term for term, _count in counts.most_common(args.terms_per_article)]
                    for ref, counts in term_bank.items()
                },
                "metrics": summarize_query_results(learned_results, top_ks=top_ks),
                "sample_results": [result.to_dict() for result in learned_results[:12]],
            }
        )

    report = {
        "schema": "refmark.bgb_oracle_reformulation_loop.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": vars(args),
        "sampled_refs": sampled_refs,
        "train_questions": len(train_rows),
        "eval_questions": len(eval_rows),
        "iterations": iteration_reports,
        "final": iteration_reports[-1] if iteration_reports else {},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def sample_article_refs(
    train_rows: list[CandidateQuestion],
    eval_rows: list[CandidateQuestion],
    *,
    count: int,
    rng: random.Random,
) -> list[str]:
    train_counts = Counter(row.article_ref for row in train_rows)
    eval_counts = Counter(row.article_ref for row in eval_rows)
    eligible = sorted(ref for ref in train_counts if train_counts[ref] >= 2 and eval_counts[ref] >= 2)
    if len(eligible) <= count:
        return eligible
    return sorted(rng.sample(eligible, count))


def document_frequency(articles: list[SearchRegion]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for article in articles:
        counts.update(set(token for token in tokenize(article.index_text(include_source=True)) if keep_term(token)))
    return counts


def candidate_terms(
    article: SearchRegion,
    *,
    doc_freq: Counter[str],
    article_count: int,
    limit: int,
    max_doc_freq_ratio: float,
) -> list[str]:
    counts = Counter(token for token in tokenize(article.index_text(include_source=True)) if keep_term(token))
    scored = []
    for term, count in counts.items():
        if doc_freq[term] / max(article_count, 1) > max_doc_freq_ratio:
            continue
        idf = article_count / max(doc_freq[term], 1)
        scored.append((count * idf, term))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [term for _score, term in scored[:limit]]


def keep_term(term: str) -> bool:
    return len(term) >= 3 and not term.isdigit() and term not in STOP


def select_hard_train_rows(
    index: PortableBM25Index,
    rows: list[CandidateQuestion],
    term_bank: dict[str, Counter[str]],
    *,
    terms_per_article: int,
    top_k: int,
    fusion_alpha: float,
) -> list[CandidateQuestion]:
    hard = []
    for row in rows:
        terms = [term for term, _count in term_bank[row.article_ref].most_common(terms_per_article)]
        rank = rank_with_terms(index, row.query, row.article_ref, terms, top_k=top_k, fusion_alpha=fusion_alpha)
        if rank is None or rank > 10:
            hard.append(row)
    return hard or rows


def greedy_oracle_terms(
    index: PortableBM25Index,
    query: str,
    article_ref: str,
    candidates: list[str],
    *,
    max_terms: int,
    top_k: int,
) -> list[str]:
    selected: list[str] = []
    best_rank = rank_with_terms(index, query, article_ref, selected, top_k=top_k, fusion_alpha=1.0)
    for _ in range(max_terms):
        best_term = None
        best_candidate_rank = best_rank
        for term in candidates:
            if term in selected:
                continue
            rank = rank_with_terms(index, query, article_ref, [*selected, term], top_k=top_k, fusion_alpha=1.0)
            if rank_better(rank, best_candidate_rank):
                best_term = term
                best_candidate_rank = rank
        if best_term is None:
            break
        selected.append(best_term)
        best_rank = best_candidate_rank
        if best_rank == 1:
            break
    return selected


def evaluate_rows(
    index: PortableBM25Index,
    rows: list[CandidateQuestion],
    term_bank: dict[str, Counter[str]],
    candidates_by_ref: dict[str, list[str]],
    *,
    terms_per_article: int,
    oracle_terms_per_query: int,
    top_ks: tuple[int, ...],
    fusion_alpha: float,
) -> list[QueryResult]:
    results = []
    for row in rows:
        learned_terms = [term for term, _count in term_bank[row.article_ref].most_common(terms_per_article)]
        oracle_terms = greedy_oracle_terms(
            index,
            row.query,
            row.article_ref,
            candidates_by_ref[row.article_ref],
            max_terms=oracle_terms_per_query,
            top_k=max(top_ks),
        )
        results.append(
            QueryResult(
                query=row.query,
                article_ref=row.article_ref,
                language=row.language,
                style=row.style,
                raw_rank=rank_with_terms(index, row.query, row.article_ref, [], top_k=max(top_ks), fusion_alpha=1.0),
                learned_rank=rank_with_terms(index, row.query, row.article_ref, learned_terms, top_k=max(top_ks), fusion_alpha=fusion_alpha),
                oracle_rank=rank_with_terms(index, row.query, row.article_ref, oracle_terms, top_k=max(top_ks), fusion_alpha=1.0),
                learned_terms=learned_terms,
                oracle_terms=oracle_terms,
            )
        )
    return results


def rank_with_terms(
    index: PortableBM25Index,
    query: str,
    article_ref: str,
    terms: list[str],
    *,
    top_k: int,
    fusion_alpha: float,
) -> int | None:
    raw_hits = index.search(query, top_k=top_k)
    if not terms:
        return first_rank([hit.stable_ref for hit in raw_hits], article_ref)
    expanded_hits = index.search(f"{query} {' '.join(terms)}", top_k=top_k)
    refs = fuse_hits(raw_hits, expanded_hits, alpha=fusion_alpha)
    return first_rank(refs[:top_k], article_ref)


def fuse_hits(raw_hits, expanded_hits, *, alpha: float) -> list[str]:
    scores: dict[str, float] = {}
    max_raw = max((hit.score for hit in raw_hits), default=1.0) or 1.0
    max_expanded = max((hit.score for hit in expanded_hits), default=1.0) or 1.0
    for hit in raw_hits:
        scores[hit.stable_ref] = scores.get(hit.stable_ref, 0.0) + (hit.score / max_raw)
    for hit in expanded_hits:
        scores[hit.stable_ref] = scores.get(hit.stable_ref, 0.0) + alpha * (hit.score / max_expanded)
    return [ref for ref, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def first_rank(stable_refs: list[str], gold_ref: str) -> int | None:
    for index, stable_ref in enumerate(stable_refs, start=1):
        if stable_ref == gold_ref:
            return index
    return None


def rank_better(candidate: int | None, current: int | None) -> bool:
    if candidate is None:
        return False
    if current is None:
        return True
    return candidate < current


def summarize_query_results(results: list[QueryResult], *, top_ks: tuple[int, ...]) -> dict[str, object]:
    raw = [result.raw_rank for result in results]
    learned = [result.learned_rank for result in results]
    oracle = [result.oracle_rank for result in results]
    return {
        "raw_bm25": summarize_ranks(raw, top_ks=top_ks),
        "learned_bank": summarize_ranks(learned, top_ks=top_ks),
        "per_query_oracle": summarize_ranks(oracle, top_ks=top_ks),
        "learned_by_style": summarize_groups(group_ranks(results, "style", "learned_rank"), top_ks=top_ks),
        "oracle_by_style": summarize_groups(group_ranks(results, "style", "oracle_rank"), top_ks=top_ks),
        "improved_count": sum(rank_better(result.learned_rank, result.raw_rank) for result in results),
        "worsened_count": sum(rank_better(result.raw_rank, result.learned_rank) for result in results),
    }


def group_ranks(results: list[QueryResult], field: str, rank_field: str) -> dict[str, list[int | None]]:
    grouped: dict[str, list[int | None]] = {}
    for result in results:
        grouped.setdefault(str(getattr(result, field)), []).append(getattr(result, rank_field))
    return grouped


if __name__ == "__main__":
    main()
