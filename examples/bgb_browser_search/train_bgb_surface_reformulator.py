"""Train a surface-conditioned BGB query reformulator.

Earlier global query -> expansion-term models learned generic legal magnet
terms and often hurt BM25. This experiment constrains the problem:

1. BM25 proposes a small article surface for a query.
2. A tiny model sees query + candidate article surface tokens.
3. It predicts local expansion terms for that candidate.
4. Candidate-specific expanded searches are fused back with BM25.

The goal is to test whether "coarse surface -> local navigation" is a better
offline path than one global reformulator.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.bgb_browser_search.adapt_bgb_static_views import article_regions, summarize_ranks  # noqa: E402
from examples.bgb_browser_search.iterate_bgb_oracle_reformulation import (  # noqa: E402
    candidate_terms,
    document_frequency,
    first_rank,
    fuse_hits,
    greedy_oracle_terms,
    rank_with_terms,
)
from examples.bgb_browser_search.train_bgb_article_candidate_generator import (  # noqa: E402
    CandidateQuestion,
    encode,
    load_split_questions,
)
from refmark.search_index import PortableBM25Index, SearchRegion, load_search_index, tokenize  # noqa: E402

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This training example requires torch.") from exc


@dataclass(frozen=True)
class SurfaceExample:
    row: CandidateQuestion
    candidate_ref: str
    query_ids: list[int]
    term_ids: list[int]
    oracle_terms: list[str]
    is_gold: bool


class SurfaceTermPredictor(nn.Module):
    def __init__(self, vocab_size: int, term_count: int, *, embed_dim: int, hidden_dim: int, layers: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        modules: list[nn.Module] = [nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.06)]
        for _ in range(max(layers - 1, 0)):
            modules.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.06)])
        modules.append(nn.Linear(hidden_dim, term_count))
        self.scorer = nn.Sequential(*modules)

    def forward(self, query_ids: torch.Tensor) -> torch.Tensor:
        mask = query_ids.ne(0).unsqueeze(-1).float()
        embedded = self.embedding(query_ids)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.scorer(pooled)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a surface-conditioned BGB query reformulator.")
    parser.add_argument("--index", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_openrouter_index.json")
    parser.add_argument("--stress-report", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--seed", type=int, default=3031)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--article-count", type=int, default=40, help="Optional sampled article count for fast probes. Use 0 for all.")
    parser.add_argument("--top-ks", default="1,3,5,10,20,50")
    parser.add_argument("--surface-k", type=int, default=8, help="BM25 candidate articles considered by the local model.")
    parser.add_argument("--negative-candidates", type=int, default=3, help="Wrong BM25 candidates per train query.")
    parser.add_argument("--candidate-terms", type=int, default=80)
    parser.add_argument("--oracle-terms-per-query", type=int, default=3)
    parser.add_argument("--target-source", choices=["article", "oracle"], default="article")
    parser.add_argument("--include-surface-oracle", action="store_true")
    parser.add_argument("--max-train-questions", type=int, default=None)
    parser.add_argument("--max-eval-questions", type=int, default=None)
    parser.add_argument("--predict-terms", default="1,2,3,4,6")
    parser.add_argument("--fusion-alphas", default="0.05,0.1,0.15,0.25")
    parser.add_argument("--max-doc-freq-ratio", type=float, default=0.06)
    parser.add_argument("--query-vocab-size", type=int, default=24000)
    parser.add_argument("--surface-terms", type=int, default=32)
    parser.add_argument("--max-query-tokens", type=int, default=96)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--positive-weight", type=float, default=10.0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    top_ks = tuple(int(part) for part in args.top_ks.split(",") if part.strip())
    predict_terms = tuple(int(part) for part in args.predict_terms.split(",") if part.strip())
    fusion_alphas = tuple(float(part) for part in args.fusion_alphas.split(",") if part.strip())

    index = load_search_index(args.index)
    articles = article_regions(index.regions)
    article_by_ref = {article.stable_ref: article for article in articles}
    bm25 = PortableBM25Index(articles, include_source=True)
    train_rows, eval_rows = load_split_questions(args.stress_report, train_fraction=args.train_fraction, seed=args.seed)
    train_rows = [row for row in train_rows if row.article_ref in article_by_ref]
    eval_rows = [row for row in eval_rows if row.article_ref in article_by_ref]
    if args.article_count > 0:
        sampled_refs = sample_article_refs(train_rows, eval_rows, count=args.article_count, rng=random.Random(args.seed))
        train_rows = [row for row in train_rows if row.article_ref in sampled_refs]
        eval_rows = [row for row in eval_rows if row.article_ref in sampled_refs]
    else:
        sampled_refs = sorted({row.article_ref for row in train_rows} | {row.article_ref for row in eval_rows})
    if args.max_train_questions is not None:
        train_rows = train_rows[: args.max_train_questions]
    if args.max_eval_questions is not None:
        eval_rows = eval_rows[: args.max_eval_questions]

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
    train_oracles = build_train_targets(
        bm25,
        train_rows,
        candidates_by_ref,
        target_source=args.target_source,
        max_terms=args.oracle_terms_per_query,
        top_k=max(top_ks),
    )
    term_vocab = build_term_vocab(train_oracles, candidates_by_ref)
    query_vocab = build_query_vocab(train_rows, article_by_ref, candidates_by_ref, surface_terms=args.surface_terms, vocab_size=args.query_vocab_size)
    train_examples = build_surface_examples(
        bm25,
        train_rows,
        train_oracles,
        query_vocab,
        term_vocab,
        candidates_by_ref,
        surface_terms=args.surface_terms,
        max_query_tokens=args.max_query_tokens,
        surface_k=args.surface_k,
        negative_candidates=args.negative_candidates,
        training=True,
    )
    train_examples_with_signal = [example for example in train_examples if example.term_ids or not example.is_gold]
    if not train_examples_with_signal:
        raise SystemExit("No train rows had oracle-improving terms or negatives.")

    model = SurfaceTermPredictor(
        len(query_vocab),
        len(term_vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)
    pos_weight = torch.full((len(term_vocab),), float(args.positive_weight))
    started = time.perf_counter()
    history = []
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_score = -1.0
    eval_examples = build_surface_examples(
        bm25,
        eval_rows,
        {},
        query_vocab,
        term_vocab,
        candidates_by_ref,
        surface_terms=args.surface_terms,
        max_query_tokens=args.max_query_tokens,
        surface_k=args.surface_k,
        negative_candidates=0,
        training=False,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        query_batch, target_batch = tensorize(train_examples_with_signal, term_count=len(term_vocab))
        optimizer.zero_grad(set_to_none=True)
        logits = model(query_batch)
        loss = F.binary_cross_entropy_with_logits(logits, target_batch, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
        row: dict[str, object] = {"epoch": epoch, "loss": round(float(loss.detach()), 6)}
        if epoch == 1 or epoch == args.epochs or epoch % 10 == 0:
            metrics = evaluate_surface_model(
                model,
                bm25,
                eval_rows,
                eval_examples,
                term_vocab,
                candidates_by_ref,
                predict_terms=predict_terms,
                fusion_alphas=fusion_alphas,
                include_surface_oracle=args.include_surface_oracle,
                top_ks=top_ks,
            )
            best = metrics["best_surface_fusion"]["metrics"]["article_hit_at_k"]
            score = float(best["mrr"]) + float(best["hit_at_k"].get("10", 0.0))
            if score > best_score:
                best_score = score
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            row.update(
                {
                    "best_predict_terms": metrics["best_surface_fusion"]["predict_terms"],
                    "best_alpha": metrics["best_surface_fusion"]["alpha"],
                    "hit_at_10": best["hit_at_k"].get("10", 0.0),
                    "mrr": best["mrr"],
                }
            )
        history.append(row)
        print(json.dumps(row))

    model.load_state_dict(best_state)
    final = evaluate_surface_model(
        model,
        bm25,
        eval_rows,
        eval_examples,
        term_vocab,
        candidates_by_ref,
        predict_terms=predict_terms,
        fusion_alphas=fusion_alphas,
        include_surface_oracle=args.include_surface_oracle,
        top_ks=top_ks,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "refmark.bgb_surface_reformulator.v1",
            "settings": vars(args),
            "sampled_refs": sampled_refs,
            "query_vocab": query_vocab,
            "term_vocab": term_vocab,
            "model_state": model.state_dict(),
        },
        output,
    )
    report = {
        "schema": "refmark.bgb_surface_reformulator_report.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": vars(args),
        "sampled_refs": sampled_refs,
        "train_questions": len(train_rows),
        "eval_questions": len(eval_rows),
        "train_examples": len(train_examples_with_signal),
        "train_gold_examples_with_oracle_terms": sum(1 for example in train_examples if example.is_gold and example.term_ids),
        "term_vocab_size": len(term_vocab),
        "query_vocab_size": len(query_vocab),
        "parameters": sum(param.numel() for param in model.parameters()),
        "artifact_bytes": output.stat().st_size,
        "artifact_megabytes": round(output.stat().st_size / 1_000_000, 4),
        "train_seconds": round(time.perf_counter() - started, 3),
        "history": history,
        "final": final,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
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


def build_train_targets(
    index: PortableBM25Index,
    rows: list[CandidateQuestion],
    candidates_by_ref: dict[str, list[str]],
    *,
    target_source: str,
    max_terms: int,
    top_k: int,
) -> dict[str, list[str]]:
    output = {}
    for row in rows:
        if target_source == "article":
            terms = candidates_by_ref[row.article_ref][:max_terms]
        else:
            terms = greedy_oracle_terms(
                index,
                row.query,
                row.article_ref,
                candidates_by_ref[row.article_ref],
                max_terms=max_terms,
                top_k=top_k,
            )
        if terms:
            output[row.query] = terms
    return output


def build_term_vocab(train_oracles: dict[str, list[str]], candidates_by_ref: dict[str, list[str]]) -> dict[str, int]:
    counts = Counter(term for terms in train_oracles.values() for term in terms)
    for terms in candidates_by_ref.values():
        for term in terms[:24]:
            counts[term] += 0.1
    return {term: index for index, (term, _count) in enumerate(counts.most_common())}


def build_query_vocab(
    rows: list[CandidateQuestion],
    article_by_ref: dict[str, SearchRegion],
    candidates_by_ref: dict[str, list[str]],
    *,
    surface_terms: int,
    vocab_size: int,
) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        counts.update(tokenize(row.query))
        counts.update(surface_text(article_by_ref[row.article_ref], candidates_by_ref[row.article_ref], surface_terms=surface_terms))
    vocab = {"<pad>": 0, "<unk>": 1, "<surface>": 2}
    for token, _count in counts.most_common(max(vocab_size - len(vocab), 0)):
        vocab.setdefault(token, len(vocab))
    return vocab


def build_surface_examples(
    index: PortableBM25Index,
    rows: list[CandidateQuestion],
    oracle_terms: dict[str, list[str]],
    query_vocab: dict[str, int],
    term_vocab: dict[str, int],
    candidates_by_ref: dict[str, list[str]],
    *,
    surface_terms: int,
    max_query_tokens: int,
    surface_k: int,
    negative_candidates: int,
    training: bool,
) -> list[SurfaceExample]:
    examples: list[SurfaceExample] = []
    article_by_ref = {region.stable_ref: region for region in index.regions}
    for row in rows:
        if row.article_ref not in candidates_by_ref:
            continue
        bm25_refs = [hit.stable_ref for hit in index.search(row.query, top_k=surface_k)]
        candidate_refs = [row.article_ref]
        if training:
            candidate_refs.extend(ref for ref in bm25_refs if ref != row.article_ref and ref in candidates_by_ref)
            candidate_refs = candidate_refs[: 1 + negative_candidates]
        else:
            candidate_refs = [ref for ref in bm25_refs if ref in candidates_by_ref]
        for candidate_ref in dict.fromkeys(candidate_refs):
            terms = oracle_terms.get(row.query, []) if candidate_ref == row.article_ref else []
            text = encode_surface_query(
                row.query,
                article_by_ref[candidate_ref],
                candidates_by_ref[candidate_ref],
                query_vocab,
                surface_terms=surface_terms,
                max_query_tokens=max_query_tokens,
            )
            examples.append(
                SurfaceExample(
                    row=row,
                    candidate_ref=candidate_ref,
                    query_ids=text,
                    term_ids=[term_vocab[term] for term in terms if term in term_vocab],
                    oracle_terms=terms,
                    is_gold=candidate_ref == row.article_ref,
                )
            )
    return examples


def encode_surface_query(
    query: str,
    article: SearchRegion,
    terms: list[str],
    vocab: dict[str, int],
    *,
    surface_terms: int,
    max_query_tokens: int,
) -> list[int]:
    tokens = tokenize(query) + ["<surface>"] + surface_text(article, terms, surface_terms=surface_terms)
    ids = [vocab.get(token, 1) for token in tokens[:max_query_tokens]]
    return ids or [1]


def surface_text(article: SearchRegion, terms: list[str], *, surface_terms: int) -> list[str]:
    text = f"{article.view.summary} {' '.join(article.view.keywords)} {' '.join(terms[:surface_terms])}"
    return tokenize(text)


def tensorize(examples: list[SurfaceExample], *, term_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(example.query_ids) for example in examples)
    query_batch = torch.tensor([example.query_ids + [0] * (max_len - len(example.query_ids)) for example in examples], dtype=torch.long)
    target_batch = torch.zeros((len(examples), term_count), dtype=torch.float32)
    for row_index, example in enumerate(examples):
        if example.term_ids:
            target_batch[row_index, example.term_ids] = 1.0
    return query_batch, target_batch


def evaluate_surface_model(
    model: SurfaceTermPredictor,
    index: PortableBM25Index,
    rows: list[CandidateQuestion],
    examples: list[SurfaceExample],
    term_vocab: dict[str, int],
    candidates_by_ref: dict[str, list[str]],
    *,
    predict_terms: tuple[int, ...],
    fusion_alphas: tuple[float, ...],
    include_surface_oracle: bool,
    top_ks: tuple[int, ...],
) -> dict[str, object]:
    term_by_id = {index: term for term, index in term_vocab.items()}
    examples_by_query: dict[str, list[SurfaceExample]] = {}
    for example in examples:
        examples_by_query.setdefault(example.row.query, []).append(example)
    raw_ranks = []
    oracle_surface_ranks = []
    predicted: dict[str, list[int | None]] = {str(count): [] for count in predict_terms}
    fused: dict[str, list[int | None]] = {f"{count}:{alpha}": [] for count in predict_terms for alpha in fusion_alphas}
    samples = []
    model.eval()
    for row in rows:
        raw_hits = index.search(row.query, top_k=max(top_ks))
        raw_refs = [hit.stable_ref for hit in raw_hits]
        raw_ranks.append(first_rank(raw_refs, row.article_ref))
        row_examples = [example for example in examples_by_query.get(row.query, []) if example.candidate_ref in candidates_by_ref]
        if include_surface_oracle:
            oracle_surface_ranks.append(surface_oracle_rank(index, row, row_examples, candidates_by_ref, top_k=max(top_ks)))
        if row_examples:
            query_batch, _targets = tensorize(row_examples, term_count=len(term_vocab))
            with torch.no_grad():
                logits_batch = model(query_batch)
        else:
            logits_batch = []
        expanded_hits_by_count: dict[int, list] = {count: [] for count in predict_terms}
        for example, logits in zip(row_examples, logits_batch, strict=False):
            ordered_terms = [term_by_id[idx] for idx in torch.argsort(logits, descending=True).tolist()]
            for count in predict_terms:
                terms = ordered_terms[:count]
                hits = index.search(f"{row.query} {' '.join(terms)}", top_k=max(top_ks))
                expanded_hits_by_count[count].extend(hits)
        for count in predict_terms:
            expanded_refs = unique_hits(expanded_hits_by_count[count], max_count=max(top_ks))
            predicted[str(count)].append(first_rank(expanded_refs, row.article_ref))
            for alpha in fusion_alphas:
                fused_refs = fuse_hits(raw_hits, expanded_hits_by_count[count], alpha=alpha)
                fused[f"{count}:{alpha}"].append(first_rank(fused_refs[: max(top_ks)], row.article_ref))
        if len(samples) < 20:
            samples.append(
                {
                    "query": row.query,
                    "gold_ref": row.article_ref,
                    "raw_rank": raw_ranks[-1],
                    "oracle_surface_rank": oracle_surface_ranks[-1] if include_surface_oracle else None,
                    "surface_refs": [example.candidate_ref for example in row_examples[:8]],
                }
            )
    predicted_summary = {count: {"article_hit_at_k": summarize_ranks(ranks, top_ks=top_ks)} for count, ranks in predicted.items()}
    fused_summary = {key: {"article_hit_at_k": summarize_ranks(ranks, top_ks=top_ks)} for key, ranks in fused.items()}
    best_key, best_metrics = max(
        fused_summary.items(),
        key=lambda item: (
            item[1]["article_hit_at_k"]["hit_at_k"].get("10", 0.0),
            item[1]["article_hit_at_k"]["mrr"],
        ),
    )
    count, alpha = best_key.split(":")
    return {
        "raw_bm25": {"article_hit_at_k": summarize_ranks(raw_ranks, top_ks=top_ks)},
        "surface_oracle": {"article_hit_at_k": summarize_ranks(oracle_surface_ranks, top_ks=top_ks)} if include_surface_oracle else None,
        "predicted_surface_append": predicted_summary,
        "predicted_surface_fusion": fused_summary,
        "best_surface_fusion": {"predict_terms": int(count), "alpha": float(alpha), "metrics": best_metrics},
        "sample_results": samples,
    }


def surface_oracle_rank(
    index: PortableBM25Index,
    row: CandidateQuestion,
    examples: list[SurfaceExample],
    candidates_by_ref: dict[str, list[str]],
    *,
    top_k: int,
) -> int | None:
    best_rank = rank_with_terms(index, row.query, row.article_ref, [], top_k=top_k, fusion_alpha=1.0)
    for example in examples:
        terms = greedy_oracle_terms(index, row.query, example.candidate_ref, candidates_by_ref[example.candidate_ref], max_terms=3, top_k=top_k)
        rank = rank_with_terms(index, row.query, row.article_ref, terms, top_k=top_k, fusion_alpha=1.0)
        if rank is not None and (best_rank is None or rank < best_rank):
            best_rank = rank
    return best_rank


def unique_hits(hits: list, *, max_count: int) -> list[str]:
    refs = []
    seen = set()
    for hit in sorted(hits, key=lambda item: (-item.score, item.stable_ref)):
        if hit.stable_ref in seen:
            continue
        seen.add(hit.stable_ref)
        refs.append(hit.stable_ref)
        if len(refs) >= max_count:
            break
    return refs


if __name__ == "__main__":
    main()
