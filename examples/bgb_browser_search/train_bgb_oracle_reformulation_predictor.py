"""Train a tiny predictor for oracle BM25-improving expansion terms.

This follows the small-slice oracle loop:

1. sample a tiny article surface;
2. derive terms that improve BM25 rank on train questions;
3. train query -> useful expansion terms;
4. evaluate whether predicted terms help held-out questions.

The goal is not a final model. It tests whether "surface narrowing + local
term navigation" has a learnable signal before scaling the idea.
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
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.bgb_browser_search.adapt_bgb_static_views import article_regions, summarize_ranks  # noqa: E402
from examples.bgb_browser_search.iterate_bgb_oracle_reformulation import (  # noqa: E402
    QueryResult,
    candidate_terms,
    document_frequency,
    first_rank,
    fuse_hits,
    greedy_oracle_terms,
    rank_with_terms,
    sample_article_refs,
)
from examples.bgb_browser_search.train_bgb_article_candidate_generator import (  # noqa: E402
    CandidateQuestion,
    encode,
    load_split_questions,
    summarize_groups,
)
from refmark.search_index import PortableBM25Index, load_search_index, tokenize  # noqa: E402

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This training example requires torch.") from exc


@dataclass(frozen=True)
class PredictorExample:
    row: CandidateQuestion
    query_ids: list[int]
    term_ids: list[int]
    oracle_terms: list[str]


class TermPredictor(nn.Module):
    def __init__(self, vocab_size: int, term_count: int, *, embed_dim: int, hidden_dim: int, layers: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        modules: list[nn.Module] = [nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.05)]
        for _ in range(max(layers - 1, 0)):
            modules.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.05)])
        modules.append(nn.Linear(hidden_dim, term_count))
        self.scorer = nn.Sequential(*modules)

    def forward(self, query_ids: torch.Tensor) -> torch.Tensor:
        mask = query_ids.ne(0).unsqueeze(-1).float()
        embedded = self.embedding(query_ids)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.scorer(pooled)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tiny query -> oracle expansion term predictor.")
    parser.add_argument("--index", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_openrouter_index.json")
    parser.add_argument("--stress-report", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--article-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--top-ks", default="1,3,5,10,20,50")
    parser.add_argument("--candidate-terms", type=int, default=80)
    parser.add_argument("--oracle-terms-per-query", type=int, default=3)
    parser.add_argument("--predict-terms", default="1,2,3,4,6")
    parser.add_argument("--fusion-alphas", default="0.05,0.1,0.15,0.25")
    parser.add_argument("--max-doc-freq-ratio", type=float, default=0.06)
    parser.add_argument("--query-vocab-size", type=int, default=12000)
    parser.add_argument("--max-query-tokens", type=int, default=72)
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--positive-weight", type=float, default=8.0)
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
    sampled_refs = sample_article_refs(train_rows, eval_rows, count=args.article_count, rng=random.Random(args.seed))
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
    train_oracles = {
        row.query: greedy_oracle_terms(
            bm25,
            row.query,
            row.article_ref,
            candidates_by_ref[row.article_ref],
            max_terms=args.oracle_terms_per_query,
            top_k=max(top_ks),
        )
        for row in train_rows
    }
    term_vocab = build_term_vocab(train_oracles, candidates_by_ref)
    query_vocab = build_query_vocab(train_rows, vocab_size=args.query_vocab_size)
    train_examples = build_examples(train_rows, train_oracles, query_vocab, term_vocab, max_query_tokens=args.max_query_tokens)
    eval_examples = build_examples(eval_rows, {}, query_vocab, term_vocab, max_query_tokens=args.max_query_tokens)
    train_examples_with_terms = [example for example in train_examples if example.term_ids]
    if not train_examples_with_terms:
        raise SystemExit("No train rows had oracle-improving terms.")

    model = TermPredictor(
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
    for epoch in range(1, args.epochs + 1):
        model.train()
        query_batch, target_batch = tensorize(train_examples_with_terms, term_count=len(term_vocab))
        optimizer.zero_grad(set_to_none=True)
        logits = model(query_batch)
        loss = F.binary_cross_entropy_with_logits(logits, target_batch, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch == args.epochs or epoch % 10 == 0:
            metrics = evaluate_predictor(
                model,
                bm25,
                eval_examples,
                candidates_by_ref,
                term_vocab,
                predict_terms=predict_terms,
                fusion_alphas=fusion_alphas,
                oracle_terms_per_query=args.oracle_terms_per_query,
                top_ks=top_ks,
            )
            best = metrics["best_fusion"]["metrics"]["article_hit_at_k"]
            row = {
                "epoch": epoch,
                "loss": round(float(loss.detach()), 6),
                "best_predict_terms": metrics["best_fusion"]["predict_terms"],
                "best_alpha": metrics["best_fusion"]["alpha"],
                "hit_at_10": best["hit_at_k"].get("10", 0.0),
                "mrr": best["mrr"],
            }
        else:
            row = {"epoch": epoch, "loss": round(float(loss.detach()), 6)}
        history.append(row)
        print(json.dumps(row))

    final = evaluate_predictor(
        model,
        bm25,
        eval_examples,
        candidates_by_ref,
        term_vocab,
        predict_terms=predict_terms,
        fusion_alphas=fusion_alphas,
        oracle_terms_per_query=args.oracle_terms_per_query,
        top_ks=top_ks,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "refmark.bgb_oracle_reformulation_predictor.v1",
            "settings": vars(args),
            "sampled_refs": sampled_refs,
            "query_vocab": query_vocab,
            "term_vocab": term_vocab,
            "model_state": model.state_dict(),
        },
        output,
    )
    report = {
        "schema": "refmark.bgb_oracle_reformulation_predictor_report.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": vars(args),
        "sampled_refs": sampled_refs,
        "train_questions": len(train_rows),
        "train_questions_with_oracle_terms": len(train_examples_with_terms),
        "eval_questions": len(eval_rows),
        "term_vocab_size": len(term_vocab),
        "query_vocab_size": len(query_vocab),
        "parameters": sum(param.numel() for param in model.parameters()),
        "artifact_bytes": output.stat().st_size,
        "artifact_megabytes": round(output.stat().st_size / 1_000_000, 4),
        "train_seconds": round(time.perf_counter() - started, 3),
        "train_oracle_terms": train_oracles,
        "history": history,
        "final": final,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def build_term_vocab(train_oracles: dict[str, list[str]], candidates_by_ref: dict[str, list[str]]) -> dict[str, int]:
    counts = Counter(term for terms in train_oracles.values() for term in terms)
    for terms in candidates_by_ref.values():
        for term in terms[:20]:
            counts[term] += 0.1
    return {term: index for index, (term, _count) in enumerate(counts.most_common())}


def build_query_vocab(rows: list[CandidateQuestion], *, vocab_size: int) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        counts.update(tokenize(row.query))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, _count in counts.most_common(max(vocab_size - len(vocab), 0)):
        vocab.setdefault(token, len(vocab))
    return vocab


def build_examples(
    rows: list[CandidateQuestion],
    oracle_terms: dict[str, list[str]],
    query_vocab: dict[str, int],
    term_vocab: dict[str, int],
    *,
    max_query_tokens: int,
) -> list[PredictorExample]:
    return [
        PredictorExample(
            row=row,
            query_ids=encode(row.query, query_vocab, max_query_tokens),
            term_ids=[term_vocab[term] for term in oracle_terms.get(row.query, []) if term in term_vocab],
            oracle_terms=oracle_terms.get(row.query, []),
        )
        for row in rows
    ]


def tensorize(examples: list[PredictorExample], *, term_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(example.query_ids) for example in examples)
    query_batch = torch.tensor([example.query_ids + [0] * (max_len - len(example.query_ids)) for example in examples], dtype=torch.long)
    target_batch = torch.zeros((len(examples), term_count), dtype=torch.float32)
    for row_index, example in enumerate(examples):
        target_batch[row_index, example.term_ids] = 1.0
    return query_batch, target_batch


def evaluate_predictor(
    model: TermPredictor,
    index: PortableBM25Index,
    examples: list[PredictorExample],
    candidates_by_ref: dict[str, list[str]],
    term_vocab: dict[str, int],
    *,
    predict_terms: tuple[int, ...],
    fusion_alphas: tuple[float, ...],
    oracle_terms_per_query: int,
    top_ks: tuple[int, ...],
) -> dict[str, object]:
    term_by_id = {index: term for term, index in term_vocab.items()}
    query_batch, _targets = tensorize(examples, term_count=len(term_vocab))
    model.eval()
    with torch.no_grad():
        logits_batch = model(query_batch)
    raw_ranks = []
    oracle_ranks = []
    predicted = {}
    fused = {}
    samples = []
    for count in predict_terms:
        predicted[str(count)] = []
        for alpha in fusion_alphas:
            fused[f"{count}:{alpha}"] = []
    for example, logits in zip(examples, logits_batch, strict=False):
        raw_hits = index.search(example.row.query, top_k=max(top_ks))
        raw_rank = first_rank([hit.stable_ref for hit in raw_hits], example.row.article_ref)
        raw_ranks.append(raw_rank)
        oracle_terms = greedy_oracle_terms(
            index,
            example.row.query,
            example.row.article_ref,
            candidates_by_ref[example.row.article_ref],
            max_terms=oracle_terms_per_query,
            top_k=max(top_ks),
        )
        oracle_ranks.append(rank_with_terms(index, example.row.query, example.row.article_ref, oracle_terms, top_k=max(top_ks), fusion_alpha=1.0))
        ordered_terms = [term_by_id[idx] for idx in torch.argsort(logits, descending=True).tolist()]
        for count in predict_terms:
            terms = ordered_terms[:count]
            expanded_hits = index.search(f"{example.row.query} {' '.join(terms)}", top_k=max(top_ks))
            predicted[str(count)].append(first_rank([hit.stable_ref for hit in expanded_hits], example.row.article_ref))
            for alpha in fusion_alphas:
                fused_refs = fuse_hits(raw_hits, expanded_hits, alpha=alpha)
                fused[f"{count}:{alpha}"].append(first_rank(fused_refs[: max(top_ks)], example.row.article_ref))
        if len(samples) < 15:
            samples.append(
                QueryResult(
                    query=example.row.query,
                    article_ref=example.row.article_ref,
                    language=example.row.language,
                    style=example.row.style,
                    raw_rank=raw_rank,
                    learned_rank=fused[f"{predict_terms[0]}:{fusion_alphas[0]}"][-1],
                    oracle_rank=oracle_ranks[-1],
                    learned_terms=ordered_terms[: predict_terms[0]],
                    oracle_terms=oracle_terms,
                ).to_dict()
            )
    predicted_summary = {count: {"article_hit_at_k": summarize_ranks(ranks, top_ks=top_ks)} for count, ranks in predicted.items()}
    fused_summary = {key: {"article_hit_at_k": summarize_ranks(ranks, top_ks=top_ks)} for key, ranks in fused.items()}
    best_fusion_key, best_fusion = max(
        fused_summary.items(),
        key=lambda item: (
            item[1]["article_hit_at_k"]["hit_at_k"].get("10", 0.0),
            item[1]["article_hit_at_k"]["mrr"],
        ),
    )
    count, alpha = best_fusion_key.split(":")
    return {
        "raw_bm25": {"article_hit_at_k": summarize_ranks(raw_ranks, top_ks=top_ks)},
        "per_query_oracle": {"article_hit_at_k": summarize_ranks(oracle_ranks, top_ks=top_ks)},
        "predicted_append": predicted_summary,
        "predicted_fusion": fused_summary,
        "best_fusion": {"predict_terms": int(count), "alpha": float(alpha), "metrics": best_fusion},
        "sample_results": samples,
    }


if __name__ == "__main__":
    main()
