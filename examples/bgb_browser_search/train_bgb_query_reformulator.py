"""Train a tiny BGB query reformulator for local lexical retrieval.

The model predicts corpus-local expansion terms from a natural query. At
runtime those terms are appended to the user query before BM25 search. This is
not a semantic embedding replacement; it is a small, inspectable bridge from
concern-style wording to the vocabulary used in the refmarked corpus.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.bgb_browser_search.adapt_bgb_static_views import article_regions, summarize_ranks  # noqa: E402
from examples.bgb_browser_search.train_bgb_article_candidate_generator import (  # noqa: E402
    CandidateQuestion,
    encode,
    load_split_questions,
    summarize_groups,
)
from refmark.search_index import PortableBM25Index, SearchRegion, load_search_index, tokenize  # noqa: E402

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This training example requires torch.") from exc


@dataclass(frozen=True)
class ReformulationExample:
    row: CandidateQuestion
    query_ids: list[int]
    target_ids: list[int]


class QueryReformulator(nn.Module):
    def __init__(self, query_vocab_size: int, term_count: int, *, embed_dim: int, hidden_dim: int, layers: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(query_vocab_size, embed_dim, padding_idx=0)
        modules: list[nn.Module] = [nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.08)]
        for _ in range(max(layers - 1, 0)):
            modules.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.08)])
        modules.append(nn.Linear(hidden_dim, term_count))
        self.classifier = nn.Sequential(*modules)

    def forward(self, query_ids: torch.Tensor) -> torch.Tensor:
        mask = query_ids.ne(0).unsqueeze(-1).float()
        embedded = self.embedding(query_ids)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.classifier(pooled)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small query -> expansion terms model for BGB BM25.")
    parser.add_argument("--index", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_openrouter_index.json")
    parser.add_argument("--stress-report", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=737)
    parser.add_argument("--top-ks", default="1,3,5,10,20,50")
    parser.add_argument("--query-vocab-size", type=int, default=32000)
    parser.add_argument("--term-vocab-size", type=int, default=8000)
    parser.add_argument("--target-terms-per-article", type=int, default=48)
    parser.add_argument("--max-query-tokens", type=int, default=72)
    parser.add_argument("--predict-terms", default="4,8,12,16,24")
    parser.add_argument("--fusion-alphas", default="0.05,0.1,0.2,0.35")
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--positive-weight", type=float, default=18.0)
    parser.add_argument("--ref-term-weight", type=float, default=0.7)
    parser.add_argument("--max-term-doc-freq-ratio", type=float, default=0.08)
    parser.add_argument("--idf-logit-weight", type=float, default=0.35)
    parser.add_argument("--eval-every", type=int, default=4)
    parser.add_argument("--epoch-eval-limit", type=int, default=600)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    top_ks = tuple(int(part) for part in args.top_ks.split(",") if part.strip())
    predict_terms = tuple(int(part) for part in args.predict_terms.split(",") if part.strip())
    fusion_alphas = tuple(float(part) for part in args.fusion_alphas.split(",") if part.strip())

    source_index = load_search_index(args.index)
    articles = article_regions(source_index.regions)
    article_by_ref = {article.stable_ref: article for article in articles}
    bm25_index = PortableBM25Index(articles, include_source=True)
    train_rows, eval_rows = load_split_questions(args.stress_report, train_fraction=args.train_fraction, seed=args.seed)
    train_rows = [row for row in train_rows if row.article_ref in article_by_ref]
    eval_rows = [row for row in eval_rows if row.article_ref in article_by_ref]

    query_vocab = build_query_vocab(articles, train_rows, vocab_size=args.query_vocab_size)
    term_vocab, article_targets, term_idf = build_term_targets(
        articles,
        vocab_size=args.term_vocab_size,
        target_terms_per_article=args.target_terms_per_article,
        ref_term_weight=args.ref_term_weight,
        max_doc_freq_ratio=args.max_term_doc_freq_ratio,
    )
    term_by_id = {index: term for term, index in term_vocab.items()}
    train_examples = build_examples(train_rows, query_vocab, article_targets, args.max_query_tokens)
    eval_examples = build_examples(eval_rows, query_vocab, article_targets, args.max_query_tokens)
    if not train_examples or not eval_examples:
        raise SystemExit("No train/eval examples matched article targets.")

    model = QueryReformulator(
        len(query_vocab),
        len(term_vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    pos_weight = torch.full((len(term_vocab),), float(args.positive_weight))
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_score = -1.0
    history: list[dict[str, object]] = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for query_batch, target_batch in make_batches(train_examples, len(term_vocab), args.batch_size, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            logits = model(query_batch)
            loss = F.binary_cross_entropy_with_logits(logits, target_batch, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        should_eval = epoch == 1 or epoch == args.epochs or epoch % max(args.eval_every, 1) == 0
        metrics = (
            evaluate_expansion(
                model,
                bm25_index,
                eval_examples[: max(args.epoch_eval_limit, 1)],
                term_by_id,
                term_idf,
                predict_terms=predict_terms,
                top_ks=top_ks,
                idf_logit_weight=args.idf_logit_weight,
                fusion_alphas=fusion_alphas,
            )
            if should_eval
            else None
        )
        if metrics is None:
            row = {
                "epoch": epoch,
                "loss": round(sum(losses) / max(len(losses), 1), 6),
                "best_predict_terms": None,
                "hit_at_1": None,
                "hit_at_10": None,
                "mrr": None,
            }
            history.append(row)
            print(json.dumps(row))
            continue
        best_terms = str(metrics["best_predict_terms"])
        score = metrics["expanded"][best_terms]["article_hit_at_k"]["mrr"] + metrics["expanded"][best_terms]["article_hit_at_k"]["hit_at_k"].get("10", 0.0)
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        row = {
            "epoch": epoch,
            "loss": round(sum(losses) / max(len(losses), 1), 6),
            "best_predict_terms": metrics["best_predict_terms"],
            "hit_at_1": metrics["expanded"][best_terms]["article_hit_at_k"]["hit_at_k"].get("1", 0.0),
            "hit_at_10": metrics["expanded"][best_terms]["article_hit_at_k"]["hit_at_k"].get("10", 0.0),
            "mrr": metrics["expanded"][best_terms]["article_hit_at_k"]["mrr"],
        }
        history.append(row)
        print(json.dumps(row))

    train_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    baseline = evaluate_bm25(bm25_index, eval_rows, top_ks=top_ks)
    final = evaluate_expansion(
        model,
        bm25_index,
        eval_examples,
        term_by_id,
        term_idf,
        predict_terms=predict_terms,
        top_ks=top_ks,
        idf_logit_weight=args.idf_logit_weight,
        fusion_alphas=fusion_alphas,
    )
    payload = {
        "schema": "refmark.bgb_query_reformulator.v1",
        "settings": vars(args),
        "query_vocab": query_vocab,
        "term_vocab": term_vocab,
        "term_idf": term_idf,
        "model_state": model.state_dict(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    artifact_bytes = output.stat().st_size
    report = {
        "schema": "refmark.bgb_query_reformulator_report.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_index": args.index,
        "stress_reports": args.stress_report,
        "settings": vars(args),
        "article_count": len(articles),
        "train_questions": len(train_rows),
        "eval_questions": len(eval_rows),
        "term_vocab_size": len(term_vocab),
        "query_vocab_size": len(query_vocab),
        "parameters": sum(param.numel() for param in model.parameters()),
        "artifact_bytes": artifact_bytes,
        "artifact_megabytes": round(artifact_bytes / 1_000_000, 4),
        "train_seconds": round(train_seconds, 3),
        "history": history,
        "best_epoch": max(
            (row for row in history if row["mrr"] is not None),
            key=lambda row: float(row["mrr"]) + float(row["hit_at_10"]),
        )["epoch"],
        "bm25_article_baseline": baseline,
        "query_reformulator": final,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def build_query_vocab(articles: list[SearchRegion], train_rows: list[CandidateQuestion], *, vocab_size: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in train_rows:
        counts.update(tokenize(row.query))
    for article in articles:
        counts.update(tokenize(article.index_text(include_source=True)))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, _count in counts.most_common(max(vocab_size - len(vocab), 0)):
        vocab.setdefault(token, len(vocab))
    return vocab


def build_term_targets(
    articles: list[SearchRegion],
    *,
    vocab_size: int,
    target_terms_per_article: int,
    ref_term_weight: float,
    max_doc_freq_ratio: float,
) -> tuple[dict[str, int], dict[str, list[int]], list[float]]:
    doc_freq: Counter[str] = Counter()
    article_counts: dict[str, Counter[str]] = {}
    for article in articles:
        counts = candidate_term_counts(article, ref_term_weight=ref_term_weight)
        article_counts[article.stable_ref] = counts
        doc_freq.update(counts.keys())
    article_count = max(len(articles), 1)
    article_targets_text: dict[str, list[str]] = {}
    for stable_ref, counts in article_counts.items():
        scored = []
        for term, count in counts.items():
            if doc_freq[term] / article_count > max_doc_freq_ratio:
                continue
            idf = 1.0 + (article_count / max(doc_freq[term], 1))
            scored.append((count * idf, term))
        scored.sort(key=lambda item: (-item[0], item[1]))
        article_targets_text[stable_ref] = [term for _score, term in scored[:target_terms_per_article]]
    selected_counts = Counter(term for terms in article_targets_text.values() for term in terms)
    term_vocab = {term: index for index, (term, _count) in enumerate(selected_counts.most_common(vocab_size))}
    article_targets = {
        stable_ref: [term_vocab[term] for term in terms if term in term_vocab]
        for stable_ref, terms in article_targets_text.items()
    }
    term_idf = [0.0] * len(term_vocab)
    for term, index in term_vocab.items():
        term_idf[index] = 1.0 + (article_count / max(doc_freq[term], 1))
    max_idf = max(term_idf) if term_idf else 1.0
    term_idf = [value / max_idf for value in term_idf]
    return term_vocab, article_targets, term_idf


def candidate_term_counts(article: SearchRegion, *, ref_term_weight: float) -> Counter[str]:
    text = "\n".join([article.view.summary, *article.view.questions, *article.view.keywords])
    counts = Counter(term for term in tokenize(text) if keep_term(term))
    for term in tokenize(article.text):
        if keep_term(term):
            counts[term] += 0.25
    for term in tokenize(article.stable_ref.replace(":", " ")):
        if keep_term(term):
            counts[term] += ref_term_weight
    return counts


def keep_term(term: str) -> bool:
    if len(term) < 3:
        return False
    if term in {"und", "oder", "the", "and", "for", "der", "die", "das", "ein", "eine", "mit", "von"}:
        return False
    return True


def build_examples(
    rows: list[CandidateQuestion],
    query_vocab: dict[str, int],
    article_targets: dict[str, list[int]],
    max_query_tokens: int,
) -> list[ReformulationExample]:
    output = []
    for row in rows:
        targets = article_targets.get(row.article_ref, [])
        if not targets:
            continue
        output.append(
            ReformulationExample(
                row=row,
                query_ids=encode(row.query, query_vocab, max_query_tokens),
                target_ids=targets,
            )
        )
    return output


def make_batches(examples: list[ReformulationExample], term_count: int, batch_size: int, *, shuffle: bool):
    ordered = list(examples)
    if shuffle:
        random.shuffle(ordered)
    for start in range(0, len(ordered), batch_size):
        batch = ordered[start : start + batch_size]
        max_len = max(len(item.query_ids) for item in batch)
        query_batch = torch.tensor([item.query_ids + [0] * (max_len - len(item.query_ids)) for item in batch], dtype=torch.long)
        target_batch = torch.zeros((len(batch), term_count), dtype=torch.float32)
        for row_index, item in enumerate(batch):
            target_batch[row_index, item.target_ids] = 1.0
        yield query_batch, target_batch


def evaluate_bm25(index: PortableBM25Index, rows: list[CandidateQuestion], *, top_ks: tuple[int, ...]) -> dict[str, object]:
    ranks = []
    by_language: dict[str, list[int | None]] = {}
    by_style: dict[str, list[int | None]] = {}
    for row in rows:
        hits = index.search(row.query, top_k=max(top_ks))
        rank = first_rank([hit.stable_ref for hit in hits], row.article_ref)
        ranks.append(rank)
        by_language.setdefault(row.language, []).append(rank)
        by_style.setdefault(row.style, []).append(rank)
    return {
        "article_hit_at_k": summarize_ranks(ranks, top_ks=top_ks),
        "by_language": summarize_groups(by_language, top_ks=top_ks),
        "by_style": summarize_groups(by_style, top_ks=top_ks),
    }


def evaluate_expansion(
    model: QueryReformulator,
    index: PortableBM25Index,
    examples: list[ReformulationExample],
    term_by_id: dict[int, str],
    term_idf: list[float],
    *,
    predict_terms: tuple[int, ...],
    top_ks: tuple[int, ...],
    idf_logit_weight: float,
    fusion_alphas: tuple[float, ...],
) -> dict[str, object]:
    model.eval()
    expanded: dict[str, dict[str, object]] = {}
    fused: dict[str, dict[str, object]] = {}
    term_samples = []
    with torch.no_grad():
        logits_rows = []
        for item in examples:
            query_batch = torch.tensor([item.query_ids], dtype=torch.long)
            logits = model(query_batch)[0]
            if term_idf and idf_logit_weight:
                logits = logits + (torch.tensor(term_idf, dtype=logits.dtype) * idf_logit_weight)
            logits_rows.append(logits)
        for count in predict_terms:
            ranks = []
            fusion_ranks: dict[str, list[int | None]] = {str(alpha): [] for alpha in fusion_alphas}
            by_language: dict[str, list[int | None]] = {}
            by_style: dict[str, list[int | None]] = {}
            for item, logits in zip(examples, logits_rows, strict=False):
                top_ids = torch.topk(logits, k=min(count, len(term_by_id))).indices.tolist()
                terms = [term_by_id[index] for index in top_ids]
                expanded_query = f"{item.row.query} {' '.join(terms)}"
                raw_hits = index.search(item.row.query, top_k=max(top_ks))
                hits = index.search(expanded_query, top_k=max(top_ks))
                rank = first_rank([hit.stable_ref for hit in hits], item.row.article_ref)
                ranks.append(rank)
                by_language.setdefault(item.row.language, []).append(rank)
                by_style.setdefault(item.row.style, []).append(rank)
                for alpha in fusion_alphas:
                    fused_refs = fuse_hits(raw_hits, hits, alpha=alpha)
                    fusion_ranks[str(alpha)].append(first_rank(fused_refs, item.row.article_ref))
                if count == predict_terms[0] and len(term_samples) < 20:
                    term_samples.append({"query": item.row.query, "gold_ref": item.row.article_ref, "terms": terms})
            expanded[str(count)] = {
                "article_hit_at_k": summarize_ranks(ranks, top_ks=top_ks),
                "by_language": summarize_groups(by_language, top_ks=top_ks),
                "by_style": summarize_groups(by_style, top_ks=top_ks),
            }
            fused[str(count)] = {
                alpha: {"article_hit_at_k": summarize_ranks(alpha_ranks, top_ks=top_ks)}
                for alpha, alpha_ranks in fusion_ranks.items()
            }
    best_count, _metrics = max(
        expanded.items(),
        key=lambda item: (
            item[1]["article_hit_at_k"]["hit_at_k"].get("10", 0.0),
            item[1]["article_hit_at_k"]["mrr"],
        ),
    )
    best_fusion = None
    if fused:
        best_fusion = max(
            (
                (count, alpha, metrics)
                for count, rows in fused.items()
                for alpha, metrics in rows.items()
            ),
            key=lambda item: (
                item[2]["article_hit_at_k"]["hit_at_k"].get("10", 0.0),
                item[2]["article_hit_at_k"]["mrr"],
            ),
        )
    return {
        "best_predict_terms": int(best_count),
        "expanded": expanded,
        "fused_with_raw_bm25": fused,
        "best_fusion": {
            "predict_terms": int(best_fusion[0]),
            "alpha": float(best_fusion[1]),
            "metrics": best_fusion[2],
        }
        if best_fusion
        else None,
        "term_samples": term_samples,
    }


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


if __name__ == "__main__":
    main()
