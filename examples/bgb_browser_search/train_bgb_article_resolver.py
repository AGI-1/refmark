"""Train a small static BGB article resolver on generated stress questions.

This is the distilled/no-vector-runtime path: BM25 produces article candidates,
then a tiny pair scorer reranks those candidates. Stress questions are split
within each article block so held-out questions are not pasted into retrieval
metadata.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.bgb_browser_search.adapt_bgb_static_views import article_regions, split_questions_by_block, stress_questions  # noqa: E402
from examples.portable_search_index.train_refmark_resolver import (  # noqa: E402
    PairResolver,
    ResolverQuestion,
    _batches,
    _build_examples,
    _build_idf,
    _build_vocab,
    _group_rows,
    _ranking_loss,
    _tensorize,
    evaluate_baseline,
    evaluate_model,
)
from refmark.search_index import PortableBM25Index, load_search_index  # noqa: E402

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This training example requires torch.") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny BGB article resolver on stress questions.")
    parser.add_argument("--index", default="examples/bgb_browser_search/output_full_qwen_turbo/bgb_openrouter_index.json")
    parser.add_argument("--stress-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--candidate-k", type=int, default=80)
    parser.add_argument("--top-ks", default="1,3,5,10,20,50")
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=727)
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--max-query-tokens", type=int, default=64)
    parser.add_argument("--max-region-tokens", type=int, default=384)
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0012)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--pos-weight", type=float, default=10.0)
    parser.add_argument("--loss", choices=["bce", "pairwise", "hybrid"], default="hybrid")
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--vector-features", action="store_true")
    parser.add_argument("--blend-alphas", default="0,0.1,0.2,0.35,0.5,0.65,0.8,1")
    args = parser.parse_args()
    args.coarse_mode = "anchor"

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    top_ks = tuple(int(part) for part in args.top_ks.split(",") if part.strip())
    blend_alphas = tuple(float(part) for part in args.blend_alphas.split(",") if part.strip())

    source_index = load_search_index(args.index)
    index = PortableBM25Index(article_regions(source_index.regions), include_source=True)
    stress = json.loads(Path(args.stress_report).read_text(encoding="utf-8"))
    train_stress, eval_stress = split_questions_by_block(
        stress_questions(stress),
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    train_questions = to_resolver_questions(train_stress)
    eval_questions = to_resolver_questions(eval_stress)

    vocab = _build_vocab(index, train_questions + eval_questions, vocab_size=args.vocab_size)
    idf = _build_idf(index)
    train_rows = _build_examples(index, train_questions, vocab=vocab, idf=idf, args=args)
    eval_rows = _build_examples(index, eval_questions, vocab=vocab, idf=idf, args=args)
    if not train_rows or not eval_rows:
        raise SystemExit("Not enough train/eval rows. Check candidate recall and stress report.")

    model = PairResolver(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        feature_dim=len(train_rows[0].features),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight))
    train_groups = _group_rows(train_rows)
    history: list[dict[str, object]] = []
    best_score = -1.0
    best_state = copy.deepcopy(model.state_dict())
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        if args.loss == "bce":
            random.shuffle(train_rows)
            batches = _batches(train_rows, args.batch_size)
        else:
            grouped = list(train_groups.values())
            random.shuffle(grouped)
            batches = grouped
        for batch in batches:
            query_batch, region_batch, feature_batch, label_batch = _tensorize(batch)
            optimizer.zero_grad(set_to_none=True)
            logits = model(query_batch, region_batch, feature_batch)
            if args.loss == "bce":
                loss = loss_fn(logits, label_batch)
            else:
                loss = _ranking_loss(logits, label_batch, margin=args.margin)
                if args.loss == "hybrid":
                    loss = loss + (0.25 * loss_fn(logits, label_batch))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        eval_metrics = evaluate_model(
            index,
            model,
            eval_rows,
            eval_questions,
            top_ks=top_ks,
            candidate_k=args.candidate_k,
            blend_alpha=1.0,
            coarse_mode="anchor",
        )
        score = float(eval_metrics["mrr"]) + float(eval_metrics["resolver_hit_at_k"]["1"])
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
        row = {
            "epoch": epoch,
            "loss": round(sum(losses) / max(len(losses), 1), 6),
            "eval_hit_at_1": eval_metrics["resolver_hit_at_k"]["1"],
            "eval_hit_at_10": eval_metrics["resolver_hit_at_k"]["10"],
            "eval_mrr": eval_metrics["mrr"],
        }
        history.append(row)
        print(json.dumps(row))

    train_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    baseline = evaluate_baseline(index, eval_questions, top_ks=top_ks, candidate_k=args.candidate_k, coarse_mode="anchor")
    resolver = evaluate_model(
        index,
        model,
        eval_rows,
        eval_questions,
        top_ks=top_ks,
        candidate_k=args.candidate_k,
        blend_alpha=1.0,
        coarse_mode="anchor",
    )
    blends = {
        str(alpha): evaluate_model(
            index,
            model,
            eval_rows,
            eval_questions,
            top_ks=top_ks,
            candidate_k=args.candidate_k,
            blend_alpha=alpha,
            coarse_mode="anchor",
        )
        for alpha in blend_alphas
    }
    best_blend_alpha, best_blend = max(
        blends.items(),
        key=lambda item: (float(item[1]["resolver_hit_at_k"]["10"]), float(item[1]["mrr"])),
    )
    payload = {
        "schema": "refmark.bgb_article_resolver.v1",
        "settings": vars(args),
        "vocab": vocab,
        "model_state": model.state_dict(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    artifact_bytes = output.stat().st_size
    report = {
        "schema": "refmark.bgb_article_resolver_report.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_index": args.index,
        "stress_report": args.stress_report,
        "settings": vars(args),
        "article_count": len(index.regions),
        "train_questions": len(train_questions),
        "eval_questions": len(eval_questions),
        "train_candidate_rows": len(train_rows),
        "eval_candidate_rows": len(eval_rows),
        "parameters": sum(param.numel() for param in model.parameters()),
        "artifact_bytes": artifact_bytes,
        "artifact_megabytes": round(artifact_bytes / 1_000_000, 4),
        "train_seconds": round(train_seconds, 3),
        "history": history,
        "best_epoch": max(history, key=lambda row: float(row["eval_mrr"]) + float(row["eval_hit_at_1"]))["epoch"],
        "baseline": baseline,
        "resolver": resolver,
        "blends": blends,
        "best_blend_alpha": best_blend_alpha,
        "best_blend": best_blend,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def to_resolver_questions(rows) -> list[ResolverQuestion]:
    output: list[ResolverQuestion] = []
    for row in rows:
        doc_id, _article_id = row.block_id.split(":", 1)
        output.append(
            ResolverQuestion(
                query=row.query,
                doc_id=doc_id,
                gold_refs=[row.block_id],
                source=row.source,
                gold_mode=f"{row.language}/{row.style}",
            )
        )
    return output


if __name__ == "__main__":
    main()
