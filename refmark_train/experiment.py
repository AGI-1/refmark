from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path

import numpy as np

from refmark_train.backends import detect_backend
from refmark_train.models import (
    BM25AnchorRetriever,
    KeywordOverlapBaseline,
    TfidfAnchorRetriever,
    TinyBoWClassifier,
    TinyEmbeddingClassifier,
    TorchBoWClassifier,
    TorchBoWCenterWidthPredictor,
    TorchBoWMLPClassifier,
    TorchBoWStartEndPredictor,
    Vocab,
    build_label_index,
    encode_examples,
    summarize_logits,
    topk_accuracy,
)
from refmark_train.real_corpus import load_bundle_from_dir
from refmark_train.synthetic import CorpusBundle, build_corpus, tokenize
from refmark_train.synthetic import save_bundle


RUNS_DIR = Path(__file__).resolve().parent / "runs"
DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass
class ExperimentConfig:
    anchors: int = 100
    seed: int = 13
    epochs: int = 60
    backend: str = "auto"
    embedding_dim: int = 24
    hidden_dim: int = 48
    mlp_hidden_dim: int = 512
    mlp_hidden2_dim: int = 0
    mlp_dropout: float = 0.2
    hybrid_exact_weight: float = 0.7
    soft_cover_reward: float = 0.92
    soft_overlap_base: float = 0.35
    soft_overlap_f1_weight: float = 0.45
    soft_overlap_precision_weight: float = 0.2
    soft_cover_breadth_penalty: float = 8.0
    soft_overlap_breadth_penalty: float = 10.0
    soft_cover_penalty_cap: float = 0.35
    soft_overlap_penalty_cap: float = 0.25
    soft_local_reward: float = 0.08
    soft_local_distance: int = 40
    range_exact_weight: float = 0.8
    range_neighbor_sigma: float = 1.5
    range_breadth_penalty: float = 0.15
    range_soft_weight: float = 0.2
    range_width_weight: float = 0.1
    center_exact_weight: float = 0.75
    center_soft_weight: float = 0.2
    center_width_weight: float = 0.05
    bm25_weight: float = 0.35
    bm25_top_k: int = 8
    bm25_margin: int = 2
    bm25_outside_penalty: float = 1.5
    bm25_enrich_train: bool = False
    broad_rerank_alpha: float = 1.0
    broad_rerank_beta: float = 0.8
    broad_rerank_gamma: float = 0.8
    broad_rerank_delta: float = 0.08
    learned_rerank_l2: float = 1e-3
    learned_rerank_center_top_k: int = 5
    learned_rerank_width_top_k: int = 5
    learned_rerank_hidden_dim: int = 64
    learned_rerank_epochs: int = 180
    learned_rerank_dropout: float = 0.1
    local_cross_hidden_dim: int = 128
    local_cross_epochs: int = 120
    local_cross_dropout: float = 0.1
    local_cross_margin_threshold: float = 0.35
    local_cross_blend_steps: int = 5
    local_cross_max_candidates: int = 24
    route_bucket_size: int = 0
    route_top_k: int = 1
    route_margin: int = 0
    route_outside_penalty: float = 1.2
    count_aux_weight: float = 0.10
    count_label_cap: int = 8
    pairwise_rerank_margin: float = 0.20
    fusion_anchor_weight: float = 0.6
    fusion_range_weight: float = 1.0
    fusion_breadth_penalty: float = 0.08
    learning_rate: float = 0.25
    batch_size: int = 32
    weight_decay: float = 1e-4


def _evaluate_bundle(bundle: CorpusBundle, config: ExperimentConfig) -> dict:
    resolved_backend = detect_backend(config.backend)
    vocab = Vocab.build(
        [example.question for example in bundle.train]
        + [anchor.text for anchor in bundle.anchors]
    )
    label_index = build_label_index(bundle.train)

    train_x, train_y = encode_examples(bundle.train, vocab, label_index)
    valid_x, valid_y = encode_examples(bundle.valid, vocab, label_index)
    reform_x, reform_y = encode_examples(bundle.reformulated, vocab, label_index)

    anchors_by_refmark = {anchor.refmark: anchor for anchor in bundle.anchors}

    baseline = KeywordOverlapBaseline()
    baseline.fit(train_x, train_y)
    baseline_valid = _summarize_with_regions(
        bundle.valid,
        baseline.logits(valid_x, len(label_index)),
        valid_y,
        label_index,
        anchors_by_refmark,
    )
    baseline_reform = _summarize_with_regions(
        bundle.reformulated,
        baseline.logits(reform_x, len(label_index)),
        reform_y,
        label_index,
        anchors_by_refmark,
    )

    retriever = TfidfAnchorRetriever(vocab)
    retriever.fit(
        [anchor.text for anchor in bundle.anchors],
        [anchor.refmark for anchor in bundle.anchors],
    )
    retrieval_valid = _summarize_with_regions(
        bundle.valid,
        retriever.logits([example.question for example in bundle.valid], label_index),
        valid_y,
        label_index,
        anchors_by_refmark,
    )
    retrieval_reform = _summarize_with_regions(
        bundle.reformulated,
        retriever.logits([example.question for example in bundle.reformulated], label_index),
        reform_y,
        label_index,
        anchors_by_refmark,
    )
    bm25 = BM25AnchorRetriever(vocab)
    bm25.fit(
        [anchor.text for anchor in bundle.anchors],
        [anchor.refmark for anchor in bundle.anchors],
    )
    bm25_valid = _summarize_with_regions(
        bundle.valid,
        bm25.logits([example.question for example in bundle.valid], label_index),
        valid_y,
        label_index,
        anchors_by_refmark,
    )
    bm25_reform = _summarize_with_regions(
        bundle.reformulated,
        bm25.logits([example.question for example in bundle.reformulated], label_index),
        reform_y,
        label_index,
        anchors_by_refmark,
    )

    if resolved_backend in {"directml", "torchcpu", "cuda"}:
        bow_train_x = TorchBoWClassifier.counts_matrix(train_x, len(vocab.id_to_token))
        bow_valid_x = TorchBoWClassifier.counts_matrix(valid_x, len(vocab.id_to_token))
        bow_reform_x = TorchBoWClassifier.counts_matrix(reform_x, len(vocab.id_to_token))
        device = "privateuseone:0" if resolved_backend == "directml" else ("cuda" if resolved_backend == "cuda" else "cpu")
        bow_model = TorchBoWClassifier(
            vocab_size=len(vocab.id_to_token),
            num_labels=len(label_index),
            device=device,
        )
        mlp_model = TorchBoWMLPClassifier(
            vocab_size=len(vocab.id_to_token),
            num_labels=len(label_index),
            device=device,
            hidden_dim=config.mlp_hidden_dim,
            hidden2_dim=config.mlp_hidden2_dim,
            dropout=config.mlp_dropout,
        )
        soft_mlp_model = TorchBoWMLPClassifier(
            vocab_size=len(vocab.id_to_token),
            num_labels=len(label_index),
            device=device,
            hidden_dim=config.mlp_hidden_dim,
            hidden2_dim=config.mlp_hidden2_dim,
            dropout=config.mlp_dropout,
        )
        hybrid_mlp_model = TorchBoWMLPClassifier(
            vocab_size=len(vocab.id_to_token),
            num_labels=len(label_index),
            device=device,
            hidden_dim=config.mlp_hidden_dim,
            hidden2_dim=config.mlp_hidden2_dim,
            dropout=config.mlp_dropout,
        )
    else:
        bow_train_x = TinyBoWClassifier._counts_matrix(train_x, len(vocab.id_to_token))
        bow_valid_x = TinyBoWClassifier._counts_matrix(valid_x, len(vocab.id_to_token))
        bow_reform_x = TinyBoWClassifier._counts_matrix(reform_x, len(vocab.id_to_token))
        bow_model = TinyBoWClassifier(
            vocab_size=len(vocab.id_to_token),
            num_labels=len(label_index),
            seed=config.seed,
        )
        mlp_model = None
        soft_mlp_model = None
        hybrid_mlp_model = None
    bow_losses = bow_model.train(
        bow_train_x,
        train_y,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        weight_decay=config.weight_decay,
        seed=config.seed,
    )

    stronger_model_results: dict[str, object] | None = None
    soft_target_model_results: dict[str, object] | None = None
    hybrid_model_results: dict[str, object] | None = None
    if mlp_model is not None:
        mlp_losses = mlp_model.train(
            bow_train_x,
            train_y,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            weight_decay=config.weight_decay,
            seed=config.seed,
        )
        stronger_model_results = {
            "train": summarize_logits(mlp_model.logits(bow_train_x), train_y),
            "valid": _summarize_with_regions(
                bundle.valid,
                mlp_model.logits(bow_valid_x),
                valid_y,
                label_index,
                anchors_by_refmark,
            ),
            "reformulated": _summarize_with_regions(
                bundle.reformulated,
                mlp_model.logits(bow_reform_x),
                reform_y,
                label_index,
                anchors_by_refmark,
            ),
            "loss_curve": mlp_losses,
        }
        soft_targets = _build_soft_targets(bundle.train, label_index, anchors_by_refmark, config)
        soft_losses = soft_mlp_model.train_soft(
            bow_train_x,
            soft_targets,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            weight_decay=config.weight_decay,
            seed=config.seed,
        )
        soft_target_model_results = {
            "train": summarize_logits(soft_mlp_model.logits(bow_train_x), train_y),
            "valid": _summarize_with_regions(
                bundle.valid,
                soft_mlp_model.logits(bow_valid_x),
                valid_y,
                label_index,
                anchors_by_refmark,
            ),
            "reformulated": _summarize_with_regions(
                bundle.reformulated,
                soft_mlp_model.logits(bow_reform_x),
                reform_y,
                label_index,
                anchors_by_refmark,
            ),
            "loss_curve": soft_losses,
        }
        hybrid_losses = hybrid_mlp_model.train_hybrid(
            bow_train_x,
            train_y,
            soft_targets,
            exact_weight=config.hybrid_exact_weight,
            soft_weight=1.0 - config.hybrid_exact_weight,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            weight_decay=config.weight_decay,
            seed=config.seed,
        )
        hybrid_model_results = {
            "train": summarize_logits(hybrid_mlp_model.logits(bow_train_x), train_y),
            "valid": _summarize_with_regions(
                bundle.valid,
                hybrid_mlp_model.logits(bow_valid_x),
                valid_y,
                label_index,
                anchors_by_refmark,
            ),
            "reformulated": _summarize_with_regions(
                bundle.reformulated,
                hybrid_mlp_model.logits(bow_reform_x),
                reform_y,
                label_index,
                anchors_by_refmark,
            ),
            "loss_curve": hybrid_losses,
        }

    model = TinyEmbeddingClassifier(
        vocab_size=len(vocab.id_to_token),
        num_labels=len(label_index),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        seed=config.seed,
    )
    losses = model.train(
        train_x,
        train_y,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        weight_decay=config.weight_decay,
        seed=config.seed,
    )
    train_metrics = summarize_logits(model.logits(train_x), train_y)
    valid_metrics = summarize_logits(model.logits(valid_x), valid_y)
    reform_metrics = summarize_logits(model.logits(reform_x), reform_y)

    return {
        "config": asdict(config),
        "dataset": {
            "anchor_count": len(bundle.anchors),
            "train_examples": len(bundle.train),
            "valid_examples": len(bundle.valid),
            "reformulated_examples": len(bundle.reformulated),
            "vocab_size": len(vocab.id_to_token),
            "backend": resolved_backend,
        },
        "baseline": {
            "valid": baseline_valid,
            "reformulated": baseline_reform,
        },
        "retrieval_baseline": {
            "valid": retrieval_valid,
            "reformulated": retrieval_reform,
        },
        "bm25_baseline": {
            "valid": bm25_valid,
            "reformulated": bm25_reform,
        },
        "tiny_model": {
            "train": summarize_logits(bow_model.logits(bow_train_x), train_y),
            "valid": _summarize_with_regions(
                bundle.valid,
                bow_model.logits(bow_valid_x),
                valid_y,
                label_index,
                anchors_by_refmark,
            ),
            "reformulated": _summarize_with_regions(
                bundle.reformulated,
                bow_model.logits(bow_reform_x),
                reform_y,
                label_index,
                anchors_by_refmark,
            ),
            "loss_curve": bow_losses,
        },
        "stronger_direct_model": stronger_model_results,
        "soft_target_direct_model": soft_target_model_results,
        "hybrid_direct_model": hybrid_model_results,
        "experimental_embedding_model": {
            "train": train_metrics,
            "valid": valid_metrics,
            "reformulated": reform_metrics,
            "loss_curve": losses,
        },
    }


def run_experiment(config: ExperimentConfig) -> tuple[dict, Path]:
    bundle = build_corpus(anchor_count=config.anchors, seed=config.seed)
    results = _evaluate_bundle(bundle, config)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = RUNS_DIR / f"run_{stamp}_seed{config.seed}_anchors{config.anchors}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results, path


def _summarize_with_regions(
    examples,
    logits: np.ndarray,
    labels: np.ndarray,
    label_index: dict[str, int],
    anchors_by_refmark: dict[str, object],
) -> dict[str, float]:
    summary = summarize_logits(logits, labels)
    region = _region_metrics(examples, logits, label_index, anchors_by_refmark)
    if region:
        summary.update(region)
    return summary


def _region_metrics(
    examples,
    logits: np.ndarray,
    label_index: dict[str, int],
    anchors_by_refmark: dict[str, object],
) -> dict[str, float]:
    inverse_labels = {idx: refmark for refmark, idx in label_index.items()}
    predicted = logits.argmax(axis=1)
    gold_regions_present = all(
        example.answer_start is not None and example.answer_end is not None and example.title
        for example in examples
    )
    anchor_regions_present = all(
        anchor.title and anchor.region_start is not None and anchor.region_end is not None
        for anchor in anchors_by_refmark.values()
    )
    if not gold_regions_present or not anchor_regions_present:
        return {}

    answer_overlap_hits = 0
    answer_full_cover_hits = 0
    answer_precision_sum = 0.0
    answer_recall_sum = 0.0
    answer_f1_sum = 0.0
    answer_breadth_ratio_sum = 0.0
    answer_excess_chars_sum = 0.0
    answer_overbroad_2x_hits = 0
    answer_undercite_hits = 0
    anchor_region_overlap_hits = 0
    anchor_region_cover_hits = 0
    anchor_region_precision_sum = 0.0
    anchor_region_recall_sum = 0.0
    anchor_region_f1_sum = 0.0
    anchor_region_iou_sum = 0.0
    anchor_region_breadth_ratio_sum = 0.0
    anchor_region_excess_chars_sum = 0.0
    anchor_region_overbroad_2x_hits = 0
    anchor_region_undercite_hits = 0

    for example, pred_idx in zip(examples, predicted, strict=True):
        gold_anchor = anchors_by_refmark.get(example.refmark)
        pred_refmark = inverse_labels.get(int(pred_idx))
        pred_anchor = anchors_by_refmark.get(pred_refmark)
        if gold_anchor is None or pred_anchor is None:
            continue

        answer_overlap = 0
        if pred_anchor.title == example.title:
            answer_overlap = _span_overlap(
                int(pred_anchor.region_start),
                int(pred_anchor.region_end),
                int(example.answer_start),
                int(example.answer_end),
            )
        answer_len = max(int(example.answer_end) - int(example.answer_start), 1)
        pred_len = max(int(pred_anchor.region_end) - int(pred_anchor.region_start), 1)
        answer_precision = answer_overlap / pred_len
        answer_recall = answer_overlap / answer_len
        answer_breadth_ratio = pred_len / answer_len
        if answer_overlap > 0:
            answer_overlap_hits += 1
        if answer_recall >= 1.0:
            answer_full_cover_hits += 1
        if 0 < answer_recall < 1.0:
            answer_undercite_hits += 1
        if answer_breadth_ratio >= 2.0:
            answer_overbroad_2x_hits += 1
        if answer_precision + answer_recall > 0.0:
            answer_f1_sum += 2.0 * answer_precision * answer_recall / (answer_precision + answer_recall)
        answer_precision_sum += answer_precision
        answer_recall_sum += answer_recall
        answer_breadth_ratio_sum += answer_breadth_ratio
        answer_excess_chars_sum += max(pred_len - answer_len, 0)

        if pred_anchor.title == gold_anchor.title:
            anchor_overlap = _span_overlap(
                int(pred_anchor.region_start),
                int(pred_anchor.region_end),
                int(gold_anchor.region_start),
                int(gold_anchor.region_end),
            )
            gold_anchor_len = max(int(gold_anchor.region_end) - int(gold_anchor.region_start), 1)
            union = max(int(pred_anchor.region_end), int(gold_anchor.region_end)) - min(
                int(pred_anchor.region_start),
                int(gold_anchor.region_start),
            )
            anchor_precision = anchor_overlap / pred_len
            anchor_recall = anchor_overlap / gold_anchor_len
            anchor_breadth_ratio = pred_len / gold_anchor_len
            if anchor_overlap > 0:
                anchor_region_overlap_hits += 1
            if anchor_recall >= 1.0:
                anchor_region_cover_hits += 1
            if 0 < anchor_recall < 1.0:
                anchor_region_undercite_hits += 1
            if anchor_breadth_ratio >= 2.0:
                anchor_region_overbroad_2x_hits += 1
            if anchor_precision + anchor_recall > 0.0:
                anchor_region_f1_sum += 2.0 * anchor_precision * anchor_recall / (anchor_precision + anchor_recall)
            anchor_region_precision_sum += anchor_precision
            anchor_region_recall_sum += anchor_recall
            anchor_region_breadth_ratio_sum += anchor_breadth_ratio
            anchor_region_excess_chars_sum += max(pred_len - gold_anchor_len, 0)
            if union > 0:
                anchor_region_iou_sum += anchor_overlap / union

    count = max(len(examples), 1)
    return {
        "answer_overlap_rate": answer_overlap_hits / count,
        "answer_cover_rate": answer_full_cover_hits / count,
        "answer_precision": answer_precision_sum / count,
        "answer_recall": answer_recall_sum / count,
        "answer_f1": answer_f1_sum / count,
        "breadth_ratio": answer_breadth_ratio_sum / count,
        "avg_excess_chars": answer_excess_chars_sum / count,
        "overbroad_2x_rate": answer_overbroad_2x_hits / count,
        "undercite_rate": answer_undercite_hits / count,
        "anchor_region_overlap_rate": anchor_region_overlap_hits / count,
        "anchor_region_cover_rate": anchor_region_cover_hits / count,
        "anchor_region_precision": anchor_region_precision_sum / count,
        "anchor_region_recall": anchor_region_recall_sum / count,
        "anchor_region_f1": anchor_region_f1_sum / count,
        "anchor_region_iou": anchor_region_iou_sum / count,
        "anchor_region_breadth_ratio": anchor_region_breadth_ratio_sum / count,
        "anchor_region_avg_excess_chars": anchor_region_excess_chars_sum / count,
        "anchor_region_overbroad_2x_rate": anchor_region_overbroad_2x_hits / count,
        "anchor_region_undercite_rate": anchor_region_undercite_hits / count,
        "predicted_anchor_count": 1.0,
    }


def _span_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    return max(0, min(end_a, end_b) - max(start_a, start_b))


def _build_soft_targets(
    examples,
    label_index: dict[str, int],
    anchors_by_refmark: dict[str, object],
    config: ExperimentConfig,
) -> np.ndarray:
    num_labels = len(label_index)
    targets = np.zeros((len(examples), num_labels), dtype=np.float32)
    can_grade = all(
        example.answer_start is not None and example.answer_end is not None and example.title
        for example in examples
    ) and all(
        anchor.title and anchor.region_start is not None and anchor.region_end is not None
        for anchor in anchors_by_refmark.values()
    )
    inverse = {idx: refmark for refmark, idx in label_index.items()}

    if not can_grade:
        for row, example in enumerate(examples):
            targets[row, label_index[example.refmark]] = 1.0
        return targets

    candidate_refs = [inverse[idx] for idx in range(num_labels)]
    for row, example in enumerate(examples):
        gold_idx = label_index[example.refmark]
        rewards = np.zeros(num_labels, dtype=np.float32)
        rewards[gold_idx] = 1.0
        gold_start = int(example.answer_start)
        gold_end = int(example.answer_end)
        gold_len = max(gold_end - gold_start, 1)
        for idx, refmark in enumerate(candidate_refs):
            if idx == gold_idx:
                continue
            anchor = anchors_by_refmark.get(refmark)
            if anchor is None:
                continue
            reward = 0.0
            if anchor.title == example.title:
                pred_start = int(anchor.region_start)
                pred_end = int(anchor.region_end)
                overlap = _span_overlap(pred_start, pred_end, gold_start, gold_end)
                pred_len = max(pred_end - pred_start, 1)
                precision = overlap / pred_len
                recall = overlap / gold_len
                breadth_ratio = pred_len / gold_len
                if recall >= 1.0 and overlap > 0:
                    breadth_penalty = min(
                        max(breadth_ratio - 1.0, 0.0) / max(config.soft_cover_breadth_penalty, 1e-6),
                        config.soft_cover_penalty_cap,
                    )
                    reward = max(reward, config.soft_cover_reward - breadth_penalty)
                if overlap > 0:
                    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
                    breadth_penalty = min(
                        max(breadth_ratio - 1.0, 0.0) / max(config.soft_overlap_breadth_penalty, 1e-6),
                        config.soft_overlap_penalty_cap,
                    )
                    reward = max(
                        reward,
                        config.soft_overlap_base
                        + config.soft_overlap_f1_weight * f1
                        + config.soft_overlap_precision_weight * precision
                        - breadth_penalty,
                    )
                distance = min(abs(pred_start - gold_start), abs(pred_end - gold_end))
                if distance <= config.soft_local_distance:
                    reward = max(reward, config.soft_local_reward)
            rewards[idx] = reward
        total = rewards.sum()
        if total <= 0.0:
            rewards[gold_idx] = 1.0
            total = 1.0
        targets[row] = rewards / total
    return targets


def prepare_dataset(anchor_count: int, seed: int) -> tuple[dict[str, str], Path]:
    bundle = build_corpus(anchor_count=anchor_count, seed=seed)
    output_dir = DATA_DIR / f"dataset_seed{seed}_anchors{anchor_count}"
    files = save_bundle(bundle, output_dir)
    return files, output_dir


def run_saved_experiment(config: ExperimentConfig, data_dir: Path) -> tuple[dict, Path]:
    bundle = load_bundle_from_dir(data_dir)
    results = _evaluate_bundle(bundle, config)
    results["dataset"]["data_dir"] = str(data_dir)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = RUNS_DIR / f"run_{stamp}_{data_dir.name}_seed{config.seed}_{results['dataset']['backend']}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results, path


def run_ensemble_saved_experiment(
    config: ExperimentConfig,
    data_dir: Path,
    *,
    seeds: list[int],
    model_kind: str = "hybrid",
    merge_mode: str = "logit",
) -> tuple[dict, Path]:
    bundle = load_bundle_from_dir(data_dir)
    resolved_backend = detect_backend(config.backend)
    if resolved_backend not in {"torchcpu", "directml", "cuda"}:
        raise ValueError("ensemble evaluation currently requires torch-backed backend")

    vocab = Vocab.build(
        [example.question for example in bundle.train]
        + [anchor.text for anchor in bundle.anchors]
    )
    label_index = build_label_index(bundle.train)
    train_x, train_y = encode_examples(bundle.train, vocab, label_index)
    valid_x, valid_y = encode_examples(bundle.valid, vocab, label_index)
    reform_x, reform_y = encode_examples(bundle.reformulated, vocab, label_index)
    bow_train_x = TorchBoWClassifier.counts_matrix(train_x, len(vocab.id_to_token))
    bow_valid_x = TorchBoWClassifier.counts_matrix(valid_x, len(vocab.id_to_token))
    bow_reform_x = TorchBoWClassifier.counts_matrix(reform_x, len(vocab.id_to_token))
    anchors_by_refmark = {anchor.refmark: anchor for anchor in bundle.anchors}
    soft_targets = _build_soft_targets(bundle.train, label_index, anchors_by_refmark, config)
    device = "privateuseone:0" if resolved_backend == "directml" else ("cuda" if resolved_backend == "cuda" else "cpu")

    valid_logits_sum = None
    reform_logits_sum = None
    train_logits_sum = None
    for seed in seeds:
        model = TorchBoWMLPClassifier(
            vocab_size=len(vocab.id_to_token),
            num_labels=len(label_index),
            device=device,
            hidden_dim=config.mlp_hidden_dim,
            hidden2_dim=config.mlp_hidden2_dim,
            dropout=config.mlp_dropout,
        )
        if model_kind == "hybrid":
            model.train_hybrid(
                bow_train_x,
                train_y,
                soft_targets,
                exact_weight=config.hybrid_exact_weight,
                soft_weight=1.0 - config.hybrid_exact_weight,
                epochs=config.epochs,
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
                weight_decay=config.weight_decay,
                seed=seed,
            )
        elif model_kind == "stronger":
            model.train(
                bow_train_x,
                train_y,
                epochs=config.epochs,
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
                weight_decay=config.weight_decay,
                seed=seed,
            )
        else:
            raise ValueError(f"unsupported model kind: {model_kind}")
        train_logits = model.logits(bow_train_x)
        valid_logits = model.logits(bow_valid_x)
        reform_logits = model.logits(bow_reform_x)
        train_logits_sum = train_logits if train_logits_sum is None else train_logits_sum + train_logits
        valid_logits_sum = valid_logits if valid_logits_sum is None else valid_logits_sum + valid_logits
        reform_logits_sum = reform_logits if reform_logits_sum is None else reform_logits_sum + reform_logits

    train_logits_avg = train_logits_sum / len(seeds)
    valid_logits_avg = valid_logits_sum / len(seeds)
    reform_logits_avg = reform_logits_sum / len(seeds)
    if merge_mode == "overlap":
        valid_logits_avg = _apply_overlap_merge(valid_logits_avg, label_index, anchors_by_refmark)
        reform_logits_avg = _apply_overlap_merge(reform_logits_avg, label_index, anchors_by_refmark)
        train_logits_avg = _apply_overlap_merge(train_logits_avg, label_index, anchors_by_refmark)
    elif merge_mode != "logit":
        raise ValueError(f"unsupported merge mode: {merge_mode}")
    results = {
        "config": asdict(config),
        "ensemble": {
            "model_kind": model_kind,
            "seeds": seeds,
            "merge_mode": merge_mode,
        },
        "dataset": {
            "anchor_count": len(bundle.anchors),
            "train_examples": len(bundle.train),
            "valid_examples": len(bundle.valid),
            "reformulated_examples": len(bundle.reformulated),
            "vocab_size": len(vocab.id_to_token),
            "backend": resolved_backend,
            "data_dir": str(data_dir),
        },
        "ensemble_model": {
            "train": summarize_logits(train_logits_avg, train_y),
            "valid": _summarize_with_regions(
                bundle.valid,
                valid_logits_avg,
                valid_y,
                label_index,
                anchors_by_refmark,
            ),
            "reformulated": _summarize_with_regions(
                bundle.reformulated,
                reform_logits_avg,
                reform_y,
                label_index,
                anchors_by_refmark,
            ),
        },
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = RUNS_DIR / f"ensemble_{stamp}_{data_dir.name}_{model_kind}_{merge_mode}_{resolved_backend}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results, path


def run_range_saved_experiment(config: ExperimentConfig, data_dir: Path) -> tuple[dict, Path]:
    bundle = load_bundle_from_dir(data_dir)
    resolved_backend = detect_backend(config.backend)
    if resolved_backend not in {"torchcpu", "directml", "cuda"}:
        raise ValueError("range evaluation currently requires torch-backed backend")

    vocab = Vocab.build(
        [example.question for example in bundle.train]
        + [anchor.text for anchor in bundle.anchors]
    )
    anchors_by_refmark = {anchor.refmark: anchor for anchor in bundle.anchors}
    sorted_anchors = sorted(bundle.anchors, key=lambda anchor: anchor.refmark)
    anchor_to_index = {anchor.refmark: idx for idx, anchor in enumerate(sorted_anchors)}
    device = "privateuseone:0" if resolved_backend == "directml" else ("cuda" if resolved_backend == "cuda" else "cpu")
    train_x = TorchBoWClassifier.counts_matrix(
        [vocab.encode(example.question) for example in bundle.train],
        len(vocab.id_to_token),
    )
    valid_x = TorchBoWClassifier.counts_matrix(
        [vocab.encode(example.question) for example in bundle.valid],
        len(vocab.id_to_token),
    )
    reform_x = TorchBoWClassifier.counts_matrix(
        [vocab.encode(example.question) for example in bundle.reformulated],
        len(vocab.id_to_token),
    )
    range_model = TorchBoWStartEndPredictor(
        vocab_size=len(vocab.id_to_token),
        num_labels=len(sorted_anchors),
        device=device,
        hidden_dim=config.mlp_hidden_dim,
        hidden2_dim=config.mlp_hidden2_dim,
        dropout=config.mlp_dropout,
    )
    train_start, train_end = _range_starts_ends(bundle.train, anchor_to_index)
    start_targets, end_targets = _build_start_end_soft_targets(
        bundle.train,
        sorted_anchors,
        anchor_to_index,
        config,
    )
    range_model.train_hybrid_ranges(
        train_x,
        train_start,
        train_end,
        start_targets,
        end_targets,
        exact_weight=config.range_exact_weight,
        soft_weight=config.range_soft_weight,
        width_weight=config.range_width_weight,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        weight_decay=config.weight_decay,
        seed=config.seed,
    )

    valid_start_logits, valid_end_logits = range_model.logits(valid_x)
    reform_start_logits, reform_end_logits = range_model.logits(reform_x)
    valid_range = _range_metrics(
        bundle.valid,
        valid_start_logits,
        valid_end_logits,
        sorted_anchors,
        anchor_to_index,
        anchors_by_refmark,
    )
    reform_range = _range_metrics(
        bundle.reformulated,
        reform_start_logits,
        reform_end_logits,
        sorted_anchors,
        anchor_to_index,
        anchors_by_refmark,
    )
    results = {
        "config": asdict(config),
        "dataset": {
            "anchor_count": len(bundle.anchors),
            "train_examples": len(bundle.train),
            "valid_examples": len(bundle.valid),
            "reformulated_examples": len(bundle.reformulated),
            "vocab_size": len(vocab.id_to_token),
            "backend": resolved_backend,
            "data_dir": str(data_dir),
        },
        "range_model": {
            "valid": valid_range,
            "reformulated": reform_range,
        },
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = RUNS_DIR / f"range_{stamp}_{data_dir.name}_seed{config.seed}_{resolved_backend}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results, path


def run_range_ensemble_saved_experiment(
    config: ExperimentConfig,
    data_dir: Path,
    *,
    seeds: list[int],
) -> tuple[dict, Path]:
    bundle = load_bundle_from_dir(data_dir)
    resolved_backend = detect_backend(config.backend)
    if resolved_backend not in {"torchcpu", "directml", "cuda"}:
        raise ValueError("range ensemble requires torch-backed backend")

    vocab = Vocab.build(
        [example.question for example in bundle.train]
        + [anchor.text for anchor in bundle.anchors]
    )
    sorted_anchors = sorted(bundle.anchors, key=lambda anchor: anchor.refmark)
    anchor_to_index = {anchor.refmark: idx for idx, anchor in enumerate(sorted_anchors)}
    train_x = TorchBoWClassifier.counts_matrix(
        [vocab.encode(example.question) for example in bundle.train],
        len(vocab.id_to_token),
    )
    valid_x = TorchBoWClassifier.counts_matrix(
        [vocab.encode(example.question) for example in bundle.valid],
        len(vocab.id_to_token),
    )
    reform_x = TorchBoWClassifier.counts_matrix(
        [vocab.encode(example.question) for example in bundle.reformulated],
        len(vocab.id_to_token),
    )
    train_start, train_end = _range_starts_ends(bundle.train, anchor_to_index)
    start_targets, end_targets = _build_start_end_soft_targets(
        bundle.train,
        sorted_anchors,
        anchor_to_index,
        config,
    )
    anchors_by_refmark = {anchor.refmark: anchor for anchor in bundle.anchors}
    device = "privateuseone:0" if resolved_backend == "directml" else ("cuda" if resolved_backend == "cuda" else "cpu")

    valid_start_sum = None
    valid_end_sum = None
    reform_start_sum = None
    reform_end_sum = None
    for seed in seeds:
        model = TorchBoWStartEndPredictor(
            vocab_size=len(vocab.id_to_token),
            num_labels=len(sorted_anchors),
            device=device,
            hidden_dim=config.mlp_hidden_dim,
            hidden2_dim=config.mlp_hidden2_dim,
            dropout=config.mlp_dropout,
        )
        model.train_hybrid_ranges(
            train_x,
            train_start,
            train_end,
            start_targets,
            end_targets,
            exact_weight=config.range_exact_weight,
            soft_weight=config.range_soft_weight,
            width_weight=config.range_width_weight,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            weight_decay=config.weight_decay,
            seed=seed,
        )
        valid_start, valid_end = model.logits(valid_x)
        reform_start, reform_end = model.logits(reform_x)
        valid_start_sum = valid_start if valid_start_sum is None else valid_start_sum + valid_start
        valid_end_sum = valid_end if valid_end_sum is None else valid_end_sum + valid_end
        reform_start_sum = reform_start if reform_start_sum is None else reform_start_sum + reform_start
        reform_end_sum = reform_end if reform_end_sum is None else reform_end_sum + reform_end

    valid_start_avg = valid_start_sum / len(seeds)
    valid_end_avg = valid_end_sum / len(seeds)
    reform_start_avg = reform_start_sum / len(seeds)
    reform_end_avg = reform_end_sum / len(seeds)
    results = {
        "config": asdict(config),
        "ensemble": {
            "seeds": seeds,
            "model_kind": "range",
        },
        "dataset": {
            "anchor_count": len(bundle.anchors),
            "train_examples": len(bundle.train),
            "valid_examples": len(bundle.valid),
            "reformulated_examples": len(bundle.reformulated),
            "vocab_size": len(vocab.id_to_token),
            "backend": resolved_backend,
            "data_dir": str(data_dir),
        },
        "range_model": {
            "valid": _range_metrics(
                bundle.valid,
                valid_start_avg,
                valid_end_avg,
                sorted_anchors,
                anchor_to_index,
                anchors_by_refmark,
            ),
            "reformulated": _range_metrics(
                bundle.reformulated,
                reform_start_avg,
                reform_end_avg,
                sorted_anchors,
                anchor_to_index,
                anchors_by_refmark,
            ),
        },
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = RUNS_DIR / f"range_ensemble_{stamp}_{data_dir.name}_{resolved_backend}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results, path


def run_center_width_saved_experiment(config: ExperimentConfig, data_dir: Path) -> tuple[dict, Path]:
    bundle = load_bundle_from_dir(data_dir)
    resolved_backend = detect_backend(config.backend)
    if resolved_backend not in {"torchcpu", "directml", "cuda"}:
        raise ValueError("center/width evaluation currently requires torch-backed backend")

    vocab = Vocab.build(
        [example.question for example in bundle.train]
        + [anchor.text for anchor in bundle.anchors]
    )
    sorted_anchors = sorted(bundle.anchors, key=lambda anchor: anchor.refmark)
    anchor_to_index = {anchor.refmark: idx for idx, anchor in enumerate(sorted_anchors)}
    device = "privateuseone:0" if resolved_backend == "directml" else ("cuda" if resolved_backend == "cuda" else "cpu")
    width_bins = _default_width_bins(len(sorted_anchors))
    train_x = TorchBoWClassifier.counts_matrix(
        [vocab.encode(example.question) for example in bundle.train],
        len(vocab.id_to_token),
    )
    valid_x = TorchBoWClassifier.counts_matrix(
        [vocab.encode(example.question) for example in bundle.valid],
        len(vocab.id_to_token),
    )
    reform_x = TorchBoWClassifier.counts_matrix(
        [vocab.encode(example.question) for example in bundle.reformulated],
        len(vocab.id_to_token),
    )
    train_center = _range_centers(bundle.train, anchor_to_index)
    valid_center = _range_centers(bundle.valid, anchor_to_index)
    reform_center = _range_centers(bundle.reformulated, anchor_to_index)
    width_labels, width_targets = _build_width_targets(bundle.train, anchor_to_index, width_bins)
    center_targets, _ = _build_range_soft_targets(
        bundle.train,
        sorted_anchors,
        anchor_to_index,
        config,
    )
    bm25_logits_valid, bm25_logits_reform = _bm25_logits_for_splits(
        bundle,
        vocab,
        anchor_to_index,
        enrich_train=config.bm25_enrich_train,
    )

    model = TorchBoWCenterWidthPredictor(
        vocab_size=len(vocab.id_to_token),
        num_labels=len(sorted_anchors),
        width_bins=width_bins,
        device=device,
        hidden_dim=config.mlp_hidden_dim,
        hidden2_dim=config.mlp_hidden2_dim,
        dropout=config.mlp_dropout,
    )
    losses = model.train_hybrid(
        train_x,
        train_center,
        width_labels,
        center_targets,
        width_targets,
        exact_weight=config.center_exact_weight,
        soft_weight=config.center_soft_weight,
        width_weight=config.center_width_weight,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        weight_decay=config.weight_decay,
        seed=config.seed,
    )
    valid_center_logits, valid_width_logits = model.logits(valid_x)
    reform_center_logits, reform_width_logits = model.logits(reform_x)
    valid_center_logits = _apply_bm25_neighborhood_prior(
        valid_center_logits,
        bm25_logits_valid,
        config,
    )
    reform_center_logits = _apply_bm25_neighborhood_prior(
        reform_center_logits,
        bm25_logits_reform,
        config,
    )
    results = {
        "config": asdict(config),
        "dataset": {
            "anchor_count": len(bundle.anchors),
            "train_examples": len(bundle.train),
            "valid_examples": len(bundle.valid),
            "reformulated_examples": len(bundle.reformulated),
            "vocab_size": len(vocab.id_to_token),
            "backend": resolved_backend,
            "data_dir": str(data_dir),
        },
        "center_width_model": {
            "width_bins": width_bins,
            "loss_curve": losses,
            "valid": _center_width_metrics(
                bundle.valid,
                valid_center_logits,
                valid_width_logits,
                sorted_anchors,
                valid_center,
                width_bins,
            ),
            "valid_by_style": _center_width_metrics_by_style(
                bundle.valid,
                valid_center_logits,
                valid_width_logits,
                sorted_anchors,
                valid_center,
                width_bins,
            ),
            "reformulated": _center_width_metrics(
                bundle.reformulated,
                reform_center_logits,
                reform_width_logits,
                sorted_anchors,
                reform_center,
                width_bins,
            ),
            "reformulated_by_style": _center_width_metrics_by_style(
                bundle.reformulated,
                reform_center_logits,
                reform_width_logits,
                sorted_anchors,
                reform_center,
                width_bins,
            ),
        },
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = RUNS_DIR / f"center_width_{stamp}_{data_dir.name}_seed{config.seed}_{resolved_backend}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results, path


def run_center_width_ensemble_saved_experiment(
    config: ExperimentConfig,
    data_dir: Path,
    *,
    seeds: list[int],
) -> tuple[dict, Path]:
    bundle = load_bundle_from_dir(data_dir)
    resolved_backend = detect_backend(config.backend)
    if resolved_backend not in {"torchcpu", "directml", "cuda"}:
        raise ValueError("center/width ensemble requires torch-backed backend")

    vocab = Vocab.build(
        [example.question for example in bundle.train]
        + [anchor.text for anchor in bundle.anchors]
    )
    sorted_anchors = sorted(bundle.anchors, key=lambda anchor: anchor.refmark)
    anchor_to_index = {anchor.refmark: idx for idx, anchor in enumerate(sorted_anchors)}
    width_bins = _default_width_bins(len(sorted_anchors))
    device = "privateuseone:0" if resolved_backend == "directml" else ("cuda" if resolved_backend == "cuda" else "cpu")
    train_x = TorchBoWClassifier.counts_matrix(
        [vocab.encode(example.question) for example in bundle.train],
        len(vocab.id_to_token),
    )
    valid_x = TorchBoWClassifier.counts_matrix(
        [vocab.encode(example.question) for example in bundle.valid],
        len(vocab.id_to_token),
    )
    reform_x = TorchBoWClassifier.counts_matrix(
        [vocab.encode(example.question) for example in bundle.reformulated],
        len(vocab.id_to_token),
    )
    train_center = _range_centers(bundle.train, anchor_to_index)
    valid_center = _range_centers(bundle.valid, anchor_to_index)
    reform_center = _range_centers(bundle.reformulated, anchor_to_index)
    width_labels, width_targets = _build_width_targets(bundle.train, anchor_to_index, width_bins)
    center_targets, _ = _build_range_soft_targets(
        bundle.train,
        sorted_anchors,
        anchor_to_index,
        config,
    )
    bm25_logits_valid, bm25_logits_reform = _bm25_logits_for_splits(
        bundle,
        vocab,
        anchor_to_index,
        enrich_train=config.bm25_enrich_train,
    )

    valid_center_sum = None
    valid_width_sum = None
    reform_center_sum = None
    reform_width_sum = None
    train_center_sum = None
    train_width_sum = None
    route_train = None
    route_valid = None
    route_reform = None
    route_valid_sum = None
    route_reform_sum = None
    route_bucket_size = max(0, config.route_bucket_size)
    if route_bucket_size > 0:
        route_train = _route_labels(bundle.train, anchor_to_index, route_bucket_size)
        route_valid = _route_labels(bundle.valid, anchor_to_index, route_bucket_size)
        route_reform = _route_labels(bundle.reformulated, anchor_to_index, route_bucket_size)
        num_route_buckets = int(np.ceil(len(sorted_anchors) / route_bucket_size))
    else:
        num_route_buckets = 0
    for seed in seeds:
        model = TorchBoWCenterWidthPredictor(
            vocab_size=len(vocab.id_to_token),
            num_labels=len(sorted_anchors),
            width_bins=width_bins,
            device=device,
            hidden_dim=config.mlp_hidden_dim,
            hidden2_dim=config.mlp_hidden2_dim,
            dropout=config.mlp_dropout,
        )
        model.train_hybrid(
            train_x,
            train_center,
            width_labels,
            center_targets,
            width_targets,
            exact_weight=config.center_exact_weight,
            soft_weight=config.center_soft_weight,
            width_weight=config.center_width_weight,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            weight_decay=config.weight_decay,
            seed=seed,
        )
        train_center_logits, train_width_logits = model.logits(train_x)
        valid_center_logits, valid_width_logits = model.logits(valid_x)
        reform_center_logits, reform_width_logits = model.logits(reform_x)
        train_center_sum = train_center_logits if train_center_sum is None else train_center_sum + train_center_logits
        train_width_sum = train_width_logits if train_width_sum is None else train_width_sum + train_width_logits
        valid_center_sum = valid_center_logits if valid_center_sum is None else valid_center_sum + valid_center_logits
        valid_width_sum = valid_width_logits if valid_width_sum is None else valid_width_sum + valid_width_logits
        reform_center_sum = reform_center_logits if reform_center_sum is None else reform_center_sum + reform_center_logits
        reform_width_sum = reform_width_logits if reform_width_sum is None else reform_width_sum + reform_width_logits
        if route_bucket_size > 0:
            route_model = TorchBoWMLPClassifier(
                vocab_size=len(vocab.id_to_token),
                num_labels=num_route_buckets,
                device=device,
                hidden_dim=max(64, config.mlp_hidden_dim // 2),
                hidden2_dim=0,
                dropout=config.mlp_dropout,
            )
            route_model.train(
                train_x,
                route_train,
                epochs=config.epochs,
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
                weight_decay=config.weight_decay,
                seed=seed,
            )
            valid_route_logits = route_model.logits(valid_x)
            reform_route_logits = route_model.logits(reform_x)
            route_valid_sum = valid_route_logits if route_valid_sum is None else route_valid_sum + valid_route_logits
            route_reform_sum = reform_route_logits if route_reform_sum is None else route_reform_sum + reform_route_logits

    valid_center_avg = _apply_bm25_neighborhood_prior(valid_center_sum / len(seeds), bm25_logits_valid, config)
    reform_center_avg = _apply_bm25_neighborhood_prior(reform_center_sum / len(seeds), bm25_logits_reform, config)
    route_info = None
    if route_bucket_size > 0 and route_valid_sum is not None and route_reform_sum is not None:
        valid_route_avg = route_valid_sum / len(seeds)
        reform_route_avg = route_reform_sum / len(seeds)
        route_info = {
            "bucket_size": route_bucket_size,
            "top_k": config.route_top_k,
            "margin": config.route_margin,
            "bucket_count": num_route_buckets,
            "outside_penalty": config.route_outside_penalty,
            "valid_accuracy": float(np.mean(np.argmax(valid_route_avg, axis=1) == route_valid)),
            "reformulated_accuracy": float(np.mean(np.argmax(reform_route_avg, axis=1) == route_reform)),
        }
        valid_center_avg = _apply_route_prior(
            valid_center_avg,
            valid_route_avg,
            bucket_size=route_bucket_size,
            top_k=config.route_top_k,
            margin=config.route_margin,
            outside_penalty=config.route_outside_penalty,
        )
        reform_center_avg = _apply_route_prior(
            reform_center_avg,
            reform_route_avg,
            bucket_size=route_bucket_size,
            top_k=config.route_top_k,
            margin=config.route_margin,
            outside_penalty=config.route_outside_penalty,
        )
    results = {
        "config": asdict(config),
        "ensemble": {
            "seeds": seeds,
            "model_kind": "center_width",
        },
        "dataset": {
            "anchor_count": len(bundle.anchors),
            "train_examples": len(bundle.train),
            "valid_examples": len(bundle.valid),
            "reformulated_examples": len(bundle.reformulated),
            "vocab_size": len(vocab.id_to_token),
            "backend": resolved_backend,
            "data_dir": str(data_dir),
        },
        "center_width_model": {
            "width_bins": width_bins,
            "route": route_info,
            "train": _center_width_metrics(
                bundle.train,
                train_center_sum / len(seeds),
                train_width_sum / len(seeds),
                sorted_anchors,
                train_center,
                width_bins,
            ),
            "valid": _center_width_metrics(
                bundle.valid,
                valid_center_avg,
                valid_width_sum / len(seeds),
                sorted_anchors,
                valid_center,
                width_bins,
            ),
            "valid_by_style": _center_width_metrics_by_style(
                bundle.valid,
                valid_center_avg,
                valid_width_sum / len(seeds),
                sorted_anchors,
                valid_center,
                width_bins,
            ),
            "reformulated": _center_width_metrics(
                bundle.reformulated,
                reform_center_avg,
                reform_width_sum / len(seeds),
                sorted_anchors,
                reform_center,
                width_bins,
            ),
            "reformulated_by_style": _center_width_metrics_by_style(
                bundle.reformulated,
                reform_center_avg,
                reform_width_sum / len(seeds),
                sorted_anchors,
                reform_center,
                width_bins,
            ),
        },
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = RUNS_DIR / f"center_width_ensemble_{stamp}_{data_dir.name}_{resolved_backend}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results, path


def run_refinement_loop_saved_experiment(
    config: ExperimentConfig,
    data_dir: Path,
    *,
    seeds: list[int],
) -> tuple[dict, Path]:
    bundle = load_bundle_from_dir(data_dir)
    resolved_backend = detect_backend(config.backend)
    if resolved_backend not in {"torchcpu", "directml", "cuda"}:
        raise ValueError("refinement loop requires torch-backed backend")

    vocab = Vocab.build(
        [example.question for example in bundle.train]
        + [anchor.text for anchor in bundle.anchors]
    )
    sorted_anchors = sorted(bundle.anchors, key=lambda anchor: anchor.refmark)
    anchor_to_index = {anchor.refmark: idx for idx, anchor in enumerate(sorted_anchors)}
    width_bins = _default_width_bins(len(sorted_anchors))
    device = "privateuseone:0" if resolved_backend == "directml" else ("cuda" if resolved_backend == "cuda" else "cpu")
    train_x = TorchBoWClassifier.counts_matrix(
        [vocab.encode(example.question) for example in bundle.train],
        len(vocab.id_to_token),
    )
    train_center = _range_centers(bundle.train, anchor_to_index)
    width_labels, width_targets = _build_width_targets(bundle.train, anchor_to_index, width_bins)
    center_targets, _ = _build_range_soft_targets(bundle.train, sorted_anchors, anchor_to_index, config)

    trained_models: list[TorchBoWCenterWidthPredictor] = []
    for seed in seeds:
        model = TorchBoWCenterWidthPredictor(
            vocab_size=len(vocab.id_to_token),
            num_labels=len(sorted_anchors),
            width_bins=width_bins,
            device=device,
            hidden_dim=config.mlp_hidden_dim,
            hidden2_dim=config.mlp_hidden2_dim,
            dropout=config.mlp_dropout,
        )
        model.train_hybrid(
            train_x,
            train_center,
            width_labels,
            center_targets,
            width_targets,
            exact_weight=config.center_exact_weight,
            soft_weight=config.center_soft_weight,
            width_weight=config.center_width_weight,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            weight_decay=config.weight_decay,
            seed=seed,
        )
        trained_models.append(model)

    valid_results = _refinement_loop_split_metrics(bundle.valid, trained_models, vocab, sorted_anchors, width_bins, anchor_to_index)
    reform_results = _refinement_loop_split_metrics(
        bundle.reformulated,
        trained_models,
        vocab,
        sorted_anchors,
        width_bins,
        anchor_to_index,
    )
    results = {
        "config": asdict(config),
        "ensemble": {
            "seeds": seeds,
            "model_kind": "refinement_loop_center_width",
        },
        "dataset": {
            "anchor_count": len(bundle.anchors),
            "train_examples": len(bundle.train),
            "valid_examples": len(bundle.valid),
            "reformulated_examples": len(bundle.reformulated),
            "vocab_size": len(vocab.id_to_token),
            "backend": resolved_backend,
            "data_dir": str(data_dir),
        },
        "refinement_loop": {
            "width_bins": width_bins,
            "valid": valid_results,
            "reformulated": reform_results,
        },
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = RUNS_DIR / f"refinement_loop_{stamp}_{data_dir.name}_{resolved_backend}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results, path


def run_refinement_two_model_saved_experiment(
    config: ExperimentConfig,
    data_dir: Path,
    *,
    seeds: list[int],
) -> tuple[dict, Path]:
    bundle = load_bundle_from_dir(data_dir)
    resolved_backend = detect_backend(config.backend)
    if resolved_backend not in {"torchcpu", "directml", "cuda"}:
        raise ValueError("two-model refinement loop requires torch-backed backend")

    vocab = Vocab.build(
        [example.question for example in bundle.train]
        + [anchor.text for anchor in bundle.anchors]
    )
    sorted_anchors = sorted(bundle.anchors, key=lambda anchor: anchor.refmark)
    anchor_to_index = {anchor.refmark: idx for idx, anchor in enumerate(sorted_anchors)}
    width_bins = _default_width_bins(len(sorted_anchors))
    device = "privateuseone:0" if resolved_backend == "directml" else ("cuda" if resolved_backend == "cuda" else "cpu")

    broad_train = [example for example in bundle.train if "broad" in example.prompt_style]
    precise_train = [example for example in bundle.train if "precise" in example.prompt_style]
    broad_models = _train_center_width_ensemble(
        broad_train,
        vocab,
        sorted_anchors,
        anchor_to_index,
        width_bins,
        device,
        config,
        seeds,
    )
    refine_models = _train_center_width_ensemble(
        precise_train,
        vocab,
        sorted_anchors,
        anchor_to_index,
        width_bins,
        device,
        config,
        seeds,
    )
    bm25_retriever = _fit_bm25_retriever(bundle, vocab, enrich_train=True)
    broad_train_center, broad_train_width = _average_center_width_logits(
        broad_models,
        [example.question for example in broad_train],
        vocab,
    )
    bm25_train_broad = bm25_retriever.logits([example.question for example in broad_train], anchor_to_index)
    broad_candidate_reranker = _fit_mlp_candidate_reranker(
        broad_train,
        broad_train_center,
        broad_train_width,
        bm25_train_broad,
        width_bins,
        sorted_anchors,
        anchor_to_index,
        config,
    )
    local_cross_reranker = _train_local_cross_reranker(
        broad_train,
        broad_train_center,
        broad_train_width,
        bm25_train_broad,
        width_bins,
        sorted_anchors,
        anchor_to_index,
        vocab,
        config,
        device,
    )
    broad_train_candidate_rows = _local_cross_candidate_scores(
        broad_train,
        broad_train_center,
        broad_train_width,
        bm25_train_broad,
        width_bins,
        sorted_anchors,
        anchor_to_index,
        vocab,
        config,
        local_cross_reranker,
    )
    local_cross_policy = _fit_local_cross_blend_policy(
        broad_train,
        broad_train_candidate_rows,
        sorted_anchors,
        anchor_to_index,
        config,
    )
    broad_valid_examples = [example for example in bundle.valid if "broad" in example.prompt_style]
    broad_reform_examples = [example for example in bundle.reformulated if "broad" in example.prompt_style]
    bm25_valid = bm25_retriever.logits([example.question for example in broad_valid_examples], anchor_to_index)
    bm25_reform = bm25_retriever.logits([example.question for example in broad_reform_examples], anchor_to_index)

    valid_results = _refinement_loop_split_metrics_two_model(
        bundle.valid,
        broad_models,
        refine_models,
        vocab,
        sorted_anchors,
        width_bins,
        anchor_to_index,
        broad_bm25_logits=bm25_valid,
        config=config,
        broad_candidate_reranker=broad_candidate_reranker,
        local_cross_reranker=local_cross_reranker,
        local_cross_policy=local_cross_policy,
    )
    reform_results = _refinement_loop_split_metrics_two_model(
        bundle.reformulated,
        broad_models,
        refine_models,
        vocab,
        sorted_anchors,
        width_bins,
        anchor_to_index,
        broad_bm25_logits=bm25_reform,
        config=config,
        broad_candidate_reranker=broad_candidate_reranker,
        local_cross_reranker=local_cross_reranker,
        local_cross_policy=local_cross_policy,
    )
    results = {
        "config": asdict(config),
        "ensemble": {
            "seeds": seeds,
            "model_kind": "refinement_loop_two_model",
            "local_cross_policy": local_cross_policy,
        },
        "dataset": {
            "anchor_count": len(bundle.anchors),
            "train_examples": len(bundle.train),
            "valid_examples": len(bundle.valid),
            "reformulated_examples": len(bundle.reformulated),
            "vocab_size": len(vocab.id_to_token),
            "backend": resolved_backend,
            "data_dir": str(data_dir),
            "broad_train_examples": len(broad_train),
            "precise_train_examples": len(precise_train),
        },
        "refinement_loop": {
            "width_bins": width_bins,
            "mode": "two_model",
            "valid": valid_results,
            "reformulated": reform_results,
        },
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = RUNS_DIR / f"refinement_two_model_{stamp}_{data_dir.name}_{resolved_backend}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results, path


def run_hybrid_fusion_saved_experiment(
    config: ExperimentConfig,
    data_dir: Path,
    *,
    seeds: list[int],
) -> tuple[dict, Path]:
    bundle = load_bundle_from_dir(data_dir)
    resolved_backend = detect_backend(config.backend)
    if resolved_backend not in {"torchcpu", "directml", "cuda"}:
        raise ValueError("hybrid fusion requires torch-backed backend")

    vocab = Vocab.build(
        [example.question for example in bundle.train]
        + [anchor.text for anchor in bundle.anchors]
    )
    sorted_anchors = sorted(bundle.anchors, key=lambda anchor: anchor.refmark)
    anchor_to_index = {anchor.refmark: idx for idx, anchor in enumerate(sorted_anchors)}
    width_bins = _default_width_bins(len(sorted_anchors))
    device = "privateuseone:0" if resolved_backend == "directml" else ("cuda" if resolved_backend == "cuda" else "cpu")
    train_x = TorchBoWClassifier.counts_matrix(
        [vocab.encode(example.question) for example in bundle.train],
        len(vocab.id_to_token),
    )
    valid_x = TorchBoWClassifier.counts_matrix(
        [vocab.encode(example.question) for example in bundle.valid],
        len(vocab.id_to_token),
    )
    reform_x = TorchBoWClassifier.counts_matrix(
        [vocab.encode(example.question) for example in bundle.reformulated],
        len(vocab.id_to_token),
    )
    train_y = np.array([anchor_to_index[example.refmark] for example in bundle.train], dtype=np.int64)
    valid_y = np.array([anchor_to_index[example.refmark] for example in bundle.valid], dtype=np.int64)
    reform_y = np.array([anchor_to_index[example.refmark] for example in bundle.reformulated], dtype=np.int64)
    center_targets, _ = _build_range_soft_targets(bundle.train, sorted_anchors, anchor_to_index, config)
    width_labels, width_targets = _build_width_targets(bundle.train, anchor_to_index, width_bins)
    soft_targets = _build_soft_targets(bundle.train, anchor_to_index, {anchor.refmark: anchor for anchor in bundle.anchors}, config)
    bm25_logits_valid, bm25_logits_reform = _bm25_logits_for_splits(
        bundle,
        vocab,
        anchor_to_index,
        enrich_train=config.bm25_enrich_train,
    )

    valid_anchor_sum = None
    reform_anchor_sum = None
    valid_center_sum = None
    reform_center_sum = None
    valid_width_sum = None
    reform_width_sum = None
    for seed in seeds:
        anchor_model = TorchBoWMLPClassifier(
            vocab_size=len(vocab.id_to_token),
            num_labels=len(sorted_anchors),
            device=device,
            hidden_dim=config.mlp_hidden_dim,
            hidden2_dim=config.mlp_hidden2_dim,
            dropout=config.mlp_dropout,
        )
        anchor_model.train_hybrid(
            train_x,
            train_y,
            soft_targets,
            exact_weight=config.hybrid_exact_weight,
            soft_weight=1.0 - config.hybrid_exact_weight,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            weight_decay=config.weight_decay,
            seed=seed,
        )
        center_model = TorchBoWCenterWidthPredictor(
            vocab_size=len(vocab.id_to_token),
            num_labels=len(sorted_anchors),
            width_bins=width_bins,
            device=device,
            hidden_dim=config.mlp_hidden_dim,
            hidden2_dim=config.mlp_hidden2_dim,
            dropout=config.mlp_dropout,
        )
        center_model.train_hybrid(
            train_x,
            train_y,
            width_labels,
            center_targets,
            width_targets,
            exact_weight=config.center_exact_weight,
            soft_weight=config.center_soft_weight,
            width_weight=config.center_width_weight,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            weight_decay=config.weight_decay,
            seed=seed,
        )
        valid_anchor_logits = anchor_model.logits(valid_x)
        reform_anchor_logits = anchor_model.logits(reform_x)
        valid_center_logits, valid_width_logits = center_model.logits(valid_x)
        reform_center_logits, reform_width_logits = center_model.logits(reform_x)
        valid_anchor_sum = valid_anchor_logits if valid_anchor_sum is None else valid_anchor_sum + valid_anchor_logits
        reform_anchor_sum = reform_anchor_logits if reform_anchor_sum is None else reform_anchor_sum + reform_anchor_logits
        valid_center_sum = valid_center_logits if valid_center_sum is None else valid_center_sum + valid_center_logits
        reform_center_sum = reform_center_logits if reform_center_sum is None else reform_center_sum + reform_center_logits
        valid_width_sum = valid_width_logits if valid_width_sum is None else valid_width_sum + valid_width_logits
        reform_width_sum = reform_width_logits if reform_width_sum is None else reform_width_sum + reform_width_logits

    valid_anchor_avg = _apply_bm25_neighborhood_prior(valid_anchor_sum / len(seeds), bm25_logits_valid, config)
    reform_anchor_avg = _apply_bm25_neighborhood_prior(reform_anchor_sum / len(seeds), bm25_logits_reform, config)
    valid_center_avg = _apply_bm25_neighborhood_prior(valid_center_sum / len(seeds), bm25_logits_valid, config)
    reform_center_avg = _apply_bm25_neighborhood_prior(reform_center_sum / len(seeds), bm25_logits_reform, config)
    results = {
        "config": asdict(config),
        "ensemble": {
            "seeds": seeds,
            "model_kind": "hybrid_fusion",
        },
        "dataset": {
            "anchor_count": len(bundle.anchors),
            "train_examples": len(bundle.train),
            "valid_examples": len(bundle.valid),
            "reformulated_examples": len(bundle.reformulated),
            "vocab_size": len(vocab.id_to_token),
            "backend": resolved_backend,
            "data_dir": str(data_dir),
        },
        "hybrid_fusion_model": {
            "width_bins": width_bins,
            "valid": _fusion_metrics(
                bundle.valid,
                valid_anchor_avg,
                valid_center_avg,
                valid_width_sum / len(seeds),
                sorted_anchors,
                valid_y,
                width_bins,
                config,
            ),
            "reformulated": _fusion_metrics(
                bundle.reformulated,
                reform_anchor_avg,
                reform_center_avg,
                reform_width_sum / len(seeds),
                sorted_anchors,
                reform_y,
                width_bins,
                config,
            ),
        },
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = RUNS_DIR / f"hybrid_fusion_{stamp}_{data_dir.name}_{resolved_backend}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results, path


def _default_width_bins(num_labels: int) -> list[int]:
    max_width = max(num_labels, 1)
    bins = list(range(1, min(8, max_width) + 1))
    width = 16
    while width < max_width:
        bins.append(width)
        width *= 2
    if bins[-1] != max_width:
        bins.append(max_width)
    return bins


def _example_range_indices(example, anchor_to_index: dict[str, int]) -> tuple[int, int]:
    start_refmark = example.gold_start_refmark or example.refmark
    end_refmark = example.gold_end_refmark or example.refmark
    start_idx = anchor_to_index[start_refmark]
    end_idx = anchor_to_index[end_refmark]
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx
    return start_idx, end_idx


def _range_centers(examples, anchor_to_index: dict[str, int]) -> np.ndarray:
    centers = []
    for example in examples:
        start_idx, end_idx = _example_range_indices(example, anchor_to_index)
        centers.append((start_idx + end_idx) // 2)
    return np.array(centers, dtype=np.int64)


def _range_starts_ends(examples, anchor_to_index: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    starts = []
    ends = []
    for example in examples:
        start_idx, end_idx = _example_range_indices(example, anchor_to_index)
        starts.append(start_idx)
        ends.append(end_idx)
    return np.array(starts, dtype=np.int64), np.array(ends, dtype=np.int64)


def _build_width_targets(examples, anchor_to_index: dict[str, int], width_bins: list[int]) -> tuple[np.ndarray, np.ndarray]:
    width_labels = np.zeros(len(examples), dtype=np.int64)
    width_targets = np.zeros((len(examples), len(width_bins)), dtype=np.float32)
    for row, example in enumerate(examples):
        start_idx, end_idx = _example_range_indices(example, anchor_to_index)
        gold_width = end_idx - start_idx + 1
        best_idx = min(range(len(width_bins)), key=lambda idx: abs(width_bins[idx] - gold_width))
        width_labels[row] = best_idx
        for idx, width in enumerate(width_bins):
            distance = abs(width - gold_width)
            width_targets[row, idx] = 1.0 / (1.0 + distance)
        width_targets[row] /= width_targets[row].sum()
    return width_labels, width_targets


def _build_count_labels(examples, anchor_to_index: dict[str, int], count_cap: int) -> np.ndarray:
    labels = np.zeros(len(examples), dtype=np.int64)
    for row, example in enumerate(examples):
        start_idx, end_idx = _example_range_indices(example, anchor_to_index)
        gold_width = end_idx - start_idx + 1
        labels[row] = min(max(gold_width, 1), count_cap) - 1
    return labels


def _bm25_logits_for_splits(bundle, vocab: Vocab, label_index: dict[str, int], *, enrich_train: bool = False) -> tuple[np.ndarray, np.ndarray]:
    retriever = _fit_bm25_retriever(bundle, vocab, enrich_train=enrich_train)
    valid_logits = retriever.logits([example.question for example in bundle.valid], label_index)
    reform_logits = retriever.logits([example.question for example in bundle.reformulated], label_index)
    return valid_logits, reform_logits


def _fit_bm25_retriever(bundle, vocab: Vocab, *, enrich_train: bool = False) -> BM25AnchorRetriever:
    retriever = BM25AnchorRetriever(vocab)
    anchor_texts = [anchor.text for anchor in bundle.anchors]
    if enrich_train:
        questions_by_refmark: dict[str, list[str]] = {anchor.refmark: [] for anchor in bundle.anchors}
        for example in bundle.train:
            questions_by_refmark.setdefault(example.refmark, []).append(example.question)
        anchor_texts = [
            anchor.text + "\n" + "\n".join(questions_by_refmark.get(anchor.refmark, []))
            for anchor in bundle.anchors
        ]
    retriever.fit(
        anchor_texts,
        [anchor.refmark for anchor in bundle.anchors],
    )
    return retriever


def _normalize_logits(logits: np.ndarray) -> np.ndarray:
    row_mean = logits.mean(axis=1, keepdims=True)
    row_std = logits.std(axis=1, keepdims=True)
    row_std[row_std < 1e-6] = 1.0
    return (logits - row_mean) / row_std


def _apply_bm25_neighborhood_prior(logits: np.ndarray, bm25_logits: np.ndarray, config: ExperimentConfig) -> np.ndarray:
    if config.bm25_weight <= 0.0 or config.bm25_top_k <= 0:
        return logits
    normalized_bm25 = _normalize_logits(bm25_logits)
    fused = logits + config.bm25_weight * normalized_bm25
    penalized = fused.copy()
    top_k = min(config.bm25_top_k, bm25_logits.shape[1])
    for row in range(bm25_logits.shape[0]):
        selected = np.argsort(bm25_logits[row])[-top_k:]
        mask = np.zeros(bm25_logits.shape[1], dtype=bool)
        for idx in selected:
            start = max(int(idx) - config.bm25_margin, 0)
            end = min(int(idx) + config.bm25_margin + 1, bm25_logits.shape[1])
            mask[start:end] = True
        penalized[row, ~mask] -= config.bm25_outside_penalty
    return penalized


def _route_labels(examples, anchor_to_index: dict[str, int], bucket_size: int) -> np.ndarray:
    labels = []
    for example in examples:
        start_idx, end_idx = _example_range_indices(example, anchor_to_index)
        labels.append(((start_idx + end_idx) // 2) // bucket_size)
    return np.array(labels, dtype=np.int64)


def _apply_route_prior(
    center_logits: np.ndarray,
    route_logits: np.ndarray,
    *,
    bucket_size: int,
    top_k: int,
    margin: int,
    outside_penalty: float,
) -> np.ndarray:
    adjusted = center_logits.copy()
    num_buckets = route_logits.shape[1]
    num_labels = center_logits.shape[1]
    top_k = max(min(top_k, num_buckets), 1)
    for row in range(center_logits.shape[0]):
        allowed = np.zeros(num_labels, dtype=bool)
        for bucket in np.argsort(route_logits[row])[-top_k:]:
            for neighbor in range(max(int(bucket) - margin, 0), min(int(bucket) + margin + 1, num_buckets)):
                start = neighbor * bucket_size
                end = min(start + bucket_size, num_labels)
                allowed[start:end] = True
        adjusted[row, ~allowed] -= outside_penalty
    return adjusted


def _decode_center_width(center_idx: int, width_idx: int, width_bins: list[int], num_labels: int) -> tuple[int, int]:
    width = max(int(width_bins[width_idx]), 1)
    half_left = (width - 1) // 2
    half_right = width // 2
    start_idx = max(center_idx - half_left, 0)
    end_idx = min(center_idx + half_right, num_labels - 1)
    return start_idx, end_idx


def _decode_center_width_boundary_aware(
    row: int,
    center_logits: np.ndarray,
    width_logits: np.ndarray,
    width_bins: list[int],
    num_labels: int,
    *,
    center_top_k: int = 5,
    width_top_k: int = 5,
) -> tuple[int, int]:
    center_scores = center_logits[row]
    width_scores = width_logits[row]
    center_candidates = np.argsort(center_scores)[-min(center_top_k, len(center_scores)):]
    width_candidates = np.argsort(width_scores)[-min(width_top_k, len(width_scores)):]
    width_probs = np.exp(width_scores - width_scores.max())
    width_probs /= max(float(width_probs.sum()), 1e-9)
    expected_width = float(sum(prob * width for prob, width in zip(width_probs, width_bins, strict=True)))
    best_score = None
    best_span = (0, 0)
    for center_idx in center_candidates:
        for width_idx in width_candidates:
            start_idx, end_idx = _decode_center_width(int(center_idx), int(width_idx), width_bins, num_labels)
            width = end_idx - start_idx + 1
            boundary_penalty = 0.04 * abs(np.log1p(width) - np.log1p(expected_width))
            edge_penalty = 0.02 if start_idx == 0 or end_idx == num_labels - 1 else 0.0
            score = float(center_scores[center_idx] + width_scores[width_idx] - boundary_penalty - edge_penalty)
            if best_score is None or score > best_score:
                best_score = score
                best_span = (start_idx, end_idx)
    return best_span


def _center_width_metrics(examples, center_logits, width_logits, sorted_anchors, labels, width_bins):
    cover = 0
    exact_match = 0
    answer_overlap_hits = 0
    answer_cover_hits = 0
    answer_breadth_sum = 0.0
    anchor_overlap_hits = 0
    anchor_cover_hits = 0
    anchor_iou_sum = 0.0
    anchor_breadth_sum = 0.0
    predicted_anchor_count_sum = 0.0
    anchor_to_index = {anchor.refmark: idx for idx, anchor in enumerate(sorted_anchors)}
    width_labels, _ = _build_width_targets(examples, anchor_to_index, width_bins)
    width_top3 = topk_accuracy(width_logits, width_labels, k=min(3, width_logits.shape[1]))
    for row, example in enumerate(examples):
        start_idx, end_idx = _decode_center_width_boundary_aware(
            row,
            center_logits,
            width_logits,
            width_bins,
            len(sorted_anchors),
        )
        gold_start_idx, gold_end_idx = _example_range_indices(example, anchor_to_index)
        anchor_overlap = max(0, min(end_idx, gold_end_idx) - max(start_idx, gold_start_idx) + 1)
        gold_anchor_count = gold_end_idx - gold_start_idx + 1
        pred_anchor_count = end_idx - start_idx + 1
        if anchor_overlap > 0:
            anchor_overlap_hits += 1
        if start_idx <= gold_start_idx and end_idx >= gold_end_idx:
            cover += 1
            anchor_cover_hits += 1
        if start_idx == gold_start_idx and end_idx == gold_end_idx:
            exact_match += 1
        anchor_union = max(end_idx, gold_end_idx) - min(start_idx, gold_start_idx) + 1
        anchor_iou_sum += anchor_overlap / max(anchor_union, 1)
        range_start = int(sorted_anchors[start_idx].region_start)
        range_end = int(sorted_anchors[end_idx].region_end)
        answer_start = int(example.answer_start)
        answer_end = int(example.answer_end)
        overlap = _span_overlap(range_start, range_end, answer_start, answer_end)
        answer_len = max(answer_end - answer_start, 1)
        range_len = max(range_end - range_start, 1)
        if overlap > 0:
            answer_overlap_hits += 1
        if overlap >= answer_len:
            answer_cover_hits += 1
        answer_breadth_sum += range_len / answer_len
        gold_range_start = int(sorted_anchors[gold_start_idx].region_start)
        gold_range_end = int(sorted_anchors[gold_end_idx].region_end)
        gold_range_len = max(gold_range_end - gold_range_start, 1)
        anchor_breadth_sum += range_len / gold_range_len
        predicted_anchor_count_sum += pred_anchor_count
    count = max(len(examples), 1)
    return {
        "exact_range_hit": cover / count,
        "exact_range_match": exact_match / count,
        "answer_overlap_rate": answer_overlap_hits / count,
        "answer_cover_rate": answer_cover_hits / count,
        "breadth_ratio": answer_breadth_sum / count,
        "anchor_region_overlap_rate": anchor_overlap_hits / count,
        "anchor_region_cover_rate": anchor_cover_hits / count,
        "anchor_region_iou": anchor_iou_sum / count,
        "anchor_region_breadth_ratio": anchor_breadth_sum / count,
        "predicted_anchor_count": predicted_anchor_count_sum / count,
        "center_top3_accuracy": topk_accuracy(center_logits, labels, k=min(3, center_logits.shape[1])),
        "width_top3_accuracy": width_top3,
        "relaxed_density": _relaxed_density_metrics(examples, center_logits, width_logits, sorted_anchors, width_bins),
    }


def _relaxed_density_metrics(examples, center_logits, width_logits, sorted_anchors, width_bins) -> dict[str, dict[str, float]]:
    anchor_to_index = {anchor.refmark: idx for idx, anchor in enumerate(sorted_anchors)}
    spans = _decode_spans(center_logits, width_logits, width_bins, len(sorted_anchors))
    output: dict[str, dict[str, float]] = {}
    for factor in (2, 3, 4, 5):
        overlap_hits = 0
        center_hits = 0
        cover_hits = 0
        for example, (pred_start, pred_end) in zip(examples, spans, strict=True):
            gold_start, gold_end = _example_range_indices(example, anchor_to_index)
            gold_width = gold_end - gold_start + 1
            target_width = max(int(round(gold_width * factor)), gold_width)
            extra = max(target_width - gold_width, 0)
            relaxed_start = max(gold_start - extra // 2, 0)
            relaxed_end = min(gold_end + (extra - extra // 2), len(sorted_anchors) - 1)
            pred_center = (pred_start + pred_end) // 2
            if max(pred_start, relaxed_start) <= min(pred_end, relaxed_end):
                overlap_hits += 1
            if relaxed_start <= pred_center <= relaxed_end:
                center_hits += 1
            if pred_start <= relaxed_start and pred_end >= relaxed_end:
                cover_hits += 1
        count = max(len(examples), 1)
        output[f"{factor}x"] = {
            "overlap_hit": overlap_hits / count,
            "center_hit": center_hits / count,
            "cover_hit": cover_hits / count,
        }
    return output


def _center_width_metrics_by_style(examples, center_logits, width_logits, sorted_anchors, labels, width_bins) -> dict[str, dict]:
    groups: dict[str, list[int]] = {}
    for idx, example in enumerate(examples):
        groups.setdefault(example.prompt_style, []).append(idx)
    output: dict[str, dict] = {}
    for style, indices in sorted(groups.items()):
        subset_examples = [examples[idx] for idx in indices]
        subset_center = center_logits[indices]
        subset_width = width_logits[indices]
        subset_labels = labels[indices]
        output[style] = _center_width_metrics(
            subset_examples,
            subset_center,
            subset_width,
            sorted_anchors,
            subset_labels,
            width_bins,
        )
    return output


def _average_center_width_logits(models, questions: list[str], vocab: Vocab) -> tuple[np.ndarray, np.ndarray]:
    x = TorchBoWClassifier.counts_matrix(
        [vocab.encode(question) for question in questions],
        len(vocab.id_to_token),
    )
    center_sum = None
    width_sum = None
    for model in models:
        center_logits, width_logits = model.logits(x)
        center_sum = center_logits if center_sum is None else center_sum + center_logits
        width_sum = width_logits if width_sum is None else width_sum + width_logits
    return center_sum / len(models), width_sum / len(models)


def _train_center_width_ensemble(
    train_examples,
    vocab: Vocab,
    sorted_anchors,
    anchor_to_index: dict[str, int],
    width_bins: list[int],
    device: str,
    config: ExperimentConfig,
    seeds: list[int],
) -> list[TorchBoWCenterWidthPredictor]:
    train_x = TorchBoWClassifier.counts_matrix(
        [vocab.encode(example.question) for example in train_examples],
        len(vocab.id_to_token),
    )
    train_center = _range_centers(train_examples, anchor_to_index)
    width_labels, width_targets = _build_width_targets(train_examples, anchor_to_index, width_bins)
    count_labels = _build_count_labels(train_examples, anchor_to_index, config.count_label_cap)
    center_targets, _ = _build_range_soft_targets(train_examples, sorted_anchors, anchor_to_index, config)
    trained_models: list[TorchBoWCenterWidthPredictor] = []
    for seed in seeds:
        model = TorchBoWCenterWidthPredictor(
            vocab_size=len(vocab.id_to_token),
            num_labels=len(sorted_anchors),
            width_bins=width_bins,
            device=device,
            hidden_dim=config.mlp_hidden_dim,
            hidden2_dim=config.mlp_hidden2_dim,
            dropout=config.mlp_dropout,
            count_cap=config.count_label_cap,
        )
        model.train_hybrid(
            train_x,
            train_center,
            width_labels,
            center_targets,
            width_targets,
            count_labels=count_labels,
            exact_weight=config.center_exact_weight,
            soft_weight=config.center_soft_weight,
            width_weight=config.center_width_weight,
            count_weight=config.count_aux_weight,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            weight_decay=config.weight_decay,
            seed=seed,
        )
        trained_models.append(model)
    return trained_models


def _decode_spans(center_logits: np.ndarray, width_logits: np.ndarray, width_bins: list[int], num_labels: int) -> list[tuple[int, int]]:
    return [
        _decode_center_width_boundary_aware(row, center_logits, width_logits, width_bins, num_labels)
        for row in range(center_logits.shape[0])
    ]


def _bm25_candidate_spans(
    bm25_logits: np.ndarray,
    num_labels: int,
    *,
    top_k: int = 5,
    margins: tuple[int, ...] = (0, 1, 2, 3),
) -> list[list[tuple[int, int]]]:
    outputs: list[list[tuple[int, int]]] = []
    for row in range(bm25_logits.shape[0]):
        candidates: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        top = np.argsort(bm25_logits[row])[-min(top_k, bm25_logits.shape[1]) :]
        for idx in top:
            for margin in margins:
                start = max(int(idx) - margin, 0)
                end = min(int(idx) + margin, num_labels - 1)
                span = (start, end)
                if span not in seen:
                    seen.add(span)
                    candidates.append(span)
        outputs.append(candidates)
    return outputs


def _span_iou(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
    overlap = max(0, min(end_a, end_b) - max(start_a, start_b) + 1)
    union = max(end_a, end_b) - min(start_a, start_b) + 1
    return overlap / max(union, 1)


def _candidate_spans_for_row(
    row: int,
    center_logits: np.ndarray,
    width_logits: np.ndarray,
    bm25_logits: np.ndarray,
    width_bins: list[int],
    config: ExperimentConfig,
) -> list[tuple[int, int]]:
    num_labels = center_logits.shape[1]
    seen: set[tuple[int, int]] = set()
    candidates: list[tuple[int, int]] = []

    def add(span: tuple[int, int]) -> None:
        start_idx, end_idx = span
        clamped = (max(start_idx, 0), min(end_idx, num_labels - 1))
        if clamped not in seen:
            seen.add(clamped)
            candidates.append(clamped)

    add(_decode_center_width_boundary_aware(row, center_logits, width_logits, width_bins, num_labels))
    center_candidates = np.argsort(center_logits[row])[-min(config.learned_rerank_center_top_k, num_labels) :]
    width_candidates = np.argsort(width_logits[row])[-min(config.learned_rerank_width_top_k, len(width_bins)) :]
    for center_idx in center_candidates:
        for width_idx in width_candidates:
            add(_decode_center_width(int(center_idx), int(width_idx), width_bins, num_labels))
    for span in _bm25_candidate_spans(
        bm25_logits[row : row + 1],
        num_labels,
        top_k=max(config.bm25_top_k, 5),
    )[0]:
        add(span)
    return candidates


def _candidate_target_score(
    example,
    span: tuple[int, int],
    sorted_anchors,
    anchor_to_index: dict[str, int],
) -> float:
    gold_start, gold_end = _example_range_indices(example, anchor_to_index)
    pred_start, pred_end = span
    overlap = max(0, min(pred_end, gold_end) - max(pred_start, gold_start) + 1)
    pred_width = pred_end - pred_start + 1
    gold_width = gold_end - gold_start + 1
    precision = overlap / max(pred_width, 1)
    recall = overlap / max(gold_width, 1)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    cover_bonus = 0.25 if pred_start <= gold_start and pred_end >= gold_end else 0.0
    exact_bonus = 0.10 if pred_start == gold_start and pred_end == gold_end else 0.0
    breadth_penalty = 0.08 * max((pred_width / max(gold_width, 1)) - 1.0, 0.0)

    pred_region_start = int(sorted_anchors[pred_start].region_start)
    pred_region_end = int(sorted_anchors[pred_end].region_end)
    gold_region_start = int(example.answer_start)
    gold_region_end = int(example.answer_end)
    answer_overlap = _span_overlap(pred_region_start, pred_region_end, gold_region_start, gold_region_end)
    answer_precision = answer_overlap / max(pred_region_end - pred_region_start, 1)
    answer_recall = answer_overlap / max(gold_region_end - gold_region_start, 1)
    answer_f1 = 0.0 if answer_precision + answer_recall == 0.0 else 2.0 * answer_precision * answer_recall / (answer_precision + answer_recall)
    return max(0.0, 0.65 * f1 + 0.20 * answer_f1 + cover_bonus + exact_bonus - breadth_penalty)


def _candidate_features(
    example,
    span: tuple[int, int],
    center_row: np.ndarray,
    width_row: np.ndarray,
    bm25_row: np.ndarray,
    width_bins: list[int],
    sorted_anchors,
    default_span: tuple[int, int],
) -> list[float]:
    start_idx, end_idx = span
    center_idx = (start_idx + end_idx) // 2
    span_width = end_idx - start_idx + 1
    width_idx = min(range(len(width_bins)), key=lambda idx: abs(width_bins[idx] - span_width))
    center_slice = center_row[start_idx : end_idx + 1]
    bm25_slice = bm25_row[start_idx : end_idx + 1]
    span_text = " ".join(anchor.text for anchor in sorted_anchors[start_idx : end_idx + 1])
    question_tokens = set(tokenize(example.question))
    span_tokens = set(tokenize(span_text))
    token_overlap = len(question_tokens & span_tokens)
    token_union = max(len(question_tokens | span_tokens), 1)
    token_recall = token_overlap / max(len(question_tokens), 1)
    default_iou = _span_iou(start_idx, end_idx, default_span[0], default_span[1])
    width_probs = np.exp(width_row - width_row.max())
    width_probs /= max(float(width_probs.sum()), 1e-9)
    expected_width = float(sum(prob * width for prob, width in zip(width_probs, width_bins, strict=True)))
    width_log_gap = abs(np.log1p(span_width) - np.log1p(expected_width))
    default_center = (default_span[0] + default_span[1]) // 2
    boundary_shift = abs(center_idx - default_center)
    bm25_peak = float(bm25_slice.max())
    bm25_mean = float(bm25_slice.mean())
    center_peak = float(center_slice.max())
    width_score = float(width_row[width_idx])
    jaccard = float(token_overlap / token_union)
    bm25_order = np.argsort(bm25_row)[::-1]
    bm25_ranks = np.empty_like(bm25_order)
    bm25_ranks[bm25_order] = np.arange(len(bm25_order))
    best_rank = int(bm25_ranks[start_idx : end_idx + 1].min())
    rank_recip = 1.0 / (1.0 + best_rank)
    return [
        1.0,
        float(center_row[center_idx]),
        center_peak,
        float(center_slice.mean()),
        width_score,
        bm25_peak,
        bm25_mean,
        float(span_width),
        float(span_width / max(len(sorted_anchors), 1)),
        jaccard,
        float(token_recall),
        float(default_iou),
        float(np.log1p(span_width)),
        float(width_log_gap),
        float(boundary_shift),
        center_peak * bm25_peak,
        width_score * bm25_peak,
        bm25_peak * jaccard,
        bm25_peak * token_recall,
        default_iou * bm25_peak,
        float(best_rank),
        float(rank_recip),
        bm25_mean * rank_recip,
    ]


def _structured_hard_negative_spans(example, anchor_to_index: dict[str, int], num_labels: int) -> list[tuple[int, int]]:
    gold_start, gold_end = _example_range_indices(example, anchor_to_index)
    gold_width = gold_end - gold_start + 1
    widths = sorted({1, max(gold_width - 1, 1), gold_width, gold_width + 1, gold_width + 2, gold_width + 3})
    spans: set[tuple[int, int]] = set()
    for shift in (-4, -2, -1, 1, 2, 4):
        start = max(gold_start + shift, 0)
        end = min(gold_end + shift, num_labels - 1)
        if start <= end:
            spans.add((start, end))
    for margin in (1, 2, 3):
        spans.add((max(gold_start - margin, 0), gold_end))
        spans.add((gold_start, min(gold_end + margin, num_labels - 1)))
        spans.add((max(gold_start - margin, 0), min(gold_end + margin, num_labels - 1)))
        if gold_width > 1:
            spans.add((min(gold_start + margin, gold_end), gold_end))
            spans.add((gold_start, max(gold_end - margin, gold_start)))
    center = (gold_start + gold_end) // 2
    for width in widths:
        half = width // 2
        spans.add((max(center - half, 0), min(center - half + width - 1, num_labels - 1)))
    for far_gap in (8, 12, 16):
        left_start = max(gold_start - far_gap, 0)
        left_end = min(left_start + gold_width - 1, num_labels - 1)
        right_end = min(gold_end + far_gap, num_labels - 1)
        right_start = max(right_end - gold_width + 1, 0)
        if left_start <= left_end:
            spans.add((left_start, left_end))
        if right_start <= right_end:
            spans.add((right_start, right_end))
    spans.discard((gold_start, gold_end))
    return sorted(span for span in spans if span[0] <= span[1])


def _fit_mlp_candidate_reranker(
    examples,
    center_logits: np.ndarray,
    width_logits: np.ndarray,
    bm25_logits: np.ndarray,
    width_bins: list[int],
    sorted_anchors,
    anchor_to_index: dict[str, int],
    config: ExperimentConfig,
) -> dict[str, object]:
    import torch

    grouped_features: list[np.ndarray] = []
    grouped_targets: list[np.ndarray] = []
    all_rows: list[np.ndarray] = []
    for row, example in enumerate(examples):
        default_span = _decode_center_width_boundary_aware(row, center_logits, width_logits, width_bins, center_logits.shape[1])
        candidate_set = set(_candidate_spans_for_row(row, center_logits, width_logits, bm25_logits, width_bins, config))
        candidate_set.update(_structured_hard_negative_spans(example, anchor_to_index, center_logits.shape[1]))
        candidates = sorted(candidate_set)
        feature_rows: list[list[float]] = []
        target_rows: list[float] = []
        for span in candidates:
            feature_rows.append(
                _candidate_features(
                    example,
                    span,
                    center_logits[row],
                    width_logits[row],
                    bm25_logits[row],
                    width_bins,
                    sorted_anchors,
                    default_span,
                )
            )
            target_rows.append(min(1.0, _candidate_target_score(example, span, sorted_anchors, anchor_to_index)))
        if not feature_rows:
            continue
        feature_array = np.asarray(feature_rows, dtype=np.float32)
        target_array = np.asarray(target_rows, dtype=np.float32)
        grouped_features.append(feature_array)
        grouped_targets.append(target_array)
        all_rows.append(feature_array)
    if not all_rows:
        return {"model": None, "mean": np.zeros(1, dtype=np.float32), "std": np.ones(1, dtype=np.float32)}
    x = np.concatenate(all_rows, axis=0)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    mean[0] = 0.0
    std[0] = 1.0
    model = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], config.learned_rerank_hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Dropout(config.learned_rerank_dropout),
        torch.nn.Linear(config.learned_rerank_hidden_dim, 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=config.learned_rerank_l2)
    rng = np.random.default_rng(config.seed)
    model.train()
    for _ in range(config.learned_rerank_epochs):
        order = rng.permutation(len(grouped_features))
        for idx in order:
            features = (grouped_features[idx] - mean) / std
            targets = grouped_targets[idx]
            if len(features) <= 1:
                continue
            xb = torch.tensor(features, dtype=torch.float32)
            target_tensor = torch.tensor(targets, dtype=torch.float32)
            logits = model(xb).squeeze(-1)
            best_idx = int(target_tensor.argmax().item())
            best_tensor = torch.tensor([best_idx], dtype=torch.long)
            positive_mask = (target_tensor >= max(float(target_tensor.max().item()) - 0.05, 0.70)).float()
            loss_rank = torch.nn.functional.cross_entropy(logits.unsqueeze(0), best_tensor)
            best_logit = logits[best_idx]
            neg_mask = torch.ones_like(logits, dtype=torch.bool)
            neg_mask[best_idx] = False
            pairwise_logits = logits[neg_mask]
            if len(pairwise_logits) > 0:
                pairwise_loss = torch.relu(config.pairwise_rerank_margin - (best_logit - pairwise_logits)).mean()
            else:
                pairwise_loss = torch.tensor(0.0, dtype=torch.float32)
            loss_aux = torch.nn.functional.binary_cross_entropy_with_logits(logits, positive_mask)
            loss_reg = torch.nn.functional.mse_loss(torch.sigmoid(logits), target_tensor)
            loss = loss_rank + 0.35 * pairwise_loss + 0.20 * loss_aux + 0.10 * loss_reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return {"model": model, "mean": mean, "std": std}


def _apply_mlp_candidate_reranker(
    examples,
    center_logits: np.ndarray,
    width_logits: np.ndarray,
    bm25_logits: np.ndarray,
    width_bins: list[int],
    sorted_anchors,
    config: ExperimentConfig,
    reranker: dict[str, object],
) -> list[tuple[int, int]]:
    import torch

    outputs: list[tuple[int, int]] = []
    model = reranker["model"]
    if model is None:
        return _decode_spans(center_logits, width_logits, width_bins, len(sorted_anchors))
    mean = reranker["mean"]
    std = reranker["std"]
    for row, example in enumerate(examples):
        default_span = _decode_center_width_boundary_aware(row, center_logits, width_logits, width_bins, center_logits.shape[1])
        candidates = _candidate_spans_for_row(row, center_logits, width_logits, bm25_logits, width_bins, config)
        best_span = default_span
        best_score = None
        for span in candidates:
            features = np.asarray(
                _candidate_features(
                    example,
                    span,
                    center_logits[row],
                    width_logits[row],
                    bm25_logits[row],
                    width_bins,
                    sorted_anchors,
                    default_span,
                ),
                dtype=np.float32,
            )
            scaled = (features - mean) / std
            xb = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
            score = float(model(xb).squeeze().item())
            if best_score is None or score > best_score:
                best_score = score
                best_span = span
        outputs.append(best_span)
    return outputs


def _local_candidate_pool(
    row: int,
    example,
    center_logits: np.ndarray,
    width_logits: np.ndarray,
    bm25_logits: np.ndarray,
    width_bins: list[int],
    anchor_to_index: dict[str, int],
    config: ExperimentConfig,
) -> list[tuple[int, int]]:
    candidate_set = set(_candidate_spans_for_row(row, center_logits, width_logits, bm25_logits, width_bins, config))
    candidate_set.update(_structured_hard_negative_spans(example, anchor_to_index, center_logits.shape[1]))
    candidates = sorted(candidate_set)
    max_candidates = max(int(config.local_cross_max_candidates), 1)
    if len(candidates) <= max_candidates:
        return candidates
    default_span = _decode_center_width_boundary_aware(row, center_logits, width_logits, width_bins, center_logits.shape[1])
    ranked = sorted(
        candidates,
        key=lambda span: _default_span_score(span, center_logits[row], width_logits[row], bm25_logits[row], width_bins),
        reverse=True,
    )
    if default_span not in ranked[:max_candidates]:
        ranked = [default_span] + [span for span in ranked if span != default_span]
    return sorted(ranked[:max_candidates])


def _span_text(sorted_anchors, span: tuple[int, int]) -> str:
    start_idx, end_idx = span
    return " ".join(anchor.text for anchor in sorted_anchors[start_idx : end_idx + 1])


def _train_local_cross_reranker(
    examples,
    center_logits: np.ndarray,
    width_logits: np.ndarray,
    bm25_logits: np.ndarray,
    width_bins: list[int],
    sorted_anchors,
    anchor_to_index: dict[str, int],
    vocab: Vocab,
    config: ExperimentConfig,
    device: str,
) -> dict[str, object]:
    import torch

    grouped_texts: list[np.ndarray] = []
    grouped_dense: list[np.ndarray] = []
    grouped_targets: list[np.ndarray] = []
    all_dense: list[np.ndarray] = []
    vocab_size = len(vocab.id_to_token)
    for row, example in enumerate(examples):
        default_span = _decode_center_width_boundary_aware(row, center_logits, width_logits, width_bins, center_logits.shape[1])
        candidates = _local_candidate_pool(row, example, center_logits, width_logits, bm25_logits, width_bins, anchor_to_index, config)
        if not candidates:
            continue
        joint_encoded = [
            vocab.encode(f"{example.question} [SEP] {_span_text(sorted_anchors, span)}")
            for span in candidates
        ]
        text_x = TorchBoWClassifier.counts_matrix(joint_encoded, vocab_size)
        dense_x = np.asarray(
            [
                _candidate_features(
                    example,
                    span,
                    center_logits[row],
                    width_logits[row],
                    bm25_logits[row],
                    width_bins,
                    sorted_anchors,
                    default_span,
                )
                for span in candidates
            ],
            dtype=np.float32,
        )
        target_y = np.asarray(
            [min(1.0, _candidate_target_score(example, span, sorted_anchors, anchor_to_index)) for span in candidates],
            dtype=np.float32,
        )
        grouped_texts.append(text_x)
        grouped_dense.append(dense_x)
        grouped_targets.append(target_y)
        all_dense.append(dense_x)
    if not grouped_texts:
        return {"model": None}
    dense_all = np.concatenate(all_dense, axis=0)
    dense_mean = dense_all.mean(axis=0)
    dense_std = dense_all.std(axis=0)
    dense_std[dense_std < 1e-6] = 1.0
    dense_mean[0] = 0.0
    dense_std[0] = 1.0
    model = torch.nn.Sequential(
        torch.nn.Linear(vocab_size + dense_all.shape[1], config.local_cross_hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Dropout(config.local_cross_dropout),
        torch.nn.Linear(config.local_cross_hidden_dim, 1),
    ).to(torch.device(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=config.learned_rerank_l2)
    rng = np.random.default_rng(config.seed)
    model.train()
    for _ in range(config.local_cross_epochs):
        order = rng.permutation(len(grouped_texts))
        for idx in order:
            if len(grouped_targets[idx]) <= 1:
                continue
            text_x = torch.tensor(grouped_texts[idx], dtype=torch.float32, device=model[0].weight.device)
            dense_x = torch.tensor((grouped_dense[idx] - dense_mean) / dense_std, dtype=torch.float32, device=model[0].weight.device)
            xb = torch.cat([text_x, dense_x], dim=1)
            targets = torch.tensor(grouped_targets[idx], dtype=torch.float32, device=model[0].weight.device)
            logits = model(xb).squeeze(-1)
            best_idx = int(targets.argmax().item())
            best_tensor = torch.tensor([best_idx], dtype=torch.long, device=model[0].weight.device)
            best_logit = logits[best_idx]
            neg_mask = torch.ones_like(logits, dtype=torch.bool)
            neg_mask[best_idx] = False
            pairwise_logits = logits[neg_mask]
            pairwise_loss = torch.relu(config.pairwise_rerank_margin - (best_logit - pairwise_logits)).mean() if len(pairwise_logits) > 0 else torch.tensor(0.0, device=model[0].weight.device)
            positive_mask = (targets >= max(float(targets.max().item()) - 0.05, 0.70)).float()
            loss_rank = torch.nn.functional.cross_entropy(logits.unsqueeze(0), best_tensor)
            loss_aux = torch.nn.functional.binary_cross_entropy_with_logits(logits, positive_mask)
            loss_reg = torch.nn.functional.mse_loss(torch.sigmoid(logits), targets)
            loss = loss_rank + 0.35 * pairwise_loss + 0.20 * loss_aux + 0.10 * loss_reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return {"model": model, "dense_mean": dense_mean, "dense_std": dense_std}


def _apply_local_cross_reranker(
    examples,
    center_logits: np.ndarray,
    width_logits: np.ndarray,
    bm25_logits: np.ndarray,
    width_bins: list[int],
    sorted_anchors,
    anchor_to_index: dict[str, int],
    vocab: Vocab,
    config: ExperimentConfig,
    reranker: dict[str, object],
) -> list[tuple[int, int]]:
    spans, _ = _apply_local_cross_reranker_with_confidence(
        examples,
        center_logits,
        width_logits,
        bm25_logits,
        width_bins,
        sorted_anchors,
        anchor_to_index,
        vocab,
        config,
        reranker,
    )
    return spans


def _apply_local_cross_reranker_with_confidence(
    examples,
    center_logits: np.ndarray,
    width_logits: np.ndarray,
    bm25_logits: np.ndarray,
    width_bins: list[int],
    sorted_anchors,
    anchor_to_index: dict[str, int],
    vocab: Vocab,
    config: ExperimentConfig,
    reranker: dict[str, object],
) -> tuple[list[tuple[int, int]], list[float]]:
    import torch

    model = reranker.get("model")
    if model is None:
        spans = _decode_spans(center_logits, width_logits, width_bins, len(sorted_anchors))
        return spans, [0.0] * len(spans)
    dense_mean = reranker["dense_mean"]
    dense_std = reranker["dense_std"]
    outputs: list[tuple[int, int]] = []
    margins: list[float] = []
    vocab_size = len(vocab.id_to_token)
    model_device = model[0].weight.device
    for row, example in enumerate(examples):
        default_span = _decode_center_width_boundary_aware(row, center_logits, width_logits, width_bins, center_logits.shape[1])
        candidates = _local_candidate_pool(row, example, center_logits, width_logits, bm25_logits, width_bins, anchor_to_index, config)
        if not candidates:
            outputs.append(default_span)
            margins.append(0.0)
            continue
        joint_encoded = [vocab.encode(f"{example.question} [SEP] {_span_text(sorted_anchors, span)}") for span in candidates]
        text_x = TorchBoWClassifier.counts_matrix(joint_encoded, vocab_size)
        dense_x = np.asarray(
            [
                _candidate_features(
                    example,
                    span,
                    center_logits[row],
                    width_logits[row],
                    bm25_logits[row],
                    width_bins,
                    sorted_anchors,
                    default_span,
                )
                for span in candidates
            ],
            dtype=np.float32,
        )
        xb = np.concatenate([text_x, (dense_x - dense_mean) / dense_std], axis=1)
        with torch.no_grad():
            scores = model(torch.tensor(xb, dtype=torch.float32, device=model_device)).squeeze(-1).detach().cpu().numpy()
        best_idx = int(scores.argmax())
        sorted_scores = np.sort(scores)
        margin = float(sorted_scores[-1] - sorted_scores[-2]) if len(sorted_scores) > 1 else 0.0
        outputs.append(candidates[best_idx])
        margins.append(margin)
    return outputs, margins


def _row_margin(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    top = np.sort(values)[-2:]
    return float(top[-1] - top[-2])


def _default_span_score(
    span: tuple[int, int],
    center_row: np.ndarray,
    width_row: np.ndarray,
    bm25_row: np.ndarray,
    width_bins: list[int],
) -> float:
    start_idx, end_idx = span
    center_idx = (start_idx + end_idx) // 2
    span_width = end_idx - start_idx + 1
    width_idx = min(range(len(width_bins)), key=lambda idx: abs(width_bins[idx] - span_width))
    bm25_support = float(bm25_row[start_idx : end_idx + 1].max())
    width_penalty = 0.03 * abs(np.log1p(span_width) - np.log1p(width_bins[width_idx]))
    return float(center_row[center_idx] + width_row[width_idx] + 0.25 * bm25_support - width_penalty)


def _default_uncertainty(
    row: int,
    default_span: tuple[int, int],
    center_logits: np.ndarray,
    width_logits: np.ndarray,
    bm25_logits: np.ndarray,
    width_bins: list[int],
) -> float:
    center_margin = _row_margin(center_logits[row])
    width_margin = _row_margin(width_logits[row])
    bm25_top = int(np.argmax(bm25_logits[row]))
    start_idx, end_idx = default_span
    bm25_outside = 0.0 if start_idx <= bm25_top <= end_idx else 1.0
    return float(1.0 / (1.0 + center_margin + 0.5 * width_margin) + 0.35 * bm25_outside)


def _local_cross_candidate_scores(
    examples,
    center_logits: np.ndarray,
    width_logits: np.ndarray,
    bm25_logits: np.ndarray,
    width_bins: list[int],
    sorted_anchors,
    anchor_to_index: dict[str, int],
    vocab: Vocab,
    config: ExperimentConfig,
    reranker: dict[str, object],
) -> list[dict[str, object]]:
    import torch

    model = reranker.get("model")
    if model is None:
        return []
    dense_mean = reranker["dense_mean"]
    dense_std = reranker["dense_std"]
    vocab_size = len(vocab.id_to_token)
    model_device = model[0].weight.device
    rows: list[dict[str, object]] = []
    for row, example in enumerate(examples):
        default_span = _decode_center_width_boundary_aware(row, center_logits, width_logits, width_bins, center_logits.shape[1])
        candidates = _local_candidate_pool(row, example, center_logits, width_logits, bm25_logits, width_bins, anchor_to_index, config)
        if not candidates:
            rows.append(
                {
                    "default_span": default_span,
                    "candidates": [default_span],
                    "local_scores": np.array([0.0], dtype=np.float32),
                    "default_scores": np.array([0.0], dtype=np.float32),
                    "uncertainty": 0.0,
                }
            )
            continue
        joint_encoded = [vocab.encode(f"{example.question} [SEP] {_span_text(sorted_anchors, span)}") for span in candidates]
        text_x = TorchBoWClassifier.counts_matrix(joint_encoded, vocab_size)
        dense_x = np.asarray(
            [
                _candidate_features(
                    example,
                    span,
                    center_logits[row],
                    width_logits[row],
                    bm25_logits[row],
                    width_bins,
                    sorted_anchors,
                    default_span,
                )
                for span in candidates
            ],
            dtype=np.float32,
        )
        xb = np.concatenate([text_x, (dense_x - dense_mean) / dense_std], axis=1)
        with torch.no_grad():
            local_scores = model(torch.tensor(xb, dtype=torch.float32, device=model_device)).squeeze(-1).detach().cpu().numpy()
        default_scores = np.asarray(
            [_default_span_score(span, center_logits[row], width_logits[row], bm25_logits[row], width_bins) for span in candidates],
            dtype=np.float32,
        )
        rows.append(
            {
                "default_span": default_span,
                "candidates": candidates,
                "local_scores": local_scores.astype(np.float32),
                "default_scores": default_scores,
                "uncertainty": _default_uncertainty(row, default_span, center_logits, width_logits, bm25_logits, width_bins),
            }
        )
    return rows


def _normalize_row_scores(scores: np.ndarray) -> np.ndarray:
    std = float(scores.std())
    if std < 1e-6:
        return scores - float(scores.mean())
    return (scores - float(scores.mean())) / std


def _choose_blended_span(row_info: dict[str, object], policy: dict[str, float]) -> tuple[int, int]:
    if float(row_info["uncertainty"]) < policy["uncertainty_threshold"]:
        return row_info["default_span"]
    local_scores = _normalize_row_scores(row_info["local_scores"])
    default_scores = _normalize_row_scores(row_info["default_scores"])
    alpha = policy["local_weight"]
    blended = alpha * local_scores + (1.0 - alpha) * default_scores
    return row_info["candidates"][int(blended.argmax())]


def _fit_local_cross_blend_policy(
    examples,
    candidate_rows: list[dict[str, object]],
    sorted_anchors,
    anchor_to_index: dict[str, int],
    config: ExperimentConfig,
) -> dict[str, float]:
    if not candidate_rows:
        return {"local_weight": 1.0, "uncertainty_threshold": 0.0}
    weights = np.linspace(0.0, 1.0, max(config.local_cross_blend_steps, 2))
    thresholds = sorted({0.0, *[round(float(row["uncertainty"]), 3) for row in candidate_rows]})
    best_policy = {"local_weight": 1.0, "uncertainty_threshold": 0.0}
    best_score = None
    for local_weight in weights:
        for threshold in thresholds:
            policy = {"local_weight": float(local_weight), "uncertainty_threshold": float(threshold)}
            score = 0.0
            for example, row_info in zip(examples, candidate_rows, strict=True):
                span = _choose_blended_span(row_info, policy)
                score += _candidate_target_score(example, span, sorted_anchors, anchor_to_index)
            if best_score is None or score > best_score:
                best_score = score
                best_policy = policy
    return best_policy


def _reranked_broad_spans(
    center_logits: np.ndarray,
    width_logits: np.ndarray,
    bm25_logits: np.ndarray,
    width_bins: list[int],
    config: ExperimentConfig,
) -> list[tuple[int, int]]:
    bm25_candidates = _bm25_candidate_spans(bm25_logits, center_logits.shape[1], top_k=max(config.bm25_top_k, 5))
    outputs: list[tuple[int, int]] = []
    for row in range(center_logits.shape[0]):
        best_default = _decode_center_width_boundary_aware(row, center_logits, width_logits, width_bins, center_logits.shape[1])
        candidates = list(bm25_candidates[row])
        if best_default not in candidates:
            candidates.append(best_default)
        best_score = None
        best_span = best_default
        for start_idx, end_idx in candidates:
            center_idx = (start_idx + end_idx) // 2
            span_width = end_idx - start_idx + 1
            width_score = -min(abs(width - span_width) for width in width_bins)
            bm25_support = float(bm25_logits[row, start_idx : end_idx + 1].max())
            score = (
                config.broad_rerank_alpha * float(center_logits[row, center_idx])
                + config.broad_rerank_beta * width_score
                + config.broad_rerank_gamma * bm25_support
                - config.broad_rerank_delta * span_width
            )
            if best_score is None or score > best_score:
                best_score = score
                best_span = (start_idx, end_idx)
        outputs.append(best_span)
    return outputs


def _refinement_loop_split_metrics(examples, models, vocab, sorted_anchors, width_bins, anchor_to_index) -> dict:
    broad_examples = [example for example in examples if "broad" in example.prompt_style]
    precise_examples = [example for example in examples if "precise" in example.prompt_style]
    paired_count = min(len(broad_examples), len(precise_examples))
    broad_examples = broad_examples[:paired_count]
    precise_examples = precise_examples[:paired_count]

    broad_center, broad_width = _average_center_width_logits(models, [example.question for example in broad_examples], vocab)
    broad_labels = _range_centers(broad_examples, anchor_to_index)
    broad_metrics = _center_width_metrics(broad_examples, broad_center, broad_width, sorted_anchors, broad_labels, width_bins)
    predicted_broad_spans = _decode_spans(broad_center, broad_width, width_bins, len(sorted_anchors))

    gold_precise_center, gold_precise_width = _average_center_width_logits(
        models,
        [example.question for example in precise_examples],
        vocab,
    )
    precise_labels = _range_centers(precise_examples, anchor_to_index)
    gold_prompt_metrics = _center_width_metrics(
        precise_examples,
        gold_precise_center,
        gold_precise_width,
        sorted_anchors,
        precise_labels,
        width_bins,
    )

    loop_questions = [
        _refinement_question_from_predicted_broad(precise, sorted_anchors[start_idx : end_idx + 1])
        for precise, (start_idx, end_idx) in zip(precise_examples, predicted_broad_spans, strict=True)
    ]
    loop_center, loop_width = _average_center_width_logits(models, loop_questions, vocab)
    loop_metrics = _center_width_metrics(precise_examples, loop_center, loop_width, sorted_anchors, precise_labels, width_bins)
    loop_spans = _decode_spans(loop_center, loop_width, width_bins, len(sorted_anchors))
    return {
        "pairs": paired_count,
        "broad_stage": broad_metrics,
        "gold_broad_prompt_precise_stage": gold_prompt_metrics,
        "predicted_broad_loop_precise_stage": loop_metrics,
        "average_predicted_broad_anchor_count": float(np.mean([end - start + 1 for start, end in predicted_broad_spans])) if predicted_broad_spans else 0.0,
        "average_loop_precise_anchor_count": float(np.mean([end - start + 1 for start, end in loop_spans])) if loop_spans else 0.0,
    }


def _refinement_loop_split_metrics_two_model(
    examples,
    broad_models,
    refine_models,
    vocab,
    sorted_anchors,
    width_bins,
    anchor_to_index,
    broad_bm25_logits: np.ndarray | None = None,
    config: ExperimentConfig | None = None,
    broad_candidate_reranker: dict[str, np.ndarray] | None = None,
    local_cross_reranker: dict[str, object] | None = None,
    local_cross_policy: dict[str, float] | None = None,
) -> dict:
    broad_examples = [example for example in examples if "broad" in example.prompt_style]
    precise_examples = [example for example in examples if "precise" in example.prompt_style]
    paired_count = min(len(broad_examples), len(precise_examples))
    broad_examples = broad_examples[:paired_count]
    precise_examples = precise_examples[:paired_count]

    broad_center, broad_width = _average_center_width_logits(broad_models, [example.question for example in broad_examples], vocab)
    broad_labels = _range_centers(broad_examples, anchor_to_index)
    broad_metrics = _center_width_metrics(broad_examples, broad_center, broad_width, sorted_anchors, broad_labels, width_bins)
    predicted_broad_spans = _decode_spans(broad_center, broad_width, width_bins, len(sorted_anchors))
    reranked_broad_spans = None
    reranked_loop_metrics = None
    reranked_loop_spans = None
    local_cross_broad_spans = None
    local_cross_loop_metrics = None
    local_cross_loop_spans = None
    hybrid_broad_spans = None
    hybrid_loop_metrics = None
    hybrid_loop_spans = None
    if broad_bm25_logits is not None and config is not None and broad_candidate_reranker is not None:
        reranked_broad_spans = _apply_mlp_candidate_reranker(
            broad_examples,
            broad_center,
            broad_width,
            broad_bm25_logits,
            width_bins,
            sorted_anchors,
            config,
            broad_candidate_reranker,
        )
    if broad_bm25_logits is not None and config is not None and local_cross_reranker is not None:
        local_cross_candidate_rows = _local_cross_candidate_scores(
            broad_examples,
            broad_center,
            broad_width,
            broad_bm25_logits,
            width_bins,
            sorted_anchors,
            anchor_to_index,
            vocab,
            config,
            local_cross_reranker,
        )
        local_cross_broad_spans = [
            row_info["candidates"][int(row_info["local_scores"].argmax())]
            for row_info in local_cross_candidate_rows
        ]
        policy = local_cross_policy or {"local_weight": 1.0, "uncertainty_threshold": 0.0}
        hybrid_broad_spans = [
            _choose_blended_span(row_info, policy)
            for row_info in local_cross_candidate_rows
        ]

    gold_precise_center, gold_precise_width = _average_center_width_logits(
        refine_models,
        [example.question for example in precise_examples],
        vocab,
    )
    precise_labels = _range_centers(precise_examples, anchor_to_index)
    gold_prompt_metrics = _center_width_metrics(
        precise_examples,
        gold_precise_center,
        gold_precise_width,
        sorted_anchors,
        precise_labels,
        width_bins,
    )

    loop_questions = [
        _refinement_question_from_predicted_broad(precise, sorted_anchors[start_idx : end_idx + 1])
        for precise, (start_idx, end_idx) in zip(precise_examples, predicted_broad_spans, strict=True)
    ]
    loop_center, loop_width = _average_center_width_logits(refine_models, loop_questions, vocab)
    loop_metrics = _center_width_metrics(precise_examples, loop_center, loop_width, sorted_anchors, precise_labels, width_bins)
    loop_spans = _decode_spans(loop_center, loop_width, width_bins, len(sorted_anchors))
    result = {
        "pairs": paired_count,
        "broad_stage": broad_metrics,
        "gold_broad_prompt_precise_stage": gold_prompt_metrics,
        "predicted_broad_loop_precise_stage": loop_metrics,
        "average_predicted_broad_anchor_count": float(np.mean([end - start + 1 for start, end in predicted_broad_spans])) if predicted_broad_spans else 0.0,
        "average_loop_precise_anchor_count": float(np.mean([end - start + 1 for start, end in loop_spans])) if loop_spans else 0.0,
    }
    if reranked_broad_spans is not None:
        reranked_questions = [
            _refinement_question_from_predicted_broad(precise, sorted_anchors[start_idx : end_idx + 1])
            for precise, (start_idx, end_idx) in zip(precise_examples, reranked_broad_spans, strict=True)
        ]
        reranked_center, reranked_width = _average_center_width_logits(refine_models, reranked_questions, vocab)
        reranked_loop_metrics = _center_width_metrics(
            precise_examples,
            reranked_center,
            reranked_width,
            sorted_anchors,
            precise_labels,
            width_bins,
        )
        reranked_loop_spans = _decode_spans(reranked_center, reranked_width, width_bins, len(sorted_anchors))
        result.update(
            {
                "learned_reranked_broad_loop_precise_stage": reranked_loop_metrics,
                "average_learned_reranked_broad_anchor_count": float(np.mean([end - start + 1 for start, end in reranked_broad_spans])),
                "average_learned_reranked_loop_precise_anchor_count": float(np.mean([end - start + 1 for start, end in reranked_loop_spans])),
            }
        )
    if local_cross_broad_spans is not None:
        local_cross_questions = [
            _refinement_question_from_predicted_broad(precise, sorted_anchors[start_idx : end_idx + 1])
            for precise, (start_idx, end_idx) in zip(precise_examples, local_cross_broad_spans, strict=True)
        ]
        local_cross_center, local_cross_width = _average_center_width_logits(refine_models, local_cross_questions, vocab)
        local_cross_loop_metrics = _center_width_metrics(
            precise_examples,
            local_cross_center,
            local_cross_width,
            sorted_anchors,
            precise_labels,
            width_bins,
        )
        local_cross_loop_spans = _decode_spans(local_cross_center, local_cross_width, width_bins, len(sorted_anchors))
        result.update(
            {
                "local_cross_broad_loop_precise_stage": local_cross_loop_metrics,
                "average_local_cross_broad_anchor_count": float(np.mean([end - start + 1 for start, end in local_cross_broad_spans])),
                "average_local_cross_loop_precise_anchor_count": float(np.mean([end - start + 1 for start, end in local_cross_loop_spans])),
            }
        )
    if hybrid_broad_spans is not None:
        hybrid_questions = [
            _refinement_question_from_predicted_broad(precise, sorted_anchors[start_idx : end_idx + 1])
            for precise, (start_idx, end_idx) in zip(precise_examples, hybrid_broad_spans, strict=True)
        ]
        hybrid_center, hybrid_width = _average_center_width_logits(refine_models, hybrid_questions, vocab)
        hybrid_loop_metrics = _center_width_metrics(
            precise_examples,
            hybrid_center,
            hybrid_width,
            sorted_anchors,
            precise_labels,
            width_bins,
        )
        hybrid_loop_spans = _decode_spans(hybrid_center, hybrid_width, width_bins, len(sorted_anchors))
        result.update(
            {
                "hybrid_local_cross_broad_loop_precise_stage": hybrid_loop_metrics,
                "average_hybrid_local_cross_broad_anchor_count": float(np.mean([end - start + 1 for start, end in hybrid_broad_spans])),
                "average_hybrid_local_cross_loop_precise_anchor_count": float(np.mean([end - start + 1 for start, end in hybrid_loop_spans])),
                "local_cross_policy": local_cross_policy or {"local_weight": 1.0, "uncertainty_threshold": 0.0},
            }
        )
    return result


def _refinement_question_from_predicted_broad(example, broad_window) -> str:
    broad_ref = f"{broad_window[0].refmark}-{broad_window[-1].refmark}" if broad_window else "unknown"
    broad_text = " ".join(anchor.text for anchor in broad_window)
    compact = " ".join(broad_text.split())
    key = _question_focus(example.question)
    return f"Previous broad range {broad_ref}: {compact[:260]}. Narrow this to only the anchors needed for {key}."


def _question_focus(question: str) -> str:
    lowered = question.lower()
    for marker in ["needed for ", "about ", "around ", "for "]:
        idx = lowered.rfind(marker)
        if idx >= 0:
            return question[idx + len(marker) :].strip().rstrip(".?")
    return question.strip().rstrip(".?")


def _best_fused_spans(anchor_logits, center_logits, width_logits, width_bins, config: ExperimentConfig):
    center_top = min(5, center_logits.shape[1])
    width_top = min(4, width_logits.shape[1])
    outputs: list[tuple[int, int]] = []
    for row in range(center_logits.shape[0]):
        center_candidates = np.argsort(center_logits[row])[-center_top:]
        width_candidates = np.argsort(width_logits[row])[-width_top:]
        best_score = None
        best_span = (0, 0)
        for center_idx in center_candidates:
            for width_idx in width_candidates:
                start_idx, end_idx = _decode_center_width(
                    int(center_idx),
                    int(width_idx),
                    width_bins,
                    center_logits.shape[1],
                )
                anchor_support = float(anchor_logits[row, start_idx : end_idx + 1].max())
                center_support = float(center_logits[row, center_idx])
                width_support = float(width_logits[row, width_idx])
                span_width = max(end_idx - start_idx + 1, 1)
                score = (
                    config.fusion_anchor_weight * anchor_support
                    + config.fusion_range_weight * (center_support + width_support)
                    - config.fusion_breadth_penalty * span_width
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_span = (start_idx, end_idx)
        outputs.append(best_span)
    return outputs


def _fusion_metrics(examples, anchor_logits, center_logits, width_logits, sorted_anchors, labels, width_bins, config):
    spans = _best_fused_spans(anchor_logits, center_logits, width_logits, width_bins, config)
    exact = 0
    answer_overlap_hits = 0
    answer_cover_hits = 0
    answer_breadth_sum = 0.0
    anchor_overlap_hits = 0
    anchor_cover_hits = 0
    anchor_breadth_sum = 0.0
    predicted_anchor_count_sum = 0.0
    for row, example in enumerate(examples):
        start_idx, end_idx = spans[row]
        gold_idx = int(labels[row])
        if start_idx <= gold_idx <= end_idx:
            exact += 1
            anchor_overlap_hits += 1
            anchor_cover_hits += 1
        range_start = int(sorted_anchors[start_idx].region_start)
        range_end = int(sorted_anchors[end_idx].region_end)
        answer_start = int(example.answer_start)
        answer_end = int(example.answer_end)
        overlap = _span_overlap(range_start, range_end, answer_start, answer_end)
        answer_len = max(answer_end - answer_start, 1)
        range_len = max(range_end - range_start, 1)
        if overlap > 0:
            answer_overlap_hits += 1
        if overlap >= answer_len:
            answer_cover_hits += 1
        answer_breadth_sum += range_len / answer_len
        gold_anchor = sorted_anchors[gold_idx]
        gold_anchor_len = max(int(gold_anchor.region_end) - int(gold_anchor.region_start), 1)
        anchor_breadth_sum += range_len / gold_anchor_len
        predicted_anchor_count_sum += end_idx - start_idx + 1
    count = max(len(examples), 1)
    return {
        "exact_range_hit": exact / count,
        "answer_overlap_rate": answer_overlap_hits / count,
        "answer_cover_rate": answer_cover_hits / count,
        "breadth_ratio": answer_breadth_sum / count,
        "anchor_region_overlap_rate": anchor_overlap_hits / count,
        "anchor_region_cover_rate": anchor_cover_hits / count,
        "anchor_region_breadth_ratio": anchor_breadth_sum / count,
        "predicted_anchor_count": predicted_anchor_count_sum / count,
        "anchor_top3_accuracy": topk_accuracy(anchor_logits, labels, k=min(3, anchor_logits.shape[1])),
        "center_top3_accuracy": topk_accuracy(center_logits, labels, k=min(3, center_logits.shape[1])),
    }


def _build_range_soft_targets(examples, sorted_anchors, anchor_to_index, config: ExperimentConfig):
    num_labels = len(sorted_anchors)
    start_targets = np.zeros((len(examples), num_labels), dtype=np.float32)
    end_targets = np.zeros((len(examples), num_labels), dtype=np.float32)
    sigma = max(config.range_neighbor_sigma, 1e-3)
    for row, example in enumerate(examples):
        start_idx, end_idx = _example_range_indices(example, anchor_to_index)
        center = (start_idx + end_idx) // 2
        for idx, anchor in enumerate(sorted_anchors):
            distance = abs(idx - center)
            score = np.exp(-(distance ** 2) / (2.0 * sigma ** 2))
            if start_idx <= idx <= end_idx:
                score = max(score, 1.0)
            else:
                edge_distance = min(abs(idx - start_idx), abs(idx - end_idx))
                score *= max(0.0, 1.0 - config.range_breadth_penalty * edge_distance)
            start_targets[row, idx] = score
            end_targets[row, idx] = score
        start_targets[row, center] += config.range_exact_weight
        end_targets[row, center] += config.range_exact_weight
        start_targets[row] /= start_targets[row].sum()
        end_targets[row] /= end_targets[row].sum()
    return start_targets, end_targets


def _build_start_end_soft_targets(examples, sorted_anchors, anchor_to_index, config: ExperimentConfig):
    num_labels = len(sorted_anchors)
    start_targets = np.zeros((len(examples), num_labels), dtype=np.float32)
    end_targets = np.zeros((len(examples), num_labels), dtype=np.float32)
    sigma = max(config.range_neighbor_sigma, 1e-3)
    for row, example in enumerate(examples):
        gold_start, gold_end = _example_range_indices(example, anchor_to_index)
        for idx in range(num_labels):
            start_distance = abs(idx - gold_start)
            end_distance = abs(idx - gold_end)
            start_targets[row, idx] = np.exp(-(start_distance ** 2) / (2.0 * sigma ** 2))
            end_targets[row, idx] = np.exp(-(end_distance ** 2) / (2.0 * sigma ** 2))
        start_targets[row, gold_start] += config.range_exact_weight
        end_targets[row, gold_end] += config.range_exact_weight
        start_targets[row] /= start_targets[row].sum()
        end_targets[row] /= end_targets[row].sum()
    return start_targets, end_targets


def _range_metrics(
    examples,
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    sorted_anchors,
    anchor_to_index: dict[str, int],
    anchors_by_refmark,
):
    start_pred = start_logits.argmax(axis=1)
    end_pred = end_logits.argmax(axis=1)
    cover = 0
    exact_match = 0
    answer_overlap_hits = 0
    answer_cover_hits = 0
    answer_breadth_sum = 0.0
    anchor_overlap_hits = 0
    anchor_cover_hits = 0
    anchor_iou_sum = 0.0
    anchor_breadth_sum = 0.0
    predicted_anchor_count_sum = 0.0
    for row, example in enumerate(examples):
        start_idx = min(int(start_pred[row]), int(end_pred[row]))
        end_idx = max(int(start_pred[row]), int(end_pred[row]))
        gold_start_idx, gold_end_idx = _example_range_indices(example, anchor_to_index)
        anchor_overlap = max(0, min(end_idx, gold_end_idx) - max(start_idx, gold_start_idx) + 1)
        if anchor_overlap > 0:
            anchor_overlap_hits += 1
        if start_idx <= gold_start_idx and end_idx >= gold_end_idx:
            cover += 1
            anchor_cover_hits += 1
        if start_idx == gold_start_idx and end_idx == gold_end_idx:
            exact_match += 1
        anchor_union = max(end_idx, gold_end_idx) - min(start_idx, gold_start_idx) + 1
        anchor_iou_sum += anchor_overlap / max(anchor_union, 1)
        range_start = int(sorted_anchors[start_idx].region_start)
        range_end = int(sorted_anchors[end_idx].region_end)
        answer_start = int(example.answer_start)
        answer_end = int(example.answer_end)
        overlap = _span_overlap(range_start, range_end, answer_start, answer_end)
        answer_len = max(answer_end - answer_start, 1)
        range_len = max(range_end - range_start, 1)
        if overlap > 0:
            answer_overlap_hits += 1
        if overlap >= answer_len:
            answer_cover_hits += 1
        answer_breadth_sum += range_len / answer_len
        gold_range_start = int(sorted_anchors[gold_start_idx].region_start)
        gold_range_end = int(sorted_anchors[gold_end_idx].region_end)
        gold_range_len = max(gold_range_end - gold_range_start, 1)
        anchor_breadth_sum += range_len / gold_range_len
        predicted_anchor_count_sum += end_idx - start_idx + 1
    count = max(len(examples), 1)
    start_labels, end_labels = _range_starts_ends(examples, anchor_to_index)
    return {
        "exact_range_hit": cover / count,
        "exact_range_match": exact_match / count,
        "answer_overlap_rate": answer_overlap_hits / count,
        "answer_cover_rate": answer_cover_hits / count,
        "breadth_ratio": answer_breadth_sum / count,
        "anchor_region_overlap_rate": anchor_overlap_hits / count,
        "anchor_region_cover_rate": anchor_cover_hits / count,
        "anchor_region_iou": anchor_iou_sum / count,
        "anchor_region_breadth_ratio": anchor_breadth_sum / count,
        "predicted_anchor_count": predicted_anchor_count_sum / count,
        "start_top3_accuracy": topk_accuracy(start_logits, start_labels, k=min(3, start_logits.shape[1])),
        "end_top3_accuracy": topk_accuracy(end_logits, end_labels, k=min(3, end_logits.shape[1])),
    }


def _apply_overlap_merge(
    logits: np.ndarray,
    label_index: dict[str, int],
    anchors_by_refmark: dict[str, object],
) -> np.ndarray:
    merged = logits.copy()
    inverse = {idx: refmark for refmark, idx in label_index.items()}
    neighbors: dict[int, list[int]] = {}
    for idx, refmark in inverse.items():
        anchor = anchors_by_refmark.get(refmark)
        if anchor is None or anchor.title is None:
            neighbors[idx] = [idx]
            continue
        linked = [idx]
        for other_idx, other_refmark in inverse.items():
            if other_idx == idx:
                continue
            other = anchors_by_refmark.get(other_refmark)
            if other is None or other.title != anchor.title:
                continue
            overlap = _span_overlap(
                int(anchor.region_start),
                int(anchor.region_end),
                int(other.region_start),
                int(other.region_end),
            )
            gap = min(
                abs(int(anchor.region_start) - int(other.region_end)),
                abs(int(other.region_start) - int(anchor.region_end)),
            )
            if overlap > 0 or gap <= 80:
                linked.append(other_idx)
        neighbors[idx] = sorted(set(linked))

    for idx, linked in neighbors.items():
        if len(linked) <= 1:
            continue
        merged[:, idx] = logits[:, linked].mean(axis=1)
    return merged


def format_report(results: dict, run_path: Path) -> str:
    lines = [
        "Refmark Train Report",
        "====================",
        "",
        f"Run artifact: {run_path}",
        "",
        "Dataset",
        f"  anchors:        {results['dataset']['anchor_count']}",
        f"  train examples: {results['dataset']['train_examples']}",
        f"  valid examples: {results['dataset']['valid_examples']}",
        f"  reformulated:   {results['dataset']['reformulated_examples']}",
        f"  vocab size:     {results['dataset']['vocab_size']}",
        f"  backend:        {results['dataset']['backend']}",
        "",
        "Baseline",
        f"  valid accuracy:        {results['baseline']['valid']['accuracy']:.3f}",
        f"  valid top-3 accuracy:  {results['baseline']['valid']['top3_accuracy']:.3f}",
        f"  valid overlap/cover:   {results['baseline']['valid'].get('answer_overlap_rate', 0.0):.3f} / {results['baseline']['valid'].get('answer_cover_rate', 0.0):.3f}",
        f"  valid broad/under:     {results['baseline']['valid'].get('breadth_ratio', 0.0):.2f}x / {results['baseline']['valid'].get('undercite_rate', 0.0):.3f}",
        f"  reform accuracy:       {results['baseline']['reformulated']['accuracy']:.3f}",
        f"  reform top-3 accuracy: {results['baseline']['reformulated']['top3_accuracy']:.3f}",
        f"  reform overlap/cover:  {results['baseline']['reformulated'].get('answer_overlap_rate', 0.0):.3f} / {results['baseline']['reformulated'].get('answer_cover_rate', 0.0):.3f}",
        f"  reform broad/under:    {results['baseline']['reformulated'].get('breadth_ratio', 0.0):.2f}x / {results['baseline']['reformulated'].get('undercite_rate', 0.0):.3f}",
        "",
        "Retrieval Baseline",
        f"  valid accuracy:        {results['retrieval_baseline']['valid']['accuracy']:.3f}",
        f"  valid top-3 accuracy:  {results['retrieval_baseline']['valid']['top3_accuracy']:.3f}",
        f"  valid overlap/cover:   {results['retrieval_baseline']['valid'].get('answer_overlap_rate', 0.0):.3f} / {results['retrieval_baseline']['valid'].get('answer_cover_rate', 0.0):.3f}",
        f"  valid broad/under:     {results['retrieval_baseline']['valid'].get('breadth_ratio', 0.0):.2f}x / {results['retrieval_baseline']['valid'].get('undercite_rate', 0.0):.3f}",
        f"  reform accuracy:       {results['retrieval_baseline']['reformulated']['accuracy']:.3f}",
        f"  reform top-3 accuracy: {results['retrieval_baseline']['reformulated']['top3_accuracy']:.3f}",
        f"  reform overlap/cover:  {results['retrieval_baseline']['reformulated'].get('answer_overlap_rate', 0.0):.3f} / {results['retrieval_baseline']['reformulated'].get('answer_cover_rate', 0.0):.3f}",
        f"  reform broad/under:    {results['retrieval_baseline']['reformulated'].get('breadth_ratio', 0.0):.2f}x / {results['retrieval_baseline']['reformulated'].get('undercite_rate', 0.0):.3f}",
        "",
        "Tiny Model",
        f"  train accuracy:        {results['tiny_model']['train']['accuracy']:.3f}",
        f"  valid accuracy:        {results['tiny_model']['valid']['accuracy']:.3f}",
        f"  valid top-3 accuracy:  {results['tiny_model']['valid']['top3_accuracy']:.3f}",
        f"  valid overlap/cover:   {results['tiny_model']['valid'].get('answer_overlap_rate', 0.0):.3f} / {results['tiny_model']['valid'].get('answer_cover_rate', 0.0):.3f}",
        f"  valid broad/under:     {results['tiny_model']['valid'].get('breadth_ratio', 0.0):.2f}x / {results['tiny_model']['valid'].get('undercite_rate', 0.0):.3f}",
        f"  reform accuracy:       {results['tiny_model']['reformulated']['accuracy']:.3f}",
        f"  reform top-3 accuracy: {results['tiny_model']['reformulated']['top3_accuracy']:.3f}",
        f"  reform overlap/cover:  {results['tiny_model']['reformulated'].get('answer_overlap_rate', 0.0):.3f} / {results['tiny_model']['reformulated'].get('answer_cover_rate', 0.0):.3f}",
        f"  reform broad/under:    {results['tiny_model']['reformulated'].get('breadth_ratio', 0.0):.2f}x / {results['tiny_model']['reformulated'].get('undercite_rate', 0.0):.3f}",
    ]
    bm25 = results.get("bm25_baseline")
    if bm25:
        retrieval_index = lines.index("Retrieval Baseline")
        lines[retrieval_index:retrieval_index] = [
            "BM25 Baseline",
            f"  valid accuracy:        {bm25['valid']['accuracy']:.3f}",
            f"  valid top-3 accuracy:  {bm25['valid']['top3_accuracy']:.3f}",
            f"  valid anchor ov/cov:   {bm25['valid'].get('anchor_region_overlap_rate', 0.0):.3f} / {bm25['valid'].get('anchor_region_cover_rate', 0.0):.3f}",
            f"  valid anchor broad:    {bm25['valid'].get('anchor_region_breadth_ratio', 0.0):.2f}x ({bm25['valid'].get('predicted_anchor_count', 0.0):.2f} anchors)",
            f"  valid overlap/cover:   {bm25['valid'].get('answer_overlap_rate', 0.0):.3f} / {bm25['valid'].get('answer_cover_rate', 0.0):.3f}",
            f"  valid broad/under:     {bm25['valid'].get('breadth_ratio', 0.0):.2f}x / {bm25['valid'].get('undercite_rate', 0.0):.3f}",
            f"  reform accuracy:       {bm25['reformulated']['accuracy']:.3f}",
            f"  reform top-3 accuracy: {bm25['reformulated']['top3_accuracy']:.3f}",
            f"  reform anchor ov/cov:  {bm25['reformulated'].get('anchor_region_overlap_rate', 0.0):.3f} / {bm25['reformulated'].get('anchor_region_cover_rate', 0.0):.3f}",
            f"  reform anchor broad:   {bm25['reformulated'].get('anchor_region_breadth_ratio', 0.0):.2f}x ({bm25['reformulated'].get('predicted_anchor_count', 0.0):.2f} anchors)",
            f"  reform overlap/cover:  {bm25['reformulated'].get('answer_overlap_rate', 0.0):.3f} / {bm25['reformulated'].get('answer_cover_rate', 0.0):.3f}",
            f"  reform broad/under:    {bm25['reformulated'].get('breadth_ratio', 0.0):.2f}x / {bm25['reformulated'].get('undercite_rate', 0.0):.3f}",
            "",
        ]
    stronger = results.get("stronger_direct_model")
    if stronger:
        lines.extend(
            [
                "",
                "Stronger Direct Model",
                f"  train accuracy:        {stronger['train']['accuracy']:.3f}",
                f"  valid accuracy:        {stronger['valid']['accuracy']:.3f}",
                f"  valid top-3 accuracy:  {stronger['valid']['top3_accuracy']:.3f}",
                f"  valid anchor ov/cov:   {stronger['valid'].get('anchor_region_overlap_rate', 0.0):.3f} / {stronger['valid'].get('anchor_region_cover_rate', 0.0):.3f}",
                f"  valid anchor broad:    {stronger['valid'].get('anchor_region_breadth_ratio', 0.0):.2f}x ({stronger['valid'].get('predicted_anchor_count', 0.0):.2f} anchors)",
                f"  valid overlap/cover:   {stronger['valid'].get('answer_overlap_rate', 0.0):.3f} / {stronger['valid'].get('answer_cover_rate', 0.0):.3f}",
                f"  valid broad/under:     {stronger['valid'].get('breadth_ratio', 0.0):.2f}x / {stronger['valid'].get('undercite_rate', 0.0):.3f}",
                f"  reform accuracy:       {stronger['reformulated']['accuracy']:.3f}",
                f"  reform top-3 accuracy: {stronger['reformulated']['top3_accuracy']:.3f}",
                f"  reform anchor ov/cov:  {stronger['reformulated'].get('anchor_region_overlap_rate', 0.0):.3f} / {stronger['reformulated'].get('anchor_region_cover_rate', 0.0):.3f}",
                f"  reform anchor broad:   {stronger['reformulated'].get('anchor_region_breadth_ratio', 0.0):.2f}x ({stronger['reformulated'].get('predicted_anchor_count', 0.0):.2f} anchors)",
                f"  reform overlap/cover:  {stronger['reformulated'].get('answer_overlap_rate', 0.0):.3f} / {stronger['reformulated'].get('answer_cover_rate', 0.0):.3f}",
                f"  reform broad/under:    {stronger['reformulated'].get('breadth_ratio', 0.0):.2f}x / {stronger['reformulated'].get('undercite_rate', 0.0):.3f}",
            ]
        )
    soft_target = results.get("soft_target_direct_model")
    if soft_target:
        lines.extend(
            [
                "",
                "Soft-Target Direct Model",
                f"  train accuracy:        {soft_target['train']['accuracy']:.3f}",
                f"  valid accuracy:        {soft_target['valid']['accuracy']:.3f}",
                f"  valid top-3 accuracy:  {soft_target['valid']['top3_accuracy']:.3f}",
                f"  valid overlap/cover:   {soft_target['valid'].get('answer_overlap_rate', 0.0):.3f} / {soft_target['valid'].get('answer_cover_rate', 0.0):.3f}",
                f"  valid broad/under:     {soft_target['valid'].get('breadth_ratio', 0.0):.2f}x / {soft_target['valid'].get('undercite_rate', 0.0):.3f}",
                f"  reform accuracy:       {soft_target['reformulated']['accuracy']:.3f}",
                f"  reform top-3 accuracy: {soft_target['reformulated']['top3_accuracy']:.3f}",
                f"  reform overlap/cover:  {soft_target['reformulated'].get('answer_overlap_rate', 0.0):.3f} / {soft_target['reformulated'].get('answer_cover_rate', 0.0):.3f}",
                f"  reform broad/under:    {soft_target['reformulated'].get('breadth_ratio', 0.0):.2f}x / {soft_target['reformulated'].get('undercite_rate', 0.0):.3f}",
            ]
        )
    hybrid = results.get("hybrid_direct_model")
    if hybrid:
        lines.extend(
            [
                "",
                "Hybrid Direct Model",
                f"  train accuracy:        {hybrid['train']['accuracy']:.3f}",
                f"  valid accuracy:        {hybrid['valid']['accuracy']:.3f}",
                f"  valid top-3 accuracy:  {hybrid['valid']['top3_accuracy']:.3f}",
                f"  valid anchor ov/cov:   {hybrid['valid'].get('anchor_region_overlap_rate', 0.0):.3f} / {hybrid['valid'].get('anchor_region_cover_rate', 0.0):.3f}",
                f"  valid anchor broad:    {hybrid['valid'].get('anchor_region_breadth_ratio', 0.0):.2f}x ({hybrid['valid'].get('predicted_anchor_count', 0.0):.2f} anchors)",
                f"  valid overlap/cover:   {hybrid['valid'].get('answer_overlap_rate', 0.0):.3f} / {hybrid['valid'].get('answer_cover_rate', 0.0):.3f}",
                f"  valid broad/under:     {hybrid['valid'].get('breadth_ratio', 0.0):.2f}x / {hybrid['valid'].get('undercite_rate', 0.0):.3f}",
                f"  reform accuracy:       {hybrid['reformulated']['accuracy']:.3f}",
                f"  reform top-3 accuracy: {hybrid['reformulated']['top3_accuracy']:.3f}",
                f"  reform anchor ov/cov:  {hybrid['reformulated'].get('anchor_region_overlap_rate', 0.0):.3f} / {hybrid['reformulated'].get('anchor_region_cover_rate', 0.0):.3f}",
                f"  reform anchor broad:   {hybrid['reformulated'].get('anchor_region_breadth_ratio', 0.0):.2f}x ({hybrid['reformulated'].get('predicted_anchor_count', 0.0):.2f} anchors)",
                f"  reform overlap/cover:  {hybrid['reformulated'].get('answer_overlap_rate', 0.0):.3f} / {hybrid['reformulated'].get('answer_cover_rate', 0.0):.3f}",
                f"  reform broad/under:    {hybrid['reformulated'].get('breadth_ratio', 0.0):.2f}x / {hybrid['reformulated'].get('undercite_rate', 0.0):.3f}",
            ]
        )
    lines.extend(
        [
            "",
            "Experimental Embedding Model",
            f"  train accuracy:        {results['experimental_embedding_model']['train']['accuracy']:.3f}",
            f"  valid accuracy:        {results['experimental_embedding_model']['valid']['accuracy']:.3f}",
            f"  valid top-3 accuracy:  {results['experimental_embedding_model']['valid']['top3_accuracy']:.3f}",
            f"  reform accuracy:       {results['experimental_embedding_model']['reformulated']['accuracy']:.3f}",
            f"  reform top-3 accuracy: {results['experimental_embedding_model']['reformulated']['top3_accuracy']:.3f}",
        ]
    )
    return "\n".join(lines)


def format_ensemble_report(results: dict, run_path: Path) -> str:
    model = results["ensemble_model"]
    lines = [
        "Refmark Train Ensemble Report",
        "=============================",
        "",
        f"Run artifact: {run_path}",
        "",
        f"Model kind:     {results['ensemble']['model_kind']}",
        f"Seeds:          {', '.join(str(seed) for seed in results['ensemble']['seeds'])}",
        f"Merge mode:     {results['ensemble']['merge_mode']}",
        f"Anchors:        {results['dataset']['anchor_count']}",
        f"Train examples: {results['dataset']['train_examples']}",
        f"Valid examples: {results['dataset']['valid_examples']}",
        f"Reformulated:   {results['dataset']['reformulated_examples']}",
        f"Backend:        {results['dataset']['backend']}",
        "",
        "Ensemble Model",
        f"  train accuracy:        {model['train']['accuracy']:.3f}",
        f"  valid accuracy:        {model['valid']['accuracy']:.3f}",
        f"  valid top-3 accuracy:  {model['valid']['top3_accuracy']:.3f}",
        f"  valid anchor ov/cov:   {model['valid'].get('anchor_region_overlap_rate', 0.0):.3f} / {model['valid'].get('anchor_region_cover_rate', 0.0):.3f}",
        f"  valid anchor broad:    {model['valid'].get('anchor_region_breadth_ratio', 0.0):.2f}x ({model['valid'].get('predicted_anchor_count', 0.0):.2f} anchors)",
        f"  valid overlap/cover:   {model['valid'].get('answer_overlap_rate', 0.0):.3f} / {model['valid'].get('answer_cover_rate', 0.0):.3f}",
        f"  valid broad/under:     {model['valid'].get('breadth_ratio', 0.0):.2f}x / {model['valid'].get('undercite_rate', 0.0):.3f}",
        f"  reform accuracy:       {model['reformulated']['accuracy']:.3f}",
        f"  reform top-3 accuracy: {model['reformulated']['top3_accuracy']:.3f}",
        f"  reform anchor ov/cov:  {model['reformulated'].get('anchor_region_overlap_rate', 0.0):.3f} / {model['reformulated'].get('anchor_region_cover_rate', 0.0):.3f}",
        f"  reform anchor broad:   {model['reformulated'].get('anchor_region_breadth_ratio', 0.0):.2f}x ({model['reformulated'].get('predicted_anchor_count', 0.0):.2f} anchors)",
        f"  reform overlap/cover:  {model['reformulated'].get('answer_overlap_rate', 0.0):.3f} / {model['reformulated'].get('answer_cover_rate', 0.0):.3f}",
        f"  reform broad/under:    {model['reformulated'].get('breadth_ratio', 0.0):.2f}x / {model['reformulated'].get('undercite_rate', 0.0):.3f}",
    ]
    return "\n".join(lines)


def format_range_report(results: dict, run_path: Path) -> str:
    model = results["range_model"]
    lines = [
        "Refmark Train Range Report",
        "==========================",
        "",
        f"Run artifact: {run_path}",
        "",
        f"Anchors:        {results['dataset']['anchor_count']}",
        f"Train examples: {results['dataset']['train_examples']}",
        f"Valid examples: {results['dataset']['valid_examples']}",
        f"Reformulated:   {results['dataset']['reformulated_examples']}",
        f"Backend:        {results['dataset']['backend']}",
        "",
        "Range Model",
        f"  valid exact-range-hit: {model['valid']['exact_range_hit']:.3f}",
        f"  valid exact match:     {model['valid'].get('exact_range_match', 0.0):.3f}",
        f"  valid anchor ov/cov:   {model['valid'].get('anchor_region_overlap_rate', 0.0):.3f} / {model['valid'].get('anchor_region_cover_rate', 0.0):.3f}",
        f"  valid anchor broad:    {model['valid'].get('anchor_region_breadth_ratio', 0.0):.2f}x ({model['valid'].get('predicted_anchor_count', 0.0):.2f} anchors)",
        f"  valid overlap/cover:   {model['valid']['answer_overlap_rate']:.3f} / {model['valid']['answer_cover_rate']:.3f}",
        f"  valid breadth:         {model['valid']['breadth_ratio']:.2f}x",
        f"  valid start/end top-3: {model['valid']['start_top3_accuracy']:.3f} / {model['valid']['end_top3_accuracy']:.3f}",
        f"  reform exact-range-hit:{model['reformulated']['exact_range_hit']:.3f}",
        f"  reform exact match:    {model['reformulated'].get('exact_range_match', 0.0):.3f}",
        f"  reform anchor ov/cov:  {model['reformulated'].get('anchor_region_overlap_rate', 0.0):.3f} / {model['reformulated'].get('anchor_region_cover_rate', 0.0):.3f}",
        f"  reform anchor broad:   {model['reformulated'].get('anchor_region_breadth_ratio', 0.0):.2f}x ({model['reformulated'].get('predicted_anchor_count', 0.0):.2f} anchors)",
        f"  reform overlap/cover:  {model['reformulated']['answer_overlap_rate']:.3f} / {model['reformulated']['answer_cover_rate']:.3f}",
        f"  reform breadth:        {model['reformulated']['breadth_ratio']:.2f}x",
        f"  reform start/end top-3:{model['reformulated']['start_top3_accuracy']:.3f} / {model['reformulated']['end_top3_accuracy']:.3f}",
    ]
    return "\n".join(lines)


def format_center_width_report(results: dict, run_path: Path) -> str:
    model = results["center_width_model"]
    lines = [
        "Refmark Train Center/Width Report",
        "=================================",
        "",
        f"Run artifact: {run_path}",
        "",
        f"Anchors:        {results['dataset']['anchor_count']}",
        f"Train examples: {results['dataset']['train_examples']}",
        f"Valid examples: {results['dataset']['valid_examples']}",
        f"Reformulated:   {results['dataset']['reformulated_examples']}",
        f"Backend:        {results['dataset']['backend']}",
        f"Width bins:     {', '.join(str(width) for width in model['width_bins'])}",
        "",
        "Center/Width Model",
        f"  train exact-range-hit: {model.get('train', {}).get('exact_range_hit', 0.0):.3f}",
        f"  valid exact-range-hit: {model['valid']['exact_range_hit']:.3f}",
        f"  valid exact match:     {model['valid'].get('exact_range_match', 0.0):.3f}",
        f"  valid anchor ov/cov:   {model['valid'].get('anchor_region_overlap_rate', 0.0):.3f} / {model['valid'].get('anchor_region_cover_rate', 0.0):.3f}",
        f"  valid anchor broad:    {model['valid'].get('anchor_region_breadth_ratio', 0.0):.2f}x ({model['valid'].get('predicted_anchor_count', 0.0):.2f} anchors)",
        f"  valid overlap/cover:   {model['valid']['answer_overlap_rate']:.3f} / {model['valid']['answer_cover_rate']:.3f}",
        f"  valid breadth:         {model['valid']['breadth_ratio']:.2f}x",
        f"  valid center/width t3: {model['valid']['center_top3_accuracy']:.3f} / {model['valid']['width_top3_accuracy']:.3f}",
        f"  valid relaxed 2-5x:    {_format_relaxed_density(model['valid'])}",
        f"  reform exact-range-hit:{model['reformulated']['exact_range_hit']:.3f}",
        f"  reform exact match:    {model['reformulated'].get('exact_range_match', 0.0):.3f}",
        f"  reform anchor ov/cov:  {model['reformulated'].get('anchor_region_overlap_rate', 0.0):.3f} / {model['reformulated'].get('anchor_region_cover_rate', 0.0):.3f}",
        f"  reform anchor broad:   {model['reformulated'].get('anchor_region_breadth_ratio', 0.0):.2f}x ({model['reformulated'].get('predicted_anchor_count', 0.0):.2f} anchors)",
        f"  reform overlap/cover:  {model['reformulated']['answer_overlap_rate']:.3f} / {model['reformulated']['answer_cover_rate']:.3f}",
        f"  reform breadth:        {model['reformulated']['breadth_ratio']:.2f}x",
        f"  reform center/width t3:{model['reformulated']['center_top3_accuracy']:.3f} / {model['reformulated']['width_top3_accuracy']:.3f}",
        f"  reform relaxed 2-5x:   {_format_relaxed_density(model['reformulated'])}",
    ]
    route = model.get("route")
    if route:
        lines.extend(
            [
                "",
                "Route Prior",
                f"  buckets:              {route['bucket_count']} x {route['bucket_size']} anchors",
                f"  top-k/margin:         {route['top_k']} / {route['margin']}",
                f"  outside penalty:      {route.get('outside_penalty', 0.0):.2f}",
                f"  valid/reform acc:     {route['valid_accuracy']:.3f} / {route['reformulated_accuracy']:.3f}",
            ]
        )
    for split_key, label in [("valid_by_style", "Valid By Style"), ("reformulated_by_style", "Reform By Style")]:
        if split_key not in model:
            continue
        lines.extend(["", label])
        for style, metrics in model[split_key].items():
            lines.append(
                f"  {style}: cover {metrics['exact_range_hit']:.3f}, exact {metrics.get('exact_range_match', 0.0):.3f}, breadth {metrics['breadth_ratio']:.2f}x, anchors {metrics['predicted_anchor_count']:.2f}"
            )
    return "\n".join(lines)


def _format_relaxed_density(metrics: dict) -> str:
    relaxed = metrics.get("relaxed_density", {})
    parts = []
    for factor in ("2x", "3x", "4x", "5x"):
        row = relaxed.get(factor, {})
        parts.append(f"{factor} ov {row.get('overlap_hit', 0.0):.3f}/ctr {row.get('center_hit', 0.0):.3f}")
    return "; ".join(parts)


def format_fusion_report(results: dict, run_path: Path) -> str:
    model = results["hybrid_fusion_model"]
    lines = [
        "Refmark Train Hybrid Fusion Report",
        "==================================",
        "",
        f"Run artifact: {run_path}",
        "",
        f"Anchors:        {results['dataset']['anchor_count']}",
        f"Train examples: {results['dataset']['train_examples']}",
        f"Valid examples: {results['dataset']['valid_examples']}",
        f"Reformulated:   {results['dataset']['reformulated_examples']}",
        f"Backend:        {results['dataset']['backend']}",
        f"Width bins:     {', '.join(str(width) for width in model['width_bins'])}",
        "",
        "Hybrid Fusion Model",
        f"  valid exact-range-hit: {model['valid']['exact_range_hit']:.3f}",
        f"  valid exact match:     {model['valid'].get('exact_range_match', 0.0):.3f}",
        f"  valid anchor ov/cov:   {model['valid'].get('anchor_region_overlap_rate', 0.0):.3f} / {model['valid'].get('anchor_region_cover_rate', 0.0):.3f}",
        f"  valid anchor broad:    {model['valid'].get('anchor_region_breadth_ratio', 0.0):.2f}x ({model['valid'].get('predicted_anchor_count', 0.0):.2f} anchors)",
        f"  valid overlap/cover:   {model['valid']['answer_overlap_rate']:.3f} / {model['valid']['answer_cover_rate']:.3f}",
        f"  valid breadth:         {model['valid']['breadth_ratio']:.2f}x",
        f"  valid anchor/center t3:{model['valid']['anchor_top3_accuracy']:.3f} / {model['valid']['center_top3_accuracy']:.3f}",
        f"  reform exact-range-hit:{model['reformulated']['exact_range_hit']:.3f}",
        f"  reform exact match:    {model['reformulated'].get('exact_range_match', 0.0):.3f}",
        f"  reform anchor ov/cov:  {model['reformulated'].get('anchor_region_overlap_rate', 0.0):.3f} / {model['reformulated'].get('anchor_region_cover_rate', 0.0):.3f}",
        f"  reform anchor broad:   {model['reformulated'].get('anchor_region_breadth_ratio', 0.0):.2f}x ({model['reformulated'].get('predicted_anchor_count', 0.0):.2f} anchors)",
        f"  reform overlap/cover:  {model['reformulated']['answer_overlap_rate']:.3f} / {model['reformulated']['answer_cover_rate']:.3f}",
        f"  reform breadth:        {model['reformulated']['breadth_ratio']:.2f}x",
        f"  reform anchor/center t3:{model['reformulated']['anchor_top3_accuracy']:.3f} / {model['reformulated']['center_top3_accuracy']:.3f}",
    ]
    return "\n".join(lines)


def format_refinement_loop_report(results: dict, run_path: Path) -> str:
    loop = results["refinement_loop"]
    lines = [
        "Refmark Train Refinement Loop Report",
        "====================================",
        "",
        f"Run artifact: {run_path}",
        "",
        f"Anchors:        {results['dataset']['anchor_count']}",
        f"Train examples: {results['dataset']['train_examples']}",
        f"Valid examples: {results['dataset']['valid_examples']}",
        f"Reformulated:   {results['dataset']['reformulated_examples']}",
        f"Backend:        {results['dataset']['backend']}",
        f"Width bins:     {', '.join(str(width) for width in loop['width_bins'])}",
        "",
    ]
    for split_key, label in [("valid", "Valid"), ("reformulated", "Reformulated")]:
        metrics = loop[split_key]
        broad = metrics["broad_stage"]
        gold = metrics["gold_broad_prompt_precise_stage"]
        predicted = metrics["predicted_broad_loop_precise_stage"]
        lines.extend(
            [
                label,
                f"  pairs:                  {metrics['pairs']}",
                f"  broad cover/breadth:    {broad['exact_range_hit']:.3f} / {broad['breadth_ratio']:.2f}x",
                f"  gold refine cover/br:   {gold['exact_range_hit']:.3f} / {gold['breadth_ratio']:.2f}x",
                f"  loop refine cover/br:   {predicted['exact_range_hit']:.3f} / {predicted['breadth_ratio']:.2f}x",
                f"  loop refine exact:      {predicted.get('exact_range_match', 0.0):.3f}",
                f"  broad -> precise count: {metrics['average_predicted_broad_anchor_count']:.2f} -> {metrics['average_loop_precise_anchor_count']:.2f} anchors",
            ]
        )
        if "learned_reranked_broad_loop_precise_stage" in metrics:
            reranked = metrics["learned_reranked_broad_loop_precise_stage"]
            lines.extend(
                [
                    f"  learned rerank cov/br:{reranked['exact_range_hit']:.3f} / {reranked['breadth_ratio']:.2f}x",
                    f"  learned rerank exact:  {reranked.get('exact_range_match', 0.0):.3f}",
                    f"  learned broad -> prec: {metrics.get('average_learned_reranked_broad_anchor_count', 0.0):.2f} -> {metrics.get('average_learned_reranked_loop_precise_anchor_count', 0.0):.2f} anchors",
                ]
            )
        if "local_cross_broad_loop_precise_stage" in metrics:
            local_cross = metrics["local_cross_broad_loop_precise_stage"]
            lines.extend(
                [
                    f"  local xenc cov/br:    {local_cross['exact_range_hit']:.3f} / {local_cross['breadth_ratio']:.2f}x",
                    f"  local xenc exact:     {local_cross.get('exact_range_match', 0.0):.3f}",
                    f"  local xenc broad->pr: {metrics.get('average_local_cross_broad_anchor_count', 0.0):.2f} -> {metrics.get('average_local_cross_loop_precise_anchor_count', 0.0):.2f} anchors",
                ]
            )
        if "hybrid_local_cross_broad_loop_precise_stage" in metrics:
            hybrid = metrics["hybrid_local_cross_broad_loop_precise_stage"]
            policy = metrics.get("local_cross_policy", {"local_weight": 1.0, "uncertainty_threshold": 0.0})
            lines.extend(
                [
                    f"  hybrid xenc cov/br:   {hybrid['exact_range_hit']:.3f} / {hybrid['breadth_ratio']:.2f}x",
                    f"  hybrid xenc exact:    {hybrid.get('exact_range_match', 0.0):.3f}",
                    f"  hybrid xenc broad->pr:{metrics.get('average_hybrid_local_cross_broad_anchor_count', 0.0):.2f} -> {metrics.get('average_hybrid_local_cross_loop_precise_anchor_count', 0.0):.2f} anchors",
                    f"  hybrid policy:        local={policy.get('local_weight', 1.0):.2f}, uncertainty>={policy.get('uncertainty_threshold', 0.0):.3f}",
                ]
            )
        lines.append("")
    return "\n".join(lines)
