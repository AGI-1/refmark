from __future__ import annotations

import argparse
from pathlib import Path

from refmark_train.diagnostics import Bm25DiagnosticsConfig, format_bm25_diagnostics, run_bm25_diagnostics
from refmark_train.experiment import (
    DATA_DIR,
    ExperimentConfig,
    format_center_width_report,
    format_ensemble_report,
    format_fusion_report,
    format_range_report,
    format_refinement_loop_report,
    format_report,
    prepare_dataset,
    run_center_width_ensemble_saved_experiment,
    run_center_width_saved_experiment,
    run_ensemble_saved_experiment,
    run_experiment,
    run_hybrid_fusion_saved_experiment,
    run_range_ensemble_saved_experiment,
    run_range_saved_experiment,
    run_refinement_loop_saved_experiment,
    run_refinement_two_model_saved_experiment,
    run_saved_experiment,
)
from refmark_train.llm_qa import LlmQaConfig, generate_llm_qa_dataset
from refmark_train.range_data import prepare_contiguous_range_dataset, prepare_refinement_dataset
from refmark_train.real_corpus import prepare_squad_dataset
from refmark_train.single_doc import prepare_single_doc_dataset
from refmark_train.synthetic import build_corpus, preview_lines


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tiny refmark training sandbox")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preview = subparsers.add_parser("preview", help="preview generated anchors and examples")
    preview.add_argument("--anchors", type=int, default=8)
    preview.add_argument("--seed", type=int, default=13)

    prepare = subparsers.add_parser("prepare", help="generate and save a dataset under refmark_train/data")
    prepare.add_argument("--anchors", type=int, default=100)
    prepare.add_argument("--seed", type=int, default=13)

    prepare_squad = subparsers.add_parser("prepare-squad", help="download SQuAD and save an anchored dataset")
    prepare_squad.add_argument("--train-limit", type=int, default=3000)
    prepare_squad.add_argument("--dev-limit", type=int, default=600)
    prepare_squad.add_argument("--min-questions-per-anchor", type=int, default=3)
    prepare_squad.add_argument("--anchor-limit", type=int)
    prepare_squad.add_argument("--sentence-window", type=int, default=0)
    prepare_squad.add_argument("--rank-mode", choices=["frequency", "confusable"], default="frequency")
    prepare_squad.add_argument("--hard-reformulated", action="store_true")
    prepare_squad.add_argument("--name", default="squad_v1")

    prepare_doc = subparsers.add_parser("prepare-doc", help="create a single-document anchored dataset from a local text file")
    prepare_doc.add_argument("--input-path", required=True)
    prepare_doc.add_argument("--name", default="single_doc")
    prepare_doc.add_argument("--unit", choices=["paragraph", "sentence", "structure"], default="paragraph")
    prepare_doc.add_argument("--paragraph-window", type=int, default=1)
    prepare_doc.add_argument("--stride", type=int, default=1)
    prepare_doc.add_argument("--anchor-limit", type=int, default=500)
    prepare_doc.add_argument("--seed", type=int, default=13)
    prepare_doc.add_argument("--question-mode", choices=["legacy", "randomized", "contextual"], default="legacy")
    prepare_doc.add_argument("--train-mutations", type=int, default=0)
    prepare_doc.add_argument("--train-questions-per-phrase", type=int)
    prepare_doc.add_argument("--valid-questions-per-phrase", type=int)
    prepare_doc.add_argument("--reform-questions-per-phrase", type=int)

    generate_llm_qa = subparsers.add_parser("generate-llm-qa", help="generate an LLM-authored QA dataset over prepared anchors")
    generate_llm_qa.add_argument("--data-dir", required=True)
    generate_llm_qa.add_argument("--output-name", required=True)
    generate_llm_qa.add_argument("--model", default="google/gemma-4-31b-it")
    generate_llm_qa.add_argument("--reviewer-model")
    generate_llm_qa.add_argument("--endpoint", default="https://openrouter.ai/api/v1/chat/completions")
    generate_llm_qa.add_argument("--max-anchors", type=int, default=24)
    generate_llm_qa.add_argument("--train-per-anchor", type=int, default=3)
    generate_llm_qa.add_argument("--valid-per-anchor", type=int, default=1)
    generate_llm_qa.add_argument("--reform-per-anchor", type=int, default=2)
    generate_llm_qa.add_argument("--seed", type=int, default=13)
    generate_llm_qa.add_argument("--temperature", type=float, default=0.4)
    generate_llm_qa.add_argument("--max-tokens", type=int, default=900)
    generate_llm_qa.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    generate_llm_qa.add_argument("--resume", action="store_true", help="reuse completed anchors from an existing raw generation file")
    generate_llm_qa.add_argument("--question-style", choices=["natural", "hard"], default="natural")

    bm25_diag = subparsers.add_parser("bm25-diagnostics", help="inspect BM25 candidate, window, and distributed-anchor behavior")
    bm25_diag.add_argument("--data-dir", required=True)
    bm25_diag.add_argument("--output-path")
    bm25_diag.add_argument("--top-ks", default="1,3,5,10")
    bm25_diag.add_argument("--window-widths", default="1,3,5,9")
    bm25_diag.add_argument("--neighborhood-top-k", type=int, default=3)
    bm25_diag.add_argument("--neighborhood-margins", default="0,1,2,4")
    bm25_diag.add_argument("--distributed-top-k", type=int, default=3)
    bm25_diag.add_argument("--distributed-min-gap", type=int, default=10)
    bm25_diag.add_argument("--distributed-pairs", type=int, default=20)

    prepare_ranges = subparsers.add_parser("prepare-ranges", help="create contiguous multi-anchor gold range examples from a prepared dataset")
    prepare_ranges.add_argument("--data-dir", required=True)
    prepare_ranges.add_argument("--output-name", required=True)
    prepare_ranges.add_argument("--max-ranges", type=int, default=120)
    prepare_ranges.add_argument("--min-width", type=int, default=2)
    prepare_ranges.add_argument("--max-width", type=int, default=4)
    prepare_ranges.add_argument("--seed", type=int, default=13)
    prepare_ranges.add_argument("--train-per-range", type=int, default=3)
    prepare_ranges.add_argument("--valid-per-range", type=int, default=1)
    prepare_ranges.add_argument("--reform-per-range", type=int, default=2)
    prepare_ranges.add_argument("--include-single-examples", type=int, default=0)

    prepare_refine = subparsers.add_parser("prepare-refinement", help="create broad-then-narrow contiguous range refinement examples")
    prepare_refine.add_argument("--data-dir", required=True)
    prepare_refine.add_argument("--output-name", required=True)
    prepare_refine.add_argument("--max-examples", type=int, default=160)
    prepare_refine.add_argument("--precise-min-width", type=int, default=1)
    prepare_refine.add_argument("--precise-max-width", type=int, default=2)
    prepare_refine.add_argument("--broad-margin-min", type=int, default=1)
    prepare_refine.add_argument("--broad-margin-max", type=int, default=3)
    prepare_refine.add_argument("--seed", type=int, default=13)

    run = subparsers.add_parser("run", help="run the end-to-end experiment")
    run.add_argument("--anchors", type=int, default=100)
    run.add_argument("--seed", type=int, default=13)
    run.add_argument("--epochs", type=int, default=60)
    run.add_argument("--backend", choices=["auto", "cpu", "cuda", "directml", "torchcpu"], default="auto")
    run.add_argument("--embedding-dim", type=int, default=24)
    run.add_argument("--hidden-dim", type=int, default=48)
    run.add_argument("--mlp-hidden-dim", type=int, default=512)
    run.add_argument("--mlp-hidden2-dim", type=int, default=0)
    run.add_argument("--mlp-dropout", type=float, default=0.2)
    run.add_argument("--hybrid-exact-weight", type=float, default=0.7)
    run.add_argument("--soft-cover-reward", type=float, default=0.92)
    run.add_argument("--soft-overlap-base", type=float, default=0.35)
    run.add_argument("--soft-overlap-f1-weight", type=float, default=0.45)
    run.add_argument("--soft-overlap-precision-weight", type=float, default=0.2)
    run.add_argument("--soft-cover-breadth-penalty", type=float, default=8.0)
    run.add_argument("--soft-overlap-breadth-penalty", type=float, default=10.0)
    run.add_argument("--soft-cover-penalty-cap", type=float, default=0.35)
    run.add_argument("--soft-overlap-penalty-cap", type=float, default=0.25)
    run.add_argument("--soft-local-reward", type=float, default=0.08)
    run.add_argument("--soft-local-distance", type=int, default=40)
    run.add_argument("--learning-rate", type=float, default=0.25)
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--weight-decay", type=float, default=1e-4)

    run_saved = subparsers.add_parser("run-saved", help="train/evaluate using a prepared dataset directory")
    run_saved.add_argument("--data-dir", required=True)
    run_saved.add_argument("--seed", type=int, default=13)
    run_saved.add_argument("--epochs", type=int, default=60)
    run_saved.add_argument("--backend", choices=["auto", "cpu", "cuda", "directml", "torchcpu"], default="auto")
    run_saved.add_argument("--embedding-dim", type=int, default=24)
    run_saved.add_argument("--hidden-dim", type=int, default=48)
    run_saved.add_argument("--mlp-hidden-dim", type=int, default=512)
    run_saved.add_argument("--mlp-hidden2-dim", type=int, default=0)
    run_saved.add_argument("--mlp-dropout", type=float, default=0.2)
    run_saved.add_argument("--hybrid-exact-weight", type=float, default=0.7)
    run_saved.add_argument("--soft-cover-reward", type=float, default=0.92)
    run_saved.add_argument("--soft-overlap-base", type=float, default=0.35)
    run_saved.add_argument("--soft-overlap-f1-weight", type=float, default=0.45)
    run_saved.add_argument("--soft-overlap-precision-weight", type=float, default=0.2)
    run_saved.add_argument("--soft-cover-breadth-penalty", type=float, default=8.0)
    run_saved.add_argument("--soft-overlap-breadth-penalty", type=float, default=10.0)
    run_saved.add_argument("--soft-cover-penalty-cap", type=float, default=0.35)
    run_saved.add_argument("--soft-overlap-penalty-cap", type=float, default=0.25)
    run_saved.add_argument("--soft-local-reward", type=float, default=0.08)
    run_saved.add_argument("--soft-local-distance", type=int, default=40)
    run_saved.add_argument("--learning-rate", type=float, default=0.25)
    run_saved.add_argument("--batch-size", type=int, default=32)
    run_saved.add_argument("--weight-decay", type=float, default=1e-4)

    ensemble_saved = subparsers.add_parser("ensemble-saved", help="train multiple seeds and merge logits on a prepared dataset")
    ensemble_saved.add_argument("--data-dir", required=True)
    ensemble_saved.add_argument("--seeds", default="13,21,34")
    ensemble_saved.add_argument("--model-kind", choices=["stronger", "hybrid"], default="hybrid")
    ensemble_saved.add_argument("--merge-mode", choices=["logit", "overlap"], default="logit")
    ensemble_saved.add_argument("--epochs", type=int, default=25)
    ensemble_saved.add_argument("--backend", choices=["auto", "cpu", "cuda", "directml", "torchcpu"], default="torchcpu")
    ensemble_saved.add_argument("--mlp-hidden-dim", type=int, default=1024)
    ensemble_saved.add_argument("--mlp-hidden2-dim", type=int, default=0)
    ensemble_saved.add_argument("--mlp-dropout", type=float, default=0.1)
    ensemble_saved.add_argument("--hybrid-exact-weight", type=float, default=0.7)
    ensemble_saved.add_argument("--soft-cover-reward", type=float, default=0.92)
    ensemble_saved.add_argument("--soft-overlap-base", type=float, default=0.35)
    ensemble_saved.add_argument("--soft-overlap-f1-weight", type=float, default=0.45)
    ensemble_saved.add_argument("--soft-overlap-precision-weight", type=float, default=0.2)
    ensemble_saved.add_argument("--soft-cover-breadth-penalty", type=float, default=8.0)
    ensemble_saved.add_argument("--soft-overlap-breadth-penalty", type=float, default=10.0)
    ensemble_saved.add_argument("--soft-cover-penalty-cap", type=float, default=0.35)
    ensemble_saved.add_argument("--soft-overlap-penalty-cap", type=float, default=0.25)
    ensemble_saved.add_argument("--soft-local-reward", type=float, default=0.08)
    ensemble_saved.add_argument("--soft-local-distance", type=int, default=40)
    ensemble_saved.add_argument("--learning-rate", type=float, default=0.25)
    ensemble_saved.add_argument("--batch-size", type=int, default=32)
    ensemble_saved.add_argument("--weight-decay", type=float, default=1e-4)

    range_saved = subparsers.add_parser("range-saved", help="train a lightweight anchor-index range predictor on a prepared dataset")
    range_saved.add_argument("--data-dir", required=True)
    range_saved.add_argument("--seed", type=int, default=13)
    range_saved.add_argument("--epochs", type=int, default=25)
    range_saved.add_argument("--backend", choices=["auto", "cpu", "cuda", "directml", "torchcpu"], default="torchcpu")
    range_saved.add_argument("--mlp-hidden-dim", type=int, default=1024)
    range_saved.add_argument("--mlp-hidden2-dim", type=int, default=0)
    range_saved.add_argument("--mlp-dropout", type=float, default=0.1)
    range_saved.add_argument("--range-exact-weight", type=float, default=0.8)
    range_saved.add_argument("--range-neighbor-sigma", type=float, default=1.5)
    range_saved.add_argument("--range-breadth-penalty", type=float, default=0.15)
    range_saved.add_argument("--range-soft-weight", type=float, default=0.2)
    range_saved.add_argument("--range-width-weight", type=float, default=0.1)
    range_saved.add_argument("--learning-rate", type=float, default=0.25)
    range_saved.add_argument("--batch-size", type=int, default=32)
    range_saved.add_argument("--weight-decay", type=float, default=1e-4)

    range_ensemble = subparsers.add_parser("range-ensemble-saved", help="train multiple range models and average start/end logits")
    range_ensemble.add_argument("--data-dir", required=True)
    range_ensemble.add_argument("--seeds", default="13,21,34")
    range_ensemble.add_argument("--epochs", type=int, default=25)
    range_ensemble.add_argument("--backend", choices=["auto", "cpu", "cuda", "directml", "torchcpu"], default="torchcpu")
    range_ensemble.add_argument("--mlp-hidden-dim", type=int, default=1024)
    range_ensemble.add_argument("--mlp-hidden2-dim", type=int, default=0)
    range_ensemble.add_argument("--mlp-dropout", type=float, default=0.1)
    range_ensemble.add_argument("--range-exact-weight", type=float, default=0.8)
    range_ensemble.add_argument("--range-neighbor-sigma", type=float, default=1.5)
    range_ensemble.add_argument("--range-breadth-penalty", type=float, default=0.15)
    range_ensemble.add_argument("--range-soft-weight", type=float, default=0.2)
    range_ensemble.add_argument("--range-width-weight", type=float, default=0.1)
    range_ensemble.add_argument("--learning-rate", type=float, default=0.25)
    range_ensemble.add_argument("--batch-size", type=int, default=32)
    range_ensemble.add_argument("--weight-decay", type=float, default=1e-4)

    center_width = subparsers.add_parser("center-width-saved", help="train a center/width anchor-index predictor on a prepared dataset")
    center_width.add_argument("--data-dir", required=True)
    center_width.add_argument("--seed", type=int, default=13)
    center_width.add_argument("--epochs", type=int, default=25)
    center_width.add_argument("--backend", choices=["auto", "cpu", "cuda", "directml", "torchcpu"], default="torchcpu")
    center_width.add_argument("--mlp-hidden-dim", type=int, default=1024)
    center_width.add_argument("--mlp-hidden2-dim", type=int, default=0)
    center_width.add_argument("--mlp-dropout", type=float, default=0.1)
    center_width.add_argument("--center-exact-weight", type=float, default=0.75)
    center_width.add_argument("--center-soft-weight", type=float, default=0.2)
    center_width.add_argument("--center-width-weight", type=float, default=0.05)
    center_width.add_argument("--range-neighbor-sigma", type=float, default=1.5)
    center_width.add_argument("--range-breadth-penalty", type=float, default=0.15)
    center_width.add_argument("--bm25-weight", type=float, default=0.35)
    center_width.add_argument("--bm25-top-k", type=int, default=8)
    center_width.add_argument("--bm25-margin", type=int, default=2)
    center_width.add_argument("--bm25-outside-penalty", type=float, default=1.5)
    center_width.add_argument("--bm25-enrich-train", action="store_true")
    center_width.add_argument("--route-bucket-size", type=int, default=0)
    center_width.add_argument("--route-top-k", type=int, default=1)
    center_width.add_argument("--route-margin", type=int, default=0)
    center_width.add_argument("--route-outside-penalty", type=float, default=1.2)
    center_width.add_argument("--learning-rate", type=float, default=0.25)
    center_width.add_argument("--batch-size", type=int, default=32)
    center_width.add_argument("--weight-decay", type=float, default=1e-4)

    center_width_ensemble = subparsers.add_parser("center-width-ensemble-saved", help="train multiple center/width models and average logits")
    center_width_ensemble.add_argument("--data-dir", required=True)
    center_width_ensemble.add_argument("--seeds", default="13,21,34")
    center_width_ensemble.add_argument("--epochs", type=int, default=25)
    center_width_ensemble.add_argument("--backend", choices=["auto", "cpu", "cuda", "directml", "torchcpu"], default="torchcpu")
    center_width_ensemble.add_argument("--mlp-hidden-dim", type=int, default=1024)
    center_width_ensemble.add_argument("--mlp-hidden2-dim", type=int, default=0)
    center_width_ensemble.add_argument("--mlp-dropout", type=float, default=0.1)
    center_width_ensemble.add_argument("--center-exact-weight", type=float, default=0.75)
    center_width_ensemble.add_argument("--center-soft-weight", type=float, default=0.2)
    center_width_ensemble.add_argument("--center-width-weight", type=float, default=0.05)
    center_width_ensemble.add_argument("--range-neighbor-sigma", type=float, default=1.5)
    center_width_ensemble.add_argument("--range-breadth-penalty", type=float, default=0.15)
    center_width_ensemble.add_argument("--bm25-weight", type=float, default=0.35)
    center_width_ensemble.add_argument("--bm25-top-k", type=int, default=8)
    center_width_ensemble.add_argument("--bm25-margin", type=int, default=2)
    center_width_ensemble.add_argument("--bm25-outside-penalty", type=float, default=1.5)
    center_width_ensemble.add_argument("--bm25-enrich-train", action="store_true")
    center_width_ensemble.add_argument("--route-bucket-size", type=int, default=0)
    center_width_ensemble.add_argument("--route-top-k", type=int, default=1)
    center_width_ensemble.add_argument("--route-margin", type=int, default=0)
    center_width_ensemble.add_argument("--route-outside-penalty", type=float, default=1.2)
    center_width_ensemble.add_argument("--learning-rate", type=float, default=0.25)
    center_width_ensemble.add_argument("--batch-size", type=int, default=32)
    center_width_ensemble.add_argument("--weight-decay", type=float, default=1e-4)

    refinement_loop = subparsers.add_parser("refinement-loop-saved", help="train center/width models and evaluate predicted-broad refinement loop")
    refinement_loop.add_argument("--data-dir", required=True)
    refinement_loop.add_argument("--seeds", default="13,21,34")
    refinement_loop.add_argument("--epochs", type=int, default=25)
    refinement_loop.add_argument("--backend", choices=["auto", "cpu", "cuda", "directml", "torchcpu"], default="torchcpu")
    refinement_loop.add_argument("--mlp-hidden-dim", type=int, default=1024)
    refinement_loop.add_argument("--mlp-hidden2-dim", type=int, default=0)
    refinement_loop.add_argument("--mlp-dropout", type=float, default=0.1)
    refinement_loop.add_argument("--center-exact-weight", type=float, default=0.75)
    refinement_loop.add_argument("--center-soft-weight", type=float, default=0.2)
    refinement_loop.add_argument("--center-width-weight", type=float, default=0.05)
    refinement_loop.add_argument("--range-neighbor-sigma", type=float, default=1.5)
    refinement_loop.add_argument("--range-breadth-penalty", type=float, default=0.15)
    refinement_loop.add_argument("--learning-rate", type=float, default=0.25)
    refinement_loop.add_argument("--batch-size", type=int, default=32)
    refinement_loop.add_argument("--weight-decay", type=float, default=1e-4)

    refinement_two_model = subparsers.add_parser("refinement-two-model-saved", help="train separate broad and refine center/width ensembles and evaluate predicted-broad loop")
    refinement_two_model.add_argument("--data-dir", required=True)
    refinement_two_model.add_argument("--seeds", default="13,21,34")
    refinement_two_model.add_argument("--epochs", type=int, default=25)
    refinement_two_model.add_argument("--backend", choices=["auto", "cpu", "cuda", "directml", "torchcpu"], default="torchcpu")
    refinement_two_model.add_argument("--mlp-hidden-dim", type=int, default=1024)
    refinement_two_model.add_argument("--mlp-hidden2-dim", type=int, default=0)
    refinement_two_model.add_argument("--mlp-dropout", type=float, default=0.1)
    refinement_two_model.add_argument("--center-exact-weight", type=float, default=0.75)
    refinement_two_model.add_argument("--center-soft-weight", type=float, default=0.2)
    refinement_two_model.add_argument("--center-width-weight", type=float, default=0.05)
    refinement_two_model.add_argument("--range-neighbor-sigma", type=float, default=1.5)
    refinement_two_model.add_argument("--range-breadth-penalty", type=float, default=0.15)
    refinement_two_model.add_argument("--local-cross-hidden-dim", type=int, default=128)
    refinement_two_model.add_argument("--local-cross-epochs", type=int, default=120)
    refinement_two_model.add_argument("--local-cross-dropout", type=float, default=0.1)
    refinement_two_model.add_argument("--local-cross-blend-steps", type=int, default=5)
    refinement_two_model.add_argument("--local-cross-max-candidates", type=int, default=24)
    refinement_two_model.add_argument("--learning-rate", type=float, default=0.25)
    refinement_two_model.add_argument("--batch-size", type=int, default=32)
    refinement_two_model.add_argument("--weight-decay", type=float, default=1e-4)

    hybrid_fusion = subparsers.add_parser("hybrid-fusion-saved", help="fuse hybrid anchor logits with center/width range logits")
    hybrid_fusion.add_argument("--data-dir", required=True)
    hybrid_fusion.add_argument("--seeds", default="13,21,34")
    hybrid_fusion.add_argument("--epochs", type=int, default=25)
    hybrid_fusion.add_argument("--backend", choices=["auto", "cpu", "cuda", "directml", "torchcpu"], default="torchcpu")
    hybrid_fusion.add_argument("--mlp-hidden-dim", type=int, default=1024)
    hybrid_fusion.add_argument("--mlp-hidden2-dim", type=int, default=0)
    hybrid_fusion.add_argument("--mlp-dropout", type=float, default=0.1)
    hybrid_fusion.add_argument("--hybrid-exact-weight", type=float, default=0.7)
    hybrid_fusion.add_argument("--soft-cover-reward", type=float, default=0.92)
    hybrid_fusion.add_argument("--soft-overlap-base", type=float, default=0.35)
    hybrid_fusion.add_argument("--soft-overlap-f1-weight", type=float, default=0.45)
    hybrid_fusion.add_argument("--soft-overlap-precision-weight", type=float, default=0.2)
    hybrid_fusion.add_argument("--soft-cover-breadth-penalty", type=float, default=8.0)
    hybrid_fusion.add_argument("--soft-overlap-breadth-penalty", type=float, default=10.0)
    hybrid_fusion.add_argument("--soft-cover-penalty-cap", type=float, default=0.35)
    hybrid_fusion.add_argument("--soft-overlap-penalty-cap", type=float, default=0.25)
    hybrid_fusion.add_argument("--soft-local-reward", type=float, default=0.08)
    hybrid_fusion.add_argument("--soft-local-distance", type=int, default=40)
    hybrid_fusion.add_argument("--center-exact-weight", type=float, default=0.75)
    hybrid_fusion.add_argument("--center-soft-weight", type=float, default=0.2)
    hybrid_fusion.add_argument("--center-width-weight", type=float, default=0.05)
    hybrid_fusion.add_argument("--range-neighbor-sigma", type=float, default=1.5)
    hybrid_fusion.add_argument("--range-breadth-penalty", type=float, default=0.15)
    hybrid_fusion.add_argument("--bm25-weight", type=float, default=0.35)
    hybrid_fusion.add_argument("--bm25-top-k", type=int, default=8)
    hybrid_fusion.add_argument("--bm25-margin", type=int, default=2)
    hybrid_fusion.add_argument("--bm25-outside-penalty", type=float, default=1.5)
    hybrid_fusion.add_argument("--bm25-enrich-train", action="store_true")
    hybrid_fusion.add_argument("--fusion-anchor-weight", type=float, default=0.6)
    hybrid_fusion.add_argument("--fusion-range-weight", type=float, default=1.0)
    hybrid_fusion.add_argument("--fusion-breadth-penalty", type=float, default=0.08)
    hybrid_fusion.add_argument("--learning-rate", type=float, default=0.25)
    hybrid_fusion.add_argument("--batch-size", type=int, default=32)
    hybrid_fusion.add_argument("--weight-decay", type=float, default=1e-4)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "preview":
        bundle = build_corpus(anchor_count=args.anchors, seed=args.seed)
        print("\n".join(preview_lines(bundle)))
        return 0

    if args.command == "prepare":
        files, output_dir = prepare_dataset(anchor_count=args.anchors, seed=args.seed)
        print(f"Prepared dataset in {output_dir}")
        for name, path in files.items():
            print(f"{name}: {path}")
        return 0

    if args.command == "prepare-squad":
        output_dir = DATA_DIR / args.name
        files = prepare_squad_dataset(
            output_dir=output_dir,
            train_limit=args.train_limit,
            dev_limit=args.dev_limit,
            min_questions_per_anchor=args.min_questions_per_anchor,
            anchor_limit=args.anchor_limit,
            sentence_window=args.sentence_window,
            rank_mode=args.rank_mode,
            hard_reformulated=args.hard_reformulated,
        )
        print(f"Prepared SQuAD dataset in {output_dir}")
        for name, path in files.items():
            print(f"{name}: {path}")
        return 0

    if args.command == "prepare-doc":
        output_dir = DATA_DIR / args.name
        files = prepare_single_doc_dataset(
            input_path=Path(args.input_path),
            output_dir=output_dir,
            unit=args.unit,
            paragraph_window=args.paragraph_window,
            stride=args.stride,
            anchor_limit=args.anchor_limit,
            seed=args.seed,
            question_mode=args.question_mode,
            train_mutations=args.train_mutations,
            train_questions_per_phrase=args.train_questions_per_phrase,
            valid_questions_per_phrase=args.valid_questions_per_phrase,
            reform_questions_per_phrase=args.reform_questions_per_phrase,
        )
        print(f"Prepared single-document dataset in {output_dir}")
        for name, path in files.items():
            print(f"{name}: {path}")
        return 0

    if args.command == "generate-llm-qa":
        output_dir = DATA_DIR / args.output_name
        files, output_dir = generate_llm_qa_dataset(
            LlmQaConfig(
                data_dir=Path(args.data_dir),
                output_dir=output_dir,
                model=args.model,
                reviewer_model=args.reviewer_model,
                endpoint=args.endpoint,
                max_anchors=args.max_anchors,
                train_per_anchor=args.train_per_anchor,
                valid_per_anchor=args.valid_per_anchor,
                reform_per_anchor=args.reform_per_anchor,
                seed=args.seed,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                api_key_env=args.api_key_env,
                resume=args.resume,
                question_style=args.question_style,
            )
        )
        print(f"Prepared LLM QA dataset in {output_dir}")
        for name, path in files.items():
            print(f"{name}: {path}")
        return 0

    if args.command == "bm25-diagnostics":
        report, output_path = run_bm25_diagnostics(
            Bm25DiagnosticsConfig(
                data_dir=Path(args.data_dir),
                output_path=Path(args.output_path) if args.output_path else None,
                top_ks=_parse_int_tuple(args.top_ks),
                window_widths=_parse_int_tuple(args.window_widths),
                neighborhood_top_k=args.neighborhood_top_k,
                neighborhood_margins=_parse_int_tuple(args.neighborhood_margins),
                distributed_top_k=args.distributed_top_k,
                distributed_min_gap=args.distributed_min_gap,
                distributed_pairs=args.distributed_pairs,
            )
        )
        print(format_bm25_diagnostics(report, output_path))
        return 0

    if args.command == "prepare-ranges":
        output_dir = DATA_DIR / args.output_name
        files = prepare_contiguous_range_dataset(
            data_dir=Path(args.data_dir),
            output_dir=output_dir,
            max_ranges=args.max_ranges,
            min_width=args.min_width,
            max_width=args.max_width,
            seed=args.seed,
            train_per_range=args.train_per_range,
            valid_per_range=args.valid_per_range,
            reform_per_range=args.reform_per_range,
            include_single_examples=args.include_single_examples,
        )
        print(f"Prepared contiguous range dataset in {output_dir}")
        for name, path in files.items():
            print(f"{name}: {path}")
        return 0

    if args.command == "prepare-refinement":
        output_dir = DATA_DIR / args.output_name
        files = prepare_refinement_dataset(
            data_dir=Path(args.data_dir),
            output_dir=output_dir,
            max_examples=args.max_examples,
            precise_min_width=args.precise_min_width,
            precise_max_width=args.precise_max_width,
            broad_margin_min=args.broad_margin_min,
            broad_margin_max=args.broad_margin_max,
            seed=args.seed,
        )
        print(f"Prepared refinement dataset in {output_dir}")
        for name, path in files.items():
            print(f"{name}: {path}")
        return 0

    if args.command == "run-saved":
        config = ExperimentConfig(
            seed=args.seed,
            epochs=args.epochs,
            backend=args.backend,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            mlp_hidden_dim=args.mlp_hidden_dim,
            mlp_hidden2_dim=args.mlp_hidden2_dim,
            mlp_dropout=args.mlp_dropout,
            hybrid_exact_weight=args.hybrid_exact_weight,
            soft_cover_reward=args.soft_cover_reward,
            soft_overlap_base=args.soft_overlap_base,
            soft_overlap_f1_weight=args.soft_overlap_f1_weight,
            soft_overlap_precision_weight=args.soft_overlap_precision_weight,
            soft_cover_breadth_penalty=args.soft_cover_breadth_penalty,
            soft_overlap_breadth_penalty=args.soft_overlap_breadth_penalty,
            soft_cover_penalty_cap=args.soft_cover_penalty_cap,
            soft_overlap_penalty_cap=args.soft_overlap_penalty_cap,
            soft_local_reward=args.soft_local_reward,
            soft_local_distance=args.soft_local_distance,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
        )
        results, run_path = run_saved_experiment(config, Path(args.data_dir))
        print(format_report(results, run_path))
        return 0

    if args.command == "ensemble-saved":
        config = ExperimentConfig(
            epochs=args.epochs,
            backend=args.backend,
            mlp_hidden_dim=args.mlp_hidden_dim,
            mlp_hidden2_dim=args.mlp_hidden2_dim,
            mlp_dropout=args.mlp_dropout,
            hybrid_exact_weight=args.hybrid_exact_weight,
            soft_cover_reward=args.soft_cover_reward,
            soft_overlap_base=args.soft_overlap_base,
            soft_overlap_f1_weight=args.soft_overlap_f1_weight,
            soft_overlap_precision_weight=args.soft_overlap_precision_weight,
            soft_cover_breadth_penalty=args.soft_cover_breadth_penalty,
            soft_overlap_breadth_penalty=args.soft_overlap_breadth_penalty,
            soft_cover_penalty_cap=args.soft_cover_penalty_cap,
            soft_overlap_penalty_cap=args.soft_overlap_penalty_cap,
            soft_local_reward=args.soft_local_reward,
            soft_local_distance=args.soft_local_distance,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
        )
        seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
        results, run_path = run_ensemble_saved_experiment(
            config,
            Path(args.data_dir),
            seeds=seeds,
            model_kind=args.model_kind,
            merge_mode=args.merge_mode,
        )
        print(format_ensemble_report(results, run_path))
        return 0

    if args.command == "range-saved":
        config = ExperimentConfig(
            seed=args.seed,
            epochs=args.epochs,
            backend=args.backend,
            mlp_hidden_dim=args.mlp_hidden_dim,
            mlp_hidden2_dim=args.mlp_hidden2_dim,
            mlp_dropout=args.mlp_dropout,
            range_exact_weight=args.range_exact_weight,
            range_neighbor_sigma=args.range_neighbor_sigma,
            range_breadth_penalty=args.range_breadth_penalty,
            range_soft_weight=args.range_soft_weight,
            range_width_weight=args.range_width_weight,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
        )
        results, run_path = run_range_saved_experiment(config, Path(args.data_dir))
        print(format_range_report(results, run_path))
        return 0

    if args.command == "range-ensemble-saved":
        config = ExperimentConfig(
            epochs=args.epochs,
            backend=args.backend,
            mlp_hidden_dim=args.mlp_hidden_dim,
            mlp_hidden2_dim=args.mlp_hidden2_dim,
            mlp_dropout=args.mlp_dropout,
            range_exact_weight=args.range_exact_weight,
            range_neighbor_sigma=args.range_neighbor_sigma,
            range_breadth_penalty=args.range_breadth_penalty,
            range_soft_weight=args.range_soft_weight,
            range_width_weight=args.range_width_weight,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
        )
        seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
        results, run_path = run_range_ensemble_saved_experiment(
            config,
            Path(args.data_dir),
            seeds=seeds,
        )
        print(format_range_report(results, run_path))
        return 0

    if args.command == "center-width-saved":
        config = ExperimentConfig(
            seed=args.seed,
            epochs=args.epochs,
            backend=args.backend,
            mlp_hidden_dim=args.mlp_hidden_dim,
            mlp_hidden2_dim=args.mlp_hidden2_dim,
            mlp_dropout=args.mlp_dropout,
            center_exact_weight=args.center_exact_weight,
            center_soft_weight=args.center_soft_weight,
            center_width_weight=args.center_width_weight,
            range_neighbor_sigma=args.range_neighbor_sigma,
            range_breadth_penalty=args.range_breadth_penalty,
            bm25_weight=args.bm25_weight,
            bm25_top_k=args.bm25_top_k,
            bm25_margin=args.bm25_margin,
            bm25_outside_penalty=args.bm25_outside_penalty,
            bm25_enrich_train=args.bm25_enrich_train,
            route_bucket_size=args.route_bucket_size,
            route_top_k=args.route_top_k,
            route_margin=args.route_margin,
            route_outside_penalty=args.route_outside_penalty,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
        )
        results, run_path = run_center_width_saved_experiment(config, Path(args.data_dir))
        print(format_center_width_report(results, run_path))
        return 0

    if args.command == "center-width-ensemble-saved":
        config = ExperimentConfig(
            epochs=args.epochs,
            backend=args.backend,
            mlp_hidden_dim=args.mlp_hidden_dim,
            mlp_hidden2_dim=args.mlp_hidden2_dim,
            mlp_dropout=args.mlp_dropout,
            center_exact_weight=args.center_exact_weight,
            center_soft_weight=args.center_soft_weight,
            center_width_weight=args.center_width_weight,
            range_neighbor_sigma=args.range_neighbor_sigma,
            range_breadth_penalty=args.range_breadth_penalty,
            bm25_weight=args.bm25_weight,
            bm25_top_k=args.bm25_top_k,
            bm25_margin=args.bm25_margin,
            bm25_outside_penalty=args.bm25_outside_penalty,
            bm25_enrich_train=args.bm25_enrich_train,
            route_bucket_size=args.route_bucket_size,
            route_top_k=args.route_top_k,
            route_margin=args.route_margin,
            route_outside_penalty=args.route_outside_penalty,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
        )
        seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
        results, run_path = run_center_width_ensemble_saved_experiment(
            config,
            Path(args.data_dir),
            seeds=seeds,
        )
        print(format_center_width_report(results, run_path))
        return 0

    if args.command == "refinement-loop-saved":
        config = ExperimentConfig(
            epochs=args.epochs,
            backend=args.backend,
            mlp_hidden_dim=args.mlp_hidden_dim,
            mlp_hidden2_dim=args.mlp_hidden2_dim,
            mlp_dropout=args.mlp_dropout,
            center_exact_weight=args.center_exact_weight,
            center_soft_weight=args.center_soft_weight,
            center_width_weight=args.center_width_weight,
            range_neighbor_sigma=args.range_neighbor_sigma,
            range_breadth_penalty=args.range_breadth_penalty,
            local_cross_hidden_dim=args.local_cross_hidden_dim,
            local_cross_epochs=args.local_cross_epochs,
            local_cross_dropout=args.local_cross_dropout,
            local_cross_blend_steps=args.local_cross_blend_steps,
            local_cross_max_candidates=args.local_cross_max_candidates,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
        )
        seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
        results, run_path = run_refinement_loop_saved_experiment(
            config,
            Path(args.data_dir),
            seeds=seeds,
        )
        print(format_refinement_loop_report(results, run_path))
        return 0

    if args.command == "refinement-two-model-saved":
        config = ExperimentConfig(
            epochs=args.epochs,
            backend=args.backend,
            mlp_hidden_dim=args.mlp_hidden_dim,
            mlp_hidden2_dim=args.mlp_hidden2_dim,
            mlp_dropout=args.mlp_dropout,
            center_exact_weight=args.center_exact_weight,
            center_soft_weight=args.center_soft_weight,
            center_width_weight=args.center_width_weight,
            range_neighbor_sigma=args.range_neighbor_sigma,
            range_breadth_penalty=args.range_breadth_penalty,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
        )
        seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
        results, run_path = run_refinement_two_model_saved_experiment(
            config,
            Path(args.data_dir),
            seeds=seeds,
        )
        print(format_refinement_loop_report(results, run_path))
        return 0

    if args.command == "hybrid-fusion-saved":
        config = ExperimentConfig(
            epochs=args.epochs,
            backend=args.backend,
            mlp_hidden_dim=args.mlp_hidden_dim,
            mlp_hidden2_dim=args.mlp_hidden2_dim,
            mlp_dropout=args.mlp_dropout,
            hybrid_exact_weight=args.hybrid_exact_weight,
            soft_cover_reward=args.soft_cover_reward,
            soft_overlap_base=args.soft_overlap_base,
            soft_overlap_f1_weight=args.soft_overlap_f1_weight,
            soft_overlap_precision_weight=args.soft_overlap_precision_weight,
            soft_cover_breadth_penalty=args.soft_cover_breadth_penalty,
            soft_overlap_breadth_penalty=args.soft_overlap_breadth_penalty,
            soft_cover_penalty_cap=args.soft_cover_penalty_cap,
            soft_overlap_penalty_cap=args.soft_overlap_penalty_cap,
            soft_local_reward=args.soft_local_reward,
            soft_local_distance=args.soft_local_distance,
            center_exact_weight=args.center_exact_weight,
            center_soft_weight=args.center_soft_weight,
            center_width_weight=args.center_width_weight,
            range_neighbor_sigma=args.range_neighbor_sigma,
            range_breadth_penalty=args.range_breadth_penalty,
            bm25_weight=args.bm25_weight,
            bm25_top_k=args.bm25_top_k,
            bm25_margin=args.bm25_margin,
            bm25_outside_penalty=args.bm25_outside_penalty,
            bm25_enrich_train=args.bm25_enrich_train,
            fusion_anchor_weight=args.fusion_anchor_weight,
            fusion_range_weight=args.fusion_range_weight,
            fusion_breadth_penalty=args.fusion_breadth_penalty,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
        )
        seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
        results, run_path = run_hybrid_fusion_saved_experiment(
            config,
            Path(args.data_dir),
            seeds=seeds,
        )
        print(format_fusion_report(results, run_path))
        return 0

    config = ExperimentConfig(
        anchors=args.anchors,
        seed=args.seed,
        epochs=args.epochs,
        backend=args.backend,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_hidden2_dim=args.mlp_hidden2_dim,
        mlp_dropout=args.mlp_dropout,
        hybrid_exact_weight=args.hybrid_exact_weight,
        soft_cover_reward=args.soft_cover_reward,
        soft_overlap_base=args.soft_overlap_base,
        soft_overlap_f1_weight=args.soft_overlap_f1_weight,
        soft_overlap_precision_weight=args.soft_overlap_precision_weight,
        soft_cover_breadth_penalty=args.soft_cover_breadth_penalty,
        soft_overlap_breadth_penalty=args.soft_overlap_breadth_penalty,
        soft_cover_penalty_cap=args.soft_cover_penalty_cap,
        soft_overlap_penalty_cap=args.soft_overlap_penalty_cap,
        soft_local_reward=args.soft_local_reward,
        soft_local_distance=args.soft_local_distance,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
    )
    results, run_path = run_experiment(config)
    print(format_report(results, run_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
