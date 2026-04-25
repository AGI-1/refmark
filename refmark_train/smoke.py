from __future__ import annotations

from refmark_train.experiment import ExperimentConfig, _evaluate_bundle
from refmark_train.synthetic import build_corpus


def run_smoke() -> dict[str, object]:
    """Train and evaluate on a tiny synthetic corpus without network or files."""
    bundle = build_corpus(anchor_count=12, seed=7)
    config = ExperimentConfig(
        anchors=12,
        seed=7,
        epochs=2,
        backend="cpu",
        batch_size=16,
        learning_rate=0.2,
        embedding_dim=8,
        hidden_dim=12,
    )
    results = _evaluate_bundle(bundle, config)
    required = ["baseline", "retrieval_baseline", "bm25_baseline", "tiny_model"]
    ok = all(name in results for name in required)
    return {
        "ok": ok,
        "anchors": len(bundle.anchors),
        "train_examples": len(bundle.train),
        "valid_examples": len(bundle.valid),
        "tiny_model_valid_accuracy": results["tiny_model"]["valid"]["accuracy"],
        "bm25_valid_accuracy": results["bm25_baseline"]["valid"]["accuracy"],
    }


def main() -> int:
    result = run_smoke()
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
