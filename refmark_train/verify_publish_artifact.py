from __future__ import annotations

import json
import hashlib
from pathlib import Path


RETAINED_DATASETS = [
    "documentation_full_paragraph_random",
    "documentation_full_paragraph_contextual",
    "documentation_full_paragraph_contextual_idf_lean2",
    "documentation_refinement_w1_2_broad_v3",
    "corporate_refinement_w1_2_broad_smoke",
]

RETAINED_RUNS = [
    "center_width_ensemble_20260424T102842Z_documentation_full_paragraph_random_torchcpu.json",
    "center_width_ensemble_20260424T105016Z_documentation_full_paragraph_contextual_torchcpu.json",
    "center_width_ensemble_20260424T115745Z_documentation_full_paragraph_contextual_idf_lean2_torchcpu.json",
    "refinement_two_model_20260424T083449Z_documentation_refinement_w1_2_broad_v3_torchcpu.json",
    "refinement_two_model_20260424T081958Z_corporate_refinement_w1_2_broad_smoke_torchcpu.json",
]


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _verify_source_docs(root: Path, errors: list[str]) -> None:
    source_root = root / "source_docs"
    manifest = _load_json(source_root / "manifest.json")
    sets_manifest = _load_json(source_root / "sets" / "manifest.json")

    for item in manifest:
        _check(bool(item.get("url")), f"missing source url in manifest row: {item}", errors)
        _check(bool(item.get("id")), f"missing source id in manifest row: {item}", errors)

    for item in sets_manifest:
        _check(bool(item.get("text_path")), f"missing source-set text_path in manifest row: {item}", errors)
        _check(bool(item.get("documents")), f"missing source-set documents in manifest row: {item}", errors)


def _verify_dataset(root: Path, dataset_name: str, errors: list[str]) -> None:
    dataset_dir = root / "data" / dataset_name
    manifest_path = dataset_dir / "manifest.json"
    _check(manifest_path.exists(), f"missing dataset manifest: {manifest_path}", errors)
    if not manifest_path.exists():
        return

    manifest = _load_json(manifest_path)
    checksums = manifest.get("file_sha256", {})
    for key, expected_count_key in [
        ("anchors", "anchor_count"),
        ("train", "train_examples"),
        ("valid", "valid_examples"),
        ("reformulated", "reformulated_examples"),
    ]:
        rel_path = manifest["files"][key]
        target = dataset_dir / rel_path
        _check(target.exists(), f"missing dataset file: {target}", errors)
        if target.exists():
            actual = _jsonl_count(target)
            expected = int(manifest[expected_count_key])
            _check(
                actual == expected,
                f"count mismatch for {target}: expected {expected}, found {actual}",
                errors,
            )
            expected_hash = checksums.get(key) if isinstance(checksums, dict) else None
            if expected_hash:
                actual_hash = _sha256(target)
                _check(
                    actual_hash == expected_hash,
                    f"sha256 mismatch for {target}: expected {expected_hash}, found {actual_hash}",
                    errors,
                )

    if "source" in manifest:
        _check(bool(manifest["source"]), f"empty source corpus lineage reference: {manifest_path}", errors)
    if "source_data_dir" in manifest:
        _check(bool(manifest["source_data_dir"]), f"empty source data lineage reference: {manifest_path}", errors)


def _verify_runs(root: Path, errors: list[str]) -> None:
    runs_dir = root / "runs"
    for run_name in RETAINED_RUNS:
        run_path = runs_dir / run_name
        _check(run_path.exists(), f"missing retained run artifact: {run_path}", errors)
        if not run_path.exists():
            continue
        run = _load_json(run_path)
        data_dir = run.get("dataset", {}).get("data_dir")
        _check(bool(data_dir), f"run artifact missing dataset.data_dir: {run_path}", errors)
        if data_dir:
            target = (runs_dir / data_dir).resolve()
            _check(target.exists(), f"run artifact data_dir does not resolve: {target}", errors)


def _verify_cli(errors: list[str]) -> None:
    from refmark_train.cli import build_parser

    parser = build_parser()
    _check(parser is not None, "failed to construct CLI parser", errors)


def main() -> int:
    root = Path(__file__).resolve().parent
    errors: list[str] = []

    for relative_path in [
        "README.md",
        "RESULTS.md",
        "PIPELINES.md",
        "PUBLISH_REVIEW.md",
        "SOURCE_PROVENANCE.md",
        "pyproject.toml",
        "requirements.txt",
    ]:
        _check((root / relative_path).exists(), f"missing publish file: {root / relative_path}", errors)

    _verify_source_docs(root, errors)
    for dataset_name in RETAINED_DATASETS:
        _verify_dataset(root, dataset_name, errors)
    _verify_runs(root, errors)
    _verify_cli(errors)

    if errors:
        print("Refmark Train publish verification failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Refmark Train publish verification passed.")
    print(f"Artifact root: {root}")
    print(f"Retained datasets: {len(RETAINED_DATASETS)}")
    print(f"Retained run artifacts: {len(RETAINED_RUNS)}")
    print("Retained dataset files: SHA-256 verified.")
    print("Source payload focus: manifests only; raw/text payloads are not redistributed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
