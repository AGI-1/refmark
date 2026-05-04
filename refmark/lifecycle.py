from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

from refmark.search_index import approx_tokens

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

LIFECYCLE_STATES = {
    "unchanged",
    "moved",
    "rewritten",
    "split_support",
    "merged",
    "deleted",
    "ambiguous",
    "alternative_support",
    "duplicate_support",
    "contradictory_support",
    "low_confidence",
    "partial_overlap",
    "semantic_drift",
    "superseded",
    "deprecated",
    "externalized",
    "invalidated",
}

DEFAULT_SUMMARY_COLUMNS = [
    "repo_url",
    "old_ref",
    "new_ref",
    "old_labels",
    "competent_silent_drift_rate",
    "competent_false_stale_rate",
    "competent_review_rate",
    "competent_preserved_rate",
    "layered_silent_drift_rate",
    "layered_review_rate",
    "layered_preserved_rate",
    "refmark_auto_rate",
    "refmark_review_rate",
    "refmark_stale_rate",
    "naive_correct_rate",
    "naive_silent_wrong_rate",
    "naive_missing_rate",
    "workload_reduction_vs_audit",
]


@dataclass(frozen=True)
class Region:
    path: str
    ref: str
    ordinal: int
    text: str
    fingerprint: str


@dataclass(frozen=True)
class LifecycleDecision:
    state: str
    reason: str
    confidence: float
    next_action: str
    candidate_ref: str | None = None
    priority: str = "medium"

    def as_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
            "next_action": self.next_action,
            "candidate_ref": self.candidate_ref,
            "priority": self.priority,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate stable refs across natural Git documentation revisions.")
    parser.add_argument("--repo-url", required=True)
    parser.add_argument("--old-ref", required=True)
    parser.add_argument("--new-refs", required=True, help="Comma-separated new refs to compare against --old-ref.")
    parser.add_argument("--subdir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", default="", help="Optional compact JSON summary rows output path.")
    parser.add_argument("--csv-output", default="", help="Optional compact CSV summary rows output path.")
    parser.add_argument("--region-tokens", type=int, default=110)
    parser.add_argument("--region-stride", type=int, default=110)
    parser.add_argument("--max-files", type=int, default=0)
    args = parser.parse_args()

    refs_to_run = [ref.strip() for ref in args.new_refs.split(",") if ref.strip()]
    payload = evaluate_git_revisions(
        repo_url=args.repo_url,
        old_ref=args.old_ref,
        new_refs=refs_to_run,
        subdir=args.subdir,
        work_dir=args.work_dir,
        output=args.output,
        summary_output=args.summary_output or None,
        csv_output=args.csv_output or None,
        region_tokens=args.region_tokens,
        region_stride=args.region_stride,
        max_files=args.max_files,
    )
    print(json.dumps(payload, indent=2))


def evaluate_git_revisions(
    *,
    repo_url: str,
    old_ref: str,
    new_refs: list[str],
    subdir: str,
    work_dir: str | Path,
    output: str | Path | None = None,
    summary_output: str | Path | None = None,
    csv_output: str | Path | None = None,
    region_tokens: int = 110,
    region_stride: int = 110,
    max_files: int = 0,
) -> dict[str, object]:
    """Evaluate evidence-label stability across natural Git revisions."""

    started = time.perf_counter()
    work = Path(work_dir)
    repo = ensure_repo(repo_url, work / "repo")
    old_root = export_ref(repo, old_ref, subdir, work / "old")
    old_regions = load_regions(old_root, region_tokens=region_tokens, region_stride=region_stride, max_files=max_files)
    revision_reports = []
    for new_ref in new_refs:
        new_root = export_ref(repo, new_ref, subdir, work / f"new_{safe_name(new_ref)}")
        new_regions = load_regions(new_root, region_tokens=region_tokens, region_stride=region_stride, max_files=max_files)
        stable = evaluate_stable_migration(old_regions, new_regions)
        naive = evaluate_naive_path_ordinal(old_regions, new_regions)
        chunk_hash = evaluate_chunk_id_content_hash(old_regions, new_regions, stable)
        source_hash = evaluate_qrels_source_hash(old_regions, new_regions, stable)
        quote_selector = evaluate_chunk_hash_quote_selector(old_regions, new_regions, stable)
        layered_selector = evaluate_layered_anchor_selector(old_regions, new_regions, stable)
        file_report = evaluate_file_lifecycle(old_regions, new_regions)
        eval_lifecycle = evaluate_eval_label_lifecycle(
            stable,
            naive,
            chunk_hash=chunk_hash,
            source_hash=source_hash,
            quote_selector=quote_selector,
            layered_selector=layered_selector,
        )
        revision_reports.append(
            {
                "new_ref": new_ref,
                "new_files": len(new_regions),
                "new_regions": sum(len(v) for v in new_regions.values()),
                "new_tokens": sum(approx_tokens(r.text) for regions in new_regions.values() for r in regions),
                "file_lifecycle": file_report,
                "stable_ref_migration": stable,
                "naive_path_ordinal_identity": naive,
                "chunk_id_content_hash_identity": chunk_hash,
                "qrels_source_hash_identity": source_hash,
                "chunk_hash_quote_selector_identity": quote_selector,
                "refmark_layered_selector_identity": layered_selector,
                "eval_label_lifecycle": eval_lifecycle,
            }
        )

    payload: dict[str, object] = {
        "schema": "refmark.git_revision_stability.v1",
        "repo_url": repo_url,
        "old_ref": old_ref,
        "new_ref": new_refs[-1] if new_refs else None,
        "new_refs": new_refs,
        "subdir": subdir,
        "settings": {
            "repo_url": repo_url,
            "old_ref": old_ref,
            "new_refs": ",".join(new_refs),
            "subdir": subdir,
            "work_dir": str(work_dir),
            "output": str(output) if output else "",
            "summary_output": str(summary_output) if summary_output else "",
            "csv_output": str(csv_output) if csv_output else "",
            "region_tokens": region_tokens,
            "region_stride": region_stride,
            "max_files": max_files,
        },
        "old_files": len(old_regions),
        "new_files": revision_reports[-1]["new_files"] if revision_reports else 0,
        "old_regions": sum(len(v) for v in old_regions.values()),
        "new_regions": revision_reports[-1]["new_regions"] if revision_reports else 0,
        "old_tokens": sum(approx_tokens(r.text) for regions in old_regions.values() for r in regions),
        "new_tokens": revision_reports[-1]["new_tokens"] if revision_reports else 0,
        "file_lifecycle": revision_reports[-1]["file_lifecycle"] if revision_reports else {},
        "stable_ref_migration": revision_reports[-1]["stable_ref_migration"] if revision_reports else {},
        "naive_path_ordinal_identity": revision_reports[-1]["naive_path_ordinal_identity"] if revision_reports else {},
        "chunk_id_content_hash_identity": revision_reports[-1]["chunk_id_content_hash_identity"] if revision_reports else {},
        "qrels_source_hash_identity": revision_reports[-1]["qrels_source_hash_identity"] if revision_reports else {},
        "chunk_hash_quote_selector_identity": revision_reports[-1]["chunk_hash_quote_selector_identity"] if revision_reports else {},
        "refmark_layered_selector_identity": revision_reports[-1]["refmark_layered_selector_identity"] if revision_reports else {},
        "eval_label_lifecycle": revision_reports[-1]["eval_label_lifecycle"] if revision_reports else {},
        "revision_reports": revision_reports,
        "summary_rows": revision_summary_rows(repo_url, old_ref, subdir, revision_reports),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "interpretation": [
            "stable_ref_migration matches old path-local refs to new regions by fingerprint/fuzzy text, first in the same file and then across renamed/moved files.",
            "naive_path_ordinal_identity assumes path plus ordinal region id remains valid after the revision.",
            "chunk_id_content_hash_identity is the competent conservative baseline: same path/ordinal plus unchanged region hash, otherwise review/stale.",
            "qrels_source_hash_identity models qrels plus source file hash: labels in changed files require review even when evidence survived.",
            "chunk_hash_quote_selector_identity models chunk id plus content hash plus quote selector plus corpus version.",
            "refmark_layered_selector_identity combines exact hashes, quote selectors, and similarity thresholds; ambiguous or low-confidence cases go to review.",
            "eval_label_lifecycle treats each old region as a maintained query->evidence label and estimates preserved/migrated/review/stale versus silent corruption.",
            "This is a natural Git revision benchmark over documentation files, not a synthetic mutation test.",
        ],
    }
    if output:
        write_json(Path(output), payload)
    if summary_output:
        write_json(Path(summary_output), payload["summary_rows"])
    if csv_output:
        write_summary_csv(Path(csv_output), payload["summary_rows"])
    return payload


def safe_name(ref: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", ref)


def ensure_repo(repo_url: str, path: Path) -> Path:
    if path.exists():
        subprocess.run(["git", "-C", str(path), "fetch", "--tags", "--quiet"], check=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", repo_url, str(path)], check=True)
    return path


def export_ref(repo: Path, ref: str, subdir: str, output: Path) -> Path:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "--quiet", ref], check=True)
    src = repo / subdir
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copytree(src, output / "docs")
    return output / "docs"


def load_regions(root: Path, *, region_tokens: int, region_stride: int, max_files: int) -> dict[str, list[Region]]:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".md", ".mdx", ".rst", ".txt"})
    if max_files:
        files = files[:max_files]
    result = {}
    for path in files:
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        regions = make_regions(rel, text, size=region_tokens, stride=region_stride)
        if regions:
            result[rel] = regions
    return result


def make_regions(path: str, text: str, *, size: int, stride: int) -> list[Region]:
    words = text.split()
    if not words:
        return []
    regions = []
    start = 0
    index = 1
    while start < len(words):
        chunk = " ".join(words[start : start + size])
        regions.append(Region(path=path, ref=f"{path}:P{index:03d}", ordinal=index, text=chunk, fingerprint=fingerprint(chunk)))
        if start + size >= len(words):
            break
        start += max(1, stride)
        index += 1
    return regions


def fingerprint(text: str) -> str:
    tokens = [token.lower() for token in TOKEN_RE.findall(text)]
    return hashlib.sha256(" ".join(tokens).encode("utf-8")).hexdigest()[:16]


def token_jaccard(a: str, b: str) -> float:
    aa = set(token.lower() for token in TOKEN_RE.findall(a))
    bb = set(token.lower() for token in TOKEN_RE.findall(b))
    return token_set_jaccard(aa, bb)


def token_set_jaccard(aa: set[str], bb: set[str]) -> float:
    if not aa and not bb:
        return 1.0
    return len(aa & bb) / max(len(aa | bb), 1)


def token_set(text: str) -> set[str]:
    return set(token.lower() for token in TOKEN_RE.findall(text))


def lifecycle_decision(
    state: str,
    *,
    reason: str,
    confidence: float,
    next_action: str | None = None,
    candidate_ref: str | None = None,
    priority: str | None = None,
) -> LifecycleDecision:
    if state not in LIFECYCLE_STATES:
        raise ValueError(f"Unknown lifecycle state: {state}")
    action = next_action or default_next_action(state)
    return LifecycleDecision(
        state=state,
        reason=reason,
        confidence=max(0.0, min(1.0, confidence)),
        next_action=action,
        candidate_ref=candidate_ref,
        priority=priority or default_priority(state),
    )


def default_next_action(state: str) -> str:
    if state == "unchanged":
        return "keep_label"
    if state == "moved":
        return "migrate_ref"
    if state == "rewritten":
        return "review_rewrite_or_preserve"
    if state == "split_support":
        return "review_range_repair"
    if state == "merged":
        return "review_merged_support"
    if state == "deleted":
        return "refresh_or_remove_label"
    if state == "ambiguous":
        return "human_disambiguation_required"
    if state == "alternative_support":
        return "review_alternative_support"
    if state == "duplicate_support":
        return "review_duplicate_support"
    if state == "contradictory_support":
        return "review_contradiction"
    if state == "low_confidence":
        return "review_low_confidence_match"
    if state == "partial_overlap":
        return "review_partial_support"
    if state == "semantic_drift":
        return "verify_semantic_equivalence"
    if state == "superseded":
        return "review_new_canonical_support"
    if state == "deprecated":
        return "flag_for_deprecation_review"
    if state == "externalized":
        return "review_external_corpus_boundary"
    if state == "invalidated":
        return "archive_or_correct_label"
    raise ValueError(f"Unknown lifecycle state: {state}")


def default_priority(state: str) -> str:
    if state in {"unchanged", "moved"}:
        return "low"
    if state in {"deleted", "contradictory_support", "semantic_drift", "externalized", "invalidated"}:
        return "high"
    return "medium"


def evaluate_stable_migration(old: dict[str, list[Region]], new: dict[str, list[Region]]) -> dict[str, object]:
    all_new = [region for regions in new.values() for region in regions]
    fuzzy_index = FuzzyRegionIndex(all_new)
    new_by_path_fp = {
        path: {region.fingerprint: region for region in regions}
        for path, regions in new.items()
    }
    new_by_fp: dict[str, list[Region]] = {}
    for region in all_new:
        new_by_fp.setdefault(region.fingerprint, []).append(region)
    counts = Counter()
    lifecycle_counts = Counter()
    examples = {"same_file_exact": [], "moved_exact": [], "fuzzy": [], "split_support": [], "stale": []}
    status_by_ref: dict[str, str] = {}
    lifecycle_by_ref: dict[str, dict[str, object]] = {}
    total = 0
    for path, regions in old.items():
        same_file_fps = new_by_path_fp.get(path, {})
        same_file_regions = new.get(path, [])
        for region in regions:
            total += 1
            same = same_file_fps.get(region.fingerprint)
            if same:
                counts["same_file_exact"] += 1
                status_by_ref[region.ref] = "same_file_exact"
                decision = lifecycle_decision(
                    "unchanged" if same.ordinal == region.ordinal else "moved",
                    reason="same_file_exact_hash" if same.ordinal == region.ordinal else "same_file_exact_hash_new_ordinal",
                    confidence=1.0,
                    candidate_ref=same.ref,
                )
                lifecycle_counts[decision.state] += 1
                lifecycle_by_ref[region.ref] = decision.as_dict()
                if same.ordinal != region.ordinal and len(examples["same_file_exact"]) < 6:
                    examples["same_file_exact"].append({"old_ref": region.ref, "new_ref": same.ref})
                continue
            moved = [candidate for candidate in new_by_fp.get(region.fingerprint, []) if candidate.path != path]
            if moved:
                counts["moved_exact"] += 1
                status_by_ref[region.ref] = "moved_exact"
                decision = lifecycle_decision(
                    "moved",
                    reason="global_exact_hash_different_path",
                    confidence=1.0,
                    candidate_ref=moved[0].ref,
                )
                lifecycle_counts[decision.state] += 1
                lifecycle_by_ref[region.ref] = decision.as_dict()
                if len(examples["moved_exact"]) < 6:
                    examples["moved_exact"].append({"old_ref": region.ref, "new_ref": moved[0].ref})
                continue
            same_file_best = fuzzy_index.best_match(region, candidates=same_file_regions)
            global_best = fuzzy_index.best_match(region)
            best = _best_match(same_file_best, global_best)
            if best and best[1] >= 0.82:
                counts["fuzzy"] += 1
                status_by_ref[region.ref] = "fuzzy"
                decision = lifecycle_decision(
                    "rewritten",
                    reason="fuzzy_text_match_requires_review",
                    confidence=best[1],
                    next_action="review_rewrite_or_preserve",
                    candidate_ref=best[0].ref,
                )
                lifecycle_counts[decision.state] += 1
                lifecycle_by_ref[region.ref] = decision.as_dict()
                if len(examples["fuzzy"]) < 6:
                    examples["fuzzy"].append({"old_ref": region.ref, "new_ref": best[0].ref, "similarity": round(best[1], 3)})
            elif split := split_support_match(region, all_new):
                support_regions, coverage = split
                counts["split_support"] += 1
                status_by_ref[region.ref] = "split_support"
                candidate_refs = [candidate.ref for candidate in support_regions]
                decision = lifecycle_decision(
                    "split_support",
                    reason="combined_neighboring_or_distributed_regions_cover_old_tokens",
                    confidence=coverage,
                    candidate_ref=",".join(candidate_refs),
                )
                lifecycle_counts[decision.state] += 1
                lifecycle_by_ref[region.ref] = decision.as_dict()
                if len(examples["split_support"]) < 6:
                    examples["split_support"].append(
                        {
                            "old_ref": region.ref,
                            "candidate_refs": candidate_refs,
                            "coverage": round(coverage, 3),
                        }
                    )
            else:
                counts["stale"] += 1
                status_by_ref[region.ref] = "stale"
                decision = lifecycle_decision(
                    "deleted",
                    reason="no_exact_or_high_confidence_fuzzy_match",
                    confidence=best[1] if best else 0.0,
                    next_action="refresh_or_remove_label",
                    candidate_ref=best[0].ref if best else None,
                )
                lifecycle_counts[decision.state] += 1
                lifecycle_by_ref[region.ref] = decision.as_dict()
                if len(examples["stale"]) < 6:
                    examples["stale"].append({"old_ref": region.ref, "best_similarity": round(best[1], 3) if best else 0.0})
    report = rates(total, counts, examples)
    report["status_by_ref"] = status_by_ref
    report["lifecycle_state_counts"] = dict(lifecycle_counts)
    report["lifecycle_by_ref"] = lifecycle_by_ref
    return report


def evaluate_naive_path_ordinal(old: dict[str, list[Region]], new: dict[str, list[Region]]) -> dict[str, object]:
    counts = Counter()
    examples = {"wrong_same_id": [], "missing": []}
    total = 0
    new_by_path_ord = {
        path: {region.ordinal: region for region in regions}
        for path, regions in new.items()
    }
    for path, regions in old.items():
        same_path = new_by_path_ord.get(path, {})
        for region in regions:
            total += 1
            candidate = same_path.get(region.ordinal)
            if candidate is None:
                counts["missing"] += 1
                if len(examples["missing"]) < 6:
                    examples["missing"].append({"old_ref": region.ref})
            elif candidate.fingerprint == region.fingerprint or token_jaccard(candidate.text, region.text) >= 0.82:
                counts["correct"] += 1
            else:
                counts["wrong_same_id"] += 1
                if len(examples["wrong_same_id"]) < 6:
                    examples["wrong_same_id"].append({"old_ref": region.ref, "same_ordinal_ref": candidate.ref, "similarity": round(token_jaccard(candidate.text, region.text), 3)})
    return rates(total, counts, examples)


def evaluate_chunk_id_content_hash(
    old: dict[str, list[Region]],
    new: dict[str, list[Region]],
    stable_report: dict[str, object],
) -> dict[str, object]:
    """Conservative same chunk id + content hash baseline.

    This is the first competent baseline: a label is preserved only when the
    same path-local ordinal exists and its region content hash is unchanged.
    Any mismatch is flagged for review/stale. That avoids silent drift but can
    over-alert when evidence moved or changed only cosmetically.
    """

    oracle = _oracle_validity(stable_report)
    new_by_path_ord = {
        path: {region.ordinal: region for region in regions}
        for path, regions in new.items()
    }
    counts = Counter()
    examples = {"preserved": [], "false_stale": [], "true_stale": []}
    total = 0
    for path, regions in old.items():
        same_path = new_by_path_ord.get(path, {})
        for region in regions:
            total += 1
            candidate = same_path.get(region.ordinal)
            valid = oracle.get(region.ref, False)
            if candidate and candidate.fingerprint == region.fingerprint:
                counts["preserved"] += 1
                if len(examples["preserved"]) < 6:
                    examples["preserved"].append({"old_ref": region.ref, "new_ref": candidate.ref})
            elif valid:
                counts["false_stale_alert"] += 1
                if len(examples["false_stale"]) < 6:
                    examples["false_stale"].append({"old_ref": region.ref})
            else:
                counts["true_stale_alert"] += 1
                if len(examples["true_stale"]) < 6:
                    examples["true_stale"].append({"old_ref": region.ref})
    return rates(total, counts, examples)


def evaluate_qrels_source_hash(
    old: dict[str, list[Region]],
    new: dict[str, list[Region]],
    stable_report: dict[str, object],
) -> dict[str, object]:
    """qrels + source-file hash baseline.

    This models an evaluator that stores stable qrels IDs plus a source file
    hash. If a file hash changes, every label in that file is flagged. It is
    simple and defensible, but can create high review workload for large files.
    """

    oracle = _oracle_validity(stable_report)
    old_file_hashes = _file_fingerprints(old)
    new_file_hashes = _file_fingerprints(new)
    counts = Counter()
    examples = {"preserved": [], "false_stale": [], "true_stale": []}
    total = 0
    for path, regions in old.items():
        unchanged_file = old_file_hashes.get(path) == new_file_hashes.get(path)
        for region in regions:
            total += 1
            valid = oracle.get(region.ref, False)
            if unchanged_file:
                counts["preserved"] += 1
                if len(examples["preserved"]) < 6:
                    examples["preserved"].append({"old_ref": region.ref})
            elif valid:
                counts["false_stale_alert"] += 1
                if len(examples["false_stale"]) < 6:
                    examples["false_stale"].append({"old_ref": region.ref})
            else:
                counts["true_stale_alert"] += 1
                if len(examples["true_stale"]) < 6:
                    examples["true_stale"].append({"old_ref": region.ref})
    return rates(total, counts, examples)


def evaluate_chunk_hash_quote_selector(
    old: dict[str, list[Region]],
    new: dict[str, list[Region]],
    stable_report: dict[str, object],
) -> dict[str, object]:
    """Competent baseline: chunk id + hash + quote selector + corpus version.

    This baseline keeps labels when the same path/ordinal hash still matches,
    otherwise it tries to re-anchor using an exact normalized quote selector
    derived from the old region. Multiple quote hits require review.
    """

    oracle = _oracle_validity(stable_report)
    all_new = [region for regions in new.values() for region in regions]
    new_by_path_ord = {
        path: {region.ordinal: region for region in regions}
        for path, regions in new.items()
    }
    quote_index = _quote_index(all_new)
    counts = Counter()
    examples = {"preserved": [], "review": [], "false_stale": [], "true_stale": [], "silent_wrong": []}
    total = 0
    for path, regions in old.items():
        same_path = new_by_path_ord.get(path, {})
        for region in regions:
            total += 1
            valid = oracle.get(region.ref, False)
            candidate = same_path.get(region.ordinal)
            if candidate and candidate.fingerprint == region.fingerprint:
                counts["preserved"] += 1
                if len(examples["preserved"]) < 6:
                    examples["preserved"].append({"old_ref": region.ref, "new_ref": candidate.ref, "via": "same_chunk_hash"})
                continue
            quote = quote_selector(region.text)
            quote_hits = quote_index.get(quote, []) if quote else []
            if len(quote_hits) == 1:
                hit = quote_hits[0]
                if valid:
                    counts["preserved"] += 1
                    if len(examples["preserved"]) < 6:
                        examples["preserved"].append({"old_ref": region.ref, "new_ref": hit.ref, "via": "quote_selector"})
                else:
                    counts["silent_wrong"] += 1
                    if len(examples["silent_wrong"]) < 6:
                        examples["silent_wrong"].append({"old_ref": region.ref, "new_ref": hit.ref})
            elif len(quote_hits) > 1:
                counts["review_needed"] += 1
                if len(examples["review"]) < 6:
                    examples["review"].append({"old_ref": region.ref, "quote_hits": len(quote_hits)})
            elif valid:
                counts["false_stale_alert"] += 1
                if len(examples["false_stale"]) < 6:
                    examples["false_stale"].append({"old_ref": region.ref})
            else:
                counts["true_stale_alert"] += 1
                if len(examples["true_stale"]) < 6:
                    examples["true_stale"].append({"old_ref": region.ref})
    return rates(total, counts, examples)


def evaluate_layered_anchor_selector(
    old: dict[str, list[Region]],
    new: dict[str, list[Region]],
    stable_report: dict[str, object],
    *,
    similarity_threshold: float = 0.82,
    same_ordinal_rewrite_threshold: float = 0.95,
) -> dict[str, object]:
    """Layered deterministic anchoring that borrows quote-selector tricks.

    This is the vNext-style safety baseline: preserve exact same path/ordinal
    hashes first, then accept a unique quote hit only when the whole region is
    still similar enough. A conservative same-path/same-ordinal rewrite gate
    catches common documentation updates when the quote still points at the
    same local chunk. Low-confidence, ambiguous, and split-looking cases are
    review-needed rather than silently accepted.
    """

    oracle = _oracle_validity(stable_report)
    all_new = [region for regions in new.values() for region in regions]
    new_by_path_ord = {
        path: {region.ordinal: region for region in regions}
        for path, regions in new.items()
    }
    quote_index = _quote_index(all_new)
    counts = Counter()
    examples = {"preserved": [], "review": [], "true_stale": [], "silent_wrong": []}
    total = 0
    for path, regions in old.items():
        same_path = new_by_path_ord.get(path, {})
        for region in regions:
            total += 1
            valid = oracle.get(region.ref, False)
            candidate = same_path.get(region.ordinal)
            if candidate and candidate.fingerprint == region.fingerprint:
                counts["preserved"] += 1
                if len(examples["preserved"]) < 6:
                    examples["preserved"].append({"old_ref": region.ref, "new_ref": candidate.ref, "via": "same_chunk_hash"})
                continue

            quote = quote_selector(region.text)
            quote_hits = quote_index.get(quote, []) if quote else []
            if candidate:
                same_ordinal_similarity = token_jaccard(region.text, candidate.text)
                short_quote = quote_selector(region.text, tokens=8)
                same_ordinal_quote_match = bool(short_quote and quote_selector(candidate.text, tokens=8) == short_quote)
                if same_ordinal_quote_match and same_ordinal_similarity >= same_ordinal_rewrite_threshold:
                    if valid:
                        counts["preserved"] += 1
                        if len(examples["preserved"]) < 6:
                            examples["preserved"].append(
                                {
                                    "old_ref": region.ref,
                                    "new_ref": candidate.ref,
                                    "via": "same_ordinal_quote_rewrite",
                                    "similarity": round(same_ordinal_similarity, 3),
                                }
                            )
                    else:
                        counts["silent_wrong"] += 1
                        if len(examples["silent_wrong"]) < 6:
                            examples["silent_wrong"].append(
                                {
                                    "old_ref": region.ref,
                                    "new_ref": candidate.ref,
                                    "via": "same_ordinal_quote_rewrite",
                                    "similarity": round(same_ordinal_similarity, 3),
                                }
                            )
                    continue
            if len(quote_hits) == 1:
                hit = quote_hits[0]
                similarity = token_jaccard(region.text, hit.text)
                if similarity >= similarity_threshold:
                    if valid:
                        counts["preserved"] += 1
                        if len(examples["preserved"]) < 6:
                            examples["preserved"].append(
                                {
                                    "old_ref": region.ref,
                                    "new_ref": hit.ref,
                                    "via": "quote_selector_similarity",
                                    "similarity": round(similarity, 3),
                                }
                            )
                    else:
                        counts["silent_wrong"] += 1
                        if len(examples["silent_wrong"]) < 6:
                            examples["silent_wrong"].append({"old_ref": region.ref, "new_ref": hit.ref, "similarity": round(similarity, 3)})
                elif valid:
                    counts["review_needed"] += 1
                    if len(examples["review"]) < 6:
                        examples["review"].append(
                            {
                                "old_ref": region.ref,
                                "candidate_ref": hit.ref,
                                "reason": "quote_hit_below_similarity_threshold",
                                "similarity": round(similarity, 3),
                            }
                        )
                else:
                    counts["true_stale_alert"] += 1
                    if len(examples["true_stale"]) < 6:
                        examples["true_stale"].append({"old_ref": region.ref})
            elif valid:
                counts["review_needed"] += 1
                if len(examples["review"]) < 6:
                    reason = "ambiguous_quote_selector" if len(quote_hits) > 1 else "no_quote_selector_hit"
                    examples["review"].append({"old_ref": region.ref, "reason": reason, "quote_hits": len(quote_hits)})
            else:
                counts["true_stale_alert"] += 1
                if len(examples["true_stale"]) < 6:
                    examples["true_stale"].append({"old_ref": region.ref})
    return rates(total, counts, examples)


def evaluate_eval_label_lifecycle(
    stable_report: dict[str, object],
    naive_report: dict[str, object],
    *,
    chunk_hash: dict[str, object] | None = None,
    source_hash: dict[str, object] | None = None,
    quote_selector: dict[str, object] | None = None,
    layered_selector: dict[str, object] | None = None,
) -> dict[str, object]:
    total = int(stable_report["total"])
    stable_counts = Counter(stable_report["counts"])
    naive_counts = Counter(naive_report["counts"])
    preserved = stable_counts["same_file_exact"] + stable_counts["moved_exact"]
    review_needed = stable_counts["fuzzy"] + stable_counts["split_support"]
    stale = stable_counts["stale"]
    naive_correct = naive_counts["correct"]
    naive_silent_wrong = naive_counts["wrong_same_id"]
    naive_missing = naive_counts["missing"]
    return {
        "total_labels": total,
        "refmark": {
            "auto_preserved_or_migrated": preserved,
            "review_needed_fuzzy": review_needed,
            "stale_or_deleted": stale,
            "rates": {
                "auto_preserved_or_migrated": round(preserved / max(total, 1), 4),
                "review_needed_fuzzy": round(review_needed / max(total, 1), 4),
                "stale_or_deleted": round(stale / max(total, 1), 4),
            },
        },
        "naive": {
            "correct": naive_correct,
            "silent_wrong": naive_silent_wrong,
            "missing": naive_missing,
            "rates": {
                "correct": round(naive_correct / max(total, 1), 4),
                "silent_wrong": round(naive_silent_wrong / max(total, 1), 4),
                "missing": round(naive_missing / max(total, 1), 4),
            },
        },
        "estimated_review_workload": {
            "refmark_review_or_refresh": review_needed + stale,
            "naive_unknown_requires_audit": total - naive_missing,
            "review_workload_reduction_vs_audit_existing_labels": round(1.0 - ((review_needed + stale) / max(total - naive_missing, 1)), 4),
        },
        "method_comparison": {
            "chunk_id_only": _method_comparison_row(
                total=total,
                silent_drift=naive_silent_wrong,
                false_stale=0,
                review=0,
                preserved=naive_correct,
            ),
            "chunk_id_content_hash": _baseline_method_row(total, chunk_hash or {}),
            "qrels_source_hash": _baseline_method_row(total, source_hash or {}),
            "chunk_hash_quote_selector": _baseline_method_row(total, quote_selector or {}),
            "refmark_layered_selector": _baseline_method_row(total, layered_selector or {}),
            "refmark": _method_comparison_row(
                total=total,
                silent_drift=0,
                false_stale=0,
                review=review_needed,
                preserved=preserved,
                stale=stale,
            ),
        },
    }


def evaluate_file_lifecycle(old: dict[str, list[Region]], new: dict[str, list[Region]]) -> dict[str, int]:
    old_paths = set(old)
    new_paths = set(new)
    return {
        "unchanged_paths": len(old_paths & new_paths),
        "removed_paths": len(old_paths - new_paths),
        "added_paths": len(new_paths - old_paths),
    }


def _oracle_validity(stable_report: dict[str, object]) -> dict[str, bool]:
    statuses = dict(stable_report.get("status_by_ref", {}))
    return {ref: status != "stale" for ref, status in statuses.items()}


def split_support_match(
    region: Region,
    candidates: list[Region],
    *,
    min_combined_coverage: float = 0.82,
    max_regions: int = 3,
) -> tuple[list[Region], float] | None:
    query_tokens = token_set(region.text)
    if not query_tokens:
        return None
    ranked: list[tuple[Region, int]] = []
    for candidate in candidates:
        overlap = len(query_tokens & token_set(candidate.text))
        if overlap:
            ranked.append((candidate, overlap))
    ranked.sort(key=lambda item: item[1], reverse=True)
    selected = [candidate for candidate, _overlap in ranked[:max_regions]]
    if len(selected) < 2:
        return None
    combined: set[str] = set()
    for candidate in selected:
        combined.update(token_set(candidate.text))
    coverage = len(query_tokens & combined) / max(len(query_tokens), 1)
    if coverage >= min_combined_coverage:
        return selected, coverage
    return None


def _file_fingerprints(regions_by_path: dict[str, list[Region]]) -> dict[str, str]:
    return {
        path: hashlib.sha256("\n".join(region.fingerprint for region in regions).encode("utf-8")).hexdigest()[:16]
        for path, regions in regions_by_path.items()
    }


def quote_selector(text: str, *, tokens: int = 20) -> str:
    normalized = [token.lower() for token in TOKEN_RE.findall(text)]
    if not normalized:
        return ""
    return " ".join(normalized[: min(tokens, len(normalized))])


def _quote_index(regions: list[Region]) -> dict[str, list[Region]]:
    index: dict[str, list[Region]] = {}
    for region in regions:
        quote = quote_selector(region.text)
        if quote:
            index.setdefault(quote, []).append(region)
    return index


def best_fuzzy_match(region: Region, candidates: list[Region]) -> tuple[Region, float] | None:
    best = None
    for candidate in candidates:
        sim = token_jaccard(region.text, candidate.text)
        if best is None or sim > best[1]:
            best = (candidate, sim)
    return best


class FuzzyRegionIndex:
    """Token index for high-threshold region similarity checks.

    The lifecycle benchmark uses Jaccard >= 0.82 as a conservative fuzzy
    migration signal. A valid candidate must share at least one token and have a
    roughly compatible token-set size, so we can avoid comparing every old
    region against every new region on large corpora.
    """

    def __init__(self, regions: list[Region], *, threshold: float = 0.82) -> None:
        self.threshold = threshold
        self.tokens_by_region = {region: token_set(region.text) for region in regions}
        self.by_token: dict[str, list[Region]] = {}
        for region, tokens in self.tokens_by_region.items():
            for token in tokens:
                self.by_token.setdefault(token, []).append(region)

    def best_match(self, region: Region, *, candidates: list[Region] | None = None) -> tuple[Region, float] | None:
        query_tokens = token_set(region.text)
        if candidates is None:
            candidate_pool = self._candidate_pool(query_tokens)
        else:
            candidate_pool = candidates
        if not query_tokens and not candidate_pool:
            return None
        query_size = len(query_tokens)
        min_size = int(query_size * self.threshold)
        max_size = int(query_size / self.threshold) + 1 if self.threshold else query_size
        best: tuple[Region, float] | None = None
        for candidate in candidate_pool:
            candidate_tokens = self.tokens_by_region.get(candidate)
            if candidate_tokens is None:
                candidate_tokens = token_set(candidate.text)
            size = len(candidate_tokens)
            if query_tokens and (size < min_size or size > max_size):
                continue
            sim = token_set_jaccard(query_tokens, candidate_tokens)
            if best is None or sim > best[1]:
                best = (candidate, sim)
        return best

    def _candidate_pool(self, query_tokens: set[str]) -> list[Region]:
        if not query_tokens:
            return list(self.tokens_by_region)
        seen: set[Region] = set()
        pool: list[Region] = []
        for token in query_tokens:
            for region in self.by_token.get(token, []):
                if region not in seen:
                    seen.add(region)
                    pool.append(region)
        return pool


def _best_match(*matches: tuple[Region, float] | None) -> tuple[Region, float] | None:
    candidates = [match for match in matches if match is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda match: match[1])


def _method_comparison_row(
    *,
    total: int,
    silent_drift: int,
    false_stale: int,
    review: int,
    preserved: int,
    stale: int = 0,
) -> dict[str, object]:
    return {
        "silent_drift": silent_drift,
        "false_stale_alerts": false_stale,
        "human_review_workload": review + stale + false_stale,
        "valid_evals_preserved": preserved,
        "rates": {
            "silent_drift": round(silent_drift / max(total, 1), 4),
            "false_stale_alerts": round(false_stale / max(total, 1), 4),
            "human_review_workload": round((review + stale + false_stale) / max(total, 1), 4),
            "valid_evals_preserved": round(preserved / max(total, 1), 4),
        },
    }


def _baseline_method_row(total: int, report: dict[str, object]) -> dict[str, object]:
    counts = Counter(report.get("counts", {}))
    return _method_comparison_row(
        total=total,
        silent_drift=counts["silent_wrong"],
        false_stale=counts["false_stale_alert"],
        review=counts["review_needed"],
        stale=counts["true_stale_alert"],
        preserved=counts["preserved"],
    )


def rates(total: int, counts: Counter, examples: dict[str, list]) -> dict[str, object]:
    return {
        "total": total,
        "counts": dict(counts),
        "rates": {key: round(value / max(total, 1), 4) for key, value in counts.items()},
        "examples": examples,
    }


def revision_summary_rows(
    repo_url: str,
    old_ref: str,
    subdir: str,
    revision_reports: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for report in revision_reports:
        stable = dict(report["stable_ref_migration"])
        naive = dict(report["naive_path_ordinal_identity"])
        lifecycle = dict(report["eval_label_lifecycle"])
        refmark = dict(lifecycle["refmark"])
        naive_lifecycle = dict(lifecycle["naive"])
        workload = dict(lifecycle["estimated_review_workload"])
        method_comparison = dict(lifecycle.get("method_comparison", {}))
        competent = dict(method_comparison.get("chunk_hash_quote_selector", {}))
        competent_rates = dict(competent.get("rates", {}))
        layered = dict(method_comparison.get("refmark_layered_selector", {}))
        layered_rates = dict(layered.get("rates", {}))
        rows.append(
            {
                "repo_url": repo_url,
                "subdir": subdir,
                "old_ref": old_ref,
                "new_ref": report["new_ref"],
                "old_labels": lifecycle["total_labels"],
                "new_regions": report["new_regions"],
                "new_tokens": report["new_tokens"],
                "competent_silent_drift_rate": competent_rates.get("silent_drift", 0.0),
                "competent_false_stale_rate": competent_rates.get("false_stale_alerts", 0.0),
                "competent_review_rate": competent_rates.get("human_review_workload", 0.0),
                "competent_preserved_rate": competent_rates.get("valid_evals_preserved", 0.0),
                "layered_silent_drift_rate": layered_rates.get("silent_drift", 0.0),
                "layered_review_rate": layered_rates.get("human_review_workload", 0.0),
                "layered_preserved_rate": layered_rates.get("valid_evals_preserved", 0.0),
                "refmark_auto_rate": _rate(refmark, "auto_preserved_or_migrated"),
                "refmark_review_rate": _rate(refmark, "review_needed_fuzzy"),
                "refmark_stale_rate": _rate(refmark, "stale_or_deleted"),
                "naive_correct_rate": _rate(naive_lifecycle, "correct"),
                "naive_silent_wrong_rate": _rate(naive_lifecycle, "silent_wrong"),
                "naive_missing_rate": _rate(naive_lifecycle, "missing"),
                "stable_same_file_exact_rate": _rate(stable, "same_file_exact"),
                "stable_moved_exact_rate": _rate(stable, "moved_exact"),
                "stable_fuzzy_rate": _rate(stable, "fuzzy"),
                "stable_stale_rate": _rate(stable, "stale"),
                "workload_reduction_vs_audit": dict(workload).get(
                    "review_workload_reduction_vs_audit_existing_labels",
                    0.0,
                ),
            }
        )
    return rows


def _rate(section: dict[str, object], key: str) -> float:
    return float(dict(section.get("rates", {})).get(key, 0.0))


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_summary_rows(paths: list[str | Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path_like in paths:
        path = Path(path_like)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows.extend(dict(row) for row in payload)
        elif "summary_rows" in payload:
            rows.extend(dict(row) for row in payload["summary_rows"])
        else:
            raise ValueError(f"{path} does not contain summary_rows")
    return rows


def render_summary_rows(
    rows: list[dict[str, object]],
    *,
    columns: list[str] | None = None,
    output_format: str = "markdown",
) -> str:
    columns = columns or DEFAULT_SUMMARY_COLUMNS
    if output_format == "json":
        return json.dumps([{column: row.get(column, "") for column in columns} for row in rows], indent=2)
    if output_format == "csv":
        from io import StringIO

        out = StringIO()
        writer = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in columns} for row in rows)
        return out.getvalue()
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(_summary_cell(row.get(column, ""), column) for column in columns) + " |" for row in rows]
    return "\n".join([header, divider, *body]) + "\n"


def _summary_cell(value: object, column: str) -> str:
    if isinstance(value, float):
        return f"{value:.1%}" if ("rate" in column or "reduction" in column) else f"{value:.3f}"
    return str(value).replace("|", "\\|")


if __name__ == "__main__":
    main()
