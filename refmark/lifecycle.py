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

DEFAULT_SUMMARY_COLUMNS = [
    "repo_url",
    "old_ref",
    "new_ref",
    "old_labels",
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
        file_report = evaluate_file_lifecycle(old_regions, new_regions)
        eval_lifecycle = evaluate_eval_label_lifecycle(stable, naive)
        revision_reports.append(
            {
                "new_ref": new_ref,
                "new_files": len(new_regions),
                "new_regions": sum(len(v) for v in new_regions.values()),
                "new_tokens": sum(approx_tokens(r.text) for regions in new_regions.values() for r in regions),
                "file_lifecycle": file_report,
                "stable_ref_migration": stable,
                "naive_path_ordinal_identity": naive,
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
        "eval_label_lifecycle": revision_reports[-1]["eval_label_lifecycle"] if revision_reports else {},
        "revision_reports": revision_reports,
        "summary_rows": revision_summary_rows(repo_url, old_ref, subdir, revision_reports),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "interpretation": [
            "stable_ref_migration matches old path-local refs to new regions by fingerprint/fuzzy text, first in the same file and then across renamed/moved files.",
            "naive_path_ordinal_identity assumes path plus ordinal region id remains valid after the revision.",
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
    if not aa and not bb:
        return 1.0
    return len(aa & bb) / max(len(aa | bb), 1)


def evaluate_stable_migration(old: dict[str, list[Region]], new: dict[str, list[Region]]) -> dict[str, object]:
    all_new = [region for regions in new.values() for region in regions]
    new_by_path_fp = {
        path: {region.fingerprint: region for region in regions}
        for path, regions in new.items()
    }
    new_by_fp: dict[str, list[Region]] = {}
    for region in all_new:
        new_by_fp.setdefault(region.fingerprint, []).append(region)
    counts = Counter()
    examples = {"same_file_exact": [], "moved_exact": [], "fuzzy": [], "stale": []}
    total = 0
    for path, regions in old.items():
        same_file_fps = new_by_path_fp.get(path, {})
        same_file_regions = new.get(path, [])
        for region in regions:
            total += 1
            same = same_file_fps.get(region.fingerprint)
            if same:
                counts["same_file_exact"] += 1
                if same.ordinal != region.ordinal and len(examples["same_file_exact"]) < 6:
                    examples["same_file_exact"].append({"old_ref": region.ref, "new_ref": same.ref})
                continue
            moved = [candidate for candidate in new_by_fp.get(region.fingerprint, []) if candidate.path != path]
            if moved:
                counts["moved_exact"] += 1
                if len(examples["moved_exact"]) < 6:
                    examples["moved_exact"].append({"old_ref": region.ref, "new_ref": moved[0].ref})
                continue
            best = best_fuzzy_match(region, same_file_regions) or best_fuzzy_match(region, all_new)
            if best and best[1] >= 0.82:
                counts["fuzzy"] += 1
                if len(examples["fuzzy"]) < 6:
                    examples["fuzzy"].append({"old_ref": region.ref, "new_ref": best[0].ref, "similarity": round(best[1], 3)})
            else:
                counts["stale"] += 1
                if len(examples["stale"]) < 6:
                    examples["stale"].append({"old_ref": region.ref, "best_similarity": round(best[1], 3) if best else 0.0})
    return rates(total, counts, examples)


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


def evaluate_eval_label_lifecycle(stable_report: dict[str, object], naive_report: dict[str, object]) -> dict[str, object]:
    total = int(stable_report["total"])
    stable_counts = Counter(stable_report["counts"])
    naive_counts = Counter(naive_report["counts"])
    preserved = stable_counts["same_file_exact"] + stable_counts["moved_exact"]
    review_needed = stable_counts["fuzzy"]
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
    }


def evaluate_file_lifecycle(old: dict[str, list[Region]], new: dict[str, list[Region]]) -> dict[str, int]:
    old_paths = set(old)
    new_paths = set(new)
    return {
        "unchanged_paths": len(old_paths & new_paths),
        "removed_paths": len(old_paths - new_paths),
        "added_paths": len(new_paths - old_paths),
    }


def best_fuzzy_match(region: Region, candidates: list[Region]) -> tuple[Region, float] | None:
    best = None
    for candidate in candidates:
        sim = token_jaccard(region.text, candidate.text)
        if best is None or sim > best[1]:
            best = (candidate, sim)
    return best


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
        rows.append(
            {
                "repo_url": repo_url,
                "subdir": subdir,
                "old_ref": old_ref,
                "new_ref": report["new_ref"],
                "old_labels": lifecycle["total_labels"],
                "new_regions": report["new_regions"],
                "new_tokens": report["new_tokens"],
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
