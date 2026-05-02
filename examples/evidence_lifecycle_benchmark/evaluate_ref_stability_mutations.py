from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ir_datasets
from refmark.search_index import approx_tokens

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class Region:
    doc_id: str
    ref: str
    ordinal: int
    text: str
    fingerprint: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled ref stability benchmark under corpus mutations.")
    parser.add_argument("--dataset", default="beir/scifact/test")
    parser.add_argument("--output", default="examples/external_qa_benchmark/output/ref_stability_scifact.json")
    parser.add_argument("--summary-output", default="", help="Optional compact JSON summary row output path.")
    parser.add_argument("--csv-output", default="", help="Optional compact CSV summary row output path.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--region-tokens", type=int, default=110)
    parser.add_argument("--region-stride", type=int, default=110)
    parser.add_argument("--insert-rate", type=float, default=0.2)
    parser.add_argument("--delete-rate", type=float, default=0.1)
    parser.add_argument("--edit-rate", type=float, default=0.1)
    parser.add_argument("--max-docs", type=int, default=0)
    args = parser.parse_args()

    started = time.perf_counter()
    rng = random.Random(args.seed)
    ds = ir_datasets.load(args.dataset)
    docs = []
    for doc in ds.docs_iter():
        docs.append((str(doc.doc_id), str(getattr(doc, "title", "") or ""), str(getattr(doc, "text", "") or "")))
        if args.max_docs and len(docs) >= args.max_docs:
            break

    base_regions_by_doc: dict[str, list[Region]] = {}
    mutated_regions_by_doc: dict[str, list[Region]] = {}
    mutation_counts: Counter[str] = Counter()
    for doc_id, title, text in docs:
        base_regions = make_regions(doc_id, title, text, size=args.region_tokens, stride=args.region_stride)
        mutated_region_texts, counts = mutate_region_sequence([r.text for r in base_regions], rng, insert_rate=args.insert_rate, delete_rate=args.delete_rate, edit_rate=args.edit_rate)
        mutation_counts.update(counts)
        mutated_regions = [
            make_region(doc_id, idx + 1, region_text)
            for idx, region_text in enumerate(mutated_region_texts)
        ]
        base_regions_by_doc[doc_id] = base_regions
        mutated_regions_by_doc[doc_id] = mutated_regions

    stable_report = evaluate_stable_ref_migration(base_regions_by_doc, mutated_regions_by_doc)
    naive_report = evaluate_naive_chunk_identity(base_regions_by_doc, mutated_regions_by_doc)
    content_report = evaluate_content_retrieval(base_regions_by_doc, mutated_regions_by_doc)

    payload = {
        "schema": "refmark.ref_stability_mutation.v1",
        "dataset": args.dataset,
        "settings": vars(args),
        "documents": len(docs),
        "base_regions": sum(len(v) for v in base_regions_by_doc.values()),
        "mutated_regions": sum(len(v) for v in mutated_regions_by_doc.values()),
        "source_tokens": sum(approx_tokens(title + "\n" + text) for _doc_id, title, text in docs),
        "mutation_counts": dict(mutation_counts),
        "stable_ref_migration": stable_report,
        "naive_chunk_identity": naive_report,
        "content_retrieval": content_report,
        "summary_rows": [
            mutation_summary_row(
                args.dataset,
                docs=len(docs),
                source_tokens=sum(approx_tokens(title + "\n" + text) for _doc_id, title, text in docs),
                base_regions=sum(len(v) for v in base_regions_by_doc.values()),
                mutated_regions=sum(len(v) for v in mutated_regions_by_doc.values()),
                stable_report=stable_report,
                naive_report=naive_report,
                content_report=content_report,
            )
        ],
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "interpretation": [
            "stable_ref_migration matches base refs to mutated regions by content fingerprints within the same parent document.",
            "naive_chunk_identity treats same ordinal chunk id as the persisted address after mutation.",
            "content_retrieval asks whether the same text can be found anywhere in the mutated document even if the ordinal changed.",
            "This is a controlled mutation test, not a natural document revision benchmark.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.summary_output:
        summary_output = Path(args.summary_output)
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(json.dumps(payload["summary_rows"], indent=2), encoding="utf-8")
    if args.csv_output:
        write_summary_csv(Path(args.csv_output), payload["summary_rows"])
    print(json.dumps(payload, indent=2))


def make_regions(doc_id: str, title: str, text: str, *, size: int, stride: int) -> list[Region]:
    words = f"{title}\n{text}".strip().split()
    if not words:
        return []
    regions = []
    start = 0
    index = 1
    while start < len(words):
        chunk = " ".join(words[start : start + size])
        regions.append(make_region(doc_id, index, chunk))
        if start + size >= len(words):
            break
        start += max(1, stride)
        index += 1
    return regions


def make_region(doc_id: str, index: int, text: str) -> Region:
    return Region(doc_id=doc_id, ref=f"{doc_id}:P{index:03d}", ordinal=index, text=text, fingerprint=fingerprint(text))


def mutate_region_sequence(regions: list[str], rng: random.Random, *, insert_rate: float, delete_rate: float, edit_rate: float) -> tuple[list[str], Counter[str]]:
    out = []
    counts: Counter[str] = Counter()
    for text in regions:
        if rng.random() < insert_rate:
            out.append(synthetic_insert_text(text, rng))
            counts["inserted_regions"] += 1
        if rng.random() < delete_rate:
            counts["deleted_regions"] += 1
            continue
        if rng.random() < edit_rate:
            out.append(light_edit(text, rng))
            counts["edited_regions"] += 1
        else:
            out.append(text)
            counts["unchanged_regions"] += 1
    return out, counts


def synthetic_insert_text(neighbor: str, rng: random.Random) -> str:
    words = TOKEN_RE.findall(neighbor.lower())
    sample = " ".join(words[: min(len(words), 12)])
    return f"Inserted update note {rng.randint(1000, 9999)}. This new paragraph contextualizes {sample}."


def light_edit(text: str, rng: random.Random) -> str:
    words = text.split()
    if len(words) < 8:
        return text + " Updated."
    pos = rng.randrange(0, len(words))
    words[pos] = words[pos] + "_updated"
    if len(words) > 20:
        del words[rng.randrange(0, len(words))]
    return " ".join(words)


def fingerprint(text: str) -> str:
    tokens = [token.lower() for token in TOKEN_RE.findall(text)]
    return hashlib.sha256(" ".join(tokens).encode("utf-8")).hexdigest()[:16]


def token_jaccard(a: str, b: str) -> float:
    aa = set(token.lower() for token in TOKEN_RE.findall(a))
    bb = set(token.lower() for token in TOKEN_RE.findall(b))
    if not aa and not bb:
        return 1.0
    return len(aa & bb) / max(len(aa | bb), 1)


def evaluate_stable_ref_migration(base: dict[str, list[Region]], mutated: dict[str, list[Region]]) -> dict[str, object]:
    total = exact = fuzzy = stale = 0
    examples = {"exact_after_shift": [], "fuzzy": [], "stale": []}
    for doc_id, regions in base.items():
        by_fp = {r.fingerprint: r for r in mutated.get(doc_id, [])}
        mutated_list = mutated.get(doc_id, [])
        for region in regions:
            total += 1
            matched = by_fp.get(region.fingerprint)
            if matched:
                exact += 1
                if matched.ordinal != region.ordinal and len(examples["exact_after_shift"]) < 6:
                    examples["exact_after_shift"].append({"base_ref": region.ref, "new_ref": matched.ref})
                continue
            best = best_fuzzy_match(region, mutated_list)
            if best and best[1] >= 0.82:
                fuzzy += 1
                if len(examples["fuzzy"]) < 6:
                    examples["fuzzy"].append({"base_ref": region.ref, "new_ref": best[0].ref, "similarity": round(best[1], 3)})
            else:
                stale += 1
                if len(examples["stale"]) < 6:
                    examples["stale"].append({"base_ref": region.ref, "best_similarity": round(best[1], 3) if best else 0.0})
    return rates(total, {"exact": exact, "fuzzy": fuzzy, "stale": stale}, examples)


def evaluate_naive_chunk_identity(base: dict[str, list[Region]], mutated: dict[str, list[Region]]) -> dict[str, object]:
    total = correct = wrong_same_id = missing = 0
    examples = {"wrong_same_id": [], "missing": []}
    for doc_id, regions in base.items():
        by_ordinal = {r.ordinal: r for r in mutated.get(doc_id, [])}
        for region in regions:
            total += 1
            candidate = by_ordinal.get(region.ordinal)
            if candidate is None:
                missing += 1
                if len(examples["missing"]) < 6:
                    examples["missing"].append({"base_ref": region.ref})
            elif candidate.fingerprint == region.fingerprint or token_jaccard(candidate.text, region.text) >= 0.82:
                correct += 1
            else:
                wrong_same_id += 1
                if len(examples["wrong_same_id"]) < 6:
                    examples["wrong_same_id"].append({"base_ref": region.ref, "same_ordinal_ref": candidate.ref, "similarity": round(token_jaccard(candidate.text, region.text), 3)})
    return rates(total, {"correct": correct, "wrong_same_id": wrong_same_id, "missing": missing}, examples)


def evaluate_content_retrieval(base: dict[str, list[Region]], mutated: dict[str, list[Region]]) -> dict[str, object]:
    total = found_exact = found_fuzzy = missing = 0
    for doc_id, regions in base.items():
        mutated_list = mutated.get(doc_id, [])
        fps = {r.fingerprint for r in mutated_list}
        for region in regions:
            total += 1
            if region.fingerprint in fps:
                found_exact += 1
            else:
                best = best_fuzzy_match(region, mutated_list)
                if best and best[1] >= 0.82:
                    found_fuzzy += 1
                else:
                    missing += 1
    return rates(total, {"found_exact": found_exact, "found_fuzzy": found_fuzzy, "missing": missing}, {})


def best_fuzzy_match(region: Region, candidates: list[Region]) -> tuple[Region, float] | None:
    best = None
    for candidate in candidates:
        sim = token_jaccard(region.text, candidate.text)
        if best is None or sim > best[1]:
            best = (candidate, sim)
    return best


def rates(total: int, counts: dict[str, int], examples: dict[str, list]) -> dict[str, object]:
    return {
        "total": total,
        "counts": counts,
        "rates": {key: round(value / max(total, 1), 4) for key, value in counts.items()},
        "examples": examples,
    }


def mutation_summary_row(
    dataset: str,
    *,
    docs: int,
    source_tokens: int,
    base_regions: int,
    mutated_regions: int,
    stable_report: dict[str, object],
    naive_report: dict[str, object],
    content_report: dict[str, object],
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "documents": docs,
        "source_tokens": source_tokens,
        "base_regions": base_regions,
        "mutated_regions": mutated_regions,
        "stable_exact_rate": _rate(stable_report, "exact"),
        "stable_fuzzy_rate": _rate(stable_report, "fuzzy"),
        "stable_stale_rate": _rate(stable_report, "stale"),
        "naive_correct_rate": _rate(naive_report, "correct"),
        "naive_silent_wrong_rate": _rate(naive_report, "wrong_same_id"),
        "naive_missing_rate": _rate(naive_report, "missing"),
        "content_found_exact_rate": _rate(content_report, "found_exact"),
        "content_found_fuzzy_rate": _rate(content_report, "found_fuzzy"),
        "content_missing_rate": _rate(content_report, "missing"),
    }


def _rate(report: dict[str, object], key: str) -> float:
    return float(dict(report.get("rates", {})).get(key, 0.0))


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
