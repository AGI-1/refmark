from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from refmark import (
    CorpusMap,
    EvalExample,
    EvalSuite,
    RegionRecord,
    export_ragas_rows,
    refmark_evidence_metrics,
)


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    original = _original_corpus()
    mutated = _mutated_corpus()

    original_suite = EvalSuite(
        examples=[
            EvalExample(
                "What is the refund window?",
                ["policy:P01"],
                metadata={"query_style": "direct"},
            ).with_source_hashes(original),
            EvalExample(
                "Which clause covers expedited shipping refunds?",
                ["policy:P02"],
                metadata={"query_style": "direct"},
            ).with_source_hashes(original),
        ],
        corpus=original,
    )
    saved_rows = [example.to_dict() for example in original_suite.examples]
    (OUTPUT / "saved_eval_rows.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in saved_rows) + "\n",
        encoding="utf-8",
    )

    current_suite = EvalSuite.from_rows(saved_rows, corpus=mutated)
    run = current_suite.evaluate(_mutated_retriever, name="mutated_demo_retriever", k=1)

    answers = {
        "What is the refund window?": "The refund window is 14 days.",
        "Which clause covers expedited shipping refunds?": "Expedited shipping refunds are handled by the carrier policy.",
    }
    plain_ragas_rows = export_ragas_rows(current_suite, run, answers=answers, include_refmark_fields=False)
    refmark_ragas_rows = export_ragas_rows(current_suite, run, answers=answers, include_refmark_fields=True)
    stale = [item.to_dict() for item in current_suite.stale_examples()]
    metrics = refmark_evidence_metrics(current_suite, run)

    _write_jsonl(OUTPUT / "ragas_plain_rows.jsonl", plain_ragas_rows)
    _write_jsonl(OUTPUT / "ragas_with_refmark_rows.jsonl", refmark_ragas_rows)
    _write_json(OUTPUT / "refmark_evidence_metrics.json", metrics)

    optional_ragas = _optional_ragas_native_dataset(refmark_ragas_rows)
    summary = {
        "schema": "refmark.ragas_mutation_demo.v1",
        "old_corpus_fingerprint": original.fingerprint,
        "new_corpus_fingerprint": mutated.fingerprint,
        "ragas_without_refmark": {
            "rows": len(plain_ragas_rows),
            "fields": sorted(plain_ragas_rows[0]),
            "limitation": "Rows contain resolved strings but no old source hashes or stale-label signal.",
        },
        "ragas_with_refmark": {
            "rows": len(refmark_ragas_rows),
            "fields": sorted(refmark_ragas_rows[0]),
            "stale_example_count": len(stale),
            "changed_refs": sorted({ref for item in stale for ref in item["changed_refs"]}),
            "missing_refs": sorted({ref for item in stale for ref in item["missing_refs"]}),
            "metrics": metrics,
        },
        "optional_ragas_native_dataset": optional_ragas,
        "interpretation": [
            "The plain Ragas-style rows are self-consistent after mutation because references are resolved from the current corpus.",
            "The Refmark-enriched rows preserve the old source hashes, so the same eval labels are marked stale before answer-quality scoring is trusted.",
            "This is the integration shape: Ragas can still score answers/contexts, while Refmark contributes evidence identity, corpus fingerprints, and label lifecycle checks.",
        ],
    }
    _write_json(OUTPUT / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _original_corpus() -> CorpusMap:
    return CorpusMap.from_records(
        [
            _record("P01", "Refunds are available within 30 days.", 1),
            _record("P02", "Expedited shipping is non-refundable after dispatch.", 2),
            _record("P03", "Audit logs are retained for 180 days.", 3),
        ],
        revision_id="policy-v1",
        metadata={"source": "ragas_refmark_mutation_demo"},
    )


def _mutated_corpus() -> CorpusMap:
    return CorpusMap.from_records(
        [
            _record("P01", "Refunds are available within 14 days.", 1),
            _record("P02", "Expedited shipping refunds are handled by the carrier policy.", 2),
            _record("P03", "Audit logs are retained for 180 days.", 3),
        ],
        revision_id="policy-v2",
        metadata={"source": "ragas_refmark_mutation_demo"},
    )


def _mutated_retriever(query: str) -> list[dict[str, Any]]:
    if "shipping" in query:
        return [{"stable_ref": "policy:P02", "score": 1.0}]
    return [{"stable_ref": "policy:P01", "score": 1.0}]


def _record(region_id: str, text: str, ordinal: int) -> RegionRecord:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return RegionRecord(
        doc_id="policy",
        region_id=region_id,
        text=text,
        start_line=ordinal,
        end_line=ordinal,
        ordinal=ordinal,
        hash=digest,
    )


def _optional_ragas_native_dataset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    except Exception as exc:
        return {"status": "not_installed", "error": f"{type(exc).__name__}: {exc}"}
    samples = [
        SingleTurnSample(
            user_input=row["user_input"],
            response=row["response"],
            retrieved_contexts=row["retrieved_contexts"],
            reference=row["reference"],
            retrieved_context_ids=row["refmark"]["context_refs"],
            reference_context_ids=row["refmark"]["gold_refs"],
        )
        for row in rows
    ]
    dataset = EvaluationDataset(samples=samples, name="refmark-ragas-mutation-demo")
    return {
        "status": "ok",
        "native_class": type(dataset).__name__,
        "sample_count": len(dataset.samples),
        "refs_preserved": dataset.samples[0].reference_context_ids == rows[0]["refmark"]["gold_refs"],
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
