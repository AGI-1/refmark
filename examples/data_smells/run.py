from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))

from refmark.adapt_plan import build_adaptation_plan
from refmark.data_smells import build_data_smell_report
from refmark.metrics import score_ref_range, summarize_scores
from refmark.pipeline import RegionRecord
from refmark.rag_eval import CorpusMap, EvalSuite, NormalizedHit


OUTPUT = ROOT / "output"
ADDRESS_SPACE = [f"F{idx:02d}" for idx in range(1, 13)]


def main() -> int:
    OUTPUT.mkdir(exist_ok=True)
    payload = json.loads((ROOT / "predictions.json").read_text(encoding="utf-8"))
    gold = payload["gold"]
    report: dict[str, object] = {"address_space": ADDRESS_SPACE, "models": {}}

    for model_name, predictions in payload["models"].items():
        rows = []
        scores = []
        for qid, gold_refs in gold.items():
            score = score_ref_range(predictions.get(qid, []), gold_refs, address_space=ADDRESS_SPACE)
            scores.append(score)
            rows.append({"id": qid, **score.to_dict()})
        report["models"][model_name] = {
            "summary": {key: round(value, 3) for key, value in summarize_scores(scores).items()},
            "rows": rows,
        }

    out_path = OUTPUT / "data_smells_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    retrieval_report, adaptation_plan = _build_retrieval_smell_demo()
    print(json.dumps(report, indent=2))
    print(f"\nWrote data-smell report to {out_path}")
    print(f"Wrote retrieval-smell report to {retrieval_report}")
    print(f"Wrote adaptation plan to {adaptation_plan}")
    return 0


def _build_retrieval_smell_demo() -> tuple[Path, Path]:
    records = [
        _record(
            "docs",
            "P01",
            "Production API tokens must be rotated every 90 days. Emergency rotation is required after suspected credential exposure.",
            ordinal=1,
        ),
        _record(
            "docs",
            "P02",
            "When a token leaks, revoke the credential, rotate dependent services, and audit recent access logs.",
            ordinal=2,
        ),
        _record(
            "docs",
            "P03",
            "Release notes: token rotation wording was updated, examples were clarified, and old credential screenshots were removed.",
            ordinal=3,
        ),
        _record(
            "docs",
            "P04",
            "Audit logs for production access are retained for 180 days and reviewed during security incidents.",
            ordinal=4,
        ),
    ]
    corpus = CorpusMap.from_records(records, revision_id="demo-v2")
    hashes = corpus.source_hashes(["docs:P01", "docs:P02", "docs:P04"])
    rows = [
        {
            "query": "How often should production API tokens be rotated?",
            "gold_refs": ["docs:P01"],
            "source_hashes": {"docs:P01": "stale-demo-hash"},
            "metadata": {"query_style": "direct"},
        },
        {
            "query": "A credential leaked, what should I do next?",
            "gold_refs": ["docs:P02"],
            "source_hashes": {"docs:P02": hashes["docs:P02"]},
            "metadata": {"query_style": "concern"},
        },
        {
            "query": "Where do I find the retention period for audit logs?",
            "gold_refs": ["docs:P04"],
            "source_hashes": {"docs:P04": hashes["docs:P04"]},
            "metadata": {"query_style": "direct"},
        },
        {
            "query": "What should be reviewed during a security incident?",
            "gold_refs": ["docs:P04"],
            "source_hashes": {"docs:P04": hashes["docs:P04"]},
            "metadata": {"query_style": "concern"},
        },
        {
            "query": "Which credential screenshots changed in the docs?",
            "gold_refs": ["docs:P03"],
            "source_hashes": {"docs:P03": corpus.source_hashes(["docs:P03"])["docs:P03"]},
            "metadata": {"query_style": "direct"},
        },
    ]
    suite = EvalSuite.from_rows(rows, corpus=corpus)

    def retriever(query: str):
        lowered = query.lower()
        if "rotated" in lowered:
            return [
                NormalizedHit("docs:P03", score=0.51, context_refs=["docs:P03", "docs:P01", "docs:P02"]),
                NormalizedHit("docs:P01", score=0.49),
            ]
        if "leaked" in lowered:
            return [
                NormalizedHit("docs:P03", score=0.42),
                NormalizedHit("docs:P01", score=0.4),
            ]
        if "audit" in lowered or "incident" in lowered:
            return [
                NormalizedHit("docs:P03", score=0.38),
                NormalizedHit("docs:P01", score=0.37),
            ]
        return [NormalizedHit("docs:P03", score=0.9)]

    run = suite.evaluate(retriever, name="noisy-demo-retriever", k=2)
    eval_path = OUTPUT / "retrieval_eval_run.json"
    run.write_json(eval_path)
    smell_report = build_data_smell_report(suite, run)
    out_path = OUTPUT / "retrieval_smells_report.json"
    smell_report.write_json(out_path)
    plan = build_adaptation_plan(smell_report.to_dict())
    plan_path = OUTPUT / "adaptation_plan.json"
    plan.write_json(plan_path)
    return out_path, plan_path


def _record(doc_id: str, region_id: str, text: str, *, ordinal: int) -> RegionRecord:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return RegionRecord(
        doc_id=doc_id,
        region_id=region_id,
        text=text,
        start_line=ordinal,
        end_line=ordinal,
        ordinal=ordinal,
        hash=digest,
        source_path="examples/data_smells/demo.md",
    )


if __name__ == "__main__":
    raise SystemExit(main())
