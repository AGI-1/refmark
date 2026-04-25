from __future__ import annotations

import json
from pathlib import Path

from refmark.metrics import score_ref_range, summarize_scores


ROOT = Path(__file__).resolve().parent
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
    print(json.dumps(report, indent=2))
    print(f"\nWrote data-smell report to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
