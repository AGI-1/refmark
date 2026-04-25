from __future__ import annotations

import json
from pathlib import Path

from refmark.metrics import citation_reward, score_ref_range


ROOT = Path(__file__).resolve().parent
PUBLISH_ROOT = ROOT.parents[1]
DATASET = PUBLISH_ROOT / "refmark_train" / "data" / "documentation_full_paragraph_contextual_idf_lean2"
OUTPUT = ROOT / "output"


def _load_jsonl(path: Path, limit: int) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
        if len(rows) >= limit:
            break
    return rows


def _neighbor(refmark: str, delta: int) -> str:
    prefix = "".join(ch for ch in refmark if ch.isalpha())
    digits = "".join(ch for ch in refmark if ch.isdigit())
    return f"{prefix}{max(1, int(digits) + delta):0{len(digits)}d}"


def main() -> int:
    OUTPUT.mkdir(exist_ok=True)
    examples = _load_jsonl(DATASET / "valid.jsonl", limit=5)

    rows = []
    for item in examples:
        gold = [item["refmark"]]
        candidates = {
            "exact": gold,
            "overcite_neighbor": [_neighbor(item["refmark"], -1), item["refmark"]],
            "wrong_location": [_neighbor(item["refmark"], 200)],
            "missing": [],
        }
        scored = {}
        for name, refs in candidates.items():
            score = score_ref_range(refs, gold)
            scored[name] = {
                "predicted_refs": refs,
                "reward": round(citation_reward(score), 4),
                "metrics": score.to_dict(),
            }
        rows.append(
            {
                "question": item["question"],
                "gold_refs": gold,
                "candidates": scored,
            }
        )

    report = {
        "dataset": str(DATASET),
        "note": "Rewards are deterministic integer/set math over refs; no LLM judge or API call is used.",
        "rows": rows,
    }
    out_path = OUTPUT / "judge_free_rewards.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote judge-free reward report to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
