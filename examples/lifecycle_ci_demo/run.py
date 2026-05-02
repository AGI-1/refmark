from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
OLD_DOC = OUTPUT / "rev_a" / "policy.md"
NEW_DOC = OUTPUT / "rev_b" / "policy.md"
OLD_MANIFEST = OUTPUT / "policy_rev_a.jsonl"
NEW_MANIFEST = OUTPUT / "policy_rev_b.jsonl"
EXAMPLES = OUTPUT / "eval_examples.jsonl"
REPORT = OUTPUT / "lifecycle_report.json"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    OLD_DOC.parent.mkdir(parents=True, exist_ok=True)
    NEW_DOC.parent.mkdir(parents=True, exist_ok=True)
    OLD_DOC.write_text(
        "# Shipping Policy\n\n"
        "Standard refunds are available within 30 days.\n\n"
        "Expedited shipping is non-refundable after dispatch.\n",
        encoding="utf-8",
    )
    NEW_DOC.write_text(
        "# Shipping Policy\n\n"
        "Standard refunds are available within 45 days.\n\n"
        "Expedited shipping is non-refundable after dispatch.\n",
        encoding="utf-8",
    )

    _run(["map", str(OLD_DOC), "--output", str(OLD_MANIFEST)])
    old_rows = _read_jsonl(OLD_MANIFEST)
    refund_ref = next(row for row in old_rows if "Standard refunds" in row["text"])
    EXAMPLES.write_text(
        json.dumps(
            {
                "query": "How long can customers request a standard refund?",
                "gold_refs": [f"{refund_ref['doc_id']}:{refund_ref['region_id']}"],
                "source_hashes": {f"{refund_ref['doc_id']}:{refund_ref['region_id']}": refund_ref["hash"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    _run(["map", str(NEW_DOC), "--output", str(NEW_MANIFEST)])
    _run(
        [
            "lifecycle-validate-labels",
            str(NEW_MANIFEST),
            str(EXAMPLES),
            "--previous-manifest",
            str(OLD_MANIFEST),
            "--previous-revision",
            "rev-a",
            "--current-revision",
            "rev-b",
            "--output",
            str(REPORT),
        ]
    )
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    print(json.dumps({"stale_example_count": payload["stale_example_count"], "status": payload["status"]}, indent=2))


def _run(args: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "refmark.cli", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
