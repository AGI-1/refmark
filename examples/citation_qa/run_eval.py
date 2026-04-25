from __future__ import annotations

import json
import shutil
from pathlib import Path

from refmark.core import inject
from refmark.highlight import highlight_refs, render_highlight_html, render_highlight_text
from refmark.metrics import score_ref_range, summarize_scores


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    OUTPUT.mkdir(exist_ok=True)
    marked_path = OUTPUT / "marked_corpus.py"
    shutil.copyfile(ROOT / "corpus.py", marked_path)

    marked, count = inject(
        marked_path.read_text(encoding="utf-8"),
        ".py",
        marker_format="typed_comment_py",
        chunker="hybrid",
    )
    marked_path.write_text(marked, encoding="utf-8")

    questions = _load_jsonl(ROOT / "questions.jsonl")
    predictions = json.loads((ROOT / "predictions.json").read_text(encoding="utf-8"))

    address_space = ["P01", "F02", "F03", "F04"]
    rows = []
    scores = []
    all_refs: list[str] = []
    for item in questions:
        predicted_refs = predictions.get(item["id"], [])
        gold_refs = item["gold_refs"]
        score = score_ref_range(predicted_refs, gold_refs, address_space=address_space)
        scores.append(score)
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "gold_refs": gold_refs,
                "predicted_refs": predicted_refs,
                **score.to_dict(),
            }
        )
        all_refs.extend(predicted_refs)

    means = {key: round(value, 3) for key, value in summarize_scores(scores).items()}
    result = {
        "marked_corpus": str(marked_path),
        "marker_count": count,
        "means": means,
        "rows": rows,
    }

    (OUTPUT / "scores.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    unique_refs = ",".join(dict.fromkeys(all_refs))
    highlighted = highlight_refs(marked_path, unique_refs, context_lines=1)
    (OUTPUT / "cited_regions.txt").write_text(render_highlight_text(highlighted), encoding="utf-8")
    (OUTPUT / "cited_regions.html").write_text(render_highlight_html(highlighted), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"\nWrote review artifacts under {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
