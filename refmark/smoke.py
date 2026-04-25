from __future__ import annotations

import tempfile
from pathlib import Path

from refmark.core import inject
from refmark.edit import apply_ref_diff
from refmark.highlight import highlight_refs


SAMPLE_SOURCE = """def normalize_name(name: str) -> str:
    return name.strip()


def greet(name: str) -> str:
    return f"Hello, {normalize_name(name)}"
"""


def _score_refs(predicted: set[str], gold: set[str]) -> dict[str, float]:
    if not predicted or not gold:
        return {"exact_match": 0.0, "overlap": 0.0, "cover": 0.0}
    intersection = predicted & gold
    return {
        "exact_match": 1.0 if predicted == gold else 0.0,
        "overlap": len(intersection) / len(predicted | gold),
        "cover": len(intersection) / len(gold),
    }


def run_smoke() -> dict[str, object]:
    """Run the deterministic smoke path shipped with the public artifact."""
    with tempfile.TemporaryDirectory(prefix="refmark_smoke_") as tmp:
        root = Path(tmp)
        path = root / "sample.py"
        state_dir = root / "shadow_state"
        marked, marker_count = inject(
            SAMPLE_SOURCE,
            ".py",
            marker_format="typed_comment_py",
            chunker="hybrid",
        )
        path.write_text(marked, encoding="utf-8")

        cited = highlight_refs(path, "F01", context_lines=0, state_dir=state_dir)
        scores = _score_refs(set(cited.refs), {"F01"})

        edit_result = apply_ref_diff(
            path,
            [
                {
                    "region_id": "F01",
                    "action": "patch_within",
                    "patch_format": "line_edits",
                    "patch": {
                        "edits": [
                            {
                                "start_line": 2,
                                "end_line": 2,
                                "expected_text": "    return name.strip()\n",
                                "new_content": "    return name.strip().title()\n",
                            }
                        ]
                    },
                }
            ],
            expect_live_markers=True,
        )
        edited = path.read_text(encoding="utf-8")

    ok = (
        marker_count >= 2
        and scores["exact_match"] == 1.0
        and bool(edit_result.get("ok"))
        and ".title()" in edited
    )
    return {
        "ok": ok,
        "markers": marker_count,
        "citation_exact_match": scores["exact_match"],
        "citation_overlap": scores["overlap"],
        "citation_cover": scores["cover"],
        "edit_ok": bool(edit_result.get("ok")),
        "syntax_ok": bool(edit_result.get("syntax_ok")),
    }


def main() -> int:
    result = run_smoke()
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
