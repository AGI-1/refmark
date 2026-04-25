from __future__ import annotations

import json
import shutil
from pathlib import Path

from refmark.core import inject
from refmark.edit import apply_ref_diff
from refmark.highlight import highlight_refs, render_highlight_text


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"


def _load_edits(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload["edits"]


def main() -> int:
    OUTPUT.mkdir(exist_ok=True)
    target = OUTPUT / "working_source.py"
    shutil.copyfile(ROOT / "source.py", target)

    marked, count = inject(
        target.read_text(encoding="utf-8"),
        ".py",
        marker_format="typed_comment_py",
        chunker="hybrid",
    )
    target.write_text(marked, encoding="utf-8")

    good_result = apply_ref_diff(target, _load_edits(ROOT / "good_edits.json"), expect_live_markers=True)
    after_good = target.read_text(encoding="utf-8")

    stale_result = apply_ref_diff(target, _load_edits(ROOT / "stale_edit.json"), expect_live_markers=True)
    after_stale = target.read_text(encoding="utf-8")
    stale_left_file_unchanged = after_good == after_stale

    highlighted = highlight_refs(target, "F01-F02", context_lines=0)
    (OUTPUT / "highlight_after_good_edits.txt").write_text(render_highlight_text(highlighted), encoding="utf-8")
    (OUTPUT / "final_source.py").write_text(after_stale, encoding="utf-8")

    result = {
        "working_source": str(target),
        "marker_count": count,
        "good_edit_ok": bool(good_result.get("ok")),
        "good_edit_result": good_result,
        "stale_edit_ok": bool(stale_result.get("ok")),
        "stale_edit_errors": stale_result.get("errors", []),
        "stale_left_file_unchanged": stale_left_file_unchanged,
        "final_source": str(OUTPUT / "final_source.py"),
        "highlight": str(OUTPUT / "highlight_after_good_edits.txt"),
    }
    (OUTPUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nWrote demo artifacts under {OUTPUT}")
    return 0 if good_result.get("ok") and stale_left_file_unchanged else 1


if __name__ == "__main__":
    raise SystemExit(main())
