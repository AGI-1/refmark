from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from refmark.pipeline import (
    align_region_records,
    build_region_manifest,
    expand_region_context,
    write_manifest,
)
from refmark.prompt import build_reference_prompt


ROOT = Path(__file__).parent
OUTPUT = ROOT / "output"


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    source_text = (ROOT / "source_policy.txt").read_text(encoding="utf-8")
    target_text = (ROOT / "target_policy.txt").read_text(encoding="utf-8")

    source_marked, source_regions = build_region_manifest(source_text, ".txt", doc_id="source_policy")
    target_marked, target_regions = build_region_manifest(target_text, ".txt", doc_id="target_policy")
    write_manifest(source_regions + target_regions, OUTPUT / "manifest.jsonl")
    (OUTPUT / "source_marked.txt").write_text(source_marked, encoding="utf-8")
    (OUTPUT / "target_marked.txt").write_text(target_marked, encoding="utf-8")

    expanded = expand_region_context(target_regions, ["P01"], after=1)
    alignments = align_region_records(source_regions, target_regions, top_k=2)
    prompt = build_reference_prompt(
        target_text,
        ".txt",
        question="Which regions support claims about expedited shipping and refund windows?",
    )

    (OUTPUT / "expanded_context.json").write_text(
        json.dumps([record.to_dict() for record in expanded], indent=2),
        encoding="utf-8",
    )
    (OUTPUT / "alignment.json").write_text(
        json.dumps([[candidate.to_dict() for candidate in row] for row in alignments], indent=2),
        encoding="utf-8",
    )
    (OUTPUT / "chat_prompt.txt").write_text(prompt.prompt, encoding="utf-8")

    print(f"wrote {OUTPUT / 'manifest.jsonl'}")
    print(f"wrote {OUTPUT / 'expanded_context.json'}")
    print(f"wrote {OUTPUT / 'alignment.json'}")
    print(f"wrote {OUTPUT / 'chat_prompt.txt'}")


if __name__ == "__main__":
    main()
