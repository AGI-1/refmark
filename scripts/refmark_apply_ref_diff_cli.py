from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from refmark.edit import apply_ref_diff


def _append_shell_log(payload: dict[str, Any]) -> None:
    log_path = os.getenv("REFMARK_SHELL_LOG_PATH")
    if not log_path:
        return
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _load_spec(args: argparse.Namespace) -> dict[str, Any]:
    sources = sum(bool(source) for source in (args.spec_file, args.spec_json, args.stdin))
    if sources != 1:
        raise ValueError("Provide exactly one of --spec-file, --spec-json, or --stdin.")

    raw: str
    if args.spec_file:
        raw = Path(args.spec_file).read_text(encoding="utf-8-sig")
    elif args.spec_json:
        raw = args.spec_json
    else:
        raw = sys.stdin.read()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Spec must be valid JSON: {exc.msg}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Spec JSON must decode to an object.")
    return parsed


def _normalize_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if "file_path" not in spec:
        raise ValueError("Spec requires file_path.")
    if "edits" not in spec:
        raise ValueError("Spec requires edits.")
    if not isinstance(spec["edits"], list):
        raise ValueError("Spec field edits must be a list.")

    normalized_edits: list[dict[str, Any]] = []
    for index, raw_edit in enumerate(spec["edits"], start=1):
        if not isinstance(raw_edit, dict):
            raise ValueError(f"Edit {index} must be an object.")
        edit = dict(raw_edit)
        if "new_content" not in edit:
            if "replacement" in edit:
                edit["new_content"] = edit.pop("replacement")
            elif "content" in edit:
                edit["new_content"] = edit.pop("content")
        if "action" not in edit and "new_content" in edit:
            edit["action"] = "replace"
        normalized_edits.append(edit)

    normalized = {
        "file_path": str(spec["file_path"]),
        "edits": normalized_edits,
    }
    if "expect_live_markers" in spec:
        normalized["expect_live_markers"] = spec["expect_live_markers"]
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply refmark edits from a strict JSON spec through a shell-friendly CLI.",
    )
    parser.add_argument("--spec-file", help="Path to a JSON file with file_path, edits, and optional expect_live_markers.")
    parser.add_argument("--spec-json", help="Inline JSON spec.")
    parser.add_argument("--stdin", action="store_true", help="Read the JSON spec from stdin.")
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON result instead of emitting a compact single-line object.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    started = time.time()

    try:
        spec = _normalize_spec(_load_spec(args))
        result = apply_ref_diff(
            spec["file_path"],
            spec["edits"],
            expect_live_markers=spec.get("expect_live_markers"),
        )
    except Exception as exc:
        payload = {
            "ok": False,
            "errors": [str(exc)],
        }
        _append_shell_log(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "tool": "apply_ref_diff_shell",
                "ok": False,
                "exception": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "latency_seconds": round(time.time() - started, 4),
                "task_id": os.getenv("REFMARK_AGENT_TASK_ID"),
                "variant_id": os.getenv("REFMARK_AGENT_VARIANT_ID"),
                "agent_name": os.getenv("REFMARK_AGENT_NAME"),
            }
        )
        if args.pretty:
            print(json.dumps(payload, indent=2))
        else:
            print(json.dumps(payload))
        return 1

    _append_shell_log(
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "tool": "apply_ref_diff_shell",
            "ok": bool(result.get("ok")),
            "file_path": spec["file_path"],
            "expect_live_markers": spec.get("expect_live_markers"),
            "edits": spec["edits"],
            "syntax_ok": result.get("syntax_ok"),
            "errors": result.get("errors", []),
            "latency_seconds": round(time.time() - started, 4),
            "task_id": os.getenv("REFMARK_AGENT_TASK_ID"),
            "variant_id": os.getenv("REFMARK_AGENT_VARIANT_ID"),
            "agent_name": os.getenv("REFMARK_AGENT_NAME"),
        }
    )

    if args.pretty:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(result))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
