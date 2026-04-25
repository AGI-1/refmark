from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from refmark.regions import _parse_blocks_with_mode
from refmark.shadow_session import apply_shadow_session_ref_diff, load_or_build_view_state


def _slice_text_window(
    text: str,
    *,
    start_line: int | None,
    end_line: int | None,
    max_chars: int,
    max_lines: int,
) -> dict[str, Any]:
    lines = text.splitlines()
    total_lines = len(lines)
    requested_start = max(1, start_line or 1)
    requested_end = min(total_lines, end_line or total_lines)
    if requested_end < requested_start:
        requested_end = requested_start - 1
    selected = lines[requested_start - 1:requested_end]
    if len(selected) > max_lines:
        selected = selected[:max_lines]
    body = "\n".join(selected)
    if total_lines and requested_end >= requested_start and text.endswith("\n"):
        body += "\n"
    if len(body) > max_chars:
        body = body[:max_chars]
    returned_end = requested_start + max(0, len(selected) - 1)
    return {
        "content": body,
        "returned_start_line": requested_start,
        "returned_end_line": returned_end if selected else requested_start - 1,
        "total_lines": total_lines,
    }


def _preview_lines_from_shadow(
    shadow_text: str,
    *,
    start_line: int,
    end_line: int,
    max_lines: int,
) -> list[str]:
    shadow_lines = shadow_text.splitlines()
    lines: list[str] = []
    for raw_line in shadow_lines[start_line - 1:]:
        stripped = raw_line.rstrip()
        if not stripped.strip():
            continue
        if stripped.lstrip().startswith(("# [@", "// [@", "[@")):
            if lines:
                break
            continue
        lines.append(stripped)
        if len(lines) >= max_lines:
            break
    return lines


def _load_spec(args: argparse.Namespace) -> dict[str, Any]:
    if args.spec_file:
        raw = Path(args.spec_file).read_text(encoding="utf-8-sig")
    elif args.spec_json:
        raw = args.spec_json
    else:
        raw = sys.stdin.read()
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Spec must decode to an object.")
    if "file_path" not in parsed or "edits" not in parsed:
        raise ValueError("Spec requires file_path and edits.")
    return parsed


def _cmd_list(args: argparse.Namespace) -> int:
    path = Path(args.file_path)
    state = load_or_build_view_state(path, Path(args.state_dir))
    if not state["supported"]:
        print(json.dumps({"ok": True, "regions": [], "refmark_supported": False}))
        return 0
    blocks = _parse_blocks_with_mode(state["view_text"], state["marker_format"], line_mode="marked")
    regions = []
    for region_id in sorted(blocks.keys()):
        block = blocks[region_id]
        regions.append(
            {
                "region_id": str(region_id),
                "start_line": int(block["line_start"]),
                "end_line": int(block["line_end"]),
                "preview_lines": _preview_lines_from_shadow(
                    state["view_text"],
                    start_line=int(block["line_start"]),
                    end_line=int(block["line_end"]),
                    max_lines=max(0, args.preview_lines),
                ),
            }
        )
    print(
        json.dumps(
            {
                "ok": True,
                "file_path": str(path),
                "regions": regions,
                "namespace_mode": state["namespace_mode"],
                "shadow_persistent": state["shadow_persistent"],
                "session_reset": state["session_reset"],
                "state_path": state.get("state_path"),
            }
        )
    )
    return 0


def _cmd_read(args: argparse.Namespace) -> int:
    path = Path(args.file_path)
    state = load_or_build_view_state(path, Path(args.state_dir))
    start_line = args.start_line
    end_line = args.end_line
    selected_region_ids: list[str] = []
    if args.region_ids:
        blocks = _parse_blocks_with_mode(state["view_text"], state["marker_format"], line_mode="marked")
        requested = [token.strip() for token in args.region_ids if token and token.strip()]
        missing = [region_id for region_id in requested if region_id not in blocks]
        if missing:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": "unknown_region_ids",
                        "missing_region_ids": missing,
                        "available_region_ids": sorted(str(key) for key in blocks.keys()),
                    }
                )
            )
            return 1
        if requested:
            selected_region_ids = requested
            start_line = min(int(blocks[region_id]["line_start"]) for region_id in requested)
            end_line = max(int(blocks[region_id]["line_end"]) for region_id in requested)
    window = _slice_text_window(
        state["view_text"],
        start_line=start_line,
        end_line=end_line,
        max_chars=args.max_chars,
        max_lines=args.max_lines,
    )
    payload = {
        "ok": True,
        "file_path": str(path),
        "namespace_mode": state["namespace_mode"],
        "shadow_persistent": state["shadow_persistent"],
        "session_reset": state["session_reset"],
        "state_path": state.get("state_path"),
        "selected_region_ids": selected_region_ids,
        **window,
    }
    print(json.dumps(payload))
    return 0


def _cmd_apply(args: argparse.Namespace) -> int:
    spec = _load_spec(args)
    result = apply_shadow_session_ref_diff(
        Path(spec["file_path"]),
        list(spec["edits"]),
        Path(args.state_dir),
    )
    print(json.dumps(result))
    return 0 if result.get("ok") else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent shadow-session CLI for refmark reads and edits.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    list_parser = sub.add_parser("list")
    list_parser.add_argument("--file-path", required=True)
    list_parser.add_argument("--state-dir", required=True)
    list_parser.add_argument("--preview-lines", type=int, default=3)
    list_parser.set_defaults(func=_cmd_list)

    read_parser = sub.add_parser("read")
    read_parser.add_argument("--file-path", required=True)
    read_parser.add_argument("--state-dir", required=True)
    read_parser.add_argument("--region-id", dest="region_ids", action="append")
    read_parser.add_argument("--region-ids", dest="region_ids", nargs="+")
    read_parser.add_argument("--region", dest="region_ids", action="append")
    read_parser.add_argument("--start-line", type=int)
    read_parser.add_argument("--end-line", type=int)
    read_parser.add_argument("--max-chars", type=int, default=12000)
    read_parser.add_argument("--max-lines", type=int, default=400)
    read_parser.set_defaults(func=_cmd_read)

    apply_parser = sub.add_parser("apply")
    apply_parser.add_argument("--state-dir", required=True)
    apply_parser.add_argument("--spec-file")
    apply_parser.add_argument("--spec-json")
    apply_parser.add_argument("--stdin", action="store_true")
    apply_parser.set_defaults(func=_cmd_apply)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
