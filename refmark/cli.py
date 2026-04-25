import sys
import argparse
import json
from pathlib import Path

from refmark.config import load_local_env

load_local_env()

from refmark.core import inject, strip
from refmark.edit import apply_ref_diff
from refmark.highlight import highlight_refs, render_highlight_html, render_highlight_json, render_highlight_text
from refmark.languages import choose_edit_chunker, choose_live_marker_format, list_supported_languages


def main():
    parser = argparse.ArgumentParser(
        description="Refmark CLI: inject, strip, highlight, and apply reference-marker edits."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # inject command
    inject_parser = subparsers.add_parser("inject", help="Inject reference markers into a file.")
    inject_parser.add_argument("file", help="Path to the file to process.")
    inject_parser.add_argument("-o", "--output", help="Output file path (default: overwrite).")
    inject_parser.add_argument("--dry-run", action="store_true", help="Print to stdout without writing to file.")
    inject_parser.add_argument("--format", default=None, help="Marker format name (e.g., bracket, hash).")
    inject_parser.add_argument("--chunker", default=None, help="Chunker name (e.g., line, paragraph, token, ast).")

    # strip command
    strip_parser = subparsers.add_parser("strip", help="Strip reference markers from a file.")
    strip_parser.add_argument("file", help="Path to the file to process.")
    strip_parser.add_argument("-o", "--output", help="Output file path (default: overwrite).")
    strip_parser.add_argument("--dry-run", action="store_true", help="Print to stdout without writing to file.")
    strip_parser.add_argument("--format", default=None, help="Marker format name to strip.")

    # languages command
    subparsers.add_parser("languages", help="List currently supported language integrations.")

    # apply-ref-diff command
    refdiff_parser = subparsers.add_parser("apply-ref-diff", help="Apply refmark-based multi-region edits from JSON.")
    refdiff_parser.add_argument("file", help="Path to the file to process.")
    refdiff_parser.add_argument("--edits-json", default=None, help="Inline JSON payload with an edits array.")
    refdiff_parser.add_argument("--edits-file", default=None, help="Path to a JSON file containing an edits array.")

    highlight_parser = subparsers.add_parser("highlight", help="Render cited refs back into reviewable source snippets.")
    highlight_parser.add_argument("file", help="Path to the file to inspect.")
    highlight_parser.add_argument("--refs", required=True, help="Comma-separated refs or ranges, e.g. F03,F05-F06.")
    highlight_parser.add_argument("--context-lines", type=int, default=2, help="Context lines around each cited region.")
    highlight_parser.add_argument("--state-dir", default=None, help="Optional shadow-session state directory.")
    highlight_parser.add_argument("--format", choices=["text", "html", "json"], default="text", help="Output format.")
    highlight_parser.add_argument("-o", "--output", default=None, help="Optional output path. Defaults to stdout.")

    smoke_parser = subparsers.add_parser("smoke", help="Run a deterministic public-artifact smoke check.")
    smoke_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    args = parser.parse_args()

    if args.command == "inject":
        _handle_inject(args)
    elif args.command == "strip":
        _handle_strip(args)
    elif args.command == "languages":
        _handle_languages()
    elif args.command == "apply-ref-diff":
        _handle_apply_ref_diff(args)
    elif args.command == "highlight":
        _handle_highlight(args)
    elif args.command == "smoke":
        _handle_smoke(args)


def _handle_inject(args):
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File '{file_path}' not found.", file=sys.stderr)
        sys.exit(1)

    content = file_path.read_text(encoding="utf-8")
    ext = file_path.suffix

    chunker_kwargs = {}
    if args.chunker == "token":
        chunker_kwargs = {"tokens_per_chunk": 200}
    elif args.chunker == "line":
        chunker_kwargs = {"lines_per_chunk": 10}

    marker_format = args.format or choose_live_marker_format(ext)
    chunker = args.chunker or choose_edit_chunker(ext)
    processed, count = inject(
        content,
        ext,
        marker_format=marker_format,
        chunker=chunker,
        chunker_kwargs=chunker_kwargs or None,
    )
    if args.dry_run:
        sys.stdout.write(processed)
    else:
        out_path = Path(args.output) if args.output else file_path
        out_path.write_text(processed, encoding="utf-8")
        print(f"[OK] Injected {count} markers into {out_path}", file=sys.stderr)


def _handle_strip(args):
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File '{file_path}' not found.", file=sys.stderr)
        sys.exit(1)

    content = file_path.read_text(encoding="utf-8")
    ext = file_path.suffix

    processed = strip(content, ext, marker_format=args.format)
    if args.dry_run:
        sys.stdout.write(processed)
    else:
        out_path = Path(args.output) if args.output else file_path
        out_path.write_text(processed, encoding="utf-8")
        print(f"[OK] Stripped markers from {out_path}", file=sys.stderr)


def _handle_languages():
    print(json.dumps({"languages": list_supported_languages()}, indent=2))


def _handle_apply_ref_diff(args):
    if bool(args.edits_json) == bool(args.edits_file):
        print("Error: Provide exactly one of --edits-json or --edits-file.", file=sys.stderr)
        sys.exit(1)

    payload_text = args.edits_json
    if args.edits_file:
        payload_text = Path(args.edits_file).read_text(encoding="utf-8-sig")

    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        print(f"Error: Invalid JSON payload: {exc}", file=sys.stderr)
        sys.exit(1)

    edits = payload.get("edits") if isinstance(payload, dict) else payload
    if not isinstance(edits, list):
        print("Error: Payload must be a JSON array or an object with an 'edits' array.", file=sys.stderr)
        sys.exit(1)

    result = apply_ref_diff(args.file, edits)
    print(json.dumps(result, indent=2))
    if not result.get("ok"):
        sys.exit(1)


def _handle_highlight(args):
    try:
        result = highlight_refs(
            args.file,
            args.refs,
            context_lines=args.context_lines,
            state_dir=args.state_dir,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.format == "html":
        output = render_highlight_html(result)
    elif args.format == "json":
        output = render_highlight_json(result)
    else:
        output = render_highlight_text(result)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)


def _handle_smoke(args):
    from refmark.smoke import run_smoke

    result = run_smoke()
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("Refmark smoke check passed.")
        print(f"markers: {result['markers']}")
        print(f"citation_exact_match: {result['citation_exact_match']:.3f}")
        print(f"citation_overlap: {result['citation_overlap']:.3f}")
        print(f"citation_cover: {result['citation_cover']:.3f}")
        print(f"edit_ok: {result['edit_ok']}")
    if not result["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
