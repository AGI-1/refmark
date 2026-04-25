import sys
import argparse
import json
from pathlib import Path

from refmark.config import load_local_env

load_local_env()

from refmark.core import inject, strip
from refmark.documents import align_documents, map_document
from refmark.document_io import extract_document_text, text_mapping_extension
from refmark.edit import apply_ref_diff
from refmark.highlight import highlight_refs, render_highlight_html, render_highlight_json, render_highlight_text
from refmark.languages import choose_edit_chunker, choose_live_marker_format, list_supported_languages
from refmark.pipeline import (
    align_region_records,
    build_region_manifest,
    evaluate_alignment_coverage,
    expand_region_context,
    read_manifest,
    render_coverage_html,
    write_manifest,
)
from refmark.prompt import build_reference_prompt
from refmark.workflow_config import WorkflowConfig, load_workflow_config, resolve_workflow_config


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

    prompt_parser = subparsers.add_parser(
        "enrich-prompt",
        help="Wrap a document in refmark citation instructions for a general chat model.",
    )
    prompt_parser.add_argument("file", help="Path to the document to include.")
    prompt_parser.add_argument("--question", default=None, help="Optional question to append after the marked document.")
    prompt_parser.add_argument("--question-file", default=None, help="Optional file containing the question.")
    prompt_parser.add_argument("--format", default="typed_bracket", help="Marker format name to use.")
    prompt_parser.add_argument("--chunker", default="paragraph", help="Chunker name to use.")
    prompt_parser.add_argument("-o", "--output", default=None, help="Optional output path. Defaults to stdout.")

    map_parser = subparsers.add_parser("map", help="Build a JSONL refmark region manifest for a file or directory.")
    map_parser.add_argument("path", help="File or directory to map.")
    map_parser.add_argument("-o", "--output", required=True, help="Manifest JSONL output path.")
    map_parser.add_argument("--marked-dir", default=None, help="Optional directory for marked document copies.")
    _add_workflow_config_args(map_parser)

    expand_parser = subparsers.add_parser("expand", help="Expand cited refs to neighboring manifest regions.")
    expand_parser.add_argument("manifest", help="Manifest JSONL path.")
    expand_parser.add_argument("--refs", required=True, help="Comma-separated refs or ranges, e.g. P01,P03-P04.")
    expand_parser.add_argument("--doc-id", default=None, help="Optional document id to scope expansion.")
    expand_parser.add_argument("--before", type=int, default=0, help="Number of previous regions to include.")
    expand_parser.add_argument("--after", type=int, default=0, help="Number of following regions to include.")
    expand_parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format.")

    align_parser = subparsers.add_parser("align", help="Map source document regions to target document regions.")
    align_parser.add_argument("source", help="Source document path.")
    align_parser.add_argument("target", help="Target document path.")
    _add_workflow_config_args(align_parser)
    align_parser.add_argument("--coverage-html", default=None, help="Optional HTML coverage review output path.")
    align_parser.add_argument("--coverage-json", default=None, help="Optional JSON coverage review output path.")
    align_parser.add_argument("--summary-json", default=None, help="Optional coverage summary JSON output path.")
    align_parser.add_argument("--layout", choices=["side-by-side", "stacked"], default="side-by-side", help="HTML report layout.")

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
    elif args.command == "enrich-prompt":
        _handle_enrich_prompt(args)
    elif args.command == "map":
        _handle_map(args)
    elif args.command == "expand":
        _handle_expand(args)
    elif args.command == "align":
        _handle_align(args)
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


def _handle_enrich_prompt(args):
    if args.question and args.question_file:
        print("Error: Provide only one of --question or --question-file.", file=sys.stderr)
        sys.exit(1)

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File '{file_path}' not found.", file=sys.stderr)
        sys.exit(1)

    question = args.question
    if args.question_file:
        question = Path(args.question_file).read_text(encoding="utf-8-sig")

    result = build_reference_prompt(
        file_path.read_text(encoding="utf-8-sig"),
        file_path.suffix or ".txt",
        question=question,
        marker_format=args.format,
        chunker=args.chunker,
    )
    if args.output:
        Path(args.output).write_text(result.prompt, encoding="utf-8")
        print(f"[OK] Wrote prompt with {result.marker_count} regions to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(result.prompt)


def _add_workflow_config_args(parser):
    parser.add_argument("--config", default=None, help="Optional flat YAML or JSON workflow config.")
    parser.add_argument("--density", choices=["dense", "balanced", "coarse", "code"], default=None)
    parser.add_argument("--marker-style", choices=["default", "machine", "explicit", "compact", "xml"], default=None)
    parser.add_argument("--format", dest="marker_format", default=None, help="Concrete marker format name to use.")
    parser.add_argument("--chunker", default=None, help="Concrete chunker name to use.")
    parser.add_argument("--min-words", type=int, default=None, help="Minimum token count for retained regions.")
    parser.add_argument("--lines-per-chunk", type=int, default=None, help="Line chunk size for dense/custom line mapping.")
    parser.add_argument("--tokens-per-chunk", type=int, default=None, help="Token chunk size for coarse/custom token mapping.")
    parser.add_argument("--ignore-title", action="store_true", help="Drop a first title-like region from document maps.")
    parser.add_argument("--top-k", type=int, default=None, help="Candidates per source region.")
    parser.add_argument("--threshold", type=float, default=None, help="Coverage threshold for covered/gap status.")
    parser.add_argument("--expand-before", type=int, default=None, help="Previous target regions for coverage context.")
    parser.add_argument("--expand-after", type=int, default=None, help="Following target regions for coverage context.")
    parser.add_argument("--no-numeric-checks", action="store_true", help="Disable small deterministic numeric conflict checks.")


def _workflow_config_from_args(args) -> WorkflowConfig:
    base = load_workflow_config(args.config) if getattr(args, "config", None) else WorkflowConfig()
    return resolve_workflow_config(
        base,
        density=getattr(args, "density", None),
        marker_style=getattr(args, "marker_style", None),
        marker_format=getattr(args, "marker_format", None),
        chunker=getattr(args, "chunker", None),
        min_words=getattr(args, "min_words", None),
        lines_per_chunk=getattr(args, "lines_per_chunk", None),
        tokens_per_chunk=getattr(args, "tokens_per_chunk", None),
        include_headings=False if getattr(args, "ignore_title", False) else None,
        top_k=getattr(args, "top_k", None),
        coverage_threshold=getattr(args, "threshold", None),
        expand_before=getattr(args, "expand_before", None),
        expand_after=getattr(args, "expand_after", None),
        numeric_checks=False if getattr(args, "no_numeric_checks", False) else None,
    )


def _handle_map(args):
    root = Path(args.path)
    if not root.exists():
        print(f"Error: Path '{root}' not found.", file=sys.stderr)
        sys.exit(1)

    files = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
    records = []
    marked_dir = Path(args.marked_dir) if args.marked_dir else None
    config = _workflow_config_from_args(args)
    for file_path in files:
        if file_path.name.startswith("."):
            continue
        try:
            doc_map = map_document(
                file_path,
                config=config,
                doc_id=file_path.stem if root.is_file() else file_path.relative_to(root).as_posix(),
            )
        except (UnicodeDecodeError, RuntimeError, KeyError) as exc:
            print(f"[SKIP] Could not extract text from {file_path}: {exc}", file=sys.stderr)
            continue
        for warning in doc_map.warnings:
            print(f"[WARN] {file_path}: {warning}", file=sys.stderr)
        records.extend(doc_map.records)
        if marked_dir:
            destination = marked_dir / (file_path.name if root.is_file() else file_path.relative_to(root))
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(doc_map.marked_text, encoding="utf-8")

    write_manifest(records, args.output)
    print(f"[OK] Wrote {len(records)} regions from {len(files)} file(s) to {args.output}", file=sys.stderr)


def _handle_expand(args):
    refs = [ref.strip() for ref in args.refs.split(",") if ref.strip()]
    records = expand_region_context(
        read_manifest(args.manifest),
        refs,
        doc_id=args.doc_id,
        before=max(args.before, 0),
        after=max(args.after, 0),
    )
    if args.format == "json":
        print(json.dumps([record.to_dict() for record in records], indent=2))
        return

    for record in records:
        print(f"[{record.doc_id}:{record.region_id}] lines {record.start_line}-{record.end_line}")
        print(record.text.rstrip())
        print()


def _handle_align(args):
    source_path = Path(args.source)
    target_path = Path(args.target)
    for path in (source_path, target_path):
        if not path.exists():
            print(f"Error: File '{path}' not found.", file=sys.stderr)
            sys.exit(1)

    config = _workflow_config_from_args(args)
    report = align_documents(source_path, target_path, config=config)
    alignments = report.alignments
    coverage = report.coverage
    if args.coverage_json:
        Path(args.coverage_json).write_text(
            json.dumps([item.to_dict() for item in coverage], indent=2),
            encoding="utf-8",
        )
    if args.coverage_html:
        report.write_html(args.coverage_html, layout=args.layout)
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(report.summary, indent=2), encoding="utf-8")
    print(json.dumps([[candidate.to_dict() for candidate in row] for row in alignments], indent=2))


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
