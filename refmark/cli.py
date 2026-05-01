import sys
import argparse
import json
import re
from pathlib import Path
import urllib.request

from refmark.config import load_local_env

load_local_env()

from refmark.citations import citation_refs_to_strings, parse_citation_refs
from refmark.core import inject, strip
from refmark.documents import align_documents, map_document
from refmark.document_io import extract_document_text, text_mapping_extension
from refmark.discovery import build_discovery_context_card, discover_corpus, load_discovery, repair_discovery_clusters, review_discovery, write_discovery
from refmark.discovery_heatmap import write_discovery_map_html
from refmark.edit import apply_ref_diff
from refmark.feedback import analyze_feedback, read_feedback_jsonl
from refmark.highlight import highlight_refs, render_highlight_html, render_highlight_json, render_highlight_text
from refmark.languages import choose_edit_chunker, choose_live_marker_format, list_supported_languages
from refmark.pipeline import (
    RegionRecord,
    align_region_records,
    build_region_manifest,
    build_section_map,
    evaluate_alignment_coverage,
    expand_region_context,
    read_manifest,
    render_coverage_html,
    write_manifest,
)
from refmark.pipeline_config import load_full_pipeline_config, write_full_pipeline_config_template
from refmark.pipeline_runner import run_full_pipeline
from refmark.prompt import build_reference_prompt
from refmark.provenance import build_eval_provenance, validate_provenance
from refmark.question_plan import build_question_plan, question_plan_to_dict
from refmark.rag_eval import CorpusMap, EvalSuite
from refmark.search_index import OPENROUTER_CHAT_URL, analyze_index_smells, build_search_index, export_browser_search_index, load_search_index
from refmark.workflow_config import WorkflowConfig, load_workflow_config, resolve_workflow_config


DEFAULT_QUESTION_PROMPT_TEMPLATE = """You are generating retrieval evaluation questions for a refmarked corpus.

Gold evidence refs: {refs}
Language: {language}
Question count: {count}

Write {count} natural user questions that are answerable from the evidence
below. The questions should require recovering the listed refs or range, not
only matching a keyword. Return JSONL only, one object per line:
{{"query":"...","gold_refs":{refs_json},"notes":"why this evidence is required"}}

Evidence:
{context}
"""


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
    refdiff_parser.add_argument("--dry-run", action="store_true", help="Validate and preview without writing the file.")
    refdiff_parser.add_argument("--base-hash", default=None, help="Expected SHA-256 of the current source file.")
    refdiff_parser.add_argument("--diff", action="store_true", help="Include a unified diff preview in the JSON result.")

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

    toc_parser = subparsers.add_parser("toc", help="Build a section/TOC map from a refmark manifest.")
    toc_parser.add_argument("manifest", help="Manifest JSONL path.")
    toc_parser.add_argument("-o", "--output", required=True, help="Section map JSON output path.")

    expand_parser = subparsers.add_parser("expand", help="Expand cited refs to neighboring manifest regions.")
    expand_parser.add_argument("manifest", help="Manifest JSONL path.")
    expand_parser.add_argument("--refs", required=True, help="Comma-separated refs or ranges, e.g. P01,P03-P04.")
    expand_parser.add_argument("--doc-id", default=None, help="Optional document id to scope expansion.")
    expand_parser.add_argument("--before", type=int, default=0, help="Number of previous regions to include.")
    expand_parser.add_argument("--after", type=int, default=0, help="Number of following regions to include.")
    expand_parser.add_argument("--same-parent", action="store_true", help="Include regions with the same parent_region_id.")
    expand_parser.add_argument("--include-parent", action="store_true", help="Include the parent region when using --same-parent.")
    expand_parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format.")

    align_parser = subparsers.add_parser("align", help="Map source document regions to target document regions.")
    align_parser.add_argument("source", help="Source document path.")
    align_parser.add_argument("target", help="Target document path.")
    _add_workflow_config_args(align_parser)
    align_parser.add_argument("--coverage-html", default=None, help="Optional HTML coverage review output path.")
    align_parser.add_argument("--coverage-json", default=None, help="Optional JSON coverage review output path.")
    align_parser.add_argument("--summary-json", default=None, help="Optional coverage summary JSON output path.")
    align_parser.add_argument("--marked-source", default=None, help="Optional marked source text output path.")
    align_parser.add_argument("--marked-target", default=None, help="Optional marked target text output path.")
    align_parser.add_argument("--no-expanded-evidence", action="store_true", help="Omit expanded evidence cards from coverage HTML.")
    align_parser.add_argument("--layout", choices=["side-by-side", "stacked"], default="side-by-side", help="HTML report layout.")

    build_index_parser = subparsers.add_parser(
        "build-index",
        help="Build a portable Refmark BM25 search index from a document corpus.",
    )
    build_index_parser.add_argument("corpus", help="Input file or directory.")
    build_index_parser.add_argument("-o", "--output", required=True, help="Output index JSON path.")
    build_index_parser.add_argument("--source", choices=["local", "openrouter"], default="local")
    build_index_parser.add_argument("--model", default="mistralai/mistral-nemo")
    build_index_parser.add_argument("--endpoint", default=OPENROUTER_CHAT_URL)
    build_index_parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    build_index_parser.add_argument("--extensions", default="txt,md,rst,html,htm,docx,pdf")
    build_index_parser.add_argument("--format", dest="marker_format", default="typed_bracket")
    build_index_parser.add_argument("--chunker", default="paragraph")
    build_index_parser.add_argument("--tokens-per-chunk", type=int, default=None)
    build_index_parser.add_argument("--lines-per-chunk", type=int, default=None)
    build_index_parser.add_argument("--min-words", type=int, default=8)
    build_index_parser.add_argument("--questions-per-region", type=int, default=4)
    build_index_parser.add_argument("--keywords-per-region", type=int, default=8)
    build_index_parser.add_argument("--metadata-only", action="store_true", help="Do not include source text in BM25 index text.")
    build_index_parser.add_argument("--limit", type=int, default=None, help="Optional region limit for probes.")
    build_index_parser.add_argument(
        "--exclude-glob",
        action="append",
        default=[],
        help="Glob pattern for source paths to skip. May be passed multiple times.",
    )
    build_index_parser.add_argument("--concurrency", type=int, default=4)
    build_index_parser.add_argument("--sleep", type=float, default=0.0)
    build_index_parser.add_argument("--view-cache", default=None, help="Optional JSONL cache for generated retrieval views.")

    search_index_parser = subparsers.add_parser(
        "search-index",
        help="Search a portable Refmark BM25 search index.",
    )
    search_index_parser.add_argument("index", help="Index JSON produced by build-index.")
    search_index_parser.add_argument("query", help="Search query.")
    search_index_parser.add_argument("--top-k", type=int, default=5)
    search_index_parser.add_argument("--strategy", choices=["flat", "hierarchical", "rerank"], default="flat")
    search_index_parser.add_argument("--doc-top-k", type=int, default=5)
    search_index_parser.add_argument("--candidate-k", type=int, default=30)
    search_index_parser.add_argument("--expand-before", type=int, default=0)
    search_index_parser.add_argument("--expand-after", type=int, default=0)
    search_index_parser.add_argument("--include-excluded", action="store_true", help="Include regions marked excluded from default search.")
    search_index_parser.add_argument("--json", action="store_true", help="Emit JSON hits.")

    inspect_index_parser = subparsers.add_parser(
        "inspect-index",
        help="Report deterministic data-smell diagnostics for a portable Refmark search index.",
    )
    inspect_index_parser.add_argument("index", help="Index JSON produced by build-index.")
    inspect_index_parser.add_argument("--duplicate-threshold", type=float, default=0.86)
    inspect_index_parser.add_argument("--max-pairs", type=int, default=25)
    inspect_index_parser.add_argument("-o", "--output", default=None, help="Optional JSON output path.")

    eval_index_parser = subparsers.add_parser(
        "eval-index",
        help="Evaluate a portable Refmark search index against JSONL query/gold_refs examples.",
    )
    eval_index_parser.add_argument("index", help="Index JSON produced by build-index.")
    eval_index_parser.add_argument("examples", help="JSONL rows with query and gold_refs fields.")
    eval_index_parser.add_argument("--manifest", default=None, help="Optional current region manifest for validation and stale-ref checks.")
    eval_index_parser.add_argument("--top-k", type=int, default=10)
    eval_index_parser.add_argument("--strategy", choices=["flat", "hierarchical", "rerank"], default="flat")
    eval_index_parser.add_argument("--doc-top-k", type=int, default=5)
    eval_index_parser.add_argument("--candidate-k", type=int, default=30)
    eval_index_parser.add_argument("--expand-before", type=int, default=0)
    eval_index_parser.add_argument("--expand-after", type=int, default=0)
    eval_index_parser.add_argument("--retriever-endpoint", default=None, help="HTTP POST retriever endpoint returning refs or hits JSON.")
    eval_index_parser.add_argument("--retriever-results", default=None, help="JSONL query -> refs/hits export from an external retriever.")
    eval_index_parser.add_argument("--retriever-timeout", type=float, default=30.0)
    eval_index_parser.add_argument("--min-hit-at-k", type=float, default=None, help="Warn or fail when hit@k is below this threshold.")
    eval_index_parser.add_argument("--min-hit-at-1", type=float, default=None, help="Warn or fail when hit@1 is below this threshold.")
    eval_index_parser.add_argument("--min-mrr", type=float, default=None, help="Warn or fail when MRR is below this threshold.")
    eval_index_parser.add_argument("--min-gold-coverage", type=float, default=None, help="Warn or fail when average gold coverage is below this threshold.")
    eval_index_parser.add_argument("--max-stale", type=int, default=None, help="Warn or fail when stale examples exceed this count.")
    eval_index_parser.add_argument("--fail-on-regression", action="store_true", help="Exit nonzero when any threshold is breached.")
    eval_index_parser.add_argument("--provenance-out", default=None, help="Optional provenance JSON output path.")
    eval_index_parser.add_argument("--expect-provenance", default=None, help="Fail if current inputs/settings differ from this provenance JSON.")
    eval_index_parser.add_argument("-o", "--output", default=None, help="Optional JSON report output path.")

    feedback_parser = subparsers.add_parser(
        "feedback-diagnostics",
        help="Aggregate production query feedback into reviewable Refmark adaptation candidates.",
    )
    feedback_parser.add_argument("feedback", help="JSONL query feedback rows.")
    feedback_parser.add_argument("--manifest", default=None, help="Optional current region manifest for ref validation.")
    feedback_parser.add_argument("--min-count", type=int, default=2, help="Minimum repeated normalized query count to report.")
    feedback_parser.add_argument("--top-n", type=int, default=25, help="Maximum clusters to emit.")
    feedback_parser.add_argument("-o", "--output", default=None, help="Optional JSON report output path.")

    pack_context_parser = subparsers.add_parser(
        "pack-context",
        help="Pack refs/ranges from a manifest into an ordered evidence text bundle.",
    )
    pack_context_parser.add_argument("manifest", help="Manifest JSONL path.")
    pack_context_parser.add_argument("--refs", required=True, help="Comma-separated refs or ranges.")
    pack_context_parser.add_argument("--format", choices=["text", "json"], default="text")
    pack_context_parser.add_argument("-o", "--output", default=None)

    question_prompt_parser = subparsers.add_parser(
        "question-prompt",
        help="Build an overridable LLM prompt for question generation from manifest refs/ranges.",
    )
    question_prompt_parser.add_argument("manifest", help="Manifest JSONL path.")
    question_prompt_parser.add_argument("--refs", required=True, help="Refs/ranges to use as the gold evidence target.")
    question_prompt_parser.add_argument("--language", default="English", help="Question language to request.")
    question_prompt_parser.add_argument("--count", type=int, default=3, help="Number of questions to request.")
    question_prompt_parser.add_argument("--template", default=None, help="Optional prompt template file.")
    question_prompt_parser.add_argument("-o", "--output", default=None)

    discover_parser = subparsers.add_parser(
        "discover",
        help="Create a corpus discovery manifest for retrieval evaluation and question generation.",
    )
    discover_parser.add_argument("manifest", help="Manifest JSONL path.")
    discover_parser.add_argument("-o", "--output", required=True, help="Discovery JSON output path.")
    discover_parser.add_argument("--mode", choices=["whole", "hierarchical", "windowed"], default="whole")
    discover_parser.add_argument("--source", choices=["local", "openrouter"], default="local")
    discover_parser.add_argument("--model", default="local")
    discover_parser.add_argument("--endpoint", default=OPENROUTER_CHAT_URL)
    discover_parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    discover_parser.add_argument("--max-input-tokens", type=int, default=180000)
    discover_parser.add_argument("--window-tokens", type=int, default=None, help="Process discovery in region-safe windows of this token size.")
    discover_parser.add_argument("--overlap-regions", type=int, default=1, help="Region overlap between discovery windows.")
    discover_parser.add_argument(
        "--cluster-strategy",
        choices=["doc_id", "source_tree", "tag_graph", "balanced_terms", "llm_topics", "llm_intents"],
        default="doc_id",
        help="Discovery cluster recipe for overview/eval boards.",
    )
    discover_parser.add_argument("--target-clusters", type=int, default=40, help="Preferred cluster count for non-structural strategies.")

    review_discovery_parser = subparsers.add_parser(
        "review-discovery",
        help="Create a deterministic review queue for noisy discovery output.",
    )
    review_discovery_parser.add_argument("discovery", help="Discovery JSON path.")
    review_discovery_parser.add_argument("--manifest", default=None, help="Optional manifest JSONL for text/heading checks.")
    review_discovery_parser.add_argument("--max-issues", type=int, default=50)
    review_discovery_parser.add_argument("-o", "--output", default=None, help="Optional JSON output path.")

    discovery_card_parser = subparsers.add_parser(
        "discovery-card",
        help="Build a compact question-generation context card for one ref.",
    )
    discovery_card_parser.add_argument("manifest", help="Manifest JSONL path.")
    discovery_card_parser.add_argument("discovery", help="Discovery JSON path.")
    discovery_card_parser.add_argument("--ref", required=True, help="Stable ref such as doc:P03.")
    discovery_card_parser.add_argument("-o", "--output", default=None, help="Optional JSON output path.")

    discovery_map_parser = subparsers.add_parser(
        "discovery-map",
        help="Render a discovery cluster manifest as an inspectable HTML map.",
    )
    discovery_map_parser.add_argument("manifest", help="Manifest JSONL path.")
    discovery_map_parser.add_argument("discovery", help="Discovery JSON path.")
    discovery_map_parser.add_argument("-o", "--output", required=True, help="HTML output path.")
    discovery_map_parser.add_argument("--title", default="Refmark Discovery Cluster Map")

    repair_clusters_parser = subparsers.add_parser(
        "repair-discovery-clusters",
        help="Ask a discovery agent to repair the cluster layer of a discovery manifest.",
    )
    repair_clusters_parser.add_argument("manifest", help="Manifest JSONL path.")
    repair_clusters_parser.add_argument("discovery", help="Discovery JSON path.")
    repair_clusters_parser.add_argument("-o", "--output", required=True, help="Repaired discovery JSON output path.")
    repair_clusters_parser.add_argument("--source", choices=["local", "openrouter"], default="openrouter")
    repair_clusters_parser.add_argument("--model", default="qwen/qwen-turbo")
    repair_clusters_parser.add_argument("--endpoint", default=OPENROUTER_CHAT_URL)
    repair_clusters_parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    repair_clusters_parser.add_argument(
        "--cluster-strategy",
        choices=["doc_id", "source_tree", "tag_graph", "balanced_terms", "llm_topics", "llm_intents"],
        default=None,
        help="Override the strategy label for repaired clusters.",
    )
    repair_clusters_parser.add_argument("--target-clusters", type=int, default=None)
    repair_clusters_parser.add_argument("--max-input-tokens", type=int, default=40000)

    question_plan_parser = subparsers.add_parser(
        "question-plan",
        help="Create an inspectable direct/concern/adversarial question plan from discovery.",
    )
    question_plan_parser.add_argument("manifest", help="Manifest JSONL path.")
    question_plan_parser.add_argument("discovery", help="Discovery JSON path.")
    question_plan_parser.add_argument("-o", "--output", default=None, help="Optional JSON output path.")
    question_plan_parser.add_argument("--direct-per-region", type=int, default=1)
    question_plan_parser.add_argument("--concern-per-region", type=int, default=1)
    question_plan_parser.add_argument("--adversarial-per-region", type=int, default=1)
    question_plan_parser.add_argument("--include-excluded", action="store_true")

    pipeline_config_parser = subparsers.add_parser(
        "init-pipeline-config",
        help="Write a template config for the full evidence-retrieval pipeline.",
    )
    pipeline_config_parser.add_argument("-o", "--output", required=True, help="YAML config path to write.")
    pipeline_config_parser.add_argument("--check", action="store_true", help="Parse the written config after creating it.")

    pipeline_config_check_parser = subparsers.add_parser(
        "check-pipeline-config",
        help="Validate and summarize a full evidence-retrieval pipeline config.",
    )
    pipeline_config_check_parser.add_argument("config", help="Pipeline YAML/JSON config path.")

    run_pipeline_parser = subparsers.add_parser(
        "run-pipeline",
        help="Run the easy-mode full evidence-retrieval pipeline from a config.",
    )
    run_pipeline_parser.add_argument("config", help="Pipeline YAML/JSON config path.")

    query_pipeline_parser = subparsers.add_parser(
        "query-pipeline",
        help="Query a processed pipeline output directory and return refs/sections.",
    )
    query_pipeline_parser.add_argument("output_dir", help="Directory produced by run-pipeline.")
    query_pipeline_parser.add_argument("query", help="Natural-language search query.")
    query_pipeline_parser.add_argument("--top-k", type=int, default=5)
    query_pipeline_parser.add_argument("--expand-after", type=int, default=1)
    query_pipeline_parser.add_argument("--include-excluded", action="store_true", help="Include query-magnet/noisy regions excluded by default.")
    query_pipeline_parser.add_argument("--json", action="store_true", help="Emit JSON.")

    browser_index_parser = subparsers.add_parser(
        "export-browser-index",
        help="Export a compact browser-searchable BM25 index from a portable Refmark index.",
    )
    browser_index_parser.add_argument("index", help="Index JSON produced by build-index.")
    browser_index_parser.add_argument("-o", "--output", required=True, help="Browser index JSON output path.")
    browser_index_parser.add_argument("--no-text", action="store_true", help="Omit region text/snippets from browser payload.")
    browser_index_parser.add_argument("--max-text-chars", type=int, default=900, help="Max snippet characters per region.")

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
    elif args.command == "toc":
        _handle_toc(args)
    elif args.command == "expand":
        _handle_expand(args)
    elif args.command == "align":
        _handle_align(args)
    elif args.command == "build-index":
        _handle_build_index(args)
    elif args.command == "search-index":
        _handle_search_index(args)
    elif args.command == "inspect-index":
        _handle_inspect_index(args)
    elif args.command == "eval-index":
        _handle_eval_index(args)
    elif args.command == "feedback-diagnostics":
        _handle_feedback_diagnostics(args)
    elif args.command == "pack-context":
        _handle_pack_context(args)
    elif args.command == "question-prompt":
        _handle_question_prompt(args)
    elif args.command == "discover":
        _handle_discover(args)
    elif args.command == "review-discovery":
        _handle_review_discovery(args)
    elif args.command == "discovery-card":
        _handle_discovery_card(args)
    elif args.command == "discovery-map":
        _handle_discovery_map(args)
    elif args.command == "repair-discovery-clusters":
        _handle_repair_discovery_clusters(args)
    elif args.command == "question-plan":
        _handle_question_plan(args)
    elif args.command == "init-pipeline-config":
        _handle_init_pipeline_config(args)
    elif args.command == "check-pipeline-config":
        _handle_check_pipeline_config(args)
    elif args.command == "run-pipeline":
        _handle_run_pipeline(args)
    elif args.command == "query-pipeline":
        _handle_query_pipeline(args)
    elif args.command == "export-browser-index":
        _handle_export_browser_index(args)
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

    result = apply_ref_diff(
        args.file,
        edits,
        dry_run=args.dry_run,
        base_hash=args.base_hash,
        include_diff=args.diff,
    )
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


def _handle_build_index(args):
    extensions = tuple(part.strip() for part in args.extensions.split(",") if part.strip())
    payload = build_search_index(
        args.corpus,
        args.output,
        source=args.source,
        model=args.model,
        endpoint=args.endpoint,
        api_key_env=args.api_key_env,
        extensions=extensions,
        marker_format=args.marker_format,
        chunker=args.chunker,
        tokens_per_chunk=args.tokens_per_chunk,
        lines_per_chunk=args.lines_per_chunk,
        min_words=args.min_words,
        questions_per_region=args.questions_per_region,
        keywords_per_region=args.keywords_per_region,
        include_source=not args.metadata_only,
        limit=args.limit,
        concurrency=args.concurrency,
        sleep=args.sleep,
        view_cache_path=args.view_cache,
        exclude_globs=args.exclude_glob,
    )
    stats = payload["stats"]
    print(f"[OK] Wrote Refmark search index to {args.output}", file=sys.stderr)
    print(
        "[OK] "
        f"{stats['documents']} documents, {stats['regions']} regions, "
        f"{stats.get('default_search_excluded_regions', 0)} excluded by default, "
        f"~{stats['approx_input_tokens']} input tokens, "
        f"estimated OpenRouter cost at mistral-nemo: ${stats['approx_openrouter_cost_usd_at_mistral_nemo']}",
        file=sys.stderr,
    )


def _handle_search_index(args):
    index = load_search_index(args.index)
    if args.strategy == "hierarchical":
        hits = index.search_hierarchical(
            args.query,
            top_k=args.top_k,
            doc_top_k=args.doc_top_k,
            candidate_k=args.candidate_k,
            expand_before=args.expand_before,
            expand_after=args.expand_after,
            include_excluded=args.include_excluded,
        )
    elif args.strategy == "rerank":
        hits = index.search_reranked(
            args.query,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            expand_before=args.expand_before,
            expand_after=args.expand_after,
            include_excluded=args.include_excluded,
        )
    else:
        hits = index.search(
            args.query,
            top_k=args.top_k,
            expand_before=args.expand_before,
            expand_after=args.expand_after,
            include_excluded=args.include_excluded,
        )
    if args.json:
        print(json.dumps([hit.to_dict() for hit in hits], indent=2))
        return
    for hit in hits:
        print(f"{hit.rank}. {hit.stable_ref} score={hit.score}")
        if hit.source_path:
            print(f"   source: {hit.source_path}")
        if hit.summary:
            print(f"   summary: {hit.summary}")
        if hit.context_refs:
            print(f"   context: {', '.join(hit.context_refs)}")
        snippet = " ".join(hit.text.split())
        if len(snippet) > 260:
            snippet = snippet[:257].rstrip() + "..."
        print(f"   text: {snippet}")


def _handle_inspect_index(args):
    index = load_search_index(args.index)
    payload = {
        "schema": "refmark.index_data_smells.v1",
        "index": args.index,
        "diagnostics": analyze_index_smells(
            index,
            duplicate_threshold=args.duplicate_threshold,
            max_pairs=args.max_pairs,
        ),
    }
    output = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"[OK] Wrote index data-smell report to {args.output}", file=sys.stderr)
    else:
        print(output)


def _handle_eval_index(args):
    index = load_search_index(args.index)
    corpus = CorpusMap.from_manifest(args.manifest, metadata={"manifest_path": args.manifest}) if args.manifest else _corpus_from_search_index(index)
    rows = _read_jsonl(args.examples)
    suite = EvalSuite.from_rows(rows, corpus=corpus).with_source_hashes()
    settings = {
        "strategy": args.strategy,
        "top_k": args.top_k,
        "doc_top_k": args.doc_top_k,
        "candidate_k": args.candidate_k,
        "expand_before": args.expand_before,
        "expand_after": args.expand_after,
        "manifest": args.manifest,
        "retriever_endpoint": bool(args.retriever_endpoint),
        "retriever_results": args.retriever_results,
    }
    provenance = build_eval_provenance(
        index_path=args.index,
        examples_path=args.examples,
        settings=settings,
        index_metadata=_read_index_metadata(args.index),
    )
    provenance_check = None
    if args.expect_provenance:
        expected = json.loads(Path(args.expect_provenance).read_text(encoding="utf-8-sig"))
        provenance_check = validate_provenance(expected, provenance)
        if not provenance_check["ok"]:
            print(json.dumps({"provenance_check": provenance_check}, indent=2), file=sys.stderr)
            sys.exit(2)

    if args.retriever_endpoint and args.retriever_results:
        print("Error: use only one of --retriever-endpoint or --retriever-results.", file=sys.stderr)
        sys.exit(1)
    if args.retriever_results:
        retrieve = _jsonl_results_retriever(args.retriever_results)
    elif args.retriever_endpoint:
        retrieve = _http_retriever(args)
    else:
        retrieve = _index_retriever(args, index)

    run = suite.evaluate(retrieve, name=args.strategy, k=args.top_k)
    validation = suite.validate_refs()
    stale = [item.to_dict() for item in suite.stale_examples()]
    ci_status = _eval_ci_status(run.metrics, len(stale), args)
    run_artifact = suite.run_artifact(
        run,
        settings=settings,
        artifacts={
            "index": args.index,
            "manifest": args.manifest,
            "examples": args.examples,
            "retriever_results": args.retriever_results,
        },
    )
    payload = {
        "schema": "refmark.eval_index_report.v1",
        "index": args.index,
        "manifest": args.manifest,
        "examples": args.examples,
        "settings": settings,
        "corpus_fingerprint": corpus.fingerprint,
        "eval_suite_fingerprint": suite.fingerprint,
        "run_fingerprint": run.fingerprint,
        "comparison_key": run_artifact["comparison_key"],
        "metrics": run.metrics,
        "diagnostics": run.diagnostics,
        "validation": validation,
        "stale_examples": stale,
        "ci_status": ci_status,
        "provenance": provenance,
        "provenance_check": provenance_check,
        "run_artifact": run_artifact,
        "results": [item.to_dict() for item in run.examples],
    }
    output = json.dumps(payload, indent=2)
    if args.provenance_out:
        Path(args.provenance_out).write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        print(f"[OK] Wrote eval provenance to {args.provenance_out}", file=sys.stderr)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"[OK] Wrote retrieval eval report to {args.output}", file=sys.stderr)
    else:
        print(output)
    if ci_status["status"] == "fail":
        print(json.dumps({"ci_status": ci_status}, indent=2), file=sys.stderr)
        sys.exit(3)


def _handle_feedback_diagnostics(args):
    events = read_feedback_jsonl(args.feedback)
    corpus = CorpusMap.from_manifest(args.manifest, metadata={"manifest_path": args.manifest}) if args.manifest else None
    report = analyze_feedback(events, corpus=corpus, min_count=args.min_count, top_n=args.top_n)
    output = json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"[OK] Wrote feedback diagnostics to {args.output}", file=sys.stderr)
    else:
        print(output)


def _handle_pack_context(args):
    try:
        refs = citation_refs_to_strings(parse_citation_refs(args.refs))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    pack = CorpusMap.from_manifest(args.manifest).context_pack(refs)
    output = json.dumps(pack.to_dict(), indent=2) if args.format == "json" else pack.text
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"[OK] Wrote context pack with {len(pack.refs)} refs to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(output)


def _handle_question_prompt(args):
    try:
        refs = citation_refs_to_strings(parse_citation_refs(args.refs))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    corpus = CorpusMap.from_manifest(args.manifest)
    pack = corpus.context_pack(refs)
    template = (
        Path(args.template).read_text(encoding="utf-8-sig")
        if args.template
        else DEFAULT_QUESTION_PROMPT_TEMPLATE
    )
    prompt = template.format(
        count=max(args.count, 1),
        language=args.language,
        refs=", ".join(pack.refs),
        refs_json=json.dumps(pack.refs, ensure_ascii=False),
        context=pack.text,
    )
    if args.output:
        Path(args.output).write_text(prompt, encoding="utf-8")
        print(f"[OK] Wrote question-generation prompt for {len(pack.refs)} refs to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(prompt)


def _handle_discover(args):
    records = read_manifest(args.manifest)
    discovery = discover_corpus(
        records,
        mode=args.mode,
        source=args.source,
        model=args.model,
        endpoint=args.endpoint,
        api_key_env=args.api_key_env,
        max_input_tokens=args.max_input_tokens,
        window_tokens=args.window_tokens,
        overlap_regions=args.overlap_regions,
        cluster_strategy=args.cluster_strategy,
        target_clusters=args.target_clusters,
    )
    write_discovery(discovery, args.output)
    print(
        f"[OK] Wrote discovery manifest for {discovery.regions} regions "
        f"({discovery.corpus_tokens} tokens) to {args.output}",
        file=sys.stderr,
    )


def _handle_review_discovery(args):
    discovery = load_discovery(args.discovery)
    records = read_manifest(args.manifest) if args.manifest else []
    issues = review_discovery(discovery, records=records, max_issues=args.max_issues)
    payload = {
        "schema": "refmark.discovery_review.v1",
        "discovery": args.discovery,
        "manifest": args.manifest,
        "issues": [issue.to_dict() for issue in issues],
        "summary": {
            "issues": len(issues),
            "high": sum(1 for issue in issues if issue.severity == "high"),
            "medium": sum(1 for issue in issues if issue.severity == "medium"),
            "low": sum(1 for issue in issues if issue.severity == "low"),
        },
    }
    output = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"[OK] Wrote discovery review with {len(issues)} issues to {args.output}", file=sys.stderr)
    else:
        print(output)


def _handle_discovery_card(args):
    records = read_manifest(args.manifest)
    discovery = load_discovery(args.discovery)
    by_ref = {f"{record.doc_id}:{record.region_id}": record for record in records}
    if args.ref not in by_ref:
        raise SystemExit(f"Unknown ref: {args.ref}")
    card = build_discovery_context_card(discovery, by_ref[args.ref], records=records)
    output = json.dumps({"schema": "refmark.discovery_context_card.v1", "card": card.to_dict()}, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"[OK] Wrote discovery context card for {args.ref} to {args.output}", file=sys.stderr)
    else:
        print(output)


def _handle_discovery_map(args):
    records = read_manifest(args.manifest)
    discovery = load_discovery(args.discovery)
    write_discovery_map_html(records, discovery, args.output, title=args.title)
    print(f"[OK] Wrote discovery cluster map with {len(discovery.clusters)} clusters to {args.output}", file=sys.stderr)


def _handle_repair_discovery_clusters(args):
    records = read_manifest(args.manifest)
    discovery = load_discovery(args.discovery)
    repaired = repair_discovery_clusters(
        discovery,
        records,
        source=args.source,
        model=args.model,
        endpoint=args.endpoint,
        api_key_env=args.api_key_env,
        cluster_strategy=args.cluster_strategy,
        target_clusters=args.target_clusters,
        max_input_tokens=args.max_input_tokens,
    )
    write_discovery(repaired, args.output)
    print(f"[OK] Wrote repaired discovery clusters ({len(repaired.clusters)} clusters) to {args.output}", file=sys.stderr)


def _handle_question_plan(args):
    records = read_manifest(args.manifest)
    discovery = load_discovery(args.discovery)
    plan = build_question_plan(
        discovery,
        records,
        direct_per_region=args.direct_per_region,
        concern_per_region=args.concern_per_region,
        adversarial_per_region=args.adversarial_per_region,
        include_excluded=args.include_excluded,
    )
    output = json.dumps(question_plan_to_dict(plan), indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"[OK] Wrote question plan with {len(plan)} plan items to {args.output}", file=sys.stderr)
    else:
        print(output)


def _handle_init_pipeline_config(args):
    write_full_pipeline_config_template(args.output)
    if args.check:
        config = load_full_pipeline_config(args.output)
        print(json.dumps(config.to_dict(), indent=2))
    else:
        print(f"[OK] Wrote pipeline config template to {args.output}", file=sys.stderr)


def _handle_check_pipeline_config(args):
    config = load_full_pipeline_config(args.config)
    payload = config.to_dict()
    payload["summary"] = {
        "question_generation": f"{config.question_generation.provider}:{config.question_generation.model}",
        "retrieval_views": f"{config.retrieval_views.provider}:{config.retrieval_views.model}",
        "judge_enabled": config.judge.enabled,
        "enabled_embeddings": [item.name for item in config.embeddings if item.enabled],
        "output_dir": config.artifacts.output_dir,
        "cache_dir": config.artifacts.cache_dir,
        "max_iterations": config.loop.max_iterations,
    }
    print(json.dumps(payload, indent=2))


def _handle_run_pipeline(args):
    summary = run_full_pipeline(args.config)
    print(json.dumps(summary.to_dict(), indent=2))


def _handle_query_pipeline(args):
    output_dir = Path(args.output_dir)
    index_path = output_dir / "docs.index.json"
    sections_path = output_dir / "sections.json"
    if not index_path.exists():
        print(f"Error: missing pipeline index: {index_path}", file=sys.stderr)
        sys.exit(1)
    index = load_search_index(index_path)
    hits = index.search_reranked(
        args.query,
        top_k=args.top_k,
        expand_after=args.expand_after,
        include_excluded=args.include_excluded,
    )
    sections = _load_section_entries(sections_path)
    rows = []
    for hit in hits:
        hit_dict = hit.to_dict()
        hit_dict["section"] = _section_for_ref(hit.stable_ref, sections)
        rows.append(hit_dict)
    if args.json:
        print(json.dumps({"query": args.query, "hits": rows}, indent=2))
        return
    for row in rows:
        section = row.get("section")
        label = f" section={section['title']} range={section['range_ref']}" if section else ""
        print(f"{row['rank']}. {row['stable_ref']} score={row['score']}{label}")
        if row.get("source_path"):
            print(f"   source: {row['source_path']}")
        if row.get("context_refs"):
            print(f"   context: {', '.join(row['context_refs'])}")
        snippet = " ".join(str(row.get("text", "")).split())
        if len(snippet) > 260:
            snippet = snippet[:257].rstrip() + "..."
        print(f"   text: {snippet}")


def _handle_export_browser_index(args):
    payload = export_browser_search_index(
        args.index,
        args.output,
        include_text=not args.no_text,
        max_text_chars=args.max_text_chars,
    )
    stats = payload["stats"]
    print(
        f"[OK] Wrote browser Refmark index to {args.output} "
        f"({stats['documents']} documents, {stats['regions']} regions, {stats['tokens']} tokens)",
        file=sys.stderr,
    )


def _read_jsonl(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object row in {path}.")
        rows.append(payload)
    return rows


def _index_retriever(args, index):
    def retrieve(query: str):
        if args.strategy == "hierarchical":
            return index.search_hierarchical(
                query,
                top_k=args.top_k,
                doc_top_k=args.doc_top_k,
                candidate_k=args.candidate_k,
                expand_before=args.expand_before,
                expand_after=args.expand_after,
            )
        if args.strategy == "rerank":
            return index.search_reranked(
                query,
                top_k=args.top_k,
                candidate_k=args.candidate_k,
                expand_before=args.expand_before,
                expand_after=args.expand_after,
            )
        return index.search(
            query,
            top_k=args.top_k,
            expand_before=args.expand_before,
            expand_after=args.expand_after,
        )

    return retrieve


def _http_retriever(args):
    def retrieve(query: str):
        body = json.dumps({"query": query, "top_k": args.top_k}).encode("utf-8")
        request = urllib.request.Request(
            args.retriever_endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=args.retriever_timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return payload.get("hits") or payload.get("refs") or payload.get("results") or []
        raise ValueError("Retriever endpoint must return a JSON list or object with hits/refs/results.")

    return retrieve


def _jsonl_results_retriever(path: str | Path):
    hits_by_query: dict[str, list[object]] = {}
    for row in _read_jsonl(path):
        query = row.get("query")
        if not query:
            raise ValueError(f"Retriever result row in {path} is missing 'query'.")
        hits = row.get("hits") or row.get("refs") or row.get("results") or []
        if not isinstance(hits, list):
            raise ValueError(f"Retriever result row for query {query!r} must contain list hits/refs/results.")
        hits_by_query[str(query)] = hits

    def retrieve(query: str):
        return hits_by_query.get(query, [])

    return retrieve


def _eval_ci_status(metrics: dict[str, float], stale_count: int, args) -> dict[str, object]:
    failures: list[str] = []
    warnings: list[str] = []

    def check_min(metric_name: str, threshold: float | None, label: str) -> None:
        if threshold is None:
            return
        value = metrics.get(metric_name, 0.0)
        if value < threshold:
            failures.append(f"{label} {value:.4f} is below {threshold:.4f}")

    check_min("hit_at_k", args.min_hit_at_k, "hit@k")
    check_min("hit_at_1", args.min_hit_at_1, "hit@1")
    check_min("mrr", args.min_mrr, "MRR")
    check_min("gold_coverage", args.min_gold_coverage, "gold coverage")
    if args.max_stale is not None and stale_count > args.max_stale:
        failures.append(f"stale examples {stale_count} exceeds {args.max_stale}")

    if failures and not args.fail_on_regression:
        warnings = failures
        failures = []

    return {
        "status": "fail" if failures else ("warn" if warnings else "pass"),
        "failures": failures,
        "warnings": warnings,
        "thresholds": {
            "min_hit_at_k": args.min_hit_at_k,
            "min_hit_at_1": args.min_hit_at_1,
            "min_mrr": args.min_mrr,
            "min_gold_coverage": args.min_gold_coverage,
            "max_stale": args.max_stale,
            "fail_on_regression": args.fail_on_regression,
        },
    }


def _load_section_entries(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return [item for item in payload.get("sections", []) if isinstance(item, dict)]


def _section_for_ref(stable_ref: str, sections: list[dict[str, object]]) -> dict[str, object] | None:
    for section in sections:
        refs = section.get("refs", [])
        if isinstance(refs, list) and stable_ref in refs:
            return section
    return None


def _corpus_from_search_index(index) -> CorpusMap:
    records = [
        RegionRecord(
            doc_id=region.doc_id,
            region_id=region.region_id,
            text=region.text,
            start_line=region.ordinal,
            end_line=region.ordinal,
            ordinal=region.ordinal,
            hash=region.hash,
            source_path=region.source_path,
            prev_region_id=region.prev_region_id,
            next_region_id=region.next_region_id,
        )
        for region in index.regions
    ]
    return CorpusMap.from_records(records)


def _read_index_metadata(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return {
        "schema": payload.get("schema"),
        "source_corpus": payload.get("source_corpus"),
        "settings": payload.get("settings", {}),
        "stats": payload.get("stats", {}),
    }


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
                doc_id=_manifest_doc_id(file_path, root=root),
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


def _manifest_doc_id(file_path: Path, *, root: Path) -> str:
    if root.is_file():
        stem = file_path.stem
    else:
        stem = file_path.relative_to(root).with_suffix("").as_posix()
    slug = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()
    return slug or file_path.stem


def _handle_toc(args):
    sections = build_section_map(read_manifest(args.manifest))
    payload = {
        "schema": "refmark.section_map.v1",
        "manifest": args.manifest,
        "sections": [section.to_dict() for section in sections],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {len(sections)} sections to {args.output}", file=sys.stderr)


def _handle_expand(args):
    try:
        refs = citation_refs_to_strings(parse_citation_refs(args.refs))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    records = expand_region_context(
        read_manifest(args.manifest),
        refs,
        doc_id=args.doc_id,
        before=max(args.before, 0),
        after=max(args.after, 0),
        same_parent=args.same_parent,
        include_parent=args.include_parent,
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
        print(f"[OK] Wrote coverage JSON with {len(coverage)} source regions to {args.coverage_json}", file=sys.stderr)
    if args.coverage_html:
        report.write_html(
            args.coverage_html,
            layout=args.layout,
            include_expanded_evidence=not args.no_expanded_evidence,
        )
        print(f"[OK] Wrote coverage HTML to {args.coverage_html}", file=sys.stderr)
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(report.summary, indent=2), encoding="utf-8")
        print(f"[OK] Wrote coverage summary to {args.summary_json}", file=sys.stderr)
    if args.marked_source:
        Path(args.marked_source).write_text(report.source.marked_text, encoding="utf-8")
        print(f"[OK] Wrote marked source to {args.marked_source}", file=sys.stderr)
    if args.marked_target:
        Path(args.marked_target).write_text(report.target.marked_text, encoding="utf-8")
        print(f"[OK] Wrote marked target to {args.marked_target}", file=sys.stderr)
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
