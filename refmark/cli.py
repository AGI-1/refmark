import sys
import argparse
import json
import re
from pathlib import Path
import urllib.request

from refmark.config import load_local_env

load_local_env()

from refmark.adapt_plan import build_adaptation_plan, read_smell_report
from refmark.citations import citation_refs_to_strings, parse_citation_refs
from refmark.core import inject, strip
from refmark.data_smells import build_data_smell_report
from refmark.documents import align_documents, map_document
from refmark.document_io import extract_document_text, text_mapping_extension
from refmark.discovery import build_discovery_context_card, discover_corpus, load_discovery, repair_discovery_clusters, review_discovery, write_discovery
from refmark.discovery_heatmap import write_discovery_map_html
from refmark.ephemeral import apply_ephemeral_edits, build_ephemeral_map, edits_from_json
from refmark.edit import apply_ref_diff
from refmark.feedback import analyze_feedback, read_feedback_jsonl
from refmark.highlight import highlight_refs, render_highlight_html, render_highlight_json, render_highlight_text
from refmark.languages import choose_edit_chunker, choose_live_marker_format, list_supported_languages
from refmark.lifecycle import evaluate_git_revisions, load_summary_rows, render_summary_rows
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
        description="Refmark CLI: map addressable regions, evaluate evidence recovery, and manage ref-based review workflows."
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

    ephemeral_map_parser = subparsers.add_parser(
        "ephemeral-map",
        help="Create a disposable marked view and address map for one-off document work.",
    )
    ephemeral_map_parser.add_argument("file", help="Input document path.")
    ephemeral_map_parser.add_argument("--doc-id", default=None, help="Optional document id for refs.")
    ephemeral_map_parser.add_argument("--json", action="store_true", help="Write a JSON payload instead of marked text.")
    ephemeral_map_parser.add_argument("--manifest", default=None, help="Optional transient manifest JSONL output.")
    ephemeral_map_parser.add_argument("--instructions", action="store_true", help="Prepend edit instructions before marked text.")
    ephemeral_map_parser.add_argument("-o", "--output", default=None, help="Optional output path. Defaults to stdout.")
    _add_workflow_config_args(ephemeral_map_parser)

    ephemeral_apply_parser = subparsers.add_parser(
        "ephemeral-apply",
        help="Apply disposable ref-addressed replacement edits to a document.",
    )
    ephemeral_apply_parser.add_argument("file", help="Input document path.")
    ephemeral_apply_parser.add_argument("-o", "--output", required=True, help="Patched output path.")
    ephemeral_apply_parser.add_argument("--edits-json", default=None, help="Inline JSON edit array or object with edits.")
    ephemeral_apply_parser.add_argument("--edits-file", default=None, help="Path to JSON edit array or object with edits.")
    ephemeral_apply_parser.add_argument("--doc-id", default=None, help="Optional document id used while mapping.")
    ephemeral_apply_parser.add_argument("--dry-run", action="store_true", help="Validate without writing output.")
    _add_workflow_config_args(ephemeral_apply_parser)

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
    eval_index_parser.add_argument("--smell-report-output", default=None, help="Optional first-class data-smell report JSON output path.")
    eval_index_parser.add_argument("--adapt-plan-output", default=None, help="Optional adaptation-plan JSON output path derived from the data-smell report.")
    eval_index_parser.add_argument("-o", "--output", default=None, help="Optional JSON report output path.")

    compare_index_parser = subparsers.add_parser(
        "compare-index",
        help="Compare built-in portable-index retrieval strategies against one eval suite.",
    )
    compare_index_parser.add_argument("index", help="Index JSON produced by build-index.")
    compare_index_parser.add_argument("examples", help="JSONL rows with query and gold_refs fields.")
    compare_index_parser.add_argument("--manifest", default=None, help="Optional current region manifest for validation and stale-ref checks.")
    compare_index_parser.add_argument("--strategies", default="flat,hierarchical,rerank", help="Comma-separated strategies: flat,hierarchical,rerank.")
    compare_index_parser.add_argument("--top-k", type=int, default=10)
    compare_index_parser.add_argument("--doc-top-k", type=int, default=5)
    compare_index_parser.add_argument("--candidate-k", type=int, default=30)
    compare_index_parser.add_argument("--expand-before", type=int, default=0)
    compare_index_parser.add_argument("--expand-after", type=int, default=0)
    compare_index_parser.add_argument("--min-best-hit-at-k", type=float, default=None, help="Fail when the best strategy hit@k is below this threshold.")
    compare_index_parser.add_argument("--fail-on-regression", action="store_true", help="Exit nonzero when a configured threshold is breached.")
    compare_index_parser.add_argument("-o", "--output", default=None, help="Optional JSON report output path.")

    compare_runs_parser = subparsers.add_parser(
        "compare-runs",
        help="Compare saved Refmark eval artifacts or eval-index reports without rerunning retrieval.",
    )
    compare_runs_parser.add_argument("reports", nargs="+", help="Eval-index reports or eval_run_artifact JSON files.")
    compare_runs_parser.add_argument("--allow-mismatch", action="store_true", help="Allow different corpus/eval fingerprints and mark the report non-comparable.")
    compare_runs_parser.add_argument("--baseline", default=None, help="Optional report/run name to use as delta baseline. Defaults to the first report.")
    compare_runs_parser.add_argument("--min-best-hit-at-k", type=float, default=None, help="Fail when the best run hit@k is below this threshold.")
    compare_runs_parser.add_argument("--fail-on-regression", action="store_true", help="Exit nonzero when compatibility or threshold checks fail.")
    compare_runs_parser.add_argument("-o", "--output", default=None, help="Optional JSON report output path.")

    ci_parser = subparsers.add_parser(
        "ci",
        help="Run the default evidence-CI loop: map, build-index, eval-index, compare-index.",
    )
    ci_parser.add_argument("corpus", help="Input file or directory to map and index.")
    ci_parser.add_argument("examples", help="JSONL rows with query and gold_refs fields.")
    ci_parser.add_argument("--out-dir", default="runs/refmark_ci", help="Directory for generated CI artifacts.")
    ci_parser.add_argument("--source", choices=["local", "openrouter"], default="local", help="Retrieval-view source for build-index.")
    ci_parser.add_argument("--model", default="mistralai/mistral-nemo", help="OpenRouter model when --source=openrouter.")
    ci_parser.add_argument("--endpoint", default=OPENROUTER_CHAT_URL)
    ci_parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    ci_parser.add_argument("--extensions", default="txt,md,rst,html,htm,docx,pdf")
    ci_parser.add_argument("--format", dest="marker_format", default="typed_bracket", help="Marker format for mapping/indexing.")
    ci_parser.add_argument("--chunker", default="paragraph")
    ci_parser.add_argument("--min-words", type=int, default=8)
    ci_parser.add_argument("--questions-per-region", type=int, default=4)
    ci_parser.add_argument("--keywords-per-region", type=int, default=8)
    ci_parser.add_argument("--top-k", type=int, default=10)
    ci_parser.add_argument("--strategies", default="flat,hierarchical,rerank")
    ci_parser.add_argument("--min-hit-at-k", type=float, default=None, help="Eval threshold for the default eval-index run.")
    ci_parser.add_argument("--min-best-hit-at-k", type=float, default=None, help="Compare-index threshold for the best built-in strategy.")
    ci_parser.add_argument("--max-stale", type=int, default=0)
    ci_parser.add_argument("--fail-on-regression", action="store_true")

    adapt_plan_parser = subparsers.add_parser(
        "adapt-plan",
        help="Convert a Refmark data-smell report into reviewable adaptation actions.",
    )
    adapt_plan_parser.add_argument("smell_report", help="JSON report with schema refmark.data_smells.v1.")
    adapt_plan_parser.add_argument("--max-actions", type=int, default=80)
    adapt_plan_parser.add_argument("-o", "--output", default=None, help="Optional JSON output path. Defaults to stdout.")

    feedback_parser = subparsers.add_parser(
        "feedback-diagnostics",
        help="Aggregate production query feedback into reviewable Refmark adaptation candidates.",
    )
    feedback_parser.add_argument("feedback", help="JSONL query feedback rows.")
    feedback_parser.add_argument("--manifest", default=None, help="Optional current region manifest for ref validation.")
    feedback_parser.add_argument("--min-count", type=int, default=2, help="Minimum repeated normalized query count to report.")
    feedback_parser.add_argument("--top-n", type=int, default=25, help="Maximum clusters to emit.")
    feedback_parser.add_argument("-o", "--output", default=None, help="Optional JSON report output path.")

    lifecycle_git_parser = subparsers.add_parser(
        "lifecycle-git",
        help="Evaluate evidence-label stability across Git documentation revisions.",
    )
    lifecycle_git_parser.add_argument("--repo-url", required=True, help="Git repository URL.")
    lifecycle_git_parser.add_argument("--old-ref", required=True, help="Base Git ref/tag.")
    lifecycle_git_parser.add_argument("--new-refs", required=True, help="Comma-separated target Git refs/tags.")
    lifecycle_git_parser.add_argument("--subdir", required=True, help="Documentation subdirectory inside the repo.")
    lifecycle_git_parser.add_argument("--work-dir", required=True, help="Working directory for clone/export cache.")
    lifecycle_git_parser.add_argument("-o", "--output", required=True, help="Full JSON benchmark output path.")
    lifecycle_git_parser.add_argument("--summary-output", default=None, help="Optional compact summary JSON rows output path.")
    lifecycle_git_parser.add_argument("--csv-output", default=None, help="Optional compact summary CSV output path.")
    lifecycle_git_parser.add_argument("--region-tokens", type=int, default=110)
    lifecycle_git_parser.add_argument("--region-stride", type=int, default=110)
    lifecycle_git_parser.add_argument("--max-files", type=int, default=0, help="Optional file cap for smoke runs.")

    lifecycle_summary_parser = subparsers.add_parser(
        "lifecycle-summarize",
        help="Render lifecycle benchmark summary_rows as Markdown, JSON, or CSV.",
    )
    lifecycle_summary_parser.add_argument("inputs", nargs="+", help="Lifecycle benchmark JSON files or summary JSON arrays.")
    lifecycle_summary_parser.add_argument("--format", choices=["markdown", "json", "csv"], default="markdown")
    lifecycle_summary_parser.add_argument(
        "--columns",
        default="",
        help="Optional comma-separated output columns. Defaults to the standard lifecycle table.",
    )
    lifecycle_summary_parser.add_argument("-o", "--output", default=None, help="Optional output path. Defaults to stdout.")

    lifecycle_validate_parser = subparsers.add_parser(
        "lifecycle-validate-labels",
        help="Validate query->gold_refs labels against a current manifest and optional previous revision.",
    )
    lifecycle_validate_parser.add_argument("current_manifest", help="Current region manifest JSONL.")
    lifecycle_validate_parser.add_argument("examples", help="Eval JSONL rows with query, gold_refs, and optional source_hashes.")
    lifecycle_validate_parser.add_argument("--previous-manifest", default=None, help="Optional previous region manifest JSONL for revision diff.")
    lifecycle_validate_parser.add_argument("--current-revision", default=None)
    lifecycle_validate_parser.add_argument("--previous-revision", default=None)
    lifecycle_validate_parser.add_argument("--attach-source-hashes", action="store_true", help="Attach current hashes before validation.")
    lifecycle_validate_parser.add_argument("--max-stale", type=int, default=None, help="Fail if stale examples exceed this count.")
    lifecycle_validate_parser.add_argument("--max-changed-refs", type=int, default=None, help="Fail if changed refs exceed this count.")
    lifecycle_validate_parser.add_argument("--max-removed-refs", type=int, default=None, help="Fail if removed/deleted refs exceed this count.")
    lifecycle_validate_parser.add_argument("-o", "--output", default=None, help="Optional JSON report output path.")

    manifest_diff_parser = subparsers.add_parser(
        "manifest-diff",
        help="Compare two region manifests and optionally report affected eval examples.",
    )
    manifest_diff_parser.add_argument("previous_manifest", help="Previous region manifest JSONL.")
    manifest_diff_parser.add_argument("current_manifest", help="Current region manifest JSONL.")
    manifest_diff_parser.add_argument("--examples", default=None, help="Optional eval JSONL rows with query and gold_refs.")
    manifest_diff_parser.add_argument("--previous-revision", default=None)
    manifest_diff_parser.add_argument("--current-revision", default=None)
    manifest_diff_parser.add_argument("--max-stale", type=int, default=None, help="Fail if affected examples exceed this count.")
    manifest_diff_parser.add_argument("--max-changed-refs", type=int, default=None, help="Fail if changed refs exceed this count.")
    manifest_diff_parser.add_argument("--max-removed-refs", type=int, default=None, help="Fail if removed/deleted refs exceed this count.")
    manifest_diff_parser.add_argument("-o", "--output", default=None, help="Optional JSON report output path.")

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
    elif args.command == "ephemeral-map":
        _handle_ephemeral_map(args)
    elif args.command == "ephemeral-apply":
        _handle_ephemeral_apply(args)
    elif args.command == "build-index":
        _handle_build_index(args)
    elif args.command == "search-index":
        _handle_search_index(args)
    elif args.command == "inspect-index":
        _handle_inspect_index(args)
    elif args.command == "eval-index":
        _handle_eval_index(args)
    elif args.command == "compare-index":
        _handle_compare_index(args)
    elif args.command == "compare-runs":
        _handle_compare_runs(args)
    elif args.command == "ci":
        _handle_ci(args)
    elif args.command == "adapt-plan":
        _handle_adapt_plan(args)
    elif args.command == "feedback-diagnostics":
        _handle_feedback_diagnostics(args)
    elif args.command == "lifecycle-git":
        _handle_lifecycle_git(args)
    elif args.command == "lifecycle-summarize":
        _handle_lifecycle_summarize(args)
    elif args.command == "lifecycle-validate-labels":
        _handle_lifecycle_validate_labels(args)
    elif args.command == "manifest-diff":
        _handle_manifest_diff(args)
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
    smell_report = build_data_smell_report(suite, run).to_dict()
    adaptation_plan = build_adaptation_plan(smell_report).to_dict()
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
        "data_smells": smell_report,
        "adaptation_plan": adaptation_plan,
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
    if args.smell_report_output:
        Path(args.smell_report_output).write_text(json.dumps(smell_report, indent=2), encoding="utf-8")
        print(f"[OK] Wrote data-smell report to {args.smell_report_output}", file=sys.stderr)
    if args.adapt_plan_output:
        Path(args.adapt_plan_output).write_text(json.dumps(adaptation_plan, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] Wrote adaptation plan to {args.adapt_plan_output}", file=sys.stderr)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"[OK] Wrote retrieval eval report to {args.output}", file=sys.stderr)
    else:
        print(output)
    if ci_status["status"] == "fail":
        print(json.dumps({"ci_status": ci_status}, indent=2), file=sys.stderr)
        sys.exit(3)


def _handle_compare_index(args):
    index = load_search_index(args.index)
    corpus = CorpusMap.from_manifest(args.manifest, metadata={"manifest_path": args.manifest}) if args.manifest else _corpus_from_search_index(index)
    rows = _read_jsonl(args.examples)
    suite = EvalSuite.from_rows(rows, corpus=corpus).with_source_hashes()
    strategies = _parse_index_strategies(args.strategies)
    base_settings = {
        "top_k": args.top_k,
        "doc_top_k": args.doc_top_k,
        "candidate_k": args.candidate_k,
        "expand_before": args.expand_before,
        "expand_after": args.expand_after,
        "manifest": args.manifest,
    }
    runs: dict[str, dict[str, object]] = {}
    run_artifacts: dict[str, dict[str, object]] = {}
    smell_summaries: dict[str, dict[str, object]] = {}
    for strategy in strategies:
        retrieve = _index_strategy_retriever(
            index,
            strategy=strategy,
            top_k=args.top_k,
            doc_top_k=args.doc_top_k,
            candidate_k=args.candidate_k,
            expand_before=args.expand_before,
            expand_after=args.expand_after,
        )
        run = suite.evaluate(retrieve, name=strategy, k=args.top_k)
        settings = {"strategy": strategy, **base_settings}
        artifact = suite.run_artifact(
            run,
            settings=settings,
            artifacts={"index": args.index, "manifest": args.manifest, "examples": args.examples},
        )
        smell_report = build_data_smell_report(suite, run).to_dict()
        runs[strategy] = {
            "metrics": run.metrics,
            "diagnostics": run.diagnostics,
            "fingerprint": run.fingerprint,
            "comparison_key": artifact["comparison_key"],
        }
        run_artifacts[strategy] = artifact
        smell_summaries[strategy] = smell_report["summary"]
    best = _best_index_strategy(runs)
    stale = [item.to_dict() for item in suite.stale_examples()]
    ci_status = _compare_index_ci_status(
        best_hit_at_k=best["metrics"].get("hit_at_k", 0.0) if best else 0.0,
        min_best_hit_at_k=args.min_best_hit_at_k,
        fail_on_regression=args.fail_on_regression,
    )
    payload = {
        "schema": "refmark.compare_index_report.v1",
        "index": args.index,
        "manifest": args.manifest,
        "examples": args.examples,
        "settings": {**base_settings, "strategies": strategies},
        "corpus_fingerprint": corpus.fingerprint,
        "eval_suite_fingerprint": suite.fingerprint,
        "validation": suite.validate_refs(),
        "stale_examples": stale,
        "stale_example_count": len(stale),
        "runs": runs,
        "smell_summaries": smell_summaries,
        "run_artifacts": run_artifacts,
        "best_by_hit_at_k": best,
        "ci_status": ci_status,
        "status": ci_status["status"],
    }
    output = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"[OK] Wrote index comparison report to {args.output}", file=sys.stderr)
    else:
        print(output)
    if ci_status["status"] == "fail":
        print(json.dumps({"ci_status": ci_status}, indent=2), file=sys.stderr)
        sys.exit(3)


def _handle_compare_runs(args):
    runs = [_load_eval_run_summary(path) for path in args.reports]
    compatibility = _run_comparison_compatibility(runs)
    if not args.allow_mismatch and not compatibility["same_corpus_and_eval"]:
        compatibility["status"] = "fail"
    baseline = _choose_baseline_run(runs, args.baseline)
    best = _best_saved_run(runs)
    table = [_saved_run_table_row(run, baseline) for run in runs]
    ci_status = _compare_runs_ci_status(
        compatibility=compatibility,
        best_hit_at_k=best["metrics"].get("hit_at_k", 0.0) if best else 0.0,
        min_best_hit_at_k=args.min_best_hit_at_k,
        fail_on_regression=args.fail_on_regression,
    )
    payload = {
        "schema": "refmark.compare_runs_report.v1",
        "inputs": args.reports,
        "compatibility": compatibility,
        "baseline": baseline["name"] if baseline else None,
        "best_by_hit_at_k": best,
        "runs": runs,
        "table": table,
        "ci_status": ci_status,
        "status": ci_status["status"],
    }
    output = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"[OK] Wrote run comparison report to {args.output}", file=sys.stderr)
    else:
        print(output)
    if ci_status["status"] == "fail":
        print(json.dumps({"ci_status": ci_status}, indent=2), file=sys.stderr)
        sys.exit(3)


def _handle_ci(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "corpus.refmark.jsonl"
    marked_dir = out_dir / "marked"
    index = out_dir / "docs.index.json"
    eval_report = out_dir / "eval.json"
    smell_report = out_dir / "smells.json"
    adaptation_plan = out_dir / "adaptation_plan.json"
    compare_report = out_dir / "compare_index.json"

    map_args = argparse.Namespace(
        path=args.corpus,
        output=str(manifest),
        marked_dir=str(marked_dir),
        config=None,
        density=None,
        marker_style=None,
        marker_format=args.marker_format,
        chunker=args.chunker,
        min_words=args.min_words,
        lines_per_chunk=None,
        tokens_per_chunk=None,
        ignore_title=False,
        top_k=None,
        threshold=None,
        expand_before=None,
        expand_after=None,
        no_numeric_checks=False,
    )
    _handle_map(map_args)

    build_args = argparse.Namespace(
        corpus=args.corpus,
        output=str(index),
        source=args.source,
        model=args.model,
        endpoint=args.endpoint,
        api_key_env=args.api_key_env,
        extensions=args.extensions,
        marker_format=args.marker_format,
        chunker=args.chunker,
        tokens_per_chunk=None,
        lines_per_chunk=None,
        min_words=args.min_words,
        questions_per_region=args.questions_per_region,
        keywords_per_region=args.keywords_per_region,
        metadata_only=False,
        limit=None,
        exclude_glob=[],
        concurrency=4,
        sleep=0.0,
        view_cache=str(out_dir / "view_cache.jsonl") if args.source == "openrouter" else None,
    )
    _handle_build_index(build_args)

    eval_args = argparse.Namespace(
        index=str(index),
        examples=args.examples,
        manifest=str(manifest),
        top_k=args.top_k,
        strategy="rerank",
        doc_top_k=5,
        candidate_k=30,
        expand_before=0,
        expand_after=0,
        retriever_endpoint=None,
        retriever_results=None,
        retriever_timeout=30.0,
        min_hit_at_k=args.min_hit_at_k,
        min_hit_at_1=None,
        min_mrr=None,
        min_gold_coverage=None,
        max_stale=args.max_stale,
        fail_on_regression=args.fail_on_regression,
        provenance_out=str(out_dir / "provenance.json"),
        expect_provenance=None,
        smell_report_output=str(smell_report),
        adapt_plan_output=str(adaptation_plan),
        output=str(eval_report),
    )
    _handle_eval_index(eval_args)

    compare_args = argparse.Namespace(
        index=str(index),
        examples=args.examples,
        manifest=str(manifest),
        strategies=args.strategies,
        top_k=args.top_k,
        doc_top_k=5,
        candidate_k=30,
        expand_before=0,
        expand_after=0,
        min_best_hit_at_k=args.min_best_hit_at_k,
        fail_on_regression=args.fail_on_regression,
        output=str(compare_report),
    )
    _handle_compare_index(compare_args)

    payload = {
        "schema": "refmark.ci_summary.v1",
        "corpus": args.corpus,
        "examples": args.examples,
        "out_dir": str(out_dir),
        "artifacts": {
            "manifest": str(manifest),
            "marked_dir": str(marked_dir),
            "index": str(index),
            "eval": str(eval_report),
            "provenance": str(out_dir / "provenance.json"),
            "smells": str(smell_report),
            "adaptation_plan": str(adaptation_plan),
            "comparison": str(compare_report),
        },
    }
    print(json.dumps(payload, indent=2))


def _handle_adapt_plan(args):
    smell_report = read_smell_report(args.smell_report)
    plan = build_adaptation_plan(smell_report, max_actions=args.max_actions)
    output = json.dumps(plan.to_dict(), indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"[OK] Wrote adaptation plan to {args.output}", file=sys.stderr)
    else:
        print(output)


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


def _handle_lifecycle_git(args):
    new_refs = [ref.strip() for ref in args.new_refs.split(",") if ref.strip()]
    if not new_refs:
        print("Error: --new-refs must contain at least one Git ref.", file=sys.stderr)
        sys.exit(1)
    payload = evaluate_git_revisions(
        repo_url=args.repo_url,
        old_ref=args.old_ref,
        new_refs=new_refs,
        subdir=args.subdir,
        work_dir=args.work_dir,
        output=args.output,
        summary_output=args.summary_output,
        csv_output=args.csv_output,
        region_tokens=args.region_tokens,
        region_stride=args.region_stride,
        max_files=args.max_files,
    )
    print(json.dumps(payload, indent=2))


def _handle_lifecycle_summarize(args):
    try:
        rows = load_summary_rows(args.inputs)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    columns = [column.strip() for column in args.columns.split(",") if column.strip()] if args.columns else None
    rendered = render_summary_rows(rows, columns=columns, output_format=args.format)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(rendered, encoding="utf-8")
        print(f"[OK] Wrote lifecycle summary to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(rendered)


def _handle_lifecycle_validate_labels(args):
    current = CorpusMap.from_manifest(
        args.current_manifest,
        revision_id=args.current_revision,
        metadata={"manifest_path": args.current_manifest},
    )
    suite = EvalSuite.from_jsonl(args.examples, corpus=current, attach_source_hashes=args.attach_source_hashes)
    validation = suite.validate_refs()
    stale = [item.to_dict() for item in suite.stale_examples()]
    diff_payload = None
    if args.previous_manifest:
        previous = CorpusMap.from_manifest(
            args.previous_manifest,
            revision_id=args.previous_revision,
            metadata={"manifest_path": args.previous_manifest},
        )
        diff_payload = current.diff_revision(previous).to_dict()
    ci_status = _lifecycle_ci_status(
        stale_example_count=len(stale),
        revision_diff=diff_payload,
        max_stale=args.max_stale,
        max_changed_refs=args.max_changed_refs,
        max_removed_refs=args.max_removed_refs,
    )
    payload = {
        "schema": "refmark.lifecycle_label_validation.v1",
        "current_manifest": args.current_manifest,
        "previous_manifest": args.previous_manifest,
        "current_revision": args.current_revision,
        "previous_revision": args.previous_revision,
        "corpus": current.snapshot().to_dict(),
        "eval_suite": suite.summary(),
        "validation": validation,
        "stale_examples": stale,
        "stale_example_count": len(stale),
        "revision_diff": diff_payload,
        "ci_status": ci_status,
        "status": ci_status["status"],
    }
    output = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"[OK] Wrote lifecycle label validation to {args.output}", file=sys.stderr)
    else:
        print(output)
    if payload["status"] == "fail":
        sys.exit(3)


def _handle_manifest_diff(args):
    previous = CorpusMap.from_manifest(
        args.previous_manifest,
        revision_id=args.previous_revision,
        metadata={"manifest_path": args.previous_manifest},
    )
    current = CorpusMap.from_manifest(
        args.current_manifest,
        revision_id=args.current_revision,
        metadata={"manifest_path": args.current_manifest},
    )
    diff = current.diff_revision(previous)
    diff_payload = diff.to_dict()
    stale: list[dict[str, object]] = []
    eval_suite_summary = None
    if args.examples:
        suite = EvalSuite.from_jsonl(args.examples, corpus=current)
        stale = [item.to_dict() for item in diff.stale_examples(suite.examples)]
        eval_suite_summary = suite.summary()
    ci_status = _lifecycle_ci_status(
        stale_example_count=len(stale),
        revision_diff=diff_payload,
        max_stale=args.max_stale,
        max_changed_refs=args.max_changed_refs,
        max_removed_refs=args.max_removed_refs,
    )
    payload = {
        "schema": "refmark.manifest_diff.v1",
        "previous_manifest": args.previous_manifest,
        "current_manifest": args.current_manifest,
        "previous_revision": args.previous_revision,
        "current_revision": args.current_revision,
        "previous_corpus": previous.snapshot().to_dict(),
        "current_corpus": current.snapshot().to_dict(),
        "revision_diff": diff_payload,
        "affected_examples": stale,
        "affected_example_count": len(stale),
        "eval_suite": eval_suite_summary,
        "summary": {
            "added_refs": len(diff.added_refs),
            "removed_refs": len(diff.removed_refs),
            "changed_refs": len(diff.changed_refs),
            "unchanged_refs": len(diff.unchanged_refs),
            "affected_refs": len(diff.affected_refs()),
            "affected_examples": len(stale),
        },
        "ci_status": ci_status,
        "status": ci_status["status"],
    }
    output = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"[OK] Wrote manifest diff to {args.output}", file=sys.stderr)
    else:
        print(output)
    if payload["status"] == "fail":
        sys.exit(3)


def _lifecycle_ci_status(
    *,
    stale_example_count: int,
    revision_diff: dict[str, object] | None,
    max_stale: int | None,
    max_changed_refs: int | None,
    max_removed_refs: int | None,
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    if max_stale is not None and stale_example_count > max_stale:
        failures.append({"metric": "stale_examples", "value": stale_example_count, "threshold": max_stale})
    changed_count = len(revision_diff.get("changed_refs", [])) if revision_diff else 0
    removed_count = len(revision_diff.get("removed_refs", [])) if revision_diff else 0
    if max_changed_refs is not None and changed_count > max_changed_refs:
        failures.append({"metric": "changed_refs", "value": changed_count, "threshold": max_changed_refs})
    if max_removed_refs is not None and removed_count > max_removed_refs:
        failures.append({"metric": "removed_refs", "value": removed_count, "threshold": max_removed_refs})
    return {
        "status": "fail" if failures else "ok",
        "exit_code": 3 if failures else 0,
        "thresholds": {
            "max_stale": max_stale,
            "max_changed_refs": max_changed_refs,
            "max_removed_refs": max_removed_refs,
        },
        "counts": {
            "stale_examples": stale_example_count,
            "changed_refs": changed_count,
            "removed_refs": removed_count,
        },
        "failures": failures,
    }


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
        return _index_strategy_retriever(
            index,
            strategy=args.strategy,
            top_k=args.top_k,
            doc_top_k=args.doc_top_k,
            candidate_k=args.candidate_k,
            expand_before=args.expand_before,
            expand_after=args.expand_after,
        )(query)

    return retrieve


def _index_strategy_retriever(
    index,
    *,
    strategy: str,
    top_k: int,
    doc_top_k: int,
    candidate_k: int,
    expand_before: int,
    expand_after: int,
):
    def retrieve(query: str):
        if strategy == "hierarchical":
            return index.search_hierarchical(
                query,
                top_k=top_k,
                doc_top_k=doc_top_k,
                candidate_k=candidate_k,
                expand_before=expand_before,
                expand_after=expand_after,
            )
        if strategy == "rerank":
            return index.search_reranked(
                query,
                top_k=top_k,
                candidate_k=candidate_k,
                expand_before=expand_before,
                expand_after=expand_after,
            )
        return index.search(
            query,
            top_k=top_k,
            expand_before=expand_before,
            expand_after=expand_after,
        )

    return retrieve


def _parse_index_strategies(raw: str) -> list[str]:
    allowed = {"flat", "hierarchical", "rerank"}
    strategies = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = sorted(set(strategies) - allowed)
    if not strategies:
        print("Error: --strategies must include at least one strategy.", file=sys.stderr)
        sys.exit(1)
    if invalid:
        print(f"Error: unsupported strategies: {', '.join(invalid)}", file=sys.stderr)
        sys.exit(1)
    return list(dict.fromkeys(strategies))


def _best_index_strategy(runs: dict[str, dict[str, object]]) -> dict[str, object]:
    def sort_key(item: tuple[str, dict[str, object]]) -> tuple[float, float, float, float, float, float]:
        strategy, payload = item
        metrics = payload["metrics"]
        if not isinstance(metrics, dict):
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return (
            float(metrics.get("hit_at_k", 0.0)),
            float(metrics.get("hit_at_1", 0.0)),
            float(metrics.get("mrr", 0.0)),
            float(metrics.get("gold_coverage", 0.0)),
            float(metrics.get("region_precision", 0.0)),
            -float(metrics.get("avg_context_refs", 0.0)),
        )

    if not runs:
        return {}
    strategy, payload = max(runs.items(), key=sort_key)
    metrics = payload["metrics"] if isinstance(payload.get("metrics"), dict) else {}
    return {"strategy": strategy, "metrics": metrics}


def _compare_index_ci_status(
    *,
    best_hit_at_k: float,
    min_best_hit_at_k: float | None,
    fail_on_regression: bool,
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    if min_best_hit_at_k is not None and best_hit_at_k < min_best_hit_at_k:
        failures.append({"metric": "best_hit_at_k", "value": best_hit_at_k, "threshold": min_best_hit_at_k})
    status = "fail" if fail_on_regression and failures else "ok"
    return {
        "status": status,
        "exit_code": 3 if status == "fail" else 0,
        "thresholds": {"min_best_hit_at_k": min_best_hit_at_k},
        "counts": {"failures": len(failures)},
        "failures": failures,
    }


def _load_eval_run_summary(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    if payload.get("schema") == "refmark.eval_index_report.v1":
        artifact = payload.get("run_artifact")
        if not isinstance(artifact, dict):
            raise ValueError(f"eval-index report {path} does not contain run_artifact.")
        name = str(payload.get("settings", {}).get("strategy") or artifact.get("run_name") or Path(path).stem)
        return _summary_from_eval_artifact(artifact, source_path=path, name=name, parent_schema=str(payload.get("schema")))
    if payload.get("schema") == "refmark.eval_run_artifact.v1":
        return _summary_from_eval_artifact(payload, source_path=path, name=str(payload.get("run_name") or Path(path).stem))
    raise ValueError(f"Unsupported eval run report schema in {path}: {payload.get('schema')!r}")


def _summary_from_eval_artifact(
    artifact: dict[str, object],
    *,
    source_path: str | Path,
    name: str,
    parent_schema: str | None = None,
) -> dict[str, object]:
    corpus = artifact.get("corpus", {})
    eval_suite = artifact.get("eval_suite", {})
    metrics = artifact.get("metrics", {})
    if not isinstance(corpus, dict) or not isinstance(eval_suite, dict) or not isinstance(metrics, dict):
        raise ValueError(f"Malformed eval run artifact in {source_path}.")
    return {
        "name": name,
        "source_path": str(source_path),
        "schema": artifact.get("schema"),
        "parent_schema": parent_schema,
        "run_fingerprint": artifact.get("run_fingerprint"),
        "comparison_key": artifact.get("comparison_key"),
        "corpus_fingerprint": corpus.get("fingerprint"),
        "eval_suite_fingerprint": eval_suite.get("fingerprint"),
        "settings": artifact.get("settings", {}),
        "artifacts": artifact.get("artifacts", {}),
        "metrics": metrics,
    }


def _run_comparison_compatibility(runs: list[dict[str, object]]) -> dict[str, object]:
    corpus_fingerprints = sorted({str(run.get("corpus_fingerprint")) for run in runs})
    eval_fingerprints = sorted({str(run.get("eval_suite_fingerprint")) for run in runs})
    same = len(corpus_fingerprints) == 1 and len(eval_fingerprints) == 1
    return {
        "status": "ok" if same else "warn",
        "same_corpus_and_eval": same,
        "corpus_fingerprints": corpus_fingerprints,
        "eval_suite_fingerprints": eval_fingerprints,
        "run_count": len(runs),
    }


def _choose_baseline_run(runs: list[dict[str, object]], baseline: str | None) -> dict[str, object] | None:
    if not runs:
        return None
    if baseline is None:
        return runs[0]
    matches = [
        run
        for run in runs
        if run.get("name") == baseline or Path(str(run.get("source_path", ""))).name == baseline or str(run.get("source_path")) == baseline
    ]
    if not matches:
        print(f"Error: --baseline did not match any run: {baseline}", file=sys.stderr)
        sys.exit(1)
    return matches[0]


def _best_saved_run(runs: list[dict[str, object]]) -> dict[str, object]:
    def sort_key(run: dict[str, object]) -> tuple[float, float, float, float, float, float]:
        metrics = run.get("metrics", {})
        if not isinstance(metrics, dict):
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return (
            float(metrics.get("hit_at_k", 0.0)),
            float(metrics.get("hit_at_1", 0.0)),
            float(metrics.get("mrr", 0.0)),
            float(metrics.get("gold_coverage", 0.0)),
            float(metrics.get("region_precision", 0.0)),
            -float(metrics.get("avg_context_refs", 0.0)),
        )

    if not runs:
        return {}
    run = max(runs, key=sort_key)
    metrics = run["metrics"] if isinstance(run.get("metrics"), dict) else {}
    return {"name": run.get("name"), "source_path": run.get("source_path"), "metrics": metrics}


def _saved_run_table_row(run: dict[str, object], baseline: dict[str, object] | None) -> dict[str, object]:
    metrics = run.get("metrics", {})
    base_metrics = baseline.get("metrics", {}) if baseline else {}
    if not isinstance(metrics, dict):
        metrics = {}
    if not isinstance(base_metrics, dict):
        base_metrics = {}
    keys = ["hit_at_1", "hit_at_k", "mrr", "gold_coverage", "region_precision", "avg_context_refs"]
    row = {
        "name": run.get("name"),
        "source_path": run.get("source_path"),
        "comparison_key": run.get("comparison_key"),
        "settings": run.get("settings", {}),
        "metrics": {key: metrics.get(key) for key in keys if key in metrics},
    }
    if baseline:
        row["delta_vs_baseline"] = {
            key: _metric_delta(metrics.get(key), base_metrics.get(key))
            for key in keys
            if key in metrics and key in base_metrics
        }
    return row


def _metric_delta(value: object, baseline: object) -> float | None:
    try:
        return round(float(value) - float(baseline), 6)
    except (TypeError, ValueError):
        return None


def _compare_runs_ci_status(
    *,
    compatibility: dict[str, object],
    best_hit_at_k: float,
    min_best_hit_at_k: float | None,
    fail_on_regression: bool,
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    if compatibility.get("status") == "fail":
        failures.append({"metric": "compatibility", "value": "mismatch", "threshold": "same_corpus_and_eval"})
    if min_best_hit_at_k is not None and best_hit_at_k < min_best_hit_at_k:
        failures.append({"metric": "best_hit_at_k", "value": best_hit_at_k, "threshold": min_best_hit_at_k})
    status = "fail" if fail_on_regression and failures else "ok"
    return {
        "status": status,
        "exit_code": 3 if status == "fail" else 0,
        "thresholds": {"min_best_hit_at_k": min_best_hit_at_k},
        "counts": {"failures": len(failures)},
        "failures": failures,
    }


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


def _handle_ephemeral_map(args):
    path = Path(args.file)
    if not path.exists():
        print(f"Error: File '{path}' not found.", file=sys.stderr)
        sys.exit(1)
    config = _workflow_config_from_args(args)
    result = build_ephemeral_map(path, config=config, doc_id=args.doc_id)
    if args.manifest:
        result.document.write_manifest(args.manifest)
        print(f"[OK] Wrote transient manifest with {len(result.document.records)} regions to {args.manifest}", file=sys.stderr)
    if args.json:
        output = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
    else:
        parts = []
        if args.instructions:
            parts.extend([result.instructions, ""])
        parts.append(result.document.marked_text)
        output = "\n".join(parts)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"[OK] Wrote ephemeral map for {path} to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(output)


def _handle_ephemeral_apply(args):
    if bool(args.edits_json) == bool(args.edits_file):
        print("Error: Provide exactly one of --edits-json or --edits-file.", file=sys.stderr)
        sys.exit(1)
    payload_text = args.edits_json
    if args.edits_file:
        payload_text = Path(args.edits_file).read_text(encoding="utf-8-sig")
    try:
        edits = edits_from_json(payload_text)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"Error: Invalid edit payload: {exc}", file=sys.stderr)
        sys.exit(1)

    config = _workflow_config_from_args(args)
    result = apply_ephemeral_edits(
        args.file,
        edits,
        output=args.output,
        config=config,
        doc_id=args.doc_id,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result.get("ok"):
        sys.exit(1)


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
