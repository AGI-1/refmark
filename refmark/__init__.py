"""Public package surface for refmark."""

from refmark.core import inject, strip
from refmark.documents import AlignmentReport, DocumentMap, align_documents, map_document
from refmark.edit import apply_ref_diff
from refmark.highlight import highlight_refs
from refmark.languages import list_supported_languages
from refmark.metrics import RefRangeScore, RewardConfig, citation_reward, score_ref_range, summarize_scores
from refmark.pipeline import (
    AlignmentCandidate,
    CoverageItem,
    RegionRecord,
    align_region_records,
    build_region_manifest,
    evaluate_alignment_coverage,
    expand_region_context,
    read_manifest,
    render_coverage_html,
    render_coverage_report_html,
    summarize_coverage,
    write_manifest,
)
from refmark.prompt import EnrichedPrompt, build_reference_prompt
from refmark.refmarker import Refmarker, RefmarkRegistry, RefmarkResult
from refmark.workflow_config import WorkflowConfig, load_workflow_config, resolve_workflow_config

__version__ = "0.1.0"

__all__ = [
    "apply_ref_diff",
    "build_reference_prompt",
    "EnrichedPrompt",
    "AlignmentCandidate",
    "AlignmentReport",
    "CoverageItem",
    "DocumentMap",
    "RegionRecord",
    "align_region_records",
    "align_documents",
    "build_region_manifest",
    "evaluate_alignment_coverage",
    "expand_region_context",
    "highlight_refs",
    "inject",
    "list_supported_languages",
    "map_document",
    "read_manifest",
    "render_coverage_html",
    "render_coverage_report_html",
    "Refmarker",
    "RefmarkRegistry",
    "RefmarkResult",
    "summarize_coverage",
    "WorkflowConfig",
    "load_workflow_config",
    "resolve_workflow_config",
    "RefRangeScore",
    "RewardConfig",
    "score_ref_range",
    "citation_reward",
    "strip",
    "summarize_scores",
    "write_manifest",
    "__version__",
]
