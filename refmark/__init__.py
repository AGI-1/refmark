"""Public package surface for refmark."""

from refmark.citations import CitationRef, parse_citation_refs, validate_citation_refs
from refmark.core import inject, strip
from refmark.discovery import DiscoveryManifest, discover_corpus, load_discovery, write_discovery
from refmark.documents import AlignmentReport, DocumentMap, align_documents, map_document
from refmark.edit import apply_ref_diff
from refmark.highlight import highlight_refs
from refmark.languages import list_supported_languages
from refmark.metrics import RefRangeScore, RewardConfig, citation_reward, score_ref_range, summarize_scores
from refmark.pipeline import (
    AlignmentCandidate,
    CoverageItem,
    RegionRecord,
    SectionEntry,
    align_region_records,
    build_section_map,
    build_region_manifest,
    evaluate_alignment_coverage,
    expand_region_context,
    read_manifest,
    render_coverage_html,
    render_coverage_report_html,
    summarize_coverage,
    write_manifest,
)
from refmark.pipeline_config import (
    EmbeddingConfig,
    FullPipelineConfig,
    ModelTierConfig,
    PipelineArtifactConfig,
    PipelineBudgetConfig,
    PipelineLoopConfig,
    default_full_pipeline_config,
    full_pipeline_config_from_dict,
    load_full_pipeline_config,
    write_full_pipeline_config_template,
)
from refmark.pipeline_runner import PipelineRunSummary, run_full_pipeline
from refmark.prompt import EnrichedPrompt, build_reference_prompt
from refmark.provenance import build_eval_provenance, file_fingerprint, validate_provenance
from refmark.rag_eval import (
    ContextPack,
    CorpusMap,
    CorpusMapSnapshot,
    CorpusRevisionDiff,
    EvalExample,
    EvalRun,
    EvalSuite,
    NormalizedHit,
    StaleExample,
    adaptation_recommendations,
    diagnose_results,
    failure_heatmap,
    selective_jump_diagnostics,
)
from refmark.refmarker import Refmarker, RefmarkRegistry, RefmarkResult
from refmark.workflow_config import WorkflowConfig, load_workflow_config, resolve_workflow_config

__version__ = "0.1.0"

__all__ = [
    "apply_ref_diff",
    "build_reference_prompt",
    "build_eval_provenance",
    "EnrichedPrompt",
    "AlignmentCandidate",
    "AlignmentReport",
    "CitationRef",
    "CoverageItem",
    "CorpusMap",
    "CorpusMapSnapshot",
    "CorpusRevisionDiff",
    "ContextPack",
    "DiscoveryManifest",
    "DocumentMap",
    "EvalExample",
    "EvalRun",
    "EvalSuite",
    "EmbeddingConfig",
    "FullPipelineConfig",
    "ModelTierConfig",
    "NormalizedHit",
    "PipelineArtifactConfig",
    "PipelineBudgetConfig",
    "PipelineLoopConfig",
    "PipelineRunSummary",
    "RegionRecord",
    "SectionEntry",
    "align_region_records",
    "align_documents",
    "adaptation_recommendations",
    "build_region_manifest",
    "build_section_map",
    "default_full_pipeline_config",
    "discover_corpus",
    "diagnose_results",
    "evaluate_alignment_coverage",
    "expand_region_context",
    "failure_heatmap",
    "file_fingerprint",
    "highlight_refs",
    "inject",
    "list_supported_languages",
    "load_discovery",
    "load_full_pipeline_config",
    "map_document",
    "parse_citation_refs",
    "read_manifest",
    "render_coverage_html",
    "render_coverage_report_html",
    "Refmarker",
    "RefmarkRegistry",
    "RefmarkResult",
    "summarize_coverage",
    "StaleExample",
    "WorkflowConfig",
    "load_workflow_config",
    "resolve_workflow_config",
    "run_full_pipeline",
    "RefRangeScore",
    "RewardConfig",
    "score_ref_range",
    "citation_reward",
    "selective_jump_diagnostics",
    "strip",
    "summarize_scores",
    "validate_provenance",
    "validate_citation_refs",
    "write_discovery",
    "write_full_pipeline_config_template",
    "write_manifest",
    "__version__",
]
