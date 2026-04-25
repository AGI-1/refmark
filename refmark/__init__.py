"""Public package surface for refmark."""

from refmark.core import inject, strip
from refmark.edit import apply_ref_diff
from refmark.highlight import highlight_refs
from refmark.languages import list_supported_languages
from refmark.metrics import RefRangeScore, RewardConfig, citation_reward, score_ref_range, summarize_scores

__version__ = "0.1.0"

__all__ = [
    "apply_ref_diff",
    "highlight_refs",
    "inject",
    "list_supported_languages",
    "RefRangeScore",
    "RewardConfig",
    "score_ref_range",
    "citation_reward",
    "strip",
    "summarize_scores",
    "__version__",
]
