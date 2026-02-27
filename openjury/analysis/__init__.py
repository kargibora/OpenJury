"""Analysis modules for evaluation artifacts.

This package is intentionally separate from runtime pipelines:
- ``openjury.pipelines`` handles generation / annotation orchestration
- ``openjury.analysis`` handles metrics/ranking computation from artifacts
"""

from openjury.analysis.agreement import analyze_agreement_annotations
from openjury.analysis.arena import analyze_arena_annotations
from openjury.analysis.common import (
    compute_agreement_metrics,
    log_agreement_summary,
    save_agreement_outputs,
)

__all__ = [
    "analyze_agreement_annotations",
    "analyze_arena_annotations",
    "compute_agreement_metrics",
    "log_agreement_summary",
    "save_agreement_outputs",
]

