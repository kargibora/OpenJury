"""Agreement-specific analysis stage helpers."""

from __future__ import annotations

from typing import Any

from openjury._logging import logger
from openjury.analysis.common import (
    compute_agreement_metrics,
    log_agreement_summary,
    save_agreement_outputs,
)
from openjury.arena.config import AgreementConfig


def analyze_agreement_annotations(
    *,
    config: AgreementConfig,
    annotation_data: dict[str, Any],
) -> dict[str, Any]:
    """Analyze persisted agreement annotations and write final outputs."""
    meta = annotation_data["metadata"]
    per_sample = annotation_data["per_sample"]
    system_prompt = annotation_data.get("system_prompt", "")

    metrics = compute_agreement_metrics(
        per_sample=per_sample,
        dataset=meta["dataset"],
        judge_model=meta["judge_model"],
        judge_mode=meta["judge_mode"],
        rubric=meta["rubric"],
        swap_debiasing=meta["swap_debiasing"],
        language=meta.get("language"),
    )
    log_agreement_summary(metrics)
    json_path, csv_path = save_agreement_outputs(
        output_dir=config.output_dir,
        metrics=metrics,
        system_prompt=system_prompt,
        per_sample=per_sample,
    )
    logger.info("")
    logger.info("Results saved:")
    logger.info("  📄 %s", json_path)
    logger.info("  📄 %s", csv_path)
    return {"metrics": metrics, "per_sample": per_sample}

