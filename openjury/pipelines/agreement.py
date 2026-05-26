"""Agreement pipeline orchestration.

Agreement is implemented as a specialized annotate run over dataset-defined
pairs. The pipeline produces a reusable annotation artifact; analysis is
performed post-hoc via separate scripts.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from openjury._logging import logger
from openjury.annotate_config import (
    AnnotateConfig,
    AnnotateGenerationConfig,
    AnnotatePairingConfig,
)
from openjury.arena.config import AgreementConfig
from openjury.pipelines.annotate import (
    save_agreement_annotations,
)
from openjury.pipelines.model_annotation import run_annotate


def _agreement_to_annotate_config(config: AgreementConfig) -> AnnotateConfig:
    """Project an agreement task onto the canonical annotate(dataset_pairs) path."""
    cache_variant = None
    for attr in ("k", "k_completions", "num_completions", "n_completions"):
        value = getattr(config, attr, None)
        if value is not None:
            cache_variant = f"{attr}_{value}"
            break
    return AnnotateConfig(
        dataset=config.dataset,
        judge=config.judge,
        challenger=None,
        output_dir=config.output_dir,
        n_instructions=config.n_instructions,
        language=config.language,
        seed=config.seed,
        balance_by=config.balance_by,
        criteria=config.criteria,
        pairing=AnnotatePairingConfig(
            source="dataset_pairs",
            strategy="dataset_defined",
            seed=config.seed,
        ),
        generation=AnnotateGenerationConfig(),
        cache_variant=cache_variant,
        ignore_score_cache=config.ignore_score_cache,
        include_completions=config.include_completions,
        include_raw_judge=False,
    )


def _pref_to_label(pref: float | None) -> str:
    """Convert a numeric preference (0.0=A, 0.5=tie, 1.0=B) to a categorical label."""
    if pref is None:
        return "parse_error"
    if pref < 0.25:
        return "A"
    if pref > 0.75:
        return "B"
    return "tie"


def _agreement_row_from_match(match: dict[str, Any]) -> dict[str, Any]:
    """Normalize an annotate match row into the agreement row shape.

    The annotate pipeline produces float preferences (0.0/0.5/1.0).
    This adapter adds aliased float-pref fields and categorical labels
    (``human_label``/``judge_label`` ∈ {"A", "B", "tie"}) for downstream
    analysis scripts.  Agreement metrics are NOT computed here.
    """
    row = dict(match)

    # ── Normalize float preference field names ───────────────────
    if "human_preference" in row and "human_pref" not in row:
        row["human_pref"] = row["human_preference"]
    if "preference" in row and "judge_pref" not in row:
        row["judge_pref"] = row["preference"]

    # ── Derive categorical labels from float preferences ─────────
    if "human_label" not in row and "human_pref" in row:
        row["human_label"] = _pref_to_label(row["human_pref"])
    if "judge_label" not in row and "judge_pref" in row:
        row["judge_label"] = _pref_to_label(row["judge_pref"])

    return row


def _is_nan_scalar(value: Any) -> bool:
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def _dict_has_nan_value(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return any(_is_nan_scalar(item) for item in value.values())


def _drop_nan_agreement_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Drop rows whose preferences or score dicts contain NaN values."""
    kept: list[dict[str, Any]] = []
    dropped = 0
    for row in rows:
        if _is_nan_scalar(row.get("human_pref")) or _is_nan_scalar(row.get("judge_pref")):
            dropped += 1
            continue
        if any(
            key.startswith("scores_") and _dict_has_nan_value(value)
            for key, value in row.items()
        ):
            dropped += 1
            continue
        kept.append(row)
    return kept, dropped


def _annotate_payload_to_agreement_payload(
    config: AgreementConfig,
    annotation_payload: dict[str, Any],
) -> dict[str, Any]:
    """Adapt canonical annotate output into the persisted agreement artifact shape."""
    matches = list(annotation_payload.get("matches", []))
    per_sample = [_agreement_row_from_match(match) for match in matches]
    per_sample, dropped_nan_rows = _drop_nan_agreement_rows(per_sample)
    if dropped_nan_rows:
        logger.warning(
            "Dropped %d agreement row(s) with NaN preferences/scores before saving.",
            dropped_nan_rows,
        )
    return {
        "metadata": {
            "dataset": config.dataset,
            "judge_model": config.judge.model,
            "judge_mode": config.judge.mode,
            "criteria": config.criteria,
            "swap_debiasing": not config.judge.no_swap,
            "n_samples_raw": len(matches),
            "n_samples": len(per_sample),
            "n_samples_dropped_nan": dropped_nan_rows,
        },
        "criterion_names": list(annotation_payload.get("criterion_names", [])),
        "criteria_definition": annotation_payload.get("criteria_definition", {}),
        "instruction_metadata": list(annotation_payload.get("instruction_metadata", [])),
        "per_sample": per_sample,
        "system_prompt": annotation_payload.get("system_prompt", ""),
    }


def run_agreement(config: AgreementConfig) -> dict[str, Any]:
    """Run the agreement pipeline: annotate and save the artifact."""
    annotate_config = _agreement_to_annotate_config(config)
    annotate_payload = run_annotate(annotate_config, persist=False)
    annotation_payload = _annotate_payload_to_agreement_payload(config, annotate_payload)

    annotation_path = save_agreement_annotations(
        config.output_dir,
        annotation_payload,
        config_snapshot=config.to_dict(),
    )
    config.save(Path(config.output_dir) / "agreement_config.json")
    logger.info("Saved agreement annotations: %s", annotation_path)

    return annotation_payload
