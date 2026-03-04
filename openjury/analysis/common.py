"""Common helpers for analysis stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from openjury._logging import logger


def compute_cohen_kappa(y1: list[str], y2: list[str]) -> float:
    """Compute Cohen's kappa coefficient for inter-rater agreement."""
    if len(y1) != len(y2):
        raise ValueError("Both lists must have the same length")
    if len(y1) == 0:
        raise ValueError("Lists cannot be empty")

    categories = sorted(set(y1) | set(y2))
    n = len(y1)

    matrix: dict[str, dict[str, int]] = {}
    for cat1 in categories:
        matrix[cat1] = {cat2: 0 for cat2 in categories}

    for label1, label2 in zip(y1, y2):
        matrix[label1][label2] += 1

    observed_agreement = sum(matrix[cat][cat] for cat in categories) / n

    expected_agreement = 0.0
    for cat in categories:
        p1 = sum(matrix[cat][c] for c in categories) / n
        p2 = sum(matrix[c][cat] for c in categories) / n
        expected_agreement += p1 * p2

    if expected_agreement == 1:
        return 1.0 if observed_agreement == 1 else 0.0

    return (observed_agreement - expected_agreement) / (1 - expected_agreement)


def compute_agreement_metrics(
    *,
    per_sample: list[dict[str, Any]],
    dataset: str,
    judge_model: str,
    judge_mode: str,
    criteria: str,
    swap_debiasing: bool,
    language: str | None = None,
) -> dict[str, Any]:
    """Compute agreement metrics from persisted per-sample annotations."""
    human_labels: list[str] = []
    judge_labels: list[str] = []

    for row in per_sample:
        human_labels.append(row["human_label"])
        judge_labels.append(row["judge_label"])

    valid_human = [h for h, j in zip(human_labels, judge_labels) if j != "parse_error"]
    valid_judge = [j for j in judge_labels if j != "parse_error"]
    n_valid = len(valid_human)
    n_total = len(per_sample)
    n_parse_failures = sum(1 for r in per_sample if r.get("parse_error"))

    n_agree = sum(1 for r in per_sample if r.get("agree") is True)
    accuracy = n_agree / n_valid if n_valid > 0 else 0.0

    kappa = compute_cohen_kappa(valid_human, valid_judge) if n_valid >= 2 else None

    decisive_h: list[str] = []
    decisive_j: list[str] = []
    for h, j in zip(valid_human, valid_judge):
        if h != "tie" and j != "tie":
            decisive_h.append(h)
            decisive_j.append(j)
    kappa_decisive = (
        compute_cohen_kappa(decisive_h, decisive_j) if len(decisive_h) >= 2 else None
    )

    valid_h_prefs: list[float] = []
    valid_j_prefs: list[float] = []
    for r in per_sample:
        if not r.get("parse_error") and r.get("judge_pref") is not None:
            valid_h_prefs.append(r["human_pref"])
            valid_j_prefs.append(r["judge_pref"])

    if len(valid_h_prefs) >= 2:
        _corr = float(np.corrcoef(valid_h_prefs, valid_j_prefs)[0, 1])
        corr = None if np.isnan(_corr) else _corr
        mae = float(np.mean(np.abs(np.array(valid_h_prefs) - np.array(valid_j_prefs))))
    else:
        corr = None
        mae = None

    def _dist(labels: list[str]) -> dict[str, int]:
        return {
            "A": sum(1 for l in labels if l == "A"),
            "B": sum(1 for l in labels if l == "B"),
            "tie": sum(1 for l in labels if l == "tie"),
        }

    metrics: dict[str, Any] = {
        "dataset": dataset,
        "judge_model": judge_model,
        "judge_mode": judge_mode,
        "criteria": criteria,
        "swap_debiasing": swap_debiasing,
        "n_samples": n_total,
        "n_valid": n_valid,
        "n_parse_errors": n_parse_failures,
        "cohens_kappa": round(kappa, 4) if kappa is not None else None,
        "cohens_kappa_decisive": (
            round(kappa_decisive, 4) if kappa_decisive is not None else None
        ),
        "accuracy": round(accuracy, 4),
        "n_agree": n_agree,
        "n_disagree": n_valid - n_agree,
        "pearson_correlation": round(corr, 4) if corr is not None else None,
        "mae": round(mae, 4) if mae is not None else None,
        "n_decisive_pairs": len(decisive_h),
        "human_distribution": _dist(valid_human),
        "judge_distribution": _dist(valid_judge),
    }
    if language:
        metrics["language"] = language
    return metrics


def log_agreement_summary(metrics: dict[str, Any]) -> None:
    """Print a concise agreement summary."""
    logger.info("")
    logger.info("═══════════════════════════════════════════════════")
    logger.info("  Human–Judge Agreement Results")
    logger.info("═══════════════════════════════════════════════════")
    logger.info(
        "  Dataset:           %s (%d samples, %d valid)",
        metrics["dataset"],
        metrics["n_samples"],
        metrics["n_valid"],
    )
    if metrics["n_parse_errors"] > 0:
        logger.info(
            "  ⚠️  Parse errors:   %d samples (excluded from metrics)",
            metrics["n_parse_errors"],
        )
    logger.info("  Judge:             %s", metrics["judge_model"])
    logger.info("  Mode:              %s", metrics["judge_mode"])
    logger.info("  Criteria:          %s", metrics["criteria"])
    logger.info("  Swap debiasing:    %s", "yes" if metrics["swap_debiasing"] else "no")
    logger.info("───────────────────────────────────────────────────")
    if metrics["cohens_kappa"] is not None:
        logger.info("  Cohen's κ:         %.4f", metrics["cohens_kappa"])
    if metrics["cohens_kappa_decisive"] is not None:
        logger.info(
            "  Cohen's κ (no tie): %.4f  (%d pairs)",
            metrics["cohens_kappa_decisive"],
            metrics["n_decisive_pairs"],
        )
    logger.info(
        "  Accuracy:          %.1f%% (%d/%d)",
        metrics["accuracy"] * 100,
        metrics["n_agree"],
        metrics["n_valid"],
    )
    if metrics["pearson_correlation"] is not None:
        logger.info("  Pearson corr:      %.4f", metrics["pearson_correlation"])
    if metrics["mae"] is not None:
        logger.info("  MAE:               %.4f", metrics["mae"])
    logger.info("───────────────────────────────────────────────────")
    h_d = metrics["human_distribution"]
    j_d = metrics["judge_distribution"]
    logger.info("  Human:   A=%-4d  B=%-4d  tie=%-4d", h_d["A"], h_d["B"], h_d["tie"])
    logger.info("  Judge:   A=%-4d  B=%-4d  tie=%-4d", j_d["A"], j_d["B"], j_d["tie"])
    logger.info("═══════════════════════════════════════════════════")


def save_agreement_outputs(
    *,
    output_dir: str | Path,
    metrics: dict[str, Any],
    system_prompt: str,
    per_sample: list[dict[str, Any]],
) -> tuple[Path, Path]:
    """Write final agreement outputs (JSON + CSV)."""
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    json_path = out_path / "agreement.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"metrics": metrics, "system_prompt": system_prompt, "per_sample": per_sample},
            f,
            indent=2,
            default=str,
        )

    csv_rows: list[dict[str, Any]] = []
    for r in per_sample:
        flat = {k: v for k, v in r.items() if k not in ("scores_a", "scores_b", "metadata")}
        for dim, val in r["scores_a"].items():
            flat[f"score_a_{dim}"] = val
        for dim, val in r["scores_b"].items():
            flat[f"score_b_{dim}"] = val
        if r.get("metadata"):
            for mk, mv in r["metadata"].items():
                flat[f"meta_{mk}"] = mv
        csv_rows.append(flat)

    csv_path = out_path / "agreement_details.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    return json_path, csv_path
