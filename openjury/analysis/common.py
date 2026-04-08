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

    # ── Multi-trial (Average-of-K) confidence metrics ─────────
    multi_trial_samples = [r for r in per_sample if r.get("n_trials", 1) > 1]
    if multi_trial_samples:
        n_trials = multi_trial_samples[0].get("n_trials", 1)
        std_prefs = [r["std_preference"] for r in multi_trial_samples if "std_preference" in r]
        self_agrs = [r["self_agreement"] for r in multi_trial_samples if "self_agreement" in r]

        metrics["n_trials"] = n_trials
        metrics["mean_self_agreement"] = (
            round(float(np.mean(self_agrs)), 4) if self_agrs else None
        )
        metrics["mean_std_preference"] = (
            round(float(np.mean(std_prefs)), 4) if std_prefs else None
        )
        metrics["median_std_preference"] = (
            round(float(np.median(std_prefs)), 4) if std_prefs else None
        )

        # ── Confidence-stratified accuracy ────────────────────
        # Split into high-confidence (std below median) and low-confidence
        if std_prefs and len(std_prefs) >= 4:
            median_std = float(np.median(std_prefs))
            high_conf = [
                r for r in multi_trial_samples
                if not r.get("parse_error")
                and r.get("std_preference", 1.0) <= median_std
            ]
            low_conf = [
                r for r in multi_trial_samples
                if not r.get("parse_error")
                and r.get("std_preference", 0.0) > median_std
            ]

            def _bucket_metrics(
                bucket: list[dict], label: str,
            ) -> dict[str, Any]:
                n_b = len(bucket)
                if n_b < 2:
                    return {"n": n_b}
                n_agr = sum(1 for r in bucket if r.get("agree") is True)
                bh = [r["human_label"] for r in bucket]
                bj = [r["judge_label"] for r in bucket]
                try:
                    bk = compute_cohen_kappa(bh, bj)
                except (ValueError, ZeroDivisionError):
                    bk = None
                return {
                    "n": n_b,
                    "accuracy": round(n_agr / n_b, 4),
                    "cohens_kappa": round(bk, 4) if bk is not None else None,
                }

            metrics["confidence_stratified"] = {
                "std_threshold": round(median_std, 4),
                "high_confidence": _bucket_metrics(high_conf, "high"),
                "low_confidence": _bucket_metrics(low_conf, "low"),
            }

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
    if metrics.get("n_trials", 1) > 1:
        logger.info("  Trials (K):        %d", metrics["n_trials"])
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

    # ── Multi-trial confidence metrics ────────────────────────
    if metrics.get("n_trials", 1) > 1:
        logger.info("───────────────────────────────────────────────────")
        logger.info("  Multi-trial confidence (K=%d)", metrics["n_trials"])
        if metrics.get("mean_self_agreement") is not None:
            logger.info("  Self-agreement:    %.4f", metrics["mean_self_agreement"])
        if metrics.get("mean_std_preference") is not None:
            logger.info("  Mean σ(pref):      %.4f", metrics["mean_std_preference"])
        if metrics.get("median_std_preference") is not None:
            logger.info("  Median σ(pref):    %.4f", metrics["median_std_preference"])

        strat = metrics.get("confidence_stratified")
        if strat:
            logger.info(
                "  Confidence split:  σ threshold = %.4f",
                strat["std_threshold"],
            )
            hi = strat["high_confidence"]
            lo = strat["low_confidence"]
            if hi.get("accuracy") is not None:
                logger.info(
                    "    High-conf:  acc=%.1f%%  κ=%s  (n=%d)",
                    hi["accuracy"] * 100,
                    f'{hi["cohens_kappa"]:.4f}' if hi.get("cohens_kappa") is not None else "n/a",
                    hi["n"],
                )
            if lo.get("accuracy") is not None:
                logger.info(
                    "    Low-conf:   acc=%.1f%%  κ=%s  (n=%d)",
                    lo["accuracy"] * 100,
                    f'{lo["cohens_kappa"]:.4f}' if lo.get("cohens_kappa") is not None else "n/a",
                    lo["n"],
                )

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
        flat = {k: v for k, v in r.items() if k not in (
            "scores_a", "scores_b", "metadata",
            "score_std_a", "score_std_b",
            "trial_raw_outputs", "trial_raw_outputs_swapped",
        )}
        for dim, val in r["scores_a"].items():
            flat[f"score_a_{dim}"] = val
        for dim, val in r["scores_b"].items():
            flat[f"score_b_{dim}"] = val
        # Multi-trial score std columns
        if r.get("score_std_a"):
            for dim, val in r["score_std_a"].items():
                flat[f"score_std_a_{dim}"] = val
        if r.get("score_std_b"):
            for dim, val in r["score_std_b"].items():
                flat[f"score_std_b_{dim}"] = val
        if r.get("metadata"):
            for mk, mv in r["metadata"].items():
                flat[f"meta_{mk}"] = mv
        csv_rows.append(flat)

    csv_path = out_path / "agreement_details.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    return json_path, csv_path
