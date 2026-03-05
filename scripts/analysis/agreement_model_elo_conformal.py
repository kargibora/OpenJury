from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("OPENJURY_LOG_LEVEL", "ERROR")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from openjury.arena.ratings import fit_multi_bt

from plotting import (
    BT_TO_ELO,
    COLORS,
    ELO_BASE,
    apply_minimal_theme as _apply_theme,
    build_matches,
    load_frame as load_annotation_frame,
    model_universe,
    resolve_artifact_path,
    shorten as shorten_model_name,
    valid_decisive as valid_pref_frame,
)


def bt_theta_to_elo(theta: dict[str, float], base: float = ELO_BASE) -> dict[str, float]:
    return {m: float(base + BT_TO_ELO * v) for m, v in theta.items()}


def fit_pool_bt(
    frame: pd.DataFrame,
    pref_col: str,
    models: list[str],
    regularization: float,
) -> tuple[dict[str, float], dict[str, float]]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.")
        theta = fit_multi_bt(
            models=models,
            matches=build_matches(frame, pref_col, decisive_only=True),
            regularization=regularization,
        )
    return theta, bt_theta_to_elo(theta)


def target_vs_anchor_rows(
    frame: pd.DataFrame,
    target_model: str,
    pref_col: str,
    allowed_anchors: set[str] | None = None,
) -> pd.DataFrame:
    rows = valid_pref_frame(frame, pref_col)
    mask = (rows["model_a"] == target_model) | (rows["model_b"] == target_model)
    rows = rows.loc[mask].copy()
    if allowed_anchors is not None:
        opp = np.where(rows["model_a"] == target_model, rows["model_b"], rows["model_a"])
        rows = rows.loc[pd.Series(opp).isin(allowed_anchors).values].copy()
    return rows.reset_index(drop=True)


def encode_target_outcomes(
    rows: pd.DataFrame,
    target_model: str,
    pref_col: str,
    anchor_theta: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    y: list[float] = []
    theta_anchor: list[float] = []

    for _, row in rows.iterrows():
        pref = float(row[pref_col])

        if row["model_a"] == target_model:
            opp = str(row["model_b"])
            target_wins = 1.0 if pref < 0.5 else 0.0
        else:
            opp = str(row["model_a"])
            target_wins = 1.0 if pref > 0.5 else 0.0

        if opp not in anchor_theta:
            continue

        y.append(target_wins)
        theta_anchor.append(anchor_theta[opp])

    return np.asarray(y, dtype=float), np.asarray(theta_anchor, dtype=float)


def solve_target_theta(
    y: np.ndarray,
    theta_anchor: np.ndarray,
    *,
    regularization: float,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> float:
    """Solve the held-out target BT parameter against fixed anchor strengths."""
    if len(y) < 2:
        return float("nan")

    theta = 0.0
    for _ in range(max_iter):
        z = theta - theta_anchor
        p = 1.0 / (1.0 + np.exp(-z))
        grad = np.sum(y - p) - regularization * theta
        hess = -np.sum(p * (1.0 - p)) - regularization
        if abs(hess) < 1e-12:
            break
        new_theta = theta - grad / hess
        if abs(new_theta - theta) < tol:
            theta = new_theta
            break
        theta = new_theta

    return float(theta)


def estimate_target_theta_against_anchors(
    rows: pd.DataFrame,
    target_model: str,
    pref_col: str,
    anchor_theta: dict[str, float],
    regularization: float,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> tuple[float, int]:
    y, theta_anchor = encode_target_outcomes(rows, target_model, pref_col, anchor_theta)
    n = len(y)
    if n < 2:
        return float("nan"), n
    return (
        solve_target_theta(
            y,
            theta_anchor,
            regularization=regularization,
            max_iter=max_iter,
            tol=tol,
        ),
        n,
    )


def bootstrap_target_theta(
    rows: pd.DataFrame,
    target_model: str,
    pref_col: str,
    anchor_theta: dict[str, float],
    *,
    anchor_frame: pd.DataFrame | None = None,
    anchor_models: list[str] | None = None,
    bootstrap_anchor_pool: bool = False,
    n_bootstrap: int,
    seed: int,
    regularization: float,
) -> pd.DataFrame:
    encoded_y, _ = encode_target_outcomes(rows, target_model, pref_col, anchor_theta)
    n = len(encoded_y)
    if n < 6:
        return pd.DataFrame(columns=["bootstrap_id", "theta", "elo"])

    rng = np.random.default_rng(seed)
    samples: list[dict[str, float]] = []

    if bootstrap_anchor_pool and (anchor_frame is None or not anchor_models):
        raise ValueError(
            "bootstrap_anchor_pool=True requires anchor_frame and anchor_models."
        )

    for b in range(n_bootstrap):
        row_idx = rng.choice(np.arange(len(rows)), size=len(rows), replace=True)
        rows_b = rows.iloc[row_idx].reset_index(drop=True)

        anchor_theta_b = anchor_theta
        if bootstrap_anchor_pool:
            anchor_idx = rng.choice(
                np.arange(len(anchor_frame)),
                size=len(anchor_frame),
                replace=True,
            )
            anchor_frame_b = anchor_frame.iloc[anchor_idx].reset_index(drop=True)
            anchor_theta_b, _ = fit_pool_bt(
                anchor_frame_b,
                pref_col,
                anchor_models,
                regularization,
            )

        y_b, theta_anchor_b = encode_target_outcomes(
            rows_b,
            target_model,
            pref_col,
            anchor_theta_b,
        )
        if len(y_b) < 2:
            continue

        theta = solve_target_theta(
            y_b,
            theta_anchor_b,
            regularization=regularization,
        )

        samples.append(
            {
                "bootstrap_id": b,
                "theta": float(theta),
                "elo": float(ELO_BASE + BT_TO_ELO * theta),
            }
        )

    return pd.DataFrame(samples)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("No valid conformal scores.")
    level = min(np.ceil((len(values) + 1) * (1 - alpha)) / len(values), 1.0)
    return float(np.quantile(values, level, method="higher"))


def eligible_target_models(
    frame: pd.DataFrame,
    *,
    min_matches: int,
) -> pd.DataFrame:
    models = model_universe(frame)
    rows: list[dict[str, object]] = []

    for model in models:
        human_n = len(target_vs_anchor_rows(frame, model, "human_pref"))
        judge_n = len(target_vs_anchor_rows(frame, model, "judge_pref"))
        rows.append(
            {
                "model": model,
                "short_model": shorten_model_name(model),
                "n_human_pairs": human_n,
                "n_judge_pairs": judge_n,
                "eligible": min(human_n, judge_n) >= min_matches,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["eligible", "n_judge_pairs", "n_human_pairs"], ascending=[False, False, False])
        .reset_index(drop=True)
    )


def build_heldout_table(
    frame: pd.DataFrame,
    *,
    min_matches: int,
    regularization: float,
    bootstrap_resamples: int,
    bootstrap_anchor_pool: bool,
    seed: int,
    partial_save_path: Path | None = None,
    progress_every: int = 1,
) -> pd.DataFrame:
    inventory = eligible_target_models(frame, min_matches=min_matches)
    targets = inventory.loc[inventory["eligible"], "model"].tolist()
    all_models = model_universe(frame)
    rows: list[dict[str, object]] = []

    if not targets:
        top_counts = inventory.head(10).to_dict(orient="records")
        raise ValueError(
            "No eligible target models for held-out calibration. "
            f"Need at least {min_matches} decisive human and judge target-vs-anchor "
            "comparisons per model. "
            f"Top availability snapshot: {top_counts}"
        )

    for idx, target_model in enumerate(targets):
        if progress_every > 0 and (idx == 0 or (idx + 1) % progress_every == 0):
            print(
                f"[heldout] target {idx + 1}/{len(targets)}: {shorten_model_name(target_model)}",
                flush=True,
            )
        anchors = [m for m in all_models if m != target_model]
        anchor_set = set(anchors)
        anchor_frame = frame[
            frame["model_a"].isin(anchor_set) & frame["model_b"].isin(anchor_set)
        ].copy()

        human_anchor_theta, _ = fit_pool_bt(
            anchor_frame, "human_pref", anchors, regularization
        )
        judge_anchor_theta, _ = fit_pool_bt(
            anchor_frame, "judge_pref", anchors, regularization
        )

        human_rows = target_vs_anchor_rows(frame, target_model, "human_pref", anchor_set)
        judge_rows = target_vs_anchor_rows(frame, target_model, "judge_pref", anchor_set)

        human_theta, n_human = estimate_target_theta_against_anchors(
            human_rows,
            target_model,
            "human_pref",
            human_anchor_theta,
            regularization,
        )
        judge_theta, n_judge = estimate_target_theta_against_anchors(
            judge_rows,
            target_model,
            "judge_pref",
            judge_anchor_theta,
            regularization,
        )

        judge_boot = bootstrap_target_theta(
            judge_rows,
            target_model,
            "judge_pref",
            judge_anchor_theta,
            anchor_frame=anchor_frame,
            anchor_models=anchors,
            bootstrap_anchor_pool=bootstrap_anchor_pool,
            n_bootstrap=bootstrap_resamples,
            seed=seed + idx * 17,
            regularization=regularization,
        )
        human_boot = bootstrap_target_theta(
            human_rows,
            target_model,
            "human_pref",
            human_anchor_theta,
            anchor_frame=anchor_frame,
            anchor_models=anchors,
            bootstrap_anchor_pool=bootstrap_anchor_pool,
            n_bootstrap=min(60, bootstrap_resamples),
            seed=seed + idx * 23,
            regularization=regularization,
        )

        judge_elo = ELO_BASE + BT_TO_ELO * judge_theta if np.isfinite(judge_theta) else float("nan")
        human_elo = ELO_BASE + BT_TO_ELO * human_theta if np.isfinite(human_theta) else float("nan")

        rows.append(
            {
                "model": target_model,
                "short_model": shorten_model_name(target_model),
                "n_human_pairs": n_human,
                "n_judge_pairs": n_judge,
                "human_bt_theta": human_theta,
                "judge_bt_theta": judge_theta,
                "human_bt_elo": human_elo,
                "judge_bt_elo": judge_elo,
                "bt_elo_residual": human_elo - judge_elo,
                "abs_bt_elo_residual": abs(human_elo - judge_elo),
                "judge_bt_elo_se": float(judge_boot["elo"].std(ddof=1)) if len(judge_boot) >= 5 else float("nan"),
                "human_bt_elo_se": float(human_boot["elo"].std(ddof=1)) if len(human_boot) >= 5 else float("nan"),
                "judge_bt_elo_p05": float(np.percentile(judge_boot["elo"], 5)) if len(judge_boot) >= 5 else float("nan"),
                "judge_bt_elo_p95": float(np.percentile(judge_boot["elo"], 95)) if len(judge_boot) >= 5 else float("nan"),
            }
        )

        if partial_save_path is not None:
            partial_df = pd.DataFrame(rows)
            partial_df.to_csv(partial_save_path, index=False)

    out = pd.DataFrame(rows)
    out["judge_bt_elo_se"] = out["judge_bt_elo_se"].replace(0.0, np.nan)
    out["normalized_residual"] = out["bt_elo_residual"] / out["judge_bt_elo_se"]
    return out.sort_values("abs_bt_elo_residual", ascending=False).reset_index(drop=True)


def split_conformal_model_calibration(
    heldout_table: pd.DataFrame,
    *,
    alpha: float,
    use_normalized: bool,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    table = heldout_table.copy()
    if use_normalized:
        table = table[
            table["judge_bt_elo_se"].notna()
            & np.isfinite(table["judge_bt_elo_se"])
            & (table["judge_bt_elo_se"] > 0)
        ].copy()
    else:
        table = table[np.isfinite(table["bt_elo_residual"])].copy()

    shuffled = table.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    if n < 8:
        raise ValueError("Need at least 8 eligible held-out models for split conformal calibration.")

    n_cal = max(4, n // 2)
    calib = shuffled.iloc[:n_cal].copy()
    test = shuffled.iloc[n_cal:].copy()

    if use_normalized:
        calib_scores = calib["abs_bt_elo_residual"] / calib["judge_bt_elo_se"]
        qhat = conformal_quantile(calib_scores.to_numpy(dtype=float), alpha)
        test["interval_lo"] = test["judge_bt_elo"] - qhat * test["judge_bt_elo_se"]
        test["interval_hi"] = test["judge_bt_elo"] + qhat * test["judge_bt_elo_se"]
        mode = "normalized"
    else:
        qhat = conformal_quantile(calib["abs_bt_elo_residual"].to_numpy(dtype=float), alpha)
        test["interval_lo"] = test["judge_bt_elo"] - qhat
        test["interval_hi"] = test["judge_bt_elo"] + qhat
        mode = "absolute"

    test["covered"] = (
        (test["human_bt_elo"] >= test["interval_lo"])
        & (test["human_bt_elo"] <= test["interval_hi"])
    )
    test["interval_width"] = test["interval_hi"] - test["interval_lo"]

    meta: dict[str, float | int | str] = {
        "alpha": float(alpha),
        "mode": mode,
        "n_total": int(n),
        "n_calibration": int(len(calib)),
        "n_test": int(len(test)),
        "qhat": float(qhat),
        "empirical_coverage": float(test["covered"].mean()),
        "median_interval_width": float(test["interval_width"].median()),
    }
    return test.sort_values("abs_bt_elo_residual", ascending=False).reset_index(drop=True), meta


def build_operational_intervals(
    heldout_table: pd.DataFrame,
    *,
    alpha: float,
) -> tuple[pd.DataFrame, float]:
    calib_all = heldout_table[
        heldout_table["judge_bt_elo_se"].notna() & (heldout_table["judge_bt_elo_se"] > 0)
    ].copy()
    qhat = conformal_quantile(
        (calib_all["abs_bt_elo_residual"] / calib_all["judge_bt_elo_se"]).to_numpy(dtype=float),
        alpha,
    )

    operational = heldout_table.copy()
    operational["deploy_interval_lo"] = operational["judge_bt_elo"] - qhat * operational["judge_bt_elo_se"]
    operational["deploy_interval_hi"] = operational["judge_bt_elo"] + qhat * operational["judge_bt_elo_se"]
    operational["deploy_interval_width"] = operational["deploy_interval_hi"] - operational["deploy_interval_lo"]
    return operational, float(qhat)


def repeated_split_conformal(
    heldout_table: pd.DataFrame,
    *,
    alpha: float,
    use_normalized: bool,
    seed: int,
    n_repeats: int,
    partial_save_path: Path | None = None,
    progress_every: int = 1,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Run split conformal repeatedly to assess seed sensitivity."""
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")

    rows: list[dict[str, float | int]] = []
    for repeat in range(n_repeats):
        if progress_every > 0 and (repeat == 0 or (repeat + 1) % progress_every == 0):
            print(f"[conformal] repeat {repeat + 1}/{n_repeats}", flush=True)
        split_seed = seed + repeat * 9973
        _, meta = split_conformal_model_calibration(
            heldout_table,
            alpha=alpha,
            use_normalized=use_normalized,
            seed=split_seed,
        )
        rows.append(
            {
                "repeat": repeat,
                "seed": split_seed,
                "qhat": float(meta["qhat"]),
                "empirical_coverage": float(meta["empirical_coverage"]),
                "median_interval_width": float(meta["median_interval_width"]),
                "n_total": int(meta["n_total"]),
                "n_calibration": int(meta["n_calibration"]),
                "n_test": int(meta["n_test"]),
            }
        )

        if partial_save_path is not None:
            pd.DataFrame(rows).to_csv(partial_save_path, index=False)

    df = pd.DataFrame(rows).sort_values("repeat").reset_index(drop=True)
    summary: dict[str, float | int] = {
        "n_repeats": int(n_repeats),
        "coverage_mean": float(df["empirical_coverage"].mean()),
        "coverage_std": float(df["empirical_coverage"].std(ddof=1)) if len(df) > 1 else 0.0,
        "coverage_min": float(df["empirical_coverage"].min()),
        "coverage_max": float(df["empirical_coverage"].max()),
        "qhat_mean": float(df["qhat"].mean()),
        "qhat_std": float(df["qhat"].std(ddof=1)) if len(df) > 1 else 0.0,
        "interval_width_median_of_medians": float(df["median_interval_width"].median()),
        "interval_width_mean": float(df["median_interval_width"].mean()),
        "interval_width_std": (
            float(df["median_interval_width"].std(ddof=1)) if len(df) > 1 else 0.0
        ),
    }
    return df, summary


def save_scatter_plot(heldout: pd.DataFrame, path: Path) -> None:
    """Judge vs Human held-out Elo scatter with error bars."""
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    fig.patch.set_facecolor("white")
    plot_df = heldout.sort_values("abs_bt_elo_residual", ascending=True).copy()

    # Bootstrap error bars
    has_se = plot_df["judge_bt_elo_se"].notna() & plot_df["human_bt_elo_se"].notna()
    ax.errorbar(
        plot_df.loc[has_se, "judge_bt_elo"],
        plot_df.loc[has_se, "human_bt_elo"],
        xerr=plot_df.loc[has_se, "judge_bt_elo_se"],
        yerr=plot_df.loc[has_se, "human_bt_elo_se"],
        fmt="none",
        ecolor=COLORS["band"],
        elinewidth=0.5,
        alpha=0.45,
        zorder=1,
    )

    sc = ax.scatter(
        plot_df["judge_bt_elo"],
        plot_df["human_bt_elo"],
        s=40,
        c=plot_df["abs_bt_elo_residual"],
        cmap="RdYlGn_r",
        edgecolor="white",
        linewidth=0.6,
        alpha=0.9,
        zorder=2,
    )

    # Annotate only top-5 outliers
    top = plot_df.nlargest(5, "abs_bt_elo_residual")
    for _, row in top.iterrows():
        ax.annotate(
            row["short_model"],
            (row["judge_bt_elo"], row["human_bt_elo"]),
            fontsize=7,
            color=COLORS["muted"],
            xytext=(6, 6),
            textcoords="offset points",
        )

    lo = min(plot_df["judge_bt_elo"].min(), plot_df["human_bt_elo"].min()) - 30
    hi = max(plot_df["judge_bt_elo"].max(), plot_df["human_bt_elo"].max()) + 30
    ax.plot([lo, hi], [lo, hi], "--", color=COLORS["muted"], linewidth=1.0, alpha=0.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    r_pearson = plot_df[["judge_bt_elo", "human_bt_elo"]].corr().iloc[0, 1]
    rho_spearman = plot_df[["judge_bt_elo", "human_bt_elo"]].corr(method="spearman").iloc[0, 1]
    mae = plot_df["abs_bt_elo_residual"].mean()

    cb = plt.colorbar(sc, ax=ax, pad=0.02, shrink=0.82)
    cb.set_label("|Elo residual|", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    _apply_theme(
        ax,
        title=f"Held-Out BT-Elo: Judge vs Human  (r={r_pearson:.3f}, \u03c1={rho_spearman:.3f}, MAE={mae:.0f})",
        xlabel="Judge BT-Elo",
        ylabel="Human BT-Elo",
    )

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _interval_plot(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    lo_col: str,
    hi_col: str,
    covered_col: str,
) -> None:
    """Shared helper for horizontal interval plots."""
    from matplotlib.lines import Line2D

    n = len(plot_df)
    y = np.arange(n)

    for i, (_, row) in enumerate(plot_df.iterrows()):
        color = COLORS["good"] if row[covered_col] else COLORS["bad"]
        ax.barh(
            i,
            row[hi_col] - row[lo_col],
            left=row[lo_col],
            height=0.6,
            color=color,
            alpha=0.22,
            edgecolor=color,
            linewidth=0.7,
        )
        ax.scatter(
            row["judge_bt_elo"], i, marker="D",
            color=COLORS["judge"], s=24, zorder=3,
            edgecolor="white", linewidth=0.4,
        )
        ax.scatter(
            row["human_bt_elo"], i, marker="o",
            color=COLORS["human"], s=32, zorder=4,
            edgecolor="white", linewidth=0.4,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["short_model"], fontsize=7.5)
    ax.spines["left"].set_visible(False)

    legend_elements = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor=COLORS["judge"],
               markersize=6, label="Judge estimate"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["human"],
               markersize=6, label="Human target"),
        Line2D([0], [0], color=COLORS["good"], linewidth=6, alpha=0.25, label="Covered"),
        Line2D([0], [0], color=COLORS["bad"], linewidth=6, alpha=0.25, label="Missed"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7.5, framealpha=0.9)


def save_conformal_plot(conformal_test: pd.DataFrame, alpha: float, path: Path) -> None:
    """Split-conformal validation intervals (test split only)."""
    plot_df = conformal_test.sort_values("judge_bt_elo").reset_index(drop=True)
    n = len(plot_df)
    coverage = plot_df["covered"].mean()

    fig, ax = plt.subplots(figsize=(9.5, max(5.2, 0.32 * n)))
    fig.patch.set_facecolor("white")
    _interval_plot(ax, plot_df, "interval_lo", "interval_hi", "covered")
    _apply_theme(
        ax,
        title=f"Split Conformal Intervals  (target {1 - alpha:.0%}, empirical {coverage:.0%}, n={n})",
        xlabel="BT-Elo",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_operational_plot(
    operational: pd.DataFrame, alpha: float, qhat: float, path: Path,
) -> None:
    """Operational intervals for all models (full calibration)."""
    plot_df = operational.sort_values("judge_bt_elo").reset_index(drop=True)
    n = len(plot_df)
    plot_df["op_covered"] = (
        (plot_df["human_bt_elo"] >= plot_df["deploy_interval_lo"])
        & (plot_df["human_bt_elo"] <= plot_df["deploy_interval_hi"])
    )
    coverage = plot_df["op_covered"].mean()

    fig, ax = plt.subplots(figsize=(10, max(6, 0.30 * n)))
    fig.patch.set_facecolor("white")
    _interval_plot(ax, plot_df, "deploy_interval_lo", "deploy_interval_hi", "op_covered")
    _apply_theme(
        ax,
        title=f"Operational Intervals  (\u0071\u0302={qhat:.2f}, coverage={coverage:.0%}, n={n})",
        xlabel="BT-Elo",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_coverage_histogram(repeat_df: pd.DataFrame, alpha: float, path: Path) -> None:
    """Histogram of empirical coverage across repeated splits."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")

    coverages = repeat_df["empirical_coverage"]
    ax.hist(
        coverages, bins=12, color=COLORS["human"],
        alpha=0.7, edgecolor="white", linewidth=0.5,
    )
    ax.axvline(
        1 - alpha, color=COLORS["bad"], linewidth=1.2, linestyle="--",
        label=f"Target ({1 - alpha:.0%})",
    )
    ax.axvline(
        coverages.mean(), color=COLORS["judge"], linewidth=1.2,
        label=f"Mean ({coverages.mean():.1%})",
    )
    ax.legend(fontsize=8, framealpha=0.9)
    _apply_theme(
        ax,
        title=f"Coverage Across {len(repeat_df)} Splits",
        xlabel="Empirical Coverage",
        ylabel="Count",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_interval_width_histogram(repeat_df: pd.DataFrame, path: Path) -> None:
    """Histogram of median interval widths across repeated splits."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")

    widths = repeat_df["median_interval_width"]
    ax.hist(
        widths, bins=12, color=COLORS["judge"],
        alpha=0.7, edgecolor="white", linewidth=0.5,
    )
    ax.axvline(
        widths.median(), color=COLORS["human"], linewidth=1.2,
        label=f"Median ({widths.median():.0f})",
    )
    ax.legend(fontsize=8, framealpha=0.9)
    _apply_theme(
        ax,
        title="Interval Width Across Splits",
        xlabel="Median Interval Width (Elo)",
        ylabel="Count",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_report_table(
    heldout: pd.DataFrame,
    operational: pd.DataFrame,
    path: Path,
) -> None:
    """Generate a clean report-ready CSV: one row per model, sorted by judge Elo."""
    merged = heldout.merge(
        operational[["model", "deploy_interval_lo", "deploy_interval_hi", "deploy_interval_width"]],
        on="model",
        how="left",
    )
    merged["covered"] = (
        (merged["human_bt_elo"] >= merged["deploy_interval_lo"])
        & (merged["human_bt_elo"] <= merged["deploy_interval_hi"])
    )
    report = merged.sort_values("judge_bt_elo", ascending=False).reset_index(drop=True)
    report.index = report.index + 1
    report.index.name = "rank"

    cols = [
        "short_model",
        "n_human_pairs",
        "n_judge_pairs",
        "human_bt_elo",
        "judge_bt_elo",
        "bt_elo_residual",
        "judge_bt_elo_se",
        "human_bt_elo_se",
        "deploy_interval_lo",
        "deploy_interval_hi",
        "deploy_interval_width",
        "covered",
    ]
    out = report[cols].copy()
    # Round Elo columns for readability
    elo_cols = [
        "human_bt_elo", "judge_bt_elo", "bt_elo_residual",
        "judge_bt_elo_se", "human_bt_elo_se",
        "deploy_interval_lo", "deploy_interval_hi", "deploy_interval_width",
    ]
    for c in elo_cols:
        if c in out.columns:
            out[c] = out[c].round(1)
    out.to_csv(path)
    print(f"[report] saved {len(out)}-row report table to {path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Held-out model conformal calibration of human-aligned BT-Elo from "
            "agreement annotations."
        )
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Agreement run directory containing agreement_annotations.json.",
    )
    parser.add_argument(
        "--artifact",
        default="agreement_annotations.json",
        help="Annotation artifact filename inside --output_dir.",
    )
    parser.add_argument(
        "--min_decisive_matches_per_model",
        type=int,
        default=25,
        help="Minimum decisive human/judge target-vs-anchor pairs for a target model.",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=0.01,
        help="L2 regularization for BT fitting.",
    )
    parser.add_argument(
        "--bootstrap_resamples",
        type=int,
        default=120,
        help="Bootstrap resamples for target judge uncertainty.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.10,
        help="Conformal miscoverage level.",
    )
    parser.add_argument(
        "--n_split_repeats",
        type=int,
        default=25,
        help="Number of repeated random split-conformal evaluations.",
    )
    parser.add_argument(
        "--bootstrap_anchor_pool",
        action="store_true",
        help=(
            "Bootstrap anchor-vs-anchor comparisons in addition to target-vs-anchor "
            "rows when estimating target Elo uncertainty. Slower but more complete."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split and bootstrap.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=1,
        help="Print progress every N targets/repeats. Set 0 to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    artifact_path = resolve_artifact_path(output_dir, args.artifact)
    heldout_partial_path = output_dir / "agreement_heldout_model_bt_elo.partial.csv"
    repeat_partial_path = output_dir / "agreement_heldout_model_bt_elo_conformal_repeats.partial.csv"

    print(f"[start] loading {artifact_path}", flush=True)
    df = load_annotation_frame(artifact_path)
    print(f"[start] loaded {len(df)} pair rows", flush=True)
    try:
        heldout = build_heldout_table(
            df,
            min_matches=args.min_decisive_matches_per_model,
            regularization=args.regularization,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_anchor_pool=args.bootstrap_anchor_pool,
            seed=args.seed,
            partial_save_path=heldout_partial_path,
            progress_every=args.progress_every,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"[heldout] completed {len(heldout)} target models", flush=True)
    conformal_test, conformal_meta = split_conformal_model_calibration(
        heldout,
        alpha=args.alpha,
        use_normalized=True,
        seed=args.seed,
    )
    operational, qhat_all = build_operational_intervals(heldout, alpha=args.alpha)
    repeat_df, repeat_summary = repeated_split_conformal(
        heldout,
        alpha=args.alpha,
        use_normalized=True,
        seed=args.seed,
        n_repeats=args.n_split_repeats,
        partial_save_path=repeat_partial_path,
        progress_every=args.progress_every,
    )
    print("[conformal] repeated split evaluation complete", flush=True)

    # --- Paths ----------------------------------------------------------------
    heldout_path = output_dir / "agreement_heldout_model_bt_elo.csv"
    conformal_path = output_dir / "agreement_heldout_model_bt_elo_conformal_test.csv"
    operational_path = output_dir / "agreement_heldout_model_bt_elo_operational.csv"
    repeat_path = output_dir / "agreement_heldout_model_bt_elo_conformal_repeats.csv"
    report_table_path = output_dir / "agreement_heldout_model_bt_elo_report.csv"

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    scatter_path = plots_dir / "conformal_scatter.png"
    interval_path = plots_dir / "conformal_intervals.png"
    operational_plot_path = plots_dir / "conformal_operational.png"
    residual_hist_path = plots_dir / "conformal_residual_histogram.png"
    residual_strength_path = plots_dir / "conformal_residual_vs_strength.png"
    coverage_path = plots_dir / "conformal_coverage_histogram.png"
    width_path = plots_dir / "conformal_interval_width_histogram.png"

    # --- Save CSVs ------------------------------------------------------------
    heldout.to_csv(heldout_path, index=False)
    conformal_test.to_csv(conformal_path, index=False)
    operational.to_csv(operational_path, index=False)
    repeat_df.to_csv(repeat_path, index=False)
    save_report_table(heldout, operational, report_table_path)

    # --- Save plots -----------------------------------------------------------
    save_scatter_plot(heldout, scatter_path)
    save_conformal_plot(conformal_test, args.alpha, interval_path)
    save_operational_plot(operational, args.alpha, qhat_all, operational_plot_path)
    save_residual_histogram(heldout, residual_hist_path)
    save_residual_vs_strength(heldout, residual_strength_path)
    save_coverage_histogram(repeat_df, args.alpha, coverage_path)
    save_interval_width_histogram(repeat_df, width_path)
    print("[plots] all plots saved", flush=True)

    if heldout_partial_path.exists():
        heldout_partial_path.unlink()
    if repeat_partial_path.exists():
        repeat_partial_path.unlink()

    # --- Compute operational coverage for summary -----------------------------
    op_covered = (
        (operational["human_bt_elo"] >= operational["deploy_interval_lo"])
        & (operational["human_bt_elo"] <= operational["deploy_interval_hi"])
    )
    rho_spearman = float(
        heldout[["judge_bt_elo", "human_bt_elo"]].corr(method="spearman").iloc[0, 1]
    )

    summary = {
        "artifact": str(artifact_path),
        "n_pair_rows": int(len(df)),
        "n_targets": int(len(heldout)),
        "mean_abs_bt_elo_residual": float(heldout["abs_bt_elo_residual"].mean()),
        "median_abs_bt_elo_residual": float(heldout["abs_bt_elo_residual"].median()),
        "median_judge_bt_elo_se": float(heldout["judge_bt_elo_se"].median()),
        "corr_pearson": float(heldout[["human_bt_elo", "judge_bt_elo"]].corr().iloc[0, 1]),
        "corr_spearman": rho_spearman,
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "bootstrap_anchor_pool": bool(args.bootstrap_anchor_pool),
        "conformal_qhat_split": float(conformal_meta["qhat"]),
        "conformal_qhat_operational": float(qhat_all),
        "conformal_empirical_coverage_split": float(conformal_meta["empirical_coverage"]),
        "conformal_operational_coverage": float(op_covered.mean()),
        "conformal_median_interval_width": float(conformal_meta["median_interval_width"]),
        "conformal_operational_median_width": float(operational["deploy_interval_width"].median()),
        "conformal_repeat_mean_coverage": float(repeat_summary["coverage_mean"]),
        "conformal_repeat_std_coverage": float(repeat_summary["coverage_std"]),
        "conformal_repeat_min_coverage": float(repeat_summary["coverage_min"]),
        "conformal_repeat_max_coverage": float(repeat_summary["coverage_max"]),
        "conformal_repeat_mean_qhat": float(repeat_summary["qhat_mean"]),
        "conformal_repeat_std_qhat": float(repeat_summary["qhat_std"]),
        "conformal_repeat_interval_width_mean": float(repeat_summary["interval_width_mean"]),
        "conformal_repeat_interval_width_std": float(repeat_summary["interval_width_std"]),
        "conformal_repeat_interval_width_median_of_medians": float(
            repeat_summary["interval_width_median_of_medians"]
        ),
    }

    summary_path = output_dir / "agreement_heldout_model_bt_elo_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    saved_files = [
        heldout_path, conformal_path, operational_path, repeat_path,
        report_table_path, scatter_path, interval_path, operational_plot_path,
        residual_hist_path, residual_strength_path,
        coverage_path, width_path, summary_path,
    ]
    for p in saved_files:
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
