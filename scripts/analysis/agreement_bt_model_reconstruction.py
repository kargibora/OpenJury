"""BT reconstruction analysis — clean visualizations.

Compares three preference-derivation strategies on the *same* rubric scores
produced by a single LLM judge run:

1. **Observed judge** — uniform average across rubric dimensions.
2. **BT (global)** — logistic-regression weights fitted on human preferences.
3. **BT (per-language)** — same, but fitted per language group.

The key question: *do rubric dimensions carry information that, when properly
weighted, closes the gap between the LLM judge and human annotators?*
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from agreement_analysis_common import (
    COLORS,
    add_figure_caption,
    add_figure_header,
    add_heatmap_grid,
    annotate_heatmap_values,
    apply_plot_theme,
    collect_group_bt_fits,
    ensure_plots_dir,
    fit_bt_weights,
    infer_score_dimensions,
    load_annotation_artifact,
    maybe_subsample_frame,
    normalized_weight_dict,
    reconstruct_preferences_from_bt_model,
    reconstruct_preferences_from_bt_model_by_group,
    resolve_artifact_path,
    save_figure,
    score_matrices,
    style_axes,
    summarize_preference_schemes,
    wrap_labels,
    write_json,
)

apply_plot_theme()

# ── Palette ────────────────────────────────────────────────────────
# Semantic scheme colours — consistent across every figure.
C_OBSERVED = "#94a3b8"  # slate-400  (baseline, muted)
C_GLOBAL = "#2563eb"  # blue-600   (BT global)
C_LANG = "#059669"  # emerald-600 (BT per-language)

SCHEME_PALETTE = {
    "Observed judge decision": C_OBSERVED,
    "Human-calibrated BT (global)": C_GLOBAL,
    "Human-calibrated BT (language-specific)": C_LANG,
}

SCHEME_SHORT = {
    "Observed judge decision": "Uniform avg\n(judge default)",
    "Human-calibrated BT (global)": "BT global\n(human-fitted)",
    "Human-calibrated BT (language-specific)": "BT per-lang\n(human-fitted)",
}


def _sc(name: str) -> str:
    """Scheme → colour."""
    return SCHEME_PALETTE.get(name, COLORS["band"])


def _ss(name: str) -> str:
    """Scheme → short label."""
    return SCHEME_SHORT.get(name, name)


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


# ═══════════════════════════════════════════════════════════════════
#  Plot 1 — Weight Profile: Uniform vs Human-Fitted BT Weights
# ═══════════════════════════════════════════════════════════════════


def save_weight_profile_plot(
    dimension_names: list[str],
    global_weights: dict[str, float],
    group_fits: dict[str, dict],
    path: Path,
) -> None:
    """Grouped bar chart: uniform vs global BT vs per-language spread."""
    k = len(dimension_names)
    uniform = {d: 1.0 / k for d in dimension_names}
    norm_global = normalized_weight_dict(global_weights)

    # Per-language normalised weights for IQR spread
    lang_matrix = None
    if group_fits:
        rows = []
        for fit in group_fits.values():
            nw = normalized_weight_dict(fit["weights"])
            rows.append([nw.get(d, 0.0) for d in dimension_names])
        if len(rows) > 1:
            lang_matrix = np.array(rows)

    dims = [d.capitalize() for d in dimension_names]
    x = np.arange(k)
    bw = 0.28

    fig, ax = plt.subplots(figsize=(10, 5.2))
    fig.subplots_adjust(top=0.82, bottom=0.14, left=0.09, right=0.96)
    add_figure_header(
        fig,
        title="Rubric Dimension Weights",
        subtitle="Uniform average vs human-calibrated Bradley-Terry weights",
    )

    # Uniform bars
    ax.bar(
        x - bw,
        [uniform[d] for d in dimension_names],
        bw,
        color=C_OBSERVED,
        edgecolor="#fff",
        linewidth=0.8,
        label="Uniform (judge default)",
        zorder=3,
    )
    # Global BT bars
    ax.bar(
        x,
        [norm_global.get(d, 0) for d in dimension_names],
        bw,
        color=C_GLOBAL,
        edgecolor="#fff",
        linewidth=0.8,
        label="BT global (human-fitted)",
        zorder=3,
    )
    # Per-language spread
    if lang_matrix is not None:
        medians = np.median(lang_matrix, axis=0)
        q25 = np.percentile(lang_matrix, 25, axis=0)
        q75 = np.percentile(lang_matrix, 75, axis=0)
        ax.bar(
            x + bw,
            medians,
            bw,
            color=C_LANG,
            edgecolor="#fff",
            linewidth=0.8,
            label=f"BT per-language median (n={len(lang_matrix)})",
            zorder=3,
        )
        ax.errorbar(
            x + bw,
            medians,
            yerr=[medians - q25, q75 - medians],
            fmt="none",
            ecolor=C_LANG,
            elinewidth=1.5,
            capsize=3,
            capthick=1.2,
            zorder=4,
            alpha=0.7,
        )

    ax.axhline(1.0 / k, ls=":", color=COLORS["muted"], lw=1, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(dims, fontsize=10.5)
    ax.set_ylabel("Normalised weight")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    style_axes(ax, grid_axis="y")
    ax.legend(
        loc="upper right",
        fontsize=9,
        frameon=True,
        facecolor="#fff",
        edgecolor=COLORS["band"],
    )
    add_figure_caption(fig, left="", right=f"{k} dimensions")
    save_figure(fig, path)


# ═══════════════════════════════════════════════════════════════════
#  Plot 2 — Alignment Uplift: grouped metric bars with deltas
# ═══════════════════════════════════════════════════════════════════


def save_alignment_uplift_plot(
    scheme_summary: pd.DataFrame,
    path: Path,
) -> None:
    """Side-by-side bars for accuracy & Elo gap with delta annotations."""
    df = scheme_summary.copy().set_index("scheme")
    schemes = [s for s in SCHEME_SHORT if s in df.index]
    if not schemes:
        return

    fig, axes = plt.subplots(
        1, 2, figsize=(11.5, 4.6), gridspec_kw={"wspace": 0.35}
    )
    fig.subplots_adjust(top=0.80, bottom=0.22, left=0.08, right=0.97)
    add_figure_header(
        fig,
        title="Does Human-Calibrated Weighting Help?",
        subtitle=(
            "Comparing uniform-average judge decisions "
            "vs BT-reweighted preferences"
        ),
    )

    short = [_ss(s) for s in schemes]
    colors = [_sc(s) for s in schemes]
    x = np.arange(len(schemes))

    # ── Left: Agreement with humans ──
    ax = axes[0]
    accs = [float(df.loc[s, "accuracy_vs_human"]) for s in schemes]
    bars = ax.bar(
        x, accs, 0.52, color=colors, edgecolor="#fff", linewidth=1, zorder=3
    )
    for bar, val in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.006,
            _pct(val),
            ha="center", va="bottom", fontsize=9.5, fontweight="semibold",
        )
    style_axes(ax, grid_axis="y")
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=9)
    ax.set_ylabel("3-class agreement")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(0, max(accs) * 1.18)
    ax.set_title(
        "Agreement with humans ↑", fontsize=11, pad=10, loc="left", color=COLORS["text"]
    )

    # ── Right: Mean Elo gap (lower = better) ──
    ax = axes[1]
    gaps = [float(df.loc[s, "mean_abs_centered_gap"]) for s in schemes]
    bars = ax.bar(
        x, gaps, 0.52, color=colors, edgecolor="#fff", linewidth=1, zorder=3
    )
    for bar, val in zip(bars, gaps):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.6,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=9.5, fontweight="semibold",
        )
    style_axes(ax, grid_axis="y")
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=9)
    ax.set_ylabel("Mean |centered Elo gap|")
    ax.set_ylim(0, max(gaps) * 1.18)
    ax.set_title(
        "Ranking gap vs humans ↓", fontsize=11, pad=10, loc="left", color=COLORS["text"]
    )

    n = int(df["n_samples"].iloc[0])
    add_figure_caption(fig, left="", right=f"n = {n:,} pairwise comparisons")
    save_figure(fig, path)


# ═══════════════════════════════════════════════════════════════════
#  Plot 3 — Per-Model Elo Scatter: scheme Elo vs human Elo
# ═══════════════════════════════════════════════════════════════════


def save_elo_scatter_plot(
    scheme_model_gaps: pd.DataFrame,
    path: Path,
) -> None:
    """One panel per scheme: centered scheme Elo vs centered human Elo."""
    schemes = scheme_model_gaps["scheme"].unique().tolist()
    n_schemes = len(schemes)
    if n_schemes == 0:
        return

    fig, axes = plt.subplots(
        1,
        n_schemes,
        figsize=(5.2 * n_schemes, 5.2),
        squeeze=False,
        sharey=True,
    )
    axes = axes[0]
    fig.subplots_adjust(
        top=0.82, bottom=0.14, wspace=0.12, left=0.07, right=0.97
    )
    add_figure_header(
        fig,
        title="Per-Model Elo: Scheme vs Human",
        subtitle="Points on the diagonal = perfect agreement with human ranking",
    )

    for i, (scheme, ax) in enumerate(zip(schemes, axes)):
        sub = scheme_model_gaps[scheme_model_gaps["scheme"] == scheme].copy()
        c = _sc(scheme)

        # Identity line + tolerance band
        lo = (
            min(
                sub["human_elo_centered"].min(),
                sub["scheme_elo_centered"].min(),
            )
            - 20
        )
        hi = (
            max(
                sub["human_elo_centered"].max(),
                sub["scheme_elo_centered"].max(),
            )
            + 20
        )
        ax.plot(
            [lo, hi], [lo, hi], ls="--", color=COLORS["band"], lw=1.2, zorder=1
        )
        ax.fill_between(
            [lo, hi],
            [lo - 30, hi - 30],
            [lo + 30, hi + 30],
            color=COLORS["band"],
            alpha=0.12,
            zorder=0,
        )

        ax.scatter(
            sub["human_elo_centered"],
            sub["scheme_elo_centered"],
            s=44,
            color=c,
            edgecolor="#fff",
            linewidth=0.6,
            alpha=0.85,
            zorder=3,
        )

        spearman = sub["human_elo_centered"].corr(
            sub["scheme_elo_centered"], method="spearman"
        )
        mae = sub["abs_centered_elo_gap"].mean()

        style_axes(ax, grid_axis=None)
        ax.set_xlabel("Human Elo (centered)")
        if i == 0:
            ax.set_ylabel("Scheme Elo (centered)")
        ax.set_title(
            f"{_ss(scheme).replace(chr(10), ' ')}   ρ={spearman:.2f}  MAE={mae:.0f}",
            fontsize=9.5, pad=8,
        )
        ax.set_aspect("equal", adjustable="datalim")

    add_figure_caption(fig, left="", right="")
    save_figure(fig, path)


# ═══════════════════════════════════════════════════════════════════
#  Plot 4 — Per-Model Heatmap (top-N most-affected models)
# ═══════════════════════════════════════════════════════════════════


def save_model_heatmap(
    scheme_summary: pd.DataFrame,
    scheme_model_gaps: pd.DataFrame,
    path: Path,
    *,
    top_models: int = 15,
) -> None:
    model_order = (
        scheme_model_gaps.groupby(["model", "short_model"], as_index=False)
        .agg(max_gap=("abs_centered_elo_gap", "max"))
        .sort_values("max_gap", ascending=False)
        .head(top_models)
    )
    scheme_order = scheme_summary.sort_values("mean_abs_centered_gap")[
        "scheme"
    ].tolist()
    heatmap = (
        scheme_model_gaps.pivot_table(
            index="scheme",
            columns="short_model",
            values="centered_elo_gap",
        ).reindex(
            index=scheme_order,
            columns=model_order["short_model"].tolist(),
        )
    )
    vmax = np.nanmax(np.abs(heatmap.to_numpy(dtype=float)))

    n_rows, n_cols = heatmap.shape
    fig_w = max(10, 0.95 * n_cols + 3)
    fig_h = max(3.6, 1.1 * n_rows + 2.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.subplots_adjust(top=0.83, bottom=0.18)
    add_figure_header(fig, title="Centered Elo Gap by Model × Scheme")

    im = ax.imshow(
        heatmap.fillna(0.0).to_numpy(),
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    style_axes(ax, grid_axis=None)
    add_heatmap_grid(ax, heatmap.shape)
    annotate_heatmap_values(
        ax, heatmap.to_numpy(dtype=float), fmt="{:+.0f}", limit=120
    )
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(heatmap.columns, rotation=40, ha="right", fontsize=8.5)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(
        [_ss(s).replace("\n", " ") for s in heatmap.index], fontsize=9
    )
    cb = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label("Scheme Elo − Human Elo (centered)", fontsize=9)
    add_figure_caption(
        fig, left="", right=f"Top {n_cols} most-affected models"
    )
    save_figure(fig, path)


# ═══════════════════════════════════════════════════════════════════
#  Plot 5 — 4-Panel Summary Dashboard
# ═══════════════════════════════════════════════════════════════════


def save_dashboard_plot(
    scheme_summary: pd.DataFrame,
    dimension_names: list[str],
    global_weights: dict[str, float],
    path: Path,
) -> None:
    """Compact 2×2 summary card."""
    df = scheme_summary.copy().set_index("scheme")
    schemes = [s for s in SCHEME_SHORT if s in df.index]
    if not schemes:
        return

    fig = plt.figure(figsize=(14, 8.5))
    fig.subplots_adjust(top=0.88, bottom=0.08, hspace=0.42, wspace=0.30)
    add_figure_header(
        fig,
        title="BT Reconstruction — Summary Dashboard",
        subtitle=(
            "Effect of human-calibrated rubric weighting "
            "on LLM judge alignment"
        ),
    )

    gs = fig.add_gridspec(2, 2, left=0.07, right=0.96, top=0.82, bottom=0.10)

    short = [_ss(s) for s in schemes]
    colors = [_sc(s) for s in schemes]

    # ── Panel A: Agreement ──
    ax = fig.add_subplot(gs[0, 0])
    accs = [float(df.loc[s, "accuracy_vs_human"]) for s in schemes]
    bars = ax.bar(
        range(len(schemes)),
        accs,
        0.55,
        color=colors,
        edgecolor="#fff",
        lw=1,
        zorder=3,
    )
    for bar, val in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            _pct(val),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="semibold",
        )
    ax.set_xticks(range(len(schemes)))
    ax.set_xticklabels(short, fontsize=8.5)
    ax.set_ylabel("Agreement")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(0, max(accs) * 1.16)
    style_axes(ax, grid_axis="y")
    ax.set_title(
        "A   Human Agreement ↑",
        loc="left",
        fontsize=10.5,
        fontweight="semibold",
    )

    # ── Panel B: Mean Elo Gap ──
    ax = fig.add_subplot(gs[0, 1])
    gaps = [float(df.loc[s, "mean_abs_centered_gap"]) for s in schemes]
    bars = ax.bar(
        range(len(schemes)),
        gaps,
        0.55,
        color=colors,
        edgecolor="#fff",
        lw=1,
        zorder=3,
    )
    for bar, val in zip(bars, gaps):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="semibold",
        )
    ax.set_xticks(range(len(schemes)))
    ax.set_xticklabels(short, fontsize=8.5)
    ax.set_ylabel("Mean |ΔElo| vs humans")
    ax.set_ylim(0, max(gaps) * 1.18)
    style_axes(ax, grid_axis="y")
    ax.set_title(
        "B   Elo Gap ↓", loc="left", fontsize=10.5, fontweight="semibold"
    )

    # ── Panel C: Weight Profile ──
    ax = fig.add_subplot(gs[1, 0])
    k = len(dimension_names)
    uniform_w = 1.0 / k
    norm_g = normalized_weight_dict(global_weights)
    dims_cap = [d.capitalize() for d in dimension_names]
    xd = np.arange(k)
    bwd = 0.35
    ax.bar(
        xd - bwd / 2,
        [uniform_w] * k,
        bwd,
        color=C_OBSERVED,
        edgecolor="#fff",
        lw=0.8,
        label="Uniform",
        zorder=3,
    )
    ax.bar(
        xd + bwd / 2,
        [norm_g.get(d, 0) for d in dimension_names],
        bwd,
        color=C_GLOBAL,
        edgecolor="#fff",
        lw=0.8,
        label="BT (human-fitted)",
        zorder=3,
    )
    ax.axhline(uniform_w, ls=":", color=COLORS["muted"], lw=0.8)
    ax.set_xticks(xd)
    ax.set_xticklabels(dims_cap, fontsize=8.5)
    ax.set_ylabel("Normalised weight")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    style_axes(ax, grid_axis="y")
    ax.legend(
        fontsize=8,
        loc="upper right",
        frameon=True,
        facecolor="#fff",
        edgecolor=COLORS["band"],
    )
    ax.set_title(
        "C   Learned Weights",
        loc="left",
        fontsize=10.5,
        fontweight="semibold",
    )

    # ── Panel D: Spearman ranking correlation ──
    ax = fig.add_subplot(gs[1, 1])
    spearman_vals = [
        float(df.loc[s, "spearman_human_vs_scheme_elo"])
        if pd.notna(df.loc[s, "spearman_human_vs_scheme_elo"])
        else 0
        for s in schemes
    ]
    bars = ax.bar(
        range(len(schemes)),
        spearman_vals,
        0.55,
        color=colors,
        edgecolor="#fff",
        lw=1,
        zorder=3,
    )
    for bar, val in zip(bars, spearman_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="semibold",
        )
    ax.set_xticks(range(len(schemes)))
    ax.set_xticklabels(short, fontsize=8.5)
    ax.set_ylabel("Spearman ρ")
    ax.set_ylim(0, 1.05)
    style_axes(ax, grid_axis="y")
    ax.set_title(
        "D   Ranking Correlation ↑",
        loc="left",
        fontsize=10.5,
        fontweight="semibold",
    )

    n = int(df["n_samples"].iloc[0])
    add_figure_caption(fig, left="", right=f"n = {n:,} pairwise comparisons")
    save_figure(fig, path)


# ═══════════════════════════════════════════════════════════════════
#  Cross-validated BT reconstruction
# ═══════════════════════════════════════════════════════════════════


def cv_reconstruct_global(
    df: pd.DataFrame,
    dimension_names: list[str],
    *,
    n_folds: int = 5,
    regularization: float = 0.01,
    tie_band: float = 0.05,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series]:
    """K-fold cross-validated BT global reconstruction.

    For each fold: fit BT weights on the training split, predict
    preferences on the held-out split.  Returns out-of-sample
    predictions for every row.
    """
    prefs = pd.Series(np.nan, index=df.index, dtype=float)
    proba_a = pd.Series(np.nan, index=df.index, dtype=float)

    valid_mask = df["human_pref"].notna()
    valid_idx = df.index[valid_mask].to_numpy()

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (train_pos, test_pos) in enumerate(kf.split(valid_idx)):
        train_idx = valid_idx[train_pos]
        test_idx = valid_idx[test_pos]

        fit = fit_bt_weights(
            df.loc[train_idx].copy(),
            "human_pref",
            dimension_names=dimension_names,
            regularization=regularization,
        )
        fold_prefs, fold_proba = reconstruct_preferences_from_bt_model(
            df.loc[test_idx],
            fit["model"],
            dimension_names=dimension_names,
            tie_band=tie_band,
        )
        prefs.loc[test_idx] = fold_prefs
        proba_a.loc[test_idx] = fold_proba

    return prefs, proba_a


def cv_reconstruct_by_group(
    df: pd.DataFrame,
    group_col: str,
    dimension_names: list[str],
    *,
    n_folds: int = 5,
    min_group_samples: int = 100,
    regularization: float = 0.01,
    tie_band: float = 0.05,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series]:
    """K-fold cross-validated BT per-group reconstruction.

    Within each fold's training split, fits a separate BT model per
    group (language).  Groups too small in the training split fall back
    to the global BT model fitted on that fold's training data.
    """
    prefs = pd.Series(np.nan, index=df.index, dtype=float)
    proba_a = pd.Series(np.nan, index=df.index, dtype=float)

    valid_mask = df["human_pref"].notna()
    valid_idx = df.index[valid_mask].to_numpy()

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (train_pos, test_pos) in enumerate(kf.split(valid_idx)):
        train_idx = valid_idx[train_pos]
        test_idx = valid_idx[test_pos]
        train_df = df.loc[train_idx]

        # Global fallback for this fold
        fold_global = fit_bt_weights(
            train_df.copy(),
            "human_pref",
            dimension_names=dimension_names,
            regularization=regularization,
        )

        # Per-group fits on this fold's training data
        fold_group_fits = collect_group_bt_fits(
            train_df,
            group_col,
            "human_pref",
            min_samples=min_group_samples,
            dimension_names=dimension_names,
            regularization=regularization,
        )

        fold_prefs, fold_proba = reconstruct_preferences_from_bt_model_by_group(
            df.loc[test_idx],
            group_col,
            fold_group_fits,
            dimension_names=dimension_names,
            fallback_fit=fold_global,
            tie_band=tie_band,
        )
        prefs.loc[test_idx] = fold_prefs
        proba_a.loc[test_idx] = fold_proba

    return prefs, proba_a


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fit human-calibrated BT weights on rubric score deltas and "
            "compare the resulting preference scheme against the raw judge."
        ),
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="Directory with agreement_annotations.json.",
    )
    p.add_argument("--artifact", default="agreement_annotations.json")
    p.add_argument(
        "--group_col",
        default="meta_lang",
        help="Metadata column for per-group BT fits.",
    )
    p.add_argument("--min_group_samples", type=int, default=100)
    p.add_argument("--regularization", type=float, default=0.01)
    p.add_argument("--elo_k", type=float, default=32.0)
    p.add_argument("--tie_band", type=float, default=0.05)
    p.add_argument("--top_models", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (0 = no CV, in-sample).",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional row cap for faster iteration.",
    )
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    plots_dir = ensure_plots_dir(output_dir)
    artifact, meta, df = load_annotation_artifact(output_dir, args.artifact)
    _ = artifact
    df = maybe_subsample_frame(df, max_samples=args.max_samples, seed=args.seed)

    dimension_names = infer_score_dimensions(df)
    if not dimension_names:
        raise SystemExit(
            "No rubric score dimensions found in the annotation artifact."
        )

    # ── Fit BT models ──────────────────────────────────────────────
    human_bt_global = fit_bt_weights(
        df[df["human_pref"].notna()].copy(),
        "human_pref",
        dimension_names=dimension_names,
        regularization=args.regularization,
    )
    human_bt_by_group = collect_group_bt_fits(
        df,
        args.group_col,
        "human_pref",
        min_samples=args.min_group_samples,
        dimension_names=dimension_names,
        regularization=args.regularization,
    )

    # ── Reconstruct preferences (cross-validated or in-sample) ─────
    use_cv = args.n_folds >= 2
    cv_tag = f" ({args.n_folds}-fold CV)" if use_cv else " (in-sample)"
    print(f"Reconstruction mode: {cv_tag.strip()}")

    if use_cv:
        global_prefs, global_proba = cv_reconstruct_global(
            df,
            dimension_names,
            n_folds=args.n_folds,
            regularization=args.regularization,
            tie_band=args.tie_band,
            seed=args.seed,
        )
    else:
        global_prefs, global_proba = reconstruct_preferences_from_bt_model(
            df,
            human_bt_global["model"],
            dimension_names=dimension_names,
            tie_band=args.tie_band,
        )

    scheme_prefs = {
        "Observed judge decision": df["judge_pref"],
        "Human-calibrated BT (global)": global_prefs,
    }
    scheme_proba = {
        "Human-calibrated BT (global)": global_proba,
    }

    if human_bt_by_group:
        if use_cv:
            group_prefs, group_proba = cv_reconstruct_by_group(
                df,
                args.group_col,
                dimension_names,
                n_folds=args.n_folds,
                min_group_samples=args.min_group_samples,
                regularization=args.regularization,
                tie_band=args.tie_band,
                seed=args.seed,
            )
        else:
            group_prefs, group_proba = reconstruct_preferences_from_bt_model_by_group(
                df,
                args.group_col,
                human_bt_by_group,
                dimension_names=dimension_names,
                fallback_fit=human_bt_global,
                tie_band=args.tie_band,
            )
        scheme_prefs["Human-calibrated BT (language-specific)"] = group_prefs
        scheme_proba["Human-calibrated BT (language-specific)"] = group_proba

    # ── Compute scheme summary ─────────────────────────────────────
    scheme_summary, scheme_model_gaps = summarize_preference_schemes(
        df,
        scheme_prefs,
        elo_k=args.elo_k,
        elo_seed=args.seed,
    )

    # ── Build reconstruction table ─────────────────────────────────
    reconstruction_table = pd.DataFrame(
        {
            "instruction_id": df.get("instruction_id"),
            "model_a": df.get("model_a"),
            "model_b": df.get("model_b"),
            "human_pref": df.get("human_pref"),
            "judge_pref": df.get("judge_pref"),
            "group": (
                df[args.group_col]
                if args.group_col in df.columns
                else pd.Series(np.nan, index=df.index)
            ),
            "bt_global_pref": global_prefs,
            "bt_global_proba_a": global_proba,
        }
    )
    if "Human-calibrated BT (language-specific)" in scheme_prefs:
        reconstruction_table["bt_lang_pref"] = scheme_prefs[
            "Human-calibrated BT (language-specific)"
        ]
        reconstruction_table["bt_lang_proba_a"] = scheme_proba[
            "Human-calibrated BT (language-specific)"
        ]

    # ── Save CSV outputs ───────────────────────────────────────────
    scheme_summary_path = output_dir / "agreement_bt_human_model_scheme_summary.csv"
    scheme_model_path = output_dir / "agreement_bt_human_model_scheme_gaps.csv"
    reconstruction_path = output_dir / "agreement_bt_human_model_reconstruction.csv"
    global_weights_path = output_dir / "agreement_bt_human_model_global_weights.csv"
    group_weights_path = output_dir / "agreement_bt_human_model_group_weights.csv"

    scheme_summary.to_csv(scheme_summary_path, index=False)
    scheme_model_gaps.to_csv(scheme_model_path, index=False)
    reconstruction_table.to_csv(reconstruction_path, index=False)

    # Save learned weights
    k = len(dimension_names)
    gw = normalized_weight_dict(human_bt_global["weights"])
    pd.DataFrame(
        [
            {
                "scope": "global",
                "dimension": d,
                "raw_weight": float(human_bt_global["weights"].get(d, 0)),
                "norm_weight": float(gw.get(d, 0)),
                "uniform_weight": 1.0 / k,
            }
            for d in dimension_names
        ]
    ).to_csv(global_weights_path, index=False)

    group_weight_rows = []
    for grp, fit in human_bt_by_group.items():
        nw = normalized_weight_dict(fit["weights"])
        for d in dimension_names:
            group_weight_rows.append(
                {
                    "group": grp,
                    "dimension": d,
                    "raw_weight": float(fit["weights"].get(d, 0)),
                    "norm_weight": float(nw.get(d, 0)),
                }
            )
    if group_weight_rows:
        pd.DataFrame(group_weight_rows).to_csv(group_weights_path, index=False)

    # ── Generate plots ─────────────────────────────────────────────
    print("Generating plots …")

    save_weight_profile_plot(
        dimension_names,
        human_bt_global["weights"],
        human_bt_by_group,
        plots_dir / "agreement_bt_human_model_weight_profile.pdf",
    )
    save_alignment_uplift_plot(
        scheme_summary,
        plots_dir / "agreement_bt_human_model_alignment_uplift.pdf",
    )
    save_elo_scatter_plot(
        scheme_model_gaps,
        plots_dir / "agreement_bt_human_model_elo_scatter.pdf",
    )
    save_model_heatmap(
        scheme_summary,
        scheme_model_gaps,
        plots_dir / "agreement_bt_human_model_scheme_heatmap.pdf",
        top_models=args.top_models,
    )
    save_dashboard_plot(
        scheme_summary,
        dimension_names,
        human_bt_global["weights"],
        plots_dir / "agreement_bt_human_model_dashboard.pdf",
    )

    # ── JSON summary ───────────────────────────────────────────────
    summary_json_path = (
        output_dir / "agreement_bt_human_model_reconstruction_summary.json"
    )
    summary = {
        "artifact": str(resolve_artifact_path(output_dir, args.artifact).resolve()),
        "dataset": meta["dataset"],
        "judge_model": meta["judge_model"],
        "group_col": args.group_col,
        "plots_dir": str(plots_dir),
        "n_pair_rows": int(len(df)),
        "n_dimensions": int(len(dimension_names)),
        "dimension_names": dimension_names,
        "n_groups_with_human_bt_model": int(len(human_bt_by_group)),
        "global_bt_training_accuracy": (
            float(human_bt_global["training_accuracy"])
            if pd.notna(human_bt_global["training_accuracy"])
            else None
        ),
        "global_bt_weights_normalized": {
            d: round(v, 4) for d, v in gw.items()
        },
        "uniform_weight": round(1.0 / k, 4),
        "tie_band": float(args.tie_band),
        "regularization": float(args.regularization),
        "n_folds": args.n_folds,
        "evaluation_mode": "cross-validated" if use_cv else "in-sample",
        "schemes": {},
    }
    for _, row in scheme_summary.iterrows():
        summary["schemes"][row["scheme"]] = {
            "accuracy_vs_human": round(float(row["accuracy_vs_human"]), 4),
            "decisive_accuracy": (
                round(float(row["decisive_accuracy_vs_human"]), 4)
                if pd.notna(row["decisive_accuracy_vs_human"])
                else None
            ),
            "mean_elo_gap": round(float(row["mean_abs_centered_gap"]), 2),
            "max_elo_gap": round(float(row["max_abs_centered_gap"]), 2),
            "tie_rate": round(float(row["scheme_tie_rate"]), 4),
            "spearman": round(float(row["spearman_human_vs_scheme_elo"]), 4),
        }
    write_json(summary_json_path, summary)

    # ── Print summary ──────────────────────────────────────────────
    print()
    print(
        scheme_summary[
            [
                "scheme",
                "accuracy_vs_human",
                "decisive_accuracy_vs_human",
                "mean_abs_centered_gap",
                "scheme_tie_rate",
                "spearman_human_vs_scheme_elo",
            ]
        ].to_string(index=False)
    )
    print()
    print("Weights (global BT vs uniform):")
    for d in dimension_names:
        print(f"  {d:>14s}:  BT = {gw[d]:5.1%}   uniform = {1 / k:5.1%}")
    print()
    for p in sorted(plots_dir.glob("*.pdf")):
        print(f"  📊 {p}")
    for p in [
        scheme_summary_path,
        scheme_model_path,
        reconstruction_path,
        global_weights_path,
        summary_json_path,
    ]:
        if p.exists():
            print(f"  📄 {p}")


if __name__ == "__main__":
    main()
