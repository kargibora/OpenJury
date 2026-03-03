from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agreement_analysis_common import (
    COLORS,
    add_heatmap_grid,
    annotate_heatmap_values,
    apply_plot_theme,
    collect_language_gap_tables,
    ensure_plots_dir,
    format_percent_axis,
    load_annotation_artifact,
    maybe_subsample_frame,
    pct,
    save_figure,
    style_axes,
    wrap_labels,
    write_json,
)

apply_plot_theme()


def save_language_dashboard(
    summary: pd.DataFrame,
    path: Path,
    *,
    top_languages: int,
) -> None:
    plot_df = summary.head(top_languages).sort_values("mean_abs_centered_gap")
    size_scale = np.sqrt(plot_df["n_valid"].clip(lower=1))
    size_min = float(size_scale.min())
    size_span = max(float(size_scale.max()) - size_min, 1e-9)
    bubble_sizes = 120 + 300 * ((size_scale - size_min) / size_span)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16.2, 7.0),
        gridspec_kw={"width_ratios": [1.05, 1.0]},
    )

    acc_min = plot_df["accuracy"].min()
    acc_span = max(plot_df["accuracy"].max() - acc_min, 1e-9)
    colors = plt.cm.YlGnBu(0.28 + 0.62 * (plot_df["accuracy"] - acc_min) / acc_span)

    axes[0].barh(
        plot_df["lang"],
        plot_df["mean_abs_centered_gap"],
        color=colors,
        edgecolor="#ffffff",
        linewidth=1.0,
    )
    style_axes(axes[0], grid_axis="x")
    axes[0].set_title("Languages Ranked by Mean Centered Elo Gap", pad=14, loc="left")
    axes[0].set_xlabel("Mean |centered judge Elo - centered human Elo|")
    axes[0].set_ylabel("Language")
    axes[0].set_xlim(0, plot_df["mean_abs_centered_gap"].max() * 1.2)
    axes[0].text(
        0.0,
        1.02,
        "Color encodes agreement with humans.",
        transform=axes[0].transAxes,
        fontsize=9,
        color=COLORS["muted"],
    )

    sc = axes[1].scatter(
        plot_df["human_tie_rate"],
        plot_df["judge_tie_rate"],
        s=bubble_sizes,
        c=plot_df["mean_abs_centered_gap"],
        cmap="inferno",
        alpha=0.95,
        edgecolor="#ffffff",
        linewidth=0.8,
    )
    style_axes(axes[1], grid_axis="both")
    lo = min(plot_df["human_tie_rate"].min(), plot_df["judge_tie_rate"].min())
    hi = max(plot_df["human_tie_rate"].max(), plot_df["judge_tie_rate"].max())
    pad = max((hi - lo) * 0.08, 0.01)
    axes[1].plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        color=COLORS["muted"],
        linewidth=1.4,
        zorder=0,
    )
    label_df = (
        plot_df.assign(tie_gap_abs=(plot_df["human_tie_rate"] - plot_df["judge_tie_rate"]).abs())
        .sort_values(["tie_gap_abs", "mean_abs_centered_gap"], ascending=False)
        .head(min(6, len(plot_df)))
    )
    for _, row in label_df.iterrows():
        axes[1].annotate(
            str(row["lang"]),
            (row["human_tie_rate"], row["judge_tie_rate"]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "#ffffff",
                "edgecolor": COLORS["band"],
                "alpha": 0.88,
            },
        )
    axes[1].set_title("Human Tie Rate vs Judge Tie Rate", pad=14, loc="left")
    axes[1].set_xlabel("Human tie rate")
    axes[1].set_ylabel("Judge tie rate")
    axes[1].set_xlim(lo - pad, hi + pad)
    axes[1].set_ylim(lo - pad, hi + pad)
    format_percent_axis(axes[1], axis="x")
    format_percent_axis(axes[1], axis="y")
    cb = plt.colorbar(sc, ax=axes[1], pad=0.02)
    cb.set_label("Mean absolute centered Elo gap")
    save_figure(fig, path)


def save_language_model_heatmap(
    language_summary: pd.DataFrame,
    language_model_gaps: pd.DataFrame,
    path: Path,
    *,
    top_models: int,
) -> None:
    language_order = language_summary["lang"].tolist()
    model_order = (
        language_model_gaps.groupby(["model", "short_model"], as_index=False)
        .agg(
            max_abs_centered_gap=("abs_centered_elo_gap", "max"),
            std_centered_gap=("centered_elo_gap", "std"),
        )
        .sort_values(["max_abs_centered_gap", "std_centered_gap"], ascending=False)
    )
    top_model_names = model_order.head(top_models)["short_model"].tolist()
    heatmap = (
        language_model_gaps.pivot_table(
            index="lang",
            columns="short_model",
            values="centered_elo_gap",
        ).reindex(index=language_order, columns=top_model_names)
    )
    vmax = np.nanmax(np.abs(heatmap.to_numpy(dtype=float)))

    fig, ax = plt.subplots(figsize=(14.8, max(5.8, 0.42 * len(heatmap))))
    im = ax.imshow(
        heatmap.fillna(0.0).to_numpy(),
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    style_axes(ax, grid_axis=None)
    add_heatmap_grid(ax, heatmap.shape)
    annotate_heatmap_values(ax, heatmap.to_numpy(dtype=float), fmt="{:+.0f}")
    ax.set_title("Centered Elo Gap by Language and Model", pad=14, loc="left")
    ax.set_xlabel("Model")
    ax.set_ylabel("Language")
    ax.set_xticks(np.arange(len(heatmap.columns)))
    ax.set_xticklabels(wrap_labels(heatmap.columns, width=16), rotation=45, ha="right")
    ax.set_yticks(np.arange(len(heatmap.index)))
    ax.set_yticklabels(wrap_labels(heatmap.index, width=14))
    cb = plt.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Centered judge Elo - centered human Elo")
    save_figure(fig, path)


def save_tie_and_stability_dashboard(
    language_summary: pd.DataFrame,
    language_model_gaps: pd.DataFrame,
    path: Path,
    *,
    top_languages: int,
    top_models: int,
) -> pd.DataFrame:
    model_language_stability = (
        language_model_gaps.groupby(["model", "short_model"], as_index=False)
        .agg(
            mean_abs_lang_gap=("abs_centered_elo_gap", "mean"),
            max_abs_lang_gap=("abs_centered_elo_gap", "max"),
            std_lang_gap=("centered_elo_gap", "std"),
        )
        .sort_values("std_lang_gap", ascending=False)
        .reset_index(drop=True)
    )

    plot_df = language_summary.head(top_languages).sort_values("mean_abs_centered_gap")
    stability_plot = model_language_stability.head(top_models).sort_values("std_lang_gap")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16.2, 7.0),
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )

    y = np.arange(len(plot_df))
    style_axes(axes[0], grid_axis="x")
    axes[0].hlines(
        y,
        plot_df["mean_abs_centered_gap_decisive"],
        plot_df["mean_abs_centered_gap"],
        color=COLORS["band"],
        linewidth=3.0,
    )
    axes[0].plot(
        plot_df["mean_abs_centered_gap"],
        y,
        marker="o",
        color=COLORS["judge"],
        linewidth=2,
        label="All valid pairs",
    )
    axes[0].plot(
        plot_df["mean_abs_centered_gap_decisive"],
        y,
        marker="D",
        color=COLORS["human"],
        linewidth=2,
        label="Decisive-only pairs",
    )
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(plot_df["lang"])
    axes[0].set_title("How Much Do Ties Explain the Gap?", pad=14, loc="left")
    axes[0].set_xlabel("Mean |centered Elo gap|")
    axes[0].legend(frameon=False, loc="lower right")

    bar_colors = plt.cm.OrRd(
        np.linspace(0.42, 0.82, max(len(stability_plot), 1))
    )
    axes[1].barh(
        stability_plot["short_model"],
        stability_plot["std_lang_gap"],
        color=bar_colors,
        edgecolor="#ffffff",
        linewidth=1.0,
    )
    style_axes(axes[1], grid_axis="x")
    for _, row in stability_plot.iterrows():
        axes[1].text(
            row["std_lang_gap"] + 0.6,
            row["short_model"],
            f"{row['std_lang_gap']:.1f}",
            va="center",
            fontsize=8.5,
            color=COLORS["muted"],
        )
    axes[1].set_title("Models with the Largest Cross-Language Gap Drift", pad=14, loc="left")
    axes[1].set_xlabel("Std. dev. of centered Elo gap across languages")
    axes[1].set_ylabel("Model")
    axes[1].set_xlim(0, stability_plot["std_lang_gap"].max() * 1.18)
    save_figure(fig, path)
    return model_language_stability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate multilingual Elo-gap diagnostics from agreement annotations."
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
        "--language_col",
        default="meta_lang",
        help="Flattened language metadata column.",
    )
    parser.add_argument(
        "--min_language_samples",
        type=int,
        default=100,
        help="Minimum samples per language slice.",
    )
    parser.add_argument(
        "--top_languages",
        type=int,
        default=18,
        help="Maximum number of languages to show per plot.",
    )
    parser.add_argument(
        "--top_models",
        type=int,
        default=18,
        help="Maximum number of models to show in the heatmap/stability plots.",
    )
    parser.add_argument(
        "--elo_k",
        type=float,
        default=32.0,
        help="Elo K-factor.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for Elo match shuffling.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional row cap for faster plot iteration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    plots_dir = ensure_plots_dir(output_dir)
    artifact, meta, df = load_annotation_artifact(output_dir, args.artifact)
    _ = artifact
    df = maybe_subsample_frame(df, max_samples=args.max_samples, seed=args.seed)

    language_summary, language_model_gaps = collect_language_gap_tables(
        df,
        meta,
        language_col=args.language_col,
        min_samples=args.min_language_samples,
        elo_k=args.elo_k,
        elo_seed=args.seed,
    )
    if language_summary.empty:
        raise SystemExit("No eligible language slices were found for multilingual analysis.")

    language_summary_path = output_dir / "agreement_language_gap_summary.csv"
    language_model_gap_path = output_dir / "agreement_language_model_centered_gaps.csv"
    stability_path = output_dir / "agreement_model_language_gap_stability.csv"
    dashboard_path = plots_dir / "agreement_language_gap_dashboard.png"
    heatmap_path = plots_dir / "agreement_language_model_gap_heatmap.png"
    tie_stability_path = plots_dir / "agreement_language_ties_and_stability.png"
    summary_json_path = output_dir / "agreement_multilingual_analysis_summary.json"

    language_summary.to_csv(language_summary_path, index=False)
    language_model_gaps.to_csv(language_model_gap_path, index=False)
    save_language_dashboard(
        language_summary,
        dashboard_path,
        top_languages=args.top_languages,
    )
    save_language_model_heatmap(
        language_summary,
        language_model_gaps,
        heatmap_path,
        top_models=args.top_models,
    )
    model_language_stability = save_tie_and_stability_dashboard(
        language_summary,
        language_model_gaps,
        tie_stability_path,
        top_languages=min(14, args.top_languages),
        top_models=min(18, args.top_models),
    )
    model_language_stability.to_csv(stability_path, index=False)

    summary = {
        "artifact": str((output_dir / args.artifact).resolve()),
        "n_pair_rows": int(len(df)),
        "n_languages": int(len(language_summary)),
        "language_col": args.language_col,
        "plots_dir": str(plots_dir),
        "min_language_samples": int(args.min_language_samples),
        "mean_language_accuracy": float(language_summary["accuracy"].mean()),
        "mean_language_gap": float(language_summary["mean_abs_centered_gap"].mean()),
        "max_language_gap": float(language_summary["max_abs_centered_gap"].max()),
        "worst_language_by_gap": str(language_summary.iloc[0]["lang"]),
        "most_stable_language_by_gap": str(
            language_summary.sort_values("mean_abs_centered_gap").iloc[0]["lang"]
        ),
    }
    write_json(summary_json_path, summary)

    print(language_summary.head(10).to_string(index=False))
    print(f"Saved: {language_summary_path}")
    print(f"Saved: {language_model_gap_path}")
    print(f"Saved: {stability_path}")
    print(f"Saved: {dashboard_path}")
    print(f"Saved: {heatmap_path}")
    print(f"Saved: {tie_stability_path}")
    print(f"Saved: {summary_json_path}")


if __name__ == "__main__":
    main()
