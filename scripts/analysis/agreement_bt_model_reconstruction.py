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
    add_figure_caption,
    add_figure_header,
    annotate_heatmap_values,
    apply_plot_theme,
    collect_group_bt_fits,
    ensure_plots_dir,
    fit_bt_weights,
    format_percent_axis,
    infer_score_dimensions,
    load_annotation_artifact,
    maybe_subsample_frame,
    reconstruct_preferences_from_bt_model,
    reconstruct_preferences_from_bt_model_by_group,
    save_figure,
    style_axes,
    summarize_preference_schemes,
    wrap_labels,
    write_json,
)

apply_plot_theme()


def scheme_color(scheme: str) -> str:
    if "Observed judge" in scheme:
        return COLORS["judge"]
    if "language-specific" in scheme:
        return COLORS["human"]
    if "(global)" in scheme:
        return COLORS["neutral"]
    return COLORS["band"]


def save_scheme_summary_plot(weight_scheme_summary: pd.DataFrame, path: Path) -> None:
    plot_df = weight_scheme_summary.copy().sort_values("mean_abs_centered_gap")
    baseline_row = plot_df.loc[plot_df["scheme"] == "Observed judge decision"]
    baseline_gap = (
        float(baseline_row["mean_abs_centered_gap"].iloc[0])
        if not baseline_row.empty
        else float(plot_df["mean_abs_centered_gap"].max())
    )
    plot_df["scheme_color"] = plot_df["scheme"].map(scheme_color)
    plot_df["gap_improvement_vs_observed"] = baseline_gap - plot_df["mean_abs_centered_gap"]
    y_labels = wrap_labels(plot_df["scheme"], width=24)
    fig, ax = plt.subplots(figsize=(11.6, 5.8))
    fig.subplots_adjust(top=0.8, bottom=0.16)
    add_figure_header(
        fig,
        title="Human-Calibrated BT Reconstruction",
        subtitle="Lower centered Elo gap means rankings closer to humans.",
    )

    y = np.arange(len(plot_df))
    style_axes(ax, grid_axis="x")
    ax.axvline(
        baseline_gap,
        linestyle="--",
        color=COLORS["muted"],
        linewidth=1.3,
        zorder=0,
    )
    for yi, (_, row) in enumerate(plot_df.iterrows()):
        ax.plot(
            [row["mean_abs_centered_gap"], baseline_gap],
            [yi, yi],
            color=COLORS["band"],
            linewidth=5.0,
            solid_capstyle="round",
            zorder=1,
        )
        ax.scatter(
            baseline_gap,
            yi,
            s=48,
            color="#cbd5e1",
            edgecolor="#ffffff",
            linewidth=0.8,
            zorder=2,
        )
        ax.scatter(
            row["mean_abs_centered_gap"],
            yi,
            s=140,
            color=row["scheme_color"],
            edgecolor="#ffffff",
            linewidth=1.1,
            zorder=3,
        )
        label = (
            f"{row['mean_abs_centered_gap']:.1f}"
            if row["scheme"] == "Observed judge decision"
            else f"{row['mean_abs_centered_gap']:.1f} ({row['gap_improvement_vs_observed']:+.1f})"
        )
        ax.text(
            max(row["mean_abs_centered_gap"], baseline_gap) + 1.0,
            yi,
            label,
            va="center",
            fontsize=8.5,
            color=COLORS["muted"],
        )
    ax.set_yticks(y)
    ax.set_yticklabels(y_labels)
    ax.text(
        0.0,
        1.02,
        "Dashed line = observed judge baseline | labels show mean gap and improvement vs baseline",
        transform=ax.transAxes,
        fontsize=9,
        color=COLORS["muted"],
    )
    ax.set_title("Mean Centered Elo Gap by Scheme", pad=14, loc="left")
    ax.set_xlabel("Mean |centered scheme Elo - centered human Elo|")
    ax.set_ylabel("Preference scheme")
    ax.set_xlim(0, max(plot_df["mean_abs_centered_gap"].max(), baseline_gap) * 1.24)
    ax.invert_yaxis()
    add_figure_caption(
        fig,
        left="Positive improvement means less Elo drift than the observed judge decision.",
        right=f"n = {int(plot_df['n_samples'].iloc[0]):,} pairwise comparisons",
    )
    save_figure(fig, path)


def save_scheme_metric_plot(weight_scheme_summary: pd.DataFrame, path: Path) -> None:
    plot_df = weight_scheme_summary.copy().sort_values("mean_abs_centered_gap")
    y = np.arange(len(plot_df))
    y_labels = wrap_labels(plot_df["scheme"], width=24)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.4, 5.8),
        sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )
    fig.subplots_adjust(top=0.8, bottom=0.16, wspace=0.18)
    add_figure_header(
        fig,
        title="BT Scheme Metrics",
        subtitle="Agreement and tie rate are shown separately to keep the comparison readable.",
    )

    metric_specs = [
        ("accuracy_vs_human", "Agreement with humans", COLORS["human"], "x"),
        ("scheme_tie_rate", "Tie rate", COLORS["judge"], "x"),
    ]
    for ax, (column, title, color, axis_name) in zip(axes, metric_specs):
        style_axes(ax, grid_axis="x")
        ax.scatter(
            plot_df[column],
            y,
            s=110,
            color=color,
            edgecolor="#ffffff",
            linewidth=1.0,
            zorder=3,
        )
        ax.hlines(y, 0, plot_df[column], color=COLORS["band"], linewidth=4.0, zorder=1)
        ax.set_title(title, pad=14, loc="left")
        ax.set_xlabel(title)
        format_percent_axis(ax, axis=axis_name)
        ax.set_xlim(0, min(1.0, float(plot_df[column].max()) + 0.08))
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(y_labels)
    axes[0].set_ylabel("Preference scheme")
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    axes[0].invert_yaxis()

    add_figure_caption(
        fig,
        left="Same schemes as the Elo-gap view, but split out so the metrics are easier to compare.",
        right=f"{len(plot_df)} scheme variants",
    )
    save_figure(fig, path)


def save_elo_gap_render_plot(weight_scheme_summary: pd.DataFrame, path: Path) -> None:
    plot_df = weight_scheme_summary.copy().sort_values("mean_abs_centered_gap")
    plot_df["scheme_color"] = plot_df["scheme"].map(scheme_color)
    baseline_row = plot_df.loc[plot_df["scheme"] == "Observed judge decision"]
    baseline_gap = (
        float(baseline_row["mean_abs_centered_gap"].iloc[0])
        if not baseline_row.empty
        else float(plot_df["mean_abs_centered_gap"].max())
    )
    plot_df["gap_improvement_vs_observed"] = baseline_gap - plot_df["mean_abs_centered_gap"]

    fig, ax = plt.subplots(figsize=(12.8, 5.8))
    fig.subplots_adjust(top=0.8, bottom=0.16)
    add_figure_header(
        fig,
        title="Centered Elo Gap Render",
        subtitle=(
            "Each row shows the typical and worst-case Elo drift for one decision rule after "
            "reconstructing preferences from the same rubric scores."
        ),
    )

    y = np.arange(len(plot_df))
    style_axes(ax, grid_axis="x")
    ax.axvline(baseline_gap, linestyle="--", color=COLORS["muted"], linewidth=1.3, zorder=0)
    for yi, (_, row) in enumerate(plot_df.iterrows()):
        ax.plot(
            [row["mean_abs_centered_gap"], row["max_abs_centered_gap"]],
            [yi, yi],
            color=COLORS["band"],
            linewidth=7.0,
            solid_capstyle="round",
            zorder=1,
        )
        ax.scatter(
            row["mean_abs_centered_gap"],
            yi,
            s=140,
            color=row["scheme_color"],
            edgecolor="#ffffff",
            linewidth=1.0,
            zorder=3,
        )
        ax.scatter(
            row["max_abs_centered_gap"],
            yi,
            s=86,
            marker="D",
            color=row["scheme_color"],
            edgecolor="#ffffff",
            linewidth=0.9,
            alpha=0.9,
            zorder=3,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(wrap_labels(plot_df["scheme"], width=24))
    ax.set_xlabel("Centered Elo gap relative to human ranking")
    ax.set_ylabel("Preference scheme")
    ax.set_title("Mean vs Worst-Case Elo Gap", pad=14, loc="left")
    ax.text(
        0.0,
        1.02,
        "Circle = mean absolute centered gap | diamond = largest absolute model-level gap",
        transform=ax.transAxes,
        fontsize=9,
        color=COLORS["muted"],
    )
    ax.set_xlim(0, plot_df["max_abs_centered_gap"].max() * 1.28)
    ax.invert_yaxis()
    add_figure_caption(
        fig,
        left="The dashed line marks the observed judge baseline.",
        right=f"n = {int(plot_df['n_samples'].iloc[0]):,} pairwise comparisons",
    )
    save_figure(fig, path)


def save_scheme_model_heatmap(
    weight_scheme_summary: pd.DataFrame,
    weight_scheme_model_gaps: pd.DataFrame,
    path: Path,
    *,
    top_models: int,
) -> None:
    model_order = (
        weight_scheme_model_gaps.groupby(["model", "short_model"], as_index=False)
        .agg(max_abs_centered_gap=("abs_centered_elo_gap", "max"))
        .sort_values("max_abs_centered_gap", ascending=False)
        .head(top_models)
    )
    scheme_order = weight_scheme_summary.sort_values("mean_abs_centered_gap")["scheme"].tolist()
    heatmap = (
        weight_scheme_model_gaps.pivot_table(
            index="scheme",
            columns="short_model",
            values="centered_elo_gap",
        ).reindex(index=scheme_order, columns=model_order["short_model"].tolist())
    )
    vmax = np.nanmax(np.abs(heatmap.to_numpy(dtype=float)))

    fig, ax = plt.subplots(figsize=(14.8, 6.1))
    fig.subplots_adjust(top=0.83, bottom=0.17)
    add_figure_header(
        fig,
        title="Model-Level Centered Elo Gap by Scheme",
        subtitle=(
            "Cells show centered scheme Elo minus centered human Elo for the models with the "
            "largest absolute drift. Negative values mean the scheme rates the model below humans."
        ),
    )
    im = ax.imshow(
        heatmap.fillna(0.0).to_numpy(),
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    style_axes(ax, grid_axis=None)
    add_heatmap_grid(ax, heatmap.shape)
    annotate_heatmap_values(ax, heatmap.to_numpy(dtype=float), fmt="{:+.0f}", limit=120)
    ax.set_title("Where Scheme-Level Elo Drift Concentrates", pad=14, loc="left")
    ax.set_xlabel("Model")
    ax.set_ylabel("Preference scheme")
    ax.set_xticks(np.arange(len(heatmap.columns)))
    ax.set_xticklabels(wrap_labels(heatmap.columns, width=16), rotation=35, ha="right")
    ax.set_yticks(np.arange(len(heatmap.index)))
    ax.set_yticklabels(wrap_labels(heatmap.index, width=24))
    cb = plt.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Centered scheme Elo - centered human Elo")
    add_figure_caption(fig, left="", right=f"Top {len(heatmap.columns)} models by absolute centered gap")
    save_figure(fig, path)


def save_probability_summary_plot(
    reconstruction_table: pd.DataFrame,
    path: Path,
    scheme_cols: list[str],
) -> None:
    rows: list[dict[str, float | str]] = []
    for scheme in scheme_cols:
        pref_col = f"{scheme}_pref"
        proba_col = f"{scheme}_proba_a"
        valid = reconstruction_table[pref_col].notna()
        if not valid.any():
            continue
        rows.append(
            {
                "scheme": scheme,
                "mean_proba_a": float(reconstruction_table.loc[valid, proba_col].mean()),
                "tie_rate": float((reconstruction_table.loc[valid, pref_col] == 0.5).mean()),
            }
        )

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        fig, ax = plt.subplots(figsize=(8.0, 3.2))
        style_axes(ax, grid_axis=None)
        ax.text(
            0.5,
            0.5,
            "No BT probability summaries available.",
            ha="center",
            va="center",
            fontsize=11,
            color=COLORS["muted"],
        )
        ax.set_axis_off()
        save_figure(fig, path)
        return

    labels = wrap_labels(plot_df["scheme"], width=20)
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.6))
    fig.subplots_adjust(top=0.82, bottom=0.16, wspace=0.24)
    add_figure_header(
        fig,
        title="BT Reconstruction Probability Summary",
        subtitle=(
            "Probability outputs come directly from the human-fitted Bradley-Terry model before "
            "the tie band converts them back into A / tie / B preferences."
        ),
    )
    axes[0].bar(
        labels,
        plot_df["mean_proba_a"],
        color=COLORS["human"],
        edgecolor="#ffffff",
        linewidth=1.0,
    )
    style_axes(axes[0], grid_axis="y")
    axes[0].axhline(0.5, linestyle="--", color=COLORS["muted"], linewidth=1.1)
    axes[0].set_title("Mean Predicted P(A wins)", pad=14, loc="left")
    axes[0].set_ylabel("Average probability")
    format_percent_axis(axes[0], axis="y")
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].set_ylim(0, min(1.0, plot_df["mean_proba_a"].max() + 0.12))

    axes[1].bar(
        labels,
        plot_df["tie_rate"],
        color=COLORS["judge"],
        edgecolor="#ffffff",
        linewidth=1.0,
    )
    style_axes(axes[1], grid_axis="y")
    axes[1].set_title("Tie Rate After BT Model Reconstruction", pad=14, loc="left")
    axes[1].set_ylabel("Tie rate")
    format_percent_axis(axes[1], axis="y")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].set_ylim(0, min(1.0, plot_df["tie_rate"].max() + 0.12))
    add_figure_caption(fig, left="", right=f"{len(plot_df)} scheme variants")
    save_figure(fig, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use human-fitted Bradley-Terry models to reconstruct preferences from "
            "saved rubric score deltas."
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
        "--group_col",
        default="meta_lang",
        help="Flattened metadata column for per-group BT fits.",
    )
    parser.add_argument(
        "--min_group_samples",
        type=int,
        default=100,
        help="Minimum samples per group for per-group BT fits.",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=0.01,
        help="L2 regularization for BT fitting.",
    )
    parser.add_argument(
        "--elo_k",
        type=float,
        default=32.0,
        help="Elo K-factor for scheme comparison.",
    )
    parser.add_argument(
        "--tie_band",
        type=float,
        default=0.05,
        help="Tie band around 0.5 when converting BT probabilities to A/tie/B labels.",
    )
    parser.add_argument(
        "--top_models",
        type=int,
        default=15,
        help="Maximum number of models to show in the scheme heatmap.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for Elo shuffling.",
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

    dimension_names = infer_score_dimensions(df)
    if not dimension_names:
        raise SystemExit("No rubric score dimensions found in the annotation artifact.")

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

    scheme_summary, scheme_model_gaps = summarize_preference_schemes(
        df,
        scheme_prefs,
        elo_k=args.elo_k,
        elo_seed=args.seed,
    )

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
            "human_calibrated_bt_global_pref": global_prefs,
            "human_calibrated_bt_global_proba_a": global_proba,
        }
    )
    if "Human-calibrated BT (language-specific)" in scheme_prefs:
        reconstruction_table[
            "human_calibrated_bt_language_specific_pref"
        ] = scheme_prefs["Human-calibrated BT (language-specific)"]
        reconstruction_table[
            "human_calibrated_bt_language_specific_proba_a"
        ] = scheme_proba["Human-calibrated BT (language-specific)"]

    scheme_summary_path = output_dir / "agreement_bt_human_model_scheme_summary.csv"
    scheme_model_path = output_dir / "agreement_bt_human_model_scheme_gaps.csv"
    reconstruction_path = output_dir / "agreement_bt_human_model_reconstruction.csv"
    summary_plot_path = plots_dir / "agreement_bt_human_model_scheme_summary.png"
    metric_plot_path = plots_dir / "agreement_bt_human_model_scheme_metrics.png"
    elo_gap_render_path = plots_dir / "agreement_bt_human_model_elo_gap_render.png"
    heatmap_path = plots_dir / "agreement_bt_human_model_scheme_heatmap.png"
    probability_plot_path = plots_dir / "agreement_bt_human_model_probability_summary.png"
    summary_json_path = output_dir / "agreement_bt_human_model_reconstruction_summary.json"

    scheme_summary.to_csv(scheme_summary_path, index=False)
    scheme_model_gaps.to_csv(scheme_model_path, index=False)
    reconstruction_table.to_csv(reconstruction_path, index=False)

    save_scheme_summary_plot(scheme_summary, summary_plot_path)
    save_scheme_metric_plot(scheme_summary, metric_plot_path)
    save_elo_gap_render_plot(scheme_summary, elo_gap_render_path)
    save_scheme_model_heatmap(
        scheme_summary,
        scheme_model_gaps,
        heatmap_path,
        top_models=args.top_models,
    )
    probability_scheme_cols = ["Human-calibrated BT (global)"]
    if "Human-calibrated BT (language-specific)" in scheme_prefs:
        probability_scheme_cols.append("Human-calibrated BT (language-specific)")
    prob_plot_df = reconstruction_table.rename(
        columns={
            "human_calibrated_bt_global_pref": "Human-calibrated BT (global)_pref",
            "human_calibrated_bt_global_proba_a": "Human-calibrated BT (global)_proba_a",
            "human_calibrated_bt_language_specific_pref": (
                "Human-calibrated BT (language-specific)_pref"
            ),
            "human_calibrated_bt_language_specific_proba_a": (
                "Human-calibrated BT (language-specific)_proba_a"
            ),
        }
    )
    save_probability_summary_plot(prob_plot_df, probability_plot_path, probability_scheme_cols)

    summary = {
        "artifact": str((output_dir / args.artifact).resolve()),
        "dataset": meta["dataset"],
        "judge_model": meta["judge_model"],
        "group_col": args.group_col,
        "plots_dir": str(plots_dir),
        "n_pair_rows": int(len(df)),
        "n_dimensions": int(len(dimension_names)),
        "n_groups_with_human_bt_model": int(len(human_bt_by_group)),
        "tie_band": float(args.tie_band),
        "regularization": float(args.regularization),
        "best_scheme_by_gap": str(scheme_summary.iloc[0]["scheme"]),
        "best_scheme_gap": float(scheme_summary.iloc[0]["mean_abs_centered_gap"]),
        "best_scheme_accuracy": float(scheme_summary.iloc[0]["accuracy_vs_human"]),
    }
    write_json(summary_json_path, summary)

    print(scheme_summary.to_string(index=False))
    print(f"Saved: {scheme_summary_path}")
    print(f"Saved: {scheme_model_path}")
    print(f"Saved: {reconstruction_path}")
    print(f"Saved: {summary_plot_path}")
    print(f"Saved: {metric_plot_path}")
    print(f"Saved: {elo_gap_render_path}")
    print(f"Saved: {heatmap_path}")
    print(f"Saved: {probability_plot_path}")
    print(f"Saved: {summary_json_path}")


if __name__ == "__main__":
    main()
