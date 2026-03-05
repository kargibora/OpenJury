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
    collect_group_bt_weights,
    ensure_plots_dir,
    fit_bt_weights,
    format_percent_axis,
    infer_score_dimensions,
    load_annotation_artifact,
    maybe_subsample_frame,
    reconstruct_preferences_by_group,
    reconstruct_preferences_from_weights,
    resolve_artifact_path,
    save_figure,
    style_axes,
    summarize_preference_schemes,
    wrap_labels,
    write_json,
)

apply_plot_theme()


def save_global_weight_plot(global_weight_view: pd.DataFrame, path: Path) -> None:
    plot_df = global_weight_view.copy().sort_values("human_norm_weight")
    y = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(11.6, 5.6))
    style_axes(ax, grid_axis="x")
    ax.hlines(
        y,
        plot_df["human_norm_weight"],
        plot_df["judge_norm_weight"],
        color=COLORS["band"],
        linewidth=3.4,
    )
    ax.scatter(
        plot_df["human_norm_weight"],
        y,
        s=120,
        color=COLORS["human"],
        edgecolor="#ffffff",
        linewidth=1.0,
        label="Human BT weights",
        zorder=3,
    )
    ax.scatter(
        plot_df["judge_norm_weight"],
        y,
        s=120,
        color=COLORS["judge"],
        edgecolor="#ffffff",
        linewidth=1.0,
        label="Judge BT weights",
        zorder=3,
    )
    ax.axvline(0.0, linestyle="--", color=COLORS["muted"], linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(wrap_labels(plot_df["dimension"], width=14))
    ax.set_xlabel("Signed L1-normalized BT weight")
    ax.set_ylabel("Rubric dimension")
    ax.set_title("Global BT Weights: Human vs Judge", pad=14, loc="left")
    ax.legend(loc="lower right")
    max_extent = float(
        np.nanmax(
            np.abs(
                plot_df[["human_norm_weight", "judge_norm_weight"]].to_numpy(dtype=float)
            )
        )
    )
    ax.set_xlim(-1.25 * max_extent, 1.45 * max_extent)
    save_figure(fig, path)


def save_human_weight_heatmap(
    language_order: list[str],
    dimension_names: list[str],
    human_bt_by_lang: pd.DataFrame,
    path: Path,
) -> None:
    human_weight_matrix = (
        human_bt_by_lang.pivot(index="group", columns="dimension", values="norm_weight")
        .reindex(index=language_order, columns=dimension_names)
    )
    n_rows = len(human_weight_matrix)
    human_matrix = human_weight_matrix.to_numpy(dtype=float)
    vmax_human = np.nanmax(np.abs(human_matrix))

    fig, ax = plt.subplots(figsize=(9, max(6.0, 0.42 * n_rows)))
    im = ax.imshow(
        human_weight_matrix.fillna(0.0).to_numpy(),
        aspect="auto",
        cmap="PuOr",
        vmin=-vmax_human,
        vmax=vmax_human,
    )
    style_axes(ax, grid_axis=None)
    add_heatmap_grid(ax, human_weight_matrix.shape)
    annotate_heatmap_values(ax, human_matrix, fmt="{:+.2f}", limit=110)
    ax.set_title("Normalized Human BT Weights by Language", pad=14, loc="left")
    ax.set_xticks(np.arange(len(dimension_names)))
    ax.set_xticklabels(wrap_labels(dimension_names, width=14), rotation=35, ha="right")
    ax.set_yticks(np.arange(len(human_weight_matrix.index)))
    ax.set_yticklabels(wrap_labels(human_weight_matrix.index, width=14))
    cb = plt.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Signed L1-normalized weight")
    save_figure(fig, path)


def save_weight_delta_heatmap(
    language_order: list[str],
    dimension_names: list[str],
    human_bt_by_lang: pd.DataFrame,
    judge_bt_by_lang: pd.DataFrame,
    path: Path,
) -> None:
    weight_delta = human_bt_by_lang.merge(
        judge_bt_by_lang,
        on=["group", "dimension"],
        suffixes=("_human", "_judge"),
    )
    weight_delta["norm_weight_delta"] = (
        weight_delta["norm_weight_judge"] - weight_delta["norm_weight_human"]
    )
    weight_delta_matrix = (
        weight_delta.pivot(index="group", columns="dimension", values="norm_weight_delta")
        .reindex(index=language_order, columns=dimension_names)
    )
    n_rows = len(weight_delta_matrix)
    vmax = np.nanmax(np.abs(weight_delta_matrix.to_numpy(dtype=float)))

    fig, ax = plt.subplots(figsize=(9, max(6.0, 0.42 * n_rows)))
    im = ax.imshow(
        weight_delta_matrix.fillna(0.0).to_numpy(),
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    style_axes(ax, grid_axis=None)
    add_heatmap_grid(ax, weight_delta_matrix.shape)
    annotate_heatmap_values(
        ax,
        weight_delta_matrix.to_numpy(dtype=float),
        fmt="{:+.2f}",
        limit=110,
    )
    ax.set_title("Judge Minus Human Normalized BT Weight", pad=14, loc="left")
    ax.set_xticks(np.arange(len(dimension_names)))
    ax.set_xticklabels(wrap_labels(dimension_names, width=14), rotation=35, ha="right")
    ax.set_yticks(np.arange(len(weight_delta_matrix.index)))
    ax.set_yticklabels(wrap_labels(weight_delta_matrix.index, width=14))
    cb = plt.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Judge norm weight - human norm weight")
    save_figure(fig, path)


def save_scheme_gap_bars(weight_scheme_summary: pd.DataFrame, path: Path) -> None:
    plot_df = weight_scheme_summary.copy().sort_values("mean_abs_centered_gap")
    y_labels = wrap_labels(plot_df["scheme"], width=24)
    fig, ax = plt.subplots(figsize=(9, 5.8))

    bar_colors = plt.cm.YlOrBr(np.linspace(0.4, 0.82, len(plot_df)))
    ax.barh(
        y_labels,
        plot_df["mean_abs_centered_gap"],
        color=bar_colors,
        edgecolor="#ffffff",
        linewidth=1.0,
    )
    style_axes(ax, grid_axis="x")
    ax.set_title(
        "Scheme-Level Elo Gap After Reconstructing Preferences",
        pad=14,
        loc="left",
    )
    ax.set_xlabel("Mean |centered scheme Elo - centered human Elo|")
    ax.set_ylabel("Preference scheme")
    ax.set_xlim(0, plot_df["mean_abs_centered_gap"].max() * 1.18)
    save_figure(fig, path)


def save_accuracy_fidelity_scatter(weight_scheme_summary: pd.DataFrame, path: Path) -> None:
    plot_df = weight_scheme_summary.copy().sort_values("mean_abs_centered_gap")
    fig, ax = plt.subplots(figsize=(9, 5.8))

    sc = ax.scatter(
        plot_df["accuracy_vs_human"],
        plot_df["mean_abs_centered_gap"],
        s=120 + 240 * plot_df["judge_match_rate"],
        c=plot_df["scheme_tie_rate"],
        cmap="viridis",
        edgecolor="#ffffff",
        linewidth=0.8,
    )
    style_axes(ax, grid_axis="both")
    ax.set_title("Accuracy vs Elo-Fidelity Tradeoff", pad=14, loc="left")
    ax.set_xlabel("Agreement with human labels")
    ax.set_ylabel("Mean |centered Elo gap|")
    format_percent_axis(ax, axis="x")
    cb = plt.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Scheme tie rate")
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

    fig, ax = plt.subplots(figsize=(14.4, 5.4))
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
    ax.set_title("Centered Elo Gap by Reconstruction Scheme and Model", pad=14, loc="left")
    ax.set_xlabel("Model")
    ax.set_ylabel("Preference scheme")
    ax.set_xticks(np.arange(len(heatmap.columns)))
    ax.set_xticklabels(wrap_labels(heatmap.columns, width=16), rotation=35, ha="right")
    ax.set_yticks(np.arange(len(heatmap.index)))
    ax.set_yticklabels(wrap_labels(heatmap.index, width=24))
    cb = plt.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Centered scheme Elo - centered human Elo")
    save_figure(fig, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit BT weights from agreement score deltas and compare uniform vs human-weighted "
            "preference reconstruction."
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

    uniform_weights = {dim: 1.0 for dim in dimension_names}
    human_bt_global = fit_bt_weights(
        df[df["human_pref"].notna()].copy(),
        "human_pref",
        dimension_names=dimension_names,
        regularization=args.regularization,
    )
    judge_bt_global = fit_bt_weights(
        df[df["judge_pref"].notna()].copy(),
        "judge_pref",
        dimension_names=dimension_names,
        regularization=args.regularization,
    )
    human_bt_by_group, human_bt_by_group_map = collect_group_bt_weights(
        df,
        args.group_col,
        "human_pref",
        min_samples=args.min_group_samples,
        dimension_names=dimension_names,
        regularization=args.regularization,
    )
    judge_bt_by_group, judge_bt_by_group_map = collect_group_bt_weights(
        df,
        args.group_col,
        "judge_pref",
        min_samples=args.min_group_samples,
        dimension_names=dimension_names,
        regularization=args.regularization,
    )

    global_weight_view = pd.DataFrame(
        {
            "dimension": dimension_names,
            "human_weight": [
                human_bt_global["weights"].get(dim, 0.0) for dim in dimension_names
            ],
            "human_norm_weight": [
                human_bt_global["normalized_weights"].get(dim, 0.0)
                for dim in dimension_names
            ],
            "judge_weight": [
                judge_bt_global["weights"].get(dim, 0.0) for dim in dimension_names
            ],
            "judge_norm_weight": [
                judge_bt_global["normalized_weights"].get(dim, 0.0)
                for dim in dimension_names
            ],
        }
    )
    global_weight_view["judge_minus_human_norm"] = (
        global_weight_view["judge_norm_weight"]
        - global_weight_view["human_norm_weight"]
    )

    weight_schemes = {
        "Observed judge decision": df["judge_pref"],
        "Equal-weight score rule": reconstruct_preferences_from_weights(
            df, uniform_weights
        ),
        "Human-calibrated score rule (global)": reconstruct_preferences_from_weights(
            df, human_bt_global["weights"]
        ),
    }
    if human_bt_by_group_map:
        weight_schemes[
            "Human-calibrated score rule (language-specific)"
        ] = reconstruct_preferences_by_group(
            df,
            args.group_col,
            human_bt_by_group_map,
            fallback_weights=human_bt_global["weights"],
        )

    weight_scheme_summary, weight_scheme_model_gaps = summarize_preference_schemes(
        df,
        weight_schemes,
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
                df[args.group_col] if args.group_col in df.columns else pd.Series(np.nan, index=df.index)
            ),
            "equal_weight_score_rule_pref": weight_schemes["Equal-weight score rule"],
            "human_calibrated_score_rule_global_pref": weight_schemes[
                "Human-calibrated score rule (global)"
            ],
        }
    )
    if "Human-calibrated score rule (language-specific)" in weight_schemes:
        reconstruction_table[
            "human_calibrated_score_rule_language_specific_pref"
        ] = weight_schemes["Human-calibrated score rule (language-specific)"]

    global_weight_path = output_dir / "agreement_bt_global_weights.csv"
    human_group_weight_path = output_dir / "agreement_bt_human_weights_by_group.csv"
    judge_group_weight_path = output_dir / "agreement_bt_judge_weights_by_group.csv"
    scheme_summary_path = output_dir / "agreement_weight_scheme_summary.csv"
    scheme_model_path = output_dir / "agreement_weight_scheme_model_gaps.csv"
    reconstruction_path = output_dir / "agreement_weighted_preference_reconstructions.csv"
    global_plot_path = plots_dir / "agreement_bt_global_weights.png"
    human_heatmap_path = plots_dir / "agreement_bt_human_weight_heatmap.png"
    delta_heatmap_path = plots_dir / "agreement_bt_weight_delta_heatmap.png"
    scheme_gap_path = plots_dir / "agreement_weight_scheme_gap_bars.png"
    scheme_scatter_path = plots_dir / "agreement_weight_scheme_accuracy_fidelity.png"
    scheme_heatmap_path = plots_dir / "agreement_weight_scheme_model_heatmap.png"
    summary_json_path = output_dir / "agreement_bt_reconstruction_summary.json"

    global_weight_view.to_csv(global_weight_path, index=False)
    human_bt_by_group.to_csv(human_group_weight_path, index=False)
    judge_bt_by_group.to_csv(judge_group_weight_path, index=False)
    weight_scheme_summary.to_csv(scheme_summary_path, index=False)
    weight_scheme_model_gaps.to_csv(scheme_model_path, index=False)
    reconstruction_table.to_csv(reconstruction_path, index=False)

    save_global_weight_plot(global_weight_view, global_plot_path)
    if not human_bt_by_group.empty and not judge_bt_by_group.empty:
        group_order = (
            df[args.group_col].fillna("__missing__").value_counts().index.tolist()
            if args.group_col in df.columns
            else []
        )
        group_order = [group for group in group_order if group in set(human_bt_by_group["group"])]
        save_human_weight_heatmap(
            group_order,
            dimension_names,
            human_bt_by_group,
            human_heatmap_path,
        )
        save_weight_delta_heatmap(
            group_order,
            dimension_names,
            human_bt_by_group,
            judge_bt_by_group,
            delta_heatmap_path,
        )
    save_scheme_gap_bars(weight_scheme_summary, scheme_gap_path)
    save_accuracy_fidelity_scatter(weight_scheme_summary, scheme_scatter_path)
    save_scheme_model_heatmap(
        weight_scheme_summary,
        weight_scheme_model_gaps,
        scheme_heatmap_path,
        top_models=args.top_models,
    )

    summary = {
        "artifact": str(resolve_artifact_path(output_dir, args.artifact).resolve()),
        "dataset": meta["dataset"],
        "judge_model": meta["judge_model"],
        "group_col": args.group_col,
        "plots_dir": str(plots_dir),
        "n_pair_rows": int(len(df)),
        "n_dimensions": int(len(dimension_names)),
        "n_groups_with_bt_fit": int(len(human_bt_by_group["group"].unique())) if not human_bt_by_group.empty else 0,
        "regularization": float(args.regularization),
        "best_scheme_by_gap": str(weight_scheme_summary.iloc[0]["scheme"]),
        "best_scheme_gap": float(weight_scheme_summary.iloc[0]["mean_abs_centered_gap"]),
        "best_scheme_accuracy": float(weight_scheme_summary.iloc[0]["accuracy_vs_human"]),
    }
    write_json(summary_json_path, summary)

    print(global_weight_view.to_string(index=False))
    print(weight_scheme_summary.to_string(index=False))
    print(f"Saved: {global_weight_path}")
    print(f"Saved: {human_group_weight_path}")
    print(f"Saved: {judge_group_weight_path}")
    print(f"Saved: {scheme_summary_path}")
    print(f"Saved: {scheme_model_path}")
    print(f"Saved: {reconstruction_path}")
    print(f"Saved: {global_plot_path}")
    if not human_bt_by_group.empty and not judge_bt_by_group.empty:
        print(f"Saved: {human_heatmap_path}")
        print(f"Saved: {delta_heatmap_path}")
    print(f"Saved: {scheme_gap_path}")
    print(f"Saved: {scheme_scatter_path}")
    print(f"Saved: {scheme_heatmap_path}")
    print(f"Saved: {summary_json_path}")


if __name__ == "__main__":
    main()
