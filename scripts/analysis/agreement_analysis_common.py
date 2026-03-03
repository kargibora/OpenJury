from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENJURY_LOG_LEVEL", "ERROR")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from openjury.analysis.common import compute_agreement_metrics
from openjury.arena.config import MatchResult
from openjury.arena.ratings import compute_elo
from openjury.bradley_terry import FeatureBradleyTerry
from openjury.common.pair_annotation import derive_preference_from_scores

COLORS = {
    "bg": "#f8fafc",
    "grid": "#d6dde8",
    "text": "#0f172a",
    "muted": "#64748b",
    "human": "#2563eb",
    "judge": "#ea580c",
    "gap_pos": "#b91c1c",
    "gap_neg": "#0369a1",
    "neutral": "#475569",
    "band": "#cbd5e1",
}


def shorten_model_name(model: str) -> str:
    return str(model).rsplit("/", 1)[-1]


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def apply_plot_theme() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["bg"],
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": COLORS["band"],
            "axes.labelcolor": COLORS["text"],
            "axes.titlecolor": COLORS["text"],
            "axes.titleweight": "semibold",
            "axes.titlesize": 13,
            "axes.labelsize": 10.5,
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.8,
            "grid.alpha": 0.8,
            "legend.frameon": False,
            "savefig.facecolor": COLORS["bg"],
            "savefig.bbox": "tight",
        }
    )


def add_figure_header(
    fig,
    *,
    title: str,
    subtitle: str | None = None,
    x: float = 0.02,
) -> None:
    fig.text(
        x,
        0.985,
        title,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="semibold",
        color=COLORS["text"],
    )
    if subtitle:
        fig.text(
            x,
            0.948,
            subtitle,
            ha="left",
            va="top",
            fontsize=10,
            color=COLORS["muted"],
        )


def add_figure_caption(
    fig,
    *,
    left: str,
    right: str | None = None,
    x: float = 0.02,
) -> None:
    if left:
        fig.text(
            x,
            0.02,
            left,
            ha="left",
            va="bottom",
            fontsize=9,
            color=COLORS["muted"],
        )
    if right:
        fig.text(
            0.98,
            0.02,
            right,
            ha="right",
            va="bottom",
            fontsize=9,
            color=COLORS["muted"],
        )


def style_axes(ax, *, grid_axis: str | None = "y") -> None:
    ax.set_facecolor("#ffffff")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["band"])
    ax.spines["bottom"].set_color(COLORS["band"])
    if grid_axis is not None:
        ax.grid(axis=grid_axis, linestyle="-", linewidth=0.8, alpha=0.75)


def wrap_label(label: object, *, width: int = 18) -> str:
    text = str(label)
    return "\n".join(
        textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    )


def wrap_labels(labels: list[object] | pd.Index, *, width: int = 18) -> list[str]:
    return [wrap_label(label, width=width) for label in labels]


def add_heatmap_grid(ax, shape: tuple[int, int]) -> None:
    n_rows, n_cols = shape
    ax.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_rows + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="#ffffff", linestyle="-", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)


def annotate_heatmap_values(
    ax,
    matrix: np.ndarray,
    *,
    fmt: str = "{:.0f}",
    limit: int = 36,
) -> None:
    if matrix.size == 0 or matrix.size > limit:
        return
    vmax = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 0.0
    for (row, col), value in np.ndenumerate(matrix):
        if not np.isfinite(value):
            continue
        color = "#ffffff" if vmax and abs(value) > 0.45 * vmax else COLORS["text"]
        ax.text(
            col,
            row,
            fmt.format(value),
            ha="center",
            va="center",
            fontsize=7.5,
            color=color,
        )


def format_percent_axis(ax, *, axis: str) -> None:
    formatter = PercentFormatter(xmax=1.0, decimals=0)
    if axis == "x":
        ax.xaxis.set_major_formatter(formatter)
    elif axis == "y":
        ax.yaxis.set_major_formatter(formatter)
    else:
        raise ValueError(f"Unsupported axis {axis!r}")


def save_figure(fig, path: Path, *, dpi: int = 220) -> None:
    fig.patch.set_facecolor(COLORS["bg"])
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def maybe_subsample_frame(
    frame: pd.DataFrame,
    *,
    max_samples: int | None,
    seed: int,
) -> pd.DataFrame:
    if max_samples is None or max_samples <= 0 or len(frame) <= max_samples:
        return frame.copy()
    return (
        frame.sample(n=max_samples, random_state=seed)
        .sort_index()
        .reset_index(drop=True)
    )


def load_annotation_artifact(
    output_dir: Path,
    artifact_name: str = "agreement_annotations.json",
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    artifact_path = output_dir / artifact_name
    with artifact_path.open(encoding="utf-8") as f:
        artifact = json.load(f)

    meta = artifact["metadata"]
    raw_df = pd.DataFrame(artifact["per_sample"])
    if "metadata" in raw_df.columns:
        meta_df = pd.json_normalize(raw_df["metadata"]).add_prefix("meta_")
        df = pd.concat([raw_df.drop(columns=["metadata"]), meta_df], axis=1)
    else:
        df = raw_df.copy()
    return artifact, meta, df


def valid_gap_frame(frame: pd.DataFrame) -> pd.DataFrame:
    mask = frame["human_pref"].notna() & frame["judge_pref"].notna()
    return frame.loc[mask].copy()


def model_universe(frame: pd.DataFrame) -> list[str]:
    return sorted(set(frame["model_a"]).union(set(frame["model_b"])))


def build_matches(frame: pd.DataFrame, preference_col: str) -> list[MatchResult]:
    matches: list[MatchResult] = []
    valid = frame.dropna(subset=[preference_col]).copy()

    for idx, row in valid.iterrows():
        matches.append(
            MatchResult(
                model_a=str(row["model_a"]),
                model_b=str(row["model_b"]),
                instruction_index=int(idx),
                scores_a=row.get("scores_a", {}) or {},
                scores_b=row.get("scores_b", {}) or {},
                preference=float(row[preference_col]),
                instruction=row.get("instruction", "") or "",
                instruction_id=row.get("instruction_id", "") or "",
                instruction_metadata={},
                completion_a="",
                completion_b="",
                raw_judge_output="",
                raw_judge_output_swapped=None,
            )
        )
    return matches


def compare_preference_sources(
    frame: pd.DataFrame,
    reference_pref_col: str,
    comparison_pref_col: str,
    *,
    reference_name: str = "human",
    comparison_name: str = "judge",
    models: list[str] | None = None,
    elo_k: float = 32.0,
    elo_seed: int = 42,
) -> pd.DataFrame:
    base = frame.dropna(subset=[reference_pref_col, comparison_pref_col]).copy()
    if len(base) == 0:
        return pd.DataFrame()
    if models is None:
        models = model_universe(base)

    ref_elo = compute_elo(
        models=models,
        matches=build_matches(base, reference_pref_col),
        k=elo_k,
        seed=elo_seed,
    )
    cmp_elo = compute_elo(
        models=models,
        matches=build_matches(base, comparison_pref_col),
        k=elo_k,
        seed=elo_seed,
    )
    appearances = pd.concat(
        [base["model_a"].rename("model"), base["model_b"].rename("model")]
    ).value_counts().rename("n_appearances")

    table = pd.DataFrame({"model": models})
    table[f"{reference_name}_elo"] = table["model"].map(ref_elo)
    table[f"{comparison_name}_elo"] = table["model"].map(cmp_elo)
    table[f"{reference_name}_elo_centered"] = (
        table[f"{reference_name}_elo"] - table[f"{reference_name}_elo"].mean()
    )
    table[f"{comparison_name}_elo_centered"] = (
        table[f"{comparison_name}_elo"] - table[f"{comparison_name}_elo"].mean()
    )
    table["short_model"] = table["model"].map(shorten_model_name)
    table["n_appearances"] = table["model"].map(appearances).fillna(0).astype(int)
    table["elo_gap"] = (
        table[f"{comparison_name}_elo"] - table[f"{reference_name}_elo"]
    )
    table["centered_elo_gap"] = (
        table[f"{comparison_name}_elo_centered"]
        - table[f"{reference_name}_elo_centered"]
    )
    table["abs_centered_elo_gap"] = table["centered_elo_gap"].abs()
    return table.sort_values("abs_centered_elo_gap", ascending=False).reset_index(
        drop=True
    )


def infer_score_dimensions(frame: pd.DataFrame) -> list[str]:
    for value in frame.get("scores_a", []):
        if isinstance(value, dict) and value:
            return sorted(value.keys())
    return []


def preference_to_label(pref: float | None) -> str:
    if pref is None or pd.isna(pref):
        return "parse_error"
    if pref < 0.5:
        return "A"
    if pref > 0.5:
        return "B"
    return "tie"


def _has_missing_scores(scores: dict[str, float] | None, dims: list[str]) -> bool:
    if not isinstance(scores, dict):
        return True
    for dim in dims:
        value = scores.get(dim, np.nan)
        if value is None or pd.isna(value):
            return True
    return False


def normalized_weight_dict(weights: dict[str, float]) -> dict[str, float]:
    denom = sum(abs(float(v)) for v in weights.values())
    if denom <= 0:
        return {dim: 0.0 for dim in weights}
    return {dim: float(v) / denom for dim, v in weights.items()}


def score_frame_pair(
    frame: pd.DataFrame,
    dimension_names: list[str],
    preference_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = frame.dropna(subset=[preference_col]).copy().reset_index(drop=True)
    scores_a = pd.DataFrame(
        [
            {dim: (row or {}).get(dim, np.nan) for dim in dimension_names}
            for row in base["scores_a"]
        ],
        columns=dimension_names,
    )
    scores_b = pd.DataFrame(
        [
            {dim: (row or {}).get(dim, np.nan) for dim in dimension_names}
            for row in base["scores_b"]
        ],
        columns=dimension_names,
    )
    return base, scores_a, scores_b


def score_matrices(
    frame: pd.DataFrame,
    dimension_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores_a = pd.DataFrame(
        [
            {dim: (row or {}).get(dim, np.nan) for dim in dimension_names}
            for row in frame["scores_a"]
        ],
        columns=dimension_names,
    )
    scores_b = pd.DataFrame(
        [
            {dim: (row or {}).get(dim, np.nan) for dim in dimension_names}
            for row in frame["scores_b"]
        ],
        columns=dimension_names,
    )
    return scores_a, scores_b


def fit_bt_weights(
    frame: pd.DataFrame,
    preference_col: str,
    *,
    dimension_names: list[str],
    regularization: float = 0.01,
) -> dict[str, object]:
    base, scores_a, scores_b = score_frame_pair(frame, dimension_names, preference_col)
    bt = FeatureBradleyTerry(
        dimension_names=dimension_names,
        regularization=regularization,
    )
    bt.fit(scores_a, scores_b, base[preference_col], verbose=False)

    prefs = base[preference_col].to_numpy(dtype=float)
    valid_mask = ~(
        scores_a.isna().any(axis=1).to_numpy()
        | scores_b.isna().any(axis=1).to_numpy()
        | np.isnan(prefs)
    )
    not_tie = np.abs(prefs - 0.5) > 0.05
    fit_mask = valid_mask & not_tie
    train_accuracy = np.nan
    if fit_mask.any():
        labels = (prefs[fit_mask] < 0.5).astype(int)
        preds = bt.predict(
            scores_a.iloc[fit_mask].reset_index(drop=True),
            scores_b.iloc[fit_mask].reset_index(drop=True),
        )
        train_accuracy = float((preds == labels).mean())

    weights = bt.weight_dict()
    return {
        "model": bt,
        "base": base,
        "scores_a": scores_a,
        "scores_b": scores_b,
        "weights": weights,
        "normalized_weights": normalized_weight_dict(weights),
        "training_accuracy": train_accuracy,
    }


def reconstruct_preferences_from_weights(
    frame: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    dims = list(weights.keys())
    prefs: list[float] = []
    for scores_a, scores_b in zip(frame["scores_a"], frame["scores_b"]):
        if _has_missing_scores(scores_a, dims) or _has_missing_scores(scores_b, dims):
            prefs.append(np.nan)
            continue
        prefs.append(
            derive_preference_from_scores(scores_a or {}, scores_b or {}, weights)
        )
    return pd.Series(prefs, index=frame.index, dtype=float)


def reconstruct_preferences_by_group(
    frame: pd.DataFrame,
    group_col: str,
    group_weight_map: dict[str, dict[str, float]],
    *,
    fallback_weights: dict[str, float],
) -> pd.Series:
    prefs: list[float] = []
    for _, row in frame.iterrows():
        group = row.get(group_col)
        if pd.isna(group):
            group = "__missing__"
        weights = group_weight_map.get(str(group), fallback_weights)
        dims = list(weights.keys())
        if _has_missing_scores(row.get("scores_a"), dims) or _has_missing_scores(
            row.get("scores_b"), dims
        ):
            prefs.append(np.nan)
            continue
        prefs.append(
            derive_preference_from_scores(
                row.get("scores_a") or {},
                row.get("scores_b") or {},
                weights,
            )
        )
    return pd.Series(prefs, index=frame.index, dtype=float)


def reconstruct_preferences_from_bt_model(
    frame: pd.DataFrame,
    bt: FeatureBradleyTerry,
    *,
    dimension_names: list[str],
    tie_band: float = 0.05,
) -> tuple[pd.Series, pd.Series]:
    scores_a, scores_b = score_matrices(frame, dimension_names)
    valid_mask = ~(
        scores_a.isna().any(axis=1).to_numpy() | scores_b.isna().any(axis=1).to_numpy()
    )

    prefs = pd.Series(np.nan, index=frame.index, dtype=float)
    proba_a = pd.Series(np.nan, index=frame.index, dtype=float)
    if valid_mask.any():
        proba = bt.predict_proba(
            scores_a.loc[valid_mask].reset_index(drop=True),
            scores_b.loc[valid_mask].reset_index(drop=True),
        )
        proba_a.loc[valid_mask] = proba
        prefs.loc[valid_mask] = np.where(
            proba >= 0.5 + tie_band,
            0.0,
            np.where(proba <= 0.5 - tie_band, 1.0, 0.5),
        )
    return prefs, proba_a


def reconstruct_preferences_from_bt_model_by_group(
    frame: pd.DataFrame,
    group_col: str,
    group_fit_map: dict[str, dict[str, object]],
    *,
    dimension_names: list[str],
    fallback_fit: dict[str, object],
    tie_band: float = 0.05,
) -> tuple[pd.Series, pd.Series]:
    prefs = pd.Series(np.nan, index=frame.index, dtype=float)
    proba_a = pd.Series(np.nan, index=frame.index, dtype=float)
    for group, group_idx in frame.groupby(frame[group_col].fillna("__missing__")).groups.items():
        fit = group_fit_map.get(str(group), fallback_fit)
        group_frame = frame.loc[group_idx].copy()
        group_prefs, group_proba = reconstruct_preferences_from_bt_model(
            group_frame,
            fit["model"],
            dimension_names=dimension_names,
            tie_band=tie_band,
        )
        prefs.loc[group_idx] = group_prefs
        proba_a.loc[group_idx] = group_proba
    return prefs, proba_a


def collect_group_bt_weights(
    frame: pd.DataFrame,
    group_col: str,
    preference_col: str,
    *,
    min_samples: int = 100,
    dimension_names: list[str],
    regularization: float = 0.01,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    if group_col not in frame.columns:
        return pd.DataFrame(), {}

    counts = frame[group_col].fillna("__missing__").value_counts()
    eligible = counts[counts >= min_samples].index.tolist()
    weight_rows: list[dict[str, object]] = []
    weight_map: dict[str, dict[str, float]] = {}

    for group in eligible:
        group_frame = frame[frame[group_col].fillna("__missing__") == group].copy()
        fit = fit_bt_weights(
            group_frame,
            preference_col,
            dimension_names=dimension_names,
            regularization=regularization,
        )
        weight_map[str(group)] = fit["weights"]
        normalized = fit["normalized_weights"]
        for dim in dimension_names:
            weight_rows.append(
                {
                    "group": str(group),
                    "dimension": dim,
                    "raw_weight": float(fit["weights"].get(dim, 0.0)),
                    "norm_weight": float(normalized.get(dim, 0.0)),
                    "n_samples": int(len(fit["base"])),
                    "training_accuracy": (
                        float(fit["training_accuracy"])
                        if pd.notna(fit["training_accuracy"])
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(weight_rows), weight_map


def collect_group_bt_fits(
    frame: pd.DataFrame,
    group_col: str,
    preference_col: str,
    *,
    min_samples: int = 100,
    dimension_names: list[str],
    regularization: float = 0.01,
) -> dict[str, dict[str, object]]:
    if group_col not in frame.columns:
        return {}

    counts = frame[group_col].fillna("__missing__").value_counts()
    eligible = counts[counts >= min_samples].index.tolist()
    fit_map: dict[str, dict[str, object]] = {}

    for group in eligible:
        group_frame = frame[frame[group_col].fillna("__missing__") == group].copy()
        fit_map[str(group)] = fit_bt_weights(
            group_frame,
            preference_col,
            dimension_names=dimension_names,
            regularization=regularization,
        )

    return fit_map


def summarize_preference_schemes(
    frame: pd.DataFrame,
    schemes: dict[str, pd.Series],
    *,
    elo_k: float = 32.0,
    elo_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    model_rows: list[pd.DataFrame] = []

    for scheme_name, prefs in schemes.items():
        scheme_frame = frame.copy()
        scheme_frame["scheme_pref"] = pd.Series(prefs, index=scheme_frame.index, dtype=float)
        scheme_frame["scheme_label"] = scheme_frame["scheme_pref"].map(preference_to_label)
        valid = scheme_frame.dropna(subset=["human_pref", "scheme_pref"]).copy()
        comparison = compare_preference_sources(
            valid,
            "human_pref",
            "scheme_pref",
            reference_name="human",
            comparison_name="scheme",
            elo_k=elo_k,
            elo_seed=elo_seed,
        )
        decisive_mask = (valid["human_label"] != "tie") & (
            valid["scheme_label"] != "tie"
        )
        decisive_accuracy = np.nan
        if decisive_mask.any():
            decisive_accuracy = float(
                (
                    valid.loc[decisive_mask, "scheme_label"]
                    == valid.loc[decisive_mask, "human_label"]
                ).mean()
            )

        spearman = np.nan
        if not comparison.empty:
            spearman = float(
                comparison["human_elo"].corr(
                    comparison["scheme_elo"], method="spearman"
                )
            )

        summary_rows.append(
            {
                "scheme": scheme_name,
                "n_samples": int(len(valid)),
                "accuracy_vs_human": float(
                    (valid["scheme_label"] == valid["human_label"]).mean()
                ),
                "decisive_accuracy_vs_human": decisive_accuracy,
                "judge_match_rate": float(
                    (valid["scheme_label"] == valid["judge_label"]).mean()
                ),
                "scheme_tie_rate": float((valid["scheme_label"] == "tie").mean()),
                "mean_abs_centered_gap": float(
                    comparison["abs_centered_elo_gap"].mean()
                ),
                "max_abs_centered_gap": float(
                    comparison["abs_centered_elo_gap"].max()
                ),
                "spearman_human_vs_scheme_elo": spearman,
            }
        )
        model_rows.append(comparison.assign(scheme=scheme_name))

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values("mean_abs_centered_gap")
        .reset_index(drop=True)
    )
    model_table = pd.concat(model_rows, ignore_index=True) if model_rows else pd.DataFrame()
    return summary, model_table


def collect_language_gap_tables(
    frame: pd.DataFrame,
    meta: dict[str, Any],
    *,
    language_col: str = "meta_lang",
    min_samples: int = 100,
    elo_k: float = 32.0,
    elo_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if language_col not in frame.columns:
        return pd.DataFrame(), pd.DataFrame()

    lang_rows: list[dict[str, object]] = []
    gap_tables: list[pd.DataFrame] = []
    counts = frame[language_col].fillna("__missing__").value_counts()
    eligible = counts[counts >= min_samples].index.tolist()

    for lang in eligible:
        lang_df = frame[frame[language_col].fillna("__missing__") == lang].copy()
        valid = valid_gap_frame(lang_df)
        if len(valid) < 2:
            continue

        metrics = compute_agreement_metrics(
            per_sample=lang_df.to_dict("records"),
            dataset=meta["dataset"],
            judge_model=meta["judge_model"],
            judge_mode=meta["judge_mode"],
            rubric=meta["rubric"],
            swap_debiasing=meta["swap_debiasing"],
            language=None if lang == "__missing__" else str(lang),
        )
        comparison = compare_preference_sources(
            lang_df,
            "human_pref",
            "judge_pref",
            reference_name="human",
            comparison_name="judge",
            elo_k=elo_k,
            elo_seed=elo_seed,
        )
        decisive_df = valid[
            (valid["human_label"] != "tie") & (valid["judge_label"] != "tie")
        ].copy()
        decisive_gap = np.nan
        if len(decisive_df) >= 2:
            decisive_gap = float(
                compare_preference_sources(
                    decisive_df,
                    "human_pref",
                    "judge_pref",
                    reference_name="human",
                    comparison_name="judge",
                    elo_k=elo_k,
                    elo_seed=elo_seed,
                )["abs_centered_elo_gap"].mean()
            )

        top_gap = comparison.iloc[0]
        lang_rows.append(
            {
                "lang": lang,
                "n_samples": int(len(lang_df)),
                "n_valid": int(metrics["n_valid"]),
                "accuracy": float(metrics["accuracy"]),
                "cohens_kappa": metrics["cohens_kappa"],
                "cohens_kappa_decisive": metrics["cohens_kappa_decisive"],
                "human_tie_rate": float((valid["human_label"] == "tie").mean()),
                "judge_tie_rate": float((valid["judge_label"] == "tie").mean()),
                "tie_rate_gap": float(
                    (valid["judge_label"] == "tie").mean()
                    - (valid["human_label"] == "tie").mean()
                ),
                "mean_abs_centered_gap": float(
                    comparison["abs_centered_elo_gap"].mean()
                ),
                "max_abs_centered_gap": float(
                    comparison["abs_centered_elo_gap"].max()
                ),
                "mean_abs_centered_gap_decisive": decisive_gap,
                "top_gap_model": str(top_gap["short_model"]),
                "top_gap_value": float(top_gap["centered_elo_gap"]),
            }
        )
        gap_tables.append(comparison.assign(lang=lang, n_valid=int(metrics["n_valid"])))

    summary = (
        pd.DataFrame(lang_rows)
        .sort_values("mean_abs_centered_gap", ascending=False)
        .reset_index(drop=True)
    )
    long_table = pd.concat(gap_tables, ignore_index=True) if gap_tables else pd.DataFrame()
    return summary, long_table


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ensure_plots_dir(output_dir: Path) -> Path:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir
