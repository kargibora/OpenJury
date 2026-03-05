"""Sliced agreement analysis: Cohen's κ, accuracy, Elo gap broken down by
language, model performance tier, and model family.

Outputs
-------
- Per-language agreement table (CSV)
- Per-model-tier agreement table (CSV)
- Per-model-family bias table (CSV)
- Plots: language bars, tier bars, family scatter, heatmap, dashboard
- JSON summary
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("OPENJURY_LOG_LEVEL", "ERROR")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from plotting import (
    COLORS,
    ELO_BASE,
    apply_minimal_theme as _apply_theme,
    fit_bt_elo,
    load_frame,
    model_universe,
    pref_to_label,
    resolve_artifact_path,
    shorten,
    valid_decisive,
)
from openjury.analysis.common import compute_cohen_kappa

TIER_PALETTE = [COLORS["tier_top"], COLORS["tier_mid"], COLORS["tier_low"]]


def infer_model_family(model: str) -> str:
    """Heuristic: extract family prefix from model name."""
    s = shorten(model).lower()
    families = [
        (r"^gpt-4o", "GPT-4o"),
        (r"^gpt-4-turbo", "GPT-4-Turbo"),
        (r"^gpt-4", "GPT-4"),
        (r"^gpt-3\.5", "GPT-3.5"),
        (r"^chatgpt-4o", "GPT-4o"),
        (r"^claude-3-5", "Claude-3.5"),
        (r"^claude-3-opus", "Claude-3-Opus"),
        (r"^claude-3-sonnet", "Claude-3-Sonnet"),
        (r"^claude-3-haiku", "Claude-3-Haiku"),
        (r"^gemini-1\.5-pro", "Gemini-1.5-Pro"),
        (r"^gemini-1\.5-flash", "Gemini-1.5-Flash"),
        (r"^gemma-2-27b", "Gemma-2-27B"),
        (r"^gemma-2-9b", "Gemma-2-9B"),
        (r"^gemma-2-2b", "Gemma-2-2B"),
        (r"^gemma-1", "Gemma-1"),
        (r"^llama-3\.1-405b", "Llama-3.1-405B"),
        (r"^llama-3\.1-70b", "Llama-3.1-70B"),
        (r"^llama-3\.1-8b", "Llama-3.1-8B"),
        (r"^llama-3-70b", "Llama-3-70B"),
        (r"^llama-3-8b", "Llama-3-8B"),
        (r"^mistral-large", "Mistral-Large"),
        (r"^mixtral-8x22b", "Mixtral-8x22B"),
        (r"^mixtral-8x7b", "Mixtral-8x7B"),
        (r"^phi-3-mini", "Phi-3-Mini"),
        (r"^phi-3-small", "Phi-3-Small"),
        (r"^phi-3-medium", "Phi-3-Medium"),
        (r"^deepseek-coder", "DeepSeek-Coder"),
        (r"^deepseek-v2", "DeepSeek-V2"),
        (r"^yi-large", "Yi-Large"),
        (r"^yi-1\.5", "Yi-1.5"),
        (r"^qwen2", "Qwen2"),
        (r"^reka-core", "Reka-Core"),
        (r"^reka-flash", "Reka-Flash"),
        (r"^command-r-plus", "Command-R+"),
        (r"^command-r", "Command-R"),
        (r"^athene", "Athene"),
        (r"^nemotron", "Nemotron"),
        (r"^glm", "GLM"),
        (r"^codestral", "Codestral"),
        (r"^eureka", "Eureka"),
        (r"^snowflake", "Snowflake"),
        (r"^dbrx", "DBRX"),
    ]
    for pattern, family in families:
        if re.search(pattern, s):
            return family
    return shorten(model)


# ── slice-level agreement computation ──────────────────────────────────────
def compute_slice_agreement(
    frame: pd.DataFrame,
    slice_name: str = "all",
) -> dict:
    """Compute agreement metrics for a dataframe slice."""
    valid = frame[frame["human_pref"].notna() & frame["judge_pref"].notna()].copy()
    h_labels = [pref_to_label(p) for p in valid["human_pref"]]
    j_labels = [pref_to_label(p) for p in valid["judge_pref"]]
    n = len(h_labels)

    if n < 2:
        return {"slice": slice_name, "n_samples": n}

    # 3-class: A/tie/B
    agree = sum(1 for h, j in zip(h_labels, j_labels) if h == j)
    acc = agree / n

    try:
        kappa = compute_cohen_kappa(h_labels, j_labels)
    except Exception:
        kappa = float("nan")

    # Decisive only
    dec_h = [h for h, j in zip(h_labels, j_labels) if h != "tie" and j != "tie"]
    dec_j = [j for h, j in zip(h_labels, j_labels) if h != "tie" and j != "tie"]
    n_decisive = len(dec_h)
    if n_decisive >= 2:
        try:
            kappa_decisive = compute_cohen_kappa(dec_h, dec_j)
        except Exception:
            kappa_decisive = float("nan")
        acc_decisive = sum(1 for h, j in zip(dec_h, dec_j) if h == j) / n_decisive
    else:
        kappa_decisive = float("nan")
        acc_decisive = float("nan")

    # Preference correlation
    h_prefs = valid["human_pref"].to_numpy(dtype=float)
    j_prefs = valid["judge_pref"].to_numpy(dtype=float)
    try:
        pearson_r = float(np.corrcoef(h_prefs, j_prefs)[0, 1])
    except Exception:
        pearson_r = float("nan")
    mae = float(np.mean(np.abs(h_prefs - j_prefs)))

    # Label distributions
    h_dist = {"A": h_labels.count("A"), "tie": h_labels.count("tie"), "B": h_labels.count("B")}
    j_dist = {"A": j_labels.count("A"), "tie": j_labels.count("tie"), "B": j_labels.count("B")}
    human_tie_rate = h_dist["tie"] / n if n else 0
    judge_tie_rate = j_dist["tie"] / n if n else 0

    return {
        "slice": slice_name,
        "n_samples": n,
        "n_decisive": n_decisive,
        "accuracy_3class": round(acc, 4),
        "accuracy_decisive": round(acc_decisive, 4) if np.isfinite(acc_decisive) else None,
        "cohens_kappa": round(kappa, 4) if np.isfinite(kappa) else None,
        "cohens_kappa_decisive": round(kappa_decisive, 4) if np.isfinite(kappa_decisive) else None,
        "pearson_r": round(pearson_r, 4) if np.isfinite(pearson_r) else None,
        "mae": round(mae, 4) if np.isfinite(mae) else None,
        "human_tie_rate": round(human_tie_rate, 4),
        "judge_tie_rate": round(judge_tie_rate, 4),
    }


def compute_slice_table(
    frame: pd.DataFrame,
    group_col: str,
    *,
    min_samples: int = 20,
) -> pd.DataFrame:
    """Compute agreement for each unique value of group_col."""
    rows = []
    for val, grp in frame.groupby(group_col):
        if len(grp) < min_samples:
            continue
        row = compute_slice_agreement(grp, slice_name=str(val))
        rows.append(row)
    # Add "all" row
    rows.append(compute_slice_agreement(frame, slice_name="ALL"))
    return pd.DataFrame(rows).sort_values("n_samples", ascending=False).reset_index(drop=True)


# ── model-level bias table ─────────────────────────────────────────────────
def compute_model_bias_table(
    frame: pd.DataFrame,
    regularization: float = 0.01,
) -> pd.DataFrame:
    """Per-model: human Elo, judge Elo, residual, family, n_appearances."""
    models = model_universe(frame)
    h_elo = fit_bt_elo(frame, "human_pref", models, regularization)
    j_elo = fit_bt_elo(frame, "judge_pref", models, regularization)

    appearances = pd.concat([
        frame["model_a"].rename("model"),
        frame["model_b"].rename("model"),
    ]).value_counts()

    rows = []
    for m in models:
        he = h_elo.get(m, float("nan"))
        je = j_elo.get(m, float("nan"))
        rows.append({
            "model": m,
            "short_model": shorten(m),
            "family": infer_model_family(m),
            "n_appearances": int(appearances.get(m, 0)),
            "human_elo": round(he, 1),
            "judge_elo": round(je, 1),
            "residual": round(he - je, 1),
            "abs_residual": round(abs(he - je), 1),
        })
    return pd.DataFrame(rows).sort_values("judge_elo", ascending=False).reset_index(drop=True)


def compute_family_bias_table(model_table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate model-level bias by family."""
    rows = []
    for family, grp in model_table.groupby("family"):
        rows.append({
            "family": family,
            "n_models": len(grp),
            "n_total_appearances": int(grp["n_appearances"].sum()),
            "mean_human_elo": round(grp["human_elo"].mean(), 1),
            "mean_judge_elo": round(grp["judge_elo"].mean(), 1),
            "mean_residual": round(grp["residual"].mean(), 1),
            "mean_abs_residual": round(grp["abs_residual"].mean(), 1),
            "models": ", ".join(grp["short_model"].tolist()),
        })
    return pd.DataFrame(rows).sort_values("mean_abs_residual", ascending=False).reset_index(drop=True)


# ── tier assignment ─────────────────────────────────────────────────────────
def assign_model_tier(frame: pd.DataFrame, model_table: pd.DataFrame) -> pd.DataFrame:
    """Add a 'tier' column based on average human Elo of the two models."""
    elo_map = dict(zip(model_table["model"], model_table["human_elo"]))
    avg_elo = (frame["model_a"].map(elo_map).fillna(ELO_BASE) +
               frame["model_b"].map(elo_map).fillna(ELO_BASE)) / 2

    # Terciles
    q33 = avg_elo.quantile(0.33)
    q66 = avg_elo.quantile(0.66)
    tier = pd.cut(avg_elo, bins=[-np.inf, q33, q66, np.inf],
                  labels=["Low", "Mid", "Top"])
    out = frame.copy()
    out["pair_tier"] = tier
    out["pair_avg_elo"] = avg_elo.round(1)
    return out


# ── plotting ───────────────────────────────────────────────────────────────
def save_language_kappa_plot(lang_table: pd.DataFrame, path: Path):
    """Horizontal bar chart: κ by language."""
    plot_df = lang_table[lang_table["slice"] != "ALL"].copy()
    plot_df = plot_df.sort_values("n_samples", ascending=True).tail(25)

    fig, ax = plt.subplots(figsize=(8, max(5, 0.3 * len(plot_df))))
    fig.patch.set_facecolor("white")
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["cohens_kappa"].fillna(0), color=COLORS["human"], alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["slice"], fontsize=8)
    _apply_theme(ax, title="Cohen's κ by Language", xlabel="κ")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_language_accuracy_plot(lang_table: pd.DataFrame, path: Path):
    """Horizontal bar chart: accuracy by language."""
    plot_df = lang_table[lang_table["slice"] != "ALL"].copy()
    plot_df = plot_df.sort_values("n_samples", ascending=True).tail(25)

    fig, ax = plt.subplots(figsize=(8, max(5, 0.3 * len(plot_df))))
    fig.patch.set_facecolor("white")
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["accuracy_3class"].fillna(0), color=COLORS["judge"], alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["slice"], fontsize=8)
    _apply_theme(ax, title="3-Class Accuracy by Language", xlabel="Accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_tier_kappa_plot(tier_table: pd.DataFrame, path: Path):
    """Grouped bar chart: κ by model-pair tier."""
    plot_df = tier_table[tier_table["slice"] != "ALL"].copy()
    tiers = plot_df["slice"].tolist()

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")
    x = np.arange(len(tiers))
    w = 0.35

    kappa_vals = plot_df["cohens_kappa"].fillna(0).tolist()
    kappa_dec = plot_df["cohens_kappa_decisive"].fillna(0).tolist()
    ax.bar(x - w/2, kappa_vals, w, color=COLORS["human"], alpha=0.8, label="κ (3-class)")
    ax.bar(x + w/2, kappa_dec, w, color=COLORS["judge"], alpha=0.8, label="κ (decisive)")
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.legend(fontsize=8, framealpha=0.9)
    _apply_theme(ax, title="Cohen's κ by Model-Pair Tier", ylabel="κ")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_tier_accuracy_plot(tier_table: pd.DataFrame, path: Path):
    """Grouped bar chart: accuracy by model-pair tier."""
    plot_df = tier_table[tier_table["slice"] != "ALL"].copy()
    tiers = plot_df["slice"].tolist()

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")
    x = np.arange(len(tiers))
    w = 0.35

    acc_vals = plot_df["accuracy_3class"].fillna(0).tolist()
    acc_dec = plot_df["accuracy_decisive"].fillna(0).tolist()
    ax.bar(x - w/2, acc_vals, w, color=COLORS["human"], alpha=0.8, label="Acc (3-class)")
    ax.bar(x + w/2, acc_dec, w, color=COLORS["judge"], alpha=0.8, label="Acc (decisive)")
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.legend(fontsize=8, framealpha=0.9)
    _apply_theme(ax, title="Accuracy by Model-Pair Tier", ylabel="Accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_family_bias_plot(family_table: pd.DataFrame, path: Path):
    """Horizontal bar chart: mean Elo residual per model family."""
    plot_df = family_table.sort_values("mean_residual").copy()

    fig, ax = plt.subplots(figsize=(9, max(5, 0.35 * len(plot_df))))
    fig.patch.set_facecolor("white")
    y = np.arange(len(plot_df))
    colors = [COLORS["good"] if r > 0 else COLORS["bad"] for r in plot_df["mean_residual"]]

    ax.barh(y, plot_df["mean_residual"], color=colors, alpha=0.8)
    ax.axvline(0, color=COLORS["text"], linewidth=0.8, alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["family"], fontsize=8)

    # Annotate sample counts
    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(
            row["mean_residual"] + (3 if row["mean_residual"] >= 0 else -3), i,
            f"n={row['n_total_appearances']}", fontsize=7, color=COLORS["muted"],
            va="center", ha="left" if row["mean_residual"] >= 0 else "right",
        )

    _apply_theme(ax, title="Judge Bias by Model Family  (+ = judge underrates)",
                 xlabel="Mean Elo Residual (Human − Judge)")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_model_scatter_plot(model_table: pd.DataFrame, path: Path):
    """Scatter: judge Elo vs human Elo, coloured by family."""
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("white")

    families = model_table["family"].unique()
    cmap = plt.cm.get_cmap("tab20", len(families))
    family_colors = {f: cmap(i) for i, f in enumerate(sorted(families))}

    for _, row in model_table.iterrows():
        ax.scatter(row["judge_elo"], row["human_elo"], s=30,
                   c=[family_colors[row["family"]]], edgecolor="white", linewidth=0.4,
                   alpha=0.85, zorder=2)

    lo = min(model_table["judge_elo"].min(), model_table["human_elo"].min()) - 30
    hi = max(model_table["judge_elo"].max(), model_table["human_elo"].max()) + 30
    ax.plot([lo, hi], [lo, hi], "--", color=COLORS["muted"], linewidth=0.8, alpha=0.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    r = model_table[["judge_elo", "human_elo"]].corr().iloc[0, 1]
    rho = model_table[["judge_elo", "human_elo"]].corr(method="spearman").iloc[0, 1]
    _apply_theme(ax, title=f"Judge vs Human Elo by Family  (r={r:.3f}, ρ={rho:.3f})",
                 xlabel="Judge Elo", ylabel="Human Elo")

    # Top outliers
    top = model_table.nlargest(5, "abs_residual")
    for _, row in top.iterrows():
        ax.annotate(row["short_model"], (row["judge_elo"], row["human_elo"]),
                    fontsize=7, color=COLORS["muted"], xytext=(5, 5),
                    textcoords="offset points")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Sliced agreement analysis.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--artifact", default="agreement_annotations.json")
    p.add_argument("--min_samples_per_slice", type=int, default=20)
    p.add_argument("--regularization", type=float, default=0.01)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    min_s = args.min_samples_per_slice

    artifact_path = resolve_artifact_path(output_dir, args.artifact)
    print(f"[start] loading {artifact_path}", flush=True)
    df = load_frame(artifact_path)
    print(f"[start] {len(df)} samples", flush=True)

    # ── Global agreement ──
    global_agreement = compute_slice_agreement(df, "ALL")
    print(f"[global] κ={global_agreement.get('cohens_kappa')}, "
          f"acc={global_agreement.get('accuracy_3class')}", flush=True)

    # ── Language slices ──
    lang_col = "meta_lang" if "meta_lang" in df.columns else None
    if lang_col:
        lang_table = compute_slice_table(df, lang_col, min_samples=min_s)
        print(f"[language] {len(lang_table)-1} languages with ≥{min_s} samples", flush=True)
    else:
        lang_table = pd.DataFrame([global_agreement])
        print("[language] no language metadata found", flush=True)

    # ── Model-level bias ──
    model_table = compute_model_bias_table(df, args.regularization)
    family_table = compute_family_bias_table(model_table)
    print(f"[model] {len(model_table)} models, {len(family_table)} families", flush=True)

    # ── Tier slices ──
    df_tier = assign_model_tier(df, model_table)
    tier_table = compute_slice_table(df_tier, "pair_tier", min_samples=min_s)
    print(f"[tier] {len(tier_table)-1} tiers", flush=True)

    # ── Save CSVs ──
    lang_path = output_dir / "agreement_sliced_by_language.csv"
    tier_path = output_dir / "agreement_sliced_by_tier.csv"
    model_path = output_dir / "agreement_model_bias.csv"
    family_path = output_dir / "agreement_family_bias.csv"

    lang_table.to_csv(lang_path, index=False)
    tier_table.to_csv(tier_path, index=False)
    model_table.to_csv(model_path, index=False)
    family_table.to_csv(family_path, index=False)

    # ── Save plots ──
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    if lang_col:
        save_language_kappa_plot(lang_table, plots_dir / "sliced_language_kappa.png")
        save_language_accuracy_plot(lang_table, plots_dir / "sliced_language_accuracy.png")
    save_tier_kappa_plot(tier_table, plots_dir / "sliced_tier_kappa.png")
    save_tier_accuracy_plot(tier_table, plots_dir / "sliced_tier_accuracy.png")
    save_family_bias_plot(family_table, plots_dir / "sliced_family_bias.png")
    save_model_scatter_plot(model_table, plots_dir / "sliced_model_scatter.png")
    print("[plots] all plots saved", flush=True)

    # ── JSON summary ──
    r_pearson = model_table[["judge_elo", "human_elo"]].corr().iloc[0, 1]
    rho_spearman = model_table[["judge_elo", "human_elo"]].corr(method="spearman").iloc[0, 1]

    summary = {
        "n_samples": int(len(df)),
        "n_models": int(len(model_table)),
        "n_families": int(len(family_table)),
        "global_cohens_kappa": global_agreement.get("cohens_kappa"),
        "global_kappa_decisive": global_agreement.get("cohens_kappa_decisive"),
        "global_accuracy_3class": global_agreement.get("accuracy_3class"),
        "global_accuracy_decisive": global_agreement.get("accuracy_decisive"),
        "global_pearson_r": global_agreement.get("pearson_r"),
        "global_mae": global_agreement.get("mae"),
        "global_human_tie_rate": global_agreement.get("human_tie_rate"),
        "global_judge_tie_rate": global_agreement.get("judge_tie_rate"),
        "elo_pearson_r": round(float(r_pearson), 4),
        "elo_spearman_rho": round(float(rho_spearman), 4),
        "elo_mean_abs_residual": round(float(model_table["abs_residual"].mean()), 1),
        "elo_median_abs_residual": round(float(model_table["abs_residual"].median()), 1),
        "top_overrated_family": family_table.iloc[0]["family"] if len(family_table) else None,
        "top_overrated_residual": float(family_table.iloc[0]["mean_residual"]) if len(family_table) else None,
    }

    # Add per-tier summary
    for _, row in tier_table.iterrows():
        tier = str(row["slice"]).lower().replace(" ", "_")
        summary[f"tier_{tier}_kappa"] = row.get("cohens_kappa")
        summary[f"tier_{tier}_accuracy"] = row.get("accuracy_3class")
        summary[f"tier_{tier}_n"] = int(row.get("n_samples", 0))

    # Add top 5 languages by sample count
    if lang_col:
        top_langs = lang_table[lang_table["slice"] != "ALL"].head(5)
        for _, row in top_langs.iterrows():
            lang = str(row["slice"]).lower().replace(" ", "_")
            summary[f"lang_{lang}_kappa"] = row.get("cohens_kappa")
            summary[f"lang_{lang}_accuracy"] = row.get("accuracy_3class")
            summary[f"lang_{lang}_n"] = int(row.get("n_samples", 0))

    js_path = output_dir / "agreement_sliced_summary.json"
    js_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    for p in [lang_path, tier_path, model_path, family_path, js_path]:
        print(f"Saved: {p}")
    print(f"Saved: {plots_dir / 'sliced_*.png'}")


if __name__ == "__main__":
    main()
