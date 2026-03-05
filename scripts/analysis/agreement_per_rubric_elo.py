"""Per-rubric Elo analysis.

For each of the 6 rubric dimensions, fit a separate BT-Elo ranking using
*only* that dimension's score gap to derive pairwise preferences.  Compare
each single-dimension ranking to the **human** ranking to reveal which rubric
dimension best predicts human preferences on its own.

IMPORTANT: the judge preference (``judge_pref``) in the data is itself the
uniform (equal-weight) average of the 6 rubric dimension scores.  Therefore
the ``judge_all`` baseline shown in plots is *not* independent of the per-
dimension scores—it is their aggregate.  The scientifically meaningful
comparison is always **single-dim Elo vs human Elo**, which is fully
independent.  ``judge_all`` is included only as a reference ceiling showing
what uniform aggregation achieves.

Outputs
-------
- per_rubric_elo.csv          — model × dimension Elo table
- per_rubric_ranking_quality.csv — Pearson r, Spearman ρ, MAE vs human per dim
- per_rubric_top_bottom_delta.csv — which dims shift which models most
- 4 plots + 1 dashboard
- JSON summary
"""
from __future__ import annotations

import argparse
import json
import math
import os
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
    DIM_PALETTE,
    apply_minimal_theme as _apply_theme,
    fit_bt_elo,
    infer_score_dimensions as infer_dimensions,
    load_frame,
    model_universe,
    resolve_artifact_path,
    shorten,
)


# ── single-dimension preference derivation ─────────────────────────────────
def derive_single_dim_pref(scores_a: dict | None, scores_b: dict | None, dim: str) -> float:
    """Compare a single rubric dimension's scores → 0.0 (A wins), 1.0 (B wins), 0.5 (tie)."""
    if not isinstance(scores_a, dict) or not isinstance(scores_b, dict):
        return np.nan
    sa = scores_a.get(dim)
    sb = scores_b.get(dim)
    if sa is None or sb is None:
        return np.nan
    try:
        sa, sb = float(sa), float(sb)
    except (TypeError, ValueError):
        return np.nan
    if math.isnan(sa) or math.isnan(sb):
        return np.nan
    if sa > sb:
        return 0.0
    if sb > sa:
        return 1.0
    return 0.5


def add_single_dim_prefs(df: pd.DataFrame, dims: list[str]) -> pd.DataFrame:
    """Add columns dim_pref_{d} for each rubric dimension."""
    out = df.copy()
    for d in dims:
        out[f"dim_pref_{d}"] = [
            derive_single_dim_pref(sa, sb, d)
            for sa, sb in zip(out["scores_a"], out["scores_b"])
        ]
    return out


# ── core analysis ──────────────────────────────────────────────────────────
def compute_per_rubric_elo(
    df: pd.DataFrame, dims: list[str], models: list[str], reg: float,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Fit BT-Elo per dimension and return a model × dim Elo table."""
    elo_maps: dict[str, dict[str, float]] = {}
    for d in dims:
        pref_col = f"dim_pref_{d}"
        n_decisive = int((np.abs(df[pref_col].astype(float) - 0.5) > 0.05).sum())
        print(f"  {d:15s}  decisive pairs: {n_decisive}", flush=True)
        elo_maps[d] = fit_bt_elo(df, pref_col, models, reg)

    # Also fit human and full-judge Elo
    elo_maps["human"] = fit_bt_elo(df, "human_pref", models, reg)
    elo_maps["judge_all"] = fit_bt_elo(df, "judge_pref", models, reg)

    rows = []
    for m in models:
        row = {"model": m, "short_model": shorten(m)}
        for source, emap in elo_maps.items():
            row[f"elo_{source}"] = round(emap.get(m, np.nan), 1)
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("elo_human", ascending=False).reset_index(drop=True)
    return table, elo_maps


def compute_ranking_quality(
    elo_table: pd.DataFrame, dims: list[str],
) -> pd.DataFrame:
    """For each dimension (+ judge_all), compute correlation with human Elo."""
    human = elo_table["elo_human"].to_numpy(dtype=float)
    rows = []
    sources = dims + ["judge_all"]
    for src in sources:
        col = f"elo_{src}"
        vals = elo_table[col].to_numpy(dtype=float)
        mask = np.isfinite(human) & np.isfinite(vals)
        if mask.sum() < 5:
            continue
        h, v = human[mask], vals[mask]

        pearson_r = float(np.corrcoef(h, v)[0, 1])
        spearman_rho = float(sp_stats.spearmanr(h, v).statistic)
        kendall_tau = float(sp_stats.kendalltau(h, v).statistic)
        mae = float(np.mean(np.abs(h - v)))
        median_ae = float(np.median(np.abs(h - v)))

        # Rank agreement: count pairwise concordance
        rows.append({
            "source": src,
            "pearson_r": round(pearson_r, 4),
            "spearman_rho": round(spearman_rho, 4),
            "kendall_tau": round(kendall_tau, 4),
            "mae": round(mae, 1),
            "median_ae": round(median_ae, 1),
            "n_models": int(mask.sum()),
        })

    return pd.DataFrame(rows).sort_values("spearman_rho", ascending=False).reset_index(drop=True)


def compute_per_model_delta(
    elo_table: pd.DataFrame, dims: list[str],
) -> pd.DataFrame:
    """For each model, compute how much each single-dim Elo deviates from human Elo.
    This reveals which dimensions shift which models most."""
    rows = []
    for _, row in elo_table.iterrows():
        h_elo = row["elo_human"]
        entry = {"model": row["model"], "short_model": row["short_model"], "elo_human": h_elo}
        for d in dims:
            d_elo = row[f"elo_{d}"]
            entry[f"delta_{d}"] = round(d_elo - h_elo, 1) if np.isfinite(d_elo) else None
        # Which dim is closest to human?
        deltas = {d: abs(row[f"elo_{d}"] - h_elo) for d in dims if np.isfinite(row[f"elo_{d}"])}
        entry["best_dim"] = min(deltas, key=deltas.get) if deltas else None
        entry["worst_dim"] = max(deltas, key=deltas.get) if deltas else None
        rows.append(entry)

    return pd.DataFrame(rows).sort_values("elo_human", ascending=False).reset_index(drop=True)


# ── plotting ───────────────────────────────────────────────────────────────
def save_ranking_quality_plot(rq: pd.DataFrame, dims: list[str], path: Path):
    """Grouped bar chart: Spearman ρ and Pearson r per source."""
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("white")

    sources = rq["source"].tolist()
    x = np.arange(len(sources))
    w = 0.3

    rho_vals = rq["spearman_rho"].tolist()
    r_vals = rq["pearson_r"].tolist()

    colors = []
    for s in sources:
        if s == "judge_all":
            colors.append(COLORS["judge"])
        elif s in dims:
            colors.append(DIM_PALETTE[dims.index(s) % len(DIM_PALETTE)])
        else:
            colors.append(COLORS["neutral"])

    bars1 = ax.bar(x - w/2, rho_vals, w, color=colors, alpha=0.85, label="Spearman ρ")
    bars2 = ax.bar(x + w/2, r_vals, w, color=colors, alpha=0.50, label="Pearson r",
                   edgecolor=colors, linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() if s != "judge_all" else "All dims" for s in sources],
                       fontsize=9, rotation=20, ha="right")
    ax.legend(fontsize=9, framealpha=0.9)
    _apply_theme(ax, title="Ranking Agreement with Human Elo by Rubric Dimension",
                 ylabel="Correlation with Human Elo")

    # Annotate values
    for bar, val, src in zip(bars1, rho_vals, sources):
        label = f"{val:.3f}"
        if src == "judge_all":
            label += "†"  # not independent of per-dim scores
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                label, ha="center", va="bottom", fontsize=7.5, color=COLORS["muted"])

    # Footnote
    ax.text(0.98, 0.02, "† judge_all = uniform avg of dims (not independent)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color=COLORS["muted"], style="italic")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_scatter_grid(elo_table: pd.DataFrame, dims: list[str], path: Path):
    """Grid of scatter plots: dim Elo vs human Elo for each dimension."""
    ncols = min(3, len(dims))
    nrows = math.ceil(len(dims) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
    fig.patch.set_facecolor("white")

    human = elo_table["elo_human"].to_numpy()

    for i, d in enumerate(dims):
        ax = axes[i // ncols][i % ncols]
        dim_elo = elo_table[f"elo_{d}"].to_numpy()
        color = DIM_PALETTE[i % len(DIM_PALETTE)]

        ax.scatter(dim_elo, human, s=22, c=color, alpha=0.7, edgecolor="white", linewidth=0.3)

        # Identity line
        lo = min(np.nanmin(dim_elo), np.nanmin(human)) - 20
        hi = max(np.nanmax(dim_elo), np.nanmax(human)) + 20
        ax.plot([lo, hi], [lo, hi], "--", color=COLORS["muted"], linewidth=0.7, alpha=0.5)

        mask = np.isfinite(dim_elo) & np.isfinite(human)
        rho = sp_stats.spearmanr(dim_elo[mask], human[mask]).statistic
        r = np.corrcoef(dim_elo[mask], human[mask])[0, 1]

        _apply_theme(ax, title=f"{d.capitalize()} (ρ={rho:.3f}, r={r:.3f})",
                     xlabel=f"{d.capitalize()} Elo", ylabel="Human Elo")

        # Label top outliers
        residuals = np.abs(human - dim_elo)
        top_idx = np.argsort(residuals)[-3:]
        for idx in top_idx:
            if np.isfinite(dim_elo[idx]):
                ax.annotate(elo_table.iloc[idx]["short_model"],
                            (dim_elo[idx], human[idx]),
                            fontsize=6.5, color=COLORS["muted"],
                            xytext=(4, 4), textcoords="offset points")

    # Hide unused subplots
    for j in range(len(dims), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle("Per-Rubric Elo vs Human Elo", fontsize=13,
                 fontweight="bold", color=COLORS["text"], y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_delta_heatmap(delta_table: pd.DataFrame, dims: list[str], path: Path):
    """Heatmap: per-model delta (dim Elo − human Elo) for each dimension."""
    # Take top 30 models by human Elo for readability
    top = delta_table.head(30)
    delta_cols = [f"delta_{d}" for d in dims]
    matrix = top[delta_cols].to_numpy(dtype=float)
    labels = top["short_model"].tolist()

    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(dims)), max(8, 0.35 * len(labels))))
    fig.patch.set_facecolor("white")

    vmax = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 100
    vmax = min(vmax, 250)  # cap for readability
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(dims)))
    ax.set_xticklabels([d.capitalize() for d in dims], fontsize=9, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.5)

    # Annotate cells
    for (i, j), val in np.ndenumerate(matrix):
        if np.isfinite(val):
            c = "white" if abs(val) > 0.45 * vmax else COLORS["text"]
            ax.text(j, i, f"{val:+.0f}", ha="center", va="center", fontsize=6.5, color=c)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Elo delta (dim − human)", fontsize=9)

    _apply_theme(ax, title="Per-Rubric Elo Deviation from Human Elo (top 30 models)")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_radar_plot(rq: pd.DataFrame, dims: list[str], path: Path):
    """Radar / spider plot of ranking quality metrics per dimension."""
    dim_rq = rq[rq["source"].isin(dims)].set_index("source").reindex(dims)

    metrics = ["spearman_rho", "pearson_r", "kendall_tau"]
    metric_labels = ["Spearman ρ", "Pearson r", "Kendall τ"]
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")

    for i, d in enumerate(dims):
        if d not in dim_rq.index:
            continue
        vals = [float(dim_rq.loc[d, m]) for m in metrics]
        vals += vals[:1]
        color = DIM_PALETTE[i % len(DIM_PALETTE)]
        ax.plot(angles, vals, "o-", color=color, linewidth=1.5, markersize=4,
                label=d.capitalize(), alpha=0.8)
        ax.fill(angles, vals, color=color, alpha=0.08)

    # Judge all
    if "judge_all" in rq["source"].values:
        jrow = rq.set_index("source").loc["judge_all"]
        vals = [float(jrow[m]) for m in metrics] + [float(jrow[metrics[0]])]
        ax.plot(angles, vals, "s--", color=COLORS["judge"], linewidth=2,
                markersize=5, label="All dims (judge)", alpha=0.9)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=10, color=COLORS["text"])
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", bbox_to_anchor=(1.25, 0), fontsize=8, framealpha=0.9)
    ax.set_title("Ranking Quality by Rubric Dimension", fontsize=12,
                 fontweight="bold", color=COLORS["text"], pad=20)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_mae_bars(rq: pd.DataFrame, dims: list[str], path: Path):
    """Horizontal bar chart: MAE per ranking source."""
    sources = rq["source"].tolist()
    colors = []
    for s in sources:
        if s == "judge_all":
            colors.append(COLORS["judge"])
        elif s in dims:
            colors.append(DIM_PALETTE[dims.index(s) % len(DIM_PALETTE)])
        else:
            colors.append(COLORS["neutral"])

    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(rq))))
    fig.patch.set_facecolor("white")
    ax.barh(np.arange(len(rq)), rq["mae"].tolist(), color=colors, alpha=0.8)
    ax.set_yticks(np.arange(len(rq)))
    ax.set_yticklabels(
        [s[:8].capitalize() if s != "judge_all" else "All" for s in sources],
        fontsize=8,
    )
    _apply_theme(ax, title="Mean |Elo − Human Elo|", xlabel="MAE (Elo points)")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_best_dim_scatter(rq: pd.DataFrame, elo_table: pd.DataFrame, dims: list[str], path: Path):
    """Scatter: best single rubric dimension Elo vs Human Elo."""
    best_dim = rq.loc[rq["source"].isin(dims)].iloc[0]["source"]
    human = elo_table["elo_human"].to_numpy()
    best_elo = elo_table[f"elo_{best_dim}"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("white")
    ax.scatter(
        best_elo, human, s=28,
        c=DIM_PALETTE[dims.index(best_dim) % len(DIM_PALETTE)],
        alpha=0.7, edgecolor="white", linewidth=0.3,
    )
    lo = min(np.nanmin(best_elo), np.nanmin(human)) - 20
    hi = max(np.nanmax(best_elo), np.nanmax(human)) + 20
    ax.plot([lo, hi], [lo, hi], "--", color=COLORS["muted"], linewidth=0.7)
    mask = np.isfinite(best_elo) & np.isfinite(human)
    rho = sp_stats.spearmanr(best_elo[mask], human[mask]).statistic
    _apply_theme(
        ax,
        title=f"Best dim: {best_dim.capitalize()} (ρ={rho:.3f})",
        xlabel=f"{best_dim.capitalize()} Elo",
        ylabel="Human Elo",
    )
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Per-rubric Elo analysis.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--artifact", default="agreement_annotations.json")
    p.add_argument("--regularization", type=float, default=0.01)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # ── Load ──
    artifact_path = resolve_artifact_path(output_dir, args.artifact)
    print(f"[load] {artifact_path}", flush=True)
    df = load_frame(artifact_path)
    dims = infer_dimensions(df)
    models = model_universe(df)
    print(f"[load] {len(df)} samples, {len(models)} models, dims: {dims}", flush=True)

    if not dims:
        print("[error] no rubric score dimensions found — exiting")
        return

    # ── Derive per-dimension preferences ──
    df = add_single_dim_prefs(df, dims)
    print("[prefs] per-dimension preferences derived", flush=True)

    # ── Fit per-rubric Elo ──
    print("[elo] fitting BT-Elo per dimension...", flush=True)
    elo_table, elo_maps = compute_per_rubric_elo(df, dims, models, args.regularization)

    # ── Ranking quality ──
    rq = compute_ranking_quality(elo_table, dims)
    print("\n[ranking quality]")
    for _, row in rq.iterrows():
        src = row["source"]
        tag = "★" if src == rq.iloc[0]["source"] else " "
        print(f"  {tag} {src:15s}  ρ={row['spearman_rho']:.3f}  "
              f"r={row['pearson_r']:.3f}  τ={row['kendall_tau']:.3f}  "
              f"MAE={row['mae']:.0f}")

    # ── Per-model deltas ──
    delta_table = compute_per_model_delta(elo_table, dims)

    # ── Save CSVs ──
    elo_table.to_csv(output_dir / "per_rubric_elo.csv", index=False)
    rq.to_csv(output_dir / "per_rubric_ranking_quality.csv", index=False)
    delta_table.to_csv(output_dir / "per_rubric_model_deltas.csv", index=False)

    # ── Plots ──
    save_ranking_quality_plot(rq, dims, plots_dir / "rubric_elo_ranking_quality.png")
    save_scatter_grid(elo_table, dims, plots_dir / "rubric_elo_scatter_grid.png")
    save_delta_heatmap(delta_table, dims, plots_dir / "rubric_elo_delta_heatmap.png")
    save_radar_plot(rq, dims, plots_dir / "rubric_elo_radar.png")
    save_mae_bars(rq, dims, plots_dir / "rubric_elo_mae.png")
    save_best_dim_scatter(rq, elo_table, dims, plots_dir / "rubric_elo_best_dim_scatter.png")
    print("[plots] all plots saved", flush=True)

    # ── JSON summary ──
    best_dim_row = rq[rq["source"].isin(dims)].iloc[0]
    worst_dim_row = rq[rq["source"].isin(dims)].iloc[-1]
    judge_all_row = rq[rq["source"] == "judge_all"]

    summary = {
        "n_samples": len(df),
        "n_models": len(models),
        "dimensions": dims,
        "best_single_dim": best_dim_row["source"],
        "best_single_dim_rho": float(best_dim_row["spearman_rho"]),
        "best_single_dim_r": float(best_dim_row["pearson_r"]),
        "best_single_dim_mae": float(best_dim_row["mae"]),
        "worst_single_dim": worst_dim_row["source"],
        "worst_single_dim_rho": float(worst_dim_row["spearman_rho"]),
        "worst_single_dim_r": float(worst_dim_row["pearson_r"]),
        "worst_single_dim_mae": float(worst_dim_row["mae"]),
    }

    if not judge_all_row.empty:
        jar = judge_all_row.iloc[0]
        summary["judge_all_rho"] = float(jar["spearman_rho"])
        summary["judge_all_r"] = float(jar["pearson_r"])
        summary["judge_all_mae"] = float(jar["mae"])
        # NOTE: judge_all is the uniform average of all dims, so this
        # comparison is NOT independent—it shows the ceiling of uniform
        # aggregation, not an independent validation.
        summary["all_vs_best_rho_delta"] = round(
            float(jar["spearman_rho"]) - float(best_dim_row["spearman_rho"]), 4)
        summary["_note_judge_all"] = (
            "judge_all uses judge_pref which is the uniform average of rubric "
            "scores; it is NOT independent of per-dimension scores. "
            "The key comparison is single-dim vs human Elo.")

    # Per-dimension summary
    for _, row in rq.iterrows():
        src = row["source"]
        summary[f"dim_{src}_rho"] = float(row["spearman_rho"])
        summary[f"dim_{src}_r"] = float(row["pearson_r"])
        summary[f"dim_{src}_tau"] = float(row["kendall_tau"])
        summary[f"dim_{src}_mae"] = float(row["mae"])

    js_path = output_dir / "per_rubric_elo_summary.json"
    js_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {js_path}")
    print(f"Saved: {plots_dir / 'rubric_elo_*.png'}")


if __name__ == "__main__":
    main()
