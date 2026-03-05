"""Bootstrap convergence analysis: how Elo ratings and agreement metrics
stabilise as sample size grows.

For each subsample fraction (e.g. 10%, 20%, ..., 100%) the script:
  1. Draws B bootstrap resamples of that size from the full data.
  2. Fits BT‑Elo for human and judge on each resample.
  3. Computes per-model Elo deviation from the full-data ("oracle") Elo.
  4. Tracks ranking-quality metrics (Pearson r, Spearman ρ, MAE of centred
     Elo gap, Cohen's κ) across resamples → median ± 90 % CI bands.

Outputs
-------
- Per-fraction summary CSV
- 4 convergence curve plots (Elo deviation, r, ρ/κ, interval width)
- 1 dashboard PNG
- JSON summary
"""
from __future__ import annotations

import argparse
import json
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


# ── core bootstrap loop ───────────────────────────────────────────────────
def bootstrap_at_fraction(
    frame: pd.DataFrame,
    fraction: float,
    *,
    models: list[str],
    oracle_human_elo: dict[str, float],
    oracle_judge_elo: dict[str, float],
    n_bootstrap: int,
    regularization: float,
    seed: int,
) -> list[dict]:
    """Run B bootstrap iterations at the given fraction of the data."""
    rng = np.random.default_rng(seed)
    n = len(frame)
    k = max(10, int(round(fraction * n)))
    rows_out = []

    for b in range(n_bootstrap):
        idx = rng.choice(n, size=k, replace=True)
        sub = frame.iloc[idx].reset_index(drop=True)

        # Fit Elo on subsample
        try:
            h_elo = fit_bt_elo(sub, "human_pref", models, regularization)
            j_elo = fit_bt_elo(sub, "judge_pref", models, regularization)
        except Exception:
            continue

        # Only compare models present in both oracle and subsample
        common = sorted(set(h_elo) & set(j_elo) & set(oracle_human_elo) & set(oracle_judge_elo))
        if len(common) < 5:
            continue

        # Per-model Elo deviation from oracle
        h_dev = [h_elo[m] - oracle_human_elo[m] for m in common]
        j_dev = [j_elo[m] - oracle_judge_elo[m] for m in common]

        # Centred Elo for ranking comparison
        h_vals = np.array([h_elo[m] for m in common])
        j_vals = np.array([j_elo[m] for m in common])
        h_cent = h_vals - h_vals.mean()
        j_cent = j_vals - j_vals.mean()

        # Ranking metrics: judge vs human on this subsample
        r_pearson = float(np.corrcoef(h_cent, j_cent)[0, 1])
        rho_spearman = float(sp_stats.spearmanr(h_cent, j_cent).statistic)
        mae_gap = float(np.mean(np.abs(h_cent - j_cent)))

        # Cohen's kappa (3-class: A/tie/B) on decisive subset of subsample
        valid_both = sub[sub["human_pref"].notna() & sub["judge_pref"].notna()]
        h_labels = [pref_to_label(p) for p in valid_both["human_pref"]]
        j_labels = [pref_to_label(p) for p in valid_both["judge_pref"]]
        try:
            kappa = compute_cohen_kappa(h_labels, j_labels)
        except Exception:
            kappa = float("nan")

        # Accuracy
        agree = sum(1 for h, j in zip(h_labels, j_labels) if h == j)
        acc = agree / len(h_labels) if h_labels else float("nan")

        rows_out.append({
            "fraction": fraction,
            "n_samples": k,
            "bootstrap_id": b,
            "n_common_models": len(common),
            "human_elo_mad": float(np.mean(np.abs(h_dev))),
            "judge_elo_mad": float(np.mean(np.abs(j_dev))),
            "pearson_r": r_pearson,
            "spearman_rho": rho_spearman,
            "mae_centered_gap": mae_gap,
            "cohens_kappa": kappa,
            "accuracy_3class": acc,
            "n_valid_pairs": len(h_labels),
        })

    return rows_out


# ── plotting ───────────────────────────────────────────────────────────────
def _band_plot(ax, summary, col, *, color, label=None):
    """Plot median + 5th-95th CI band."""
    x = summary["n_samples"]
    med = summary[f"{col}_median"]
    lo = summary[f"{col}_p05"]
    hi = summary[f"{col}_p95"]
    ax.fill_between(x, lo, hi, alpha=0.18, color=color)
    ax.plot(x, med, "-o", color=color, markersize=4, linewidth=1.5, label=label)


def save_elo_convergence_plot(summary: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    _band_plot(ax, summary, "human_elo_mad", color=COLORS["human"], label="Human Elo MAD")
    _band_plot(ax, summary, "judge_elo_mad", color=COLORS["judge"], label="Judge Elo MAD")
    ax.legend(fontsize=9, framealpha=0.9)
    _apply_theme(ax, title="Elo Deviation from Full-Data Estimate",
                 xlabel="# bootstrap samples", ylabel="Mean |Elo − Elo_oracle|")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_correlation_convergence_plot(summary: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    _band_plot(ax, summary, "pearson_r", color=COLORS["human"], label="Pearson r")
    _band_plot(ax, summary, "spearman_rho", color=COLORS["judge"], label="Spearman \u03c1")
    ax.legend(fontsize=9, framealpha=0.9)
    _apply_theme(ax, title="Ranking Correlation (Judge vs Human)",
                 xlabel="# bootstrap samples", ylabel="Correlation")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_mae_convergence_plot(summary: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    _band_plot(ax, summary, "mae_centered_gap", color=COLORS["neutral"], label="MAE centred gap")
    ax.legend(fontsize=9, framealpha=0.9)
    _apply_theme(ax, title="Mean |Centred Elo Gap|",
                 xlabel="# bootstrap samples", ylabel="Elo")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_kappa_convergence_plot(summary: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    _band_plot(ax, summary, "cohens_kappa", color=COLORS["human"], label="Cohen's κ")
    ax.legend(fontsize=9, framealpha=0.9)
    _apply_theme(ax, title="Cohen's κ vs Sample Size",
                 xlabel="# bootstrap samples", ylabel="κ")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_accuracy_convergence_plot(summary: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    _band_plot(ax, summary, "accuracy_3class", color=COLORS["judge"], label="3-class accuracy")
    ax.legend(fontsize=9, framealpha=0.9)
    _apply_theme(ax, title="3-Class Agreement vs Sample Size",
                 xlabel="# bootstrap samples", ylabel="Accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Bootstrap convergence analysis.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--artifact", default="agreement_annotations.json")
    p.add_argument("--n_bootstrap", type=int, default=50,
                   help="Bootstrap resamples per fraction.")
    p.add_argument("--fractions", type=str,
                   default="0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.0",
                   help="Comma-separated subsample fractions.")
    p.add_argument("--regularization", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    fractions = [float(f) for f in args.fractions.split(",")]

    artifact_path = resolve_artifact_path(output_dir, args.artifact)
    print(f"[start] loading {artifact_path}", flush=True)
    df = load_frame(artifact_path)
    print(f"[start] {len(df)} samples, fractions={fractions}", flush=True)

    models = model_universe(df)
    print(f"[start] {len(models)} models", flush=True)

    # Oracle = full-data BT Elo
    oracle_human = fit_bt_elo(df, "human_pref", models, args.regularization)
    oracle_judge = fit_bt_elo(df, "judge_pref", models, args.regularization)
    print("[oracle] full-data Elo computed", flush=True)

    all_rows = []
    for fi, frac in enumerate(fractions):
        print(f"[bootstrap] fraction {frac:.0%} ({fi+1}/{len(fractions)})", flush=True)
        rows = bootstrap_at_fraction(
            df, frac,
            models=models,
            oracle_human_elo=oracle_human,
            oracle_judge_elo=oracle_judge,
            n_bootstrap=args.n_bootstrap,
            regularization=args.regularization,
            seed=args.seed + fi * 997,
        )
        all_rows.extend(rows)

    raw_df = pd.DataFrame(all_rows)
    print(f"[bootstrap] {len(raw_df)} total iterations", flush=True)

    # Aggregate: median + 5th/95th percentiles per fraction
    metrics_cols = [
        "human_elo_mad", "judge_elo_mad", "pearson_r", "spearman_rho",
        "mae_centered_gap", "cohens_kappa", "accuracy_3class",
    ]
    agg_rows = []
    for frac, grp in raw_df.groupby("fraction"):
        row = {"fraction": frac, "n_samples": int(grp["n_samples"].iloc[0]),
               "n_bootstrap_ok": len(grp)}
        for col in metrics_cols:
            vals = grp[col].dropna()
            row[f"{col}_median"] = float(vals.median()) if len(vals) else float("nan")
            row[f"{col}_p05"] = float(vals.quantile(0.05)) if len(vals) else float("nan")
            row[f"{col}_p95"] = float(vals.quantile(0.95)) if len(vals) else float("nan")
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{col}_std"] = float(vals.std()) if len(vals) > 1 else float("nan")
        agg_rows.append(row)
    summary = pd.DataFrame(agg_rows).sort_values("fraction").reset_index(drop=True)

    # Save CSVs
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    raw_path = output_dir / "agreement_bootstrap_convergence_raw.csv"
    summary_path = output_dir / "agreement_bootstrap_convergence_summary.csv"
    raw_df.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    # Save plots
    save_elo_convergence_plot(summary, plots_dir / "convergence_elo_deviation.png")
    save_correlation_convergence_plot(summary, plots_dir / "convergence_ranking_correlation.png")
    save_mae_convergence_plot(summary, plots_dir / "convergence_mae_gap.png")
    save_kappa_convergence_plot(summary, plots_dir / "convergence_kappa.png")
    save_accuracy_convergence_plot(summary, plots_dir / "convergence_accuracy.png")

    # JSON summary
    full_row = summary[summary["fraction"] == 1.0]
    js = {
        "n_total_samples": int(len(df)),
        "n_models": len(models),
        "n_fractions": len(fractions),
        "n_bootstrap_per_fraction": args.n_bootstrap,
        "fractions": fractions,
    }
    if len(full_row):
        fr = full_row.iloc[0]
        js["full_data_pearson_r_median"] = float(fr["pearson_r_median"])
        js["full_data_spearman_rho_median"] = float(fr["spearman_rho_median"])
        js["full_data_kappa_median"] = float(fr["cohens_kappa_median"])
        js["full_data_accuracy_median"] = float(fr["accuracy_3class_median"])

    js_path = output_dir / "agreement_bootstrap_convergence_summary.json"
    js_path.write_text(json.dumps(js, indent=2), encoding="utf-8")

    print(f"Saved: {raw_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {js_path}")
    print(f"Saved: {plots_dir / 'convergence_*.png'}")
    print(json.dumps(js, indent=2))


if __name__ == "__main__":
    main()
