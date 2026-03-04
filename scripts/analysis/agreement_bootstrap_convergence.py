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

from openjury.analysis.common import compute_cohen_kappa
from openjury.arena.config import MatchResult
from openjury.arena.ratings import fit_multi_bt

# ---------------------------------------------------------------------------
BT_TO_ELO = 400.0 / math.log(10.0)
ELO_BASE = 1500.0

COLORS = {
    "human": "#0f766e",
    "judge": "#b45309",
    "neutral": "#334155",
    "band": "#d6d3c8",
    "text": "#1f1f1a",
    "muted": "#6f7268",
    "good": "#0f766e",
    "bad": "#dc2626",
}


# ── helpers ────────────────────────────────────────────────────────────────
def _apply_theme(ax, *, title="", xlabel="", ylabel=""):
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8, color=COLORS["text"])
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=COLORS["text"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=COLORS["text"])
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(COLORS["band"])
    ax.spines["bottom"].set_color(COLORS["band"])
    ax.tick_params(colors=COLORS["text"], labelsize=9)
    ax.grid(True, alpha=0.25, linewidth=0.5, color=COLORS["band"])


def shorten(name: str) -> str:
    return str(name).rsplit("/", 1)[-1]


def load_frame(artifact_path: Path) -> pd.DataFrame:
    with artifact_path.open(encoding="utf-8") as f:
        artifact = json.load(f)
    df = pd.DataFrame(artifact["per_sample"])
    if "metadata" in df.columns:
        meta = pd.json_normalize(df["metadata"]).add_prefix("meta_")
        df = pd.concat([df.drop(columns=["metadata"]), meta], axis=1)
    return df


def valid_decisive(frame: pd.DataFrame, pref_col: str) -> pd.DataFrame:
    out = frame[frame[pref_col].notna()].copy()
    out = out[np.abs(out[pref_col].astype(float) - 0.5) > 0.05]
    return out.reset_index(drop=True)


def pref_to_label(p: float) -> str:
    if p < 0.5:
        return "A"
    if p > 0.5:
        return "B"
    return "tie"


def model_universe(frame: pd.DataFrame) -> list[str]:
    return sorted(set(frame["model_a"]).union(set(frame["model_b"])))


def _build_matches(frame: pd.DataFrame, pref_col: str) -> list[MatchResult]:
    valid = valid_decisive(frame, pref_col)
    matches = []
    for idx, row in valid.iterrows():
        matches.append(
            MatchResult(
                model_a=str(row["model_a"]),
                model_b=str(row["model_b"]),
                instruction_index=int(idx),
                scores_a=row.get("scores_a", {}) or {},
                scores_b=row.get("scores_b", {}) or {},
                preference=float(row[pref_col]),
                instruction="",
                instruction_id="",
                instruction_metadata={},
                completion_a="",
                completion_b="",
                raw_judge_output="",
                raw_judge_output_swapped=None,
            )
        )
    return matches


def fit_bt_elo(frame, pref_col, models, reg=0.01):
    theta = fit_multi_bt(
        models=models,
        matches=_build_matches(frame, pref_col),
        regularization=reg,
    )
    return {m: ELO_BASE + BT_TO_ELO * v for m, v in theta.items()}


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


def save_ranking_convergence_plot(summary: pd.DataFrame, path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("white")

    _band_plot(ax1, summary, "pearson_r", color=COLORS["human"], label="Pearson r")
    _band_plot(ax1, summary, "spearman_rho", color=COLORS["judge"], label="Spearman ρ")
    ax1.legend(fontsize=9, framealpha=0.9)
    _apply_theme(ax1, title="Ranking Correlation (Judge vs Human)",
                 xlabel="# bootstrap samples", ylabel="Correlation")
    ax1.set_ylim(0, 1.05)

    _band_plot(ax2, summary, "mae_centered_gap", color=COLORS["neutral"], label="MAE centred gap")
    ax2.legend(fontsize=9, framealpha=0.9)
    _apply_theme(ax2, title="Mean |Centred Elo Gap|",
                 xlabel="# bootstrap samples", ylabel="Elo")
    fig.tight_layout(w_pad=3)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_agreement_convergence_plot(summary: pd.DataFrame, path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("white")

    _band_plot(ax1, summary, "cohens_kappa", color=COLORS["human"], label="Cohen's κ")
    ax1.legend(fontsize=9, framealpha=0.9)
    _apply_theme(ax1, title="Cohen's κ vs Sample Size",
                 xlabel="# bootstrap samples", ylabel="κ")

    _band_plot(ax2, summary, "accuracy_3class", color=COLORS["judge"], label="3-class accuracy")
    ax2.legend(fontsize=9, framealpha=0.9)
    _apply_theme(ax2, title="3-Class Agreement vs Sample Size",
                 xlabel="# bootstrap samples", ylabel="Accuracy")
    fig.tight_layout(w_pad=3)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_dashboard(summary: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("white")

    ax = axes[0, 0]
    _band_plot(ax, summary, "human_elo_mad", color=COLORS["human"], label="Human")
    _band_plot(ax, summary, "judge_elo_mad", color=COLORS["judge"], label="Judge")
    ax.legend(fontsize=8, framealpha=0.9)
    _apply_theme(ax, title="Elo Deviation from Oracle", ylabel="Mean |ΔElo|")

    ax = axes[0, 1]
    _band_plot(ax, summary, "pearson_r", color=COLORS["human"], label="r")
    _band_plot(ax, summary, "spearman_rho", color=COLORS["judge"], label="ρ")
    ax.legend(fontsize=8, framealpha=0.9)
    _apply_theme(ax, title="Ranking Correlation", ylabel="Correlation")
    ax.set_ylim(0, 1.05)

    ax = axes[1, 0]
    _band_plot(ax, summary, "cohens_kappa", color=COLORS["human"], label="κ")
    ax.legend(fontsize=8, framealpha=0.9)
    _apply_theme(ax, title="Cohen's κ", ylabel="κ")

    ax = axes[1, 1]
    _band_plot(ax, summary, "accuracy_3class", color=COLORS["judge"], label="Accuracy")
    ax.legend(fontsize=8, framealpha=0.9)
    _apply_theme(ax, title="3-Class Accuracy", ylabel="Accuracy")

    for ax in axes.flat:
        ax.set_xlabel("# samples", fontsize=9)

    fig.suptitle("Bootstrap Convergence Dashboard", fontsize=14,
                 fontweight="bold", color=COLORS["text"], y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
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

    print(f"[start] loading {output_dir / args.artifact}", flush=True)
    df = load_frame(output_dir / args.artifact)
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
    save_ranking_convergence_plot(summary, plots_dir / "convergence_ranking.png")
    save_agreement_convergence_plot(summary, plots_dir / "convergence_agreement.png")
    save_dashboard(summary, plots_dir / "convergence_dashboard.png")

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
