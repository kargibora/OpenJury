"""Rubric dimension disagreement analysis.

When human and judge disagree on a pairwise comparison, *which* rubric
dimensions are responsible?  This script computes:

1. Per-dimension "alignment" with human vs judge preferences.
2. Per-dimension score-gap distributions on agree vs disagree cases.
3. Which rubric dimensions best *predict* whether human and judge will agree.
4. "Disagreement profiles" — the rubric fingerprint of systematic failures.

IMPORTANT: ``judge_pref`` is derived from a uniform (equal-weight) average of
the 6 rubric scores, so ``r(gap_d, judge_pref)`` is partially tautological—
each dimension is a component of the aggregate.  The key independent signal is
``r(gap_d, human_pref)``.  The *difference* ``r_judge − r_human`` still
meaningfully reveals which dimensions the judge over-relies on relative to
humans, because a dimension's weight in the uniform aggregate may not match
its importance to human raters.

Outputs
-------
- dimension_alignment.csv     — per-dim correlation with human/judge pref
- disagree_rubric_profile.csv — mean rubric gap on agree vs disagree splits
- dimension_predictive.csv    — logistic-regression coefficients predicting agreement
- 5 plots + 1 dashboard
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

# ---------------------------------------------------------------------------
COLORS = {
    "human": "#0f766e",
    "judge": "#b45309",
    "agree": "#2563eb",
    "disagree": "#dc2626",
    "neutral": "#334155",
    "band": "#d6d3c8",
    "text": "#1f1f1a",
    "muted": "#6f7268",
}

DIM_PALETTE = [
    "#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#db2777",
]


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


def load_frame(artifact_path: Path) -> pd.DataFrame:
    with artifact_path.open(encoding="utf-8") as f:
        artifact = json.load(f)
    df = pd.DataFrame(artifact["per_sample"])
    if "metadata" in df.columns:
        meta = pd.json_normalize(df["metadata"]).add_prefix("meta_")
        df = pd.concat([df.drop(columns=["metadata"]), meta], axis=1)
    return df


def infer_dimensions(frame: pd.DataFrame) -> list[str]:
    for val in frame.get("scores_a", []):
        if isinstance(val, dict) and val:
            return sorted(val.keys())
    return []


def pref_to_sign(p: float) -> float:
    """Map preference to direction: A-wins → -1, B-wins → +1, tie → 0."""
    if p < 0.5:
        return -1.0
    if p > 0.5:
        return 1.0
    return 0.0


def pref_to_label(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "parse_error"
    p = float(p)
    if p < 0.5:
        return "A"
    if p > 0.5:
        return "B"
    return "tie"


# ── core data preparation ─────────────────────────────────────────────────
def build_rubric_frame(df: pd.DataFrame, dims: list[str]) -> pd.DataFrame:
    """Build a flat frame with per-dimension gaps and alignment signals.

    Columns added per dimension d:
        gap_d       = score_a[d] - score_b[d]   (positive favours A)
        abs_gap_d   = |gap_d|
    Plus:
        human_sign  = direction of human preference (-1/0/+1)
        judge_sign  = direction of judge preference (-1/0/+1)
        agree       = human_label == judge_label  (3-class)
        agree_decisive = agree on decisive-only subset
    """
    valid = df[df["human_pref"].notna() & df["judge_pref"].notna()].copy()

    # Extract score gaps
    for d in dims:
        sa = valid["scores_a"].apply(lambda x: (x or {}).get(d, np.nan) if isinstance(x, dict) else np.nan)
        sb = valid["scores_b"].apply(lambda x: (x or {}).get(d, np.nan) if isinstance(x, dict) else np.nan)
        valid[f"gap_{d}"] = sa - sb
        valid[f"abs_gap_{d}"] = (sa - sb).abs()
        valid[f"sa_{d}"] = sa
        valid[f"sb_{d}"] = sb

    valid["human_sign"] = valid["human_pref"].apply(pref_to_sign)
    valid["judge_sign"] = valid["judge_pref"].apply(pref_to_sign)
    valid["human_label"] = valid["human_pref"].apply(pref_to_label)
    valid["judge_label"] = valid["judge_pref"].apply(pref_to_label)
    valid["agree"] = valid["human_label"] == valid["judge_label"]

    # Decisive subset flag
    valid["human_decisive"] = valid["human_pref"].apply(lambda p: abs(p - 0.5) > 0.05)
    valid["judge_decisive"] = valid["judge_pref"].apply(lambda p: abs(p - 0.5) > 0.05)
    valid["both_decisive"] = valid["human_decisive"] & valid["judge_decisive"]

    return valid.reset_index(drop=True)


# ── analysis 1: dimension alignment ───────────────────────────────────────
def compute_dimension_alignment(rf: pd.DataFrame, dims: list[str]) -> pd.DataFrame:
    """For each dimension, how well does its score gap align with
    human preference vs judge preference?

    'Alignment' = point-biserial correlation between sign(gap_d) and pref_sign.
    We also compute mean signed gap conditioned on agree/disagree.
    """
    rows = []
    for d in dims:
        gap = rf[f"gap_{d}"].to_numpy(dtype=float)
        mask = np.isfinite(gap)
        if mask.sum() < 10:
            continue

        h_sign = rf["human_sign"].to_numpy(dtype=float)
        j_sign = rf["judge_sign"].to_numpy(dtype=float)

        # Alignment with human: when gap_d > 0 (A is better on d),
        # does human also prefer A (sign = -1)?  We want correlation of
        # -gap_d with human_sign (since gap>0 favours A, sign<0 favours A)
        # Simpler: correlation of gap_d * (-human_sign)
        # Actually, let's just correlate gap_d with judge_sign - human_sign
        # to see which dimension the *judge* over-relies on.

        # Pearson r(gap_d, human_sign): negative means "higher A score on d ↔ human prefers A"
        m = mask & np.isfinite(h_sign) & np.isfinite(j_sign)
        r_human = float(np.corrcoef(gap[m], h_sign[m])[0, 1]) if m.sum() > 2 else np.nan
        r_judge = float(np.corrcoef(gap[m], j_sign[m])[0, 1]) if m.sum() > 2 else np.nan

        # Mean gap on agree vs disagree
        agree_mask = rf["agree"].to_numpy() & m
        disagree_mask = ~rf["agree"].to_numpy() & m
        mean_abs_gap_agree = float(np.abs(gap[agree_mask]).mean()) if agree_mask.sum() else np.nan
        mean_abs_gap_disagree = float(np.abs(gap[disagree_mask]).mean()) if disagree_mask.sum() else np.nan

        # "Dimension conflict": how often does this dimension's gap direction
        # match judge but contradict human?
        gap_sign = np.sign(gap[m])
        judge_match = (gap_sign == j_sign[m]) | (gap_sign == 0)
        human_match = (gap_sign == h_sign[m]) | (gap_sign == 0)
        # Judge-aligned but human-opposed
        conflict_rate = float(((gap_sign != 0) & judge_match & ~human_match).mean()) if m.sum() > 0 else np.nan

        rows.append({
            "dimension": d,
            "n_valid": int(m.sum()),
            "r_with_human_pref": round(r_human, 4) if np.isfinite(r_human) else None,
            "r_with_judge_pref": round(r_judge, 4) if np.isfinite(r_judge) else None,
            "r_gap_judge_minus_human": round(r_judge - r_human, 4) if (np.isfinite(r_human) and np.isfinite(r_judge)) else None,
            "mean_abs_gap_agree": round(mean_abs_gap_agree, 4) if np.isfinite(mean_abs_gap_agree) else None,
            "mean_abs_gap_disagree": round(mean_abs_gap_disagree, 4) if np.isfinite(mean_abs_gap_disagree) else None,
            "gap_ratio_disagree_over_agree": round(mean_abs_gap_disagree / mean_abs_gap_agree, 3) if (np.isfinite(mean_abs_gap_agree) and mean_abs_gap_agree > 0 and np.isfinite(mean_abs_gap_disagree)) else None,
            "judge_conflict_rate": round(conflict_rate, 4) if np.isfinite(conflict_rate) else None,
        })
    return pd.DataFrame(rows)


# ── analysis 2: disagree profiles ─────────────────────────────────────────
def compute_disagree_profile(rf: pd.DataFrame, dims: list[str]) -> pd.DataFrame:
    """Per-dimension statistics split by agree/disagree."""
    rows = []
    for d in dims:
        gap = rf[f"gap_{d}"]
        for split, mask in [("agree", rf["agree"]), ("disagree", ~rf["agree"])]:
            g = gap[mask].dropna()
            if len(g) < 5:
                continue
            rows.append({
                "dimension": d,
                "split": split,
                "n": len(g),
                "mean_gap": round(float(g.mean()), 4),
                "mean_abs_gap": round(float(g.abs().mean()), 4),
                "std_gap": round(float(g.std()), 4),
                "median_abs_gap": round(float(g.abs().median()), 4),
                "q75_abs_gap": round(float(g.abs().quantile(0.75)), 4),
            })
    return pd.DataFrame(rows)


# ── analysis 3: predictive power via logistic-like scoring ─────────────────
def compute_predictive_importance(rf: pd.DataFrame, dims: list[str]) -> pd.DataFrame:
    """Which rubric dimension features best predict agreement?

    We use a simple approach: for each dimension, compute the
    AUC (agreement ~ abs_gap_d) via the Mann-Whitney U statistic.
    Higher AUC means the dimension is more informative for predicting
    agreement (e.g., 'when abs_gap is large, they tend to agree more').
    """
    y = rf["agree"].to_numpy().astype(float)
    rows = []
    for d in dims:
        x = rf[f"abs_gap_{d}"].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 20:
            continue

        # Mann-Whitney U: does abs_gap differ between agree/disagree?
        agree_vals = x[m & (y == 1)]
        disagree_vals = x[m & (y == 0)]
        if len(agree_vals) < 5 or len(disagree_vals) < 5:
            continue

        u_stat, u_pval = sp_stats.mannwhitneyu(agree_vals, disagree_vals, alternative="two-sided")
        # Convert to AUC
        auc = u_stat / (len(agree_vals) * len(disagree_vals))

        # Point-biserial correlation
        rpb, rpb_pval = sp_stats.pointbiserialr(y[m], x[m])

        # Mean difference
        mean_diff = float(agree_vals.mean() - disagree_vals.mean())

        rows.append({
            "dimension": d,
            "n_agree": len(agree_vals),
            "n_disagree": len(disagree_vals),
            "mean_abs_gap_agree": round(float(agree_vals.mean()), 4),
            "mean_abs_gap_disagree": round(float(disagree_vals.mean()), 4),
            "mean_diff": round(mean_diff, 4),
            "auc_agree_vs_disagree": round(float(auc), 4),
            "point_biserial_r": round(float(rpb), 4),
            "pval": float(rpb_pval),
            "significant": rpb_pval < 0.05,
        })
    return pd.DataFrame(rows).sort_values("auc_agree_vs_disagree", ascending=False).reset_index(drop=True)


# ── analysis 4: disagreement direction breakdown ──────────────────────────
def compute_direction_breakdown(rf: pd.DataFrame, dims: list[str]) -> pd.DataFrame:
    """On disagreement cases, break down by direction:
    - Judge says A, Human says B  (judge_A_human_B)
    - Judge says B, Human says A  (judge_B_human_A)
    - One says tie, other doesn't (tie mismatch)

    For each direction, compute mean signed rubric gap per dimension.
    """
    dis = rf[~rf["agree"]].copy()
    rows = []

    conditions = [
        ("judge_A_human_B", (dis["judge_label"] == "A") & (dis["human_label"] == "B")),
        ("judge_B_human_A", (dis["judge_label"] == "B") & (dis["human_label"] == "A")),
        ("judge_tie_human_decisive", (dis["judge_label"] == "tie") & (dis["human_label"].isin(["A", "B"]))),
        ("judge_decisive_human_tie", (dis["judge_label"].isin(["A", "B"])) & (dis["human_label"] == "tie")),
    ]

    for label, mask in conditions:
        subset = dis[mask]
        if len(subset) < 5:
            continue
        row = {"direction": label, "n": len(subset)}
        for d in dims:
            g = subset[f"gap_{d}"].dropna()
            row[f"mean_gap_{d}"] = round(float(g.mean()), 3) if len(g) > 0 else None
            row[f"mean_abs_gap_{d}"] = round(float(g.abs().mean()), 3) if len(g) > 0 else None
        rows.append(row)

    return pd.DataFrame(rows)


# ── plotting ───────────────────────────────────────────────────────────────
def save_alignment_plot(alignment: pd.DataFrame, path: Path):
    """Grouped bar chart: r(gap, human_pref) vs r(gap, judge_pref) per dimension."""
    dims = alignment["dimension"].tolist()
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    x = np.arange(len(dims))
    w = 0.35

    rh = alignment["r_with_human_pref"].fillna(0).tolist()
    rj = alignment["r_with_judge_pref"].fillna(0).tolist()
    ax.bar(x - w/2, rh, w, color=COLORS["human"], alpha=0.85, label="r(gap, human pref)")
    ax.bar(x + w/2, rj, w, color=COLORS["judge"], alpha=0.85, label="r(gap, judge pref)")

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in dims], fontsize=9)
    ax.axhline(0, color=COLORS["text"], linewidth=0.6, alpha=0.4)
    ax.legend(fontsize=9, framealpha=0.9)
    _apply_theme(ax, title="Rubric Dimension Alignment with Preferences",
                 ylabel="Pearson r (gap vs preference direction)")

    # Footnote: judge pref is not independent of rubric scores
    ax.text(0.98, 0.02,
            "Note: judge pref = uniform avg of dims (r with judge is partially tautological)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color=COLORS["muted"], style="italic")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_gap_distribution_plot(rf: pd.DataFrame, dims: list[str], path: Path):
    """Box plots: abs rubric gap on agree vs disagree, per dimension."""
    fig, axes = plt.subplots(1, len(dims), figsize=(3 * len(dims), 5), sharey=True)
    fig.patch.set_facecolor("white")
    if len(dims) == 1:
        axes = [axes]

    for i, d in enumerate(dims):
        ax = axes[i]
        agree_vals = rf.loc[rf["agree"], f"abs_gap_{d}"].dropna()
        disagree_vals = rf.loc[~rf["agree"], f"abs_gap_{d}"].dropna()

        bp = ax.boxplot(
            [agree_vals, disagree_vals],
            labels=["Agree", "Disagree"],
            patch_artist=True,
            widths=0.6,
            medianprops=dict(color=COLORS["text"], linewidth=1.5),
        )
        bp["boxes"][0].set_facecolor(COLORS["agree"])
        bp["boxes"][0].set_alpha(0.4)
        bp["boxes"][1].set_facecolor(COLORS["disagree"])
        bp["boxes"][1].set_alpha(0.4)
        _apply_theme(ax, title=d.capitalize())
        if i == 0:
            ax.set_ylabel("|Score gap|", fontsize=10)

    fig.suptitle("Rubric Score Gap: Agree vs Disagree Cases",
                 fontsize=12, fontweight="bold", color=COLORS["text"], y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_conflict_plot(alignment: pd.DataFrame, path: Path):
    """Bar chart: judge-conflict rate per dimension."""
    dims = alignment["dimension"].tolist()
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")

    conflict = alignment["judge_conflict_rate"].fillna(0).tolist()
    x = np.arange(len(dims))
    bars = ax.bar(x, conflict, color=[DIM_PALETTE[i % len(DIM_PALETTE)] for i in range(len(dims))],
                  alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in dims], fontsize=9)
    _apply_theme(ax, title="Judge-Conflict Rate per Dimension",
                 ylabel="P(dim aligns with judge but contradicts human)")

    # Annotate values
    for bar, val in zip(bars, conflict):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=8, color=COLORS["muted"])

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_predictive_plot(predictive: pd.DataFrame, path: Path):
    """Horizontal bar: AUC of each dimension for predicting agreement."""
    pred = predictive.sort_values("auc_agree_vs_disagree").copy()
    fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * len(pred))))
    fig.patch.set_facecolor("white")

    y = np.arange(len(pred))
    colors = [COLORS["agree"] if a > 0.5 else COLORS["disagree"]
              for a in pred["auc_agree_vs_disagree"]]
    ax.barh(y, pred["auc_agree_vs_disagree"], color=colors, alpha=0.8)
    ax.axvline(0.5, color=COLORS["text"], linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([d.capitalize() for d in pred["dimension"]], fontsize=9)

    # Star significant ones
    for i, (_, row) in enumerate(pred.iterrows()):
        marker = " *" if row["significant"] else ""
        ax.text(row["auc_agree_vs_disagree"] + 0.005, i,
                f"{row['auc_agree_vs_disagree']:.3f}{marker}",
                va="center", fontsize=8, color=COLORS["muted"])

    _apply_theme(ax, title="Predictive Power: |Score Gap| → Agreement  (* p < 0.05)",
                 xlabel="AUC (agree vs disagree)")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_direction_heatmap(direction_df: pd.DataFrame, dims: list[str], path: Path):
    """Heatmap: disagreement direction × rubric dimension mean gap."""
    if direction_df.empty:
        return

    gap_cols = [f"mean_gap_{d}" for d in dims]
    labels = direction_df["direction"].tolist()
    matrix = direction_df[gap_cols].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(dims)), max(3, 0.6 * len(labels))))
    fig.patch.set_facecolor("white")

    vmax = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1.0
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(dims)))
    ax.set_xticklabels([d.capitalize() for d in dims], fontsize=9, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels([l.replace("_", " ") for l in labels], fontsize=9)

    # Annotate cells
    for (i, j), val in np.ndenumerate(matrix):
        if np.isfinite(val):
            color = "white" if abs(val) > 0.45 * vmax else COLORS["text"]
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean signed gap (A−B)", fontsize=9)

    _apply_theme(ax, title="Rubric Gap Profile by Disagreement Direction")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_dashboard(alignment, rf, dims, predictive, direction_df, path: Path):
    """2×2 dashboard summarising rubric disagreement analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("white")

    # (0,0) Alignment bars
    ax = axes[0, 0]
    x = np.arange(len(dims))
    w = 0.35
    rh = alignment.set_index("dimension").reindex(dims)["r_with_human_pref"].fillna(0).tolist()
    rj = alignment.set_index("dimension").reindex(dims)["r_with_judge_pref"].fillna(0).tolist()
    ax.bar(x - w/2, rh, w, color=COLORS["human"], alpha=0.85, label="Human")
    ax.bar(x + w/2, rj, w, color=COLORS["judge"], alpha=0.85, label="Judge")
    ax.set_xticks(x)
    ax.set_xticklabels([d[:8].capitalize() for d in dims], fontsize=8)
    ax.axhline(0, color=COLORS["text"], linewidth=0.5, alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")
    _apply_theme(ax, title="Alignment: r(gap, pref)")

    # (0,1) Box plots (agree vs disagree) for all dims
    ax = axes[0, 1]
    agree_means = []
    disagree_means = []
    for d in dims:
        agree_means.append(rf.loc[rf["agree"], f"abs_gap_{d}"].dropna().mean())
        disagree_means.append(rf.loc[~rf["agree"], f"abs_gap_{d}"].dropna().mean())
    x = np.arange(len(dims))
    ax.bar(x - w/2, agree_means, w, color=COLORS["agree"], alpha=0.7, label="Agree")
    ax.bar(x + w/2, disagree_means, w, color=COLORS["disagree"], alpha=0.7, label="Disagree")
    ax.set_xticks(x)
    ax.set_xticklabels([d[:8].capitalize() for d in dims], fontsize=8)
    ax.legend(fontsize=7, loc="upper right")
    _apply_theme(ax, title="Mean |Gap| by Agree/Disagree")

    # (1,0) Conflict rate
    ax = axes[1, 0]
    al = alignment.set_index("dimension").reindex(dims)
    conflict = al["judge_conflict_rate"].fillna(0).tolist()
    bars = ax.bar(np.arange(len(dims)), conflict,
                  color=[DIM_PALETTE[i % len(DIM_PALETTE)] for i in range(len(dims))], alpha=0.85)
    ax.set_xticks(np.arange(len(dims)))
    ax.set_xticklabels([d[:8].capitalize() for d in dims], fontsize=8)
    _apply_theme(ax, title="Judge-Conflict Rate")

    # (1,1) Direction heatmap (if available)
    ax = axes[1, 1]
    if not direction_df.empty:
        gap_cols = [f"mean_gap_{d}" for d in dims]
        matrix = direction_df[gap_cols].to_numpy(dtype=float)
        labels = direction_df["direction"].tolist()
        vmax = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1.0
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(len(dims)))
        ax.set_xticklabels([d[:6].capitalize() for d in dims], fontsize=7, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels([l.replace("_", "\n") for l in labels], fontsize=7)
        for (i, j), val in np.ndenumerate(matrix):
            if np.isfinite(val):
                c = "white" if abs(val) > 0.45 * vmax else COLORS["text"]
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=7, color=c)
        _apply_theme(ax, title="Gap Profile by Direction")
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color=COLORS["muted"])
        _apply_theme(ax, title="Gap Profile by Direction")

    fig.suptitle("Rubric Dimension Disagreement Analysis", fontsize=14,
                 fontweight="bold", color=COLORS["text"], y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Rubric dimension disagreement analysis.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--artifact", default="agreement_annotations.json")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # ── Load ──
    print(f"[load] {output_dir / args.artifact}", flush=True)
    df = load_frame(output_dir / args.artifact)
    dims = infer_dimensions(df)
    print(f"[load] {len(df)} samples, {len(dims)} rubric dims: {dims}", flush=True)

    if not dims:
        print("[error] no rubric score dimensions found — exiting")
        return

    # ── Build rubric frame ──
    rf = build_rubric_frame(df, dims)
    n_agree = rf["agree"].sum()
    n_disagree = (~rf["agree"]).sum()
    print(f"[data] {n_agree} agree, {n_disagree} disagree "
          f"({100*n_agree/len(rf):.1f}% / {100*n_disagree/len(rf):.1f}%)", flush=True)

    # ── Analysis 1: Dimension alignment ──
    alignment = compute_dimension_alignment(rf, dims)
    alignment.to_csv(output_dir / "rubric_dimension_alignment.csv", index=False)
    print("[alignment]")
    for _, row in alignment.iterrows():
        print(f"  {row['dimension']:15s}  r_human={row['r_with_human_pref']:+.3f}  "
              f"r_judge={row['r_with_judge_pref']:+.3f}  "
              f"conflict={row['judge_conflict_rate']:.1%}")

    # ── Analysis 2: Disagree profile ──
    profile = compute_disagree_profile(rf, dims)
    profile.to_csv(output_dir / "rubric_disagree_profile.csv", index=False)

    # ── Analysis 3: Predictive importance ──
    predictive = compute_predictive_importance(rf, dims)
    predictive.to_csv(output_dir / "rubric_dimension_predictive.csv", index=False)
    print("[predictive]")
    for _, row in predictive.iterrows():
        sig = " *" if row["significant"] else ""
        print(f"  {row['dimension']:15s}  AUC={row['auc_agree_vs_disagree']:.3f}  "
              f"rpb={row['point_biserial_r']:+.3f}{sig}")

    # ── Analysis 4: Direction breakdown ──
    direction_df = compute_direction_breakdown(rf, dims)
    direction_df.to_csv(output_dir / "rubric_direction_breakdown.csv", index=False)

    # ── Plots ──
    save_alignment_plot(alignment, plots_dir / "rubric_alignment.png")
    save_gap_distribution_plot(rf, dims, plots_dir / "rubric_gap_distributions.png")
    save_conflict_plot(alignment, plots_dir / "rubric_conflict_rate.png")
    save_predictive_plot(predictive, plots_dir / "rubric_predictive_power.png")
    save_direction_heatmap(direction_df, dims, plots_dir / "rubric_direction_heatmap.png")
    save_dashboard(alignment, rf, dims, predictive, direction_df,
                   plots_dir / "rubric_dashboard.png")
    print("[plots] all plots saved", flush=True)

    # ── JSON summary ──
    # Find most judge-overweighted dimension (highest r_gap)
    if not alignment.empty:
        al_sorted = alignment.sort_values("r_gap_judge_minus_human", ascending=False)
        most_overweighted = al_sorted.iloc[0]["dimension"]
        most_overweighted_gap = al_sorted.iloc[0]["r_gap_judge_minus_human"]
        most_underweighted = al_sorted.iloc[-1]["dimension"]
        most_underweighted_gap = al_sorted.iloc[-1]["r_gap_judge_minus_human"]
    else:
        most_overweighted = most_underweighted = None
        most_overweighted_gap = most_underweighted_gap = None

    summary = {
        "n_samples": len(rf),
        "n_agree": int(n_agree),
        "n_disagree": int(n_disagree),
        "agree_rate": round(float(n_agree / len(rf)), 4),
        "dimensions": dims,
        "most_judge_overweighted_dim": most_overweighted,
        "most_judge_overweighted_r_gap": float(most_overweighted_gap) if most_overweighted_gap is not None else None,
        "most_judge_underweighted_dim": most_underweighted,
        "most_judge_underweighted_r_gap": float(most_underweighted_gap) if most_underweighted_gap is not None else None,
    }

    # Per-dimension summary
    for _, row in alignment.iterrows():
        d = row["dimension"]
        summary[f"dim_{d}_r_human"] = row["r_with_human_pref"]
        summary[f"dim_{d}_r_judge"] = row["r_with_judge_pref"]
        summary[f"dim_{d}_conflict_rate"] = row["judge_conflict_rate"]

    for _, row in predictive.iterrows():
        d = row["dimension"]
        summary[f"dim_{d}_auc"] = row["auc_agree_vs_disagree"]
        summary[f"dim_{d}_significant"] = bool(row["significant"])

    js_path = output_dir / "rubric_disagreement_summary.json"
    js_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {js_path}")
    print(f"Saved: {plots_dir / 'rubric_*.png'}")


if __name__ == "__main__":
    main()
