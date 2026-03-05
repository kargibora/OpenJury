#!/usr/bin/env python3
"""Compare two or more LLM judge runs side-by-side.

Each ``--run`` argument points to an output directory that contains an
``agreement_annotations.json`` artifact (or a custom name via ``--artifact``).

The script produces:
  • A summary CSV + JSON with per-judge metrics
  • Bar charts comparing κ, accuracy, Elo correlation, tie rates
  • A scatter overlay of judge Elo vs human Elo across judges
  • A radar chart when ≥3 judges are compared
  • A combined dashboard PNG

Usage
-----
python scripts/analysis/compare_judges.py \\
    --run results/gpt4o  --run results/claude35 \\
    --output_dir results/comparison

# Optionally label runs:
python scripts/analysis/compare_judges.py \\
    --run results/gpt4o:GPT-4o  --run results/claude35:Claude-3.5 \\
    --output_dir results/comparison
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENJURY_LOG_LEVEL", "ERROR")

# Ensure sibling modules are importable
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from plotting import (
    COLORS,
    apply_minimal_theme as _theme,
    apply_theme,
    fit_bt_elo,
    load_annotation_artifact,
    model_universe,
    pref_to_label,
    save_figure,
    shorten,
    style_axes,
    valid_decisive,
)
from openjury.analysis.common import compute_agreement_metrics, compute_cohen_kappa


# ── Palette for multiple judges ────────────────────────────────────────────
JUDGE_PALETTE = [
    "#2563eb",  # blue
    "#ea580c",  # orange
    "#059669",  # green
    "#d97706",  # amber
    "#7c3aed",  # violet
    "#db2777",  # pink
    "#0891b2",  # cyan
    "#4f46e5",  # indigo
]


def judge_color(idx: int) -> str:
    return JUDGE_PALETTE[idx % len(JUDGE_PALETTE)]


# ── Per-run summary ────────────────────────────────────────────────────────
def summarize_run(
    label: str,
    artifact: dict,
    meta: dict,
    df: pd.DataFrame,
    *,
    elo_reg: float = 0.01,
) -> dict:
    """Compute a comprehensive metrics dict for one judge run."""
    valid = df[df["human_pref"].notna() & df["judge_pref"].notna()].copy()
    h_labels = [pref_to_label(p) for p in valid["human_pref"]]
    j_labels = [pref_to_label(p) for p in valid["judge_pref"]]

    n = len(valid)
    acc_3 = sum(h == j for h, j in zip(h_labels, j_labels)) / n if n else float("nan")

    try:
        kappa = compute_cohen_kappa(h_labels, j_labels)
    except Exception:
        kappa = float("nan")

    # Decisive subset
    dec = [(h, j) for h, j in zip(h_labels, j_labels) if h != "tie" and j != "tie"]
    n_dec = len(dec)
    acc_dec = sum(h == j for h, j in dec) / n_dec if n_dec else float("nan")
    try:
        kappa_dec = compute_cohen_kappa([h for h, _ in dec], [j for _, j in dec])
    except Exception:
        kappa_dec = float("nan")

    # Tie rates
    human_tie = sum(1 for h in h_labels if h == "tie") / n if n else 0
    judge_tie = sum(1 for j in j_labels if j == "tie") / n if n else 0

    # Elo correlation
    models = model_universe(valid)
    h_elo = fit_bt_elo(valid, "human_pref", models, elo_reg)
    j_elo = fit_bt_elo(valid, "judge_pref", models, elo_reg)
    common = sorted(set(h_elo) & set(j_elo))
    h_vals = np.array([h_elo[m] for m in common])
    j_vals = np.array([j_elo[m] for m in common])

    r_pearson = float(np.corrcoef(h_vals, j_vals)[0, 1]) if len(common) >= 3 else float("nan")
    rho = float(sp_stats.spearmanr(h_vals, j_vals).statistic) if len(common) >= 3 else float("nan")
    mae = float(np.mean(np.abs(h_vals - j_vals))) if common else float("nan")

    # Centred gap
    h_c = h_vals - h_vals.mean()
    j_c = j_vals - j_vals.mean()
    mae_centered = float(np.mean(np.abs(h_c - j_c))) if common else float("nan")

    return {
        "label": label,
        "judge_model": meta.get("judge_model", "?"),
        "dataset": meta.get("dataset", "?"),
        "rubric": meta.get("rubric", "?"),
        "n_samples": n,
        "n_decisive": n_dec,
        "n_models": len(common),
        "accuracy_3class": round(acc_3, 4),
        "accuracy_decisive": round(acc_dec, 4),
        "cohens_kappa": round(kappa, 4) if np.isfinite(kappa) else None,
        "cohens_kappa_decisive": round(kappa_dec, 4) if np.isfinite(kappa_dec) else None,
        "human_tie_rate": round(human_tie, 4),
        "judge_tie_rate": round(judge_tie, 4),
        "elo_pearson_r": round(r_pearson, 4) if np.isfinite(r_pearson) else None,
        "elo_spearman_rho": round(rho, 4) if np.isfinite(rho) else None,
        "elo_mae": round(mae, 1) if np.isfinite(mae) else None,
        "elo_mae_centered": round(mae_centered, 1) if np.isfinite(mae_centered) else None,
        # Stash for plotting
        "_h_elo": h_elo,
        "_j_elo": j_elo,
    }


# ── Plots ──────────────────────────────────────────────────────────────────
def save_metrics_bar_chart(summaries: list[dict], path: Path) -> None:
    """Grouped bar chart: κ, accuracy, Elo ρ for each judge."""
    labels = [s["label"] for s in summaries]
    metrics = [
        ("cohens_kappa", "Cohen's κ"),
        ("accuracy_3class", "Accuracy (3-class)"),
        ("accuracy_decisive", "Accuracy (decisive)"),
        ("elo_spearman_rho", "Elo Spearman ρ"),
    ]
    n_judges = len(labels)
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    width = 0.8 / max(n_judges, 1)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("white")

    for i, s in enumerate(summaries):
        vals = [float(s.get(m, 0) or 0) for m, _ in metrics]
        offset = (i - n_judges / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=s["label"],
                       color=judge_color(i), alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7,
                    color=COLORS["muted"])

    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in metrics], fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=8, framealpha=0.9)
    _theme(ax, title="Judge Agreement Metrics", ylabel="Score")
    save_figure(fig, path)


def save_tie_rate_chart(summaries: list[dict], path: Path) -> None:
    """Side-by-side bar chart: human vs judge tie rates."""
    labels = [s["label"] for s in summaries]
    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(7, 2 * len(labels)), 5))
    fig.patch.set_facecolor("white")

    h_tie = [s["human_tie_rate"] for s in summaries]
    j_tie = [s["judge_tie_rate"] for s in summaries]

    ax.bar(x - w / 2, h_tie, w, color=COLORS["human"], alpha=0.8, label="Human tie rate")
    ax.bar(x + w / 2, j_tie, w, color=[judge_color(i) for i in range(len(labels))],
           alpha=0.8, label="Judge tie rate")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=8, framealpha=0.9)
    _theme(ax, title="Tie Rate Comparison", ylabel="Tie rate")
    save_figure(fig, path)


def save_elo_scatter_overlay(summaries: list[dict], path: Path) -> None:
    """Scatter: judge Elo vs human Elo for each run on the same axes."""
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("white")

    all_vals = []
    for i, s in enumerate(summaries):
        h_elo = s["_h_elo"]
        j_elo = s["_j_elo"]
        common = sorted(set(h_elo) & set(j_elo))
        hv = [h_elo[m] for m in common]
        jv = [j_elo[m] for m in common]
        all_vals.extend(hv + jv)

        ax.scatter(jv, hv, s=30, c=judge_color(i), alpha=0.7,
                   edgecolor="white", linewidth=0.4, label=s["label"], zorder=2)

    if all_vals:
        lo = min(all_vals) - 30
        hi = max(all_vals) + 30
        ax.plot([lo, hi], [lo, hi], "--", color=COLORS["muted"],
                linewidth=0.8, alpha=0.5, zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")

    ax.legend(fontsize=8, framealpha=0.9, loc="upper left")
    _theme(ax, title="Judge Elo vs Human Elo", xlabel="Judge Elo", ylabel="Human Elo")
    save_figure(fig, path)


def save_radar_chart(summaries: list[dict], path: Path) -> None:
    """Radar plot of key metrics (only useful with ≥ 2 judges)."""
    metric_keys = [
        "cohens_kappa", "accuracy_3class", "accuracy_decisive",
        "elo_spearman_rho", "elo_pearson_r",
    ]
    metric_labels = ["κ", "Acc (3-cls)", "Acc (dec)", "Elo ρ", "Elo r"]
    N = len(metric_keys)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")

    for i, s in enumerate(summaries):
        vals = [float(s.get(k, 0) or 0) for k in metric_keys]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", color=judge_color(i), linewidth=1.5,
                markersize=5, label=s["label"], alpha=0.85)
        ax.fill(angles, vals, color=judge_color(i), alpha=0.07)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=10, color=COLORS["text"])
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", bbox_to_anchor=(1.3, 0), fontsize=8, framealpha=0.9)
    ax.set_title("Judge Comparison Radar", fontsize=13, fontweight="bold",
                 color=COLORS["text"], pad=20)
    fig.tight_layout()
    save_figure(fig, path)


def save_dashboard(summaries: list[dict], path: Path) -> None:
    """2×2 dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.patch.set_facecolor("white")

    labels = [s["label"] for s in summaries]
    n_judges = len(labels)

    # (0,0) Agreement metrics bars
    ax = axes[0, 0]
    metrics = [("cohens_kappa", "κ"), ("accuracy_3class", "Acc"),
               ("elo_spearman_rho", "Elo ρ")]
    x = np.arange(len(metrics))
    w = 0.8 / max(n_judges, 1)
    for i, s in enumerate(summaries):
        vals = [float(s.get(m, 0) or 0) for m, _ in metrics]
        offset = (i - n_judges / 2 + 0.5) * w
        ax.bar(x + offset, vals, w, color=judge_color(i), alpha=0.85, label=s["label"])
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in metrics], fontsize=9)
    ax.legend(fontsize=7, framealpha=0.9)
    _theme(ax, title="Key Metrics")

    # (0,1) Tie rates
    ax = axes[0, 1]
    x = np.arange(n_judges)
    ax.bar(x - 0.18, [s["human_tie_rate"] for s in summaries], 0.35,
           color=COLORS["human"], alpha=0.8, label="Human")
    ax.bar(x + 0.18, [s["judge_tie_rate"] for s in summaries], 0.35,
           color=[judge_color(i) for i in range(n_judges)], alpha=0.8, label="Judge")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(fontsize=7, framealpha=0.9)
    _theme(ax, title="Tie Rates")

    # (1,0) Elo scatter overlay
    ax = axes[1, 0]
    all_vals = []
    for i, s in enumerate(summaries):
        h_elo, j_elo = s["_h_elo"], s["_j_elo"]
        common = sorted(set(h_elo) & set(j_elo))
        hv = [h_elo[m] for m in common]
        jv = [j_elo[m] for m in common]
        all_vals.extend(hv + jv)
        ax.scatter(jv, hv, s=20, c=judge_color(i), alpha=0.7,
                   edgecolor="white", linewidth=0.3, label=s["label"])
    if all_vals:
        lo, hi = min(all_vals) - 20, max(all_vals) + 20
        ax.plot([lo, hi], [lo, hi], "--", color=COLORS["muted"], linewidth=0.7, alpha=0.5)
    ax.legend(fontsize=7, framealpha=0.9)
    _theme(ax, title="Judge vs Human Elo", xlabel="Judge Elo", ylabel="Human Elo")

    # (1,1) Summary table
    ax = axes[1, 1]
    ax.axis("off")
    col_labels = ["Judge", "κ", "Acc", "Elo ρ", "MAE", "Tie gap"]
    cell_text = []
    for s in summaries:
        tie_gap = abs(s["judge_tie_rate"] - s["human_tie_rate"])
        cell_text.append([
            s["label"],
            f"{s['cohens_kappa']:.3f}" if s["cohens_kappa"] else "—",
            f"{s['accuracy_3class']:.3f}",
            f"{s['elo_spearman_rho']:.3f}" if s["elo_spearman_rho"] else "—",
            f"{s['elo_mae_centered']:.0f}" if s["elo_mae_centered"] else "—",
            f"{tie_gap:.1%}",
        ])
    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e2e8f0")
            cell.set_text_props(fontweight="bold")
        cell.set_edgecolor(COLORS["band"])

    fig.suptitle("Multi-Judge Comparison", fontsize=14,
                 fontweight="bold", color=COLORS["text"], y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, path)


# ── CLI ─────────────────────────────────────────────────────────────────────
def parse_run_arg(raw: str) -> tuple[Path, str]:
    """Parse 'path[:label]' into (Path, label)."""
    if ":" in raw and not Path(raw).exists():
        path_str, label = raw.rsplit(":", 1)
    else:
        path_str = raw
        label = Path(raw).resolve().name
    return Path(path_str).resolve(), label


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare multiple LLM judge runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--run", action="append", required=True, dest="runs",
        metavar="DIR[:LABEL]",
        help="Path to a judge run output directory. Optionally append :Label.",
    )
    p.add_argument(
        "--output_dir", required=True,
        help="Where to write comparison outputs.",
    )
    p.add_argument(
        "--artifact", default="agreement_annotations.json",
        help="Annotation artifact filename inside each run dir.",
    )
    p.add_argument(
        "--regularization", type=float, default=0.01,
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    apply_theme()

    summaries: list[dict] = []
    for raw in args.runs:
        run_dir, label = parse_run_arg(raw)
        artifact_path = run_dir / args.artifact
        if not artifact_path.exists():
            print(f"⚠  Skipping {label}: {artifact_path} not found", flush=True)
            continue
        print(f"Loading {label} from {run_dir}…", flush=True)
        artifact, meta, df = load_annotation_artifact(run_dir, args.artifact)
        s = summarize_run(label, artifact, meta, df, elo_reg=args.regularization)
        summaries.append(s)
        print(f"  {label}: n={s['n_samples']}, κ={s['cohens_kappa']}, "
              f"acc={s['accuracy_3class']}, ρ={s['elo_spearman_rho']}", flush=True)

    if len(summaries) < 2:
        raise SystemExit("Need at least 2 valid runs to compare.")

    # ── Save CSVs / JSON ──
    export = [{k: v for k, v in s.items() if not k.startswith("_")} for s in summaries]
    summary_df = pd.DataFrame(export)
    summary_df.to_csv(out_dir / "judge_comparison.csv", index=False)
    (out_dir / "judge_comparison.json").write_text(
        json.dumps(export, indent=2, default=str), encoding="utf-8",
    )
    print(f"\nSaved: {out_dir / 'judge_comparison.csv'}", flush=True)

    # ── Plots ──
    save_metrics_bar_chart(summaries, plots_dir / "judge_metrics_bars.png")
    save_tie_rate_chart(summaries, plots_dir / "judge_tie_rates.png")
    save_elo_scatter_overlay(summaries, plots_dir / "judge_elo_scatter.png")
    if len(summaries) >= 2:
        save_radar_chart(summaries, plots_dir / "judge_radar.png")
    save_dashboard(summaries, plots_dir / "judge_comparison_dashboard.png")
    print(f"Saved: {plots_dir / 'judge_*.png'}", flush=True)

    # Print summary table
    print(f"\n{'─' * 72}")
    print(f"{'Judge':>25s}  {'κ':>7s}  {'Acc':>7s}  {'ρ':>7s}  {'MAE':>6s}  {'Tie gap':>8s}")
    print(f"{'─' * 72}")
    for s in summaries:
        tg = abs(s["judge_tie_rate"] - s["human_tie_rate"])
        kstr = f"{s['cohens_kappa']:.3f}" if s["cohens_kappa"] else "  —"
        rstr = f"{s['elo_spearman_rho']:.3f}" if s["elo_spearman_rho"] else "  —"
        mstr = f"{s['elo_mae_centered']:.0f}" if s["elo_mae_centered"] else " —"
        print(f"{s['label']:>25s}  {kstr:>7s}  {s['accuracy_3class']:7.3f}  "
              f"{rstr:>7s}  {mstr:>6s}  {tg:8.1%}")
    print(f"{'─' * 72}")


if __name__ == "__main__":
    main()
