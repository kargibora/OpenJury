#!/usr/bin/env python3
"""Unified analysis runner.

Run one, several, or all analysis scripts from a single CLI invocation.
Each analysis writes its outputs into a dedicated sub-folder under the
base ``--output_dir``, so nothing gets mixed up.

Usage examples
--------------
# Run everything (default)
python scripts/analysis/run_analysis.py --output_dir results/

# Run only two analyses
python scripts/analysis/run_analysis.py --output_dir results/ \\
    --analyses bootstrap_convergence sliced

# List available analyses
python scripts/analysis/run_analysis.py --list
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path

# ── Registry ────────────────────────────────────────────────────────────────
# Maps short name → (module_name, description, extra_default_args)
ANALYSIS_REGISTRY: dict[str, tuple[str, str, dict]] = {
    "bootstrap_convergence": (
        "agreement_bootstrap_convergence",
        "Bootstrap convergence of Elo ratings & agreement metrics",
        {"n_bootstrap": "50"},
    ),
    "bt_reconstruction": (
        "agreement_bt_reconstruction",
        "BT-weighted preference reconstruction (global + per-language)",
        {},
    ),
    "bt_model_reconstruction": (
        "agreement_bt_model_reconstruction",
        "BT model reconstruction with cross-validation",
        {},
    ),
    "conformal": (
        "agreement_model_elo_conformal",
        "Held-out conformal calibration of model Elo",
        {},
    ),
    "per_rubric_elo": (
        "agreement_per_rubric_elo",
        "Per-rubric-dimension Elo ranking quality",
        {},
    ),
    "rubric_disagreement": (
        "agreement_rubric_disagreement",
        "Rubric dimension disagreement analysis",
        {},
    ),
    "sliced": (
        "agreement_sliced_analysis",
        "Sliced agreement by language, tier, model family",
        {},
    ),
    "multilingual": (
        "agreement_multilingual_analysis",
        "Multilingual Elo gap & language dashboard",
        {},
    ),
}

# Recommended execution order (faster / prerequisite analyses first)
DEFAULT_ORDER = [
    "sliced",
    "per_rubric_elo",
    "rubric_disagreement",
    "bt_reconstruction",
    "bt_model_reconstruction",
    "multilingual",
    "bootstrap_convergence",
    "conformal",
]


def _run_analysis(
    analysis_name: str,
    base_output_dir: Path,
    artifact_path: Path,
    extra_args: list[str],
    *,
    flat: bool = False,
) -> bool:
    """Import and run a single analysis module.

    Returns True on success, False on error.
    """
    module_name, description, default_extra = ANALYSIS_REGISTRY[analysis_name]

    if flat:
        out_dir = base_output_dir
    else:
        out_dir = base_output_dir / analysis_name
        out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n{'═' * 72}\n"
        f"  ▶  {analysis_name}  —  {description}\n"
        f"     output → {out_dir}\n"
        f"{'═' * 72}",
        flush=True,
    )

    # Build sys.argv for the analysis script's argparse
    argv_backup = sys.argv[:]
    sys.argv = [
        module_name,
        "--output_dir", str(out_dir),
        "--artifact", str(artifact_path),
    ]
    # Apply registry defaults first, then user overrides
    for k, v in default_extra.items():
        flag = f"--{k}"
        if flag not in extra_args:
            sys.argv.extend([flag, v])
    sys.argv.extend(extra_args)

    t0 = time.perf_counter()
    try:
        mod = importlib.import_module(module_name)
        mod.main()  # type: ignore[attr-defined]
        elapsed = time.perf_counter() - t0
        print(f"  ✓  {analysis_name} finished in {elapsed:.1f}s", flush=True)
        return True
    except SystemExit:
        # Some scripts call sys.exit on error
        elapsed = time.perf_counter() - t0
        print(f"  ✗  {analysis_name} exited after {elapsed:.1f}s", flush=True)
        return False
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(
            f"  ✗  {analysis_name} failed after {elapsed:.1f}s: {exc}",
            flush=True,
        )
        return False
    finally:
        sys.argv = argv_backup


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run one or more OpenJury agreement analyses.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Pass extra flags after '--' to forward them to every analysis script.",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help=(
            "Base output directory.  Must contain the annotation artifact "
            "(agreement_annotations.json).  Each analysis creates a sub-folder here."
        ),
    )
    p.add_argument(
        "--artifact",
        default="agreement_annotations.json",
        help="Annotation artifact filename inside --output_dir.",
    )
    p.add_argument(
        "--analyses",
        nargs="*",
        default=None,
        metavar="NAME",
        help=(
            "Which analyses to run (space-separated).  "
            "Omit to run all.  Use --list to see available names."
        ),
    )
    p.add_argument(
        "--flat",
        action="store_true",
        help="Write all outputs directly into --output_dir (no sub-folders).",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Print available analysis names and exit.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        print("Available analyses (in default run order):\n")
        for name in DEFAULT_ORDER:
            _, desc, _ = ANALYSIS_REGISTRY[name]
            print(f"  {name:30s}  {desc}")
        print(f"\nTotal: {len(DEFAULT_ORDER)} analyses")
        return

    base = Path(args.output_dir).resolve()
    artifact = base / args.artifact
    if not artifact.exists():
        raise SystemExit(
            f"Artifact not found: {artifact}\n"
            f"Run the agreement evaluation first, or check --output_dir."
        )

    # Ensure scripts/analysis is on sys.path so the modules can find each other
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    analyses = args.analyses if args.analyses else DEFAULT_ORDER
    unknown = set(analyses) - set(ANALYSIS_REGISTRY)
    if unknown:
        raise SystemExit(
            f"Unknown analyses: {sorted(unknown)}\n"
            f"Available: {sorted(ANALYSIS_REGISTRY)}"
        )

    # Separate extra forwarded args (after --)
    extra_args: list[str] = []
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        extra_args = sys.argv[idx + 1 :]

    print(
        f"OpenJury Analysis Runner\n"
        f"  base dir  : {base}\n"
        f"  artifact  : {args.artifact}\n"
        f"  analyses  : {', '.join(analyses)}\n"
        f"  sub-folders: {'no (flat)' if args.flat else 'yes'}",
        flush=True,
    )

    results: dict[str, bool] = {}
    t_total = time.perf_counter()

    for name in analyses:
        ok = _run_analysis(
            name,
            base,
            artifact,
            extra_args,
            flat=args.flat,
        )
        results[name] = ok

    elapsed_total = time.perf_counter() - t_total
    n_ok = sum(results.values())
    n_fail = len(results) - n_ok

    print(
        f"\n{'─' * 72}\n"
        f"  Done  ({elapsed_total:.1f}s total)  "
        f"✓ {n_ok} passed   ✗ {n_fail} failed\n"
        f"{'─' * 72}",
        flush=True,
    )

    if n_fail:
        failed = [k for k, v in results.items() if not v]
        print(f"  Failed: {', '.join(failed)}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
