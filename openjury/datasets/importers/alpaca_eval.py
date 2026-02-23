"""Import pre-existing Alpaca-Eval model outputs into the CompletionCache.

The `tatsu-lab/alpaca_eval <https://github.com/tatsu-lab/alpaca_eval>`_
repository ships completions from **~160 models** under::

    results/{model_name}/model_outputs.json

Each JSON file contains 805 entries (one per instruction) with fields
``instruction``, ``output``, ``generator``, and optionally ``dataset``.

This importer:

1. Clones / updates the repo into ``$OPENJURY_DATA/alpaca_eval_repo/``.
2. Discovers all ``results/*/model_outputs.json`` files.
3. For each model, validates instruction alignment with the canonical
   alpaca-eval dataset (by comparing instruction text at each index).
4. Stores completions in the :class:`CompletionCache` as parquets under
   the **``AlpacaEval/{generator_name}``** namespace, so they're
   automatically discovered by the arena pipeline.

Usage::

    # Import all models
    uv run python -m openjury.datasets.importers.alpaca_eval

    # List available models without importing
    uv run python -m openjury.datasets.importers.alpaca_eval --dry-run

    # Import specific models only
    uv run python -m openjury.datasets.importers.alpaca_eval \\
        --models gpt-4-turbo-2024-04-09 claude-3-opus-20240229

    # Use an existing clone
    uv run python -m openjury.datasets.importers.alpaca_eval \\
        --repo-path /path/to/alpaca_eval

After import, completions are available in the arena pipeline::

    {
        "models": [
            "AlpacaEval/gpt-4-turbo-2024-04-09",
            "AlpacaEval/claude-3-opus-20240229",
            "VLLM/Qwen/Qwen2.5-7B-Instruct"
        ],
        ...
    }
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from openjury._logging import logger
from openjury.completion_cache import cache
from openjury.datasets import load_dataset
from openjury.utils import data_root


# ═════════════════════════════════════════════════════════════════════
#  Constants
# ═════════════════════════════════════════════════════════════════════

REPO_URL = "https://github.com/tatsu-lab/alpaca_eval.git"
DATASET_NAME = "alpaca-eval"
MODEL_PREFIX = "AlpacaEval"


# ═════════════════════════════════════════════════════════════════════
#  Core import logic
# ═════════════════════════════════════════════════════════════════════


def _ensure_repo(repo_path: Path | None = None) -> Path:
    """Clone or update the alpaca_eval repository.

    Args:
        repo_path: Existing local clone.  If ``None``, clones into
            ``$OPENJURY_DATA/alpaca_eval_repo/``.

    Returns:
        Path to the repository root.
    """
    if repo_path is not None:
        repo_path = Path(repo_path)
        if not (repo_path / "results").is_dir():
            raise FileNotFoundError(
                f"No results/ directory in {repo_path}. "
                "Is this a valid alpaca_eval clone?"
            )
        return repo_path

    repo_path = data_root / "alpaca_eval_repo"
    if (repo_path / ".git").is_dir():
        logger.info("Updating existing clone at %s …", repo_path)
        subprocess.run(
            ["git", "-C", str(repo_path), "pull", "--ff-only"],
            check=False,
            capture_output=True,
        )
    else:
        logger.info("Cloning %s → %s …", REPO_URL, repo_path)
        # Shallow clone — we only need the results/ directory
        subprocess.run(
            [
                "git", "clone",
                "--depth", "1",
                "--filter=blob:none",
                "--sparse",
                REPO_URL,
                str(repo_path),
            ],
            check=True,
        )
        # Sparse-checkout only results/ to save space
        subprocess.run(
            ["git", "-C", str(repo_path), "sparse-checkout", "set", "results"],
            check=True,
        )

    return repo_path


def discover_models(repo_path: Path) -> list[str]:
    """List all model names that have ``model_outputs.json``.

    Args:
        repo_path: Path to the alpaca_eval repository clone.

    Returns:
        Sorted list of model directory names.
    """
    results_dir = repo_path / "results"
    models = []
    for d in sorted(results_dir.iterdir()):
        if d.is_dir() and (d / "model_outputs.json").exists():
            models.append(d.name)
    return models


def _load_reference_instructions() -> list[str]:
    """Load the canonical alpaca-eval instruction list."""
    ds = load_dataset(DATASET_NAME)
    return ds.instructions


def import_model(
    model_name: str,
    repo_path: Path,
    reference_instructions: list[str],
    *,
    force: bool = False,
) -> bool:
    """Import a single model's completions into the cache.

    Args:
        model_name: Directory name under ``results/`` (e.g.
            ``"gpt-4-turbo-2024-04-09"``).
        repo_path: Path to the alpaca_eval repository clone.
        reference_instructions: Canonical instruction list for
            alignment verification.
        force: Overwrite existing cache entries.

    Returns:
        ``True`` if completions were imported (or already cached),
        ``False`` on error.
    """
    cache_model = f"{MODEL_PREFIX}/{model_name}"

    # ── Already cached? ──────────────────────────────────────────
    if not force and cache.exists(cache_model, DATASET_NAME):
        logger.info("  ✓ %s — already cached, skipping", cache_model)
        return True

    # ── Load JSON ────────────────────────────────────────────────
    json_path = repo_path / "results" / model_name / "model_outputs.json"
    if not json_path.exists():
        logger.warning("  ✗ %s — model_outputs.json not found", model_name)
        return False

    try:
        with open(json_path, encoding="utf-8") as f:
            entries: list[dict[str, Any]] = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("  ✗ %s — failed to read JSON: %s", model_name, exc)
        return False

    # ── Validate count ───────────────────────────────────────────
    n_expected = len(reference_instructions)
    if len(entries) != n_expected:
        logger.warning(
            "  ✗ %s — expected %d entries, got %d",
            model_name, n_expected, len(entries),
        )
        return False

    # ── Validate instruction alignment & build DataFrame ─────────
    completions: list[str] = []
    indices: list[int] = []
    mismatches = 0

    for i, entry in enumerate(entries):
        output = entry.get("output", "")
        instruction = entry.get("instruction", "")

        # Verify alignment: compare first 200 chars to handle
        # minor whitespace differences
        ref = reference_instructions[i]
        if instruction.strip()[:200] != ref.strip()[:200]:
            if mismatches < 3:
                logger.warning(
                    "  ⚠ %s index %d instruction mismatch:\n"
                    "    repo: %s\n"
                    "    ours: %s",
                    model_name, i,
                    instruction.strip()[:80],
                    ref.strip()[:80],
                )
            mismatches += 1

        completions.append(output)
        indices.append(i)

    if mismatches > 0:
        logger.warning(
            "  ⚠ %s — %d/%d instruction mismatches (importing anyway)",
            model_name, mismatches, n_expected,
        )

    # ── Store in cache ───────────────────────────────────────────
    df = pd.DataFrame({
        "completion": completions,
        "instruction_index": indices,
    })
    cache.put(df, cache_model, DATASET_NAME)
    logger.info("  ✓ %s — imported %d completions", cache_model, len(df))
    return True


def import_alpaca_eval_outputs(
    *,
    repo_path: Path | None = None,
    models: list[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, bool]:
    """Import Alpaca-Eval model outputs into the CompletionCache.

    Args:
        repo_path: Path to an existing alpaca_eval clone, or ``None``
            to auto-clone into ``$OPENJURY_DATA/alpaca_eval_repo/``.
        models: Specific model names to import.  ``None`` = all.
        force: Overwrite existing cache entries.
        dry_run: Only list available models, don't import.

    Returns:
        Dict mapping model name → success boolean.
    """
    repo = _ensure_repo(repo_path)
    available = discover_models(repo)

    logger.info("Found %d models with completions in %s", len(available), repo)

    if dry_run:
        for m in available:
            cached = cache.exists(f"{MODEL_PREFIX}/{m}", DATASET_NAME)
            status = "cached" if cached else "available"
            print(f"  [{status:>9}] {MODEL_PREFIX}/{m}")
        return {}

    # ── Filter models ────────────────────────────────────────────
    if models:
        to_import = [m for m in models if m in available]
        missing = [m for m in models if m not in available]
        if missing:
            logger.warning("Models not found in repo: %s", missing)
    else:
        to_import = available

    if not to_import:
        logger.warning("No models to import.")
        return {}

    # ── Load reference instructions once ─────────────────────────
    logger.info("Loading reference instructions …")
    ref_instructions = _load_reference_instructions()
    logger.info("Reference: %d instructions", len(ref_instructions))

    # ── Import each model ────────────────────────────────────────
    results: dict[str, bool] = {}
    for i, model_name in enumerate(to_import, 1):
        logger.info("[%d/%d] Importing %s …", i, len(to_import), model_name)
        results[model_name] = import_model(
            model_name, repo, ref_instructions, force=force,
        )

    # ── Summary ──────────────────────────────────────────────────
    ok = sum(v for v in results.values())
    fail = len(results) - ok
    logger.info(
        "Import complete: %d succeeded, %d failed (out of %d)",
        ok, fail, len(results),
    )
    return results


# ═════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="openjury.datasets.importers.alpaca_eval",
        description=(
            "Import pre-existing Alpaca-Eval model completions into the "
            "OpenJury CompletionCache with AlpacaEval/ prefix."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:

  # List all available models
  uv run python -m openjury.datasets.importers.alpaca_eval --dry-run

  # Import all ~160 models
  uv run python -m openjury.datasets.importers.alpaca_eval

  # Import specific models
  uv run python -m openjury.datasets.importers.alpaca_eval \\
      --models gpt-4-turbo-2024-04-09 claude-3-opus-20240229

  # Use existing clone & force re-import
  uv run python -m openjury.datasets.importers.alpaca_eval \\
      --repo-path /path/to/alpaca_eval --force
""",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Specific model names to import (default: all).",
    )
    parser.add_argument(
        "--repo-path", default=None,
        help="Path to existing alpaca_eval clone (default: auto-clone).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing cache entries.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List available models without importing.",
    )

    args = parser.parse_args()

    import_alpaca_eval_outputs(
        repo_path=Path(args.repo_path) if args.repo_path else None,
        models=args.models,
        force=args.force,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
