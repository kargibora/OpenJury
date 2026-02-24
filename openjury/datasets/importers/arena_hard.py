"""Import pre-existing Arena-Hard model answers into the CompletionCache.

The `lmarena/arena-hard-auto <https://github.com/lmarena/arena-hard-auto>`_
repository ships model answers as JSONL files under::

    data/arena-hard-v0.1/model_answer/{model_name}.jsonl

Each line is a JSON object with fields ``uid``, ``model``, ``messages``
(chat-format list), and ``tstamp``.  The **last message** in the
``messages`` list is the assistant response.

This importer:

1. Clones / updates the repo (sparse checkout of ``data/``) into
   ``$OPENJURY_DATA/arena_hard_repo/``.
2. Discovers all ``*.jsonl`` files under ``data/arena-hard-v0.1/model_answer/``.
3. For each model, matches entries to the canonical arena-hard dataset
   by ``uid`` ↔ ``question_id``.
4. Stores completions in the :class:`CompletionCache` under the
   **``ArenaHard/{model_name}``** namespace.

Usage::

    # Import all models
    uv run python -m openjury.datasets.importers.arena_hard

    # List available models without importing
    uv run python -m openjury.datasets.importers.arena_hard --dry-run

    # Import specific models only
    uv run python -m openjury.datasets.importers.arena_hard \\
        --models gpt-4-turbo-2024-04-09 claude-3-opus-20240229

    # Use an existing clone
    uv run python -m openjury.datasets.importers.arena_hard \\
        --repo-path /path/to/arena-hard-auto

After import, completions are available in the arena pipeline::

    {
        "models": [
            "ArenaHard/gpt-4-turbo-2024-04-09",
            "ArenaHard/claude-3-opus-20240229",
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
from openjury.cache.completions import cache
from openjury.datasets import load_dataset
from openjury.common.paths import data_root


# ═════════════════════════════════════════════════════════════════════
#  Constants
# ═════════════════════════════════════════════════════════════════════

REPO_URL = "https://github.com/lmarena/arena-hard-auto.git"
DATASET_NAME = "arena-hard"
MODEL_PREFIX = "ArenaHard"
MODEL_ANSWER_DIR = "data/arena-hard-v0.1/model_answer"


# ═════════════════════════════════════════════════════════════════════
#  Core import logic
# ═════════════════════════════════════════════════════════════════════


def _ensure_repo(repo_path: Path | None = None) -> Path:
    """Clone or update the arena-hard-auto repository.

    Args:
        repo_path: Existing local clone.  If ``None``, clones into
            ``$OPENJURY_DATA/arena_hard_repo/``.

    Returns:
        Path to the repository root.
    """
    if repo_path is not None:
        repo_path = Path(repo_path)
        model_dir = repo_path / MODEL_ANSWER_DIR
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"No {MODEL_ANSWER_DIR}/ directory in {repo_path}. "
                "Is this a valid arena-hard-auto clone?"
            )
        return repo_path

    repo_path = data_root / "arena_hard_repo"
    if (repo_path / ".git").is_dir():
        logger.info("Updating existing clone at %s …", repo_path)
        subprocess.run(
            ["git", "-C", str(repo_path), "pull", "--ff-only"],
            check=False,
            capture_output=True,
        )
    else:
        logger.info("Cloning %s → %s …", REPO_URL, repo_path)
        # Shallow clone — sparse checkout only data/ directory
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
        # Sparse-checkout the data directory
        subprocess.run(
            ["git", "-C", str(repo_path), "sparse-checkout", "set", "data"],
            check=True,
        )

    return repo_path


def discover_models(repo_path: Path) -> list[str]:
    """List all model names that have JSONL answer files.

    Args:
        repo_path: Path to the arena-hard-auto repository clone.

    Returns:
        Sorted list of model names (derived from JSONL filenames).
    """
    model_dir = repo_path / MODEL_ANSWER_DIR
    models = []
    for f in sorted(model_dir.iterdir()):
        if f.is_file() and f.suffix == ".jsonl":
            models.append(f.stem)
    return models


def _extract_completion(messages: list[dict[str, Any]]) -> str:
    """Extract the assistant completion from the messages list.

    The last message is the assistant response.  Its ``content`` may be
    either a plain string or a dict with an ``answer`` key (some models
    use the dict form).

    Args:
        messages: Chat-format message list from the JSONL entry.

    Returns:
        The completion text.
    """
    if not messages:
        return ""
    last_msg = messages[-1]
    content = last_msg.get("content", "")
    if isinstance(content, dict):
        return content.get("answer", str(content))
    return str(content)


def _build_uid_to_index(dataset_samples: list) -> dict[str, int]:
    """Build a mapping from question_id (uid) → row index.

    The arena-hard loader stores ``question_id`` in each sample's metadata.

    Args:
        dataset_samples: List of EvalSample from the arena-hard dataset.

    Returns:
        Dict mapping hex-hash ``question_id`` → positional index.
    """
    uid_map: dict[str, int] = {}
    for i, sample in enumerate(dataset_samples):
        qid = sample.metadata.get("question_id", "")
        if qid:
            uid_map[qid] = i
    return uid_map


def import_model(
    model_name: str,
    repo_path: Path,
    uid_to_index: dict[str, int],
    n_instructions: int,
    *,
    force: bool = False,
) -> bool:
    """Import a single model's completions into the cache.

    Args:
        model_name: JSONL filename stem (e.g. ``"gpt-3.5-turbo-0125"``).
        repo_path: Path to the arena-hard-auto repository clone.
        uid_to_index: Mapping from question_id (uid) → row index.
        n_instructions: Expected number of instructions in the dataset.
        force: Overwrite existing cache entries.

    Returns:
        ``True`` if completions were imported, ``False`` on error.
    """
    cache_model = f"{MODEL_PREFIX}/{model_name}"

    # ── Already cached? ──────────────────────────────────────────
    if not force and cache.exists(cache_model, DATASET_NAME):
        logger.info("  ✓ %s — already cached, skipping", cache_model)
        return True

    # ── Load JSONL ───────────────────────────────────────────────
    jsonl_path = repo_path / MODEL_ANSWER_DIR / f"{model_name}.jsonl"
    if not jsonl_path.exists():
        logger.warning("  ✗ %s — JSONL file not found", model_name)
        return False

    entries: list[dict[str, Any]] = []
    try:
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("  ✗ %s — failed to read JSONL: %s", model_name, exc)
        return False

    # ── Match by uid → index ─────────────────────────────────────
    completions: list[str] = [""] * n_instructions
    matched = 0
    missing_uids = 0

    for entry in entries:
        uid = entry.get("uid", "")
        messages = entry.get("messages", [])

        idx = uid_to_index.get(uid)
        if idx is None:
            missing_uids += 1
            if missing_uids <= 3:
                logger.warning(
                    "  ⚠ %s — uid %s not found in dataset", model_name, uid[:16],
                )
            continue

        completion = _extract_completion(messages)
        completions[idx] = completion
        matched += 1

    if missing_uids > 3:
        logger.warning(
            "  ⚠ %s — %d total UIDs not found in dataset",
            model_name, missing_uids,
        )

    if matched == 0:
        logger.warning("  ✗ %s — 0 entries matched, skipping", model_name)
        return False

    # ── Check coverage ───────────────────────────────────────────
    empty = sum(1 for c in completions if c == "")
    if empty > 0:
        logger.warning(
            "  ⚠ %s — %d/%d instructions have no completion",
            model_name, empty, n_instructions,
        )

    # ── Store in cache ───────────────────────────────────────────
    df = pd.DataFrame({
        "completion": completions,
        "instruction_index": list(range(n_instructions)),
    })
    cache.put(df, cache_model, DATASET_NAME)
    logger.info(
        "  ✓ %s — imported %d completions (%d matched)",
        cache_model, n_instructions, matched,
    )
    return True


def import_arena_hard_outputs(
    *,
    repo_path: Path | None = None,
    models: list[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, bool]:
    """Import Arena-Hard model answers into the CompletionCache.

    Args:
        repo_path: Path to an existing arena-hard-auto clone, or ``None``
            to auto-clone into ``$OPENJURY_DATA/arena_hard_repo/``.
        models: Specific model names to import.  ``None`` = all.
        force: Overwrite existing cache entries.
        dry_run: Only list available models, don't import.

    Returns:
        Dict mapping model name → success boolean.
    """
    repo = _ensure_repo(repo_path)
    available = discover_models(repo)

    logger.info("Found %d models with answers in %s", len(available), repo)

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

    # ── Load reference dataset once ──────────────────────────────
    logger.info("Loading reference arena-hard dataset …")
    ds = load_dataset(DATASET_NAME)
    uid_to_index = _build_uid_to_index(ds.samples)
    n_instructions = len(ds)
    logger.info(
        "Reference: %d instructions, %d with question_id",
        n_instructions, len(uid_to_index),
    )

    if not uid_to_index:
        logger.error(
            "No question_id metadata found in arena-hard dataset. "
            "Make sure the loader populates metadata['question_id']."
        )
        return {}

    # ── Import each model ────────────────────────────────────────
    results: dict[str, bool] = {}
    for i, model_name in enumerate(to_import, 1):
        logger.info("[%d/%d] Importing %s …", i, len(to_import), model_name)
        results[model_name] = import_model(
            model_name, repo, uid_to_index, n_instructions, force=force,
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
        prog="openjury.datasets.importers.arena_hard",
        description=(
            "Import pre-existing Arena-Hard model answers into the "
            "OpenJury CompletionCache with ArenaHard/ prefix."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:

  # List all available models
  uv run python -m openjury.datasets.importers.arena_hard --dry-run

  # Import all models
  uv run python -m openjury.datasets.importers.arena_hard

  # Import specific models
  uv run python -m openjury.datasets.importers.arena_hard \\
      --models gpt-4-turbo-2024-04-09 claude-3-opus-20240229

  # Use existing clone & force re-import
  uv run python -m openjury.datasets.importers.arena_hard \\
      --repo-path /path/to/arena-hard-auto --force
""",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Specific model names to import (default: all).",
    )
    parser.add_argument(
        "--repo-path", default=None,
        help="Path to existing arena-hard-auto clone (default: auto-clone).",
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

    import_arena_hard_outputs(
        repo_path=Path(args.repo_path) if args.repo_path else None,
        models=args.models,
        force=args.force,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
