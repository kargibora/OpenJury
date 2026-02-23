"""Pre-download all OpenJury datasets for offline use on compute nodes.

Run this once from a **login node** (with internet access) before
submitting SLURM jobs::

    uv run python -m openjury.datasets.download          # all datasets
    uv run python -m openjury.datasets.download lmsys    # single dataset
    uv run python -m openjury.datasets.download --list   # show available

Datasets are cached by HuggingFace Hub / ``datasets`` library in the
standard ``$HF_HOME`` cache directory.  Subsequent ``load_dataset()``
and ``snapshot_download()`` calls work offline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from openjury._logging import logger


# ═════════════════════════════════════════════════════════════════════
#  Dataset download registry
# ═════════════════════════════════════════════════════════════════════

# Each entry: (name, description, download_fn)
# download_fn() should download everything needed for that dataset.

_DATASETS: dict[str, tuple[str, callable]] = {}


def _register(name: str, description: str):
    """Decorator to register a dataset download function."""
    def wrapper(fn):
        _DATASETS[name] = (description, fn)
        return fn
    return wrapper


# ─────────────────────────────────────────────────────────────────────
#  Individual dataset downloaders
# ─────────────────────────────────────────────────────────────────────


@_register("lmsys", "LMSys Arena 100k human preferences")
def _download_lmsys():
    from huggingface_hub import snapshot_download

    repo_id = "lmarena-ai/arena-human-preference-100k"
    logger.info("Downloading %s ...", repo_id)
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns="*.parquet",
    )
    n_files = len(list(Path(path).rglob("*.parquet")))
    logger.info("  ✅ %s — %d parquet file(s) cached at %s", repo_id, n_files, path)


@_register("comparia", "ComparIA French human preferences")
def _download_comparia():
    from huggingface_hub import snapshot_download

    repo_id = "ministere-culture/comparia-votes"
    logger.info("Downloading %s ...", repo_id)
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
    )
    n_files = len(list(Path(path).rglob("*.parquet")))
    logger.info("  ✅ %s — %d parquet file(s) cached at %s", repo_id, n_files, path)


@_register("m-arena-hard", "Multilingual Arena-Hard v1 (23 languages)")
def _download_m_arenahard_v1():
    from openjury.datasets.loaders.m_arenahard import download_dataset, _REPO_V1

    download_dataset(_REPO_V1)


@_register("m-arena-hard-v2", "Multilingual Arena-Hard v2 (23 languages)")
def _download_m_arenahard_v2():
    from openjury.datasets.loaders.m_arenahard import download_dataset, _REPO_V2

    download_dataset(_REPO_V2)


@_register("alpaca-eval", "AlpacaEval instruction dataset")
def _download_alpaca_eval():
    from openjury.utils import download_hf, data_root

    logger.info("Downloading alpaca-eval tables ...")
    local_path = data_root / "tables"
    download_hf(name="alpaca-eval", local_path=local_path)
    logger.info("  ✅ alpaca-eval cached at %s", local_path)


@_register("arena-hard", "Arena-Hard instruction dataset")
def _download_arena_hard():
    from openjury.utils import download_hf, data_root

    logger.info("Downloading arena-hard tables ...")
    local_path = data_root / "tables"
    download_hf(name="arena-hard", local_path=local_path)
    logger.info("  ✅ arena-hard cached at %s", local_path)


@_register("contexts", "Multilingual contexts for completion")
def _download_contexts():
    from huggingface_hub import snapshot_download
    from openjury.utils import data_root

    repo_id = "geoalgo/multilingual-contexts-to-be-completed"
    logger.info("Downloading %s ...", repo_id)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns="*",
        local_dir=data_root / "contexts",
        force_download=False,
    )
    logger.info("  ✅ contexts cached at %s", data_root / "contexts")


# ═════════════════════════════════════════════════════════════════════
#  Public API
# ═════════════════════════════════════════════════════════════════════


def download(names: list[str] | None = None) -> None:
    """Download one or more datasets.

    Args:
        names: Dataset names to download. ``None`` = all registered.
    """
    targets = names if names else list(_DATASETS.keys())

    unknown = [n for n in targets if n not in _DATASETS]
    if unknown:
        logger.error(
            "Unknown dataset(s): %s\nAvailable: %s",
            ", ".join(unknown),
            ", ".join(_DATASETS.keys()),
        )
        sys.exit(1)

    logger.info("═══════════════════════════════════════════════════")
    logger.info("  OpenJury — Dataset Pre-Download")
    logger.info("  Datasets: %s", ", ".join(targets))
    logger.info("═══════════════════════════════════════════════════")
    logger.info("")

    succeeded: list[str] = []
    failed: list[str] = []

    for name in targets:
        desc, fn = _DATASETS[name]
        logger.info("── [%s] %s ──", name, desc)
        try:
            fn()
            succeeded.append(name)
        except Exception as e:
            logger.error("  ❌ %s failed: %s", name, e)
            failed.append(name)
        logger.info("")

    # ── Summary ──────────────────────────────────────────────────
    logger.info("═══════════════════════════════════════════════════")
    logger.info("  Download Summary")
    logger.info("═══════════════════════════════════════════════════")
    logger.info("  ✅ Succeeded: %d — %s", len(succeeded), ", ".join(succeeded) or "none")
    if failed:
        logger.info("  ❌ Failed:    %d — %s", len(failed), ", ".join(failed))
    logger.info("")
    logger.info("  Datasets are cached for offline use on compute nodes.")
    logger.info("═══════════════════════════════════════════════════")


# ═════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        prog="openjury-download",
        description="Pre-download OpenJury datasets for offline compute nodes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n\n"
            "  # Download everything\n"
            "  uv run python -m openjury.datasets.download\n\n"
            "  # Download specific datasets\n"
            "  uv run python -m openjury.datasets.download lmsys comparia\n\n"
            "  # List available datasets\n"
            "  uv run python -m openjury.datasets.download --list\n"
        ),
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        default=None,
        help="Dataset(s) to download. Default: all.",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available datasets and exit.",
    )

    args = parser.parse_args()

    if args.list:
        print("Available datasets:\n")
        for name, (desc, _) in _DATASETS.items():
            print(f"  {name:20s} {desc}")
        print(f"\nTotal: {len(_DATASETS)} datasets")
        return

    download(args.datasets or None)


if __name__ == "__main__":
    main()
