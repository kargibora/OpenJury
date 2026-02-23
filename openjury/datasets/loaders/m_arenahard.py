"""Multilingual Arena-Hard dataset loaders (v1 and v2).

Two registered datasets:

- ``m-arena-hard`` — `CohereLabs/m-ArenaHard`_ (v1, 500 prompts × 23 langs).
  Each language subset has columns: ``question_id``, ``cluster``,
  ``category``, ``prompt``.

- ``m-arena-hard-v2`` — `CohereLabs/m-ArenaHard-v2.0`_ (v2, 498 prompts × 23 langs).
  Each language subset has columns: ``question_id``, ``category``,
  ``subcategory``, ``prompt``, ``language``.

Both support dynamic language suffixes via the registry, e.g.
``m-arena-hard-en``, ``m-arena-hard-v2-fr``, ``m-arena-hard-EU``.

Pre-download on login node::

    python -m openjury.datasets.loaders.m_arenahard --download
    python -m openjury.datasets.loaders.m_arenahard --download --version v2

.. _CohereLabs/m-ArenaHard: https://huggingface.co/datasets/CohereLabs/m-ArenaHard
.. _CohereLabs/m-ArenaHard-v2.0: https://huggingface.co/datasets/CohereLabs/m-ArenaHard-v2.0
"""

from __future__ import annotations

import argparse

import pandas as pd

from openjury._logging import logger
from openjury.datasets.registry import DatasetRegistry
from openjury.datasets.schema import EvalDataset, EvalSample


# ── Constants ────────────────────────────────────────────────────────

_REPO_V1 = "CohereLabs/m-ArenaHard"
_REPO_V2 = "CohereLabs/m-ArenaHard-v2.0"

_EU_LANGUAGES = [
    "cs", "de", "el", "en", "es", "fr", "it", "nl", "pl", "pt", "ro", "uk",
]

_ALL_LANGUAGES = [
    "ar", "cs", "de", "el", "en", "es", "fa", "fr", "he", "hi",
    "id", "it", "ja", "ko", "nl", "pl", "pt", "ro", "ru",
    "tr", "uk", "vi", "zh",
]


# ── Shared helpers ───────────────────────────────────────────────────


def _resolve_languages(language: str | None) -> list[str]:
    """Return the list of language codes to load.

    Args:
        language: ``None`` (all), ``"EU"`` (European subset), or a
            single ISO-639-1 code.

    Returns:
        Sorted list of language codes.

    Raises:
        ValueError: If *language* is not recognised.
    """
    if language is None:
        return list(_ALL_LANGUAGES)
    if language == "EU":
        return list(_EU_LANGUAGES)
    if language not in _ALL_LANGUAGES:
        raise ValueError(
            f"Unknown language '{language}'. "
            f"Valid: {_ALL_LANGUAGES + ['EU']}"
        )
    return [language]


def download_dataset(repo_id: str, languages: list[str] | None = None) -> str:
    """Pre-download a multilingual Arena-Hard dataset to the HF cache.

    Call this from a login node with internet access. Subsequent
    ``load_dataset()`` calls (even on offline compute nodes) will
    use the cached data.

    Args:
        repo_id: HuggingFace repo ID (e.g. ``CohereLabs/m-ArenaHard``).
        languages: List of language codes to download. ``None`` = all.

    Returns:
        Path to the cached dataset directory.
    """
    from datasets import load_dataset

    langs = languages or list(_ALL_LANGUAGES)
    logger.info("Pre-downloading %s — %d language subset(s)...", repo_id, len(langs))

    cached_count = 0
    for lang in langs:
        try:
            ds = load_dataset(repo_id, lang, split="test")
            logger.info("  ✅ %s — %d rows", lang, len(ds))
            cached_count += 1
        except Exception as e:
            logger.warning("  ⚠️  %s — failed: %s", lang, e)

    logger.info("Cached %d/%d subsets for %s", cached_count, len(langs), repo_id)
    return repo_id


def _load_hf_subsets(
    repo_id: str,
    languages: list[str],
) -> pd.DataFrame:
    """Load one or more language subsets via ``datasets.load_dataset``.

    Each HF subset corresponds to a language code (e.g. ``"en"``).
    Returns a concatenated DataFrame with an added ``lang`` column.

    Works offline if the dataset was pre-downloaded via
    :func:`download_dataset`.
    """
    from datasets import load_dataset

    dfs: list[pd.DataFrame] = []
    for lang in languages:
        try:
            ds = load_dataset(repo_id, lang, split="test")
        except Exception:
            logger.warning("  ⚠️  Subset '%s' not found in %s — skipping", lang, repo_id)
            continue
        df = ds.to_pandas()
        df["lang"] = lang
        dfs.append(df)

    if not dfs:
        raise ValueError(
            f"No data found for languages={languages} in {repo_id}. "
            f"Did you pre-download? Run: python -m openjury.datasets.loaders.m_arenahard --download"
        )

    return pd.concat(dfs, ignore_index=True)


# ═════════════════════════════════════════════════════════════════════
#  m-ArenaHard v1
# ═════════════════════════════════════════════════════════════════════


@DatasetRegistry.register("m-arena-hard")
def load_m_arenahard(
    *,
    n: int | None = None,
    language: str | None = None,
    **_kwargs,
) -> EvalDataset:
    """Load the multilingual Arena-Hard v1 dataset.

    HF repo: ``CohereLabs/m-ArenaHard`` (previously ``m_ArenaHard``).

    Columns per subset: ``question_id``, ``cluster``, ``category``, ``prompt``.
    Language is determined by the HF config/subset name.

    Metadata per sample:
        - ``lang`` — ISO-639-1 language code.
        - ``original_question_id`` — shared across translations.
        - ``cluster`` — topic cluster.
        - ``category`` — benchmark category.

    Args:
        n: Maximum number of samples.
        language: ``None`` = all, ``"EU"`` = European subset, or a
            single ISO-639-1 code (e.g. ``"en"``).

    Returns:
        :class:`EvalDataset` with rich language metadata.
    """
    languages = _resolve_languages(language)
    logger.info("Loading m-ArenaHard v1 — languages: %s", languages)

    df = _load_hf_subsets(_REPO_V1, languages)

    # Stable sort
    df.sort_values(["question_id", "lang"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    samples: list[EvalSample] = []
    for _, row in df.iterrows():
        qid = str(row["question_id"])
        lang = str(row["lang"])

        metadata: dict[str, str] = {
            "lang": lang,
            "original_question_id": qid,
        }
        if "cluster" in row and pd.notna(row["cluster"]):
            metadata["cluster"] = str(row["cluster"])
        if "category" in row and pd.notna(row["category"]):
            metadata["category"] = str(row["category"])

        samples.append(EvalSample(
            instruction=str(row["prompt"]),
            instruction_id=f"{qid}-{lang}",
            metadata=metadata,
        ))

    if n is not None:
        samples = samples[:n]

    dataset_name = "m-arena-hard"
    if language:
        dataset_name = f"m-arena-hard-{language}"

    langs_found = sorted({s.metadata["lang"] for s in samples})
    logger.info(
        "m-ArenaHard v1: %d samples, %d language(s): %s",
        len(samples), len(langs_found), langs_found,
    )

    return EvalDataset(
        name=dataset_name,
        samples=samples,
        metadata_schema={
            "lang": "ISO-639-1 language code",
            "original_question_id": "Original question ID (shared across translations)",
            "cluster": "Topic cluster",
            "category": "Benchmark category (e.g. arena-hard-v0.1)",
        },
    )


# ═════════════════════════════════════════════════════════════════════
#  m-ArenaHard v2
# ═════════════════════════════════════════════════════════════════════


@DatasetRegistry.register("m-arena-hard-v2")
def load_m_arenahard_v2(
    *,
    n: int | None = None,
    language: str | None = None,
    **_kwargs,
) -> EvalDataset:
    """Load the multilingual Arena-Hard v2.0 dataset.

    HF repo: ``CohereLabs/m-ArenaHard-v2.0``.

    Columns per subset: ``question_id``, ``category``, ``subcategory``,
    ``prompt``, ``language``.

    Metadata per sample:
        - ``lang`` — ISO-639-1 language code.
        - ``original_question_id`` — shared across translations.
        - ``category`` — prompt category from the original dataset.
        - ``subcategory`` — finer-grained prompt category (e.g. ``"coding"``).

    Args:
        n: Maximum number of samples.
        language: ``None`` = all, ``"EU"`` = European subset, or a
            single ISO-639-1 code (e.g. ``"ko"``).

    Returns:
        :class:`EvalDataset` with rich language metadata.
    """
    languages = _resolve_languages(language)
    logger.info("Loading m-ArenaHard v2 — languages: %s", languages)

    df = _load_hf_subsets(_REPO_V2, languages)

    # Stable sort
    df.sort_values(["question_id", "lang"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    samples: list[EvalSample] = []
    for _, row in df.iterrows():
        qid = str(row["question_id"])
        lang = str(row["lang"])

        metadata: dict[str, str] = {
            "lang": lang,
            "original_question_id": qid,
        }
        if "category" in row and pd.notna(row["category"]):
            metadata["category"] = str(row["category"])
        if "subcategory" in row and pd.notna(row["subcategory"]):
            metadata["subcategory"] = str(row["subcategory"])

        samples.append(EvalSample(
            instruction=str(row["prompt"]),
            instruction_id=f"{qid}-{lang}",
            metadata=metadata,
        ))

    if n is not None:
        samples = samples[:n]

    dataset_name = "m-arena-hard-v2"
    if language:
        dataset_name = f"m-arena-hard-v2-{language}"

    langs_found = sorted({s.metadata["lang"] for s in samples})
    logger.info(
        "m-ArenaHard v2: %d samples, %d language(s): %s",
        len(samples), len(langs_found), langs_found,
    )

    return EvalDataset(
        name=dataset_name,
        samples=samples,
        metadata_schema={
            "lang": "ISO-639-1 language code",
            "original_question_id": "Original question ID (shared across translations)",
            "category": "Prompt category from original dataset",
            "subcategory": "Finer-grained prompt category (e.g. coding)",
        },
    )


# ═════════════════════════════════════════════════════════════════════
#  CLI: pre-download
# ═════════════════════════════════════════════════════════════════════


def main():
    """CLI entry point for pre-downloading m-ArenaHard datasets."""
    parser = argparse.ArgumentParser(
        description="Pre-download multilingual Arena-Hard datasets to HF cache.",
    )
    parser.add_argument(
        "--version", choices=["v1", "v2", "both"], default="both",
        help="Which version to download. Default: both",
    )
    parser.add_argument(
        "--languages", nargs="*", default=None,
        help="Specific languages to download. Default: all 23.",
    )
    args = parser.parse_args()

    langs = args.languages

    if args.version in ("v1", "both"):
        download_dataset(_REPO_V1, langs)

    if args.version in ("v2", "both"):
        download_dataset(_REPO_V2, langs)

    logger.info("✅ Done. Datasets are cached for offline use.")


if __name__ == "__main__":
    main()