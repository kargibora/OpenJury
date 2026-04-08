"""ComparIA human-preference dataset loader.

Converts the ComparIA (ministere-culture/comparia-votes) dataset into
:class:`EvalSample` objects with completions and human preference labels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from openjury._logging import logger
from openjury.datasets.registry import DatasetRegistry
from openjury.datasets.schema import EvalDataset, EvalSample


@DatasetRegistry.register("comparia")
def load_comparia(
    *,
    n: int | None = None,
    language: str = "fr",
    single_turn_only: bool = True,
    seed: int = 42,
    **_kwargs,
) -> EvalDataset:
    """Load ComparIA human-preference samples.

    Metadata per sample:
        - ``lang`` — detected language (ISO-639-1).
        - ``source`` — always ``"comparia"``.
        - ``model_a`` / ``model_b`` — the two models being compared.
        - ``original_question_id`` — original dataset ID.

    Each sample carries:
        - ``completions``: ``{"model_a_name": text, "model_b_name": text}``
        - ``human_pref``: 0.0 = A wins, 0.5 = tie, 1.0 = B wins

    Args:
        n: Maximum number of samples.
        language: ISO-639-1 code for language filtering.
        single_turn_only: Drop multi-turn conversations.
        seed: Random seed for sub-sampling.

    Returns:
        :class:`EvalDataset` with completions and human preferences.
    """
    from huggingface_hub import snapshot_download

    logger.info("Loading ComparIA dataset …")
    repo_path = snapshot_download(
        repo_id="ministere-culture/comparia-votes",
        repo_type="dataset",
    )

    parquets = list(Path(repo_path).rglob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(
            f"No parquet files found in {repo_path}."
        )
    df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)

    # ── Filter single-turn ───────────────────────────────────────
    if single_turn_only and "conversation_a" in df.columns:
        df = df[df["conversation_a"].apply(lambda c: len(c) <= 2)]

    # ── Language filter ──────────────────────────────────────────
    if language:
        try:
            from fast_langdetect import detect_language

            df["_lang"] = df["conversation_a"].apply(
                lambda c: detect_language(c[0]["content"]).lower()
                if len(c) > 0 else ""
            )
            before = len(df)
            df = df[df["_lang"] == language]
            logger.info(
                "Language filter (%s): %d → %d samples",
                language, before, len(df),
            )
        except ImportError:
            logger.warning(
                "fast-langdetect not installed — skipping language filter"
            )

    # ── Sub-sample ───────────────────────────────────────────────
    if n is not None and len(df) > n:
        df = df.sample(n=n, random_state=seed)

    # ── Build samples ────────────────────────────────────────────
    samples: list[EvalSample] = []
    for row_idx, (_, row) in enumerate(df.iterrows()):
        conv_a = row.get("conversation_a", [])
        conv_b = row.get("conversation_b", [])
        instruction = conv_a[0]["content"] if len(conv_a) > 0 else ""
        comp_a = conv_a[1]["content"] if len(conv_a) > 1 else ""
        comp_b = conv_b[1]["content"] if len(conv_b) > 1 else ""

        model_a_name = str(row.get("model_a_name", ""))
        model_b_name = str(row.get("model_b_name", ""))
        chosen = row.get("chosen_model_name")
        both_equal = row.get("both_equal", False)
        # both_equal may be a numpy scalar/array — coerce to plain bool
        if isinstance(both_equal, (np.ndarray, np.generic)):
            both_equal = bool(np.any(both_equal))
        else:
            both_equal = bool(both_equal)

        if both_equal or chosen is None or (isinstance(chosen, float) and np.isnan(chosen)):
            pref = 0.5
        elif str(chosen) == model_a_name:
            pref = 0.0
        elif str(chosen) == model_b_name:
            pref = 1.0
        else:
            pref = 0.5  # fallback

        question_id = str(row.get("id", row_idx))

        samples.append(EvalSample(
            instruction=instruction,
            instruction_id=f"comparia-{question_id}",
            metadata={
                "lang": language,
                "source": "comparia",
                "model_a": model_a_name,
                "model_b": model_b_name,
                "original_question_id": question_id,
            },
            completions={model_a_name: comp_a, model_b_name: comp_b},
            human_pref=pref,
        ))

    logger.info("Loaded %d ComparIA samples (lang=%s)", len(samples), language)

    return EvalDataset(
        name="comparia",
        samples=samples,
        metadata_schema={
            "lang": "Detected language (ISO-639-1)",
            "source": "Dataset source (always 'comparia')",
            "model_a": "Name of model A",
            "model_b": "Name of model B",
            "original_question_id": "Original ComparIA record ID",
        },
    )
