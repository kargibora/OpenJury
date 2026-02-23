"""LMSys Arena human-preference dataset loader.

Converts the LMSys Arena 100k dataset into :class:`EvalSample` objects
with pre-existing completions and human preference labels.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from openjury._logging import logger
from openjury.datasets.registry import DatasetRegistry
from openjury.datasets.schema import EvalDataset, EvalSample


@DatasetRegistry.register("lmsys")
def load_lmsys(
    *,
    n: int | None = None,
    language: str | None = None,
    single_turn_only: bool = True,
    seed: int = 42,
    **_kwargs,
) -> EvalDataset:
    """Load LMSys Arena 100k human-preference samples.

    Metadata per sample:
        - ``lang`` — detected language (ISO-639-1).
        - ``source`` — always ``"lmsys"``.
        - ``model_a`` / ``model_b`` — the two models being compared.
        - ``original_question_id`` — original dataset question ID.

    Each sample carries:
        - ``completions``: ``{"model_a_name": text, "model_b_name": text}``
        - ``human_pref``: 0.0 = A wins, 0.5 = tie, 1.0 = B wins

    Args:
        n: Maximum number of samples.
        language: ISO-639-1 code for language filtering (requires
            ``fast-langdetect``).
        single_turn_only: Drop multi-turn conversations.
        seed: Random seed for sub-sampling.

    Returns:
        :class:`EvalDataset` with completions and human preferences.
    """
    from huggingface_hub import snapshot_download

    logger.info("Loading LMSys Arena dataset …")
    repo_path = snapshot_download(
        repo_id="lmarena-ai/arena-human-preference-100k",
        repo_type="dataset",
        allow_patterns="*.parquet",
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
        # Map ISO-639-1 codes to the full names used in the dataset
        _LANG_MAP = {
            "en": "English", "ru": "Russian", "zh": "Chinese",
            "fr": "French", "de": "German", "es": "Spanish",
            "it": "Italian", "pt": "Portuguese", "ja": "Japanese",
            "ko": "Korean", "ar": "Arabic", "nl": "Dutch",
            "pl": "Polish", "tr": "Turkish", "vi": "Vietnamese",
            "cs": "Czech", "ro": "Romanian", "uk": "Ukrainian",
            "el": "Greek", "he": "Hebrew", "hi": "Hindi",
            "id": "Indonesian", "fa": "Persian", "th": "Thai",
            "sv": "Swedish", "da": "Danish", "fi": "Finnish",
            "no": "Norwegian", "hu": "Hungarian",
        }
        if "language" in df.columns:
            lang_full = _LANG_MAP.get(language, language)
            before = len(df)
            df = df[df["language"].str.lower() == lang_full.lower()]
            logger.info(
                "Language filter (%s → %s): %d → %d samples",
                language, lang_full, before, len(df),
            )
        else:
            logger.warning(
                "No 'language' column in dataset — skipping language filter"
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

        winner = str(row.get("winner", "tie")).lower()
        if "model_a" in winner:
            pref = 0.0
        elif "model_b" in winner:
            pref = 1.0
        else:
            pref = 0.5

        model_a = str(row.get("model_a", ""))
        model_b = str(row.get("model_b", ""))
        question_id = str(row.get("question_id", row_idx))

        # Use per-row language from dataset, not the filter parameter
        row_lang = str(row.get("language", "")).strip() if "language" in row.index else None

        samples.append(EvalSample(
            instruction=instruction,
            instruction_id=f"lmsys-{question_id}",
            metadata={
                "lang": row_lang,
                "source": "lmsys",
                "model_a": model_a,
                "model_b": model_b,
                "original_question_id": question_id,
            },
            completions={model_a: comp_a, model_b: comp_b},
            human_pref=pref,
        ))

    logger.info("Loaded %d LMSys samples (lang=%s)", len(samples), language)

    return EvalDataset(
        name="lmsys",
        samples=samples,
        metadata_schema={
            "lang": "Detected language (ISO-639-1)",
            "source": "Dataset source (always 'lmsys')",
            "model_a": "Name of model A",
            "model_b": "Name of model B",
            "original_question_id": "Original LMSys question ID",
        },
    )
