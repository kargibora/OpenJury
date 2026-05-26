"""LMSYS / LM Arena human-preference dataset loaders.

Supports both the original 100k preference dataset and the newer 140k
variant under separate dataset names:

- ``lmsys`` -> ``lmarena-ai/arena-human-preference-100k``
- ``lmsys-140k`` -> ``lmarena-ai/arena-human-preference-140k``
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from openjury._logging import logger
from openjury.datasets.registry import DatasetRegistry
from openjury.datasets.schema import EvalDataset, EvalSample

_REPO_100K = "lmarena-ai/arena-human-preference-100k"
_REPO_140K = "lmarena-ai/arena-human-preference-140k"

_LANG_NAME_TO_ISO = {
    "english": "en",
    "russian": "ru",
    "chinese": "zh",
    "french": "fr",
    "german": "de",
    "spanish": "es",
    "italian": "it",
    "portuguese": "pt",
    "japanese": "ja",
    "korean": "ko",
    "arabic": "ar",
    "dutch": "nl",
    "polish": "pl",
    "turkish": "tr",
    "vietnamese": "vi",
    "czech": "cs",
    "romanian": "ro",
    "ukrainian": "uk",
    "greek": "el",
    "hebrew": "he",
    "hindi": "hi",
    "indonesian": "id",
    "persian": "fa",
    "thai": "th",
    "swedish": "sv",
    "danish": "da",
    "finnish": "fi",
    "norwegian": "no",
    "hungarian": "hu",
}


def _normalize_lang(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in _LANG_NAME_TO_ISO:
        return _LANG_NAME_TO_ISO[lowered]
    return lowered


def _flatten_content(content: Any) -> str:
    """Flatten LM Arena content blocks into plain text."""
    if content is None:
        return ""
    if hasattr(content, "tolist"):
        content = content.tolist()
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, (list, tuple)):
        parts = [_flatten_content(item) for item in content]
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"].strip()
        if "content" in content:
            return _flatten_content(content["content"])
        if isinstance(content.get("value"), str):
            return content["value"].strip()
    return ""


def _extract_turn_text(turn: Any) -> str:
    if isinstance(turn, dict):
        return _flatten_content(turn.get("content"))
    return _flatten_content(turn)


def _extract_instruction(conv: list[Any]) -> str:
    for turn in conv:
        text = _extract_turn_text(turn)
        if text:
            return text
    return ""


def _extract_completion(conv: list[Any]) -> str:
    if len(conv) > 1:
        text = _extract_turn_text(conv[1])
        if text:
            return text
    for turn in conv[1:]:
        text = _extract_turn_text(turn)
        if text:
            return text
    return ""


def _winner_to_pref(winner: Any) -> float:
    label = str(winner or "tie").lower()
    if "model_a" in label:
        return 0.0
    if "model_b" in label:
        return 1.0
    return 0.5


def _hf_hub_offline() -> bool:
    """Return True when HF-backed loaders should stay strictly local."""
    for key in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "HF_HUB_LOCAL_FILES_ONLY",
    ):
        value = os.environ.get(key)
        if value and value.strip().lower() not in {"0", "false", "no", "off"}:
            return True
    return False


def _load_lmsys_repo(
    *,
    repo_id: str,
    dataset_name: str,
    source_name: str,
    n: int | None = None,
    language: str | None = None,
    single_turn_only: bool = True,
    seed: int = 42,
) -> EvalDataset:
    from huggingface_hub import snapshot_download

    logger.info("Loading %s from %s …", dataset_name, repo_id)
    repo_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns="*.parquet",
        local_files_only=_hf_hub_offline(),
    )

    parquets = list(Path(repo_path).rglob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquet files found in {repo_path}.")
    df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)

    if single_turn_only and "conversation_a" in df.columns:
        df = df[df["conversation_a"].apply(lambda c: len(c) <= 2)]

    if language and "language" in df.columns:
        lang_code = _normalize_lang(language)
        before = len(df)
        df = df[df["language"].apply(_normalize_lang) == lang_code]
        logger.info(
            "Language filter (%s): %d → %d samples",
            language,
            before,
            len(df),
        )

    if n is not None and len(df) > n:
        df = df.sample(n=n, random_state=seed)

    def _to_list(val):
        """Safely convert a value to a Python list (handles numpy arrays)."""
        if val is None:
            return []
        if hasattr(val, "tolist"):  # numpy array
            return val.tolist()
        if isinstance(val, list):
            return val
        return list(val) if hasattr(val, "__iter__") else []

    samples: list[EvalSample] = []
    dropped_empty_rows = 0
    for row_idx, (_, row) in enumerate(df.iterrows()):
        conv_a = _to_list(row.get("conversation_a"))
        conv_b = _to_list(row.get("conversation_b"))

        instruction = _extract_instruction(conv_a)
        comp_a = _extract_completion(conv_a)
        comp_b = _extract_completion(conv_b)

        if not instruction.strip() or not comp_a.strip() or not comp_b.strip():
            dropped_empty_rows += 1
            continue

        model_a = str(row.get("model_a", ""))
        model_b = str(row.get("model_b", ""))
        sample_id = str(row.get("question_id") or row.get("id") or row_idx)
        row_lang = _normalize_lang(row.get("language"))

        samples.append(
            EvalSample(
                instruction=instruction,
                instruction_id=f"{dataset_name}-{sample_id}",
                metadata={
                    "lang": row_lang,
                    "source": source_name,
                    "model_a": model_a,
                    "model_b": model_b,
                    "original_question_id": sample_id,
                },
                completions={model_a: comp_a, model_b: comp_b},
                human_pref=_winner_to_pref(row.get("winner")),
            )
        )

    if dropped_empty_rows:
        logger.warning(
            "Dropped %d %s rows with missing instruction/completion text",
            dropped_empty_rows,
            dataset_name,
        )
    logger.info("Loaded %d %s samples (lang=%s)", len(samples), dataset_name, language)

    return EvalDataset(
        name=dataset_name,
        samples=samples,
        metadata_schema={
            "lang": "Detected language (ISO-639-1 when available)",
            "source": f"Dataset source (always '{source_name}')",
            "model_a": "Name of model A",
            "model_b": "Name of model B",
            "original_question_id": "Original Arena question/evaluation ID",
        },
    )


@DatasetRegistry.register("lmsys")
def load_lmsys(
    *,
    n: int | None = None,
    language: str | None = None,
    single_turn_only: bool = True,
    seed: int = 42,
    **_kwargs,
) -> EvalDataset:
    """Load LMSYS Arena 100k human-preference samples."""
    return _load_lmsys_repo(
        repo_id=_REPO_100K,
        dataset_name="lmsys",
        source_name="lmsys",
        n=n,
        language=language,
        single_turn_only=single_turn_only,
        seed=seed,
    )


@DatasetRegistry.register("lmsys-140k")
def load_lmsys_140k(
    *,
    n: int | None = None,
    language: str | None = None,
    single_turn_only: bool = True,
    seed: int = 42,
    **_kwargs,
) -> EvalDataset:
    """Load LM Arena 140k human-preference samples."""
    return _load_lmsys_repo(
        repo_id=_REPO_140K,
        dataset_name="lmsys-140k",
        source_name="lmsys-140k",
        n=n,
        language=language,
        single_turn_only=single_turn_only,
        seed=seed,
    )
