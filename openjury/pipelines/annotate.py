"""Helpers for persisting intermediate annotation artifacts.

These artifacts are the boundary between the ``annotate`` and ``analyze``
stages in evaluation pipelines.
"""

from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any


_DROP = object()


def _sanitize_json_value(value: Any) -> Any:
    """Recursively drop NaN values so persisted JSON stays strict and clean."""
    try:
        if math.isnan(value):  # handles float + numpy float scalars
            return _DROP
    except (TypeError, ValueError):
        pass

    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            sanitized = _sanitize_json_value(item)
            if sanitized is not _DROP:
                cleaned[key] = sanitized
        return cleaned

    if isinstance(value, list):
        cleaned_list: list[Any] = []
        for item in value:
            sanitized = _sanitize_json_value(item)
            if sanitized is not _DROP:
                cleaned_list.append(sanitized)
        return cleaned_list

    if isinstance(value, tuple):
        cleaned_tuple: list[Any] = []
        for item in value:
            sanitized = _sanitize_json_value(item)
            if sanitized is not _DROP:
                cleaned_tuple.append(sanitized)
        return cleaned_tuple

    return value


def _artifact_path(output_dir: str | Path, filename: str) -> Path:
    path = Path(output_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path / filename


def save_annotation_artifact(
    *,
    output_dir: str | Path,
    filename: str,
    task: str,
    payload: dict[str, Any],
    config_snapshot: dict[str, Any] | None = None,
) -> Path:
    """Save an intermediate annotation artifact as JSON."""
    path = _artifact_path(output_dir, filename)
    data = {
        "task": task,
        "schema_version": "2.0",
        "created_at": datetime.now().isoformat(),
        "config_snapshot": config_snapshot or {},
        **payload,
    }
    data = _sanitize_json_value(data)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, allow_nan=False)
    return path


def load_annotation_artifact(
    *,
    output_dir: str | Path,
    filename: str,
    expected_task: str,
) -> dict[str, Any]:
    """Load and validate an annotation artifact."""
    path = _artifact_path(output_dir, filename)
    if not path.exists():
        raise FileNotFoundError(
            f"Annotation artifact not found: {path}. Run the annotate stage first."
        )
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    task = data.get("task")
    if task != expected_task:
        raise ValueError(
            f"Expected {expected_task!r} annotation artifact, found {task!r} at {path}"
        )
    # Backward compatibility: older artifacts used {"version": "1"} and had no
    # config_snapshot/created_at fields.
    if "schema_version" not in data:
        data["schema_version"] = str(data.get("version", "1"))
    data.setdefault("config_snapshot", {})
    data.setdefault("created_at", "")
    return data


def save_agreement_annotations(
    output_dir: str | Path,
    payload: dict[str, Any],
    *,
    config_snapshot: dict[str, Any] | None = None,
) -> Path:
    return save_annotation_artifact(
        output_dir=output_dir,
        filename="agreement_annotations.json",
        task="agreement",
        payload=payload,
        config_snapshot=config_snapshot,
    )


def load_agreement_annotations(output_dir: str | Path) -> dict[str, Any]:
    return load_annotation_artifact(
        output_dir=output_dir,
        filename="agreement_annotations.json",
        expected_task="agreement",
    )


def save_arena_annotations(
    output_dir: str | Path,
    payload: dict[str, Any],
    *,
    config_snapshot: dict[str, Any] | None = None,
) -> Path:
    return save_annotation_artifact(
        output_dir=output_dir,
        filename="arena_annotations.json",
        task="arena",
        payload=payload,
        config_snapshot=config_snapshot,
    )


def load_arena_annotations(output_dir: str | Path) -> dict[str, Any]:
    return load_annotation_artifact(
        output_dir=output_dir,
        filename="arena_annotations.json",
        expected_task="arena",
    )


def save_annotate_annotations(
    output_dir: str | Path,
    payload: dict[str, Any],
    *,
    config_snapshot: dict[str, Any] | None = None,
) -> Path:
    return save_annotation_artifact(
        output_dir=output_dir,
        filename="annotate_annotations.json",
        task="annotate",
        payload=payload,
        config_snapshot=config_snapshot,
    )


def load_annotate_annotations(output_dir: str | Path) -> dict[str, Any]:
    return load_annotation_artifact(
        output_dir=output_dir,
        filename="annotate_annotations.json",
        expected_task="annotate",
    )
