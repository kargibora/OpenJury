"""Helpers for persisting intermediate annotation artifacts.

These artifacts are the boundary between the ``annotate`` and ``analyze``
stages in evaluation pipelines.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any


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
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
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
