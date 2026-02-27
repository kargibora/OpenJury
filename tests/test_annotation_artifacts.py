from __future__ import annotations

import json

from openjury.pipelines.annotate import (
    load_annotation_artifact,
    load_arena_annotations,
    save_arena_annotations,
)


def test_annotation_artifact_contract_fields_are_persisted(tmp_path):
    payload = {
        "metadata": {"dataset": "alpaca-eval"},
        "matches": [],
        "model_scores": {},
    }
    snap = {"dataset": "alpaca-eval", "n_instructions": 10}

    path = save_arena_annotations(tmp_path, payload, config_snapshot=snap)
    data = load_arena_annotations(tmp_path)

    assert path.name == "arena_annotations.json"
    assert data["task"] == "arena"
    assert data["schema_version"] == "2.0"
    assert data["created_at"]
    assert data["config_snapshot"] == snap
    assert data["metadata"]["dataset"] == "alpaca-eval"


def test_annotation_artifact_backward_compat_with_version_v1(tmp_path):
    path = tmp_path / "arena_annotations.json"
    legacy = {
        "task": "arena",
        "version": "1",
        "metadata": {"dataset": "alpaca-eval"},
        "matches": [],
        "model_scores": {},
    }
    path.write_text(json.dumps(legacy), encoding="utf-8")

    data = load_annotation_artifact(
        output_dir=tmp_path,
        filename="arena_annotations.json",
        expected_task="arena",
    )
    assert data["schema_version"] == "1"
    assert data["config_snapshot"] == {}
    assert data["created_at"] == ""

