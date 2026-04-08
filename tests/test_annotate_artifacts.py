from __future__ import annotations

from openjury.pipelines.annotate import (
    load_annotate_annotations,
    save_annotate_annotations,
)


def test_annotate_artifact_roundtrip(tmp_path):
    payload = {
        "metadata": {"dataset": "lmsys", "models": ["A", "B"]},
        "matches": [],
        "model_scores": {},
    }
    snap = {"dataset": "lmsys", "challenger": {"name": "A"}}

    path = save_annotate_annotations(tmp_path, payload, config_snapshot=snap)
    data = load_annotate_annotations(tmp_path)

    assert path.name == "annotate_annotations.json"
    assert data["task"] == "annotate"
    assert data["config_snapshot"] == snap
    assert data["metadata"]["dataset"] == "lmsys"
