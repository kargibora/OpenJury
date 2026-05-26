import json

from openjury.pipelines.annotate import save_annotation_artifact


def test_save_annotation_artifact_strips_nested_nan_values(tmp_path):
    path = save_annotation_artifact(
        output_dir=tmp_path,
        filename="artifact.json",
        task="agreement",
        payload={
            "per_sample": [
                {
                    "scores_a": {"adherence": 7.0, "clarity": float("nan")},
                    "scores_b": {"adherence": float("nan"), "clarity": 8.0},
                    "judge_pref": 0.5,
                }
            ]
        },
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    row = data["per_sample"][0]

    assert row["scores_a"] == {"adherence": 7.0}
    assert row["scores_b"] == {"clarity": 8.0}
