from __future__ import annotations

from pathlib import Path

from openjury.arena.config import AgreementConfig, JudgeConfig
from openjury.pipelines import agreement as agreement_pipeline


def _agreement_config(tmp_path: Path) -> AgreementConfig:
    return AgreementConfig(
        dataset="lmsys",
        judge=JudgeConfig(
            model="VLLM/Qwen/Qwen3-32B",
            gpus=4,
            mode="pairwise",
        ),
        output_dir=str(tmp_path / "agreement_out"),
        n_instructions=25,
        seed=7,
        balance_by="models",
        criteria="default",
    )


def test_run_agreement_annotate_uses_canonical_annotate_path(tmp_path, monkeypatch):
    config = _agreement_config(tmp_path)
    captured: dict[str, object] = {}

    def fake_run_annotate(annotate_config):
        captured["annotate_config"] = annotate_config
        return {
            "matches": [
                {
                    "sample_id": "s1",
                    "preference": 0.2,
                    "human_preference": 0.0,
                    "scores_a": {"helpfulness": 7.0},
                    "scores_b": {"helpfulness": 5.0},
                }
            ],
            "criterion_names": ["helpfulness"],
            "criteria_definition": {"name": "default"},
            "instruction_metadata": [{"source": "lmsys"}],
            "system_prompt": "judge prompt",
        }

    def fake_save(output_dir, payload, config_snapshot=None):
        captured["saved_output_dir"] = output_dir
        captured["saved_payload"] = payload
        captured["saved_snapshot"] = config_snapshot
        return str(Path(output_dir) / "agreement_annotations.json")

    monkeypatch.setattr(agreement_pipeline, "run_annotate", fake_run_annotate)
    monkeypatch.setattr(agreement_pipeline, "save_agreement_annotations", fake_save)
    monkeypatch.setattr(
        agreement_pipeline.AgreementConfig,
        "save",
        lambda self, path: captured.setdefault("config_path", str(path)),
    )

    result = agreement_pipeline.run_agreement(config, stage="annotate")

    annotate_config = captured["annotate_config"]
    assert annotate_config.pairing.source == "dataset_pairs"
    assert annotate_config.pairing.strategy == "dataset_defined"
    assert annotate_config.judge.model == config.judge.model
    assert annotate_config.challenger is None

    saved_payload = captured["saved_payload"]
    assert saved_payload["metadata"]["dataset"] == "lmsys"
    assert saved_payload["metadata"]["judge_model"] == config.judge.model
    assert saved_payload["metadata"]["swap_debiasing"] is True
    assert saved_payload["per_sample"][0]["human_pref"] == 0.0
    assert saved_payload["per_sample"][0]["judge_pref"] == 0.2
    assert result["per_sample"][0]["judge_pref"] == 0.2


def test_run_agreement_all_passes_normalized_payload_to_analysis(tmp_path, monkeypatch):
    config = _agreement_config(tmp_path)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        agreement_pipeline,
        "run_annotate",
        lambda annotate_config: {
            "matches": [
                {
                    "sample_id": "s1",
                    "preference": 0.8,
                    "human_preference": 1.0,
                    "scores_a": {"helpfulness": 4.0},
                    "scores_b": {"helpfulness": 8.0},
                }
            ],
            "criterion_names": ["helpfulness"],
            "criteria_definition": {"name": "default"},
            "instruction_metadata": [],
            "system_prompt": "judge prompt",
        },
    )
    monkeypatch.setattr(
        agreement_pipeline,
        "save_agreement_annotations",
        lambda output_dir, payload, config_snapshot=None: str(
            Path(output_dir) / "agreement_annotations.json"
        ),
    )
    monkeypatch.setattr(
        agreement_pipeline.AgreementConfig,
        "save",
        lambda self, path: None,
    )

    def fake_analyze(*, config, annotation_data):
        captured["analysis_payload"] = annotation_data
        return {"status": "ok", "n": len(annotation_data["per_sample"])}

    monkeypatch.setattr(agreement_pipeline, "analyze_agreement_stage", fake_analyze)

    result = agreement_pipeline.run_agreement(config, stage="all")

    assert result == {"status": "ok", "n": 1}
    payload = captured["analysis_payload"]
    assert payload["per_sample"][0]["human_pref"] == 1.0
    assert payload["per_sample"][0]["judge_pref"] == 0.8
    assert payload["metadata"]["judge_mode"] == "pairwise"
