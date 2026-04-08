from __future__ import annotations

from pathlib import Path

from openjury.arena.config import ArenaConfig, JudgeConfig, MatchmakerConfig, ModelEntry
from openjury.pipelines import arena as arena_pipeline


def _arena_config(tmp_path: Path) -> ArenaConfig:
    return ArenaConfig(
        dataset="alpaca-eval",
        models=[
            ModelEntry(name="VLLM/model-a", gpus=1),
            ModelEntry(name="VLLM/model-b", gpus=1),
        ],
        judge=JudgeConfig(
            model="VLLM/Qwen/Qwen3-32B",
            gpus=2,
            mode="pairwise",
        ),
        output_dir=str(tmp_path / "arena_out"),
        n_instructions=20,
        seed=9,
        criteria="default",
        matchmaker=MatchmakerConfig(strategy="balanced_random", n_matches=40),
        include_completions=True,
        include_raw_judge=True,
    )


def test_run_arena_annotate_uses_canonical_annotate_matchmaker(tmp_path, monkeypatch):
    config = _arena_config(tmp_path)
    captured: dict[str, object] = {}

    def fake_run_annotate(annotate_config, *, persist=True):
        captured["annotate_config"] = annotate_config
        captured["persist"] = persist
        return {
            "metadata": {
                "models": ["VLLM/model-a", "VLLM/model-b"],
                "dataset_cache_key": "alpaca-eval",
                "n_instructions": 20,
            },
            "criterion_names": ["helpfulness"],
            "criteria_definition": {"helpfulness": {"weight": 1.0}},
            "instruction_metadata": [{"split": "test"}],
            "model_scores": {},
            "matches": [
                {
                    "model_a": "VLLM/model-a",
                    "model_b": "VLLM/model-b",
                    "instruction_index": 0,
                    "scores_a": {"helpfulness": 7.0},
                    "scores_b": {"helpfulness": 6.0},
                    "preference": 0.4,
                    "instruction_id": "0",
                    "instruction_metadata": {"split": "test"},
                }
            ],
            "system_prompt": "judge prompt",
        }

    def fake_save(output_dir, payload, config_snapshot=None):
        captured["saved_output_dir"] = output_dir
        captured["saved_payload"] = payload
        captured["saved_snapshot"] = config_snapshot
        return str(Path(output_dir) / "arena_annotations.json")

    monkeypatch.setattr(arena_pipeline, "run_annotate", fake_run_annotate)
    monkeypatch.setattr(arena_pipeline, "save_arena_annotations", fake_save)
    monkeypatch.setattr(
        arena_pipeline.ArenaConfig,
        "save",
        lambda self, path: captured.setdefault("config_path", str(path)),
    )

    result = arena_pipeline.run_arena(config, stage="annotate")

    assert result is None
    annotate_config = captured["annotate_config"]
    assert annotate_config.pairing.source == "matchmaker"
    assert annotate_config.pairing.strategy == "balanced_random"
    assert annotate_config.pairing.n_matches == 40
    assert [model.name for model in annotate_config.models] == [
        "VLLM/model-a",
        "VLLM/model-b",
    ]
    assert captured["persist"] is False

    saved_payload = captured["saved_payload"]
    assert saved_payload["metadata"]["matchmaker"] == "balanced_random"
    assert saved_payload["metadata"]["judge_model"] == config.judge.model
    assert saved_payload["matches"][0]["preference"] == 0.4
    assert saved_payload["system_prompt"] == "judge prompt"
