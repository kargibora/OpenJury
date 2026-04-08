from __future__ import annotations

from openjury.annotate_config import AnnotateConfig


def test_annotate_config_supports_matchmaker_models():
    cfg = AnnotateConfig.from_dict(
        {
            "dataset": "alpaca-eval",
            "judge": {
                "model": "VLLM/Qwen/Qwen3-32B",
                "gpus": 2,
                "mode": "pairwise",
            },
            "models": [
                {"name": "VLLM/model-a", "gpus": 1},
                {"name": "VLLM/model-b", "gpus": 1},
            ],
            "criteria": "default",
            "pairing": {
                "source": "matchmaker",
                "strategy": "balanced_random",
                "seed": 11,
                "n_matches": 50,
            },
        }
    )

    assert cfg.requires_models is True
    assert cfg.requires_challenger is False
    assert [model.name for model in cfg.models] == ["VLLM/model-a", "VLLM/model-b"]
    assert cfg.pairing.source == "matchmaker"
    assert cfg.pairing.strategy == "balanced_random"
    assert cfg.pairing.n_matches == 50

    dumped = cfg.to_dict()
    assert dumped["pairing"]["source"] == "matchmaker"
    assert dumped["pairing"]["strategy"] == "balanced_random"
    assert dumped["pairing"]["n_matches"] == 50
    assert [row["name"] for row in dumped["models"]] == ["VLLM/model-a", "VLLM/model-b"]
