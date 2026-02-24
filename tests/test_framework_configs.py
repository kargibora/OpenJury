from __future__ import annotations

import openjury.models.factory as model_factory
from openjury.arena import ArenaConfig
from openjury.models.config import OpenAIConfig, VLLMConfig
from openjury.models.factory import build_config_for_model


def test_build_config_for_model_vllm_reads_chat_template_file(tmp_path):
    template_path = tmp_path / "chat_template.jinja"
    template_path.write_text("{{ messages | length }}", encoding="utf-8")

    cfg = build_config_for_model(
        "VLLM/Qwen/Qwen2.5-0.5B-Instruct",
        max_tokens=128,
        tensor_parallel_size=2,
        quantization="fp8",
        chat_template_file=str(template_path),
        generation_kwargs={"top_k": 32},
        api_base_url="https://ignored.example/v1",
    )

    assert isinstance(cfg, VLLMConfig)
    assert cfg.max_tokens == 128
    assert cfg.tensor_parallel_size == 2
    assert cfg.quantization == "fp8"
    assert cfg.chat_template == "{{ messages | length }}"
    assert cfg.generation_kwargs == {"top_k": 32}


def test_make_model_dict_config_uses_provider_specific_config(monkeypatch):
    def fake_create(*, provider, model_name, config):
        assert provider == "OpenRouter"
        assert model_name == "deepseek/deepseek-chat-v3.1"
        return config

    monkeypatch.setattr(
        model_factory.ModelRegistry,
        "create",
        staticmethod(fake_create),
    )

    cfg = model_factory.make_model(
        "OpenRouter/deepseek/deepseek-chat-v3.1",
        config={
            "api_base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
            "temperature": 0.0,
        },
    )

    assert isinstance(cfg, OpenAIConfig)
    assert cfg.base_url == "https://openrouter.ai/api/v1"
    assert cfg.api_key_env == "OPENROUTER_API_KEY"
    assert cfg.temperature == 0.0


def test_make_model_vllm_chat_template_shortcut_builds_vllm_config(tmp_path, monkeypatch):
    template_path = tmp_path / "judge_template.jinja"
    template_path.write_text("{% for m in messages %}{{ m['content'] }}{% endfor %}", encoding="utf-8")

    def fake_create(*, provider, model_name, config):
        assert provider == "VLLM"
        assert model_name == "Qwen/Qwen2.5-0.5B-Instruct"
        return config

    monkeypatch.setattr(
        model_factory.ModelRegistry,
        "create",
        staticmethod(fake_create),
    )

    cfg = model_factory.make_model(
        "VLLM/Qwen/Qwen2.5-0.5B-Instruct",
        max_tokens=256,
        config={"tensor_parallel_size": 2, "quantization": "fp8"},
        chat_template_file=str(template_path),
    )

    assert isinstance(cfg, VLLMConfig)
    assert cfg.max_tokens == 256
    assert cfg.tensor_parallel_size == 2
    assert cfg.quantization == "fp8"
    assert "messages" in (cfg.chat_template or "")


def test_arena_config_roundtrip_preserves_model_and_judge_overrides(tmp_path):
    chat_template_path = tmp_path / "template.jinja"
    chat_template_path.write_text("{{ messages }}", encoding="utf-8")

    raw = {
        "dataset": "alpaca-eval",
        "models": [
            "Dummy/model-a",
            {
                "name": "VLLM/Qwen/Qwen2.5-0.5B-Instruct",
                "gpus": 2,
                "tp": 2,
                "quantization": "fp8",
                "completions": "results/completions/model.parquet",
                "chat_template_file": str(chat_template_path),
                "max_tokens": 1024,
                "temperature": 0.2,
                "top_p": 0.9,
                "generation_kwargs": {"top_k": 20},
            },
        ],
        "judge": {
            "model": "VLLM/Qwen/Qwen2.5-1.5B-Instruct",
            "gpus": 2,
            "tp": 2,
            "mode": "pairwise",
            "max_tokens": 2048,
            "temperature": 0.0,
            "top_p": 1.0,
            "quantization": "fp8",
            "chat_template": "{{ messages }}",
            "provide_explanation": True,
            "no_swap": True,
            "enable_thinking": False,
            "generation_kwargs": {"top_k": 8},
        },
        "generation": {
            "max_tokens": 4096,
            "truncate_input_chars": 4096,
            "ignore_cache": True,
            "ignore_score_cache": True,
        },
        "ratings": {"bt_regularization": 0.1, "elo_k": 24.0},
        "output_dir": "results/arena/demo",
    }

    cfg = ArenaConfig.from_dict(raw)
    saved = cfg.to_dict()

    model_entry = saved["models"][1]
    assert isinstance(model_entry, dict)
    assert model_entry["tp"] == 2
    assert model_entry["quantization"] == "fp8"
    assert model_entry["completions"] == "results/completions/model.parquet"
    assert model_entry["chat_template_file"] == str(chat_template_path)
    assert model_entry["max_tokens"] == 1024
    assert model_entry["generation_kwargs"] == {"top_k": 20}

    judge = saved["judge"]
    assert judge["tp"] == 2
    assert judge["quantization"] == "fp8"
    assert judge["chat_template"] == "{{ messages }}"
    assert judge["provide_explanation"] is True
    assert judge["no_swap"] is True
    assert judge["enable_thinking"] is False
    assert judge["generation_kwargs"] == {"top_k": 8}

