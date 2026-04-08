from __future__ import annotations

import pytest

import openjury.models.factory as model_factory
from openjury.arena import ArenaConfig
from openjury.arena.config import AgreementConfig, JudgeConfig, ModelEntry
from openjury.datasets.options import DatasetOptions
from openjury.generate_config import GenerateConfig, GenerateRuntimeConfig
from openjury.models.config import OpenAIConfig, SGLangConfig, VLLMConfig
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


def test_build_config_for_model_sglang_maps_local_engine_fields(tmp_path):
    template_path = tmp_path / "chat_template.jinja"
    template_path.write_text("{{ messages | length }}", encoding="utf-8")

    cfg = build_config_for_model(
        "SGLang/Qwen/Qwen3-8B",
        max_tokens=256,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.82,
        quantization="fp8",
        chat_template_file=str(template_path),
        generation_kwargs={"top_k": 16},
    )

    assert isinstance(cfg, SGLangConfig)
    assert cfg.max_tokens == 256
    assert cfg.tensor_parallel_size == 2
    assert cfg.mem_fraction_static == 0.82
    assert cfg.quantization == "fp8"
    assert cfg.chat_template == "{{ messages | length }}"
    assert cfg.generation_kwargs == {"top_k": 16}


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
            "pairwise_prompt_style": "legacy",
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
    assert judge["pairwise_prompt_style"] == "legacy"
    assert judge["provide_explanation"] is True
    assert judge["no_swap"] is True
    assert judge["enable_thinking"] is False
    assert judge["generation_kwargs"] == {"top_k": 8}


def test_judge_config_to_dict_omits_default_valued_fields():
    cfg = JudgeConfig(model="VLLM/Qwen/Qwen3-32B")

    assert cfg.to_dict() == {"model": "VLLM/Qwen/Qwen3-32B"}


def test_judge_config_from_raw_accepts_dataclass_fields_and_normalizes_legacy_values():
    raw = {
        "model": "VLLM/Qwen/Qwen3-32B",
        "pairwise_prompt_style": "rubric",
        "max_model_len": 16384,
        "enforce_eager": True,
        "enable_thinking": False,
        "generation_kwargs": {"top_k": 8},
        "unknown_field": "ignored",
    }

    cfg = JudgeConfig.from_raw(raw)

    assert cfg.model == "VLLM/Qwen/Qwen3-32B"
    assert cfg.pairwise_prompt_style == "criteria"
    assert cfg.max_model_len == 16384
    assert cfg.enforce_eager is True
    assert cfg.enable_thinking is False
    assert cfg.generation_kwargs == {"top_k": 8}


def test_model_entry_roundtrip_uses_sparse_serialization():
    default_entry = ModelEntry(name="Dummy/model-a")
    rich_entry = ModelEntry(
        name="Dummy/model-b",
        gpus=2,
        tp=4,
        quantization="fp8",
        generation_kwargs={"top_k": 10},
    )

    assert default_entry.to_dict() == "Dummy/model-a"
    assert rich_entry.to_dict() == {
        "name": "Dummy/model-b",
        "gpus": 2,
        "tp": 4,
        "quantization": "fp8",
        "generation_kwargs": {"top_k": 10},
    }
    assert ModelEntry.from_raw(rich_entry.to_dict()) == rich_entry


def test_generate_config_roundtrip_uses_model_entry_serialization(tmp_path):
    template_path = tmp_path / "generate_template.jinja"
    template_path.write_text("{{ messages }}", encoding="utf-8")

    cfg = GenerateConfig(
        model=ModelEntry(
            name="VLLM/Qwen/Qwen2.5-0.5B-Instruct",
            gpus=2,
            quantization="fp8",
            chat_template_file=str(template_path),
            max_tokens=1024,
        ),
        dataset=DatasetOptions(name="alpaca-eval"),
        output=str(tmp_path / "gen.parquet"),
        runtime=GenerateRuntimeConfig(
            gpu_memory_utilization=0.82,
            api_base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
        ),
    )

    saved = cfg.to_dict()
    roundtrip = GenerateConfig.from_dict(saved)

    assert saved["dataset"] == {
        "name": "alpaca-eval",
        "seed": 42,
    }
    assert saved["model"] == {
        "name": "VLLM/Qwen/Qwen2.5-0.5B-Instruct",
        "gpus": 2,
        "quantization": "fp8",
        "chat_template_file": str(template_path),
        "max_tokens": 1024,
    }
    assert saved["runtime"] == {
        "truncate_input_chars": 8192,
        "gpu_memory_utilization": 0.82,
        "api_base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "use_tqdm": False,
        "base_model": False,
        "ignore_cache": False,
    }
    assert "n_instructions" not in saved
    assert "language" not in saved
    assert "balance_by" not in saved
    assert "max_tokens" not in saved
    assert "tensor_parallel_size" not in saved
    assert "quantization" not in saved
    assert "chat_template_file" not in saved
    assert "gpu_memory_utilization" not in saved
    assert "api_base_url" not in saved
    assert "api_key_env" not in saved
    assert roundtrip.dataset_options == cfg.dataset_options
    assert roundtrip.model_entry == cfg.model_entry
    assert roundtrip.runtime == cfg.runtime
    assert roundtrip.gpu_memory_utilization == 0.82
    assert roundtrip.api_base_url == "https://openrouter.ai/api/v1"
    assert roundtrip.api_key_env == "OPENROUTER_API_KEY"


def test_generate_config_rejects_legacy_flat_shape():
    with pytest.raises(ValueError, match="nested 'dataset' object"):
        GenerateConfig.from_dict(
            {
                "model": "VLLM/Qwen/Qwen2.5-0.5B-Instruct",
                "dataset": "alpaca-eval",
                "output": "out.parquet",
                "gpu_memory_utilization": 0.82,
                "gpu_devices": "0,1",
                "api_base_url": "https://openrouter.ai/api/v1",
                "api_key_env": "OPENROUTER_API_KEY",
                "use_tqdm": True,
                "base_model": True,
                "ignore_cache": True,
                "truncate_input_chars": 4096,
            }
        )


def test_generate_config_upgrade_legacy_dict_rewrites_nested_shape():
    upgraded = GenerateConfig.upgrade_legacy_dict(
        {
            "model": "VLLM/Qwen/Qwen2.5-0.5B-Instruct",
            "dataset": "alpaca-eval",
            "output": "out.parquet",
            "n_instructions": 25,
            "language": "en",
            "seed": 13,
            "balance_by": "lang",
            "max_tokens": 2048,
            "tensor_parallel_size": 2,
            "quantization": "fp8",
            "gpu_memory_utilization": 0.82,
            "api_key_env": "OPENROUTER_API_KEY",
            "ignore_cache": True,
        }
    )

    assert upgraded == {
        "model": {
            "name": "VLLM/Qwen/Qwen2.5-0.5B-Instruct",
            "gpus": 2,
            "quantization": "fp8",
            "max_tokens": 2048,
        },
        "dataset": {
            "name": "alpaca-eval",
            "n_instructions": 25,
            "language": "en",
            "seed": 13,
            "balance_by": "lang",
        },
        "output": "out.parquet",
        "runtime": {
            "gpu_memory_utilization": 0.82,
            "api_key_env": "OPENROUTER_API_KEY",
            "ignore_cache": True,
        },
    }


def test_agreement_config_uses_judge_to_dict_for_serialization():
    cfg = AgreementConfig(
        dataset="lmsys",
        judge=JudgeConfig(
            model="VLLM/Qwen/Qwen3-32B",
            max_model_len=16384,
            enforce_eager=True,
        ),
    )

    saved = cfg.to_dict()

    assert saved["judge"] == {
        "model": "VLLM/Qwen/Qwen3-32B",
        "max_model_len": 16384,
        "enforce_eager": True,
    }
