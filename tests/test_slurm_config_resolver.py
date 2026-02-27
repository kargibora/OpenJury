from __future__ import annotations

import argparse

import pytest

from openjury.arena.config import AgreementConfig, JudgeConfig
from openjury.generate_config import GenerateConfig
from openjury.resolution.argparse_overrides import collect_explicit_dests
from openjury.slurm.config_resolver import resolve_args_from_config, validate_mode_args


def test_collect_explicit_dests_detects_value_and_equals_forms():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge_max_tokens", type=int, default=2048)
    parser.add_argument("--config", default=None)

    explicit = collect_explicit_dests(
        parser,
        ["--config", "cfg.json", "--seed", "42", "--judge_max_tokens=2048"],
    )

    assert {"config", "seed", "judge_max_tokens"} <= explicit


def test_slurm_config_resolver_respects_explicit_default_valued_overrides(tmp_path):
    cfg = AgreementConfig(
        dataset="lmsys",
        judge=JudgeConfig(
            model="OpenRouter/qwen/qwen3-32b",
            max_tokens=999,
        ),
        seed=7,
    )
    cfg_path = tmp_path / "agreement.json"
    cfg.save(cfg_path)

    args = argparse.Namespace(
        config=str(cfg_path),
        mode="agreement",
        dataset=None,
        judge_model=None,
        judge_gpus=1,
        judge_quantization=None,
        judge_mode="samplewise",
        judge_max_tokens=2048,
        enable_thinking=False,
        chat_template=None,
        chat_template_file=None,
        rubric="default",
        n_instructions=None,
        language=None,
        seed=42,
        truncate_instruction=500,
        ignore_score_cache=False,
        models=None,
    )

    payload = resolve_args_from_config(
        args,
        explicit_dests={"seed", "judge_max_tokens"},
    )

    assert args.dataset == "lmsys"
    assert args.judge_model == "OpenRouter/qwen/qwen3-32b"
    assert args.seed == 42  # explicit CLI default wins over config seed=7
    assert args.judge_max_tokens == 2048  # explicit CLI default wins over config 999
    assert payload.kind == "agreement"
    assert payload.payload and payload.payload["seed"] == 7


def test_slurm_config_resolver_supports_generate_config(tmp_path):
    cfg = GenerateConfig(
        model="VLLM/Qwen/Qwen2.5-0.5B-Instruct",
        dataset="alpaca-eval",
        output=str(tmp_path / "gen.parquet"),
        n_instructions=11,
        language="en",
        seed=13,
        balance_by="lang",
        max_tokens=1234,
        truncate_input_chars=4321,
        tensor_parallel_size=2,
        quantization="fp8",
        ignore_cache=True,
    )
    cfg_path = tmp_path / "generate.json"
    cfg.save(cfg_path)

    args = argparse.Namespace(
        config=str(cfg_path),
        mode="generate",
        dataset=None,
        n_instructions=None,
        language=None,
        seed=42,
        balance_by=None,
        generation_max_tokens=4096,
        truncate_input_chars=8192,
        models=None,
        model_gpus=None,
        model_quantizations=None,
        ignore_cache=False,
    )

    payload = resolve_args_from_config(args, explicit_dests=set())

    assert args.dataset == "alpaca-eval"
    assert args.n_instructions == 11
    assert args.language == "en"
    assert args.seed == 13
    assert args.balance_by == "lang"
    assert args.generation_max_tokens == 1234
    assert args.truncate_input_chars == 4321
    assert args.models == ["VLLM/Qwen/Qwen2.5-0.5B-Instruct"]
    assert args.model_gpus == [2]
    assert args.model_quantizations == ["fp8"]
    assert args.ignore_cache is True
    assert payload.kind == "generate"
    assert payload.payload and payload.payload["output"] == str(tmp_path / "gen.parquet")


def test_validate_mode_args_rejects_stage_for_generate_and_judge():
    parser = argparse.ArgumentParser()

    with pytest.raises(SystemExit):
        validate_mode_args(
            parser,
            argparse.Namespace(
                mode="generate",
                stage="annotate",
                models=["Dummy/A"],
                judge_model=None,
                dataset="alpaca-eval",
            ),
        )

    with pytest.raises(SystemExit):
        validate_mode_args(
            parser,
            argparse.Namespace(
                mode="judge",
                stage="analyze",
                models=["Dummy/A", "Dummy/B"],
                judge_model="Dummy/J",
                dataset="alpaca-eval",
            ),
        )
