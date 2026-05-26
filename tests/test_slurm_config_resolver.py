from __future__ import annotations

import argparse

import pytest

from openjury.arena.config import AgreementConfig, ArenaConfig, JudgeConfig, ModelEntry
from openjury.generate_config import GenerateConfig
from openjury.resolution.argparse_overrides import collect_explicit_dests
from openjury.slurm.config_resolver import resolve_args_from_config, validate_mode_args
from openjury.slurm.generate_slurm import build_run_plan_from_env_and_args


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
    assert len(args.model_entries) == 1
    assert args.model_entries[0].name == "VLLM/Qwen/Qwen2.5-0.5B-Instruct"
    assert args.model_entries[0].gpus == 2
    assert args.model_entries[0].quantization == "fp8"
    assert args.ignore_cache is True
    assert payload.kind == "generate"
    assert payload.payload and payload.payload["output"] == str(tmp_path / "gen.parquet")
    assert payload.payload["dataset"] == {
        "name": "alpaca-eval",
        "n_instructions": 11,
        "language": "en",
        "seed": 13,
        "balance_by": "lang",
    }
    assert payload.payload["runtime"] == {
        "truncate_input_chars": 4321,
        "gpu_memory_utilization": 0.9,
        "use_tqdm": False,
        "base_model": False,
        "ignore_cache": True,
    }


def test_slurm_config_resolver_preserves_rich_arena_model_entries(tmp_path):
    cfg = ArenaConfig(
        dataset="alpaca-eval",
        models=[
            ModelEntry(
                name="VLLM/Dummy/A",
                gpus=2,
                quantization="fp8",
                max_tokens=123,
                generation_kwargs={"top_k": 7},
            ),
            ModelEntry(name="OpenRouter/Dummy/B", chat_template="{{ messages }}"),
        ],
        judge=JudgeConfig(model="OpenRouter/Dummy/J"),
    )
    cfg_path = tmp_path / "arena.json"
    cfg.save(cfg_path)

    args = argparse.Namespace(
        config=str(cfg_path),
        mode="arena",
        models=None,
        model_gpus=None,
        model_quantizations=None,
        judge_model=None,
        dataset=None,
        n_instructions=None,
        language=None,
        seed=42,
        balance_by=None,
        criteria="default",
        judge_gpus=1,
        judge_quantization=None,
        judge_mode="samplewise",
        pairwise_prompt_style="criteria",
        enable_thinking=False,
        chat_template=None,
        chat_template_file=None,
        matchmaker="round_robin",
        n_matches=None,
        generation_max_tokens=4096,
        truncate_input_chars=8192,
        ignore_cache=False,
        ignore_score_cache=False,
        max_model_len=None,
        judge_max_model_len=None,
        enforce_eager=False,
        judge_enforce_eager=False,
        truncate_instruction=500,
    )

    payload = resolve_args_from_config(args, explicit_dests=set())

    assert payload.kind == "arena"
    assert args.models == ["VLLM/Dummy/A", "OpenRouter/Dummy/B"]
    assert args.model_gpus == [2, 1]
    assert args.model_quantizations == ["fp8", "none"]
    assert args.model_entries[0].max_tokens == 123
    assert args.model_entries[0].generation_kwargs == {"top_k": 7}
    assert args.model_entries[1].chat_template == "{{ messages }}"


def test_build_run_plan_from_env_and_args_uses_structured_model_entries(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("USER_WORK_DIR", str(tmp_path / "user_work"))
    monkeypatch.setenv("ACCOUNT", "acct")
    monkeypatch.setenv("PARTITION", "part")
    monkeypatch.setenv("TIME_LIMIT", "00:30:00")

    args = argparse.Namespace(
        dataset="alpaca-eval",
        judge_model="OpenRouter/Dummy/J",
        judge_gpus=1,
        judge_quantization=None,
        judge_mode="samplewise",
        pairwise_prompt_style="criteria",
        judge_max_tokens=2048,
        no_swap=False,
        provide_explanation=False,
        enable_thinking=False,
        chat_template=None,
        chat_template_file=None,
        max_model_len=None,
        judge_max_model_len=None,
        enforce_eager=False,
        judge_enforce_eager=False,
        gen_kwargs=None,
        criteria="default",
        n_instructions=10,
        generation_max_tokens=4096,
        truncate_input_chars=8192,
        ignore_cache=False,
        ignore_score_cache=False,
        language=None,
        seed=42,
        balance_by=None,
        truncate_instruction=500,
        stage="all",
        mode="arena",
        models=None,
        model_gpus=None,
        model_quantizations=None,
        model_entries=[
            ModelEntry(name="VLLM/Dummy/A", gpus=2, quantization="fp8"),
            ModelEntry(name="OpenRouter/Dummy/B"),
        ],
        partition=None,
        account=None,
        time_generate=None,
        time_judge=None,
        qos="normal",
        project_dir=None,
        tag="TESTTAG",
    )

    plan = build_run_plan_from_env_and_args(args)

    assert [model.name for model in plan.models] == [
        "VLLM/Dummy/A",
        "OpenRouter/Dummy/B",
    ]
    assert plan.models[0].gpus == 2
    assert plan.models[0].quantization == "fp8"
    assert plan.judge.model == "OpenRouter/Dummy/J"
    assert plan.execution.partition == "part"


def test_build_run_plan_from_env_and_args_resolves_container_defaults(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("USER_WORK_DIR", str(tmp_path / "user_work"))
    monkeypatch.setenv("ACCOUNT", "acct")
    monkeypatch.setenv("PARTITION", "part")
    monkeypatch.setenv("TIME_LIMIT", "00:30:00")
    monkeypatch.setenv("OPENJURY_CONTAINER_RUNTIME", "apptainer")
    monkeypatch.setenv("OPENJURY_CONTAINER_IMAGE", str(tmp_path / "runtime.sif"))
    monkeypatch.setenv("OPENJURY_CONTAINER_HOME", str(tmp_path / "container-home"))

    args = argparse.Namespace(
        dataset="lmsys",
        judge_model="VLLM/Qwen/Qwen3-8B",
        judge_gpus=1,
        judge_quantization=None,
        judge_mode="samplewise",
        pairwise_prompt_style="criteria",
        judge_max_tokens=2048,
        no_swap=False,
        provide_explanation=False,
        enable_thinking=False,
        chat_template=None,
        chat_template_file=None,
        max_model_len=None,
        judge_max_model_len=None,
        enforce_eager=False,
        judge_enforce_eager=False,
        gen_kwargs=None,
        criteria="default",
        n_instructions=10,
        generation_max_tokens=4096,
        truncate_input_chars=8192,
        ignore_cache=False,
        ignore_score_cache=False,
        language=None,
        seed=42,
        balance_by=None,
        truncate_instruction=500,
        stage="all",
        mode="agreement",
        models=[],
        model_gpus=None,
        model_quantizations=None,
        model_entries=[],
        partition=None,
        account=None,
        time_generate=None,
        time_judge=None,
        qos="normal",
        project_dir=None,
        tag="TAG",
        container_runtime=None,
        container_image=None,
        container_home=None,
    )

    plan = build_run_plan_from_env_and_args(args)

    assert plan.execution.container_runtime == "apptainer"
    assert plan.execution.container_image == str(tmp_path / "runtime.sif")
    assert plan.execution.container_home == str(tmp_path / "container-home")


def test_build_run_plan_from_env_and_args_allows_container_cli_overrides(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("USER_WORK_DIR", str(tmp_path / "user_work"))
    monkeypatch.setenv("ACCOUNT", "acct")
    monkeypatch.setenv("PARTITION", "part")
    monkeypatch.setenv("TIME_LIMIT", "00:30:00")
    monkeypatch.setenv("OPENJURY_CONTAINER_RUNTIME", "none")

    args = argparse.Namespace(
        dataset="lmsys",
        judge_model="VLLM/Qwen/Qwen3-8B",
        judge_gpus=1,
        judge_quantization=None,
        judge_mode="samplewise",
        pairwise_prompt_style="criteria",
        judge_max_tokens=2048,
        no_swap=False,
        provide_explanation=False,
        enable_thinking=False,
        chat_template=None,
        chat_template_file=None,
        max_model_len=None,
        judge_max_model_len=None,
        enforce_eager=False,
        judge_enforce_eager=False,
        gen_kwargs=None,
        criteria="default",
        n_instructions=10,
        generation_max_tokens=4096,
        truncate_input_chars=8192,
        ignore_cache=False,
        ignore_score_cache=False,
        language=None,
        seed=42,
        balance_by=None,
        truncate_instruction=500,
        stage="all",
        mode="agreement",
        models=[],
        model_gpus=None,
        model_quantizations=None,
        model_entries=[],
        partition=None,
        account=None,
        time_generate=None,
        time_judge=None,
        qos="normal",
        project_dir=None,
        tag="TAG",
        container_runtime="apptainer",
        container_image=str(tmp_path / "cli-runtime.sif"),
        container_home=str(tmp_path / "cli-home"),
    )

    plan = build_run_plan_from_env_and_args(args)

    assert plan.execution.container_runtime == "apptainer"
    assert plan.execution.container_image == str(tmp_path / "cli-runtime.sif")
    assert plan.execution.container_home == str(tmp_path / "cli-home")


def test_build_run_plan_from_env_and_args_defaults_container_home_from_hf_home(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("USER_WORK_DIR", str(tmp_path / "user_work"))
    monkeypatch.setenv("ACCOUNT", "acct")
    monkeypatch.setenv("PARTITION", "part")
    monkeypatch.setenv("TIME_LIMIT", "00:30:00")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-cache"))
    monkeypatch.setenv("OPENJURY_CONTAINER_RUNTIME", "apptainer")
    monkeypatch.setenv("OPENJURY_CONTAINER_IMAGE", str(tmp_path / "runtime.sif"))
    monkeypatch.delenv("OPENJURY_CONTAINER_HOME", raising=False)

    args = argparse.Namespace(
        dataset="lmsys",
        judge_model="VLLM/Qwen/Qwen3-8B",
        judge_gpus=1,
        judge_quantization=None,
        judge_mode="samplewise",
        pairwise_prompt_style="criteria",
        judge_max_tokens=2048,
        no_swap=False,
        provide_explanation=False,
        enable_thinking=False,
        chat_template=None,
        chat_template_file=None,
        max_model_len=None,
        judge_max_model_len=None,
        enforce_eager=False,
        judge_enforce_eager=False,
        gen_kwargs=None,
        criteria="default",
        n_instructions=10,
        generation_max_tokens=4096,
        truncate_input_chars=8192,
        ignore_cache=False,
        ignore_score_cache=False,
        language=None,
        seed=42,
        balance_by=None,
        truncate_instruction=500,
        stage="all",
        mode="agreement",
        models=[],
        model_gpus=None,
        model_quantizations=None,
        model_entries=[],
        partition=None,
        account=None,
        time_generate=None,
        time_judge=None,
        qos="normal",
        project_dir=None,
        tag="TAG",
        container_runtime=None,
        container_image=None,
        container_home=None,
    )

    plan = build_run_plan_from_env_and_args(args)

    assert plan.execution.container_home == str(tmp_path / "hf-cache" / "openjury_container_home")


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
