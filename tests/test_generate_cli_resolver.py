from __future__ import annotations

import argparse

import pytest

from openjury.cli._resolvers.generate import resolve_generate_cli
from openjury.cli_args import add_dataset_args, add_dataset_selection_args


def _generate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", required=False)
    add_dataset_args(parser, dataset_required=False)
    add_dataset_selection_args(parser)
    parser.add_argument("--output", required=False)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--truncate_input_chars", type=int, default=8192)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--gpu_devices", default=None)
    parser.add_argument("--chat_template", default=None)
    parser.add_argument("--chat_template_file", default=None)
    parser.add_argument("--api_base_url", default=None)
    parser.add_argument("--api_key_env", default=None)
    parser.add_argument("--use_tqdm", action="store_true")
    parser.add_argument("--base_model", action="store_true")
    parser.add_argument("--ignore_cache", action="store_true")
    parser.add_argument("--slurm", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--slurm_output_dir", default="slurm_scripts")
    return parser


def test_generate_resolver_builds_slurm_forward_and_dataset_options(tmp_path):
    parser = _generate_parser()
    argv = [
        "--model", "VLLM/Qwen/Qwen2.5-0.5B-Instruct",
        "--dataset", "alpaca-eval",
        "--output", str(tmp_path / "gen.parquet"),
        "--n_instructions", "10",
        "--language", "en",
        "--seed", "13",
        "--max_tokens", "1024",
        "--tensor_parallel_size", "2",
        "--quantization", "fp8",
        "--ignore_cache",
        "--slurm",
        "--submit",
    ]
    args = parser.parse_args(argv)
    resolved = resolve_generate_cli(parser, args, argv)

    assert resolved.config.dataset_options.name == "alpaca-eval"
    assert resolved.config.dataset_options.n_instructions == 10
    assert resolved.config.dataset_options.loader_kwargs() == {"language": "en", "seed": 13}
    assert resolved.config.max_tokens == 1024
    assert resolved.config.tensor_parallel_size == 2
    assert resolved.config.quantization == "fp8"
    assert resolved.config.ignore_cache is True
    assert resolved.slurm_forward is not None
    assert resolved.slurm_forward.mode == "generate"
    assert resolved.slurm_forward.submit is True


def test_generate_resolver_config_plus_explicit_default_overrides(tmp_path):
    from openjury.generate_config import GenerateConfig

    cfg = GenerateConfig(
        model="VLLM/Qwen/Qwen2.5-0.5B-Instruct",
        dataset="alpaca-eval",
        output=str(tmp_path / "cfg.parquet"),
        seed=7,
        max_tokens=999,
    )
    cfg_path = tmp_path / "generate.json"
    cfg.save(cfg_path)

    parser = _generate_parser()
    argv = [
        "--config", str(cfg_path),
        "--output", str(tmp_path / "override.parquet"),
        "--seed", "42",          # explicit default-valued override
        "--max_tokens", "4096",  # explicit default-valued override
    ]
    args = parser.parse_args(argv)
    resolved = resolve_generate_cli(parser, args, argv)

    assert resolved.config.output == str(tmp_path / "override.parquet")
    assert resolved.config.seed == 42
    assert resolved.config.max_tokens == 4096


def test_generate_resolver_rejects_unsupported_slurm_flags(tmp_path):
    parser = _generate_parser()
    argv = [
        "--model", "VLLM/Qwen/Qwen2.5-0.5B-Instruct",
        "--dataset", "alpaca-eval",
        "--output", str(tmp_path / "gen.parquet"),
        "--chat_template", "{{ messages }}",
        "--slurm",
    ]
    args = parser.parse_args(argv)
    with pytest.raises(SystemExit):
        resolve_generate_cli(parser, args, argv)
