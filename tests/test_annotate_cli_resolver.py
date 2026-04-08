from __future__ import annotations

from openjury.cli.annotate import build_parser
from openjury.cli._resolvers.annotate import resolve_annotate_cli


def test_annotate_resolver_builds_config_from_cli():
    parser = build_parser()
    argv = [
        "--dataset", "lmsys",
        "--challenger_model", "VLLM/openai/gpt-oss-20b",
        "--challenger_gpus", "4",
        "--judge_model", "VLLM/Qwen/Qwen3-32B",
        "--judge_gpus", "4",
        "--judge_mode", "pairwise",
        "--criteria", "default",
        "--generation_max_tokens", "2048",
        "--truncate_input_chars", "4096",
        "--pairing_strategy", "all",
        "--n_instructions", "500",
        "--balance_by", "models",
        "--output_dir", "results/annotate/test",
        "--max_model_len", "16384",
        "--enforce_eager",
    ]
    args = parser.parse_args(argv)
    resolved = resolve_annotate_cli(parser, args, argv)

    assert resolved.config.dataset == "lmsys"
    assert resolved.config.challenger.name == "VLLM/openai/gpt-oss-20b"
    assert resolved.config.challenger.gpus == 4
    assert resolved.config.judge.model == "VLLM/Qwen/Qwen3-32B"
    assert resolved.config.judge.max_model_len == 16384
    assert resolved.config.judge.enforce_eager is True
    assert resolved.config.generation.max_tokens == 2048
    assert resolved.config.pairing.strategy == "all"
    assert resolved.config.pairing.source == "challenger_vs_dataset_inline"


def test_annotate_resolver_builds_matchmaker_config_from_cli():
    parser = build_parser()
    argv = [
        "--dataset", "alpaca-eval",
        "--pairing_source", "matchmaker",
        "--pairing_strategy", "balanced_random",
        "--n_matches", "300",
        "--models", "VLLM/model-a", "VLLM/model-b", "VLLM/model-c",
        "--judge_model", "VLLM/Qwen/Qwen3-32B",
        "--judge_gpus", "4",
        "--judge_mode", "pairwise",
        "--criteria", "default",
    ]
    args = parser.parse_args(argv)
    resolved = resolve_annotate_cli(parser, args, argv)

    assert resolved.config.dataset == "alpaca-eval"
    assert resolved.config.pairing.source == "matchmaker"
    assert resolved.config.pairing.strategy == "balanced_random"
    assert resolved.config.pairing.n_matches == 300
    assert resolved.config.challenger is None
    assert [model.name for model in resolved.config.models or []] == [
        "VLLM/model-a",
        "VLLM/model-b",
        "VLLM/model-c",
    ]
