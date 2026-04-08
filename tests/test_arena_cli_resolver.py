from __future__ import annotations

import argparse

import pytest

from openjury.arena.config import ArenaConfig, JudgeConfig, ModelEntry
from openjury.cli._resolvers.arena import resolve_arena_cli
from openjury.cli_args import add_arena_pipeline_args


def _arena_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_arena_pipeline_args(parser)
    parser.add_argument("--bt_regularization", type=float, default=0.01)
    parser.add_argument("--elo_k", type=float, default=32.0)
    parser.add_argument("--include_completions", action="store_true")
    parser.add_argument("--include_raw_judge", action="store_true")
    parser.add_argument("--stage", choices=("all", "annotate", "analyze"), default="all")
    parser.add_argument("--slurm", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--slurm_output_dir", default="slurm_scripts")
    return parser


def test_arena_resolver_config_plus_explicit_overrides(tmp_path):
    cfg = ArenaConfig(
        dataset="alpaca-eval",
        models=[ModelEntry(name="Dummy/A"), ModelEntry(name="Dummy/B")],
        judge=JudgeConfig(model="Dummy/J", max_tokens=999),
        bt_regularization=0.01,
        include_raw_judge=False,
    )
    cfg_path = tmp_path / "arena.json"
    cfg.save(cfg_path)

    parser = _arena_parser()
    argv = [
        "--config", str(cfg_path),
        "--judge_max_tokens", "2048",  # explicit default-valued override
        "--max_model_len", "8192",
        "--enforce_eager",
        "--no_swap",
        "--bt_regularization", "0.2",
        "--include_raw_judge",
        "--gen-kwargs", "top_k=11",
    ]
    args = parser.parse_args(argv)
    resolved = resolve_arena_cli(parser, args, argv)

    assert resolved.config.judge.max_tokens == 2048
    assert resolved.config.judge.max_model_len == 8192
    assert resolved.config.judge.enforce_eager is True
    assert resolved.config.judge.no_swap is True
    assert resolved.config.bt_regularization == 0.2
    assert resolved.config.include_raw_judge is True
    assert resolved.config.judge.generation_kwargs == {"top_k": 11}


def test_arena_resolver_auto_loads_stage_analyze_config(tmp_path):
    out = tmp_path / "arena_out"
    out.mkdir()
    cfg = ArenaConfig(
        dataset="arena-hard",
        models=[ModelEntry(name="Dummy/A"), ModelEntry(name="Dummy/B")],
        judge=JudgeConfig(model="Dummy/J"),
        output_dir=str(out),
    )
    cfg.save(out / "arena_config.json")

    parser = _arena_parser()
    argv = ["--stage", "analyze", "--output_dir", str(out)]
    args = parser.parse_args(argv)
    resolved = resolve_arena_cli(parser, args, argv)

    assert resolved.stage == "analyze"
    assert resolved.config.dataset == "arena-hard"


def test_arena_resolver_inline_requires_models_and_judge_and_dataset():
    parser = _arena_parser()
    argv = ["--models", "Dummy/A", "Dummy/B"]
    args = parser.parse_args(argv)
    with pytest.raises(SystemExit):
        resolve_arena_cli(parser, args, argv)
