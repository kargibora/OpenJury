"""Resolver for the arena CLI (config loading + explicit CLI overrides)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from openjury._logging import logger
from openjury.arena.config import ArenaConfig, JudgeConfig, MatchmakerConfig, ModelEntry
from openjury.cli._forwarding.slurm import SlurmForwardRequest
from openjury.cli_args import parse_kwargs
from openjury.resolution.argparse_overrides import (
    apply_mapping_to_config,
    collect_explicit_dests,
)


@dataclass(frozen=True)
class ResolvedArenaCLI:
    """Resolved arena CLI inputs ready for execution."""

    config: ArenaConfig
    stage: str
    slurm_forward: SlurmForwardRequest | None = None


def _candidate_auto_config(args: argparse.Namespace) -> str | None:
    if args.stage != "analyze" or args.config:
        return None
    candidate = Path(args.output_dir) / "arena_config.json"
    return str(candidate) if candidate.exists() else None


def _apply_arena_cli_overrides(
    config: ArenaConfig,
    args: argparse.Namespace,
    explicit: set[str],
    *,
    gen_kwargs: dict,
) -> None:
    apply_mapping_to_config(
        config,
        args,
        explicit,
        [
            ("dataset", lambda c, v: setattr(c, "dataset", v)),
            ("output_dir", lambda c, v: setattr(c, "output_dir", v)),
            ("n_instructions", lambda c, v: setattr(c, "n_instructions", v)),
            ("language", lambda c, v: setattr(c, "language", v)),
            ("seed", lambda c, v: setattr(c, "seed", v)),
            ("balance_by", lambda c, v: setattr(c, "balance_by", v)),
            ("criteria", lambda c, v: setattr(c, "criteria", v)),
            ("generation_max_tokens", lambda c, v: setattr(c, "generation_max_tokens", v)),
            ("truncate_input_chars", lambda c, v: setattr(c, "truncate_input_chars", v)),
            ("judge_model", lambda c, v: setattr(c.judge, "model", v)),
            ("judge_mode", lambda c, v: setattr(c.judge, "mode", v)),
            ("pairwise_prompt_style", lambda c, v: setattr(c.judge, "pairwise_prompt_style", v)),
            ("judge_max_tokens", lambda c, v: setattr(c.judge, "max_tokens", v)),
            ("judge_gpus", lambda c, v: setattr(c.judge, "gpus", v)),
            ("judge_quantization", lambda c, v: setattr(c.judge, "quantization", v)),
            ("chat_template", lambda c, v: setattr(c.judge, "chat_template", v)),
            ("chat_template_file", lambda c, v: setattr(c.judge, "chat_template_file", v)),
            ("matchmaker", lambda c, v: setattr(c.matchmaker, "strategy", v)),
            ("n_matches", lambda c, v: setattr(c.matchmaker, "n_matches", v)),
            ("bt_regularization", lambda c, v: setattr(c, "bt_regularization", v)),
            ("elo_k", lambda c, v: setattr(c, "elo_k", v)),
            ("models", lambda c, v: setattr(c, "models", [ModelEntry(name=m) for m in v])),
        ],
    )

    if "ignore_cache" in explicit and args.ignore_cache:
        config.ignore_cache = True
    if "ignore_score_cache" in explicit and args.ignore_score_cache:
        config.ignore_score_cache = True
    if "no_swap" in explicit and args.no_swap:
        config.judge.no_swap = True
    if "provide_explanation" in explicit and args.provide_explanation:
        config.judge.provide_explanation = True
    if "enable_thinking" in explicit and args.enable_thinking:
        config.judge.enable_thinking = True
    if "include_completions" in explicit and args.include_completions:
        config.include_completions = True
    if "include_raw_judge" in explicit and args.include_raw_judge:
        config.include_raw_judge = True
    if "gen_kwargs" in explicit and gen_kwargs:
        config.judge.generation_kwargs = gen_kwargs


def resolve_arena_cli(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argv: list[str] | None,
) -> ResolvedArenaCLI:
    """Resolve arena CLI inputs into a typed config plus execution request."""
    explicit = collect_explicit_dests(parser, argv)
    gen_kwargs = parse_kwargs(getattr(args, "gen_kwargs", None))

    cfg_path = args.config or _candidate_auto_config(args)
    if cfg_path:
        config = ArenaConfig.load(cfg_path)
        logger.info("Loaded arena config from %s", cfg_path)
        _apply_arena_cli_overrides(config, args, explicit, gen_kwargs=gen_kwargs)
    else:
        if not args.models or len(args.models) < 2:
            parser.error("Provide --config OR --models with at least 2 models.")
        if not args.judge_model:
            parser.error("--judge_model is required when not using --config.")
        if not args.dataset:
            parser.error("--dataset is required when not using --config.")

        config = ArenaConfig(
            dataset=args.dataset,
            models=[ModelEntry(name=m) for m in args.models],
            judge=JudgeConfig(
                model=args.judge_model,
                gpus=args.judge_gpus,
                mode=args.judge_mode,
                pairwise_prompt_style=args.pairwise_prompt_style,
                max_tokens=args.judge_max_tokens,
                quantization=args.judge_quantization,
                chat_template=args.chat_template,
                chat_template_file=args.chat_template_file,
                provide_explanation=args.provide_explanation,
                no_swap=args.no_swap,
                enable_thinking=args.enable_thinking if args.enable_thinking else None,
                generation_kwargs=gen_kwargs,
            ),
            output_dir=args.output_dir,
            n_instructions=args.n_instructions,
            language=args.language,
            seed=args.seed,
            balance_by=args.balance_by,
            criteria=args.criteria,
            matchmaker=MatchmakerConfig(
                strategy=args.matchmaker,
                n_matches=args.n_matches,
            ),
            generation_max_tokens=args.generation_max_tokens,
            truncate_input_chars=args.truncate_input_chars,
            ignore_cache=args.ignore_cache,
            ignore_score_cache=args.ignore_score_cache,
            bt_regularization=args.bt_regularization,
            elo_k=args.elo_k,
            include_completions=args.include_completions,
            include_raw_judge=args.include_raw_judge,
        )

    slurm_forward: SlurmForwardRequest | None = None
    if args.slurm:
        slurm_forward = SlurmForwardRequest(
            mode="arena",
            submit=bool(getattr(args, "submit", False)),
            detach=bool(getattr(args, "detach", False)),
            slurm_output_dir=getattr(args, "slurm_output_dir", "slurm_scripts"),
            remote=bool(getattr(args, "remote", False)),
            cluster=getattr(args, "cluster", None),
            remote_project_dir=getattr(args, "remote_project_dir", None),
            wait_timeout=getattr(args, "wait_timeout", None),
            stage=args.stage,
            warn_output_dir_semantics=(args.output_dir != "results/arena/"),
        )

    return ResolvedArenaCLI(config=config, stage=args.stage, slurm_forward=slurm_forward)
