"""Helpers for resolving CLI args + task configs for the SLURM planner.

This module keeps precedence rules in one place:

1. CLI flags explicitly provided by the user
2. Config file values (``--config``)
3. Argparse defaults

The goal is to avoid scattering `if args.foo == default` checks across the CLI.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from openjury.resolution.argparse_overrides import (
    apply_mapping_from_config,
    collect_explicit_dests,
    present_nonempty,
)


@dataclass
class ResolvedTaskConfigPayload:
    """Optional typed task config payload preserved for SLURM config emission."""
    kind: str | None = None
    payload: dict[str, Any] | None = None


def resolve_args_from_config(
    args: argparse.Namespace,
    *,
    explicit_dests: set[str] | None = None,
) -> ResolvedTaskConfigPayload:
    """Apply ``--config`` values into parsed CLI args when present.

    CLI flags explicitly passed by the user win over config-file values.
    """
    if not getattr(args, "config", None):
        return ResolvedTaskConfigPayload()

    explicit = explicit_dests or set()
    if args.mode == "agreement":
        return _apply_agreement_config(args, explicit)
    if args.mode == "generate":
        return _apply_generate_config(args, explicit)
    else:  # ``arena`` / ``judge`` reuse ArenaConfig today.
        return _apply_arena_config(args, explicit)


def _apply_generate_config(
    args: argparse.Namespace,
    explicit: set[str],
) -> ResolvedTaskConfigPayload:
    from openjury.generate_config import GenerateConfig

    cfg = GenerateConfig.load(args.config)

    apply_mapping_from_config(
        args,
        explicit,
        cfg,
        [
            ("dataset", lambda c: c.dataset),
            ("n_instructions", lambda c: c.n_instructions),
            ("language", lambda c: c.language),
            ("seed", lambda c: c.seed),
            ("balance_by", lambda c: c.balance_by),
            ("generation_max_tokens", lambda c: c.max_tokens),
            ("truncate_input_chars", lambda c: c.truncate_input_chars),
            ("models", lambda c: [c.model]),
            ("model_gpus", lambda c: [c.tensor_parallel_size]),
            ("model_quantizations", lambda c: [c.quantization or "none"]),
        ],
        present=present_nonempty,
    )

    if cfg.ignore_cache:
        args.ignore_cache = True

    return ResolvedTaskConfigPayload(kind="generate", payload=cfg.to_dict())


def _apply_agreement_config(
    args: argparse.Namespace,
    explicit: set[str],
) -> ResolvedTaskConfigPayload:
    from openjury.arena.config import AgreementConfig

    cfg = AgreementConfig.load(args.config)

    apply_mapping_from_config(
        args,
        explicit,
        cfg,
        [
            ("dataset", lambda c: c.dataset),
            ("judge_model", lambda c: c.judge.model),
            ("judge_gpus", lambda c: c.judge.gpus),
            ("judge_quantization", lambda c: c.judge.quantization),
            ("judge_mode", lambda c: c.judge.mode),
            ("pairwise_prompt_style", lambda c: c.judge.pairwise_prompt_style),
            ("judge_max_tokens", lambda c: c.judge.max_tokens),
            (
                "enable_thinking",
                lambda c: (
                    bool(c.judge.enable_thinking)
                    if c.judge.enable_thinking is not None
                    else None
                ),
            ),
            ("chat_template", lambda c: c.judge.chat_template),
            ("chat_template_file", lambda c: c.judge.chat_template_file),
            ("criteria", lambda c: c.criteria),
            ("n_instructions", lambda c: c.n_instructions),
            ("language", lambda c: c.language),
            ("seed", lambda c: c.seed),
            ("balance_by", lambda c: c.balance_by),
            ("truncate_instruction", lambda c: c.truncate_instruction),
        ],
    )

    if cfg.ignore_score_cache:
        args.ignore_score_cache = True
    if not args.models:
        args.models = []
    return ResolvedTaskConfigPayload(kind="agreement", payload=cfg.to_dict())


def _apply_arena_config(
    args: argparse.Namespace,
    explicit: set[str],
) -> ResolvedTaskConfigPayload:
    from openjury.arena.config import ArenaConfig

    cfg = ArenaConfig.load(args.config)

    apply_mapping_from_config(
        args,
        explicit,
        cfg,
        [
            ("models", lambda c: c.model_names),
            ("judge_model", lambda c: c.judge.model),
            ("model_gpus", lambda c: [m.gpus for m in c.models]),
            (
                "model_quantizations",
                lambda c: [m.quantization or "none" for m in c.models],
            ),
            ("dataset", lambda c: c.dataset),
            ("n_instructions", lambda c: c.n_instructions),
            ("language", lambda c: c.language),
            ("seed", lambda c: c.seed),
            ("balance_by", lambda c: c.balance_by),
            ("criteria", lambda c: c.criteria),
            ("judge_gpus", lambda c: c.judge.gpus),
            ("judge_quantization", lambda c: c.judge.quantization),
            ("judge_mode", lambda c: c.judge.mode),
            ("pairwise_prompt_style", lambda c: c.judge.pairwise_prompt_style),
            (
                "enable_thinking",
                lambda c: (
                    bool(c.judge.enable_thinking)
                    if c.judge.enable_thinking is not None
                    else None
                ),
            ),
            ("chat_template", lambda c: c.judge.chat_template),
            ("chat_template_file", lambda c: c.judge.chat_template_file),
            ("matchmaker", lambda c: c.matchmaker.strategy),
            ("n_matches", lambda c: c.matchmaker.n_matches),
            ("generation_max_tokens", lambda c: c.generation_max_tokens),
            ("truncate_input_chars", lambda c: c.truncate_input_chars),
        ],
        present=present_nonempty,
    )

    if cfg.ignore_cache:
        args.ignore_cache = True
    if cfg.ignore_score_cache:
        args.ignore_score_cache = True
    return ResolvedTaskConfigPayload(kind="arena", payload=cfg.to_dict())


def validate_mode_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate mode-specific required args after config resolution."""
    stage = getattr(args, "stage", "all")
    if args.mode in {"generate", "judge"} and stage != "all":
        parser.error("--stage is only supported for --mode arena or --mode agreement.")

    if args.mode == "arena":
        if not args.models or len(args.models) < 2:
            parser.error(
                "--models with at least 2 models is required for arena mode "
                "(or use --config)"
            )
        if not args.judge_model:
            parser.error(
                "--judge_model is required for arena mode "
                "(or use --config)"
            )
    elif args.mode == "generate":
        if not args.models or len(args.models) < 1:
            parser.error(
                "--models with at least 1 model is required for generate mode"
            )
        if not args.judge_model:
            args.judge_model = "none"
    elif args.mode == "judge":
        if not args.models or len(args.models) < 2:
            parser.error(
                "--models with at least 2 models is required for judge mode "
                "(or use --config)"
            )
        if not args.judge_model:
            parser.error(
                "--judge_model is required for judge mode "
                "(or use --config)"
            )
    elif args.mode == "agreement":
        if not args.judge_model:
            parser.error(
                "--judge_model is required for agreement mode "
                "(or use --config)"
            )
        if not args.models:
            args.models = []

    if not args.dataset:
        parser.error("--dataset is required (or supply it via --config).")
