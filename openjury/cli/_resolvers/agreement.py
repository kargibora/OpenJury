"""Resolver for the agreement CLI (config loading + explicit CLI overrides)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from openjury._logging import logger
from openjury.arena.config import AgreementConfig, JudgeConfig
from openjury.cli._forwarding.slurm import SlurmForwardRequest
from openjury.cli_args import parse_kwargs
from openjury.resolution.argparse_overrides import (
    apply_mapping_to_config,
    collect_explicit_dests,
)


@dataclass(frozen=True)
class ResolvedAgreementCLI:
    """Resolved agreement CLI inputs ready for execution."""

    config: AgreementConfig
    stage: str
    slurm_forward: SlurmForwardRequest | None = None


def _candidate_auto_config(args: argparse.Namespace) -> str | None:
    if args.stage != "analyze" or args.config:
        return None
    candidate = Path(args.output_dir) / "agreement_config.json"
    return str(candidate) if candidate.exists() else None


def _apply_agreement_cli_overrides(
    config: AgreementConfig,
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
            ("rubric", lambda c, v: setattr(c, "rubric", v)),
            ("truncate_instruction", lambda c, v: setattr(c, "truncate_instruction", v)),
            ("judge_model", lambda c, v: setattr(c.judge, "model", v)),
            ("judge_mode", lambda c, v: setattr(c.judge, "mode", v)),
            ("judge_max_tokens", lambda c, v: setattr(c.judge, "max_tokens", v)),
            ("judge_gpus", lambda c, v: setattr(c.judge, "gpus", v)),
            ("judge_quantization", lambda c, v: setattr(c.judge, "quantization", v)),
            ("chat_template", lambda c, v: setattr(c.judge, "chat_template", v)),
            ("chat_template_file", lambda c, v: setattr(c.judge, "chat_template_file", v)),
        ],
    )

    if "ignore_score_cache" in explicit and args.ignore_score_cache:
        config.ignore_score_cache = True
    if "no_swap" in explicit and args.no_swap:
        config.judge.no_swap = True
    if "provide_explanation" in explicit and args.provide_explanation:
        config.judge.provide_explanation = True
    if "enable_thinking" in explicit and args.enable_thinking:
        config.judge.enable_thinking = True
    if "gen_kwargs" in explicit and gen_kwargs:
        config.judge.generation_kwargs = gen_kwargs


def resolve_agreement_cli(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argv: list[str] | None,
) -> ResolvedAgreementCLI:
    """Resolve agreement CLI inputs into a typed config plus execution request."""
    explicit = collect_explicit_dests(parser, argv)
    gen_kwargs = parse_kwargs(getattr(args, "gen_kwargs", None))

    cfg_path = args.config or _candidate_auto_config(args)
    if cfg_path:
        config = AgreementConfig.load(cfg_path)
        logger.info("Loaded agreement config from %s", cfg_path)
        _apply_agreement_cli_overrides(config, args, explicit, gen_kwargs=gen_kwargs)
        if not config.judge.model:
            parser.error("judge.model is required in the config file.")
    else:
        if not args.dataset:
            parser.error("--dataset is required when not using --config.")
        if not args.judge_model:
            parser.error("--judge_model is required when not using --config.")

        config = AgreementConfig(
            dataset=args.dataset,
            judge=JudgeConfig(
                model=args.judge_model,
                gpus=args.judge_gpus,
                mode=args.judge_mode,
                max_tokens=args.judge_max_tokens,
                temperature=0.0,
                quantization=args.judge_quantization,
                no_swap=args.no_swap,
                provide_explanation=args.provide_explanation,
                enable_thinking=True if args.enable_thinking else None,
                chat_template=args.chat_template,
                chat_template_file=args.chat_template_file,
                generation_kwargs=gen_kwargs or {},
            ),
            output_dir=args.output_dir,
            n_instructions=args.n_instructions,
            language=args.language,
            seed=args.seed,
            balance_by=args.balance_by,
            rubric=args.rubric,
            ignore_score_cache=args.ignore_score_cache,
            truncate_instruction=args.truncate_instruction,
        )

    slurm_forward: SlurmForwardRequest | None = None
    if args.slurm:
        slurm_forward = SlurmForwardRequest(
            mode="agreement",
            submit=bool(getattr(args, "submit", False)),
            detach=bool(getattr(args, "detach", False)),
            slurm_output_dir=getattr(args, "slurm_output_dir", "slurm_scripts"),
            remote=bool(getattr(args, "remote", False)),
            cluster=getattr(args, "cluster", None),
            remote_project_dir=getattr(args, "remote_project_dir", None),
            wait_timeout=getattr(args, "wait_timeout", None),
            stage=args.stage,
            warn_output_dir_semantics=(args.output_dir != "results/agreement"),
        )

    return ResolvedAgreementCLI(
        config=config,
        stage=args.stage,
        slurm_forward=slurm_forward,
    )
