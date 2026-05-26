"""Resolver for the generic annotate CLI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

from openjury.annotate_config import (
    AnnotateConfig,
    AnnotateGenerationConfig,
    AnnotatePairingConfig,
)
from openjury.arena.config import ModelEntry
from openjury.cli._resolvers._judge import apply_judge_cli_overrides, judge_from_args
from openjury.cli._forwarding.slurm import SlurmForwardRequest
from openjury.cli_args import parse_kwargs
from openjury.resolution.argparse_overrides import collect_explicit_dests


@dataclass(frozen=True)
class ResolvedAnnotateCLI:
    config: AnnotateConfig
    slurm_forward: SlurmForwardRequest | None = None


def _models_from_args(args: argparse.Namespace) -> list[ModelEntry] | None:
    if not args.models:
        return None
    return [ModelEntry(name=model) for model in args.models]


def _challenger_from_args(args: argparse.Namespace) -> ModelEntry:
    return ModelEntry(
        name=args.challenger_model,
        gpus=args.challenger_gpus,
        quantization=args.challenger_quantization,
        completions=args.challenger_completions,
        chat_template=args.challenger_chat_template,
        chat_template_file=args.challenger_chat_template_file,
    )


def _update_challenger(
    config: AnnotateConfig,
    args: argparse.Namespace,
    explicit: set[str],
) -> None:
    if config.pairing.source == "matchmaker":
        return

    challenger_fields = {
        "challenger_model",
        "challenger_gpus",
        "challenger_quantization",
        "challenger_completions",
        "challenger_chat_template",
        "challenger_chat_template_file",
    }
    if config.challenger is None and not (challenger_fields & explicit):
        return

    challenger = config.challenger or ModelEntry(name="", gpus=args.challenger_gpus)
    if "challenger_model" in explicit:
        challenger = replace(challenger, name=args.challenger_model)
    if "challenger_gpus" in explicit:
        challenger = replace(challenger, gpus=args.challenger_gpus)
    if "challenger_quantization" in explicit:
        challenger = replace(challenger, quantization=args.challenger_quantization)
    if "challenger_completions" in explicit:
        challenger = replace(challenger, completions=args.challenger_completions)
    if "challenger_chat_template" in explicit:
        challenger = replace(challenger, chat_template=args.challenger_chat_template)
    if "challenger_chat_template_file" in explicit:
        challenger = replace(
            challenger,
            chat_template_file=args.challenger_chat_template_file,
        )
    config.challenger = challenger


def _apply_annotate_cli_overrides(
    config: AnnotateConfig,
    args: argparse.Namespace,
    explicit: set[str],
    *,
    gen_kwargs: dict,
) -> None:
    if "dataset" in explicit:
        config.dataset = args.dataset
    if "n_instructions" in explicit:
        config.n_instructions = args.n_instructions
    if "language" in explicit:
        config.language = args.language
    if "seed" in explicit:
        config.seed = args.seed
    if "balance_by" in explicit:
        config.balance_by = args.balance_by
    if "criteria" in explicit:
        config.criteria = args.criteria
    if "output_dir" in explicit:
        config.output_dir = args.output_dir
    if "ignore_cache" in explicit and args.ignore_cache:
        config.generation.ignore_cache = True
    if "ignore_score_cache" in explicit and args.ignore_score_cache:
        config.ignore_score_cache = True
    if "include_completions" in explicit and args.include_completions:
        config.include_completions = True
    if "include_raw_judge" in explicit and args.include_raw_judge:
        config.include_raw_judge = True

    if "pairing_source" in explicit:
        config.pairing.source = args.pairing_source
        if "pairing_strategy" not in explicit:
            config.pairing.strategy = (
                "dataset_defined"
                if args.pairing_source == "dataset_pairs"
                else (
                    "round_robin"
                    if args.pairing_source == "matchmaker"
                    else "random_single"
                )
            )
    if "pairing_strategy" in explicit:
        config.pairing.strategy = args.pairing_strategy
    if "pairing_n_matches" in explicit:
        config.pairing.n_matches = args.pairing_n_matches
    if "pairing_seed" in explicit:
        config.pairing.seed = args.pairing_seed
    if "pairing_include_models" in explicit:
        config.pairing.include_models = list(args.pairing_include_models or []) or None
    if "pairing_exclude_models" in explicit:
        config.pairing.exclude_models = list(args.pairing_exclude_models or []) or None
    config.pairing.__post_init__()

    if "models" in explicit:
        config.models = _models_from_args(args)

    _update_challenger(config, args, explicit)

    generation = config.generation
    if "generation_max_tokens" in explicit:
        generation.max_tokens = args.generation_max_tokens
    if "truncate_input_chars" in explicit:
        generation.truncate_input_chars = args.truncate_input_chars
    if "gpu_memory_utilization" in explicit:
        generation.gpu_memory_utilization = args.gpu_memory_utilization
    if "gpu_devices" in explicit:
        generation.gpu_devices = args.gpu_devices
    if "api_base_url" in explicit:
        generation.api_base_url = args.api_base_url
    if "api_key_env" in explicit:
        generation.api_key_env = args.api_key_env
    if "use_tqdm" in explicit and args.use_tqdm:
        generation.use_tqdm = True

    apply_judge_cli_overrides(config, args, explicit, gen_kwargs=gen_kwargs)


def resolve_annotate_cli(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argv: list[str] | None,
) -> ResolvedAnnotateCLI:
    explicit = collect_explicit_dests(parser, argv)
    gen_kwargs = parse_kwargs(getattr(args, "gen_kwargs", None))

    if args.config:
        config = AnnotateConfig.load(args.config)
        _apply_annotate_cli_overrides(config, args, explicit, gen_kwargs=gen_kwargs)
    else:
        if not args.dataset:
            parser.error("--dataset is required when not using --config.")
        if not args.judge_model:
            parser.error("--judge_model is required when not using --config.")
        if args.pairing_source == "matchmaker" and (not args.models or len(args.models) < 2):
            parser.error(
                "--pairing_source matchmaker requires --models with at least 2 models."
            )
        challenger = (
            _challenger_from_args(args)
            if args.pairing_source == "challenger_vs_dataset_inline"
            else (
                _challenger_from_args(args)
                if args.challenger_model
                else None
            )
        )
        config = AnnotateConfig(
            dataset=args.dataset,
            judge=judge_from_args(args, gen_kwargs=gen_kwargs),
            challenger=challenger,
            models=_models_from_args(args),
            output_dir=args.output_dir,
            n_instructions=args.n_instructions,
            language=args.language,
            seed=args.seed,
            balance_by=args.balance_by,
            criteria=args.criteria,
            pairing=AnnotatePairingConfig(
                source=args.pairing_source,
                strategy=(
                    args.pairing_strategy
                    or (
                        "dataset_defined"
                        if args.pairing_source == "dataset_pairs"
                        else (
                            "round_robin"
                            if args.pairing_source == "matchmaker"
                            else "random_single"
                        )
                    )
                ),
                seed=args.pairing_seed,
                n_matches=args.pairing_n_matches,
                include_models=list(args.pairing_include_models or []) or None,
                exclude_models=list(args.pairing_exclude_models or []) or None,
            ),
            generation=AnnotateGenerationConfig(
                max_tokens=args.generation_max_tokens,
                truncate_input_chars=args.truncate_input_chars,
                ignore_cache=args.ignore_cache,
                gpu_memory_utilization=args.gpu_memory_utilization,
                gpu_devices=args.gpu_devices,
                api_base_url=args.api_base_url,
                api_key_env=args.api_key_env,
                use_tqdm=args.use_tqdm,
            ),
            ignore_score_cache=args.ignore_score_cache,
            include_completions=args.include_completions,
            include_raw_judge=args.include_raw_judge,
        )
        _apply_annotate_cli_overrides(config, args, explicit, gen_kwargs=gen_kwargs)

    if not config.dataset:
        parser.error("dataset is required (via CLI or --config).")
    if config.requires_challenger and not (config.challenger and config.challenger.name):
        parser.error(
            "challenger model is required when pairing.source is "
            "'challenger_vs_dataset_inline'."
        )
    if config.requires_models and (not config.models or len(config.models) < 2):
        parser.error(
            "pairing.source 'matchmaker' requires at least 2 models "
            "(via --models or --config)."
        )
    slurm_forward: SlurmForwardRequest | None = None
    if getattr(args, "slurm", False):
        slurm_forward = SlurmForwardRequest(
            mode="annotate",
            submit=bool(getattr(args, "submit", False)),
            detach=bool(getattr(args, "detach", False)),
            slurm_output_dir=getattr(args, "slurm_output_dir", "slurm_scripts"),
            tag=getattr(args, "slurm_tag", None),
            container_runtime=getattr(args, "slurm_container_runtime", None),
            container_image=getattr(args, "slurm_container_image", None),
            container_home=getattr(args, "slurm_container_home", None),
            remote=bool(getattr(args, "remote", False)),
            cluster=getattr(args, "cluster", None),
            remote_project_dir=getattr(args, "remote_project_dir", None),
            wait_timeout=getattr(args, "wait_timeout", None),
            warn_output_dir_semantics=(args.output_dir != "results/annotate"),
        )
    return ResolvedAnnotateCLI(config=config, slurm_forward=slurm_forward)
