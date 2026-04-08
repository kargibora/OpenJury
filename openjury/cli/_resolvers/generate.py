"""Resolver for the generate CLI (config loading + local/slurm dispatch)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from openjury._logging import logger
from openjury.arena.config import ModelEntry
from openjury.cli._forwarding.slurm import SlurmForwardRequest
from openjury.generate_config import GenerateConfig, GenerateRuntimeConfig
from openjury.datasets.options import DatasetOptions
from openjury.resolution.argparse_overrides import collect_explicit_dests


@dataclass(frozen=True)
class ResolvedGenerateCLI:
    """Resolved generate CLI inputs ready for execution."""

    config: GenerateConfig
    slurm_forward: SlurmForwardRequest | None = None


def _generate_cli_overrides(args: argparse.Namespace, explicit: set[str]) -> dict[str, Any]:
    """Collect explicit CLI overrides for GenerateConfig."""
    override_fields = {
        "model",
        "dataset",
        "output",
        "n_instructions",
        "language",
        "seed",
        "balance_by",
        "max_tokens",
        "truncate_input_chars",
        "tensor_parallel_size",
        "gpu_memory_utilization",
        "quantization",
        "gpu_devices",
        "chat_template",
        "chat_template_file",
        "api_base_url",
        "api_key_env",
    }
    overrides = {
        field: getattr(args, field)
        for field in override_fields
        if field in explicit
    }
    if "use_tqdm" in explicit and args.use_tqdm:
        overrides["use_tqdm"] = True
    if "base_model" in explicit and args.base_model:
        overrides["base_model"] = True
    if "ignore_cache" in explicit and args.ignore_cache:
        overrides["ignore_cache"] = True
    return overrides


def _apply_generate_cli_overrides(
    config: GenerateConfig,
    args: argparse.Namespace,
    explicit: set[str],
) -> None:
    config.apply_overrides(_generate_cli_overrides(args, explicit))


def _unsupported_generate_slurm_flags(config: GenerateConfig) -> list[str]:
    runtime = config.runtime
    model_entry = config.model_entry
    unsupported = []
    if runtime.base_model:
        unsupported.append("--base_model")
    if runtime.gpu_memory_utilization != 0.9:
        unsupported.append("--gpu_memory_utilization")
    if runtime.gpu_devices is not None:
        unsupported.append("--gpu_devices")
    if model_entry.chat_template is not None:
        unsupported.append("--chat_template")
    if model_entry.chat_template_file is not None:
        unsupported.append("--chat_template_file")
    if runtime.api_base_url is not None:
        unsupported.append("--api_base_url")
    if runtime.api_key_env is not None:
        unsupported.append("--api_key_env")
    return unsupported


def resolve_generate_cli(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argv: list[str] | None = None,
) -> ResolvedGenerateCLI:
    """Resolve generate CLI args into a typed config + optional SLURM request."""
    explicit = collect_explicit_dests(parser, argv)

    if args.submit and not args.slurm:
        parser.error("--submit is only valid together with --slurm.")
    if getattr(args, "detach", False) and not args.slurm:
        parser.error("--detach is only valid together with --slurm.")
    if getattr(args, "detach", False) and not args.submit:
        parser.error("--detach requires --submit.")

    if args.config:
        config = GenerateConfig.load(args.config)
        logger.info("Loaded generate config from %s", args.config)
        _apply_generate_cli_overrides(config, args, explicit)
    else:
        if not args.output:
            parser.error("--output is required when not using --config.")
        config = GenerateConfig(
            model=ModelEntry(
                name=args.model,
                gpus=args.tensor_parallel_size,
                quantization=args.quantization,
                chat_template=args.chat_template,
                chat_template_file=args.chat_template_file,
                max_tokens=args.max_tokens,
            ),
            runtime=GenerateRuntimeConfig(
                truncate_input_chars=args.truncate_input_chars,
                gpu_memory_utilization=args.gpu_memory_utilization,
                gpu_devices=args.gpu_devices,
                api_base_url=args.api_base_url,
                api_key_env=args.api_key_env,
                use_tqdm=args.use_tqdm,
                base_model=args.base_model,
                ignore_cache=args.ignore_cache,
            ),
            dataset=DatasetOptions(
                name=args.dataset,
                n_instructions=args.n_instructions,
                language=args.language,
                seed=args.seed,
                balance_by=args.balance_by,
            ),
            output=args.output,
        )
        config.apply_overrides(_generate_cli_overrides(args, explicit))

    if not config.model_entry.name:
        parser.error("model is required (via CLI or --config).")
    if not config.dataset_options.name:
        parser.error("dataset is required (via CLI or --config).")
    if not config.output and not args.slurm:
        parser.error("output is required for local generation (via CLI or --config).")

    slurm_forward: SlurmForwardRequest | None = None
    if args.slurm:
        unsupported = _unsupported_generate_slurm_flags(config)
        if unsupported:
            parser.error(
                "The following flags are not yet supported with --slurm in "
                "openjury-generate forwarding: "
                + ", ".join(unsupported)
                + ". Use openjury-slurm directly for advanced scheduling options."
            )
        slurm_forward = SlurmForwardRequest(
            mode="generate",
            submit=bool(getattr(args, "submit", False)),
            detach=bool(getattr(args, "detach", False)),
            slurm_output_dir=getattr(args, "slurm_output_dir", "slurm_scripts"),
            tag=getattr(args, "slurm_tag", None),
            warn_output_dir_semantics=True,
        )

    return ResolvedGenerateCLI(config=config, slurm_forward=slurm_forward)
