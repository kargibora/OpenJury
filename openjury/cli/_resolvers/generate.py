"""Resolver for the generate CLI (config loading + local/slurm dispatch)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from openjury._logging import logger
from openjury.cli._forwarding.slurm import SlurmForwardRequest
from openjury.generate_config import GenerateConfig
from openjury.resolution.argparse_overrides import (
    apply_mapping_to_config,
    collect_explicit_dests,
)


@dataclass(frozen=True)
class ResolvedGenerateCLI:
    """Resolved generate CLI inputs ready for execution."""

    config: GenerateConfig
    slurm_forward: SlurmForwardRequest | None = None


def _apply_generate_cli_overrides(
    config: GenerateConfig,
    args: argparse.Namespace,
    explicit: set[str],
) -> None:
    apply_mapping_to_config(
        config,
        args,
        explicit,
        [
            ("model", lambda c, v: setattr(c, "model", v)),
            ("dataset", lambda c, v: setattr(c, "dataset", v)),
            ("output", lambda c, v: setattr(c, "output", v)),
            ("n_instructions", lambda c, v: setattr(c, "n_instructions", v)),
            ("language", lambda c, v: setattr(c, "language", v)),
            ("seed", lambda c, v: setattr(c, "seed", v)),
            ("max_tokens", lambda c, v: setattr(c, "max_tokens", v)),
            ("truncate_input_chars", lambda c, v: setattr(c, "truncate_input_chars", v)),
            ("tensor_parallel_size", lambda c, v: setattr(c, "tensor_parallel_size", v)),
            ("gpu_memory_utilization", lambda c, v: setattr(c, "gpu_memory_utilization", v)),
            ("quantization", lambda c, v: setattr(c, "quantization", v)),
            ("gpu_devices", lambda c, v: setattr(c, "gpu_devices", v)),
            ("chat_template", lambda c, v: setattr(c, "chat_template", v)),
            ("chat_template_file", lambda c, v: setattr(c, "chat_template_file", v)),
            ("api_base_url", lambda c, v: setattr(c, "api_base_url", v)),
            ("api_key_env", lambda c, v: setattr(c, "api_key_env", v)),
        ],
    )
    if "use_tqdm" in explicit and args.use_tqdm:
        config.use_tqdm = True
    if "base_model" in explicit and args.base_model:
        config.base_model = True
    if "ignore_cache" in explicit and args.ignore_cache:
        config.ignore_cache = True


def _unsupported_generate_slurm_flags(config: GenerateConfig) -> list[str]:
    unsupported = []
    if config.base_model:
        unsupported.append("--base_model")
    if config.gpu_memory_utilization != 0.9:
        unsupported.append("--gpu_memory_utilization")
    if config.gpu_devices is not None:
        unsupported.append("--gpu_devices")
    if config.chat_template is not None:
        unsupported.append("--chat_template")
    if config.chat_template_file is not None:
        unsupported.append("--chat_template_file")
    if config.api_base_url is not None:
        unsupported.append("--api_base_url")
    if config.api_key_env is not None:
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

    if args.config:
        config = GenerateConfig.load(args.config)
        logger.info("Loaded generate config from %s", args.config)
        _apply_generate_cli_overrides(config, args, explicit)
    else:
        if not args.output:
            parser.error("--output is required when not using --config.")
        config = GenerateConfig(
            model=args.model,
            dataset=args.dataset,
            output=args.output,
            n_instructions=args.n_instructions,
            language=args.language,
            seed=args.seed,
            max_tokens=args.max_tokens,
            truncate_input_chars=args.truncate_input_chars,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            quantization=args.quantization,
            gpu_devices=args.gpu_devices,
            chat_template=args.chat_template,
            chat_template_file=args.chat_template_file,
            api_base_url=args.api_base_url,
            api_key_env=args.api_key_env,
            use_tqdm=args.use_tqdm,
            base_model=args.base_model,
            ignore_cache=args.ignore_cache,
        )

    if not config.model:
        parser.error("model is required (via CLI or --config).")
    if not config.dataset:
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
            slurm_output_dir=getattr(args, "slurm_output_dir", "slurm_scripts"),
            warn_output_dir_semantics=True,
        )

    return ResolvedGenerateCLI(config=config, slurm_forward=slurm_forward)
