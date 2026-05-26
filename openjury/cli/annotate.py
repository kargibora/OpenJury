"""Generic annotation CLI."""

from __future__ import annotations

import argparse

from openjury._logging import logger
from openjury.cli._resolvers.annotate import resolve_annotate_cli
from openjury.cli._forwarding.slurm import forward_task_config_to_slurm
from openjury.cli_args import (
    add_cache_args,
    add_dataset_args,
    add_dataset_selection_args,
    add_generation_args,
    add_judge_args,
    add_models_arg,
    add_output_dir_arg,
)
from openjury.pipelines.model_annotation import run_annotate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openjury-annotate",
        description=(
            "Create judge annotations from existing dataset pairs or from a "
            "challenger model aligned to dataset-provided completions."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n\n"
            "  # Dataset-defined pairs (LMSYS/comparia-style)\n"
            "  uv run openjury-evaluate annotate \\\n"
            "      --dataset lmsys \\\n"
            "      --pairing_source dataset_pairs \\\n"
            "      --judge_model VLLM/Qwen/Qwen3-32B \\\n"
            "      --judge_mode pairwise\n\n"
            "  # Challenger vs dataset inline completions\n"
            "  uv run openjury-evaluate annotate \\\n"
            "      --dataset lmsys \\\n"
            "      --pairing_source challenger_vs_dataset_inline \\\n"
            "      --challenger_model VLLM/openai/gpt-oss-20b \\\n"
            "      --judge_model VLLM/Qwen/Qwen3-32B\n\n"
            "  # Matchmaker annotation over a model pool\n"
            "  uv run openjury-evaluate annotate \\\n"
            "      --dataset alpaca-eval \\\n"
            "      --pairing_source matchmaker \\\n"
            "      --models VLLM/model-a VLLM/model-b VLLM/model-c \\\n"
            "      --pairing_strategy balanced_random \\\n"
            "      --n_matches 300 \\\n"
            "      --judge_model VLLM/Qwen/Qwen3-32B\n"
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a JSON/YAML annotate config file. CLI flags can override fields.",
    )
    add_dataset_args(parser, dataset_required=False)
    add_dataset_selection_args(parser)
    add_models_arg(parser)
    parser.add_argument("--challenger_model", default=None, help="Optional challenger model.")
    parser.add_argument(
        "--challenger_gpus",
        type=int,
        default=1,
        help="Number of GPUs for the challenger model (default: 1).",
    )
    parser.add_argument(
        "--challenger_quantization",
        default=None,
        help="Quantization method for the challenger model.",
    )
    parser.add_argument(
        "--challenger_completions",
        default=None,
        help=(
            "Optional parquet file with unified challenger completions. "
            "If omitted and the pairing mode requires a challenger, completions "
            "are generated or loaded from the completion cache."
        ),
    )
    parser.add_argument(
        "--challenger_chat_template",
        default=None,
        help="Explicit vLLM Jinja chat template override for the challenger model.",
    )
    parser.add_argument(
        "--challenger_chat_template_file",
        default=None,
        help="Path to a file containing a vLLM chat template override for the challenger model.",
    )
    add_judge_args(parser)
    add_generation_args(parser)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--gpu_devices", default=None)
    parser.add_argument("--api_base_url", default=None)
    parser.add_argument("--api_key_env", default=None)
    parser.add_argument("--use_tqdm", action="store_true")
    parser.add_argument(
        "--pairing_source",
        "--opponent_source",
        dest="pairing_source",
        choices=["dataset_pairs", "challenger_vs_dataset_inline", "matchmaker"],
        default="challenger_vs_dataset_inline",
        help=(
            "How to build annotation pairs: use dataset-defined pairs directly, "
            "compare a challenger model against dataset inline completions, "
            "or sample battles from a model pool."
        ),
    )
    parser.add_argument(
        "--pairing_strategy",
        "--opponent_strategy",
        dest="pairing_strategy",
        choices=[
            "dataset_defined",
            "random_single",
            "all",
            "round_robin",
            "random_pairs",
            "balanced_random",
        ],
        default=None,
        help=(
            "Pairing strategy. Use dataset_defined for dataset_pairs, and "
            "random_single/all for challenger_vs_dataset_inline, and "
            "round_robin/random_pairs/balanced_random for matchmaker."
        ),
    )
    parser.add_argument(
        "--pairing_n_matches",
        "--n_matches",
        dest="pairing_n_matches",
        type=int,
        default=None,
        help="Optional match count for matchmaker pairing strategies.",
    )
    parser.add_argument(
        "--pairing_seed",
        "--opponent_seed",
        dest="pairing_seed",
        type=int,
        default=42,
        help="Seed for any randomized pairing strategy.",
    )
    parser.add_argument(
        "--pairing_include_models",
        "--opponent_include_models",
        dest="pairing_include_models",
        nargs="*",
        default=None,
        help="Optional allowlist of dataset models to keep during pair construction.",
    )
    parser.add_argument(
        "--pairing_exclude_models",
        "--opponent_exclude_models",
        dest="pairing_exclude_models",
        nargs="*",
        default=None,
        help="Optional blocklist of dataset models to exclude during pair construction.",
    )
    add_cache_args(parser)
    add_output_dir_arg(parser, default="results/annotate")
    parser.add_argument(
        "--include_completions",
        action="store_true",
        help="Persist completion texts in the saved annotation artifact.",
    )
    parser.add_argument(
        "--include_raw_judge",
        action="store_true",
        help="Persist raw judge outputs in the saved annotation artifact.",
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="Generate SLURM scripts instead of running locally (uses openjury-slurm).",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="With --slurm: submit generated scripts immediately.",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help="With --slurm --submit: run submission in background and return immediately.",
    )
    parser.add_argument(
        "--slurm_output_dir",
        default="slurm_scripts",
        help="With --slurm: root directory for generated SLURM scripts.",
    )
    parser.add_argument(
        "--slurm_tag",
        default=None,
        help=(
            "With --slurm: explicit run tag used in the generated SLURM run "
            "directory name. Defaults to an auto-generated timestamp."
        ),
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help=(
            "With --slurm: submit to a remote cluster via slurmpilot instead of "
            "the local cluster."
        ),
    )
    parser.add_argument(
        "--cluster",
        default=None,
        help="With --slurm --remote: slurmpilot cluster name.",
    )
    parser.add_argument(
        "--remote_project_dir",
        default=None,
        help=(
            "With --slurm --remote: path to the OpenJury project on the remote "
            "cluster."
        ),
    )
    parser.add_argument(
        "--wait_timeout",
        type=int,
        default=7200,
        help=(
            "With --slurm --remote: max seconds to wait per submitted job "
            "(default: 7200)."
        ),
    )
    parser.add_argument(
        "--slurm_container_runtime",
        choices=["none", "apptainer"],
        default=None,
        help="With --slurm: override the container runtime for eligible compute jobs.",
    )
    parser.add_argument(
        "--slurm_container_image",
        default=None,
        help="With --slurm: override the configured Apptainer/Singularity image path.",
    )
    parser.add_argument(
        "--slurm_container_home",
        default=None,
        help="With --slurm: override the persistent writable container home.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.submit and not args.slurm:
        parser.error("--submit is only valid together with --slurm.")
    if args.detach and not args.slurm:
        parser.error("--detach is only valid together with --slurm.")
    if args.remote and not args.slurm:
        parser.error("--remote is only valid together with --slurm.")
    if args.detach and not args.submit:
        parser.error("--detach requires --submit.")
    if args.detach and args.remote:
        parser.error("--detach is not supported together with --remote.")
    resolved = resolve_annotate_cli(parser, args, argv)
    if resolved.slurm_forward is not None:
        if resolved.slurm_forward.warn_output_dir_semantics:
            logger.warning(
                "--slurm forwarding does not preserve local annotate --output_dir semantics "
                "(got %s). Results will be written under the SLURM run directory managed "
                "by openjury-slurm.",
                args.output_dir,
            )
        forward_task_config_to_slurm(
            "annotate",
            resolved.config,
            resolved.slurm_forward,
        )
        return
    run_annotate(resolved.config)


if __name__ == "__main__":
    main()
