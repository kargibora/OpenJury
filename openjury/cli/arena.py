"""Arena CLI — config/CLI parsing + execution dispatch."""

from __future__ import annotations

import argparse

from openjury._logging import logger
from openjury.cli._resolvers.arena import resolve_arena_cli
from openjury.cli._forwarding.slurm import forward_task_config_to_slurm
from openjury.cli_args import add_arena_pipeline_args
from openjury.pipelines.arena import run_arena


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="openjury-arena",
        description=(
            "Run a K-model arena evaluation. Supply a config file (--config) "
            "or specify models inline via CLI arguments."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:

  # Config file (recommended)
  uv run openjury-evaluate arena --config arena.json

  # Inline 2-model battle
  uv run openjury-evaluate arena \\
      --models VLLM/Qwen/Qwen2.5-0.5B-Instruct \\
               VLLM/Qwen/Qwen2.5-1.5B-Instruct \\
      --judge_model VLLM/Qwen/Qwen3-32B \\
      --dataset alpaca-eval --output_dir results/arena/

  # 5-model arena with budget
  uv run openjury-evaluate arena \\
      --models M1 M2 M3 M4 M5 \\
      --judge_model OpenRouter/qwen/qwen3-235b-a22b \\
      --dataset alpaca-eval \\
      --matchmaker balanced_random --n_matches 500 \\
      --output_dir results/arena/
""",
    )

    add_arena_pipeline_args(parser)

    parser.add_argument("--bt_regularization", type=float, default=0.01)
    parser.add_argument("--elo_k", type=float, default=32.0)
    parser.add_argument("--include_completions", action="store_true")
    parser.add_argument("--include_raw_judge", action="store_true")
    parser.add_argument(
        "--stage",
        choices=("all", "annotate", "analyze"),
        default="all",
        help=(
            "Pipeline stage to run: annotate only, analyze only (from saved "
            "arena_annotations.json), or all (default)."
        ),
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
        help="With --slurm: root directory for generated SLURM scripts (default: slurm_scripts).",
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

    args = parser.parse_args(argv)

    if args.submit and not args.slurm:
        parser.error("--submit is only valid together with --slurm.")
    if args.detach and not args.slurm:
        parser.error("--detach is only valid together with --slurm.")
    if args.detach and not args.submit:
        parser.error("--detach requires --submit.")
    resolved = resolve_arena_cli(parser, args, argv)
    if resolved.slurm_forward is not None:
        if resolved.slurm_forward.warn_output_dir_semantics:
            logger.warning(
                "--slurm forwarding does not preserve local arena --output_dir semantics "
                "(got %s). Results will be written under the SLURM run directory managed "
                "by openjury-slurm.",
                args.output_dir,
            )
        forward_task_config_to_slurm("arena", resolved.config, resolved.slurm_forward)
        return

    run_arena(resolved.config)


if __name__ == "__main__":
    main()
