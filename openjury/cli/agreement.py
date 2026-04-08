"""Agreement CLI — compare human preferences with LLM judge preferences.

Runs the agreement pipeline on datasets that contain inline completions and
human preference labels (e.g. ``lmsys``, ``comparia``).
"""

from __future__ import annotations

import argparse

from openjury._logging import logger
from openjury.cli._resolvers.agreement import resolve_agreement_cli
from openjury.cli._forwarding.slurm import forward_task_config_to_slurm
from openjury.cli_args import (
    add_dataset_args,
    add_dataset_selection_args,
    add_judge_args,
)
from openjury.datasets import load_dataset
from openjury.pipelines.agreement import run_agreement


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="openjury-agreement",
        description=(
            "Compare human preferences with LLM judge preferences. "
            "Loads a dataset with inline completions and human labels "
            "(e.g. lmsys, comparia), runs the judge on the same pairs, "
            "and saves per-sample results for downstream analysis."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n\n"
            "  # From a config file\n"
            "  uv run openjury-evaluate agreement \\\n"
            "      --config configs/agreement_lmsys.json\n\n"
            "  # Samplewise (default) on LMSys\n"
            "  uv run openjury-evaluate agreement \\\n"
            "      --dataset lmsys --judge_model OpenRouter/qwen/qwen3-32b \\\n"
            "      --n_instructions 200\n\n"
            "  # Pairwise on ComparIA (French)\n"
            "  uv run openjury-evaluate agreement \\\n"
            "      --dataset comparia --judge_model VLLM/Qwen/Qwen3-32B \\\n"
            "      --judge_mode pairwise --language fr\n"
        ),
    )

    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to a JSON/YAML agreement config file. CLI flags can still "
            "override individual settings."
        ),
    )

    add_dataset_args(parser, dataset_required=False)
    add_judge_args(parser)
    parser.add_argument(
        "--output_dir",
        default="results/agreement",
        help="Output directory (default: results/agreement).",
    )
    parser.add_argument(
        "--ignore_score_cache",
        action="store_true",
        help="Force re-scoring by the judge even if scores are cached.",
    )
    add_dataset_selection_args(parser)
    parser.add_argument(
        "--truncate_instruction",
        type=int,
        default=500,
        help=(
            "Truncate instructions in output to this many chars (display only, "
            "full text is judged). Default: 500."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=("all", "annotate", "analyze"),
        default="all",
        help=(
            "Pipeline stage to run: annotate only, analyze only (from saved "
            "agreement_annotations.json), or all (default)."
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
        "--slurm_time",
        default=None,
        help=(
            "With --slurm: wall-time for the judge SLURM job "
            "(e.g. '04:00:00'). Overrides $TIME_LIMIT env var."
        ),
    )

    args = parser.parse_args(argv)
    if args.submit and not args.slurm:
        parser.error("--submit is only valid together with --slurm.")
    if args.detach and not args.slurm:
        parser.error("--detach is only valid together with --slurm.")
    if args.detach and not args.submit:
        parser.error("--detach requires --submit.")
    resolved = resolve_agreement_cli(parser, args, argv)

    # ── Pre-flight dataset check ──────────────────────────────────────
    # Verify the dataset is available (and trigger download if needed)
    # *before* submitting a SLURM job that has no internet access.
    logger.info(
        "Pre-flight: verifying dataset %r is available locally…",
        resolved.config.dataset,
    )
    try:
        load_dataset(resolved.config.dataset, n=5)
        logger.info("Pre-flight: dataset %r OK.", resolved.config.dataset)
    except (FileNotFoundError, KeyError) as exc:
        parser.error(
            f"Dataset {resolved.config.dataset!r} is not available and could "
            f"not be downloaded: {exc}\n"
            f"Ensure the dataset is cached locally before submitting a SLURM job."
        )
    except Exception as exc:
        parser.error(
            f"Dataset {resolved.config.dataset!r} failed to load: "
            f"{type(exc).__name__}: {exc}\n"
            f"Fix the dataset loader before submitting."
        )

    if resolved.slurm_forward is not None:
        if resolved.slurm_forward.warn_output_dir_semantics:
            logger.warning(
                "--slurm forwarding does not preserve local agreement --output_dir semantics "
                "(got %s). Results will be written under the SLURM run directory managed "
                "by openjury-slurm.",
                args.output_dir,
            )
        forward_task_config_to_slurm(
            "agreement",
            resolved.config,
            resolved.slurm_forward,
        )
        return

    run_agreement(resolved.config)


if __name__ == "__main__":
    main()
