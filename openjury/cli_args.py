"""Shared CLI argument builders for OpenJury entry-points.

Quick usage — one import, one call::

    from openjury.cli_args import add_arena_pipeline_args

    parser = argparse.ArgumentParser(...)
    add_arena_pipeline_args(parser)   # adds --config, --models,
                                      # --dataset, --judge_*, etc.

This registers all the standard flags needed for an arena evaluation
pipeline (``arena_step.py``, ``generate_slurm.py``, etc.).

For scripts that only need a subset (e.g. ``generate_step.py``),
individual ``add_*_args()`` helpers are still available.
"""

from __future__ import annotations

import argparse
from typing import Any


# ═════════════════════════════════════════════════════════════════════
#  Key=value kwarg parser
# ═════════════════════════════════════════════════════════════════════

def parse_kwargs(pairs: list[str] | None) -> dict[str, Any]:
    """Parse ``key=value`` pairs into a dict with automatic type coercion.

    Used by the ``--gen-kwargs`` CLI flag to allow arbitrary backend
    parameters without defining one ``--flag`` per knob.

    Type coercion rules (applied in order):

    - ``true`` / ``yes`` / ``false`` / ``no`` → ``bool``
    - ``none`` / ``null`` → ``None``
    - Integer literal → ``int``
    - Float literal → ``float``
    - Everything else → ``str``

    Examples::

        parse_kwargs(["temperature=0.7", "top_k=50", "do_sample=true"])
        # → {"temperature": 0.7, "top_k": 50, "do_sample": True}

    Args:
        pairs: List of ``"key=value"`` strings, or ``None``.

    Returns:
        Dict of parsed kwargs.  Empty dict if *pairs* is ``None``.

    Raises:
        argparse.ArgumentTypeError: If a pair is missing ``=``.
    """
    if not pairs:
        return {}
    result: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise argparse.ArgumentTypeError(
                f"Invalid kwarg format: '{pair}'. Expected key=value."
            )
        key, val_str = pair.split("=", 1)
        key = key.strip()
        val_str = val_str.strip()

        # Auto-coerce types
        if val_str.lower() in ("true", "yes"):
            result[key] = True
        elif val_str.lower() in ("false", "no"):
            result[key] = False
        elif val_str.lower() in ("none", "null"):
            result[key] = None
        else:
            try:
                result[key] = int(val_str)
            except ValueError:
                try:
                    result[key] = float(val_str)
                except ValueError:
                    result[key] = val_str
    return result


# ═════════════════════════════════════════════════════════════════════
#  Dataset
# ═════════════════════════════════════════════════════════════════════

def add_dataset_args(
    parser: argparse.ArgumentParser,
    *,
    dataset_required: bool = False,
) -> None:
    """``--dataset`` and ``--n_instructions``."""
    parser.add_argument(
        "--dataset",
        default=None,
        required=dataset_required,
        help="Dataset name (e.g. alpaca-eval, arena-hard).",
    )
    parser.add_argument(
        "--n_instructions",
        type=int,
        default=None,
        help="Limit to the first N instructions from the dataset.",
    )


def add_dataset_selection_args(parser: argparse.ArgumentParser) -> None:
    """Optional dataset selection / filtering flags shared across tasks.

    These options are only used by datasets/loaders that support them.
    """
    parser.add_argument(
        "--language",
        default=None,
        help="Dataset language filter (where supported, e.g. comparia/lmsys variants).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Dataset sub-sampling seed (where supported). Default: 42.",
    )
    parser.add_argument(
        "--balance_by",
        default=None,
        help=(
            "Balanced sub-sampling metadata field (e.g. 'lang'). When set with "
            "--n_instructions, the loader samples approximately uniformly across "
            "field values and redistributes deficits automatically."
        ),
    )


# ═════════════════════════════════════════════════════════════════════
#  Judge model
# ═════════════════════════════════════════════════════════════════════

def add_judge_args(parser: argparse.ArgumentParser) -> None:
    """Arguments that control the LLM judge."""
    parser.add_argument(
        "--judge_model",
        default=None,
        help="Judge model spec, e.g. OpenRouter/qwen/qwen3-32b.",
    )
    parser.add_argument(
        "--judge_mode",
        choices=["samplewise", "pairwise"],
        default="samplewise",
        help="Judge evaluation mode. 'samplewise' (default): scores each "
             "model independently. 'pairwise': judge sees A and B together.",
    )
    parser.add_argument(
        "--pairwise_prompt_style",
        choices=["rubric", "legacy"],
        default="rubric",
        help=(
            "Prompt format for pairwise judging. 'rubric' (default) asks for "
            "dimension-level JSON scores. 'legacy' reuses the old overall "
            "A/B scoring prompt and is intended for single-dimension rubrics "
            "such as 'overall'."
        ),
    )
    parser.add_argument(
        "--judge_max_tokens",
        type=int,
        default=2048,
        help="Max tokens for the judge model (default: 2048).",
    )
    parser.add_argument(
        "--judge_gpus",
        type=int,
        default=1,
        help="Number of GPUs for the judge model (default: 1).",
    )
    parser.add_argument(
        "--judge_quantization",
        default=None,
        choices=["awq", "gptq", "squeezellm", "fp8"],
        help="Quantization method for the judge model.",
    )
    parser.add_argument(
        "--no_swap",
        action="store_true",
        help="Disable position-swap debiasing in pairwise mode.",
    )
    parser.add_argument(
        "--provide_explanation",
        action="store_true",
        help="Ask the judge to provide a textual explanation.",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        default=False,
        help="Enable thinking/reasoning mode (e.g. Qwen3 <think> tokens). "
             "Only affects VLLM backends that support it.",
    )
    parser.add_argument(
        "--chat_template",
        default=None,
        help=(
            "Explicit vLLM Jinja chat template override for the judge model. "
            "Ignored by non-VLLM providers."
        ),
    )
    parser.add_argument(
        "--chat_template_file",
        default=None,
        help=(
            "Path to a file containing a vLLM chat template override for the "
            "judge model. Ignored if --chat_template is provided."
        ),
    )
    parser.add_argument(
        "--gen-kwargs",
        nargs="*",
        metavar="KEY=VALUE",
        default=None,
        help=(
            "Extra generation kwargs passed through to the backend. "
            "For VLLM: merged into SamplingParams. For API: merged into "
            "the request body.  Example: "
            "--gen-kwargs temperature=0.7 top_p=0.9 top_k=50 "
            "repetition_penalty=1.05"
        ),
    )
    parser.add_argument(
        "--rubric",
        default="default",
        help="Rubric name for judge evaluation (default: 'default').",
    )


# ═════════════════════════════════════════════════════════════════════
#  Generation
# ═════════════════════════════════════════════════════════════════════

def add_generation_args(parser: argparse.ArgumentParser) -> None:
    """Arguments for model completion generation."""
    parser.add_argument(
        "--generation_max_tokens",
        type=int,
        default=4096,
        help="Max tokens for generation models (default: 4096).",
    )
    parser.add_argument(
        "--truncate_input_chars",
        type=int,
        default=8192,
        help="Truncate input instructions to this many characters "
             "(default: 8192).",
    )


# ═════════════════════════════════════════════════════════════════════
#  Arena matchmaker
# ═════════════════════════════════════════════════════════════════════

def add_matchmaker_args(parser: argparse.ArgumentParser) -> None:
    """``--matchmaker`` and ``--n_matches``."""
    parser.add_argument(
        "--matchmaker",
        default="round_robin",
        choices=["round_robin", "random_pairs", "balanced_random"],
        help="Matchmaking strategy for arena mode (default: round_robin).",
    )
    parser.add_argument(
        "--n_matches",
        type=int,
        default=None,
        help="Max matches budget for arena mode "
             "(random_pairs / balanced_random).",
    )


# ═════════════════════════════════════════════════════════════════════
#  Cache control
# ═════════════════════════════════════════════════════════════════════

def add_cache_args(parser: argparse.ArgumentParser) -> None:
    """``--ignore_cache`` and ``--ignore_score_cache``."""
    parser.add_argument(
        "--ignore_cache",
        action="store_true",
        help="Force regeneration of completions even if cached.",
    )
    parser.add_argument(
        "--ignore_score_cache",
        action="store_true",
        help="Force re-scoring by the judge even if scores are cached. "
             "Only applies to arena samplewise mode.",
    )


# ═════════════════════════════════════════════════════════════════════
#  Arena config file
# ═════════════════════════════════════════════════════════════════════

def add_arena_config_arg(parser: argparse.ArgumentParser) -> None:
    """``--config`` (path to arena config JSON / YAML)."""
    parser.add_argument(
        "--config",
        default=None,
        help="Path to arena config file (JSON or YAML). "
             "When provided, most other CLI flags are ignored.",
    )


# ═════════════════════════════════════════════════════════════════════
#  Model list
# ═════════════════════════════════════════════════════════════════════

def add_models_arg(parser: argparse.ArgumentParser) -> None:
    """``--models`` (one or more model specs)."""
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to evaluate (e.g. VLLM/meta-llama/Llama-3.1-70B-Instruct).",
    )


# ═════════════════════════════════════════════════════════════════════
#  Output
# ═════════════════════════════════════════════════════════════════════

def add_output_dir_arg(
    parser: argparse.ArgumentParser,
    *,
    default: str = "results/arena/",
) -> None:
    """``--output_dir``."""
    parser.add_argument(
        "--output_dir",
        default=default,
        help=f"Output directory (default: {default}).",
    )


# ═════════════════════════════════════════════════════════════════════
#  Unified builder — one call for the full arena pipeline
# ═════════════════════════════════════════════════════════════════════

def add_arena_pipeline_args(
    parser: argparse.ArgumentParser,
    *,
    output_dir_default: str = "results/arena/",
) -> None:
    """Register all standard arena-pipeline arguments in one call.

    Adds: ``--config``, ``--models``, ``--dataset``, ``--n_instructions``,
    ``--judge_model``, ``--judge_mode``, ``--judge_max_tokens``,
    ``--judge_gpus``, ``--judge_quantization``, ``--no_swap``,
    ``--provide_explanation``, ``--rubric``, ``--generation_max_tokens``,
    ``--truncate_input_chars``, ``--matchmaker``, ``--n_matches``,
    ``--ignore_cache``, ``--ignore_score_cache``, ``--output_dir``.

    Args:
        parser: The argument parser to populate.
        output_dir_default: Default value for ``--output_dir``.
    """
    add_arena_config_arg(parser)
    add_models_arg(parser)
    add_dataset_args(parser)
    add_dataset_selection_args(parser)
    add_judge_args(parser)
    add_generation_args(parser)
    add_matchmaker_args(parser)
    add_cache_args(parser)
    add_output_dir_arg(parser, default=output_dir_default)
