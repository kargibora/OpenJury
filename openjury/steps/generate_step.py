"""Step 1: Generate completions from a single model and save to parquet.

Designed to run as a standalone SLURM job so that the CUDA context is
fully released between models (no in-process cleanup issues).

Usage::

    uv run python -m openjury.steps.generate_step \\
        --model VLLM/meta-llama/Llama-3.1-70B-Instruct \\
        --dataset alpaca-eval \\
        --output /path/to/completions_A.parquet \\
        --tensor_parallel_size 4 \\
        --n_instructions 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from openjury._logging import logger
from openjury.cli_args import add_dataset_args
from openjury.completion_cache import cache
from openjury.generate import generate_instructions, generate_base
from openjury.instruction_dataset import load_instructions
from openjury.models.factory import build_config_for_model


def main():
    parser = argparse.ArgumentParser(
        prog="generate_step",
        description="Generate completions from a single model and save to parquet.",
    )
    parser.add_argument("--model", required=True,
                        help="Model spec, e.g. VLLM/meta-llama/Llama-3.1-70B-Instruct")
    add_dataset_args(parser, dataset_required=True)
    parser.add_argument("--output", required=True,
                        help="Output parquet file path.")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Max tokens to generate per completion (default: 4096).")
    parser.add_argument("--truncate_input_chars", type=int, default=8192)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--gpu_devices", default=None)
    parser.add_argument("--chat_template", default=None,
                        help="Explicit vLLM Jinja chat template override.")
    parser.add_argument("--chat_template_file", default=None,
                        help="Path to a file containing a vLLM chat template override.")
    parser.add_argument("--api_base_url", default=None,
                        help="API base URL (for ChatOpenAI, OpenRouter, LiteLLM providers).")
    parser.add_argument("--api_key_env", default=None,
                        help="Env var name for API key (e.g. OPENAI_API_KEY).")
    parser.add_argument("--use_tqdm", action="store_true")
    parser.add_argument("--base_model", action="store_true",
                        help="Use generate_base (for fluency tasks) instead of generate_instructions.")
    parser.add_argument("--ignore_cache", action="store_true",
                        help="Force regeneration even if completions are cached.")

    args = parser.parse_args()
    output_path = Path(args.output)

    # ── Check completion cache first ─────────────────────────────
    if not args.ignore_cache:
        cached_df = cache.get(args.model, args.dataset, args.n_instructions)
        if cached_df is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cached_df.to_parquet(output_path, index=False)
            logger.info(
                "Using cached completions → %s (%d rows)",
                output_path, len(cached_df),
            )
            return

    # ── Load instructions ────────────────────────────────────────
    instructions_df = load_instructions(
        dataset=args.dataset, n_instructions=args.n_instructions,
    )
    instructions = instructions_df["instruction"]
    if args.n_instructions:
        instructions = instructions[: args.n_instructions]

    logger.info(
        "Generating %d completions with %s (tp=%d)",
        len(instructions), args.model, args.tensor_parallel_size,
    )

    # Build provider-appropriate config (VLLM flags are ignored for API providers)
    config = build_config_for_model(
        args.model,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        quantization=args.quantization,
        gpu_devices=args.gpu_devices,
        chat_template=args.chat_template,
        chat_template_file=args.chat_template_file,
        api_base_url=args.api_base_url,
        api_key_env=args.api_key_env,
    )

    # Generate
    gen_fn = generate_base if args.base_model else generate_instructions
    df = gen_fn(
        instructions=instructions,
        model=args.model,
        truncate_input_chars=args.truncate_input_chars,
        max_tokens=args.max_tokens,
        use_tqdm=args.use_tqdm,
        config=config,
    )

    # Save + cache
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    cache.put(df, args.model, args.dataset, args.n_instructions)
    logger.info("Saved %d completions to %s (+ cached)", len(df), output_path)


if __name__ == "__main__":
    main()
