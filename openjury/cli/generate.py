"""Step 1: Generate completions from a single model and save to parquet.

Designed to run as a standalone SLURM job so that the CUDA context is
fully released between models (no in-process cleanup issues).

Usage::

    uv run openjury-generate \\
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
from openjury.cli._resolvers.generate import resolve_generate_cli
from openjury.cli._forwarding.slurm import forward_task_config_to_slurm
from openjury.cli_args import add_dataset_args, add_dataset_selection_args
from openjury.cache.completions import cache
from openjury.datasets import load_dataset
from openjury.generate_config import GenerateConfig
from openjury.pipelines.generation import generate_instructions, generate_base
from openjury.models.factory import build_config_for_model


def _run_generate_local(cfg: GenerateConfig) -> None:
    """Execute local generation from a resolved config."""
    output_path = Path(cfg.output)
    ds_opts = cfg.dataset_options
    cache_dataset = ds_opts.cache_key()

    # ── Check completion cache first ─────────────────────────────
    if not cfg.ignore_cache:
        cached_df = cache.get(
            cfg.model,
            cache_dataset,
            ds_opts.n_instructions,
        )
        if cached_df is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cached_df.to_parquet(output_path, index=False)
            logger.info(
                "Using cached completions → %s (%d rows)",
                output_path, len(cached_df),
            )
            return

    # ── Load instructions (unified dataset API) ──────────────────
    ds = load_dataset(
        ds_opts.name,
        n=ds_opts.n_instructions,
        **ds_opts.loader_kwargs(),
    )
    instructions = pd.Series(
        [s.instruction for s in ds.samples],
        index=[s.instruction_id for s in ds.samples],
        name="instruction",
    )

    logger.info(
        "Generating %d completions with %s (tp=%d)",
        len(instructions), cfg.model, cfg.tensor_parallel_size,
    )

    # Build provider-appropriate config (VLLM flags are ignored for API providers)
    model_cfg = build_config_for_model(
        cfg.model,
        max_tokens=cfg.max_tokens,
        tensor_parallel_size=cfg.tensor_parallel_size,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        quantization=cfg.quantization,
        gpu_devices=cfg.gpu_devices,
        chat_template=cfg.chat_template,
        chat_template_file=cfg.chat_template_file,
        api_base_url=cfg.api_base_url,
        api_key_env=cfg.api_key_env,
    )

    gen_fn = generate_base if cfg.base_model else generate_instructions
    df = gen_fn(
        instructions=instructions,
        model=cfg.model,
        truncate_input_chars=cfg.truncate_input_chars,
        max_tokens=cfg.max_tokens,
        use_tqdm=cfg.use_tqdm,
        config=model_cfg,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    cache.put(df, cfg.model, cache_dataset, ds_opts.n_instructions)
    logger.info("Saved %d completions to %s (+ cached)", len(df), output_path)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="openjury-generate",
        description="Generate completions from a single model and save to parquet.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a JSON/YAML generate config file. CLI flags can override fields.",
    )
    parser.add_argument("--model", required=False,
                        help="Model spec, e.g. VLLM/meta-llama/Llama-3.1-70B-Instruct")
    add_dataset_args(parser, dataset_required=False)
    add_dataset_selection_args(parser)
    parser.add_argument("--output", required=False,
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
        "--slurm_output_dir",
        default="slurm_scripts",
        help="With --slurm: root directory for generated SLURM scripts (default: slurm_scripts).",
    )

    args = parser.parse_args(argv)
    resolved = resolve_generate_cli(parser, args, argv)

    if resolved.slurm_forward is not None:
        if resolved.slurm_forward.warn_output_dir_semantics:
            logger.warning(
                "--slurm mode ignores --output (%s); completions will be generated "
                "via openjury-slurm into its managed work/cache directories.",
                resolved.config.output,
            )
        forward_task_config_to_slurm("generate", resolved.config, resolved.slurm_forward)
        return

    _run_generate_local(resolved.config)


if __name__ == "__main__":
    main()
