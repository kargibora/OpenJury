"""Arena step — the unified entry point for K-model evaluation.

Handles the **complete pipeline**: resolve completions (from cache, file,
or generate on-the-fly), run the arena judge, compute ratings, and save
``arena.json``.

Can be driven by a **config file** (JSON or YAML) or CLI arguments::

    # Config file (recommended for ≥3 models)
    uv run python -m openjury.steps.arena_step --config arena.json

    # CLI (convenient for 2-model battles)
    uv run python -m openjury.steps.arena_step \\
        --models VLLM/Qwen/Qwen2.5-0.5B-Instruct \\
                 VLLM/Qwen/Qwen2.5-1.5B-Instruct \\
        --judge_model VLLM/Qwen/Qwen3-32B \\
        --dataset alpaca-eval \\
        --output_dir results/arena/

Minimal config file (arena.json)::

    {
        "dataset": "alpaca-eval",
        "models": [
            "VLLM/Qwen/Qwen2.5-0.5B-Instruct",
            "VLLM/Qwen/Qwen2.5-1.5B-Instruct"
        ],
        "judge": {"model": "VLLM/Qwen/Qwen3-32B", "gpus": 2},
        "output_dir": "results/arena/"
    }

Completion resolution (per model, in order):
    1. ``completions`` field set on model entry → load that parquet file
    2. Completion cache hit → use cached
    3. Cache miss → generate using model backend (VLLM/API), then cache
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from openjury._logging import logger
from openjury.arena.arena_judge import ArenaJudge
from openjury.cli_args import add_arena_pipeline_args, parse_kwargs
from openjury.arena.config import (
    ArenaConfig,
    ArenaResult,
    JudgeConfig,
    MatchmakerConfig,
    ModelEntry,
)
from openjury.arena.matchmaker import get_matchmaker
from openjury.arena.ratings import compute_ratings
from openjury.arena.save_arena import save_arena
from openjury.completion_cache import cache
from openjury.datasets import load_dataset
from openjury.generate import generate_instructions
from openjury.models.factory import make_model
from openjury.rubrics import get_rubric
from openjury.score_cache import score_cache


# ═════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═════════════════════════════════════════════════════════════════════


def run_arena(config: ArenaConfig) -> ArenaResult:
    """Execute the full arena pipeline from an :class:`ArenaConfig`.

    Steps:
        1. Load instructions from the dataset.
        2. Resolve completions for every model (cache → generate → file).
        3. Generate pairwise matches via the matchmaker.
        4. Run the arena judge (samplewise or pairwise).
        5. Compute ratings (BT strengths, Elo, dimension weights).
        6. Assemble and save :class:`ArenaResult`.

    Args:
        config: A fully populated :class:`ArenaConfig`.

    Returns:
        The :class:`ArenaResult` containing all evaluation data.
    """
    # ── 1. Load rubric ───────────────────────────────────────────
    rubric = _load_rubric(config.rubric)

    # ── 2. Load instructions ─────────────────────────────────────
    dataset = load_dataset(config.dataset, n=config.n_instructions)
    instructions = dataset.instructions
    instruction_ids = dataset.instruction_ids
    instruction_metadata = [s.metadata for s in dataset.samples]
    n = len(instructions)
    logger.info("Loaded %d instructions from %s", n, config.dataset)

    # Build a Series for compatibility with cache / generate
    instructions_series = pd.Series(instructions, name="instruction")

    # ── 3. Resolve completions ───────────────────────────────────
    completions = _resolve_completions(
        models=config.models,
        dataset=config.dataset,
        n_instructions=config.n_instructions,
        instructions=instructions_series,
        generation_max_tokens=config.generation_max_tokens,
        truncate_input_chars=config.truncate_input_chars,
        ignore_cache=config.ignore_cache,
    )

    # ── 4. Generate matches ──────────────────────────────────────
    model_names = config.model_names
    instruction_indices = list(range(n))
    matchmaker_fn = get_matchmaker(config.matchmaker.strategy)
    matches = matchmaker_fn(
        models=model_names,
        instruction_indices=instruction_indices,
        n_matches=config.matchmaker.n_matches,
    )
    logger.info(
        "Matchmaker '%s': %d matches for %d models",
        config.matchmaker.strategy, len(matches), config.n_models,
    )

    # ── 5. Build judge model ─────────────────────────────────────
    judge_cfg = config.judge
    judge_model_config = judge_cfg.to_model_config()
    judge_model = make_model(judge_cfg.model, config=judge_model_config)

    # Resolve rubric name key for score caching
    rubric_name_key = config.rubric if not Path(config.rubric).is_file() else rubric.name

    # ── 6. Run arena judge ───────────────────────────────────────
    arena_judge = ArenaJudge(
        judge_model=judge_model,
        rubric=rubric,
        provide_explanation=judge_cfg.provide_explanation,
    )

    model_scores = {}
    if judge_cfg.mode == "samplewise":
        # Resolve per-model scores from cache (score only on miss)
        precomputed = _resolve_scores(
            arena_judge=arena_judge,
            models=config.models,
            completions=completions,
            instructions=instructions,
            judge_model=judge_cfg.model,
            rubric_name=rubric_name_key,
            dataset=config.dataset,
            n_instructions=config.n_instructions,
            ignore_score_cache=config.ignore_score_cache,
        )

        model_scores, match_results = arena_judge.run_samplewise(
            models=model_names,
            completions=completions,
            instructions=instructions,
            matches=matches,
            use_tqdm=True,
            precomputed_scores=precomputed,
        )
    else:
        match_results = arena_judge.run_pairwise(
            completions=completions,
            instructions=instructions,
            matches=matches,
            swap_to_debias=not judge_cfg.no_swap,
            use_tqdm=True,
        )

    # ── 7. Compute ratings ───────────────────────────────────────
    ratings = compute_ratings(
        models=model_names,
        matches=match_results,
        dimension_names=rubric.dimension_names,
        bt_regularization=config.bt_regularization,
        elo_k=config.elo_k,
    )

    # ── 6. Build rubric definition ───────────────────────────────
    rubric_def: dict = {}
    for dim in rubric.dimensions:
        rubric_def[dim.name] = {
            "description": dim.description,
            "scale_min": dim.scale_min,
            "scale_max": dim.scale_max,
            "weight": dim.weight,
        }

    # ── 10. Assemble ArenaResult ─────────────────────────────────
    # Attach instruction_id + metadata to each match result
    for m in match_results:
        idx = m.instruction_index
        if 0 <= idx < n:
            m.instruction_id = instruction_ids[idx]
            m.instruction_metadata = instruction_metadata[idx]

    # Capture the system prompt for reproducibility
    prompt_key = "pairwise" if judge_cfg.mode == "pairwise" else "samplewise"
    system_prompt_used = arena_judge.scorer.system_prompt.get(prompt_key, "")

    arena_result = ArenaResult(
        models=model_names,
        dataset=config.dataset,
        judge_model=judge_cfg.model,
        judge_mode=judge_cfg.mode,
        matchmaker_strategy=config.matchmaker.strategy,
        rubric_name=rubric_name_key,
        rubric_definition=rubric_def,
        n_instructions=n,
        instruction_metadata=instruction_metadata,
        model_scores=model_scores,
        matches=match_results,
        bt_strengths=ratings["bt_strengths"],
        elo_ratings=ratings["elo"],
        win_matrix=ratings["win_matrix"],
        aggregate_win_rates=ratings["aggregate_win_rates"],
        dimension_weights=ratings["dimension_weights"],
        dimension_weight_accuracy=ratings["dimension_weight_accuracy"],
        system_prompt=system_prompt_used,
    )

    # ── 11. Save ─────────────────────────────────────────────────
    save_arena(
        arena_result,
        config.output_dir,
        include_completions=config.include_completions,
        include_raw_judge=config.include_raw_judge,
    )

    # Also save the config alongside the results for reproducibility
    config.save(Path(config.output_dir) / "arena_config.json")

    # ── 12. Print leaderboard ────────────────────────────────────
    _print_leaderboard(arena_result)

    logger.info("Arena evaluation complete. Output: %s", config.output_dir)
    return arena_result


# ═════════════════════════════════════════════════════════════════════
#  Completion resolution
# ═════════════════════════════════════════════════════════════════════


def _resolve_completions(
    models: list[ModelEntry],
    dataset: str,
    n_instructions: int | None,
    instructions: pd.Series,
    generation_max_tokens: int,
    truncate_input_chars: int = 8192,
    ignore_cache: bool = False,
) -> dict[str, pd.DataFrame]:
    """Resolve completions for every model.

    Resolution order per model:
        1. ``model.completions`` is set → load from that parquet file.
        2. Cache hit → use cached completions.
        3. Cache miss → generate using the model backend, then cache.

    For GPU models (VLLM, LlamaCpp), generation loads the model into GPU
    memory, runs inference, then frees the GPU so the next model can load.
    API models (OpenRouter, ChatOpenAI) simply call the remote API.

    Args:
        models: List of :class:`ModelEntry` from the config.
        dataset: Dataset name for cache key.
        n_instructions: Number of instructions (``None`` = all).
        instructions: The instruction Series for generation.
        generation_max_tokens: Default max tokens for generation.
        ignore_cache: If ``True``, always regenerate.

    Returns:
        Dict mapping model name → DataFrame with ``completion`` column.
    """
    completions: dict[str, pd.DataFrame] = {}

    for entry in models:
        model = entry.name

        # ── 1. Pre-specified completions file ────────────────────
        if entry.completions:
            pq = Path(entry.completions)
            if not pq.exists():
                raise FileNotFoundError(
                    f"Completions file not found: {pq} (model: {model})"
                )
            df = pd.read_parquet(pq)
            completions[model] = df
            logger.info(
                "Loaded %d completions for %s from file: %s",
                len(df), entry.short_name, pq,
            )
            continue

        # ── 2 & 3. Cache → generate on miss ─────────────────────
        max_tok = entry.max_tokens or generation_max_tokens

        def _generate(
            _model: str = model,
            _entry: ModelEntry = entry,
            _max_tok: int = max_tok,
        ) -> pd.DataFrame:
            """Generate completions for a single model."""
            # Apply the global generation_max_tokens default if the entry
            # doesn't specify its own max_tokens.
            effective_entry = _entry
            if _entry.max_tokens is None:
                from dataclasses import replace
                effective_entry = replace(_entry, max_tokens=_max_tok)
            model_config = effective_entry.to_model_config()
            return generate_instructions(
                instructions=instructions,
                model=_model,
                max_tokens=_max_tok,
                config=model_config,
                truncate_input_chars=truncate_input_chars,
            )

        df = cache.get_or_generate(
            model=model,
            dataset=dataset,
            n=n_instructions,
            generate_fn=_generate,
            ignore_cache=ignore_cache,
        )
        completions[model] = df
        logger.info(
            "Completions for %s: %d rows (%s)",
            entry.short_name, len(df),
            "cache" if cache.exists(model, dataset, n_instructions) else "generated",
        )

    return completions


# ═════════════════════════════════════════════════════════════════════
#  Score resolution (per-model judge-score caching)
# ═════════════════════════════════════════════════════════════════════


def _resolve_scores(
    arena_judge: ArenaJudge,
    models: list[ModelEntry],
    completions: dict[str, pd.DataFrame],
    instructions: list[str],
    judge_model: str,
    rubric_name: str,
    dataset: str,
    n_instructions: int | None,
    ignore_score_cache: bool,
) -> dict[str, list]:
    """Resolve per-model judge scores, using cache when possible.

    For each model:
        1. Check the score cache for existing scores.
        2. On cache miss, call ``arena_judge.score_model()`` and cache the result.

    This enables **incremental evaluation**: adding a new model to the
    arena only requires scoring the new model — existing scores are
    loaded from cache.

    Args:
        arena_judge: The :class:`ArenaJudge` instance (with rubric + judge model).
        models: List of :class:`ModelEntry` from the config.
        completions: Resolved completions (from ``_resolve_completions``).
        instructions: List of instruction strings.
        judge_model: Judge model specification (for cache key).
        rubric_name: Rubric name (for cache key).
        dataset: Dataset name (for cache key).
        n_instructions: Number of instructions (for cache key).
        ignore_score_cache: If ``True``, always re-score.

    Returns:
        Dict mapping model name → ``list[ModelScore]``.  All models
        are guaranteed to have scores (either cached or freshly computed).
    """
    from openjury.arena.config import ModelScore  # local to avoid circular

    precomputed: dict[str, list[ModelScore]] = {}

    for entry in models:
        model = entry.name

        def _score(
            _model: str = model,
            _df: pd.DataFrame = completions[model],
        ) -> list[ModelScore]:
            return arena_judge.score_model(
                model=_model,
                completions_df=_df,
                instructions=instructions,
            )

        scores = score_cache.get_or_score(
            judge=judge_model,
            rubric=rubric_name,
            model=model,
            dataset=dataset,
            n=n_instructions,
            score_fn=_score,
            ignore_cache=ignore_score_cache,
        )
        precomputed[model] = scores
        logger.info(
            "Scores for %s: %d entries (%s)",
            entry.short_name,
            len(scores),
            "cache" if score_cache.exists(judge_model, rubric_name, model, dataset, n_instructions) else "scored",
        )

    return precomputed


# ═════════════════════════════════════════════════════════════════════
#  Rubric loading
# ═════════════════════════════════════════════════════════════════════


def _load_rubric(rubric_spec: str):
    """Load rubric from a registry name or a JSON file path."""
    if Path(rubric_spec).is_file():
        import json
        from openjury.rubrics.schema import Rubric, RubricDimension

        with open(rubric_spec, encoding="utf-8") as f:
            rdata = json.load(f)
        return Rubric(
            name=rdata.get("name", "custom"),
            description=rdata.get("description", ""),
            dimensions=[RubricDimension(**d) for d in rdata["dimensions"]],
        )
    return get_rubric(rubric_spec)


# ═════════════════════════════════════════════════════════════════════
#  Pretty printing
# ═════════════════════════════════════════════════════════════════════


def _print_leaderboard(result: ArenaResult) -> None:
    """Print the arena leaderboard to the logger."""
    logger.info("")
    logger.info("═" * 70)

    if result.n_models == 2:
        _print_head_to_head(result)
    else:
        _print_k_model_leaderboard(result)

    # Dimension weights — which rubric features predict who wins
    _print_dimension_weights(result)

    logger.info("═" * 70)


def _print_head_to_head(result: ArenaResult) -> None:
    """Print a concise head-to-head accuracy report for 2 models."""
    ma, mb = result.models[0], result.models[1]
    short_a = ma.rsplit("/", 1)[-1]
    short_b = mb.rsplit("/", 1)[-1]

    logger.info("  ⚖️  HEAD-TO-HEAD: %s  vs  %s", short_a, short_b)
    logger.info("  📊 Dataset: %s  |  Judge: %s", result.dataset, result.judge_model)
    logger.info("  📈 %d instructions evaluated", result.n_instructions)
    logger.info("═" * 70)

    # Count wins, losses, ties from match results
    wins_a = sum(1 for m in result.matches if m.preference < 0.4)
    wins_b = sum(1 for m in result.matches if m.preference > 0.6)
    ties = result.n_matches - wins_a - wins_b
    total = max(result.n_matches, 1)

    logger.info("")
    logger.info("    %-35s  %4d  (%5.1f%%)", f"✅ {short_a} wins", wins_a, wins_a / total * 100)
    logger.info("    %-35s  %4d  (%5.1f%%)", f"✅ {short_b} wins", wins_b, wins_b / total * 100)
    logger.info("    %-35s  %4d  (%5.1f%%)", "🤝 Ties", ties, ties / total * 100)
    logger.info("")

    # Determine winner
    if wins_a > wins_b:
        logger.info("    ► %s wins with %.1f%% accuracy", short_a, wins_a / total * 100)
    elif wins_b > wins_a:
        logger.info("    ► %s wins with %.1f%% accuracy", short_b, wins_b / total * 100)
    else:
        logger.info("    ► Draw — both models tied overall")

    # Aggregate win rates
    wr_a = result.aggregate_win_rates.get(ma, 0.5) * 100
    wr_b = result.aggregate_win_rates.get(mb, 0.5) * 100
    logger.info("")
    logger.info("    Aggregate win rate: %s %.1f%%  |  %s %.1f%%", short_a, wr_a, short_b, wr_b)
    logger.info("")


def _print_k_model_leaderboard(result: ArenaResult) -> None:
    """Print the full K-model leaderboard with BT strengths, Elo, and win rate."""
    logger.info("  🏆 ARENA LEADERBOARD")
    logger.info("  📊 Dataset: %s  |  ⚖️ Judge: %s", result.dataset, result.judge_model)
    logger.info("  📈 %d models, %d matches (%s)", result.n_models, result.n_matches, result.matchmaker_strategy)
    logger.info("═" * 70)
    logger.info("  %-4s %-35s %8s %6s %7s", "Rank", "Model", "BT θ", "Elo", "WinR%")
    logger.info("  %s", "-" * 64)

    for rank, (model, bt, elo) in enumerate(result.leaderboard(), 1):
        short = model.rsplit("/", 1)[-1]
        wr = result.aggregate_win_rates.get(model, 0.5) * 100
        logger.info("  %-4d %-35s %+8.4f %6.0f %6.1f%%", rank, short, bt, elo, wr)

    # Win matrix
    logger.info("")
    logger.info("  Win matrix (row wins over column):")
    short_names = [m.rsplit("/", 1)[-1][:15] for m in result.models]
    header = "  " + " " * 16 + "".join(f"{s:>16}" for s in short_names)
    logger.info(header)
    for i, mi in enumerate(result.models):
        row = f"  {short_names[i]:<16}"
        for mj in result.models:
            wr = result.win_matrix.get(mi, {}).get(mj, 0.5)
            row += f"{wr:>15.1%} "
        logger.info(row)


def _print_dimension_weights(result: ArenaResult) -> None:
    """Print rubric dimension weights (shared by 2-model and K-model)."""
    if not result.dimension_weights:
        return
    logger.info("")
    logger.info("  Rubric dimension weights (BT feature importance):")
    for dim, w in sorted(result.dimension_weights.items(), key=lambda x: -abs(x[1])):
        bar = "█" * min(int(abs(w) * 20), 40)
        sign = "+" if w >= 0 else "-"
        logger.info("    %-15s %s%s  %.4f", dim, sign, bar, w)
    logger.info(
        "  Dimension weight accuracy: %.1f%%",
        result.dimension_weight_accuracy * 100,
    )


# ═════════════════════════════════════════════════════════════════════
#  CLI — config file or inline arguments
# ═════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        prog="arena_step",
        description=(
            "Run a K-model arena evaluation.  Supply a config file "
            "(--config) or specify models inline via CLI arguments."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:

  # Config file (recommended)
  uv run python -m openjury.steps.arena_step --config arena.json

  # Inline 2-model battle
  uv run python -m openjury.steps.arena_step \\
      --models VLLM/Qwen/Qwen2.5-0.5B-Instruct \\
               VLLM/Qwen/Qwen2.5-1.5B-Instruct \\
      --judge_model VLLM/Qwen/Qwen3-32B \\
      --dataset alpaca-eval --output_dir results/arena/

  # 5-model arena with budget
  uv run python -m openjury.steps.arena_step \\
      --models M1 M2 M3 M4 M5 \\
      --judge_model OpenRouter/qwen/qwen3-235b-a22b \\
      --dataset alpaca-eval \\
      --matchmaker balanced_random --n_matches 500 \\
      --output_dir results/arena/
""",
    )

    # ── Pipeline args (config, models, dataset, judge, etc.) ────
    add_arena_pipeline_args(parser)

    # Rating config
    parser.add_argument("--bt_regularization", type=float, default=0.01)
    parser.add_argument("--elo_k", type=float, default=32.0)

    # Output options
    parser.add_argument("--include_completions", action="store_true")
    parser.add_argument("--include_raw_judge", action="store_true")

    args = parser.parse_args()

    # Parse --gen-kwargs key=value pairs
    _gen_kwargs = parse_kwargs(getattr(args, "gen_kwargs", None))

    # ── Build ArenaConfig ────────────────────────────────────────
    if args.config:
        config = ArenaConfig.load(args.config)
        logger.info("Loaded arena config from %s", args.config)

        # Allow CLI overrides for key operational fields
        if args.output_dir != "results/arena/":
            config.output_dir = args.output_dir
        if args.n_instructions is not None:
            config.n_instructions = args.n_instructions
        if args.ignore_cache:
            config.ignore_cache = True
        if args.ignore_score_cache:
            config.ignore_score_cache = True
    else:
        # Build config entirely from CLI args
        if not args.models or len(args.models) < 2:
            parser.error(
                "Provide --config OR --models with at least 2 models."
            )
        if not args.judge_model:
            parser.error("--judge_model is required when not using --config.")
        if not args.dataset:
            parser.error("--dataset is required when not using --config.")

        config = ArenaConfig(
            dataset=args.dataset,
            models=[ModelEntry(name=m) for m in args.models],
            judge=JudgeConfig(
                model=args.judge_model,
                gpus=args.judge_gpus,
                mode=args.judge_mode,
                max_tokens=args.judge_max_tokens,
                quantization=args.judge_quantization,
                provide_explanation=args.provide_explanation,
                no_swap=args.no_swap,
                enable_thinking=args.enable_thinking if args.enable_thinking else None,
                generation_kwargs=_gen_kwargs,
            ),
            output_dir=args.output_dir,
            n_instructions=args.n_instructions,
            rubric=args.rubric,
            matchmaker=MatchmakerConfig(
                strategy=args.matchmaker,
                n_matches=args.n_matches,
            ),
            generation_max_tokens=args.generation_max_tokens,
            truncate_input_chars=args.truncate_input_chars,
            ignore_cache=args.ignore_cache,
            ignore_score_cache=args.ignore_score_cache,
            bt_regularization=args.bt_regularization,
            elo_k=args.elo_k,
            include_completions=args.include_completions,
            include_raw_judge=args.include_raw_judge,
        )

    # ── Run ──────────────────────────────────────────────────────
    run_arena(config)


if __name__ == "__main__":
    main()
