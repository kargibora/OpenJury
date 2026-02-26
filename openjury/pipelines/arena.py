"""Arena pipeline orchestration (generate/annotate/analyze)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from openjury._logging import logger
from openjury.arena.arena_judge import ArenaJudge
from openjury.arena.config import (
    ArenaConfig,
    ArenaResult,
    MatchResult,
    ModelScore,
    ModelEntry,
)
from openjury.arena.matchmaker import get_matchmaker
from openjury.arena.ratings import compute_ratings
from openjury.arena.save_arena import save_arena
from openjury.cache.completions import cache
from openjury.datasets import load_dataset
from openjury.pipelines.generation import generate_instructions
from openjury.models.factory import make_model
from openjury.rubrics import get_rubric
from openjury.cache.scores import score_cache
from openjury.pipelines.annotate import load_arena_annotations, save_arena_annotations


# ═════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═════════════════════════════════════════════════════════════════════


def _annotate_arena(config: ArenaConfig) -> dict[str, Any]:
    """Run generation + judging and return annotation payload for analysis."""
    rubric = _load_rubric(config.rubric)

    ds_opts = config.dataset_options
    dataset = load_dataset(ds_opts.name, n=ds_opts.n_instructions, **ds_opts.loader_kwargs())
    instructions = dataset.instructions
    instruction_ids = dataset.instruction_ids
    instruction_metadata = [s.metadata for s in dataset.samples]
    n = len(instructions)
    logger.info("Loaded %d instructions from %s", n, config.dataset)

    instructions_series = pd.Series(instructions, name="instruction")
    completions = _resolve_completions(
        models=config.models,
        dataset=config.dataset,
        n_instructions=config.n_instructions,
        instructions=instructions_series,
        generation_max_tokens=config.generation_max_tokens,
        truncate_input_chars=config.truncate_input_chars,
        ignore_cache=config.ignore_cache,
    )

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
        config.matchmaker.strategy,
        len(matches),
        config.n_models,
    )

    judge_cfg = config.judge
    judge_model_config = judge_cfg.to_model_config()
    judge_model = make_model(judge_cfg.model, config=judge_model_config)
    rubric_name_key = config.rubric if not Path(config.rubric).is_file() else rubric.name

    arena_judge = ArenaJudge(
        judge_model=judge_model,
        rubric=rubric,
        provide_explanation=judge_cfg.provide_explanation,
    )

    model_scores: dict[str, list[ModelScore]] = {}
    if judge_cfg.mode == "samplewise":
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

    rubric_def: dict[str, Any] = {}
    for dim in rubric.dimensions:
        rubric_def[dim.name] = {
            "description": dim.description,
            "scale_min": dim.scale_min,
            "scale_max": dim.scale_max,
            "weight": dim.weight,
            **({"score_references": dim.score_references} if dim.score_references else {}),
        }

    for m in match_results:
        idx = m.instruction_index
        if 0 <= idx < n:
            m.instruction_id = instruction_ids[idx]
            m.instruction_metadata = instruction_metadata[idx]

    prompt_key = "pairwise" if judge_cfg.mode == "pairwise" else "samplewise"
    system_prompt_used = arena_judge.scorer.system_prompt.get(prompt_key, "")

    return {
        "metadata": {
            "models": model_names,
            "dataset": config.dataset,
            "judge_model": judge_cfg.model,
            "judge_mode": judge_cfg.mode,
            "matchmaker": config.matchmaker.strategy,
            "rubric": rubric_name_key,
            "n_instructions": n,
        },
        "rubric_definition": rubric_def,
        "dimension_names": list(rubric.dimension_names),
        "instruction_metadata": instruction_metadata,
        "model_scores": {
            model: [
                {
                    "model": s.model,
                    "instruction_index": s.instruction_index,
                    "scores": s.scores,
                    "sample_id": s.sample_id,
                    "completion": s.completion,
                    "raw_judge_output": s.raw_judge_output,
                }
                for s in scores
            ]
            for model, scores in model_scores.items()
        },
        "matches": [
            {
                "model_a": m.model_a,
                "model_b": m.model_b,
                "instruction_index": m.instruction_index,
                "scores_a": m.scores_a,
                "scores_b": m.scores_b,
                "preference": m.preference,
                **({"instruction_id": m.instruction_id} if m.instruction_id else {}),
                **({"instruction_metadata": m.instruction_metadata} if m.instruction_metadata else {}),
            }
            for m in match_results
        ],
        "system_prompt": system_prompt_used,
    }


def _load_arena_annotation_runtime(output_dir: str | Path) -> dict[str, Any]:
    """Load a saved arena annotation artifact and reconstruct runtime objects."""
    data = load_arena_annotations(output_dir)
    model_scores: dict[str, list[ModelScore]] = {}
    for model, rows in data.get("model_scores", {}).items():
        model_scores[model] = [
            ModelScore(
                model=row.get("model", model),
                instruction_index=row["instruction_index"],
                scores=row["scores"],
                sample_id=row.get("sample_id", ""),
                completion=row.get("completion", ""),
                raw_judge_output=row.get("raw_judge_output", ""),
            )
            for row in rows
        ]
    matches = [
        MatchResult(
            model_a=m["model_a"],
            model_b=m["model_b"],
            instruction_index=m["instruction_index"],
            scores_a=m["scores_a"],
            scores_b=m["scores_b"],
            preference=m["preference"],
            instruction=m.get("instruction", ""),
            instruction_id=m.get("instruction_id", ""),
            instruction_metadata=m.get("instruction_metadata", {}),
            completion_a=m.get("completion_a", ""),
            completion_b=m.get("completion_b", ""),
            raw_judge_output=m.get("raw_judge_output", ""),
            raw_judge_output_swapped=m.get("raw_judge_output_swapped"),
        )
        for m in data.get("matches", [])
    ]
    return {
        "metadata": data["metadata"],
        "rubric_definition": data.get("rubric_definition", {}),
        "dimension_names": data.get("dimension_names", []),
        "instruction_metadata": data.get("instruction_metadata", []),
        "model_scores": model_scores,
        "matches": matches,
        "system_prompt": data.get("system_prompt", ""),
    }


def _analyze_arena_annotations(
    config: ArenaConfig,
    ann: dict[str, Any],
) -> ArenaResult:
    """Compute ratings and save arena outputs from annotation payload."""
    meta = ann["metadata"]
    model_names = list(meta["models"])
    match_results: list[MatchResult] = ann["matches"]
    dimension_names = ann.get("dimension_names", [])

    ratings = compute_ratings(
        models=model_names,
        matches=match_results,
        dimension_names=dimension_names,
        bt_regularization=config.bt_regularization,
        elo_k=config.elo_k,
    )

    arena_result = ArenaResult(
        models=model_names,
        dataset=meta["dataset"],
        judge_model=meta["judge_model"],
        judge_mode=meta["judge_mode"],
        matchmaker_strategy=meta["matchmaker"],
        rubric_name=meta["rubric"],
        rubric_definition=ann.get("rubric_definition", {}),
        n_instructions=meta["n_instructions"],
        instruction_metadata=ann.get("instruction_metadata", []),
        model_scores=ann.get("model_scores", {}),
        matches=match_results,
        bt_strengths=ratings["bt_strengths"],
        elo_ratings=ratings["elo"],
        win_matrix=ratings["win_matrix"],
        aggregate_win_rates=ratings["aggregate_win_rates"],
        dimension_weights=ratings["dimension_weights"],
        dimension_weight_accuracy=ratings["dimension_weight_accuracy"],
        system_prompt=ann.get("system_prompt", ""),
    )

    save_arena(
        arena_result,
        config.output_dir,
        include_completions=config.include_completions,
        include_raw_judge=config.include_raw_judge,
    )
    config.save(Path(config.output_dir) / "arena_config.json")
    _print_leaderboard(arena_result)
    logger.info("Arena evaluation complete. Output: %s", config.output_dir)
    return arena_result


def run_arena(config: ArenaConfig, stage: str = "all") -> ArenaResult | None:
    """Execute the arena pipeline with stage control."""
    if stage not in {"all", "annotate", "analyze"}:
        raise ValueError(f"Unsupported arena stage: {stage}")

    if stage == "analyze":
        ann_runtime = _load_arena_annotation_runtime(config.output_dir)
        return _analyze_arena_annotations(config, ann_runtime)

    ann_payload = _annotate_arena(config)
    ann_path = save_arena_annotations(config.output_dir, ann_payload)
    config.save(Path(config.output_dir) / "arena_config.json")
    logger.info("Saved arena annotations: %s", ann_path)

    if stage == "annotate":
        logger.info("Arena annotate stage complete. Run --stage analyze to compute ratings.")
        return None

    ann_runtime = _load_arena_annotation_runtime(config.output_dir)
    return _analyze_arena_annotations(config, ann_runtime)


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

        was_cached = (not ignore_cache) and cache.exists(model, dataset, n_instructions)
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
            "cache" if was_cached else "generated",
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

        was_cached = (
            not ignore_score_cache
            and score_cache.exists(judge_model, rubric_name, model, dataset, n_instructions)
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
            "cache" if was_cached else "scored",
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
