"""Arena-specific analysis stage helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from openjury._logging import logger
from openjury.arena.config import ArenaConfig, ArenaResult, MatchResult
from openjury.arena.ratings import compute_ratings
from openjury.arena.save_arena import save_arena


def analyze_arena_annotations(
    *,
    config: ArenaConfig,
    ann: dict[str, Any],
    leaderboard_printer: Callable[[ArenaResult], None] | None = None,
) -> ArenaResult:
    """Compute ratings and persist arena outputs from annotation payload."""
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
    if leaderboard_printer is not None:
        leaderboard_printer(arena_result)
    logger.info("Arena analysis complete. Output: %s", config.output_dir)
    return arena_result

