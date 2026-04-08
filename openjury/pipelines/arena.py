"""Arena pipeline orchestration over canonical annotate.

The pipeline produces a reusable annotation artifact; analysis is
performed post-hoc via separate scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openjury._logging import logger
from openjury.annotate_config import (
    AnnotateConfig,
    AnnotateGenerationConfig,
    AnnotatePairingConfig,
)
from openjury.arena.config import (
    ArenaConfig,
    MatchResult,
    ModelScore,
    ModelEntry,
)
from openjury.pipelines.model_annotation import run_annotate
from openjury.pipelines.annotate import save_arena_annotations


# ═════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═════════════════════════════════════════════════════════════════════


def _arena_to_annotate_config(config: ArenaConfig) -> AnnotateConfig:
    """Project an arena task onto the canonical annotate(matchmaker) path."""
    return AnnotateConfig(
        dataset=config.dataset,
        judge=config.judge,
        challenger=None,
        models=[ModelEntry.from_raw(model.to_dict()) for model in config.models],
        output_dir=config.output_dir,
        n_instructions=config.n_instructions,
        language=config.language,
        seed=config.seed,
        balance_by=config.balance_by,
        criteria=config.criteria,
        pairing=AnnotatePairingConfig(
            source="matchmaker",
            strategy=config.matchmaker.strategy,
            seed=config.seed,
            n_matches=config.matchmaker.n_matches,
        ),
        generation=AnnotateGenerationConfig(
            max_tokens=config.generation_max_tokens,
            truncate_input_chars=config.truncate_input_chars,
            ignore_cache=config.ignore_cache,
        ),
        ignore_score_cache=config.ignore_score_cache,
        include_completions=config.include_completions,
        include_raw_judge=config.include_raw_judge,
    )


def _annotate_payload_to_arena_payload(
    config: ArenaConfig,
    annotate_payload: dict[str, Any],
) -> dict[str, Any]:
    """Normalize canonical annotate output into the persisted arena artifact shape."""
    metadata = dict(annotate_payload.get("metadata", {}))
    return {
        "metadata": {
            "models": list(metadata.get("models", config.model_names)),
            "dataset": config.dataset,
            "dataset_cache_key": metadata.get(
                "dataset_cache_key",
                config.dataset_options.cache_key(),
            ),
            "judge_model": config.judge.model,
            "judge_mode": config.judge.mode,
            "pairwise_prompt_style": config.judge.pairwise_prompt_style,
            "matchmaker": config.matchmaker.strategy,
            "criteria": config.criteria,
            "n_instructions": metadata.get("n_instructions", config.n_instructions),
        },
        "criteria_definition": annotate_payload.get("criteria_definition", {}),
        "criterion_names": list(annotate_payload.get("criterion_names", [])),
        "instruction_metadata": list(annotate_payload.get("instruction_metadata", [])),
        "model_scores": dict(annotate_payload.get("model_scores", {})),
        "matches": list(annotate_payload.get("matches", [])),
        "system_prompt": annotate_payload.get("system_prompt", ""),
    }


def _annotate_arena(config: ArenaConfig) -> dict[str, Any]:
    """Run the canonical annotate engine for arena matchmaker evaluation."""
    annotate_config = _arena_to_annotate_config(config)
    annotate_payload = run_annotate(annotate_config, persist=False)
    return _annotate_payload_to_arena_payload(config, annotate_payload)


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
        "criteria_definition": data.get("criteria_definition", data.get("rubric_definition", {})),
        "criterion_names": data.get("criterion_names", data.get("dimension_names", [])),
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
    return analyze_arena_stage(
        config=config,
        ann=ann,
        leaderboard_printer=_print_leaderboard,
    )


def run_arena(config: ArenaConfig) -> None:
    """Execute the arena pipeline: annotate and save the artifact."""
    ann_payload = _annotate_arena(config)
    ann_path = save_arena_annotations(
        config.output_dir,
        ann_payload,
        config_snapshot=config.to_dict(),
    )
    config.save(Path(config.output_dir) / "arena_config.json")
    logger.info("Saved arena annotations: %s", ann_path)


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

    # Dimension weights — which criteria features predict who wins
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
    """Print criteria dimension weights (shared by 2-model and K-model)."""
    if not result.dimension_weights:
        return
    logger.info("")
    logger.info("  Criteria dimension weights (BT feature importance):")
    for dim, w in sorted(result.dimension_weights.items(), key=lambda x: -abs(x[1])):
        bar = "█" * min(int(abs(w) * 20), 40)
        sign = "+" if w >= 0 else "-"
        logger.info("    %-15s %s%s  %.4f", dim, sign, bar, w)
    logger.info(
        "  Dimension weight accuracy: %.1f%%",
        result.dimension_weight_accuracy * 100,
    )
