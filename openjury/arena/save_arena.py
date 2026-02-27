"""Save arena results as ``arena.json``.

The ``arena.json`` file is the **canonical** self-contained artifact for
downstream analysis: leaderboards, BT re-fitting, cross-judge comparison,
human-preference correlation, per-topic slicing, etc.

Schema (v2.1)::

    {
      "version": "2.1",
      "metadata": {
        "models": ["VLLM/A", "VLLM/B", "VLLM/C"],
        "dataset": "alpaca-eval",
        "judge_model": "VLLM/Qwen/Qwen3-32B",
        "judge_mode": "samplewise",
        "matchmaker": "round_robin",
        "rubric": "default",
        "n_instructions": 100,
        "n_matches": 300,
        "date": "2026-02-16T12:00:00"
      },
      "rubric_definition": { ... },
      "model_scores": {
        "VLLM/A": [
          {"instruction_index": 0, "scores": {"fluency": 4, ...}},
          ...
        ]
      },
      "matches": [
        {
          "model_a": "VLLM/A", "model_b": "VLLM/B",
          "instruction_index": 0,
          "scores_a": {...}, "scores_b": {...},
          "preference": 0.8
        }
      ],
      "ratings": {
        "bt_strengths": {"VLLM/A": 1.23, ...},
        "elo": {"VLLM/A": 1550, ...},
        "win_matrix": {"VLLM/A": {"VLLM/B": 0.6, ...}, ...},
        "dimension_weights": {"fluency": 0.42, ...},
        "dimension_weight_accuracy": 0.78
      }
    }
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from openjury._logging import logger
from openjury.arena.config import ArenaResult


# ═════════════════════════════════════════════════════════════════════
#  Save
# ═════════════════════════════════════════════════════════════════════


def save_arena(
    result: ArenaResult,
    output_dir: str | Path,
    *,
    include_completions: bool = False,
    include_raw_judge: bool = False,
) -> Path:
    """Write ``arena.json`` to the output directory.

    Args:
        result: Complete arena result.
        output_dir: Directory to write into (created if needed).
        include_completions: Store full completion text in each match
            record.  Can make the file very large for many matches.
        include_raw_judge: Store raw judge output in match records.

    Returns:
        Path to the written ``arena.json`` file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {
        "version": "2.1",
        "metadata": {
            "models": result.models,
            "dataset": result.dataset,
            "judge_model": result.judge_model,
            "judge_mode": result.judge_mode,
            "matchmaker": result.matchmaker_strategy,
            "rubric": result.rubric_name,
            "n_instructions": result.n_instructions,
            "n_matches": result.n_matches,
            "n_models": result.n_models,
            "date": datetime.now().isoformat(),
            **result.extra_metadata,
        },
        "rubric_definition": result.rubric_definition,
    }

    # ── System prompt (for reproducibility) ──────────────────────
    if result.system_prompt:
        data["system_prompt"] = result.system_prompt

    # ── Model scores (samplewise) ───────────────────────────────
    if result.model_scores:
        ms_data: dict[str, list[dict]] = {}
        for model, scores in result.model_scores.items():
            ms_data[model] = [
                {
                    "instruction_index": s.instruction_index,
                    "scores": s.scores,
                    **({"completion": s.completion} if include_completions else {}),
                    **({"raw_judge_output": s.raw_judge_output} if include_raw_judge else {}),
                }
                for s in scores
            ]
        data["model_scores"] = ms_data

    # ── Matches ──────────────────────────────────────────────────
    matches_data: list[dict] = []
    for m in result.matches:
        record: dict[str, Any] = {
            "model_a": m.model_a,
            "model_b": m.model_b,
            "instruction_index": m.instruction_index,
            "scores_a": m.scores_a,
            "scores_b": m.scores_b,
            "preference": m.preference,
        }
        if m.instruction:
            record["instruction"] = m.instruction
        if m.instruction_id:
            record["instruction_id"] = m.instruction_id
        if m.instruction_metadata:
            record["instruction_metadata"] = m.instruction_metadata
        if include_completions:
            record["completion_a"] = m.completion_a
            record["completion_b"] = m.completion_b
        if include_raw_judge:
            record["raw_judge_output"] = m.raw_judge_output
            if m.raw_judge_output_swapped is not None:
                record["raw_judge_output_swapped"] = m.raw_judge_output_swapped
        matches_data.append(record)
    data["matches"] = matches_data

    # ── Instruction metadata (columnar) ──────────────────────────
    if result.instruction_metadata:
        # Store as a columnar dict: {"lang": ["en", "fr", ...], ...}
        all_keys: set[str] = set()
        for md in result.instruction_metadata:
            all_keys.update(md.keys())
        columnar: dict[str, list] = {k: [] for k in sorted(all_keys)}
        for md in result.instruction_metadata:
            for k in columnar:
                columnar[k].append(md.get(k))
        data["instruction_metadata"] = columnar

    # ── Ratings ──────────────────────────────────────────────────
    data["ratings"] = {
        "bt_strengths": result.bt_strengths,
        "elo": result.elo_ratings,
        "win_matrix": result.win_matrix,
        "aggregate_win_rates": result.aggregate_win_rates,
        "dimension_weights": result.dimension_weights,
        "dimension_weight_accuracy": result.dimension_weight_accuracy,
    }

    # ── Write ────────────────────────────────────────────────────
    path = output_dir / "arena.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(
        "Saved arena.json: %d models, %d matches → %s",
        result.n_models, result.n_matches, path,
    )
    return path

