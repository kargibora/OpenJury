"""Save and load arena results as ``arena.json``.

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
from openjury.arena.config import ArenaResult, MatchResult, ModelScore


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


# ═════════════════════════════════════════════════════════════════════
#  Load
# ═════════════════════════════════════════════════════════════════════


def load_arena(path: str | Path) -> ArenaResult:
    """Load an ``arena.json`` file into an :class:`ArenaResult`.

    Args:
        path: Path to the ``arena.json`` file, or a directory containing one.

    Returns:
        A fully populated :class:`ArenaResult`.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "arena.json"
    if not path.exists():
        raise FileNotFoundError(f"Arena file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    meta = data["metadata"]
    ratings = data.get("ratings", {})

    # ── Parse model scores ───────────────────────────────────────
    model_scores: dict[str, list[ModelScore]] = {}
    for model, entries in data.get("model_scores", {}).items():
        model_scores[model] = [
            ModelScore(
                model=model,
                instruction_index=e["instruction_index"],
                scores=e["scores"],
                completion=e.get("completion", ""),
                raw_judge_output=e.get("raw_judge_output", ""),
            )
            for e in entries
        ]

    # ── Parse matches ────────────────────────────────────────────
    matches: list[MatchResult] = []
    for m in data.get("matches", []):
        matches.append(MatchResult(
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
        ))

    # ── Reconstruct per-instruction metadata from columnar ───────
    instruction_metadata: list[dict[str, Any]] = []
    if "instruction_metadata" in data:
        col_data = data["instruction_metadata"]
        if col_data:
            first_key = next(iter(col_data))
            n_rows = len(col_data[first_key])
            for i in range(n_rows):
                instruction_metadata.append(
                    {k: col_data[k][i] for k in col_data}
                )

    result = ArenaResult(
        models=meta["models"],
        dataset=meta["dataset"],
        judge_model=meta["judge_model"],
        judge_mode=meta["judge_mode"],
        matchmaker_strategy=meta.get("matchmaker", "round_robin"),
        rubric_name=meta.get("rubric", "default"),
        rubric_definition=data.get("rubric_definition", {}),
        n_instructions=meta.get("n_instructions", 0),
        instruction_metadata=instruction_metadata,
        model_scores=model_scores,
        matches=matches,
        bt_strengths=ratings.get("bt_strengths", {}),
        elo_ratings=ratings.get("elo", {}),
        win_matrix=ratings.get("win_matrix", {}),
        aggregate_win_rates=ratings.get("aggregate_win_rates", {}),
        dimension_weights=ratings.get("dimension_weights", {}),
        dimension_weight_accuracy=ratings.get("dimension_weight_accuracy", 0.0),
        system_prompt=data.get("system_prompt", ""),
    )

    logger.info(
        "Loaded arena: %d models, %d matches, dataset=%s, judge=%s",
        result.n_models, result.n_matches, result.dataset, result.judge_model,
    )
    return result
