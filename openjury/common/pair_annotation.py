"""Shared pair-annotation primitives used by arena and agreement pipelines.

This module provides a small, reusable core:

- a normalized pair sample representation
- shared pairwise/samplewise annotation routines
- cache restore helpers keyed by stable sample IDs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from openjury.arena.config import ModelScore
from openjury.cache.scores import score_cache


@dataclass(frozen=True)
class PairSample:
    """A normalized completion pair to be judged."""

    sample_id: str
    instruction_index: int
    instruction: str
    model_a: str
    model_b: str
    completion_a: str
    completion_b: str


@dataclass(frozen=True)
class PairJudgement:
    """Judgement output for one :class:`PairSample`."""

    sample_id: str
    instruction_index: int
    model_a: str
    model_b: str
    scores_a: dict[str, float]
    scores_b: dict[str, float]
    preference: float
    raw_judge_output: str = ""
    raw_judge_output_swapped: str | None = None


@dataclass(frozen=True)
class PairwiseCacheConfig:
    """Cache-key configuration for pairwise judging."""

    judge: str
    rubric: str
    model_key: str
    dataset_exact: str
    n_instructions: int | None
    dataset_base: str | None = None


@dataclass(frozen=True)
class SamplewiseCacheConfig:
    """Cache-key configuration for two-side samplewise judging."""

    judge: str
    rubric: str
    model_key_a: str
    model_key_b: str
    dataset_exact: str
    n_instructions: int | None
    dataset_base: str | None = None


def derive_preference_from_scores(
    scores_a: dict[str, float],
    scores_b: dict[str, float],
    weights: dict[str, float],
) -> float:
    """Derive preference from weighted rubric averages.

    Returns:
        ``0.0`` if A wins, ``1.0`` if B wins, ``0.5`` for tie/unknown.
    """

    def _weighted_avg(scores: dict[str, float]) -> float:
        total_w, total_s = 0.0, 0.0
        for dim, w in weights.items():
            s = scores.get(dim, float("nan"))
            if s == s:  # not NaN
                total_w += w
                total_s += w * s
        return total_s / total_w if total_w > 0 else float("nan")

    avg_a = _weighted_avg(scores_a)
    avg_b = _weighted_avg(scores_b)
    if np.isnan(avg_a) or np.isnan(avg_b):
        return 0.5
    if avg_a > avg_b:
        return 0.0
    if avg_b > avg_a:
        return 1.0
    return 0.5


def _restore_scores_by_sample_id(
    cached_scores: list[ModelScore],
    sample_ids: list[str],
) -> list[ModelScore] | None:
    """Restore cached model scores in current sample order using stable IDs."""
    if not cached_scores:
        return []

    if all(ms.sample_id for ms in cached_scores):
        by_id: dict[str, ModelScore] = {}
        for ms in cached_scores:
            if ms.sample_id not in by_id:
                by_id[ms.sample_id] = ms
        missing = [sid for sid in sample_ids if sid not in by_id]
        if missing:
            return None
        return [by_id[sid] for sid in sample_ids]

    # Legacy cache rows without sample IDs: only safe when lengths match.
    if len(cached_scores) != len(sample_ids):
        return None
    return sorted(cached_scores, key=lambda ms: ms.instruction_index)


def _restore_pairwise_scores_by_sample_id(
    cached_scores: list[ModelScore],
    sample_ids: list[str],
) -> list[tuple[ModelScore, ModelScore]] | None:
    """Restore interleaved pairwise cache entries (A then B) by sample ID."""
    if not cached_scores:
        return []
    if len(cached_scores) % 2 != 0:
        return None

    if not all(ms.sample_id for ms in cached_scores):
        if len(cached_scores) != 2 * len(sample_ids):
            return None
        ordered = sorted(cached_scores, key=lambda ms: ms.instruction_index)
        return [(ordered[i], ordered[i + 1]) for i in range(0, len(ordered), 2)]

    by_id: dict[str, tuple[ModelScore, ModelScore]] = {}
    for i in range(0, len(cached_scores), 2):
        ms_a = cached_scores[i]
        ms_b = cached_scores[i + 1]
        sid_a = ms_a.sample_id
        sid_b = ms_b.sample_id
        if not sid_a or not sid_b or sid_a != sid_b:
            return None
        if sid_a not in by_id:
            by_id[sid_a] = (ms_a, ms_b)

    missing = [sid for sid in sample_ids if sid not in by_id]
    if missing:
        return None
    return [by_id[sid] for sid in sample_ids]


def score_pairs_pairwise(
    *,
    scorer: Any,
    pairs: list[PairSample],
    swap_to_debias: bool,
    use_tqdm: bool = False,
    cache_config: PairwiseCacheConfig | None = None,
    ignore_cache: bool = False,
) -> list[PairJudgement]:
    """Judge normalized pairs in pairwise mode, with optional score cache."""
    if not pairs:
        return []

    sample_ids = [p.sample_id for p in pairs]

    def _load_cached() -> list[tuple[ModelScore, ModelScore]] | None:
        if ignore_cache or cache_config is None:
            return None

        cached = score_cache.get(
            judge=cache_config.judge,
            rubric=cache_config.rubric,
            model=cache_config.model_key,
            dataset=cache_config.dataset_exact,
            n=cache_config.n_instructions,
        )
        restored = (
            _restore_pairwise_scores_by_sample_id(cached, sample_ids)
            if cached is not None
            else None
        )
        if restored is not None:
            return restored

        if (
            cache_config.dataset_base
            and cache_config.dataset_base != cache_config.dataset_exact
        ):
            cached_all = score_cache.get(
                judge=cache_config.judge,
                rubric=cache_config.rubric,
                model=cache_config.model_key,
                dataset=cache_config.dataset_base,
                n=None,
            )
            restored_all = (
                _restore_pairwise_scores_by_sample_id(cached_all, sample_ids)
                if cached_all is not None
                else None
            )
            if restored_all is not None:
                return restored_all

        return None

    cached_pairs = _load_cached()
    if cached_pairs is not None:
        out: list[PairJudgement] = []
        for pair, (ms_a, ms_b) in zip(pairs, cached_pairs):
            scores_a = dict(ms_a.scores)
            preference = float(scores_a.pop("__preference__", 0.5))
            out.append(
                PairJudgement(
                    sample_id=pair.sample_id,
                    instruction_index=pair.instruction_index,
                    model_a=pair.model_a,
                    model_b=pair.model_b,
                    scores_a=scores_a,
                    scores_b=dict(ms_b.scores),
                    preference=preference,
                    raw_judge_output=ms_a.raw_judge_output,
                    raw_judge_output_swapped=ms_b.raw_judge_output or None,
                )
            )
        return out

    results = scorer.score_pairwise(
        instructions=[p.instruction for p in pairs],
        completions_A=[p.completion_a for p in pairs],
        completions_B=[p.completion_b for p in pairs],
        swap_to_debias=swap_to_debias,
        use_tqdm=use_tqdm,
    )

    out: list[PairJudgement] = []
    for pair, res in zip(pairs, results):
        out.append(
            PairJudgement(
                sample_id=pair.sample_id,
                instruction_index=pair.instruction_index,
                model_a=pair.model_a,
                model_b=pair.model_b,
                scores_a=res.scores_A,
                scores_b=res.scores_B,
                preference=res.preference,
                raw_judge_output=res.raw_judge_output,
                raw_judge_output_swapped=res.raw_judge_output_swapped,
            )
        )

    if cache_config is not None:
        cache_entries: list[ModelScore] = []
        for pair, ann in zip(pairs, out):
            cache_entries.append(
                ModelScore(
                    model=pair.model_a,
                    instruction_index=pair.instruction_index,
                    sample_id=pair.sample_id,
                    scores={**ann.scores_a, "__preference__": ann.preference},
                    completion=pair.completion_a,
                    raw_judge_output=ann.raw_judge_output,
                )
            )
            cache_entries.append(
                ModelScore(
                    model=pair.model_b,
                    instruction_index=pair.instruction_index,
                    sample_id=pair.sample_id,
                    scores=ann.scores_b,
                    completion=pair.completion_b,
                    raw_judge_output=ann.raw_judge_output_swapped or "",
                )
            )
        score_cache.put(
            cache_entries,
            judge=cache_config.judge,
            rubric=cache_config.rubric,
            model=cache_config.model_key,
            dataset=cache_config.dataset_exact,
            n=cache_config.n_instructions,
        )

    return out


def score_pairs_samplewise(
    *,
    scorer: Any,
    pairs: list[PairSample],
    dimension_weights: dict[str, float],
    use_tqdm: bool = False,
    cache_config: SamplewiseCacheConfig | None = None,
    ignore_cache: bool = False,
    side_a_label: str = "side_A",
    side_b_label: str = "side_B",
) -> list[PairJudgement]:
    """Judge normalized pairs samplewise (score A and B independently)."""
    if not pairs:
        return []

    sample_ids = [p.sample_id for p in pairs]

    def _score_side(completions: list[str], side_label: str) -> list[ModelScore]:
        rs_list = scorer.score(
            instructions=[p.instruction for p in pairs],
            completions=completions,
            model_name=side_label,
            use_tqdm=use_tqdm,
        )
        return [
            ModelScore(
                model=side_label,
                instruction_index=pairs[i].instruction_index,
                sample_id=pairs[i].sample_id,
                scores=rs.scores,
                completion=completions[i],
                raw_judge_output=rs.raw_judge_output,
            )
            for i, rs in enumerate(rs_list)
        ]

    def _load_cached_side(cache_model_key: str) -> list[ModelScore] | None:
        if ignore_cache or cache_config is None:
            return None

        cached = score_cache.get(
            judge=cache_config.judge,
            rubric=cache_config.rubric,
            model=cache_model_key,
            dataset=cache_config.dataset_exact,
            n=cache_config.n_instructions,
        )
        restored = (
            _restore_scores_by_sample_id(cached, sample_ids)
            if cached is not None
            else None
        )
        if restored is not None:
            return restored

        if (
            cache_config.dataset_base
            and cache_config.dataset_base != cache_config.dataset_exact
        ):
            cached_all = score_cache.get(
                judge=cache_config.judge,
                rubric=cache_config.rubric,
                model=cache_model_key,
                dataset=cache_config.dataset_base,
                n=None,
            )
            restored_all = (
                _restore_scores_by_sample_id(cached_all, sample_ids)
                if cached_all is not None
                else None
            )
            if restored_all is not None:
                return restored_all

        return None

    cache_key_a = cache_config.model_key_a if cache_config else side_a_label
    cache_key_b = cache_config.model_key_b if cache_config else side_b_label

    ms_a_list = _load_cached_side(cache_key_a)
    if ms_a_list is None:
        ms_a_list = _score_side([p.completion_a for p in pairs], side_a_label)
        if cache_config is not None:
            score_cache.put(
                ms_a_list,
                judge=cache_config.judge,
                rubric=cache_config.rubric,
                model=cache_key_a,
                dataset=cache_config.dataset_exact,
                n=cache_config.n_instructions,
            )

    ms_b_list = _load_cached_side(cache_key_b)
    if ms_b_list is None:
        ms_b_list = _score_side([p.completion_b for p in pairs], side_b_label)
        if cache_config is not None:
            score_cache.put(
                ms_b_list,
                judge=cache_config.judge,
                rubric=cache_config.rubric,
                model=cache_key_b,
                dataset=cache_config.dataset_exact,
                n=cache_config.n_instructions,
            )

    out: list[PairJudgement] = []
    for pair, ms_a, ms_b in zip(pairs, ms_a_list, ms_b_list):
        preference = derive_preference_from_scores(
            ms_a.scores,
            ms_b.scores,
            dimension_weights,
        )
        out.append(
            PairJudgement(
                sample_id=pair.sample_id,
                instruction_index=pair.instruction_index,
                model_a=pair.model_a,
                model_b=pair.model_b,
                scores_a=ms_a.scores,
                scores_b=ms_b.scores,
                preference=preference,
                raw_judge_output=ms_a.raw_judge_output,
                raw_judge_output_swapped=None,
            )
        )

    return out
