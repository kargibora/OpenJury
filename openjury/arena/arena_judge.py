"""Arena judge — score K models and derive pairwise match results.

Two modes:

- **samplewise** (default, recommended for K ≥ 3):
  Score each model's completions independently (K × N judge calls), then
  derive pairwise preferences from criteria-score differences for every
  sampled matchup.  Efficient: judge cost is O(K·N) regardless of how
  many pairs the matchmaker generates.

- **pairwise**:
  For each matchup the judge sees both completions side-by-side (with
  optional swap debiasing).  Cost is O(|matches|) — can be expensive for
  large K.

Usage::

    from openjury.arena.arena_judge import ArenaJudge

    judge = ArenaJudge(
        judge_model=make_model("VLLM/Qwen/Qwen3-32B"),
        criteria=get_criteria("default"),
    )

    # Samplewise (default): score each model, derive all pairs
    results = judge.run_samplewise(
        models=["VLLM/A", "VLLM/B", "VLLM/C"],
        completions={"VLLM/A": df_a, "VLLM/B": df_b, "VLLM/C": df_c},
        instructions=instructions,
        matches=matchmaker_output,
    )
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from openjury._logging import logger
from openjury.arena.config import Match, MatchResult, ModelScore
from openjury.common.pair_annotation import (
    PairSample,
    derive_preference_from_scores,
    score_pairs_pairwise,
)
from openjury.criteria import CriteriaScorer
from openjury.criteria.schema import Criteria, CriteriaScore


class ArenaJudge:
    """Orchestrate criteria scoring for K models in an arena.

    Wraps :class:`CriteriaScorer` and adds arena-specific logic:
    model-indexed completion lookup, samplewise→pairwise derivation,
    and match-result construction.

    Args:
        judge_model: An LLM backend with ``.invoke`` / ``.batch``.
        criteria: The criteria set to evaluate against.
        provide_explanation: Ask judge for explanations (slower).
    """

    def __init__(
        self,
        judge_model: Any,
        criteria: Criteria,
        provide_explanation: bool = False,
        pairwise_prompt_style: str = "criteria",
    ):
        self.criteria = criteria
        self.scorer = CriteriaScorer(
            judge_model=judge_model,
            criteria=criteria,
            provide_explanation=provide_explanation,
            pairwise_prompt_style=pairwise_prompt_style,
        )

    # ─────────────────────────────────────────────────────────────
    #  Single-model scoring (public — enables per-model caching)
    # ─────────────────────────────────────────────────────────────

    def score_model(
        self,
        model: str,
        completions_df: pd.DataFrame,
        instructions: list[str],
        reference_answers: list[str] | None = None,
        use_tqdm: bool = False,
    ) -> list[ModelScore]:
        """Score a single model's completions against the criteria.

        This is the **cacheable unit**: one model, one dataset, one criteria set.
        Call this per-model and cache the results so that adding a new
        model to the arena only requires scoring the new model.

        Args:
            model: Model name.
            completions_df: DataFrame with ``completion`` column, aligned
                with *instructions* by row order.
            instructions: List of instruction strings.
            reference_answers: Optional reference answers for calibration.
            use_tqdm: Show progress bar during judge inference.

        Returns:
            List of :class:`ModelScore` (one per instruction).
        """
        n = len(instructions)
        model_completions = completions_df["completion"].tolist()[:n]

        logger.info(
            "Scoring model %s on %d instructions",
            model.rsplit("/", 1)[-1], len(model_completions),
        )

        criteria_scores: list[CriteriaScore] = self.scorer.score(
            instructions=instructions[:len(model_completions)],
            completions=model_completions,
            model_name=model,
            use_tqdm=use_tqdm,
            reference_answers=reference_answers,
        )

        model_score_list: list[ModelScore] = []
        for rs in criteria_scores:
            ms = ModelScore(
                model=model,
                instruction_index=rs.instruction_index,
                scores=rs.scores,
                completion=model_completions[rs.instruction_index]
                if rs.instruction_index < len(model_completions)
                else "",
                raw_judge_output=rs.raw_judge_output,
            )
            model_score_list.append(ms)

        return model_score_list

    # ─────────────────────────────────────────────────────────────
    #  Samplewise mode (default — O(K·N) judge calls)
    # ─────────────────────────────────────────────────────────────

    def run_samplewise(
        self,
        models: list[str],
        completions: dict[str, pd.DataFrame],
        instructions: list[str],
        matches: list[Match],
        reference_answers: list[str] | None = None,
        use_tqdm: bool = False,
        precomputed_scores: dict[str, list[ModelScore]] | None = None,
    ) -> tuple[dict[str, list[ModelScore]], list[MatchResult]]:
        """Score each model independently, then derive pairwise results.

        Args:
            models: List of model names participating.
            completions: ``{model_name: DataFrame}`` with ``completion`` column
                indexed by instruction order.
            instructions: List of instruction strings (aligned with completions).
            matches: List of :class:`Match` from the matchmaker.
            reference_answers: Optional reference answers for calibration.
            use_tqdm: Show progress bar during judge inference.
            precomputed_scores: Optional ``{model: list[ModelScore]}``.
                Models present here are **not re-scored** — their cached
                scores are used directly.  Models absent are scored
                via ``score_model()``.

        Returns:
            ``(model_scores, match_results)`` where:
            - ``model_scores``: ``{model: list[ModelScore]}``
            - ``match_results``: List of :class:`MatchResult` for every match.
        """
        n = len(instructions)
        pre = precomputed_scores or {}

        models_to_score = [m for m in models if m not in pre]
        logger.info(
            "Arena samplewise scoring: %d models × %d instructions "
            "(%d cached, %d to score = %d calls)",
            len(models), n, len(pre), len(models_to_score),
            len(models_to_score) * n,
        )

        # ── Score each model (skip if precomputed) ───────────────
        all_model_scores: dict[str, list[ModelScore]] = {}
        # Index for fast lookup: model → instruction_index → ModelScore
        score_index: dict[str, dict[int, ModelScore]] = defaultdict(dict)

        # Insert precomputed scores into index
        for model, scores in pre.items():
            all_model_scores[model] = scores
            for ms in scores:
                score_index[model][ms.instruction_index] = ms

        # Score remaining models
        for model in models_to_score:
            df = completions[model]
            model_score_list = self.score_model(
                model=model,
                completions_df=df,
                instructions=instructions,
                reference_answers=reference_answers,
                use_tqdm=use_tqdm,
            )
            all_model_scores[model] = model_score_list
            for ms in model_score_list:
                score_index[model][ms.instruction_index] = ms

        # ── Derive pairwise results from score differences ────────
        match_results = self.derive_pairwise(
            matches=matches,
            score_index=score_index,
            instructions=instructions,
            completions=completions,
        )

        logger.info(
            "Arena samplewise: %d matches derived from %d model scores",
            len(match_results), sum(len(v) for v in all_model_scores.values()),
        )
        return all_model_scores, match_results

    def derive_pairwise(
        self,
        matches: list[Match],
        score_index: dict[str, dict[int, ModelScore]],
        instructions: list[str],
        completions: dict[str, pd.DataFrame],
    ) -> list[MatchResult]:
        """Derive pairwise preferences from samplewise criteria scores.

        For each match, computes a weighted average of criteria scores for
        both models and converts the difference into a preference.

        This method is public so that the arena pipeline can call it
        separately after restoring cached per-model scores — without
        re-invoking the judge.

        Args:
            matches: List of :class:`Match` from the matchmaker.
            score_index: ``{model: {instruction_index: ModelScore}}``.
            instructions: Instruction strings.
            completions: ``{model: DataFrame}`` with ``completion`` column.

        Returns:
            List of :class:`MatchResult`.
        """
        results: list[MatchResult] = []
        weights = {c.name: c.weight for c in self.criteria.criteria}

        for match in matches:
            ms_a = score_index.get(match.model_a, {}).get(match.instruction_index)
            ms_b = score_index.get(match.model_b, {}).get(match.instruction_index)

            if ms_a is None or ms_b is None:
                logger.warning(
                    "Missing scores for match %s vs %s @ instruction %d",
                    match.model_a, match.model_b, match.instruction_index,
                )
                continue

            preference = derive_preference_from_scores(
                ms_a.scores,
                ms_b.scores,
                weights,
            )

            # Completion text
            comp_a = _get_completion(completions, match.model_a, match.instruction_index)
            comp_b = _get_completion(completions, match.model_b, match.instruction_index)

            results.append(MatchResult(
                model_a=match.model_a,
                model_b=match.model_b,
                instruction_index=match.instruction_index,
                scores_a=ms_a.scores,
                scores_b=ms_b.scores,
                preference=preference,
                instruction=instructions[match.instruction_index]
                if match.instruction_index < len(instructions)
                else "",
                completion_a=comp_a,
                completion_b=comp_b,
            ))

        return results

    # ─────────────────────────────────────────────────────────────
    #  Pairwise mode (O(|matches|) judge calls)
    # ─────────────────────────────────────────────────────────────

    def run_pairwise(
        self,
        completions: dict[str, pd.DataFrame],
        instructions: list[str],
        matches: list[Match],
        swap_to_debias: bool = True,
        use_tqdm: bool = False,
    ) -> list[MatchResult]:
        """Run pairwise judge on each match directly.

        Normalizes matches into pair samples, then delegates pairwise
        annotation to the shared pair-annotation core.

        Args:
            completions: ``{model_name: DataFrame}`` with ``completion`` column.
            instructions: Instruction strings.
            matches: Matchmaker output.
            swap_to_debias: Run A/B + B/A and average.
            use_tqdm: Show progress bar.

        Returns:
            List of :class:`MatchResult`.
        """
        logger.info("Arena pairwise scoring: %d matches", len(matches))

        pairs: list[PairSample] = []
        for i, match in enumerate(matches):
            idx = match.instruction_index
            pairs.append(
                PairSample(
                    sample_id=f"arena::{i}::{match.model_a}::{match.model_b}::{idx}",
                    instruction_index=idx,
                    instruction=instructions[idx] if idx < len(instructions) else "",
                    model_a=match.model_a,
                    model_b=match.model_b,
                    completion_a=_get_completion(completions, match.model_a, idx),
                    completion_b=_get_completion(completions, match.model_b, idx),
                )
            )

        judgements = score_pairs_pairwise(
            scorer=self.scorer,
            pairs=pairs,
            swap_to_debias=swap_to_debias,
            use_tqdm=use_tqdm,
            cache_config=None,  # arena pairwise cache remains task-specific future work
            ignore_cache=True,
        )

        all_results = [
            MatchResult(
                model_a=j.model_a,
                model_b=j.model_b,
                instruction_index=j.instruction_index,
                scores_a=j.scores_a,
                scores_b=j.scores_b,
                preference=j.preference,
                instruction=pair.instruction,
                completion_a=pair.completion_a,
                completion_b=pair.completion_b,
                raw_judge_output=j.raw_judge_output,
                raw_judge_output_swapped=j.raw_judge_output_swapped,
            )
            for pair, j in zip(pairs, judgements)
        ]

        logger.info(
            "Arena pairwise: %d match results collected", len(all_results),
        )
        return all_results


# ═════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════


def _get_completion(
    completions: dict[str, pd.DataFrame],
    model: str,
    instruction_index: int,
) -> str:
    """Safely retrieve a completion string from the completions dict."""
    df = completions.get(model)
    if df is None:
        return ""
    if instruction_index < len(df):
        return str(df.iloc[instruction_index]["completion"])
    return ""
