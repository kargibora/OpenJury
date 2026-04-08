"""Score completions on a criteria set using an LLM judge.

Supports two modes:

- **Pairwise** (default): Judge sees both A and B side-by-side, scores each
  on criteria, then gives an overall preference. K calls (2K with
  swap-debiasing). Best for ranking.
- **Sample-wise**: Judge scores each completion independently (2K calls).
  Preferences are derived from the weighted average of criterion
  scores — no extra judge call. Optionally uses a reference answer as an
  anchor. Best for absolute quality assessment.

Example (pairwise)::

    from openjury.criteria import CriteriaScorer, get_criteria
    from openjury.models.factory import make_model

    judge = make_model("VLLM/Qwen/Qwen3-32B")
    scorer = CriteriaScorer(judge_model=judge, criteria=get_criteria("default"))

    results = scorer.score_pairwise(
        instructions=["Write a haiku about AI"],
        completions_A=["Silicon dreams hum / ..."],
        completions_B=["AI is good / ..."],
    )

Example (sample-wise with reference)::

    scores = scorer.score(
        instructions=["What is 2+2?"],
        completions=["4"],
        model_name="gpt-4o",
        reference_answers=["The answer is 4."],
    )
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd

from openjury._logging import logger
from openjury.prompts import load_prompt
from openjury.criteria.schema import (
    Criteria,
    CriteriaScore,
    MultiTrialPairwiseResult,
    MultiTrialSamplewiseResult,
    PairwiseCriteriaResult,
)
from openjury.inference import do_inference, do_inference_multi


# ═════════════════════════════════════════════════════════════════════
#  Prompt Templates — loaded from openjury/prompts/*.txt
# ═════════════════════════════════════════════════════════════════════

SAMPLEWISE_SYSTEM_PROMPT = load_prompt("criteria_samplewise_system")
SAMPLEWISE_USER_TEMPLATE = load_prompt("criteria_samplewise_user")
PAIRWISE_SYSTEM_PROMPT = load_prompt("criteria_pairwise_system")
PAIRWISE_USER_TEMPLATE = load_prompt("criteria_pairwise_user")
LEGACY_PAIRWISE_SYSTEM_PROMPT = load_prompt("legacy_pairwise_system_prompt")
LEGACY_PAIRWISE_USER_TEMPLATE = load_prompt("prompt")
LEGACY_PAIRWISE_USER_TEMPLATE_WITH_EXPLANATION = load_prompt(
    "legacy_pairwise_prompt_with_explanation"
)


# ═════════════════════════════════════════════════════════════════════
#  Scorer
# ═════════════════════════════════════════════════════════════════════


class CriteriaScorer:
    """Scores completions along criteria using an LLM judge.

    Supports pairwise (A vs B) and sample-wise (independent) scoring.

    Args:
        judge_model: A model backend (anything with ``.invoke`` / ``.batch``).
        criteria: The criteria set to evaluate against.
        provide_explanation: Whether to ask the judge for explanations.
        pairwise_prompt_style: ``"criteria"`` (structured JSON) or ``"legacy"``
            (simple A/B scoring for single-criterion sets).
            ``"rubric"`` is accepted as a backward-compatible alias for
            ``"criteria"``.
    """

    def __init__(
        self,
        judge_model: Any,
        criteria: Criteria,
        provide_explanation: bool = False,
        pairwise_prompt_style: str = "criteria",
    ):
        self.judge_model = judge_model
        self.criteria = criteria
        self.provide_explanation = provide_explanation
        # Normalize legacy alias
        if pairwise_prompt_style == "rubric":
            pairwise_prompt_style = "criteria"
        self.pairwise_prompt_style = pairwise_prompt_style

        if self.pairwise_prompt_style not in {"criteria", "legacy"}:
            raise ValueError(
                "pairwise_prompt_style must be one of {'criteria', 'legacy'}"
            )
        if self.pairwise_prompt_style == "legacy" and self.criteria.num_criteria != 1:
            raise ValueError(
                "legacy pairwise prompt style requires a single-criterion criteria set "
                "(for example criteria='overall')."
            )

        explanation_block = (
            "Before providing scores, briefly explain your reasoning for each criterion."
            if provide_explanation
            else "Provide ONLY the JSON output, no explanation needed."
        )

        # Generate example JSON that uses the *actual* criterion names
        # so the model sees consistent names between criteria block and example.
        _sample_vals = [7, 8, 6, 9, 7, 8, 5, 10, 6, 9]
        _example = {
            c.name: min(
                c.scale_max,
                max(c.scale_min, _sample_vals[i % len(_sample_vals)]),
            )
            for i, c in enumerate(criteria.criteria)
        }
        _example_json = json.dumps(_example)
        _example_b = {
            c.name: min(
                c.scale_max,
                max(c.scale_min, _sample_vals[i % len(_sample_vals)] - 1),
            )
            for i, c in enumerate(criteria.criteria)
        }
        _example_pairwise = json.dumps(
            {"scores_A": _example, "scores_B": _example_b, "preference": "A"},
            indent=2,
        )

        # Build system prompts for both modes
        self._samplewise_system = SAMPLEWISE_SYSTEM_PROMPT.format(
            criteria_block=criteria.prompt_block(),
            reference_block="",  # Filled dynamically when reference is provided
            explanation_block=explanation_block,
            example_json=_example_json,
        )
        self._samplewise_system_with_ref = SAMPLEWISE_SYSTEM_PROMPT.format(
            criteria_block=criteria.prompt_block(),
            reference_block=(
                "You are also given a REFERENCE ANSWER. Use it as an anchor to "
                "calibrate your scores. A perfect completion should match or exceed "
                "the reference answer in quality."
            ),
            explanation_block=explanation_block,
            example_json=_example_json,
        )
        if self.pairwise_prompt_style == "legacy":
            self._pairwise_system = LEGACY_PAIRWISE_SYSTEM_PROMPT
        else:
            self._pairwise_system = PAIRWISE_SYSTEM_PROMPT.format(
                criteria_block=criteria.prompt_block(),
                explanation_block=explanation_block,
                example_json_pairwise=_example_pairwise,
            )

    @property
    def system_prompt(self) -> dict[str, str]:
        """Return the formatted system prompts for debugging/reproducibility.

        Returns:
            Dict with keys ``samplewise``, ``samplewise_with_ref``,
            ``pairwise`` — each the fully rendered system prompt string.
        """
        return {
            "samplewise": self._samplewise_system,
            "samplewise_with_ref": self._samplewise_system_with_ref,
            "pairwise": self._pairwise_system,
        }

    # ─────────────────────────────────────────────────────────────
    #  Key normalisation
    # ─────────────────────────────────────────────────────────────

    def _normalize_score_keys(
        self, scores: dict[str, float],
    ) -> dict[str, float]:
        """Map parsed score keys back to canonical criterion names.

        The judge prompt renders criteria via ``prompt_block()`` which
        applies ``.title()`` casing (e.g. ``Instruction_Adherence``).  The
        model's JSON output therefore typically uses title-cased keys,
        but all downstream code expects the canonical lowercase
        names (``instruction_adherence``).  This method performs a
        case-insensitive mapping to reconcile the two.
        """
        lookup = {c.lower(): c for c in self.criteria.criterion_names}
        return {
            lookup.get(k.lower(), k): v
            for k, v in scores.items()
        }

    def _coerce_score_value(self, value: Any) -> float:
        """Best-effort coercion for judge outputs that nest numeric scores.

        Some models emit values like ``[7]`` or ``{"score": 7}`` even when the
        prompt asks for a plain scalar. Treat these as recoverable parser noise
        rather than crashing the evaluation run.
        """
        if isinstance(value, bool) or value is None:
            raise TypeError(f"Unsupported score value: {value!r}")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            match = re.search(r"-?\d+(?:\.\d+)?", value.strip())
            if match:
                return float(match.group(0))
            raise ValueError(f"Could not parse numeric score from string: {value!r}")
        if isinstance(value, dict):
            for key in ("score", "value", "rating"):
                if key in value:
                    return self._coerce_score_value(value[key])
            for nested in value.values():
                try:
                    return self._coerce_score_value(nested)
                except (TypeError, ValueError):
                    continue
            raise TypeError(f"Unsupported nested score object: {value!r}")
        if isinstance(value, (list, tuple)):
            for item in value:
                try:
                    return self._coerce_score_value(item)
                except (TypeError, ValueError):
                    continue
            raise TypeError(f"Unsupported score list: {value!r}")
        raise TypeError(f"Unsupported score value type: {type(value)!r}")

    def _coerce_scores_dict(self, scores: dict[str, Any]) -> dict[str, float]:
        """Coerce a parsed JSON dict into criterion -> float scores."""
        return self._normalize_score_keys(
            {k: self._coerce_score_value(v) for k, v in scores.items()}
        )

    def _nan_scores_dict(self) -> dict[str, float]:
        """Return a NaN-filled score dict aligned with this criteria set."""
        return {cname: float("nan") for cname in self.criteria.criterion_names}

    # ─────────────────────────────────────────────────────────────
    #  Pairwise scoring (recommended for ranking)
    # ─────────────────────────────────────────────────────────────

    def score_pairwise(
        self,
        instructions: list[str],
        completions_A: list[str],
        completions_B: list[str],
        swap_to_debias: bool = True,
        use_tqdm: bool = False,
        force_async: bool = False,
        n_trials: int = 1,
    ) -> list[PairwiseCriteriaResult] | list[MultiTrialPairwiseResult]:
        """Compare A vs B on criteria.

        The judge sees both completions, scores each on every criterion,
        and states an overall preference. If ``swap_to_debias=True``, runs
        each pair twice (A/B then B/A) and averages to cancel position bias.

        When ``n_trials > 1``, the judge is called K times per sample (with
        ``SamplingParams(n=K)`` for local models) and results are aggregated
        into :class:`MultiTrialPairwiseResult` objects with per-sample
        confidence statistics.

        Args:
            instructions: List of instructions/prompts.
            completions_A: Completions from model A.
            completions_B: Completions from model B.
            swap_to_debias: Run A/B + B/A and average (recommended).
            use_tqdm: Show progress bar.
            force_async: Use async inference for API models.
            n_trials: Number of independent judge calls per sample.
                ``1`` = original behaviour. ``K > 1`` = Average-of-K with
                confidence estimation. Requires ``temperature > 0``.

        Returns:
            ``n_trials == 1``: ``list[PairwiseCriteriaResult]`` (one per instruction).
            ``n_trials > 1``: ``list[MultiTrialPairwiseResult]`` with aggregated stats.
        """
        assert len(instructions) == len(completions_A) == len(completions_B)
        num_samples = len(instructions)

        if n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {n_trials}")

        # ── Single-trial fast path (original behaviour) ──────────
        if n_trials == 1:
            return self._score_pairwise_single(
                instructions, completions_A, completions_B,
                swap_to_debias=swap_to_debias,
                use_tqdm=use_tqdm,
                force_async=force_async,
            )

        # ── Multi-trial path: K trials per sample ────────────────
        logger.info(
            "Multi-trial pairwise scoring: %d pairs × %d trials on %d criteria",
            num_samples, n_trials, self.criteria.num_criteria,
        )

        prompts_ab = self._build_pairwise_prompts(
            instructions, completions_A, completions_B,
        )

        # Returns list[list[str]] — [sample_i][trial_k]
        logger.info("  A/B order: %d pairs × %d trials", num_samples, n_trials)
        multi_ab = do_inference_multi(
            chat_model=self.judge_model,
            inputs=prompts_ab,
            n=n_trials,
            use_tqdm=use_tqdm,
            force_async=force_async,
        )

        multi_ba: list[list[str]] | None = None
        if swap_to_debias:
            prompts_ba = self._build_pairwise_prompts(
                instructions=instructions,
                completions_A=completions_B,
                completions_B=completions_A,
            )
            logger.info("  B/A order: %d pairs × %d trials (debiasing)", num_samples, n_trials)
            multi_ba = do_inference_multi(
                chat_model=self.judge_model,
                inputs=prompts_ba,
                n=n_trials,
                use_tqdm=use_tqdm,
                force_async=force_async,
            )

        # Parse each trial independently, then aggregate
        results: list[MultiTrialPairwiseResult] = []
        for i in range(num_samples):
            trials_for_sample: list[PairwiseCriteriaResult] = []
            for k in range(n_trials):
                text_ab = multi_ab[i][k]
                try:
                    parsed_ab = self._parse_pairwise(text_ab)
                except Exception as exc:
                    logger.warning(
                        "Parse failure sample %d trial %d (A/B): %s", i, k, exc,
                    )
                    parsed_ab = {
                        "scores_A": self._nan_scores_dict(),
                        "scores_B": self._nan_scores_dict(),
                        "preference": 0.5,
                    }

                if swap_to_debias and multi_ba is not None:
                    text_ba = multi_ba[i][k]
                    try:
                        parsed_ba = self._parse_pairwise(text_ba)
                    except Exception as exc:
                        logger.warning(
                            "Parse failure sample %d trial %d (B/A): %s", i, k, exc,
                        )
                        parsed_ba = {
                            "scores_A": self._nan_scores_dict(),
                            "scores_B": self._nan_scores_dict(),
                            "preference": 0.5,
                        }
                    merged = self._merge_swapped(parsed_ab, parsed_ba)
                    merged_raw_swapped = text_ba
                else:
                    merged = parsed_ab
                    merged_raw_swapped = None

                trials_for_sample.append(PairwiseCriteriaResult(
                    instruction_index=i,
                    scores_A=merged["scores_A"],
                    scores_B=merged["scores_B"],
                    preference=merged["preference"],
                    raw_judge_output=text_ab,
                    raw_judge_output_swapped=merged_raw_swapped,
                    scores_A_original=merged.get("scores_A_original"),
                    scores_B_original=merged.get("scores_B_original"),
                    preference_original=merged.get("preference_original"),
                    scores_A_swapped=merged.get("scores_A_swapped"),
                    scores_B_swapped=merged.get("scores_B_swapped"),
                    preference_swapped=merged.get("preference_swapped"),
                ))

            results.append(
                MultiTrialPairwiseResult.from_trials(i, trials_for_sample)
            )

        valid = sum(1 for r in results if r.mean_preference != 0.5)
        logger.info(
            "Multi-trial pairwise: %d/%d valid preferences "
            "(K=%d, mean self-agreement=%.2f)%s",
            valid, num_samples, n_trials,
            np.mean([r.self_agreement for r in results]),
            " (position-swap debiased)" if swap_to_debias else "",
        )
        return results

    def _score_pairwise_single(
        self,
        instructions: list[str],
        completions_A: list[str],
        completions_B: list[str],
        swap_to_debias: bool = True,
        use_tqdm: bool = False,
        force_async: bool = False,
    ) -> list[PairwiseCriteriaResult]:
        """Single-trial pairwise scoring (original implementation)."""
        n = len(instructions)

        prompts_ab = self._build_pairwise_prompts(
            instructions, completions_A, completions_B,
        )

        logger.info(
            "Pairwise criteria scoring: %d pairs on %d criteria (A/B order)",
            n, self.criteria.num_criteria,
        )
        raw_ab = do_inference(
            chat_model=self.judge_model,
            inputs=prompts_ab,
            use_tqdm=use_tqdm,
            force_async=force_async,
        )

        raw_ba = None
        if swap_to_debias:
            prompts_ba = self._build_pairwise_prompts(
                instructions=instructions,
                completions_A=completions_B,
                completions_B=completions_A,
            )
            logger.info(
                "Pairwise criteria scoring: %d pairs on %d criteria (B/A debiasing)",
                n, self.criteria.num_criteria,
            )
            raw_ba = do_inference(
                chat_model=self.judge_model,
                inputs=prompts_ba,
                use_tqdm=use_tqdm,
                force_async=force_async,
            )

        results: list[PairwiseCriteriaResult] = []
        for i in range(n):
            text_ab = raw_ab[i] if isinstance(raw_ab[i], str) else raw_ab[i].content
            try:
                parsed_ab = self._parse_pairwise(text_ab)
            except Exception as exc:
                logger.warning(
                    "Unexpected pairwise parse failure for sample %d (A/B order): %s",
                    i,
                    exc,
                )
                parsed_ab = {
                    "scores_A": self._nan_scores_dict(),
                    "scores_B": self._nan_scores_dict(),
                    "preference": 0.5,
                }

            if swap_to_debias and raw_ba is not None:
                text_ba = raw_ba[i] if isinstance(raw_ba[i], str) else raw_ba[i].content
                try:
                    parsed_ba = self._parse_pairwise(text_ba)
                except Exception as exc:
                    logger.warning(
                        "Unexpected pairwise parse failure for sample %d (B/A order): %s",
                        i,
                        exc,
                    )
                    parsed_ba = {
                        "scores_A": self._nan_scores_dict(),
                        "scores_B": self._nan_scores_dict(),
                        "preference": 0.5,
                    }
                merged = self._merge_swapped(parsed_ab, parsed_ba)
                merged_raw_swapped = text_ba
            else:
                merged = parsed_ab
                merged_raw_swapped = None

            results.append(PairwiseCriteriaResult(
                instruction_index=i,
                scores_A=merged["scores_A"],
                scores_B=merged["scores_B"],
                preference=merged["preference"],
                raw_judge_output=text_ab,
                raw_judge_output_swapped=merged_raw_swapped,
                scores_A_original=merged.get("scores_A_original"),
                scores_B_original=merged.get("scores_B_original"),
                preference_original=merged.get("preference_original"),
                scores_A_swapped=merged.get("scores_A_swapped"),
                scores_B_swapped=merged.get("scores_B_swapped"),
                preference_swapped=merged.get("preference_swapped"),
            ))

        valid = sum(1 for r in results if r.preference != 0.5)
        logger.info(
            "Pairwise criteria: %d/%d valid preferences%s",
            valid, n,
            " (position-swap debiased)" if swap_to_debias else "",
        )

        return results

    def _build_pairwise_prompts(
        self,
        instructions: list[str],
        completions_A: list[str],
        completions_B: list[str],
    ) -> list[list[tuple[str, str]]]:
        """Build chat prompts for pairwise comparison."""
        prompts = []
        for instr, comp_a, comp_b in zip(instructions, completions_A, completions_B):
            if self.pairwise_prompt_style == "legacy":
                template = (
                    LEGACY_PAIRWISE_USER_TEMPLATE_WITH_EXPLANATION
                    if self.provide_explanation
                    else LEGACY_PAIRWISE_USER_TEMPLATE
                )
                user_msg = template.format(
                    user_prompt=instr,
                    completion_A=comp_a,
                    completion_B=comp_b,
                )
            else:
                user_msg = PAIRWISE_USER_TEMPLATE.format(
                    instruction=instr,
                    completion_A=comp_a,
                    completion_B=comp_b,
                )
            prompts.append([
                ("system", self._pairwise_system),
                ("user", user_msg),
            ])
        return prompts

    def _parse_pairwise(self, raw_output: str) -> dict:
        """Parse pairwise judge output into scores_A, scores_B, preference.

        Returns:
            Dict with keys: scores_A, scores_B (dicts), preference (float).
        """
        if self.pairwise_prompt_style == "legacy":
            return self._parse_pairwise_legacy(raw_output)

        nan_scores = {c: float("nan") for c in self.criteria.criterion_names}
        default = {"scores_A": nan_scores.copy(), "scores_B": nan_scores.copy(), "preference": 0.5}

        # Try ```json ... ``` blocks
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_output, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return self._extract_pairwise_from_dict(data)
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback: find any large JSON object
        json_match = re.search(r"\{[^{}]*\{[^{}]*\}[^{}]*\}", raw_output, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return self._extract_pairwise_from_dict(data)
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Last resort: try regex for individual score patterns
        scores_a, scores_b = {}, {}
        for cname in self.criteria.criterion_names:
            match_a = re.search(
                rf'(?:scores?_?A|completion_?A).*?{re.escape(cname)}.*?(\d+(?:\.\d+)?)',
                raw_output, re.IGNORECASE | re.DOTALL,
            )
            match_b = re.search(
                rf'(?:scores?_?B|completion_?B).*?{re.escape(cname)}.*?(\d+(?:\.\d+)?)',
                raw_output, re.IGNORECASE | re.DOTALL,
            )
            if match_a:
                scores_a[cname] = float(match_a.group(1))
            if match_b:
                scores_b[cname] = float(match_b.group(1))

        pref_match = re.search(r'preference["\s:]*["\']?(A|B|tie)', raw_output, re.IGNORECASE)
        pref = 0.5
        if pref_match:
            p = pref_match.group(1).upper()
            pref = 0.0 if p == "A" else 1.0 if p == "B" else 0.5

        if scores_a or scores_b:
            for cname in self.criteria.criterion_names:
                scores_a.setdefault(cname, float("nan"))
                scores_b.setdefault(cname, float("nan"))
            return {"scores_A": scores_a, "scores_B": scores_b, "preference": pref}

        logger.warning("Could not parse pairwise criteria output: %s", raw_output[:200])
        return default

    def _parse_pairwise_legacy(self, raw_output: str) -> dict:
        """Parse the legacy overall pairwise format (score_A / score_B)."""
        cname = self.criteria.criterion_names[0]
        default = {
            "scores_A": {cname: float("nan")},
            "scores_B": {cname: float("nan")},
            "preference": 0.5,
        }

        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_output, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return self._extract_pairwise_legacy_from_dict(data)
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        json_match = re.search(r"\{.*?\}", raw_output, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return self._extract_pairwise_legacy_from_dict(data)
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        match_a = re.search(
            r"score[\s_]*A[\s:=]+(-?\d+(?:\.\d+)?)",
            raw_output,
            re.IGNORECASE,
        )
        match_b = re.search(
            r"score[\s_]*B[\s:=]+(-?\d+(?:\.\d+)?)",
            raw_output,
            re.IGNORECASE,
        )
        if match_a and match_b:
            score_a = float(match_a.group(1))
            score_b = float(match_b.group(1))
            if score_a > score_b:
                pref = 0.0
            elif score_b > score_a:
                pref = 1.0
            else:
                pref = 0.5
            return {
                "scores_A": {cname: score_a},
                "scores_B": {cname: score_b},
                "preference": pref,
            }

        logger.warning("Could not parse legacy pairwise output: %s", raw_output[:200])
        return default

    def _extract_pairwise_legacy_from_dict(self, data: dict[str, Any]) -> dict:
        """Extract legacy pairwise scores from a parsed JSON dict."""
        cname = self.criteria.criterion_names[0]
        score_a_raw = data.get("score_A", data.get("score_a"))
        score_b_raw = data.get("score_B", data.get("score_b"))
        if score_a_raw is None or score_b_raw is None:
            raise ValueError("Legacy pairwise JSON must include score_A and score_B.")

        score_a = float(score_a_raw)
        score_b = float(score_b_raw)
        if score_a > score_b:
            pref = 0.0
        elif score_b > score_a:
            pref = 1.0
        else:
            pref = 0.5
        return {
            "scores_A": {cname: score_a},
            "scores_B": {cname: score_b},
            "preference": pref,
        }

    def _extract_pairwise_from_dict(self, data: dict) -> dict:
        """Extract pairwise scores from a parsed JSON dict."""
        scores_a = data.get("scores_A", data.get("scores_a", {}))
        scores_b = data.get("scores_B", data.get("scores_b", {}))
        pref_raw = data.get("preference", "tie")

        if isinstance(pref_raw, str):
            pref_raw = pref_raw.strip().upper()
            pref = 0.0 if pref_raw == "A" else 1.0 if pref_raw == "B" else 0.5
        elif isinstance(pref_raw, (int, float)):
            pref = float(pref_raw)
        else:
            pref = 0.5

        scores_a = self._coerce_scores_dict(scores_a)
        scores_b = self._coerce_scores_dict(scores_b)

        return {"scores_A": scores_a, "scores_B": scores_b, "preference": pref}

    def _merge_swapped(self, ab: dict, ba: dict) -> dict:
        """Merge A/B and B/A (swapped) results to cancel position bias.

        In the B/A run, positions are swapped: what the judge calls "A" is
        actually B, and vice versa. So we flip them back before averaging.

        Returns the merged scores plus the per-position originals.
        """
        merged_A, merged_B = {}, {}
        # Per-position scores (A/B call = original, B/A call = flipped back)
        original_A, original_B = {}, {}
        swapped_A, swapped_B = {}, {}
        for cname in self.criteria.criterion_names:
            # ab: scores_A = actual A, scores_B = actual B
            # ba: scores_A = actual B (swapped), scores_B = actual A (swapped)
            a_from_ab = ab["scores_A"].get(cname, float("nan"))
            a_from_ba = ba["scores_B"].get(cname, float("nan"))  # flip back
            b_from_ab = ab["scores_B"].get(cname, float("nan"))
            b_from_ba = ba["scores_A"].get(cname, float("nan"))  # flip back

            merged_A[cname] = float(np.nanmean([a_from_ab, a_from_ba]))
            merged_B[cname] = float(np.nanmean([b_from_ab, b_from_ba]))
            original_A[cname] = a_from_ab
            original_B[cname] = b_from_ab
            swapped_A[cname] = a_from_ba
            swapped_B[cname] = b_from_ba

        # Merge preference: ab.preference is P(B wins in A/B order)
        # ba.preference is P(A-actual wins in B/A order) = P(B wins) in swapped = 1 - P(A wins)
        pref_ab = ab["preference"]
        pref_ba = 1.0 - ba["preference"]  # Flip back
        merged_pref = (pref_ab + pref_ba) / 2.0

        return {
            "scores_A": merged_A, "scores_B": merged_B, "preference": merged_pref,
            "scores_A_original": original_A, "scores_B_original": original_B,
            "preference_original": pref_ab,
            "scores_A_swapped": swapped_A, "scores_B_swapped": swapped_B,
            "preference_swapped": pref_ba,
        }

    # ─────────────────────────────────────────────────────────────
    #  Sample-wise scoring (independent, optional reference anchor)
    # ─────────────────────────────────────────────────────────────

    def _build_prompts(
        self,
        instructions: list[str],
        completions: list[str],
        reference_answers: list[str] | None = None,
    ) -> list[list[tuple[str, str]]]:
        """Build chat prompts for sample-wise scoring."""
        has_ref = reference_answers is not None
        system = self._samplewise_system_with_ref if has_ref else self._samplewise_system

        prompts = []
        for i, (instruction, completion) in enumerate(zip(instructions, completions)):
            ref_section = ""
            if has_ref and reference_answers[i]:
                ref_section = f"## Reference Answer\n{reference_answers[i]}"

            user_msg = SAMPLEWISE_USER_TEMPLATE.format(
                instruction=instruction,
                reference_section=ref_section,
                completion=completion,
            )
            prompts.append([
                ("system", system),
                ("user", user_msg),
            ])
        return prompts

    def _parse_scores(self, raw_output: str) -> dict[str, float]:
        """Extract criterion scores from judge output.

        Tries to find JSON in ```json``` blocks first, then falls back to
        finding any JSON object in the text, and finally to key: value patterns.

        Args:
            raw_output: Raw text output from the judge.

        Returns:
            Dict mapping criterion name → score. NaN for unparseable scores.
        """
        # Try ```json ... ``` blocks
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_output, re.DOTALL)
        if json_match:
            try:
                scores = json.loads(json_match.group(1))
                return self._coerce_scores_dict(scores)
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback: find any JSON object
        json_match = re.search(r"\{[^{}]*\}", raw_output)
        if json_match:
            try:
                scores = json.loads(json_match.group(0))
                return self._coerce_scores_dict(scores)
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Last resort: try to parse key: value patterns
        scores = {}
        for cname in self.criteria.criterion_names:
            pattern = rf'["\']?{re.escape(cname)}["\']?\s*[:=]\s*(\d+(?:\.\d+)?)'
            match = re.search(pattern, raw_output, re.IGNORECASE)
            if match:
                scores[cname] = float(match.group(1))

        if scores:
            return scores

        logger.warning(
            "Could not parse criteria scores from judge output: %s",
            raw_output[:200],
        )
        return {cname: float("nan") for cname in self.criteria.criterion_names}

    def score(
        self,
        instructions: list[str],
        completions: list[str],
        model_name: str,
        use_tqdm: bool = False,
        force_async: bool = False,
        reference_answers: list[str] | None = None,
        n_trials: int = 1,
    ) -> list[CriteriaScore] | list[MultiTrialSamplewiseResult]:
        """Score completions independently (sample-wise) on the criteria.

        Each completion is evaluated in isolation. If ``reference_answers``
        is provided, the judge uses them as quality anchors for calibration.

        When ``n_trials > 1``, the judge is called K times per sample and
        results are aggregated into :class:`MultiTrialSamplewiseResult`
        objects with per-sample score variance statistics.

        Args:
            instructions: List of instructions/prompts.
            completions: List of model completions.
            model_name: Name of the model that generated the completions.
            use_tqdm: Whether to show progress bar.
            force_async: Use async inference for API models.
            reference_answers: Optional list of reference/ground-truth answers
                to anchor scoring.
            n_trials: Number of independent judge calls per sample.
                ``1`` = original behaviour. ``K > 1`` = Average-of-K.

        Returns:
            ``n_trials == 1``: ``list[CriteriaScore]`` (one per completion).
            ``n_trials > 1``: ``list[MultiTrialSamplewiseResult]`` with stats.
        """
        assert len(instructions) == len(completions), (
            f"instructions ({len(instructions)}) and completions ({len(completions)}) "
            "must have the same length"
        )
        if reference_answers is not None:
            assert len(reference_answers) == len(instructions), (
                f"reference_answers ({len(reference_answers)}) must match "
                f"instructions ({len(instructions)})"
            )
        if n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {n_trials}")

        prompts = self._build_prompts(instructions, completions, reference_answers)

        mode_label = "sample-wise (with reference)" if reference_answers else "sample-wise"

        # ── Single-trial fast path ────────────────────────────────
        if n_trials == 1:
            logger.info(
                "Scoring %d completions from [model]%s[/model] on %d criteria (%s)",
                len(prompts), model_name, self.criteria.num_criteria, mode_label,
            )

            raw_outputs = do_inference(
                chat_model=self.judge_model,
                inputs=prompts,
                use_tqdm=use_tqdm,
                force_async=force_async,
            )

            results = []
            for i, raw in enumerate(raw_outputs):
                raw_text = raw if isinstance(raw, str) else raw.content
                try:
                    scores = self._parse_scores(raw_text)
                except Exception as exc:
                    logger.warning(
                        "Unexpected samplewise parse failure for sample %d: %s",
                        i,
                        exc,
                    )
                    scores = self._nan_scores_dict()
                results.append(
                    CriteriaScore(
                        instruction_index=i,
                        model=model_name,
                        scores=scores,
                        raw_judge_output=raw_text,
                    )
                )

            valid_scores = [
                r
                for r in results
                if not any(v != v for v in r.scores.values())
            ]
            logger.info(
                "Successfully parsed %d/%d criteria scores",
                len(valid_scores),
                len(results),
            )
            return results

        # ── Multi-trial path ──────────────────────────────────────
        logger.info(
            "Multi-trial scoring: %d completions from [model]%s[/model] "
            "× %d trials on %d criteria (%s)",
            len(prompts), model_name, n_trials,
            self.criteria.num_criteria, mode_label,
        )

        multi_outputs = do_inference_multi(
            chat_model=self.judge_model,
            inputs=prompts,
            n=n_trials,
            use_tqdm=use_tqdm,
            force_async=force_async,
        )

        multi_results: list[MultiTrialSamplewiseResult] = []
        for i, trial_outputs in enumerate(multi_outputs):
            trials: list[CriteriaScore] = []
            for k, raw_text in enumerate(trial_outputs):
                try:
                    scores = self._parse_scores(raw_text)
                except Exception as exc:
                    logger.warning(
                        "Parse failure sample %d trial %d: %s", i, k, exc,
                    )
                    scores = self._nan_scores_dict()
                trials.append(CriteriaScore(
                    instruction_index=i,
                    model=model_name,
                    scores=scores,
                    raw_judge_output=raw_text,
                ))
            multi_results.append(
                MultiTrialSamplewiseResult.from_trials(i, model_name, trials)
            )

        valid = sum(
            1 for r in multi_results
            if not any(v != v for v in r.mean_scores.values())
        )
        logger.info(
            "Multi-trial samplewise: parsed %d/%d (K=%d)",
            valid, len(multi_results), n_trials,
        )
        return multi_results

    # ─────────────────────────────────────────────────────────────
    #  Utilities
    # ─────────────────────────────────────────────────────────────

    def score_to_dataframe(
        self,
        criteria_scores: list[CriteriaScore],
    ) -> pd.DataFrame:
        """Convert criteria scores to a DataFrame.

        Columns: instruction_index, model, crit1, crit2, ..., critK, raw_judge_output.
        """
        rows = []
        for cs in criteria_scores:
            row = {
                "instruction_index": cs.instruction_index,
                "model": cs.model,
                **cs.scores,
                "raw_judge_output": cs.raw_judge_output,
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def pairwise_to_dataframes(
        self,
        results: list[PairwiseCriteriaResult],
        model_A_name: str,
        model_B_name: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Convert pairwise results to DataFrames + preferences.

        Returns:
            (df_scores_A, df_scores_B, preferences) where preferences is a
            Series of floats (0.0 = A wins, 0.5 = tie, 1.0 = B wins).
        """
        rows_A, rows_B = [], []
        prefs = []
        for r in results:
            rows_A.append({
                "instruction_index": r.instruction_index,
                "model": model_A_name,
                **r.scores_A,
            })
            rows_B.append({
                "instruction_index": r.instruction_index,
                "model": model_B_name,
                **r.scores_B,
            })
            prefs.append(r.preference)

        return pd.DataFrame(rows_A), pd.DataFrame(rows_B), pd.Series(prefs)

    def samplewise_to_dataframes(
        self,
        scores_A: list[CriteriaScore],
        scores_B: list[CriteriaScore],
        model_A_name: str,
        model_B_name: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Convert samplewise scores to DataFrames and derive preferences.

        Preferences are computed from the **weighted average** of criterion
        scores — no extra judge call required. Weights come from
        ``Criterion.weight`` (all 1.0 by default, i.e. plain mean).

        Returns:
            (df_scores_A, df_scores_B, preferences)
        """
        df_A = self.score_to_dataframe(scores_A)
        df_B = self.score_to_dataframe(scores_B)

        weights = {c.name: c.weight for c in self.criteria.criteria}

        prefs: list[float] = []
        for sa, sb in zip(scores_A, scores_B):
            avg_a = self._weighted_criteria_avg(sa.scores, weights)
            avg_b = self._weighted_criteria_avg(sb.scores, weights)

            if np.isnan(avg_a) or np.isnan(avg_b):
                prefs.append(0.5)          # can't decide — treat as tie
            elif avg_a > avg_b:
                prefs.append(0.0)          # A wins
            elif avg_b > avg_a:
                prefs.append(1.0)          # B wins
            else:
                prefs.append(0.5)          # tie

        prefs_series = pd.Series(prefs)

        n_a = (prefs_series < 0.5).sum()
        n_b = (prefs_series > 0.5).sum()
        n_t = (prefs_series == 0.5).sum()
        logger.info(
            "Samplewise preferences (from criteria weighted avg): "
            "A wins=%d, B wins=%d, ties=%d (of %d)",
            n_a, n_b, n_t, len(prefs),
        )

        return df_A, df_B, prefs_series

    @staticmethod
    def _weighted_criteria_avg(
        scores: dict[str, float],
        weights: dict[str, float],
    ) -> float:
        """Weighted average of criterion scores, skipping NaN criteria."""
        total_w = 0.0
        total_s = 0.0
        for cname, w in weights.items():
            s = scores.get(cname, float("nan"))
            if s == s:  # not NaN
                total_w += w
                total_s += w * s
        return total_s / total_w if total_w > 0 else float("nan")
