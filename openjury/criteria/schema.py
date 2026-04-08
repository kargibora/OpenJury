"""Schema for multi-criteria evaluation.

Defines data structures for criteria sets, individual criteria,
and per-completion criteria scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Criterion:
    """A single scoring criterion in a criteria set.

    Attributes:
        name: Short identifier (e.g. "fluency", "usefulness").
        description: Human-readable description shown to the judge.
        scale_min: Minimum score (inclusive).
        scale_max: Maximum score (inclusive).
        weight: Optional prior weight for aggregation (not used in BT fitting).
        score_references: Optional score anchors shown to the judge in the prompt.
            Mapping ``score -> description`` (e.g. ``{7: "...", 5: "...", 3: "...", 1: "..."}``).
    """

    name: str
    description: str
    scale_min: int = 1
    scale_max: int = 10
    weight: float = 1.0
    score_references: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scale_min >= self.scale_max:
            raise ValueError(
                f"Invalid criteria scale for '{self.name}': "
                f"{self.scale_min} > {self.scale_max}"
            )

        # JSON object keys arrive as strings; normalize to ints.
        normalized: dict[int, str] = {}
        for raw_score, text in self.score_references.items():
            score = int(raw_score)
            if not (self.scale_min <= score <= self.scale_max):
                raise ValueError(
                    f"Score reference {score} for '{self.name}' is outside "
                    f"the configured scale [{self.scale_min}, {self.scale_max}]"
                )
            normalized[score] = str(text).strip()
        self.score_references = normalized

    def prompt_block(self) -> str:
        """Render this criterion as a scoring instruction for the judge."""
        base = (
            f"**{self.name.title()}** ({self.scale_min}–{self.scale_max}): "
            f"{self.description}"
        )
        if not self.score_references:
            return base

        refs = "\n".join(
            f"   - {score}: {self.score_references[score]}"
            for score in sorted(self.score_references.keys(), reverse=True)
        )
        return f"{base}\n   Score references:\n{refs}"

@dataclass
class Criteria:
    """A collection of scoring criteria forming a complete criteria set.

    Attributes:
        name: Criteria identifier (e.g. "default", "my_custom").
        criteria: List of scoring criteria.
        description: Optional description of when to use this criteria set.
    """

    name: str
    criteria: list[Criterion]
    description: str = ""

    @property
    def criterion_names(self) -> list[str]:
        """Names of all criteria in order."""
        return [c.name for c in self.criteria]

    @property
    def num_criteria(self) -> int:
        """Number of criteria."""
        return len(self.criteria)

    def prompt_block(self) -> str:
        """Render the full criteria as scoring instructions."""
        lines = ["Score the following completion on each criterion:\n"]
        for i, dim in enumerate(self.criteria, 1):
            lines.append(f"{i}. {dim.prompt_block()}")
        return "\n".join(lines)


@dataclass
class CriteriaScore:
    """Scores for a single (instruction, completion) pair across all criteria.

    Attributes:
        instruction_index: Index linking back to the instruction.
        model: Model that generated the completion.
        scores: Dict mapping criterion name → score.
        raw_judge_output: Raw text from the judge (for debugging).
    """

    instruction_index: int | str
    model: str
    scores: dict[str, float]
    raw_judge_output: str = ""

    def to_list(self, criterion_names: list[str]) -> list[float]:
        """Return scores as an ordered vector matching criterion_names."""
        return [self.scores.get(name, 0.0) for name in criterion_names]


@dataclass
class PairwiseCriteriaResult:
    """Result of a pairwise criteria comparison: both A and B scored in one call.

    The judge sees both completions side-by-side, scores each on every
    criterion, and gives an overall preference.

    When swap-debiasing is used, ``scores_A`` / ``scores_B`` / ``preference``
    contain the **merged** (averaged) values.  The per-position originals are
    preserved in the ``*_original`` / ``*_swapped`` fields.

    Attributes:
        instruction_index: Index linking back to the instruction.
        scores_A: Dict mapping criterion name → (merged) score for completion A.
        scores_B: Dict mapping criterion name → (merged) score for completion B.
        preference: (Merged) overall preference: 0.0 = A wins, 0.5 = tie, 1.0 = B wins.
        raw_judge_output: Raw text from the A/B judge call.
        raw_judge_output_swapped: Raw text from the B/A judge call (if debiasing).
        scores_A_original: Per-criterion A scores from the A/B call (before merge).
        scores_B_original: Per-criterion B scores from the A/B call (before merge).
        preference_original: Preference from the A/B call (before merge).
        scores_A_swapped: Per-criterion A scores from the B/A call, flipped back.
        scores_B_swapped: Per-criterion B scores from the B/A call, flipped back.
        preference_swapped: Preference from the B/A call, flipped back.
    """

    instruction_index: int | str
    scores_A: dict[str, float]
    scores_B: dict[str, float]
    preference: float  # 0.0 = A wins, 0.5 = tie, 1.0 = B wins
    raw_judge_output: str = ""
    raw_judge_output_swapped: str | None = None
    # Per-position scores (before merge)
    scores_A_original: dict[str, float] | None = None
    scores_B_original: dict[str, float] | None = None
    preference_original: float | None = None
    scores_A_swapped: dict[str, float] | None = None
    scores_B_swapped: dict[str, float] | None = None
    preference_swapped: float | None = None

    def as_criteria_scores(
        self, model_A: str, model_B: str,
    ) -> tuple[CriteriaScore, CriteriaScore]:
        """Convert to two CriteriaScore objects (one per model)."""
        return (
            CriteriaScore(
                instruction_index=self.instruction_index,
                model=model_A,
                scores=self.scores_A,
                raw_judge_output=self.raw_judge_output,
            ),
            CriteriaScore(
                instruction_index=self.instruction_index,
                model=model_B,
                scores=self.scores_B,
                raw_judge_output=self.raw_judge_output,
            ),
        )


# ─────────────────────────────────────────────────────────────────
#  Multi-trial (Average-of-K) aggregated results
# ─────────────────────────────────────────────────────────────────


def _pref_to_label(pref: float) -> str:
    """Convert numeric preference to categorical label for majority vote."""
    if pref < 0.25:
        return "A"
    if pref > 0.75:
        return "B"
    return "tie"


@dataclass
class MultiTrialPairwiseResult:
    """Aggregated result of K pairwise judge trials for one sample.

    Stores all raw trial results plus aggregated statistics.  When
    ``n_trials == 1`` this is a thin wrapper around the single trial.

    Attributes:
        instruction_index: Index linking back to the instruction.
        trials: All K raw :class:`PairwiseCriteriaResult` objects.
        mean_scores_A: Mean criterion scores for A across trials.
        mean_scores_B: Mean criterion scores for B across trials.
        mean_preference: Mean continuous preference across trials.
        std_preference: Standard deviation of preference across trials.
        score_std_A: Per-criterion std for A across trials.
        score_std_B: Per-criterion std for B across trials.
        majority_label: ``"A"`` | ``"B"`` | ``"tie"`` by majority vote.
        self_agreement: Fraction of trials agreeing with majority label.
        n_trials: Number of trials.
    """

    instruction_index: int | str
    trials: list[PairwiseCriteriaResult]
    mean_scores_A: dict[str, float]
    mean_scores_B: dict[str, float]
    mean_preference: float
    std_preference: float
    score_std_A: dict[str, float]
    score_std_B: dict[str, float]
    majority_label: str
    self_agreement: float
    n_trials: int

    @classmethod
    def from_trials(
        cls, instruction_index: int | str, trials: list[PairwiseCriteriaResult],
    ) -> "MultiTrialPairwiseResult":
        """Aggregate K trial results into a single multi-trial result."""
        import numpy as _np

        k = len(trials)
        if k == 0:
            raise ValueError("Cannot aggregate zero trials")

        crit_names = list(trials[0].scores_A.keys())

        # Gather per-criterion arrays
        arr_A = {c: [t.scores_A.get(c, float("nan")) for t in trials] for c in crit_names}
        arr_B = {c: [t.scores_B.get(c, float("nan")) for t in trials] for c in crit_names}
        prefs = [t.preference for t in trials]

        mean_A = {c: float(_np.nanmean(arr_A[c])) for c in crit_names}
        mean_B = {c: float(_np.nanmean(arr_B[c])) for c in crit_names}
        std_A = {c: float(_np.nanstd(arr_A[c])) for c in crit_names}
        std_B = {c: float(_np.nanstd(arr_B[c])) for c in crit_names}
        mean_pref = float(_np.mean(prefs))
        std_pref = float(_np.std(prefs)) if k > 1 else 0.0

        # Majority vote
        labels = [_pref_to_label(p) for p in prefs]
        from collections import Counter
        counts = Counter(labels)
        majority = counts.most_common(1)[0][0]
        self_agr = counts[majority] / k

        return cls(
            instruction_index=instruction_index,
            trials=trials,
            mean_scores_A=mean_A,
            mean_scores_B=mean_B,
            mean_preference=mean_pref,
            std_preference=std_pref,
            score_std_A=std_A,
            score_std_B=std_B,
            majority_label=majority,
            self_agreement=self_agr,
            n_trials=k,
        )

    def as_single_result(self) -> PairwiseCriteriaResult:
        """Return the aggregated result as a standard PairwiseCriteriaResult.

        Uses mean scores/preference so downstream code that expects the
        single-trial type still works.
        """
        raw_outputs = [t.raw_judge_output for t in self.trials]
        raw_swapped = [t.raw_judge_output_swapped for t in self.trials if t.raw_judge_output_swapped]
        return PairwiseCriteriaResult(
            instruction_index=self.instruction_index,
            scores_A=self.mean_scores_A,
            scores_B=self.mean_scores_B,
            preference=self.mean_preference,
            raw_judge_output=raw_outputs[0] if raw_outputs else "",
            raw_judge_output_swapped=raw_swapped[0] if raw_swapped else None,
        )


@dataclass
class MultiTrialSamplewiseResult:
    """Aggregated result of K samplewise judge trials for one completion.

    Attributes:
        instruction_index: Index linking back to the instruction.
        model: Model that generated the completion.
        trials: All K raw :class:`CriteriaScore` objects.
        mean_scores: Mean criterion scores across trials.
        score_std: Per-criterion std across trials.
        n_trials: Number of trials.
    """

    instruction_index: int | str
    model: str
    trials: list[CriteriaScore]
    mean_scores: dict[str, float]
    score_std: dict[str, float]
    n_trials: int

    @classmethod
    def from_trials(
        cls,
        instruction_index: int | str,
        model: str,
        trials: list[CriteriaScore],
    ) -> "MultiTrialSamplewiseResult":
        """Aggregate K samplewise trial results."""
        import numpy as _np

        k = len(trials)
        if k == 0:
            raise ValueError("Cannot aggregate zero trials")

        crit_names = list(trials[0].scores.keys())
        arr = {c: [t.scores.get(c, float("nan")) for t in trials] for c in crit_names}
        mean_s = {c: float(_np.nanmean(arr[c])) for c in crit_names}
        std_s = {c: float(_np.nanstd(arr[c])) for c in crit_names}

        return cls(
            instruction_index=instruction_index,
            model=model,
            trials=trials,
            mean_scores=mean_s,
            score_std=std_s,
            n_trials=k,
        )

    def as_single_result(self) -> CriteriaScore:
        """Return the aggregated result as a standard CriteriaScore."""
        return CriteriaScore(
            instruction_index=self.instruction_index,
            model=self.model,
            scores=self.mean_scores,
            raw_judge_output=self.trials[0].raw_judge_output if self.trials else "",
        )

