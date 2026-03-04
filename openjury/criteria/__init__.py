"""Criteria package for structured LLM evaluation.

Provides structured, interpretable scoring of LLM completions along
configurable criteria (fluency, usefulness, clarity, style, adherence, etc.)
using an LLM judge with custom criteria sets.

Quick start::

    from openjury.criteria import CriteriaScorer, get_criteria
    from openjury.models.factory import make_model

    judge = make_model("VLLM/Qwen/Qwen2.5-32B-Instruct")
    criteria = get_criteria("default")
    scorer = CriteriaScorer(judge_model=judge, criteria=criteria)

    scores = scorer.score(
        instructions=["Write a poem"],
        completions=["Roses are red..."],
        model_name="test",
    )

Modules:
    - ``openjury.criteria.schema``: dataclasses for criteria definitions and scores
    - ``openjury.criteria.defaults``: built-in criteria registry
    - ``openjury.criteria.io``: custom criteria loading/registration from JSON
    - ``openjury.criteria.scorer``: criteria scoring with an LLM judge
    - ``openjury.criteria.pipeline``: shared criteria pipeline output helpers
"""

from openjury.criteria.schema import (
    Criteria,
    Criterion,
    CriteriaScore,
    PairwiseCriteriaResult,
)
from openjury.criteria.defaults import (
    CRITERIA_REGISTRY,
    get_criteria,
    register_criteria,
)
from openjury.criteria.scorer import CriteriaScorer

__all__ = [
    "Criteria",
    "Criterion",
    "CriteriaScore",
    "PairwiseCriteriaResult",
    "CRITERIA_REGISTRY",
    "CriteriaScorer",
    "get_criteria",
    "register_criteria",
]
