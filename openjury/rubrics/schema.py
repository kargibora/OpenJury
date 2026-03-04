"""Backward-compatibility shim — schema aliases from ``criteria`` package.

.. deprecated::
    Import from ``openjury.criteria.schema`` instead.
"""

from openjury.criteria.schema import (  # noqa: F401
    Criterion as RubricDimension,
    Criteria as Rubric,
    CriteriaScore as RubricScore,
    PairwiseCriteriaResult as PairwiseRubricResult,
)

__all__ = ["RubricDimension", "Rubric", "RubricScore", "PairwiseRubricResult"]
