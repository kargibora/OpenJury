"""Backward-compatibility shim for ``openjury.rubrics``.

.. deprecated::
    Use ``openjury.criteria`` directly. This package re-exports all public
    symbols under their legacy names for backward compatibility.

Quick migration guide::

    # OLD
    from openjury.rubrics import RubricScorer, get_rubric, Rubric

    # NEW
    from openjury.criteria import CriteriaScorer, get_criteria, Criteria
"""

from openjury.rubrics.schema import (  # noqa: F401
    Rubric,
    RubricDimension,
    RubricScore,
    PairwiseRubricResult,
)
from openjury.rubrics.defaults import (  # noqa: F401
    DEFAULT_RUBRICS,
    RUBRIC_REGISTRY,
    get_rubric,
    register_rubric,
)
from openjury.rubrics.scorer import RubricScorer  # noqa: F401

__all__ = [
    "Rubric",
    "RubricDimension",
    "RubricScore",
    "PairwiseRubricResult",
    "DEFAULT_RUBRICS",
    "RUBRIC_REGISTRY",
    "RubricScorer",
    "get_rubric",
    "register_rubric",
]
