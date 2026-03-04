"""Backward-compatibility shim — scorer alias from ``criteria`` package.

.. deprecated::
    Import from ``openjury.criteria.scorer`` instead.
"""

from openjury.criteria.scorer import CriteriaScorer as RubricScorer  # noqa: F401

__all__ = ["RubricScorer"]
