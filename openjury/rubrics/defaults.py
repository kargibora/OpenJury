"""Backward-compatibility shim — defaults aliases from ``criteria`` package.

.. deprecated::
    Import from ``openjury.criteria.defaults`` instead.
"""

from openjury.criteria.defaults import (  # noqa: F401
    CRITERIA_REGISTRY as RUBRIC_REGISTRY,
    get_criteria as get_rubric,
    register_criteria as register_rubric,
)

# Convenience alias
DEFAULT_RUBRICS = RUBRIC_REGISTRY

__all__ = ["RUBRIC_REGISTRY", "DEFAULT_RUBRICS", "get_rubric", "register_rubric"]
