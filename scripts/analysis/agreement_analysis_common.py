"""Shared helpers for OpenJury analysis scripts (backward-compat shim).

This module re-exports everything from :mod:`plotting` so that existing
``from agreement_analysis_common import …`` imports keep working.  **New
code should import from** ``plotting`` **directly.**
"""
# ruff: noqa: F401, F403
from plotting import *  # re-export the full public API

# Legacy aliases — some older scripts use these names.
from plotting import apply_theme as apply_plot_theme  # noqa: F401
from plotting import shorten as shorten_model_name  # noqa: F401
from plotting import pref_to_label as preference_to_label  # noqa: F401
