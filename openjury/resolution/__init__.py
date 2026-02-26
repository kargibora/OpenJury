"""Shared helpers for CLI/config resolution and precedence handling."""

from .argparse_overrides import (
    apply_mapping_from_config,
    apply_mapping_to_config,
    collect_explicit_dests,
    present_nonempty,
)

__all__ = [
    "apply_mapping_from_config",
    "apply_mapping_to_config",
    "collect_explicit_dests",
    "present_nonempty",
]
