"""Argparse override helpers for config-first CLIs.

These utilities keep precedence logic explicit and reusable:
explicit CLI flags win over config file values, which win over parser defaults.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from typing import Any

Getter = Callable[[Any], Any]
Setter = Callable[[Any, Any], None]
Presence = Callable[[Any], bool]


def collect_explicit_dests(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None,
) -> set[str]:
    """Return argparse dest names explicitly set in ``argv``.

    Supports both ``--flag value`` and ``--flag=value`` patterns.
    """
    raw = list(sys.argv[1:] if argv is None else argv)
    option_to_dest: dict[str, str] = {}
    for action in parser._actions:  # noqa: SLF001 - pragmatic argparse introspection
        for opt in getattr(action, "option_strings", ()):
            option_to_dest[opt] = action.dest

    explicit: set[str] = set()
    for token in raw:
        if token == "--":
            break
        if not token.startswith("-") or token == "-":
            continue
        opt = token.split("=", 1)[0]
        dest = option_to_dest.get(opt)
        if dest:
            explicit.add(dest)
    return explicit


def present_nonempty(value: Any) -> bool:
    """Return ``True`` for non-None values and non-empty containers/strings."""
    if value is None:
        return False
    if isinstance(value, (str, list, tuple, dict, set)) and not value:
        return False
    return True


def apply_mapping_from_config(
    args: argparse.Namespace,
    explicit_dests: set[str],
    cfg: Any,
    mappings: list[tuple[str, Getter]],
    *,
    present: Presence | None = None,
) -> None:
    """Apply config values into ``args`` when the CLI dest was not explicit."""
    is_present = present or (lambda v: v is not None)
    for dest, getter in mappings:
        if dest in explicit_dests:
            continue
        value = getter(cfg)
        if not is_present(value):
            continue
        setattr(args, dest, value)


def apply_mapping_to_config(
    config_obj: Any,
    args: argparse.Namespace,
    explicit_dests: set[str],
    mappings: list[tuple[str, Setter]],
    *,
    present: Presence | None = None,
) -> None:
    """Apply explicitly provided CLI values into a typed config object.

    ``mappings`` is a list of ``(arg_dest, setter)`` tuples where ``setter``
    mutates ``config_obj`` with the arg value.
    """
    is_present = present or (lambda _v: True)
    for dest, setter in mappings:
        if dest not in explicit_dests:
            continue
        value = getattr(args, dest, None)
        if not is_present(value):
            continue
        setter(config_obj, value)
