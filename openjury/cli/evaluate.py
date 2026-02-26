"""Umbrella CLI for evaluation tasks (arena, agreement).

Primary public interface:

- ``openjury-evaluate arena ...``
- ``openjury-evaluate agreement ...``

Task-specific commands (``openjury-arena``, ``openjury-agreement``) remain
supported as backward-compatible aliases.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

from openjury.cli import agreement as agreement_cli
from openjury.cli import arena as arena_cli


@dataclass(frozen=True)
class _EvalTask:
    name: str
    help: str
    handler: Callable[[list[str] | None], None]


_TASKS: dict[str, _EvalTask] = {
    "arena": _EvalTask(
        name="arena",
        help="K-model arena evaluation and ratings",
        handler=arena_cli.main,
    ),
    "agreement": _EvalTask(
        name="agreement",
        help="Human-vs-judge agreement evaluation on preference datasets",
        handler=agreement_cli.main,
    ),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="openjury-evaluate",
        description=(
            "Run OpenJury evaluation tasks. Use a subcommand for a specific "
            "evaluation pipeline (e.g. arena, agreement)."
        ),
    )
    subparsers = parser.add_subparsers(dest="task", metavar="TASK")
    for task in _TASKS.values():
        subparsers.add_parser(task.name, add_help=False, help=task.help)

    ns, rest = parser.parse_known_args(argv)

    if ns.task is None:
        parser.print_help()
        raise SystemExit(2)

    _TASKS[ns.task].handler(rest)


if __name__ == "__main__":
    main()

