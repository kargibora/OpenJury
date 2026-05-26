"""Helpers for locating and validating the OpenJury project root."""

from __future__ import annotations

from pathlib import Path

from openjury._logging import logger


def is_project_root(path: Path) -> bool:
    """Return True when *path* looks like the OpenJury repo root."""
    return (
        path.is_dir()
        and (path / "pyproject.toml").is_file()
        and (path / "openjury").is_dir()
    )


def normalize_project_dir(path_str: str) -> str:
    """Normalize an OpenJury project directory.

    Accepts either the repo root itself or a wrapper directory that contains
    exactly one immediate child repo root. Raises early if the path still does
    not resolve to a usable OpenJury project root.
    """
    candidate = Path(path_str).expanduser().resolve()
    if is_project_root(candidate):
        return str(candidate)

    child_matches = [child for child in candidate.iterdir() if is_project_root(child)]
    if len(child_matches) == 1:
        resolved = child_matches[0]
        logger.warning(
            "Resolved OPENJURY project dir %s -> %s based on nested pyproject/openjury package",
            candidate,
            resolved,
        )
        return str(resolved)

    raise EnvironmentError(
        "OPENJURY project dir is not a valid repo root: "
        f"{candidate}. Expected a directory containing both pyproject.toml "
        "and openjury/."
    )
