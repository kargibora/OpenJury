"""Shared dataset selection options used across pipelines and executors."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetOptions:
    """Dataset identity + optional selection/filter knobs.

    This is intentionally small and conservative so both task configs
    (arena/agreement) and executors (local/SLURM) can share it without
    forcing task-specific fields into a single schema.
    """

    name: str
    n_instructions: int | None = None
    language: str | None = None
    seed: int = 42

    def loader_kwargs(self) -> dict[str, object]:
        """Build kwargs for ``openjury.datasets.load_dataset``."""
        kwargs: dict[str, object] = {}
        if self.language:
            kwargs["language"] = self.language
        if self.seed != 42:
            kwargs["seed"] = self.seed
        return kwargs
