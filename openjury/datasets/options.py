"""Shared dataset selection options used across pipelines and executors."""

from __future__ import annotations

from dataclasses import dataclass
import re


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
    balance_by: str | None = None

    def loader_kwargs(self) -> dict[str, object]:
        """Build kwargs for ``openjury.datasets.load_dataset``."""
        kwargs: dict[str, object] = {}
        if self.language:
            kwargs["language"] = self.language
        if self.seed != 42:
            kwargs["seed"] = self.seed
        if self.balance_by:
            kwargs["balance_by"] = self.balance_by
        return kwargs

    def cache_key(self) -> str:
        """Dataset cache namespace including selection knobs.

        Cache entries for different dataset selections must not collide.
        This key keeps the original dataset name when no selection knobs are
        active, but adds a stable suffix for language / seeded sampling /
        balanced sampling variants.
        """
        parts: list[str] = []
        if self.language:
            parts.append(f"lang={self.language}")
        if self.balance_by:
            parts.append(f"balance={self.balance_by}")
        # Seed only affects cacheable sub-sampling when n is requested.
        if self.n_instructions is not None:
            parts.append(f"seed={self.seed}")
        if not parts:
            return self.name

        def _slug(v: str) -> str:
            return re.sub(r"[^A-Za-z0-9._=-]+", "-", v).strip("-")

        suffix = "__".join(_slug(p) for p in parts)
        return f"{self.name}__sel__{suffix}"
