"""Arena package — K-model evaluation framework.

Generate completions for K models, sample pairwise matchups, judge them,
and compute ratings (BT strengths, Elo, dimension weights).

Quick start::

    from openjury.arena import ArenaConfig

    config = ArenaConfig.load("arena.json")

Or run directly::

    uv run python -m openjury.cli.arena --config arena.json
"""

from openjury.arena.config import (
    AgreementConfig,
    ArenaConfig,
    ArenaResult,
    JudgeConfig,
    Match,
    MatchmakerConfig,
    MatchResult,
    ModelEntry,
    ModelScore,
)

__all__ = [
    "AgreementConfig",
    "ArenaConfig",
    "ArenaResult",
    "JudgeConfig",
    "Match",
    "MatchmakerConfig",
    "MatchResult",
    "ModelEntry",
    "ModelScore",
]
