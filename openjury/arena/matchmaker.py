"""Matchmaker — select pairwise matchups from K models.

Strategies control *which* (model_i, model_j, instruction) triples get
judged.  For K models and N instructions the full grid has C(K,2) × N
entries — the matchmaker lets you trade thoroughness for budget.

Usage::

    from openjury.arena.matchmaker import get_matchmaker

    matches = get_matchmaker("round_robin")(
        models=["VLLM/A", "VLLM/B", "VLLM/C"],
        instruction_indices=list(range(100)),
    )
"""

from __future__ import annotations

import random
from itertools import combinations
from typing import Callable

from openjury.arena.config import Match


# ═══════════════════════════════════════════════════════════════════
#  Strategies
# ═══════════════════════════════════════════════════════════════════


def round_robin(
    models: list[str],
    instruction_indices: list[int],
    **_kwargs,
) -> list[Match]:
    """All C(K,2) model pairs × all instructions.

    Total matches = C(K,2) × N.  Guarantees every pair is compared on
    every instruction — the most thorough (and most expensive) strategy.
    """
    matches: list[Match] = []
    for i, j in combinations(range(len(models)), 2):
        for idx in instruction_indices:
            matches.append(Match(
                model_a=models[i],
                model_b=models[j],
                instruction_index=idx,
            ))
    return matches


def random_pairs(
    models: list[str],
    instruction_indices: list[int],
    *,
    n_matches: int | None = None,
    seed: int = 42,
    **_kwargs,
) -> list[Match]:
    """Budget-constrained random sampling.

    Randomly samples ``n_matches`` from the full grid of
    C(K,2) × N possible matchups.  Defaults to ``K × N`` matches if
    ``n_matches`` is not given — roughly equivalent to each model
    appearing in N matchups.
    """
    rng = random.Random(seed)
    all_pairs = list(combinations(range(len(models)), 2))

    if n_matches is None:
        n_matches = len(models) * len(instruction_indices)

    pool: list[Match] = []
    for _ in range(n_matches):
        i, j = rng.choice(all_pairs)
        idx = rng.choice(instruction_indices)
        pool.append(Match(
            model_a=models[i],
            model_b=models[j],
            instruction_index=idx,
        ))
    return pool


def balanced_random(
    models: list[str],
    instruction_indices: list[int],
    *,
    n_matches: int | None = None,
    seed: int = 42,
    **_kwargs,
) -> list[Match]:
    """Random sampling that ensures every model pair appears roughly equally.

    Cycles through all C(K,2) pairs in round-robin order, picking a random
    instruction for each, until the budget is exhausted.  This guarantees
    balanced coverage even with a small budget.
    """
    rng = random.Random(seed)
    all_pairs = list(combinations(range(len(models)), 2))
    rng.shuffle(all_pairs)

    if n_matches is None:
        n_matches = len(models) * len(instruction_indices)

    matches: list[Match] = []
    pair_idx = 0
    while len(matches) < n_matches:
        i, j = all_pairs[pair_idx % len(all_pairs)]
        idx = rng.choice(instruction_indices)
        matches.append(Match(
            model_a=models[i],
            model_b=models[j],
            instruction_index=idx,
        ))
        pair_idx += 1
    return matches


# ═══════════════════════════════════════════════════════════════════
#  Registry
# ═══════════════════════════════════════════════════════════════════

MatchmakerFn = Callable[..., list[Match]]

MATCHMAKER_REGISTRY: dict[str, MatchmakerFn] = {
    "round_robin": round_robin,
    "random_pairs": random_pairs,
    "balanced_random": balanced_random,
}


def get_matchmaker(name: str) -> MatchmakerFn:
    """Look up a matchmaker strategy by name.

    Args:
        name: Strategy identifier.

    Returns:
        A callable ``(models, instruction_indices, **kwargs) -> list[Match]``.

    Raises:
        KeyError: If name is not registered.
    """
    if name not in MATCHMAKER_REGISTRY:
        available = ", ".join(sorted(MATCHMAKER_REGISTRY))
        raise KeyError(f"Unknown matchmaker '{name}'. Available: {available}")
    return MATCHMAKER_REGISTRY[name]
