"""Compute arena ratings from pairwise match results.

Three complementary rating systems:

1. **Multi-player Bradley-Terry** — maximum-likelihood strength parameters
   θ_m for each model, fitted via logistic regression on all pairwise
   outcomes: P(i ≻ j) = σ(θ_i - θ_j).

2. **Elo ratings** — iterative updates (classic chess-style) that converge
   quickly and are easy to interpret.

3. **Dimension weights** — re-uses the existing
   :class:`FeatureBradleyTerry` to learn *which rubric dimensions matter*
   by stacking all pairwise score differences across the arena.

Usage::

    from openjury.arena.ratings import compute_ratings

    ratings = compute_ratings(
        models=["A", "B", "C"],
        matches=match_results,
        dimension_names=["fluency", "usefulness", "clarity"],
    )
    print(ratings["bt_strengths"])   # {model: θ}
    print(ratings["elo"])            # {model: Elo}
    print(ratings["win_matrix"])     # {model_i: {model_j: win_rate}}
    print(ratings["dimension_weights"])  # {dim: weight}
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from openjury._logging import logger
from openjury.arena.config import MatchResult
from openjury.bradley_terry import FeatureBradleyTerry


# ═════════════════════════════════════════════════════════════════════
#  Public API
# ═════════════════════════════════════════════════════════════════════


def compute_ratings(
    models: list[str],
    matches: list[MatchResult],
    dimension_names: list[str],
    bt_regularization: float = 0.01,
    elo_k: float = 32.0,
    elo_initial: float = 1500.0,
    include_log_length: bool = True,
) -> dict:
    """Compute all arena ratings from pairwise match results.

    Args:
        models: List of participating model names.
        matches: All :class:`MatchResult` from the arena.
        dimension_names: Rubric dimension names (for weight fitting).
        bt_regularization: L2 regularization for BT fitting.
        elo_k: Elo K-factor (sensitivity to each game).
        elo_initial: Starting Elo rating.
        include_log_length: If ``True``, adds ``log_length`` (log of
            completion character count) as an extra feature in the
            Feature-BT regression to detect judge length bias.

    Returns:
        Dict with keys ``bt_strengths``, ``elo``, ``win_matrix``,
        ``aggregate_win_rates``, ``dimension_weights``,
        ``dimension_weight_accuracy``.
    """
    win_mat = compute_win_matrix(models, matches)
    agg_wr = compute_aggregate_win_rates(models, win_mat)
    bt = fit_multi_bt(models, matches, regularization=bt_regularization)
    elo = compute_elo(models, matches, k=elo_k, initial=elo_initial)
    dim_w, dim_acc = fit_dimension_weights(
        matches, dimension_names, regularization=bt_regularization,
        include_log_length=include_log_length,
    )

    return {
        "bt_strengths": bt,
        "elo": elo,
        "win_matrix": win_mat,
        "aggregate_win_rates": agg_wr,
        "dimension_weights": dim_w,
        "dimension_weight_accuracy": dim_acc,
    }


# ═════════════════════════════════════════════════════════════════════
#  Win matrix
# ═════════════════════════════════════════════════════════════════════


def compute_win_matrix(
    models: list[str],
    matches: list[MatchResult],
) -> dict[str, dict[str, float]]:
    """Compute pairwise win rates between all model pairs.

    Returns:
        Nested dict ``{model_i: {model_j: win_rate_of_i_over_j}}``.
        Win rate is in [0, 1].  Ties count as 0.5 for each side.
    """
    wins: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for m in matches:
        a, b = m.model_a, m.model_b
        counts[a][b] += 1
        counts[b][a] += 1

        if m.preference < 0.5:
            wins[a][b] += 1.0
            wins[b][a] += 0.0
        elif m.preference > 0.5:
            wins[a][b] += 0.0
            wins[b][a] += 1.0
        else:
            wins[a][b] += 0.5
            wins[b][a] += 0.5

    matrix: dict[str, dict[str, float]] = {}
    for mi in models:
        matrix[mi] = {}
        for mj in models:
            if mi == mj:
                matrix[mi][mj] = 0.5
            elif counts[mi][mj] > 0:
                matrix[mi][mj] = wins[mi][mj] / counts[mi][mj]
            else:
                matrix[mi][mj] = 0.5  # no data
    return matrix


# ═════════════════════════════════════════════════════════════════════
#  Aggregate win rate
# ═════════════════════════════════════════════════════════════════════


def compute_aggregate_win_rates(
    models: list[str],
    win_matrix: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Compute aggregate (average) win rate for each model.

    For model *i*, the aggregate win rate is the mean of its pairwise
    win rates against all other models:

    .. math::

        \\text{agg\\_wr}(i) = \\frac{1}{K-1} \\sum_{j \\neq i} \\text{win\\_rate}(i, j)

    This is an intuitive, easy-to-interpret metric that complements
    BT strengths and Elo ratings.

    Args:
        models: List of participating model names.
        win_matrix: Pairwise win-rate matrix from :func:`compute_win_matrix`.

    Returns:
        ``{model_name: aggregate_win_rate}`` — in [0, 1], higher is better.
    """
    K = len(models)
    if K < 2:
        return {m: 0.5 for m in models}

    agg: dict[str, float] = {}
    for mi in models:
        wr_sum = 0.0
        n_opponents = 0
        for mj in models:
            if mi == mj:
                continue
            wr_sum += win_matrix.get(mi, {}).get(mj, 0.5)
            n_opponents += 1
        agg[mi] = wr_sum / max(n_opponents, 1)

    logger.info("Aggregate win rates:")
    for m, wr in sorted(agg.items(), key=lambda x: -x[1]):
        logger.info("  %s: %.1f%%", m.rsplit("/", 1)[-1], wr * 100)

    return agg


# ═════════════════════════════════════════════════════════════════════
#  Multi-player Bradley-Terry (per-model strengths)
# ═════════════════════════════════════════════════════════════════════


def fit_multi_bt(
    models: list[str],
    matches: list[MatchResult],
    regularization: float = 0.01,
) -> dict[str, float]:
    """Fit multi-player BT strengths θ_m for each model.

    Uses the formulation: P(i ≻ j) = σ(θ_i − θ_j).
    Each match becomes one training sample with a one-hot feature vector:
    +1 for model_a, −1 for model_b.  Logistic regression on this matrix
    yields the strength parameters.

    Args:
        models: List of model names.
        matches: Pairwise match results.
        regularization: L2 penalty.

    Returns:
        ``{model_name: θ}`` — higher is better.
    """
    n_models = len(models)
    model_to_idx = {m: i for i, m in enumerate(models)}

    # Build feature matrix: one row per match, one column per model
    rows_X: list[list[float]] = []
    rows_y: list[float] = []

    for m in matches:
        if m.model_a not in model_to_idx or m.model_b not in model_to_idx:
            continue
        # Skip ties
        if abs(m.preference - 0.5) <= 0.05:
            continue

        row = [0.0] * n_models
        row[model_to_idx[m.model_a]] = 1.0
        row[model_to_idx[m.model_b]] = -1.0
        rows_X.append(row)

        # y = 1 means model_a wins (preference < 0.5)
        rows_y.append(1.0 if m.preference < 0.5 else 0.0)

    if len(rows_y) < 2:
        logger.warning("Too few decisive matches (%d) to fit BT strengths", len(rows_y))
        return {m: 0.0 for m in models}

    X = np.array(rows_X)
    y = np.array(rows_y)

    C = 1.0 / max(regularization, 1e-12)
    lr = LogisticRegression(
        C=C,
        fit_intercept=False,  # strengths are relative
        max_iter=1000,
        solver="lbfgs",
    )
    lr.fit(X, y)

    strengths = lr.coef_[0]
    # Center so mean = 0
    strengths = strengths - strengths.mean()

    result = {m: float(strengths[i]) for i, m in enumerate(models)}

    logger.info("BT strengths fitted on %d matches:", len(rows_y))
    for m, s in sorted(result.items(), key=lambda x: -x[1]):
        logger.info("  %s: %+.4f", m.rsplit("/", 1)[-1], s)

    return result


# ═════════════════════════════════════════════════════════════════════
#  Elo ratings
# ═════════════════════════════════════════════════════════════════════


def compute_elo(
    models: list[str],
    matches: list[MatchResult],
    k: float = 32.0,
    initial: float = 1500.0,
    seed: int = 42,
) -> dict[str, float]:
    """Compute Elo ratings from match results.

    Processes matches in random order (seeded for reproducibility) and
    applies standard Elo updates.

    Args:
        models: List of model names.
        matches: Match results.
        k: K-factor (update sensitivity).
        initial: Starting Elo for all models.
        seed: Random seed for match ordering.

    Returns:
        ``{model_name: Elo}`` — higher is better.
    """
    import random as _random

    rng = _random.Random(seed)
    elo: dict[str, float] = {m: initial for m in models}

    shuffled = list(matches)
    rng.shuffle(shuffled)

    for m in shuffled:
        ra = elo.get(m.model_a, initial)
        rb = elo.get(m.model_b, initial)

        ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        eb = 1.0 - ea

        # Actual score: 1 = win, 0.5 = tie, 0 = loss
        if m.preference < 0.5:
            sa, sb = 1.0, 0.0
        elif m.preference > 0.5:
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5

        elo[m.model_a] = ra + k * (sa - ea)
        elo[m.model_b] = rb + k * (sb - eb)

    logger.info("Elo ratings computed from %d matches:", len(matches))
    for m, r in sorted(elo.items(), key=lambda x: -x[1]):
        logger.info("  %s: %.0f", m.rsplit("/", 1)[-1], r)

    return elo


# ═════════════════════════════════════════════════════════════════════
#  Dimension weights (which rubric dimensions predict the winner?)
# ═════════════════════════════════════════════════════════════════════


def fit_dimension_weights(
    matches: list[MatchResult],
    dimension_names: list[str],
    regularization: float = 0.01,
    include_log_length: bool = True,
) -> tuple[dict[str, float], float]:
    """Fit rubric-dimension weights across all arena matches.

    Stacks the score differences from every match into one dataset and
    fits the existing :class:`FeatureBradleyTerry` to learn which rubric
    dimensions predict the winner.

    When ``include_log_length`` is True, ``log(len(completion) + 1)``
    (character count) is appended as an extra feature.  A positive
    weight indicates the judge (or human) prefers longer completions;
    this is a well-known bias signal in LLM-as-judge evaluation.

    Args:
        matches: All match results with ``scores_a`` / ``scores_b``.
        dimension_names: Ordered list of rubric dimension names.
        regularization: L2 regularization.
        include_log_length: Add ``log_length`` feature.

    Returns:
        ``(weight_dict, accuracy)`` — dimension weights and training accuracy.
    """
    if not matches or not dimension_names:
        return {d: 0.0 for d in dimension_names}, 0.0

    # Decide effective feature names
    feature_names = list(dimension_names)
    if include_log_length:
        feature_names.append("log_length")

    # Stack all pairwise score diffs
    rows_a: list[dict[str, float]] = []
    rows_b: list[dict[str, float]] = []
    prefs: list[float] = []

    for m in matches:
        row_a = {d: m.scores_a.get(d, float("nan")) for d in dimension_names}
        row_b = {d: m.scores_b.get(d, float("nan")) for d in dimension_names}
        # log(char_length + 1) as a proxy for token count
        if include_log_length:
            row_a["log_length"] = float(np.log1p(len(m.completion_a)))
            row_b["log_length"] = float(np.log1p(len(m.completion_b)))
        rows_a.append(row_a)
        rows_b.append(row_b)
        prefs.append(m.preference)

    scores_A = pd.DataFrame(rows_a, columns=feature_names)
    scores_B = pd.DataFrame(rows_b, columns=feature_names)
    preferences = pd.Series(prefs)

    bt = FeatureBradleyTerry(
        dimension_names=feature_names,
        regularization=regularization,
    )

    try:
        bt.fit(scores_A, scores_B, preferences, verbose=True)
    except (ValueError, RuntimeError) as e:
        logger.warning("Could not fit dimension weights: %s", e)
        return {d: 0.0 for d in dimension_names}, 0.0

    # Bootstrap CIs (log significant dimensions)
    if len(preferences) >= 20:
        try:
            boot = bt.bootstrap_weights(
                scores_A, scores_B, preferences, n_bootstrap=1000,
            )
            for dim, stats in boot.items():
                if dim == "_intercept":
                    continue
                sig = "✓" if stats.get("significant") else "✗"
                logger.info(
                    "  %s %-22s w=%+.4f  95%% CI=[%+.4f, %+.4f]",
                    sig, dim, stats["weight"],
                    stats["ci_lower"], stats["ci_upper"],
                )
        except (ValueError, RuntimeError) as e:
            logger.warning("Could not compute bootstrap CIs: %s", e)

    # Accuracy
    acc = 0.0
    try:
        proba = bt.predict_proba(scores_A, scores_B)
        preds = (proba >= 0.5).astype(int)
        labels = (preferences.values < 0.5).astype(int)
        not_tie = np.abs(preferences.values - 0.5) > 0.05
        if not_tie.sum() > 0:
            acc = float((preds[not_tie] == labels[not_tie]).mean())
    except (ValueError, RuntimeError):
        pass

    return bt.weight_dict(), acc
