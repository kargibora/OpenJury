"""Unified judge-score cache for OpenJury arena.

Stores per-model samplewise rubric scores as **parquet** files under a
deterministic path::

    $OPENJURY_DATA/scores/{judge_key}/{rubric}/{dataset}/{model_key}/{n_key}.parquet

Cache key = ``(judge_model, rubric_name, model, dataset, n_instructions)``.

The design mirrors :mod:`openjury.completion_cache` — filesystem-based,
human-inspectable, and subsettable (``all.parquet`` can serve any ``nK``
request).

Quick start::

    from openjury.score_cache import score_cache

    # Look up
    scores = score_cache.get(
        judge="VLLM/Qwen/Qwen3-32B", rubric="default",
        model="VLLM/Qwen/Qwen2.5-0.5B-Instruct", dataset="alpaca-eval", n=100,
    )

    # Store
    score_cache.put(
        model_scores, judge="VLLM/Qwen/Qwen3-32B", rubric="default",
        model="VLLM/Qwen/Qwen2.5-0.5B-Instruct", dataset="alpaca-eval", n=100,
    )

    # Get or score (lazy — only scores on cache miss)
    scores = score_cache.get_or_score(
        judge="VLLM/Qwen/Qwen3-32B", rubric="default",
        model="VLLM/Qwen/Qwen2.5-0.5B-Instruct", dataset="alpaca-eval", n=100,
        score_fn=lambda: arena_judge.score_model(...),
    )

CLI::

    uv run python -m openjury.score_cache list
    uv run python -m openjury.score_cache list --dataset alpaca-eval
    uv run python -m openjury.score_cache clear --judge VLLM/Qwen/Qwen3-32B --rubric default
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from openjury._logging import logger
from openjury.arena.config import ModelScore
from openjury.utils import data_root


# ═════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════


def _fs_key(name: str) -> str:
    """Filesystem-safe key: ``VLLM/Qwen/Qwen3-32B`` → ``VLLM__Qwen__Qwen3-32B``."""
    return name.replace("/", "__")


def _n_key(n: int | None) -> str:
    return "all" if n is None else f"n{n}"


# ═════════════════════════════════════════════════════════════════════
#  Serialisation helpers (ModelScore ↔ DataFrame row)
# ═════════════════════════════════════════════════════════════════════


def _scores_to_df(model_scores: list[ModelScore]) -> pd.DataFrame:
    """Convert a list of :class:`ModelScore` to a DataFrame for storage.

    Columns: ``[model, instruction_index, sample_id, scores_json, completion, raw_judge_output]``
    """
    rows = []
    for ms in model_scores:
        rows.append({
            "model": ms.model,
            "instruction_index": ms.instruction_index,
            "sample_id": ms.sample_id,
            "scores_json": json.dumps(ms.scores, ensure_ascii=False),
            "completion": ms.completion,
            "raw_judge_output": ms.raw_judge_output,
        })
    return pd.DataFrame(rows)


def _df_to_scores(df: pd.DataFrame) -> list[ModelScore]:
    """Restore ``list[ModelScore]`` from a cached DataFrame."""
    scores: list[ModelScore] = []
    for _, row in df.iterrows():
        scores.append(ModelScore(
            model=row["model"],
            instruction_index=int(row["instruction_index"]),
            sample_id=str(row.get("sample_id", "") or ""),
            scores=json.loads(row["scores_json"]),
            completion=row.get("completion", ""),
            raw_judge_output=row.get("raw_judge_output", ""),
        ))
    return scores


# ═════════════════════════════════════════════════════════════════════
#  Cache entry
# ═════════════════════════════════════════════════════════════════════


@dataclass
class ScoreCacheEntry:
    """Metadata for a single cached score file."""

    judge: str
    rubric: str
    model: str
    dataset: str
    n: int | None
    path: Path
    size_mb: float

    def __str__(self) -> str:
        n_str = str(self.n) if self.n is not None else "all"
        return (
            f"{self.judge} | {self.rubric} | "
            f"{self.dataset}/{self.model} (n={n_str}, {self.size_mb:.1f}MB)"
        )


# ═════════════════════════════════════════════════════════════════════
#  ScoreCache
# ═════════════════════════════════════════════════════════════════════


class ScoreCache:
    """Filesystem-based judge-score cache.

    Directory layout::

        {root}/scores/{judge_key}/{rubric}/{dataset}/{model_key}/{n_key}.parquet

    Parameters:
        root: Base data directory.  Defaults to ``$OPENJURY_DATA``
              (typically ``~/openjury-eval-data``).

    Lookup strategy for ``get(judge, rubric, model, dataset, n=K)``:
        1. Exact match ``nK.parquet`` → return it.
        2. Fall back to ``all.parquet`` (superset) → load and return first K rows.
    """

    def __init__(self, root: Path | None = None):
        self.root = (root or data_root) / "scores"

    # ── Path helpers ─────────────────────────────────────────────

    def _path(
        self,
        judge: str,
        rubric: str,
        model: str,
        dataset: str,
        n: int | None,
    ) -> Path:
        return (
            self.root
            / _fs_key(judge)
            / rubric
            / dataset
            / _fs_key(model)
            / f"{_n_key(n)}.parquet"
        )

    # ── Core API ─────────────────────────────────────────────────

    def get(
        self,
        judge: str,
        rubric: str,
        model: str,
        dataset: str,
        n: int | None = None,
    ) -> list[ModelScore] | None:
        """Look up cached scores.

        Returns a list of :class:`ModelScore`, or ``None`` if nothing
        is cached.
        """
        # 1. Exact match
        exact = self._path(judge, rubric, model, dataset, n)
        if exact.exists():
            logger.info("Score cache hit: %s", exact)
            df = pd.read_parquet(exact)
            return _df_to_scores(df)

        # 2. Superset: "all" cache can serve any smaller n
        if n is not None:
            all_path = self._path(judge, rubric, model, dataset, None)
            if all_path.exists():
                logger.info("Score cache hit (subsetting from 'all'): %s", all_path)
                df = pd.read_parquet(all_path)
                df = df.head(n)
                return _df_to_scores(df)

        return None

    def put(
        self,
        model_scores: list[ModelScore],
        judge: str,
        rubric: str,
        model: str,
        dataset: str,
        n: int | None = None,
    ) -> Path:
        """Store scores in cache.  Returns the written path."""
        df = _scores_to_df(model_scores)
        path = self._path(judge, rubric, model, dataset, n)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info("Cached %d scores → %s", len(model_scores), path)
        return path

    def get_or_score(
        self,
        judge: str,
        rubric: str,
        model: str,
        dataset: str,
        n: int | None,
        score_fn: Callable[[], list[ModelScore]],
        ignore_cache: bool = False,
    ) -> list[ModelScore]:
        """Return cached scores or invoke judge + cache them.

        Args:
            judge: Judge model specification string.
            rubric: Rubric name.
            model: Candidate model being scored.
            dataset: Dataset name.
            n: Number of instructions (``None`` = all).
            score_fn: Zero-arg callable that returns ``list[ModelScore]``.
            ignore_cache: If ``True``, always re-score.

        Returns:
            List of :class:`ModelScore`.
        """
        if not ignore_cache:
            cached = self.get(judge, rubric, model, dataset, n)
            if cached is not None:
                return cached

        logger.info(
            "Scoring model: judge=%s rubric=%s model=%s dataset=%s n=%s",
            judge, rubric, model, dataset, n,
        )
        scores = score_fn()
        self.put(scores, judge, rubric, model, dataset, n)
        return scores

    # ── Query helpers ────────────────────────────────────────────

    def exists(
        self,
        judge: str,
        rubric: str,
        model: str,
        dataset: str,
        n: int | None = None,
    ) -> bool:
        """Check whether scores are cached (exact or superset)."""
        if self._path(judge, rubric, model, dataset, n).exists():
            return True
        if n is not None and self._path(judge, rubric, model, dataset, None).exists():
            return True
        return False

    def resolve(
        self,
        judge: str,
        rubric: str,
        model: str,
        dataset: str,
        n: int | None = None,
    ) -> Path | None:
        """Return the cache *file* path that would serve this request, or ``None``."""
        exact = self._path(judge, rubric, model, dataset, n)
        if exact.exists():
            return exact
        if n is not None:
            all_path = self._path(judge, rubric, model, dataset, None)
            if all_path.exists():
                return all_path
        return None

    # ── Listing ──────────────────────────────────────────────────

    def list(
        self,
        judge: str | None = None,
        rubric: str | None = None,
        dataset: str | None = None,
    ) -> list[ScoreCacheEntry]:
        """List all cached scores, optionally filtered."""
        entries: list[ScoreCacheEntry] = []
        if not self.root.exists():
            return entries

        judge_dirs = (
            [self.root / _fs_key(judge)] if judge else sorted(self.root.iterdir())
        )
        for j_dir in judge_dirs:
            if not j_dir.is_dir():
                continue
            judge_name = j_dir.name.replace("__", "/")

            rubric_dirs = (
                [j_dir / rubric] if rubric else sorted(j_dir.iterdir())
            )
            for r_dir in rubric_dirs:
                if not r_dir.is_dir():
                    continue
                rubric_name = r_dir.name

                ds_dirs = (
                    [r_dir / dataset] if dataset else sorted(r_dir.iterdir())
                )
                for ds_dir in ds_dirs:
                    if not ds_dir.is_dir():
                        continue
                    ds_name = ds_dir.name

                    for model_dir in sorted(ds_dir.iterdir()):
                        if not model_dir.is_dir():
                            continue
                        model_name = model_dir.name.replace("__", "/")

                        for parquet in sorted(model_dir.glob("*.parquet")):
                            stem = parquet.stem
                            n = None if stem == "all" else int(stem.lstrip("n"))
                            size_mb = parquet.stat().st_size / (1024 * 1024)
                            entries.append(ScoreCacheEntry(
                                judge=judge_name,
                                rubric=rubric_name,
                                model=model_name,
                                dataset=ds_name,
                                n=n,
                                path=parquet,
                                size_mb=size_mb,
                            ))
        return entries

    # ── Cleanup ──────────────────────────────────────────────────

    def clear(
        self,
        judge: str | None = None,
        rubric: str | None = None,
        model: str | None = None,
        dataset: str | None = None,
    ) -> int:
        """Remove cached scores.  Returns count of removed files.

        - ``clear()`` — remove everything
        - ``clear(judge="...")`` — remove all for a judge
        - ``clear(judge="...", rubric="default", model="...", dataset="...")``
          — remove specific
        """
        removed = 0
        if judge and rubric and model and dataset:
            model_dir = (
                self.root / _fs_key(judge) / rubric / dataset / _fs_key(model)
            )
            if model_dir.exists():
                for f in model_dir.glob("*.parquet"):
                    f.unlink()
                    removed += 1
                if not any(model_dir.iterdir()):
                    model_dir.rmdir()
        else:
            # Broader sweep — use listing to find matching entries
            for entry in self.list(judge=judge, rubric=rubric, dataset=dataset):
                if model and entry.model != model:
                    continue
                entry.path.unlink(missing_ok=True)
                removed += 1

        if removed:
            logger.info("Cleared %d cached score files", removed)
        return removed


# ═════════════════════════════════════════════════════════════════════
#  Module-level singleton (mirrors completion_cache.cache)
# ═════════════════════════════════════════════════════════════════════

score_cache = ScoreCache()


# ═════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════


def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        prog="score_cache",
        description="Manage the OpenJury judge-score cache.",
    )
    sub = parser.add_subparsers(dest="command")

    # list
    ls = sub.add_parser("list", help="List cached scores")
    ls.add_argument("--judge", default=None)
    ls.add_argument("--rubric", default=None)
    ls.add_argument("--dataset", default=None)

    # clear
    clr = sub.add_parser("clear", help="Remove cached scores")
    clr.add_argument("--judge", default=None)
    clr.add_argument("--rubric", default=None)
    clr.add_argument("--model", default=None)
    clr.add_argument("--dataset", default=None)

    # path
    pth = sub.add_parser("path", help="Show cache path for a key")
    pth.add_argument("--judge", required=True)
    pth.add_argument("--rubric", required=True)
    pth.add_argument("--model", required=True)
    pth.add_argument("--dataset", required=True)
    pth.add_argument("--n", type=int, default=None)

    args = parser.parse_args()

    sc = ScoreCache()

    if args.command == "list":
        entries = sc.list(
            judge=args.judge, rubric=args.rubric, dataset=args.dataset,
        )
        if not entries:
            print("(no cached scores)")
        for e in entries:
            print(e)

    elif args.command == "clear":
        n = sc.clear(
            judge=args.judge,
            rubric=args.rubric,
            model=args.model,
            dataset=args.dataset,
        )
        print(f"Removed {n} cached score file(s).")

    elif args.command == "path":
        print(sc._path(args.judge, args.rubric, args.model, args.dataset, args.n))

    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
