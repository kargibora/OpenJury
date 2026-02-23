"""Unified completion cache for OpenJury.

Stores model completions as **parquet** files under a deterministic path::

    $OPENJURY_DATA/completions/{dataset}/{model_key}/{n_key}.parquet

Cache key = ``(model, dataset, n_instructions)``.

Quick start::

    from openjury.completion_cache import cache

    # Look up
    df = cache.get("VLLM/Qwen/Qwen2.5-0.5B-Instruct", "alpaca-eval", n=100)

    # Store
    cache.put(df, "VLLM/Qwen/Qwen2.5-0.5B-Instruct", "alpaca-eval", n=100)

    # Get or generate (lazy — only generates on cache miss)
    df = cache.get_or_generate(
        model="VLLM/Qwen/Qwen2.5-0.5B-Instruct",
        dataset="alpaca-eval",
        n=100,
        generate_fn=lambda: generate_instructions(...),
    )

    # List cached completions
    for entry in cache.list():
        print(entry)

CLI::

    uv run python -m openjury.completion_cache list
    uv run python -m openjury.completion_cache list --dataset alpaca-eval
    uv run python -m openjury.completion_cache clear --model VLLM/Qwen/Qwen2.5-0.5B-Instruct --dataset alpaca-eval
    uv run python -m openjury.completion_cache path --model VLLM/Qwen/Qwen2.5-0.5B-Instruct --dataset alpaca-eval --n 100
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from openjury._logging import logger
from openjury.utils import data_root


# ═════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════


def _model_key(model: str) -> str:
    """Filesystem-safe model key.

    ``VLLM/Qwen/Qwen2.5-0.5B-Instruct`` → ``VLLM__Qwen__Qwen2.5-0.5B-Instruct``
    """
    return model.replace("/", "__")


def _n_key(n: int | None) -> str:
    """Cache filename stem for a given n_instructions value."""
    return "all" if n is None else f"n{n}"


# ═════════════════════════════════════════════════════════════════════
#  Cache entry
# ═════════════════════════════════════════════════════════════════════


@dataclass
class CacheEntry:
    """Metadata for a single cached completion file."""

    model: str
    dataset: str
    n: int | None
    path: Path
    size_mb: float

    def __str__(self) -> str:
        n_str = str(self.n) if self.n is not None else "all"
        return f"{self.dataset}/{self.model} (n={n_str}, {self.size_mb:.1f}MB)"


# ═════════════════════════════════════════════════════════════════════
#  CompletionCache
# ═════════════════════════════════════════════════════════════════════


class CompletionCache:
    """Filesystem-based completion cache.

    Directory layout::

        {root}/completions/{dataset}/{model_key}/{n_key}.parquet

    Parameters:
        root: Base data directory.  Defaults to ``$OPENJURY_DATA``
              (typically ``~/openjury-eval-data``).

    Lookup strategy for ``get(model, dataset, n=K)``:
        1. Exact match ``nK.parquet`` → return it.
        2. Fall back to ``all.parquet`` (superset) → load and return first K rows.
    """

    def __init__(self, root: Path | None = None):
        self.root = (root or data_root) / "completions"

    # ── Path helpers ─────────────────────────────────────────────

    def _path(self, model: str, dataset: str, n: int | None) -> Path:
        return self.root / dataset / _model_key(model) / f"{_n_key(n)}.parquet"

    # ── Core API ─────────────────────────────────────────────────

    def get(
        self,
        model: str,
        dataset: str,
        n: int | None = None,
    ) -> pd.DataFrame | None:
        """Look up cached completions.

        Returns a DataFrame with columns ``[completion, instruction_index]``,
        or ``None`` if nothing is cached.
        """
        # 1. Exact match
        exact = self._path(model, dataset, n)
        if exact.exists():
            logger.info("Cache hit: %s", exact)
            return pd.read_parquet(exact)

        # 2. Superset: "all" cache can serve any smaller n
        if n is not None:
            all_path = self._path(model, dataset, None)
            if all_path.exists():
                logger.info("Cache hit (subsetting from 'all'): %s", all_path)
                df = pd.read_parquet(all_path)
                return df.head(n)

        return None

    def put(
        self,
        df: pd.DataFrame,
        model: str,
        dataset: str,
        n: int | None = None,
    ) -> Path:
        """Store completions in cache.  Returns the written path."""
        path = self._path(model, dataset, n)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info("Cached %d completions → %s", len(df), path)
        return path

    def get_or_generate(
        self,
        model: str,
        dataset: str,
        n: int | None,
        generate_fn: Callable[[], pd.DataFrame],
        ignore_cache: bool = False,
    ) -> pd.DataFrame:
        """Return cached completions or generate + cache them.

        Args:
            model: Model specification string.
            dataset: Dataset name.
            n: Number of instructions (``None`` = all).
            generate_fn: Zero-arg callable that returns a DataFrame with
                columns ``[completion, instruction_index]``.
            ignore_cache: If ``True``, always regenerate.

        Returns:
            DataFrame with columns ``[completion, instruction_index]``.
        """
        if not ignore_cache:
            df = self.get(model, dataset, n)
            if df is not None:
                return df

        logger.info("Generating completions: model=%s dataset=%s n=%s", model, dataset, n)
        df = generate_fn()
        self.put(df, model, dataset, n)
        return df

    # ── Query helpers ────────────────────────────────────────────

    def exists(self, model: str, dataset: str, n: int | None = None) -> bool:
        """Check whether completions are cached (exact or superset)."""
        if self._path(model, dataset, n).exists():
            return True
        if n is not None and self._path(model, dataset, None).exists():
            return True
        return False

    def resolve(self, model: str, dataset: str, n: int | None = None) -> Path | None:
        """Return the cache *file* path that would serve this request, or ``None``."""
        exact = self._path(model, dataset, n)
        if exact.exists():
            return exact
        if n is not None:
            all_path = self._path(model, dataset, None)
            if all_path.exists():
                return all_path
        return None

    # ── Listing ──────────────────────────────────────────────────

    def list(self, dataset: str | None = None) -> list[CacheEntry]:
        """List all cached completions, optionally filtered by dataset."""
        entries: list[CacheEntry] = []
        if not self.root.exists():
            return entries

        ds_dirs = [self.root / dataset] if dataset else sorted(self.root.iterdir())
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
                    entries.append(CacheEntry(
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
        model: str | None = None,
        dataset: str | None = None,
    ) -> int:
        """Remove cached completions.  Returns count of removed files.

        - ``clear()`` — remove everything
        - ``clear(dataset="alpaca-eval")`` — remove all for a dataset
        - ``clear(model="VLLM/...", dataset="alpaca-eval")`` — remove specific
        """
        removed = 0
        if model and dataset:
            model_dir = self.root / dataset / _model_key(model)
            if model_dir.exists():
                for f in model_dir.glob("*.parquet"):
                    f.unlink()
                    removed += 1
                if not any(model_dir.iterdir()):
                    model_dir.rmdir()
        elif dataset:
            ds_dir = self.root / dataset
            if ds_dir.exists():
                for f in ds_dir.rglob("*.parquet"):
                    f.unlink()
                    removed += 1
                # Clean up empty dirs
                _cleanup_empty_dirs(ds_dir)
        else:
            if self.root.exists():
                for f in self.root.rglob("*.parquet"):
                    f.unlink()
                    removed += 1
                _cleanup_empty_dirs(self.root)

        if removed:
            logger.info("Cleared %d cached completion file(s)", removed)
        return removed


def _cleanup_empty_dirs(base: Path) -> None:
    """Remove empty subdirectories bottom-up."""
    for d in sorted(base.rglob("*"), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()


# ═════════════════════════════════════════════════════════════════════
#  Module-level singleton
# ═════════════════════════════════════════════════════════════════════

cache = CompletionCache()


# ═════════════════════════════════════════════════════════════════════
#  CLI: ``uv run python -m openjury.completion_cache <command>``
# ═════════════════════════════════════════════════════════════════════


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="openjury.completion_cache",
        description="Manage the OpenJury completion cache.",
    )
    sub = parser.add_subparsers(dest="command")

    # list
    ls = sub.add_parser("list", aliases=["ls"], help="List cached completions.")
    ls.add_argument("--dataset", default=None)

    # clear
    cl = sub.add_parser("clear", help="Clear cached completions.")
    cl.add_argument("--dataset", default=None)
    cl.add_argument("--model", default=None)
    cl.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip confirmation prompt.",
    )

    # path — show the cache path for a specific model+dataset+n
    p = sub.add_parser("path", help="Show cache path for a model+dataset.")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--n", type=int, default=None)

    args = parser.parse_args()

    if args.command in ("list", "ls"):
        entries = cache.list(dataset=args.dataset)
        if not entries:
            print("No cached completions found.")
            print(f"Cache root: {cache.root}")
            return
        print(f"{'Dataset':<20} {'Model':<45} {'N':>6} {'Size':>8}  Path")
        print("-" * 120)
        for e in entries:
            n_str = str(e.n) if e.n is not None else "all"
            print(f"{e.dataset:<20} {e.model:<45} {n_str:>6} {e.size_mb:>7.1f}M  {e.path}")
        print(f"\nTotal: {len(entries)} cached completion(s)")

    elif args.command == "clear":
        target = "all cached completions"
        if args.model and args.dataset:
            target = f"{args.model} / {args.dataset}"
        elif args.dataset:
            target = f"all models for {args.dataset}"
        if not args.yes:
            resp = input(f"Clear {target}? [y/N] ")
            if resp.lower() != "y":
                print("Aborted.")
                return
        n = cache.clear(model=args.model, dataset=args.dataset)
        print(f"Cleared {n} cached file(s).")

    elif args.command == "path":
        resolved = cache.resolve(args.model, args.dataset, args.n)
        if resolved:
            print(f"Cached: {resolved}")
        else:
            expected = cache._path(args.model, args.dataset, args.n)
            print(f"Not cached.")
            print(f"Expected path: {expected}")

    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
