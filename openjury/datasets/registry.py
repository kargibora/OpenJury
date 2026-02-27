"""Dataset registry — decorator-based plugin system for dataset loaders.

Add a new dataset by writing a loader function and decorating it::

    from openjury.datasets.registry import DatasetRegistry

    @DatasetRegistry.register("my-dataset")
    def load_my_dataset(*, n: int | None = None, **kwargs) -> EvalDataset:
        ...

Then load it anywhere::

    from openjury.datasets import load_dataset
    ds = load_dataset("my-dataset", n=100)
"""

from __future__ import annotations

from typing import Any, Callable

from openjury._logging import logger
from openjury.datasets.schema import EvalDataset


# Type for a loader function:
#   (*, n: int | None, **kwargs) -> EvalDataset
LoaderFn = Callable[..., EvalDataset]


class DatasetRegistry:
    """Central registry mapping dataset names to loader functions.

    Follows the same decorator pattern as :class:`ModelRegistry`.
    """

    _loaders: dict[str, LoaderFn] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[LoaderFn], LoaderFn]:
        """Decorator to register a dataset loader under *name*.

        The decorated function must accept ``n: int | None`` as a keyword
        argument (number of samples to return) and may accept arbitrary
        ``**kwargs`` for dataset-specific options (e.g. ``language``).

        Example::

            @DatasetRegistry.register("alpaca-eval")
            def _load_alpaca_eval(*, n: int | None = None, **kw) -> EvalDataset:
                ...
        """

        def decorator(fn: LoaderFn) -> LoaderFn:
            if name in cls._loaders:
                logger.warning(
                    "Dataset '%s' is being re-registered (previous loader will be overwritten).",
                    name,
                )
            cls._loaders[name] = fn
            return fn

        return decorator

    @classmethod
    def load(cls, name: str, *, n: int | None = None, **kwargs: Any) -> EvalDataset:
        """Load a dataset by name.

        For datasets with dynamic suffixes (e.g. ``m-arena-hard-en``),
        the registry first tries an exact match, then strips the last
        ``-suffix`` and passes it as a ``language`` kwarg.

        Args:
            name: Registered dataset name or a dynamic variant.
            n: Maximum number of samples (``None`` = all).
            **kwargs: Forwarded to the loader function.

        Returns:
            An :class:`EvalDataset` with at most *n* samples.

        Raises:
            KeyError: If no loader matches *name*.
        """
        balance_by = kwargs.pop("balance_by", None)
        sampler_seed = int(kwargs.get("seed", 42) or 42)

        def _load_with_optional_balancing(loader: LoaderFn, **loader_kwargs: Any) -> EvalDataset:
            # Balanced sampling needs the full candidate pool to rebalance
            # deficits across classes. We therefore bypass loader-side n-cuts.
            if balance_by and n is not None:
                ds_full = loader(n=None, **loader_kwargs)
                before = len(ds_full)
                ds_bal = ds_full.sample_balanced(by=str(balance_by), n=n, seed=sampler_seed)
                logger.info(
                    "Balanced sample by '%s': requested n=%d -> %d/%d samples",
                    balance_by, n, len(ds_bal), before,
                )
                return ds_bal
            return loader(n=n, **loader_kwargs)

        # ── Exact match ──────────────────────────────────────────
        if name in cls._loaders:
            ds = _load_with_optional_balancing(cls._loaders[name], **kwargs)
            logger.info(
                "Loaded dataset '%s': %d samples (%s)",
                name, len(ds), ", ".join(sorted(ds.metadata_keys)) or "no metadata",
            )
            return ds

        # ── Dynamic suffix match (e.g. m-arena-hard-en) ─────────
        parts = name.rsplit("-", 1)
        if len(parts) == 2:
            base, suffix = parts
            if base in cls._loaders:
                ds = _load_with_optional_balancing(
                    cls._loaders[base],
                    language=suffix,
                    **kwargs,
                )
                # Override the dataset name to include the suffix
                ds.name = name
                logger.info(
                    "Loaded dataset '%s' (base='%s', language='%s'): %d samples",
                    name, base, suffix, len(ds),
                )
                return ds

        available = ", ".join(sorted(cls._loaders.keys()))
        raise KeyError(
            f"Unknown dataset '{name}'. "
            f"Available: {available}. "
            f"Register new datasets with @DatasetRegistry.register()."
        )

    @classmethod
    def list_datasets(cls) -> list[str]:
        """Return sorted list of registered dataset names."""
        return sorted(cls._loaders.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a dataset (exact or dynamic suffix) is loadable."""
        if name in cls._loaders:
            return True
        base = name.rsplit("-", 1)[0]
        return base in cls._loaders
