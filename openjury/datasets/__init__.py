"""Unified dataset abstraction for OpenJury.

Every evaluation dataset — instruction-only, multilingual, or human-preference —
is normalised to :class:`EvalSample` / :class:`EvalDataset` and loaded through
the :class:`DatasetRegistry`.

Quick start::

    from openjury.datasets import load_dataset

    # Instruction dataset
    ds = load_dataset("alpaca-eval", n=100)

    # Multilingual (dynamic suffix → language kwarg)
    ds = load_dataset("m-arena-hard-en", n=50)

    # Human-preference dataset
    ds = load_dataset("lmsys", n=200, language="en")
    assert ds.has_human_pref

    # List available datasets
    from openjury.datasets import DatasetRegistry
    print(DatasetRegistry.list_datasets())

Adding a new dataset
--------------------
Create a loader in ``openjury/datasets/loaders/`` and decorate it::

    @DatasetRegistry.register("my-dataset")
    def load_my_dataset(*, n=None, **kw) -> EvalDataset:
        ...

Then import it in ``openjury/datasets/loaders/__init__.py``.
"""

from openjury.datasets.schema import EvalDataset, EvalSample
from openjury.datasets.registry import DatasetRegistry

# Importing loaders triggers all @register decorators
import openjury.datasets.loaders  # noqa: F401


def load_dataset(name: str, *, n: int | None = None, **kwargs) -> EvalDataset:
    """Convenience function — delegates to :meth:`DatasetRegistry.load`.

    Args:
        name: Dataset name (e.g. ``"alpaca-eval"``, ``"m-arena-hard-en"``).
        n: Maximum number of samples.
        **kwargs: Forwarded to the dataset loader.

    Returns:
        An :class:`EvalDataset`.
    """
    return DatasetRegistry.load(name, n=n, **kwargs)


__all__ = [
    "EvalSample",
    "EvalDataset",
    "DatasetRegistry",
    "load_dataset",
]
