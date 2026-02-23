"""Legacy instruction dataset loader — **backward-compatibility shim**.

New code should use :func:`openjury.datasets.load_dataset` instead.
This module delegates to the unified :class:`DatasetRegistry` and
returns a :class:`~pandas.DataFrame` for backward compatibility with
``generate_step.py`` and any external scripts.
"""

import pandas as pd

from openjury._logging import logger
from openjury.datasets import load_dataset as _load_dataset


def load_instructions(dataset: str, n_instructions: int | None = None) -> pd.DataFrame:
    """Load instructions as a DataFrame (backward-compatible API).

    .. deprecated::
        Use :func:`openjury.datasets.load_dataset` for the new
        :class:`~openjury.datasets.EvalDataset` interface with
        full metadata support.

    Args:
        dataset: Dataset name (e.g. ``"alpaca-eval"``, ``"m-arena-hard-en"``).
        n_instructions: Maximum number of instructions.

    Returns:
        DataFrame indexed by ``instruction_index`` with an ``instruction``
        column (and ``lang`` for multilingual datasets).
    """
    ds = _load_dataset(dataset, n=n_instructions)

    records = []
    for sample in ds.samples:
        row: dict = {
            "instruction_index": sample.instruction_id,
            "instruction": sample.instruction,
        }
        # Preserve language column for backward compat
        if "lang" in sample.metadata:
            row["lang"] = sample.metadata["lang"]
        records.append(row)

    df = pd.DataFrame(records)
    df = df.set_index("instruction_index").sort_index()

    logger.info(
        "Loaded [number]%d[/number] instructions for [dataset]%s[/dataset] (via legacy shim)",
        len(df), dataset,
    )
    return df


if __name__ == "__main__":
    instructions = load_instructions(dataset="alpaca-eval")
    logger.info("%s", instructions)
