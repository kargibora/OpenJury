"""Alpaca-Eval instruction dataset loader."""

from __future__ import annotations

from openjury.datasets.registry import DatasetRegistry
from openjury.datasets.schema import EvalDataset, EvalSample
from openjury.common.paths import data_root
from openjury.io.download import download_hf
from openjury.io.tabular import read_df


@DatasetRegistry.register("alpaca-eval")
def load_alpaca_eval(*, n: int | None = None, **_kwargs) -> EvalDataset:
    """Load the Alpaca-Eval instruction dataset.

    Metadata per sample:
        - ``domain`` — topic domain (e.g. ``"helpful_base"``, ``"koala"``).
        - ``subdataset`` — source sub-dataset within Alpaca-Eval.

    Args:
        n: Maximum number of samples.

    Returns:
        :class:`EvalDataset` with ``instruction`` and per-sample metadata.
    """
    local_path_tables = data_root / "tables"
    download_hf(name="alpaca-eval", local_path=local_path_tables)
    df = read_df(local_path_tables / "instructions" / "alpaca-eval.csv")

    samples: list[EvalSample] = []
    for i, row in df.iterrows():
        metadata: dict = {}
        if "domain" in row and row["domain"]:
            metadata["domain"] = str(row["domain"])
        if "dataset" in row and row["dataset"]:
            metadata["subdataset"] = str(row["dataset"])
        inst_id = str(row["instruction_index"]) if "instruction_index" in row else str(i)
        samples.append(EvalSample(
            instruction=row["instruction"],
            instruction_id=inst_id,
            metadata=metadata,
        ))

    if n is not None:
        samples = samples[:n]

    return EvalDataset(
        name="alpaca-eval",
        samples=samples,
        metadata_schema={
            "domain": "Topic domain (e.g. helpful_base, koala)",
            "subdataset": "Source sub-dataset within Alpaca-Eval",
        },
    )
