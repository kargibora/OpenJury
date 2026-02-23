"""Arena-Hard instruction dataset loader.

Loads 500 instructions from the Arena-Hard-Auto benchmark with metadata:
``question_id`` (hex hash), ``category``, and ``cluster``.
"""

from __future__ import annotations

from openjury.datasets.registry import DatasetRegistry
from openjury.datasets.schema import EvalDataset, EvalSample
from openjury.utils import data_root, download_hf, read_df


@DatasetRegistry.register("arena-hard")
def load_arena_hard(*, n: int | None = None, **_kwargs) -> EvalDataset:
    """Load the Arena-Hard instruction dataset.

    Metadata per sample:
        - ``question_id``: Hex hash identifier (matches ``uid`` in the
          ``lmarena/arena-hard-auto`` repository).
        - ``category``: High-level category (e.g. ``arena-hard-v0.1``).
        - ``cluster``: Topic cluster (e.g. ``ABC Sequence Puzzles & Groups``).

    Args:
        n: Maximum number of samples.

    Returns:
        :class:`EvalDataset` with instructions, stable IDs, and metadata.
    """
    local_path_tables = data_root / "tables"
    download_hf(name="arena-hard", local_path=local_path_tables)
    df = read_df(local_path_tables / "instructions" / "arena-hard.csv")

    samples: list[EvalSample] = []
    for i, row in df.iterrows():
        inst_id = str(row["instruction_index"]) if "instruction_index" in row else str(i)

        meta: dict[str, str] = {}
        if "question_id" in row:
            meta["question_id"] = str(row["question_id"])
        if "category" in row:
            meta["category"] = str(row["category"])
        if "domain" in row:
            meta["cluster"] = str(row["domain"])

        samples.append(EvalSample(
            instruction=row["instruction"],
            instruction_id=inst_id,
            metadata=meta,
        ))

    if n is not None:
        samples = samples[:n]

    return EvalDataset(
        name="arena-hard",
        samples=samples,
        metadata_schema={
            "question_id": "Hex hash UID matching the arena-hard-auto repo",
            "category": "Benchmark category (e.g. arena-hard-v0.1)",
            "cluster": "Topic cluster",
        },
    )
