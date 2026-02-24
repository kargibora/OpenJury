from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_df(filename: Path, **pandas_kwargs) -> pd.DataFrame:
    assert filename.exists(), f"Dataframe file not found at {filename}"
    if filename.name.endswith(".csv.zip") or filename.name.endswith(".csv"):
        return pd.read_csv(filename, **pandas_kwargs)
    assert filename.name.endswith(".parquet"), f"Unsupported extension {filename}"
    return pd.read_parquet(filename, **pandas_kwargs)
