from __future__ import annotations

import os
from pathlib import Path

from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache

data_root = Path(
    os.environ.get("OPENJURY_DATA", Path("~/openjury-eval-data/").expanduser())
).expanduser()


def set_langchain_cache() -> None:
    set_llm_cache(SQLiteCache(database_path=str(data_root / ".langchain.db")))
