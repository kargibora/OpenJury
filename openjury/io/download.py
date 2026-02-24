from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

from openjury.common.paths import data_root


def download_hf(name: str, local_path: Path) -> None:
    local_path.mkdir(exist_ok=True, parents=True)
    snapshot_download(
        repo_id="geoalgo/llmjudge",
        repo_type="dataset",
        allow_patterns=f"*{name}*",
        local_dir=local_path,
        force_download=False,
    )


def download_all() -> None:
    print(f"Downloading all dataset in {data_root}")
    for dataset in ["alpaca-eval", "arena-hard", "m-arena-hard"]:
        local_path_tables = data_root / "tables"
        download_hf(name=dataset, local_path=local_path_tables)

    snapshot_download(
        repo_id="geoalgo/multilingual-contexts-to-be-completed",
        repo_type="dataset",
        allow_patterns="*",
        local_dir=data_root / "contexts",
        force_download=False,
    )
