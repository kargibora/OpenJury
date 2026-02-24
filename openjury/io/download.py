from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


def download_hf(name: str, local_path: Path) -> None:
    local_path.mkdir(exist_ok=True, parents=True)
    snapshot_download(
        repo_id="geoalgo/llmjudge",
        repo_type="dataset",
        allow_patterns=f"*{name}*",
        local_dir=local_path,
        force_download=False,
    )
