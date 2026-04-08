from __future__ import annotations

import sys
import types

import pandas as pd

from openjury.datasets.loaders.lmsys import load_lmsys_140k


def test_lmsys_140k_loader_flattens_structured_content(tmp_path, monkeypatch):
    repo_dir = tmp_path / "hf_repo"
    repo_dir.mkdir()
    parquet_path = repo_dir / "train.parquet"

    df = pd.DataFrame(
        [
            {
                "id": "abc123",
                "model_a": "model-a",
                "model_b": "model-b",
                "winner": "model_b",
                "language": "en",
                "conversation_a": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Compare Tormenta20 with DnD5e"},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Answer A first paragraph."},
                            {"type": "text", "text": "Answer A second paragraph."},
                        ],
                    },
                ],
                "conversation_b": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Compare Tormenta20 with DnD5e"},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Answer B only paragraph."},
                        ],
                    },
                ],
            }
        ]
    )
    df.to_parquet(parquet_path, index=False)

    fake_hf = types.SimpleNamespace(snapshot_download=lambda **_kwargs: str(repo_dir))
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    ds = load_lmsys_140k(n=None, language="en", single_turn_only=True, seed=42)

    assert ds.name == "lmsys-140k"
    assert len(ds.samples) == 1

    sample = ds.samples[0]
    assert sample.instruction_id == "lmsys-140k-abc123"
    assert sample.instruction == "Compare Tormenta20 with DnD5e"
    assert sample.completions == {
        "model-a": "Answer A first paragraph.\nAnswer A second paragraph.",
        "model-b": "Answer B only paragraph.",
    }
    assert sample.human_pref == 1.0
    assert sample.metadata["lang"] == "en"
    assert sample.metadata["source"] == "lmsys-140k"
