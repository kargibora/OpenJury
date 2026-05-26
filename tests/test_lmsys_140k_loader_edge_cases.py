import pandas as pd

from openjury.datasets.loaders.lmsys import load_lmsys_140k


def _turn(role: str, text: str | None):
    content = [] if text is None else [{"type": "text", "text": text}]
    return {"role": role, "content": content}


def test_lmsys_140k_loader_drops_rows_with_empty_completion(tmp_path, monkeypatch):
    df = pd.DataFrame(
        [
            {
                "id": "good-row",
                "model_a": "model-a",
                "model_b": "model-b",
                "winner": "model_a",
                "language": "en",
                "conversation_a": [
                    _turn("user", "What is 2+2?"),
                    _turn("assistant", "It is 4."),
                ],
                "conversation_b": [
                    _turn("user", "What is 2+2?"),
                    _turn("assistant", "The answer is four."),
                ],
            },
            {
                "id": "bad-row",
                "model_a": "model-c",
                "model_b": "model-d",
                "winner": "model_b",
                "language": "en",
                "conversation_a": [
                    _turn("user", "Explain BT."),
                    _turn("assistant", None),
                ],
                "conversation_b": [
                    _turn("user", "Explain BT."),
                    _turn("assistant", "Bradley-Terry is a pairwise model."),
                ],
            },
        ]
    )
    parquet_path = tmp_path / "sample.parquet"
    df.to_parquet(parquet_path, index=False)

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **_kwargs: str(tmp_path),
    )

    dataset = load_lmsys_140k(n=None, single_turn_only=True)

    assert len(dataset.samples) == 1
    sample = dataset.samples[0]
    assert sample.instruction == "What is 2+2?"
    assert sample.completions == {
        "model-a": "It is 4.",
        "model-b": "The answer is four.",
    }
    assert sample.human_pref == 0.0
