from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from openjury.annotate_config import AnnotateConfig, PairingConfig
from openjury.arena.config import JudgeConfig, ModelEntry
from openjury.common.pair_annotation import PairJudgement
from openjury.datasets.schema import EvalDataset, EvalSample
from openjury.pipelines.model_annotation import run_annotate


def test_run_annotate_builds_sparse_dataset_inline_matches(monkeypatch, tmp_path):
    dataset = EvalDataset(
        name="lmsys",
        samples=[
            EvalSample(
                instruction="i1",
                instruction_id="id1",
                metadata={"lang": "en"},
                completions={"Model/X": "x1", "Model/Y": "y1"},
                human_pref=0.0,
            ),
            EvalSample(
                instruction="i2",
                instruction_id="id2",
                metadata={"lang": "en"},
                completions={"Model/Y": "y2", "Model/Z": "z2"},
                human_pref=1.0,
            ),
        ],
    )

    monkeypatch.setattr(
        "openjury.pipelines.model_annotation.load_dataset",
        lambda *args, **kwargs: dataset,
    )
    monkeypatch.setattr(
        "openjury.pipelines.model_annotation._resolve_challenger_completions",
        lambda **kwargs: pd.DataFrame(
            {
                "instruction_index": ["id1", "id2"],
                "completion": ["c1", "c2"],
            }
        ),
    )
    monkeypatch.setattr(
        "openjury.pipelines.model_annotation.make_model",
        lambda *args, **kwargs: object(),
    )
    fake_criteria = SimpleNamespace(
        criteria=[
            SimpleNamespace(
                name="quality",
                description="Quality",
                scale_min=1,
                scale_max=10,
                weight=1.0,
                score_references=None,
            )
        ],
        criterion_names=["quality"],
    )
    monkeypatch.setattr(
        "openjury.pipelines.model_annotation._load_criteria",
        lambda *_args, **_kwargs: fake_criteria,
    )

    class _FakeScorer:
        system_prompt = {"pairwise": "sys"}

        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(
        "openjury.pipelines.model_annotation.CriteriaScorer",
        _FakeScorer,
    )

    def _fake_score_pairs_pairwise(**kwargs):
        pairs = kwargs["pairs"]
        return [
            PairJudgement(
                sample_id=pair.sample_id,
                instruction_index=pair.instruction_index,
                model_a=pair.model_a,
                model_b=pair.model_b,
                scores_a={"quality": 8.0},
                scores_b={"quality": 6.0},
                preference=0.0,
                raw_judge_output="raw",
            )
            for pair in pairs
        ]

    monkeypatch.setattr(
        "openjury.pipelines.model_annotation.score_pairs_pairwise",
        _fake_score_pairs_pairwise,
    )

    cfg = AnnotateConfig(
        dataset="lmsys",
        n_instructions=2,
        challenger=ModelEntry(name="VLLM/openai/gpt-oss-20b", gpus=4),
        judge=JudgeConfig(model="VLLM/Qwen/Qwen3-32B", gpus=4, mode="pairwise"),
        pairing=PairingConfig(source="challenger_vs_dataset_inline", strategy="all"),
        output_dir=str(tmp_path),
    )

    payload = run_annotate(cfg)

    assert payload["metadata"]["challenger_model"] == "VLLM/openai/gpt-oss-20b"
    assert payload["metadata"]["models"] == [
        "VLLM/openai/gpt-oss-20b",
        "Model/X",
        "Model/Y",
        "Model/Z",
    ]
    assert len(payload["matches"]) == 4
    assert (tmp_path / "annotate_annotations.json").exists()


def test_run_annotate_builds_dataset_pair_matches(monkeypatch, tmp_path):
    dataset = EvalDataset(
        name="lmsys",
        samples=[
            EvalSample(
                instruction="i1",
                instruction_id="id1",
                metadata={"lang": "en"},
                completions={"Model/A": "a1", "Model/B": "b1"},
                human_pref=0.5,
            ),
        ],
    )

    monkeypatch.setattr(
        "openjury.pipelines.model_annotation.load_dataset",
        lambda *args, **kwargs: dataset,
    )
    monkeypatch.setattr(
        "openjury.pipelines.model_annotation.make_model",
        lambda *args, **kwargs: object(),
    )
    fake_criteria = SimpleNamespace(
        criteria=[
            SimpleNamespace(
                name="quality",
                description="Quality",
                scale_min=1,
                scale_max=10,
                weight=1.0,
                score_references=None,
            )
        ],
        criterion_names=["quality"],
    )
    monkeypatch.setattr(
        "openjury.pipelines.model_annotation._load_criteria",
        lambda *_args, **_kwargs: fake_criteria,
    )

    class _FakeScorer:
        system_prompt = {"pairwise": "sys"}

        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(
        "openjury.pipelines.model_annotation.CriteriaScorer",
        _FakeScorer,
    )

    monkeypatch.setattr(
        "openjury.pipelines.model_annotation.score_pairs_pairwise",
        lambda **kwargs: [
            PairJudgement(
                sample_id=kwargs["pairs"][0].sample_id,
                instruction_index=kwargs["pairs"][0].instruction_index,
                model_a=kwargs["pairs"][0].model_a,
                model_b=kwargs["pairs"][0].model_b,
                scores_a={"quality": 8.0},
                scores_b={"quality": 6.0},
                preference=0.0,
            )
        ],
    )

    cfg = AnnotateConfig(
        dataset="lmsys",
        n_instructions=1,
        judge=JudgeConfig(model="VLLM/Qwen/Qwen3-32B", gpus=4, mode="pairwise"),
        pairing=PairingConfig(source="dataset_pairs"),
        output_dir=str(tmp_path),
    )

    payload = run_annotate(cfg)

    assert payload["metadata"]["pairing_source"] == "dataset_pairs"
    assert payload["metadata"]["challenger_model"] is None
    assert payload["matches"][0]["human_preference"] == 0.5
