from __future__ import annotations

from openjury.annotate_config import (
    AnnotateConfig,
    AnnotateGenerationConfig,
    PairingConfig,
)
from openjury.arena.config import JudgeConfig, ModelEntry


def test_annotate_config_roundtrip(tmp_path):
    cfg = AnnotateConfig(
        dataset="lmsys",
        n_instructions=500,
        seed=7,
        balance_by="models",
        challenger=ModelEntry(
            name="VLLM/openai/gpt-oss-20b",
            gpus=4,
            quantization="fp8",
        ),
        judge=JudgeConfig(
            model="VLLM/Qwen/Qwen3-32B",
            gpus=4,
            mode="pairwise",
            max_model_len=16384,
            enforce_eager=True,
        ),
        criteria="default",
        pairing=PairingConfig(
            source="challenger_vs_dataset_inline",
            strategy="all",
            seed=9,
        ),
        generation=AnnotateGenerationConfig(
            max_tokens=2048,
            truncate_input_chars=4096,
            ignore_cache=True,
            gpu_memory_utilization=0.85,
            use_tqdm=True,
        ),
        output_dir="results/annotate/test",
        ignore_score_cache=True,
        include_completions=True,
        include_raw_judge=True,
    )
    path = tmp_path / "annotate.json"
    cfg.save(path)
    loaded = AnnotateConfig.load(path)

    assert loaded.to_dict() == cfg.to_dict()
    assert loaded.challenger.name == "VLLM/openai/gpt-oss-20b"
    assert loaded.generation.gpu_memory_utilization == 0.85
    assert loaded.pairing.strategy == "all"
    assert loaded.pairing.source == "challenger_vs_dataset_inline"
