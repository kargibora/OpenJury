from __future__ import annotations

import json

from openjury.arena.config import AgreementConfig, ArenaConfig, JudgeConfig, ModelEntry
from openjury.cli import agreement as agreement_cli
from openjury.cli import arena as arena_cli
from openjury.cli import generate as generate_cli
from openjury.slurm.generate_slurm import PipelineConfig


def test_agreement_cli_slurm_forwarding_uses_agreement_mode(monkeypatch):
    captured: dict[str, object] = {}

    def fake_slurm_main(argv=None):
        argv_list = list(argv or [])
        captured["argv"] = argv_list
        cfg_path = argv_list[argv_list.index("--config") + 1]
        captured["config"] = json.loads(open(cfg_path, encoding="utf-8").read())

    monkeypatch.setattr("openjury.slurm.generate_slurm.main", fake_slurm_main)

    agreement_cli.main([
        "--dataset", "lmsys",
        "--judge_model", "OpenRouter/qwen/qwen3-32b",
        "--n_instructions", "25",
        "--judge_mode", "pairwise",
        "--pairwise_prompt_style", "legacy",
        "--language", "fr",
        "--seed", "7",
        "--slurm",
        "--slurm_output_dir", "slurm_scripts",
        "--stage", "analyze",
    ])

    argv = captured["argv"]
    assert argv[:2] == ["--mode", "agreement"]
    assert "--config" in argv
    assert "--stage" in argv
    assert argv[argv.index("--stage") + 1] == "analyze"
    assert "--dataset" not in argv
    cfg = captured["config"]
    assert cfg["dataset"] == "lmsys"
    assert cfg["judge"]["model"] == "OpenRouter/qwen/qwen3-32b"
    assert cfg["judge"]["mode"] == "pairwise"
    assert cfg["judge"]["pairwise_prompt_style"] == "legacy"
    assert cfg["language"] == "fr"
    assert cfg["seed"] == 7


def test_generate_cli_slurm_forwarding_includes_dataset_selection(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def fake_slurm_main(argv=None):
        argv_list = list(argv or [])
        captured["argv"] = argv_list
        cfg_path = argv_list[argv_list.index("--config") + 1]
        captured["config"] = json.loads(open(cfg_path, encoding="utf-8").read())

    monkeypatch.setattr("openjury.slurm.generate_slurm.main", fake_slurm_main)

    out = tmp_path / "gen.parquet"
    generate_cli.main([
        "--model", "VLLM/Qwen/Qwen2.5-0.5B-Instruct",
        "--dataset", "alpaca-eval",
        "--output", str(out),
        "--n_instructions", "10",
        "--language", "en",
        "--seed", "13",
        "--slurm",
    ])

    argv = captured["argv"]
    assert argv[:2] == ["--mode", "generate"]
    assert "--config" in argv
    assert "--dataset" not in argv
    cfg = captured["config"]
    assert cfg["dataset"] == "alpaca-eval"
    assert cfg["n_instructions"] == 10
    assert cfg["language"] == "en"
    assert cfg["seed"] == 13


def test_arena_cli_slurm_forwarding_uses_unified_entry(monkeypatch):
    captured: dict[str, object] = {}

    def fake_slurm_main(argv=None):
        argv_list = list(argv or [])
        captured["argv"] = argv_list
        cfg_path = argv_list[argv_list.index("--config") + 1]
        captured["config"] = json.loads(open(cfg_path, encoding="utf-8").read())

    monkeypatch.setattr("openjury.slurm.generate_slurm.main", fake_slurm_main)

    arena_cli.main([
        "--models", "Dummy/A", "Dummy/B",
        "--judge_model", "OpenRouter/qwen/qwen3-32b",
        "--dataset", "alpaca-eval",
        "--judge_mode", "pairwise",
        "--pairwise_prompt_style", "legacy",
        "--bt_regularization", "0.2",
        "--elo_k", "64",
        "--include_completions",
        "--include_raw_judge",
        "--gen-kwargs", "top_k=33",
        "--stage", "annotate",
        "--slurm",
    ])

    argv = captured["argv"]
    assert argv[:2] == ["--mode", "arena"]
    assert "--config" in argv
    assert "--stage" in argv
    assert argv[argv.index("--stage") + 1] == "annotate"
    assert "--models" not in argv
    cfg = captured["config"]
    assert cfg["dataset"] == "alpaca-eval"
    assert cfg["judge"]["model"] == "OpenRouter/qwen/qwen3-32b"
    assert cfg["judge"]["mode"] == "pairwise"
    assert cfg["judge"]["pairwise_prompt_style"] == "legacy"
    assert cfg["ratings"]["bt_regularization"] == 0.2
    assert cfg["ratings"]["elo_k"] == 64.0
    assert cfg["include_completions"] is True
    assert cfg["include_raw_judge"] is True
    assert cfg["judge"]["generation_kwargs"] == {"top_k": 33}


def test_dataset_options_properties_are_available_on_task_and_slurm_configs():
    arena_cfg = ArenaConfig(
        dataset="alpaca-eval",
        models=[ModelEntry(name="Dummy/A"), ModelEntry(name="Dummy/B")],
        judge=JudgeConfig(model="Dummy/J"),
    )
    aopts = arena_cfg.dataset_options
    assert aopts.name == "alpaca-eval"
    assert aopts.n_instructions is None
    assert aopts.loader_kwargs() == {}

    agreement_cfg = AgreementConfig(
        dataset="lmsys",
        judge=JudgeConfig(model="Dummy/J"),
        n_instructions=50,
        language="fr",
        seed=99,
        balance_by="lang",
    )
    gopts = agreement_cfg.dataset_options
    assert gopts.name == "lmsys"
    assert gopts.n_instructions == 50
    assert gopts.loader_kwargs() == {"language": "fr", "seed": 99, "balance_by": "lang"}
    assert "balance=lang" in gopts.cache_key()

    slurm_cfg = PipelineConfig(
        dataset="comparia",
        judge_model="Dummy/J",
        n_instructions=10,
        language="de",
        seed=3,
        balance_by="lang",
    )
    sopts = slurm_cfg.dataset_options
    assert sopts.name == "comparia"
    assert sopts.n_instructions == 10
    assert sopts.loader_kwargs() == {"language": "de", "seed": 3, "balance_by": "lang"}
    assert "balance=lang" in sopts.cache_key()
