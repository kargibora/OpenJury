from __future__ import annotations

import json

from openjury.arena.config import AgreementConfig, ArenaConfig, JudgeConfig, ModelEntry
from openjury.cli import agreement as agreement_cli
from openjury.cli import annotate as annotate_cli
from openjury.cli import arena as arena_cli
from openjury.cli import generate as generate_cli
from openjury.slurm.plans import SlurmExecutionConfig, SlurmRunPlan


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
        "--slurm_container_runtime", "apptainer",
        "--slurm_container_image", "/containers/vllm.sif",
        "--slurm_container_home", "/container-home",
        "--stage", "analyze",
    ])

    argv = captured["argv"]
    assert argv[:2] == ["--mode", "agreement"]
    assert "--config" in argv
    assert "--stage" in argv
    assert argv[argv.index("--stage") + 1] == "analyze"
    assert argv[argv.index("--container_runtime") + 1] == "apptainer"
    assert argv[argv.index("--container_image") + 1] == "/containers/vllm.sif"
    assert argv[argv.index("--container_home") + 1] == "/container-home"
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
        "--slurm_container_runtime", "apptainer",
        "--slurm_container_image", "/containers/vllm.sif",
        "--slurm_container_home", "/container-home",
    ])

    argv = captured["argv"]
    assert argv[:2] == ["--mode", "generate"]
    assert "--config" in argv
    assert argv[argv.index("--container_runtime") + 1] == "apptainer"
    assert argv[argv.index("--container_image") + 1] == "/containers/vllm.sif"
    assert argv[argv.index("--container_home") + 1] == "/container-home"
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
        "--slurm_container_runtime", "apptainer",
        "--slurm_container_image", "/containers/vllm.sif",
        "--slurm_container_home", "/container-home",
    ])

    argv = captured["argv"]
    assert argv[:2] == ["--mode", "arena"]
    assert "--config" in argv
    assert "--stage" in argv
    assert argv[argv.index("--stage") + 1] == "annotate"
    assert argv[argv.index("--container_runtime") + 1] == "apptainer"
    assert argv[argv.index("--container_image") + 1] == "/containers/vllm.sif"
    assert argv[argv.index("--container_home") + 1] == "/container-home"
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


def test_annotate_cli_slurm_forwarding_uses_annotate_mode(monkeypatch):
    captured: dict[str, object] = {}

    def fake_slurm_main(argv=None):
        argv_list = list(argv or [])
        captured["argv"] = argv_list
        cfg_path = argv_list[argv_list.index("--config") + 1]
        captured["config"] = json.loads(open(cfg_path, encoding="utf-8").read())

    monkeypatch.setattr("openjury.slurm.generate_slurm.main", fake_slurm_main)

    annotate_cli.main([
        "--dataset", "lmsys",
        "--challenger_model", "VLLM/openai/gpt-oss-20b",
        "--challenger_gpus", "4",
        "--judge_model", "VLLM/Qwen/Qwen3-32B",
        "--judge_gpus", "4",
        "--judge_mode", "pairwise",
        "--pairing_source", "challenger_vs_dataset_inline",
        "--pairing_strategy", "all",
        "--n_instructions", "25",
        "--slurm",
        "--slurm_container_runtime", "apptainer",
        "--slurm_container_image", "/containers/vllm.sif",
        "--slurm_container_home", "/container-home",
    ])

    argv = captured["argv"]
    assert argv[:2] == ["--mode", "annotate"]
    assert "--config" in argv
    assert argv[argv.index("--container_runtime") + 1] == "apptainer"
    assert argv[argv.index("--container_image") + 1] == "/containers/vllm.sif"
    assert argv[argv.index("--container_home") + 1] == "/container-home"
    assert "--dataset" not in argv
    cfg = captured["config"]
    assert cfg["dataset"] == "lmsys"
    assert cfg["challenger"]["name"] == "VLLM/openai/gpt-oss-20b"
    assert cfg["judge"]["model"] == "VLLM/Qwen/Qwen3-32B"
    assert cfg["pairing"]["source"] == "challenger_vs_dataset_inline"
    assert cfg["pairing"]["strategy"] == "all"


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

    slurm_cfg = SlurmRunPlan(
        task=AgreementConfig(
            dataset="comparia",
            judge=JudgeConfig(model="Dummy/J"),
            n_instructions=10,
            language="de",
            seed=3,
            balance_by="lang",
        ),
        execution=SlurmExecutionConfig(),
    )
    sopts = slurm_cfg.dataset_options
    assert sopts.name == "comparia"
    assert sopts.n_instructions == 10
    assert sopts.loader_kwargs() == {"language": "de", "seed": 3, "balance_by": "lang"}
    assert "balance=lang" in sopts.cache_key()


def test_slurm_run_plan_stores_task_and_execution_separately():
    slurm_cfg = SlurmRunPlan(
        task=ArenaConfig(
            dataset="lmsys",
            models=[
                ModelEntry(name="Dummy/A", gpus=2, quantization="fp8"),
                ModelEntry(name="Dummy/B"),
            ],
            judge=JudgeConfig(
                model="Dummy/J",
                gpus=2,
                mode="pairwise",
                generation_kwargs={"top_k": 5},
            ),
        ),
        execution=SlurmExecutionConfig(tag="TAG"),
    )

    assert slurm_cfg.judge.model == "Dummy/J"
    assert slurm_cfg.judge.gpus == 2
    assert slurm_cfg.judge.mode == "pairwise"
    assert slurm_cfg.judge.generation_kwargs == {"top_k": 5}
    assert slurm_cfg.model_names == ["Dummy/A", "Dummy/B"]
    assert slurm_cfg.models[0].quantization == "fp8"
    assert slurm_cfg.execution.tag == "TAG"
