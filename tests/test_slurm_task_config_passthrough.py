from __future__ import annotations

import json

from openjury.arena.config import AgreementConfig, ArenaConfig, JudgeConfig, ModelEntry
from openjury.slurm import generate_slurm


def test_slurm_arena_config_passthrough_preserves_rich_fields(tmp_path, monkeypatch):
    cfg = ArenaConfig(
        dataset="alpaca-eval",
        models=[
            ModelEntry(name="OpenRouter/qwen/model-a"),
            ModelEntry(name="OpenRouter/qwen/model-b"),
        ],
        judge=JudgeConfig(
            model="OpenRouter/qwen/judge",
            mode="pairwise",
            generation_kwargs={"top_k": 17},
            chat_template="{{ messages }}",
            provide_explanation=True,
            no_swap=True,
        ),
        output_dir="results/arena/local",
        bt_regularization=0.25,
        elo_k=77.0,
        include_completions=True,
        include_raw_judge=True,
        generation_max_tokens=1234,
        truncate_input_chars=5678,
    )
    cfg_path = tmp_path / "arena_input.json"
    cfg.save(cfg_path)

    monkeypatch.setenv("USER_WORK_DIR", str(tmp_path / "user_work"))
    monkeypatch.setattr("openjury.cache.completions.cache.exists", lambda **_k: True)
    monkeypatch.setattr("openjury.cache.scores.score_cache.exists", lambda **_k: True)

    out_root = tmp_path / "slurm_out"
    tag = "PASSTHRU"
    generate_slurm.main([
        "--mode", "arena",
        "--config", str(cfg_path),
        "--output_dir", str(out_root),
        "--tag", tag,
    ])

    run_dir = out_root / f"{cfg.dataset}_arena_{tag}"
    data = json.loads((run_dir / "arena_config.json").read_text(encoding="utf-8"))

    assert data["ratings"]["bt_regularization"] == 0.25
    assert data["ratings"]["elo_k"] == 77.0
    assert data["include_completions"] is True
    assert data["include_raw_judge"] is True
    assert data["judge"]["generation_kwargs"] == {"top_k": 17}
    assert data["judge"]["chat_template"] == "{{ messages }}"
    assert data["judge"]["no_swap"] is True
    assert data["judge"]["provide_explanation"] is True
    assert data["output_dir"] != "results/arena/local"
    assert data["output_dir"].endswith("/results")


def test_slurm_agreement_config_passthrough_preserves_rich_fields(tmp_path, monkeypatch):
    cfg = AgreementConfig(
        dataset="lmsys",
        judge=JudgeConfig(
            model="OpenRouter/qwen/judge",
            mode="pairwise",
            generation_kwargs={"top_p": 0.8},
            chat_template="{{ custom }}",
            provide_explanation=True,
        ),
        output_dir="results/agreement/local",
        n_instructions=25,
        language="fr",
        seed=9,
        ignore_score_cache=True,
    )
    cfg_path = tmp_path / "agreement_input.json"
    cfg.save(cfg_path)

    monkeypatch.setenv("USER_WORK_DIR", str(tmp_path / "user_work"))

    out_root = tmp_path / "slurm_out"
    tag = "PASSTHRU"
    generate_slurm.main([
        "--mode", "agreement",
        "--config", str(cfg_path),
        "--output_dir", str(out_root),
        "--tag", tag,
    ])

    run_dir = out_root / f"{cfg.dataset}_agreement_{tag}"
    data = json.loads((run_dir / "agreement_config.json").read_text(encoding="utf-8"))

    assert data["judge"]["generation_kwargs"] == {"top_p": 0.8}
    assert data["judge"]["chat_template"] == "{{ custom }}"
    assert data["judge"]["provide_explanation"] is True
    assert data["language"] == "fr"
    assert data["seed"] == 9
    assert data["generation"]["ignore_score_cache"] is True
    assert data["output_dir"] != "results/agreement/local"
    assert data["output_dir"].endswith("/results")


def test_slurm_agreement_cli_path_preserves_nested_judge_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_WORK_DIR", str(tmp_path / "user_work"))

    out_root = tmp_path / "slurm_out"
    tag = "CLIJUDGE"
    generate_slurm.main([
        "--mode", "agreement",
        "--dataset", "lmsys",
        "--judge_model", "OpenRouter/qwen/judge",
        "--judge_mode", "pairwise",
        "--pairwise_prompt_style", "legacy",
        "--n_instructions", "12",
        "--max_model_len", "16384",
        "--enforce_eager",
        "--gen-kwargs", "top_k=9", "top_p=0.8",
        "--output_dir", str(out_root),
        "--tag", tag,
    ])

    run_dir = out_root / f"lmsys_agreement_{tag}"
    data = json.loads((run_dir / "agreement_config.json").read_text(encoding="utf-8"))

    assert data["judge"]["model"] == "OpenRouter/qwen/judge"
    assert data["judge"]["mode"] == "pairwise"
    assert data["judge"]["pairwise_prompt_style"] == "legacy"
    assert data["judge"]["generation_kwargs"] == {"top_k": 9, "top_p": 0.8}
    assert data["judge"]["max_model_len"] == 16384
    assert data["judge"]["enforce_eager"] is True


def test_slurm_arena_stage_analyze_skips_generation_scripts(tmp_path, monkeypatch):
    cfg = ArenaConfig(
        dataset="alpaca-eval",
        models=[
            ModelEntry(name="OpenRouter/qwen/model-a"),
            ModelEntry(name="OpenRouter/qwen/model-b"),
        ],
        judge=JudgeConfig(model="OpenRouter/qwen/judge"),
        output_dir="results/arena/local",
    )
    cfg_path = tmp_path / "arena_input.json"
    cfg.save(cfg_path)

    monkeypatch.setenv("USER_WORK_DIR", str(tmp_path / "user_work"))

    out_root = tmp_path / "slurm_out"
    tag = "ANALYZE"
    generate_slurm.main([
        "--mode", "arena",
        "--stage", "analyze",
        "--config", str(cfg_path),
        "--output_dir", str(out_root),
        "--tag", tag,
    ])

    run_dir = out_root / f"{cfg.dataset}_arena_{tag}"
    scripts = sorted(run_dir.glob("*.sh"))
    script_names = [p.name for p in scripts]

    assert "01_arena_analyze.sh" in script_names
    assert not any("_generate_" in name for name in script_names)

    analyze_script = (run_dir / "01_arena_analyze.sh").read_text(encoding="utf-8")
    assert '--stage "analyze"' in analyze_script


def test_slurm_project_dir_env_is_normalized_to_nested_repo_root(tmp_path, monkeypatch):
    outer = tmp_path / "outer"
    inner = outer / "OpenJury"
    inner.mkdir(parents=True)
    (inner / "pyproject.toml").write_text("[project]\nname='openjury'\n", encoding="utf-8")
    (inner / "openjury").mkdir()

    cfg = AgreementConfig(
        dataset="lmsys",
        judge=JudgeConfig(model="OpenRouter/qwen/judge"),
        n_instructions=5,
        output_dir="results/agreement/local",
    )
    cfg_path = tmp_path / "agreement_input.json"
    cfg.save(cfg_path)

    monkeypatch.setenv("USER_WORK_DIR", str(tmp_path / "user_work"))
    monkeypatch.setenv("OPENJURY_PROJECT_DIR", str(outer))

    out_root = tmp_path / "slurm_out"
    tag = "PROJECTDIR"
    generate_slurm.main([
        "--mode", "agreement",
        "--config", str(cfg_path),
        "--output_dir", str(out_root),
        "--tag", tag,
    ])

    run_dir = out_root / f"{cfg.dataset}_agreement_{tag}"
    script = (run_dir / "01_agreement.sh").read_text(encoding="utf-8")
    assert f"cd {inner}" in script
