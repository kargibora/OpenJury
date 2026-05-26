from __future__ import annotations

from types import SimpleNamespace

from openjury.arena.config import AgreementConfig, JudgeConfig, ModelEntry
from openjury.slurm import render as slurm_render
from openjury.slurm.plans import GenerateTaskConfig, SlurmExecutionConfig, SlurmRunPlan


def _job(**overrides):
    base = dict(
        job_name="oj_test",
        n_gpus=1,
        partition="boost_usr_prod",
        account="OELLM_prod2026",
        n_nodes=1,
        cpus_per_task=4,
        memory="32G",
        time="00:30:00",
        qos="normal",
        extra_sbatch=[],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _execution(tmp_path, *, runtime: str = "apptainer") -> SlurmExecutionConfig:
    return SlurmExecutionConfig(
        mode="generate",
        stage="all",
        judge_local=True,
        tag="TAG",
        partition="boost_usr_prod",
        account="OELLM_prod2026",
        time_generate="00:30:00",
        time_judge="00:30:00",
        qos="normal",
        project_dir=str(tmp_path / "project"),
        work_dir=str(tmp_path / "work"),
        logs_dir=str(tmp_path / "logs"),
        container_runtime=runtime,
        container_image=str(tmp_path / "vllm.sif"),
        container_home=str(tmp_path / "container-home"),
    )


def test_vllm_generate_script_uses_apptainer_wrapper(tmp_path):
    execution = _execution(tmp_path)
    task = GenerateTaskConfig(
        dataset="alpaca-eval",
        models=[ModelEntry(name="VLLM/Qwen/Qwen3-8B")],
    )

    script = slurm_render._generate_step_script(
        "VLLM/Qwen/Qwen3-8B",
        str(tmp_path / "work" / "completions.parquet"),
        task,
        execution,
        _job(n_gpus=2),
        local=True,
    )

    assert 'OPENJURY_CONTAINER_IMAGE="${OPENJURY_CONTAINER_IMAGE:-' in script
    assert 'OPENJURY_CONTAINER_HOME="${OPENJURY_CONTAINER_HOME:-' in script
    assert "_openjury_container_exec_script" in script
    assert "command -v apptainer" in script
    assert "command -v singularity" in script
    assert "exec --nv" in script
    assert "--home \"$OPENJURY_CONTAINER_HOME:$OPENJURY_CONTAINER_HOME\"" in script
    assert "openjury_container_cmd.XXXXXX.sh" in script
    assert str(tmp_path / "work") in script
    assert "Run 'openjury-container-prepare' first" in script
    assert 'openjury-generate \\' in script
    assert 'OPENJURY_VENV="${OPENJURY_VENV:-$PWD/.venv}"' not in script


def test_non_vllm_generate_script_keeps_host_venv_even_with_container_defaults(tmp_path):
    execution = _execution(tmp_path)
    task = GenerateTaskConfig(
        dataset="alpaca-eval",
        models=[ModelEntry(name="SGLang/Qwen/Qwen3-8B")],
    )

    script = slurm_render._generate_step_script(
        "SGLang/Qwen/Qwen3-8B",
        str(tmp_path / "work" / "completions.parquet"),
        task,
        execution,
        _job(n_gpus=2),
        local=True,
    )

    assert "_openjury_container_exec_script" not in script
    assert "exec --nv" not in script
    assert 'OPENJURY_VENV="${OPENJURY_VENV:-$PWD/.venv}"' in script


def test_vllm_agreement_script_uses_apptainer_wrapper_and_checks(tmp_path):
    plan = SlurmRunPlan(
        task=AgreementConfig(
            dataset="lmsys",
            judge=JudgeConfig(model="VLLM/Qwen/Qwen3-8B", gpus=2),
            n_instructions=8,
        ),
        execution=_execution(tmp_path),
    )

    script = slurm_render._agreement_step_script(
        plan,
        _job(n_gpus=2),
        output_dir=str(tmp_path / "results"),
        config_path=str(tmp_path / "agreement_config.json"),
        stage="annotate",
        local=True,
    )

    assert "_openjury_container_exec_script" in script
    assert "Container runtime is enabled but OPENJURY_CONTAINER_IMAGE is not set." in script
    assert "Container runtime is enabled but OPENJURY_CONTAINER_HOME is not set." in script
    assert "REQUESTS_CA_BUNDLE" in script
    assert "CURL_CA_BUNDLE" in script
    assert "unset SSL_CERT_DIR" in script
    assert 'openjury-evaluate agreement --config "' in script
