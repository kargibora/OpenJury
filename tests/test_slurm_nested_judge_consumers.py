from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from openjury.arena.config import AgreementConfig, ArenaConfig, JudgeConfig, MatchmakerConfig, ModelEntry
from openjury.slurm import modes as slurm_modes
from openjury.slurm import render as slurm_render
from openjury.slurm import submit_builders as slurm_submit_builders
from openjury.slurm.plans import SlurmExecutionConfig, SlurmRunPlan


def _job(**overrides):
    base = dict(
        job_name="oj_test",
        n_gpus=3,
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


def _arena_plan(tmp_path: Path, *, judge_local: bool = True) -> SlurmRunPlan:
    return SlurmRunPlan(
        task=ArenaConfig(
            dataset="lmsys",
            models=[ModelEntry(name="Dummy/A"), ModelEntry(name="Dummy/B")],
            judge=JudgeConfig(
                model="OpenRouter/qwen/judge",
                gpus=3,
                mode="pairwise",
            ),
            criteria="default",
            matchmaker=MatchmakerConfig(strategy="random"),
            n_instructions=8,
        ),
        execution=SlurmExecutionConfig(
            judge_local=judge_local,
            project_dir=str(tmp_path / "project"),
            work_dir=str(tmp_path / "work"),
            logs_dir=str(tmp_path / "logs"),
            partition="boost_usr_prod",
            account="OELLM_prod2026",
            time_judge="00:30:00",
            time_generate="00:30:00",
            qos="normal",
            tag="TAG",
            mode="arena",
        ),
    )


def _agreement_plan(tmp_path: Path, *, judge_local: bool = True) -> SlurmRunPlan:
    return SlurmRunPlan(
        task=AgreementConfig(
            dataset="lmsys",
            judge=JudgeConfig(
                model="OpenRouter/qwen/judge",
                gpus=3,
                mode="pairwise",
            ),
            criteria="default",
            n_instructions=8,
        ),
        execution=SlurmExecutionConfig(
            judge_local=judge_local,
            project_dir=str(tmp_path / "project"),
            work_dir=str(tmp_path / "work"),
            logs_dir=str(tmp_path / "logs"),
            partition="boost_usr_prod",
            account="OELLM_prod2026",
            time_judge="00:30:00",
            time_generate="00:30:00",
            qos="normal",
            tag="TAG",
            mode="agreement",
        ),
    )


def test_slurm_render_scripts_use_nested_judge_fields(tmp_path):
    arena_plan = _arena_plan(tmp_path, judge_local=False)
    arena_script = slurm_render._arena_step_script(
        arena_plan,
        _job(),
        output_dir=str(tmp_path / "results"),
        config_path=str(tmp_path / "arena_config.json"),
        stage="annotate",
        local=False,
    )
    agreement_plan = _agreement_plan(tmp_path, judge_local=False)
    agreement_script = slurm_render._agreement_step_script(
        agreement_plan,
        _job(),
        output_dir=str(tmp_path / "results"),
        config_path=str(tmp_path / "agreement_config.json"),
        stage="annotate",
        local=False,
    )

    assert "Judge:      OpenRouter/qwen/judge" in arena_script
    assert "Mode:       pairwise" in arena_script
    assert "Judge:             OpenRouter/qwen/judge" in agreement_script
    assert "Mode:              pairwise" in agreement_script


def test_slurm_submit_builders_use_nested_judge_fields(tmp_path):
    arena_plan = _arena_plan(tmp_path, judge_local=True)
    agreement_plan = _agreement_plan(tmp_path, judge_local=True)

    judge_script = slurm_submit_builders.build_judge_submit_all(
        arena_plan,
        path_judge=tmp_path / "01_arena_judge.sh",
        results_dir=str(tmp_path / "results"),
        tag_suffix="_TAG",
    )
    agreement_script = slurm_submit_builders.build_agreement_submit_all(
        agreement_plan,
        path_agreement=tmp_path / "01_agreement.sh",
        results_dir=str(tmp_path / "results"),
        tag_suffix="_TAG",
        stage="annotate",
    )

    assert "OpenRouter/qwen/judge (compute node, 3 GPU(s))" in judge_script
    assert "Mode:     pairwise" in judge_script
    assert "OpenRouter/qwen/judge (compute node, 3 GPU(s))" in agreement_script
    assert "Mode:     pairwise" in agreement_script


def test_slurm_agreement_mode_scripts_work_with_nested_judge_only(tmp_path):
    plan = _agreement_plan(tmp_path, judge_local=True)

    captured_jobs: list[dict[str, object]] = []
    written: dict[str, str] = {}

    def job_factory(**kwargs):
        captured_jobs.append(kwargs)
        return _job(**kwargs)

    def write_script(path: Path, content: str) -> Path:
        path.write_text(content, encoding="utf-8")
        written[path.name] = content
        return path

    slurm_modes.agreement_mode_scripts(
        plan=plan,
        output_dir=tmp_path,
        tag_suffix="_TAG",
        write_script=write_script,
        job_factory=job_factory,
        generated_files=[],
    )

    assert captured_jobs[0]["n_gpus"] == 3
    assert "OpenRouter/qwen/judge" in written["01_agreement.sh"]
    assert "pairwise" in written["01_agreement.sh"]

    config = json.loads((tmp_path / "agreement_config.json").read_text(encoding="utf-8"))
    assert config["judge"]["model"] == "OpenRouter/qwen/judge"
