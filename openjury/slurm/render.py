"""Shell script renderers for OpenJury SLURM/login-node execution."""

from __future__ import annotations

import os
from textwrap import dedent
from typing import Any

from openjury.annotate_config import AnnotateConfig
from openjury.arena.config import AgreementConfig, ArenaConfig
from openjury.models.utils import api_key_env_for_model
from openjury.slurm.plans import GenerateTaskConfig, SlurmExecutionConfig, SlurmRunPlan


def _build_env_preamble(execution: SlurmExecutionConfig, *, is_compute: bool) -> str:
    """Build an env-setup block for generated scripts."""
    lines: list[str] = []
    env_file = os.path.join(execution.project_dir, ".env")
    lines.append("# Load .env if present (API keys, cluster settings)")
    lines.append(f'if [ -f "{env_file}" ]; then')
    lines.append(f'    set -a; source "{env_file}"; set +a')
    lines.append("fi")
    lines.append("")

    if is_compute:
        lines.append("# Activate venv (OPENJURY_VENV from .env, fallback to .venv)")
        lines.append('OPENJURY_VENV="${OPENJURY_VENV:-$PWD/.venv}"')
        lines.append('if [ -f "$OPENJURY_VENV/bin/activate" ]; then')
        lines.append('    source "$OPENJURY_VENV/bin/activate"')
        lines.append('else')
        lines.append('    echo "WARNING: venv not found at $OPENJURY_VENV"')
        lines.append('fi')
        lines.append("")
        lines.append("# Compute node — enable HF offline mode")
        lines.append('if [ -z "${HF_HUB_OFFLINE:-}" ]; then')
        lines.append("    export HF_HUB_OFFLINE=1")
        lines.append("    export TRANSFORMERS_OFFLINE=1")
        lines.append("    export HF_DATASETS_OFFLINE=1")
        lines.append("fi")
        lines.append("export VLLM_NO_USAGE_STATS=1")
    else:
        lines.append("# Activate venv (OPENJURY_VENV from .env, fallback to .venv)")
        lines.append('OPENJURY_VENV="${OPENJURY_VENV:-$PWD/.venv}"')
        lines.append('if [ -f "$OPENJURY_VENV/bin/activate" ]; then')
        lines.append('    source "$OPENJURY_VENV/bin/activate"')
        lines.append('else')
        lines.append('    echo "WARNING: venv not found at $OPENJURY_VENV"')
        lines.append('fi')
        lines.append("")
        lines.append("# Login node — ensure offline flags are NOT set")
        lines.append(
            "unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE 2>/dev/null || true"
        )

    return "\n".join(lines)


SBATCH_HEADER = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={logs_dir}/{job_name}_%j.out
#SBATCH --error={logs_dir}/{job_name}_%j.err
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --nodes={n_nodes}
{gres_line}#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={memory}
#SBATCH --time={time}
#SBATCH --qos={qos}
{extra_sbatch}

# ── Environment Setup ─────────────────────────────────────────
set -euo pipefail

echo "═══════════════════════════════════════════════════════"
echo "  Job:  {job_name}"
echo "  Date: $(date)"
echo "  Node: $(hostname)"
echo "  GPUs: ${{CUDA_VISIBLE_DEVICES:-not set}}"
echo "═══════════════════════════════════════════════════════"

{env_preamble}

cd {project_dir}

# Ensure work directories exist
mkdir -p {work_dir} {logs_dir}
"""


LOCAL_HEADER = """\
#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  {job_name}  (runs on LOGIN NODE — requires network access)
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

echo "═══════════════════════════════════════════════════════"
echo "  Job:  {job_name} (login node)"
echo "  Date: $(date)"
echo "  Node: $(hostname)"
echo "═══════════════════════════════════════════════════════"

{env_preamble}
{api_key_block}
cd {project_dir}

# Ensure work directories exist
mkdir -p {work_dir} {logs_dir}
"""


def _render_header(job: Any, execution: SlurmExecutionConfig) -> str:
    extra = "\n".join(f"#SBATCH {line}" for line in job.extra_sbatch)
    gres_line = f"#SBATCH --gres=gpu:{job.n_gpus}\n" if job.n_gpus > 0 else ""
    return SBATCH_HEADER.format(
        job_name=job.job_name,
        logs_dir=execution.logs_dir,
        partition=job.partition or execution.partition,
        account=job.account or execution.account,
        n_nodes=job.n_nodes,
        gres_line=gres_line,
        cpus_per_task=job.cpus_per_task,
        memory=job.memory,
        time=job.time or execution.time_generate,
        qos=job.qos,
        extra_sbatch=extra,
        env_preamble=_build_env_preamble(execution, is_compute=True),
        project_dir=execution.project_dir,
        work_dir=execution.work_dir,
    )


def _render_local_header(
    job_name: str,
    execution: SlurmExecutionConfig,
    model: str,
) -> str:
    return _render_local_header_for_models(job_name, execution, [model])


def _render_local_header_for_models(
    job_name: str,
    execution: SlurmExecutionConfig,
    models: list[str],
) -> str:
    key_envs: list[str] = []
    for model in models:
        key_env = api_key_env_for_model(model)
        if key_env and key_env not in key_envs:
            key_envs.append(key_env)

    if key_envs:
        checks = []
        for key_env in key_envs:
            checks.append(
                f'if [ -z "${{{key_env}:-}}" ]; then\n'
                f'    echo "ERROR: ${key_env} is not set. Export it or add it to .env"\n'
                f"    exit 1\n"
                f"fi"
            )
        api_key_block = "\n# Ensure required API keys are available\n" + "\n".join(checks) + "\n"
    else:
        api_key_block = ""

    return LOCAL_HEADER.format(
        job_name=job_name,
        env_preamble=_build_env_preamble(execution, is_compute=False),
        api_key_block=api_key_block,
        project_dir=execution.project_dir,
        work_dir=execution.work_dir,
        logs_dir=execution.logs_dir,
    )


def _model_name(model: Any) -> str:
    """Return a model name from either a string or a structured entry."""
    return getattr(model, "name", model)


def _generate_step_script(
    model: str,
    output_parquet: str,
    task: ArenaConfig | GenerateTaskConfig,
    execution: SlurmExecutionConfig,
    job: Any,
    quantization: str | None = None,
    *,
    local: bool = True,
) -> str:
    """Render a single-model generation script."""
    header = (
        _render_header(job, execution)
        if local
        else _render_local_header(job.job_name, execution, model)
    )

    parts: list[str] = [
        "openjury-generate \\",
        f'    --model "{model}" \\',
        f'    --dataset "{task.dataset}" \\',
        f'    --output "{output_parquet}" \\',
        f"    --tensor_parallel_size {job.n_gpus} \\",
        f"    --max_tokens {task.generation_max_tokens} \\",
        f"    --truncate_input_chars {task.truncate_input_chars} \\",
    ]
    if task.n_instructions:
        parts.append(f"    --n_instructions {task.n_instructions} \\")
    if getattr(task, "language", None):
        parts.append(f'    --language "{task.language}" \\')
    if getattr(task, "seed", 42) != 42:
        parts.append(f"    --seed {task.seed} \\")
    if getattr(task, "balance_by", None):
        parts.append(f'    --balance_by "{task.balance_by}" \\')
    if getattr(task, "ignore_cache", False):
        parts.append("    --ignore_cache \\")
    if quantization:
        parts.append(f"    --quantization {quantization}")
    else:
        parts[-1] = parts[-1].rstrip(" \\")
    cmd = "\n".join(parts)

    body = dedent(f"""\
        # ── Generate Completions ──────────────────────────────────
        echo "Model:     {model}"
        echo "TP size:   {job.n_gpus}"
        echo "Output:    {output_parquet}"

        {cmd}

        echo ""
        echo "✅ Generate step completed: $(date)"
        echo "   Output: {output_parquet} ($(du -sh {output_parquet} | cut -f1))"
    """)
    return header + body


def _arena_step_script(
    plan: SlurmRunPlan,
    job: Any,
    output_dir: str,
    config_path: str,
    stage: str = "all",
    *,
    local: bool = True,
) -> str:
    """Render an arena evaluation script."""
    task = plan.task
    if not isinstance(task, ArenaConfig):
        raise TypeError("Arena step rendering requires an ArenaConfig task.")

    header = (
        _render_header(job, plan.execution)
        if local
        else _render_local_header(job.job_name, plan.execution, task.judge.model)
    )
    cmd = f'openjury-evaluate arena --config "{config_path}" --stage "{stage}"'
    model_list_str = "\n".join(
        f'echo "  [{i+1}] {_model_name(model)}"'
        for i, model in enumerate(task.models)
    )

    body = dedent(f"""\
        # ── Arena Judge (K-model evaluation) ──────────────────────
        echo "Arena evaluation: {len(task.models)} models"
        {model_list_str}
        echo "Judge:      {task.judge.model} (TP={job.n_gpus})"
        echo "Criteria:     {task.criteria}"
        echo "Mode:       {task.judge.mode}"
        echo "Stage:      {stage}"
        echo "Matchmaker: {task.matchmaker.strategy}"
        echo "Config:     {config_path}"

        {cmd}

        echo ""
        echo "✅ Arena step completed: $(date)"
        echo "   Results: {output_dir}/"
    """)
    return header + body


def _agreement_step_script(
    plan: SlurmRunPlan,
    job: Any,
    output_dir: str,
    config_path: str,
    stage: str = "all",
    *,
    local: bool = True,
) -> str:
    """Render an agreement evaluation script."""
    task = plan.task
    if not isinstance(task, AgreementConfig):
        raise TypeError("Agreement step rendering requires an AgreementConfig task.")

    header = (
        _render_header(job, plan.execution)
        if local
        else _render_local_header(job.job_name, plan.execution, task.judge.model)
    )
    cmd = f'openjury-evaluate agreement --config "{config_path}" --stage "{stage}"'

    body = dedent(f"""\
        # ── Agreement Evaluation (human vs judge) ────────────────
        echo "Agreement dataset: {task.dataset}"
        echo "Judge:             {task.judge.model} (TP={job.n_gpus})"
        echo "Criteria:            {task.criteria}"
        echo "Mode:              {task.judge.mode}"
        echo "Stage:             {stage}"
        echo "Language filter:   {task.language or 'all'}"
        echo "Balanced by:       {task.balance_by or 'none'}"
        echo "Config:            {config_path}"

        {cmd}

        echo ""
        echo "✅ Agreement step completed: $(date)"
        echo "   Results: {output_dir}/"
    """)
    return header + body


def _annotate_step_script(
    plan: SlurmRunPlan,
    job: Any,
    output_dir: str,
    config_path: str,
    *,
    local: bool = True,
) -> str:
    """Render an annotate evaluation script."""
    task = plan.task
    if not isinstance(task, AnnotateConfig):
        raise TypeError("Annotate step rendering requires an AnnotateConfig task.")

    if local:
        header = _render_header(job, plan.execution)
    else:
        api_models = [task.judge.model]
        if (
            task.challenger is not None
            and task.challenger.name
            and not task.challenger.completions
        ):
            api_models.append(task.challenger.name)
        header = _render_local_header_for_models(job.job_name, plan.execution, api_models)

    cmd = f'openjury-evaluate annotate --config "{config_path}"'
    challenger_line = (
        f'echo "Challenger:        {task.challenger.name}"\n'
        if task.challenger is not None and task.challenger.name
        else ""
    )

    body = dedent(f"""\
        # ── Annotate Evaluation ──────────────────────────────────
        echo "Dataset:           {task.dataset}"
        {challenger_line}echo "Judge:             {task.judge.model} (TP={job.n_gpus})"
        echo "Criteria:          {task.criteria}"
        echo "Mode:              {task.judge.mode}"
        echo "Pairing source:    {task.pairing.source}"
        echo "Pairing strategy:  {task.pairing.strategy}"
        echo "Config:            {config_path}"

        {cmd}

        echo ""
        echo "✅ Annotate step completed: $(date)"
        echo "   Results: {output_dir}/"
    """)
    return header + body
