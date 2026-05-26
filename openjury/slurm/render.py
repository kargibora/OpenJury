"""Shell script renderers for OpenJury SLURM/login-node execution."""

from __future__ import annotations

import os
from textwrap import dedent
from typing import Any

from openjury.annotate_config import AnnotateConfig
from openjury.arena.config import AgreementConfig, ArenaConfig
from openjury.container_support import is_vllm_model
from openjury.models.utils import api_key_env_for_model
from openjury.slurm.plans import GenerateTaskConfig, SlurmExecutionConfig, SlurmRunPlan


def _build_env_preamble(
    execution: SlurmExecutionConfig,
    *,
    is_compute: bool,
    use_container: bool = False,
) -> str:
    """Build an env-setup block for generated scripts."""
    lines: list[str] = []
    env_file = os.path.join(execution.project_dir, ".env")
    lines.append("# Load .env if present (API keys, cluster settings)")
    lines.append(f'if [ -f "{env_file}" ]; then')
    lines.append(f'    set -a; source "{env_file}"; set +a')
    lines.append("fi")
    lines.append("")

    if is_compute and not use_container:
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
    elif is_compute:
        lines.append("# Compute node — containerized VLLM runtime")
        lines.append("# Host venv activation is skipped; the prepared container home is used instead.")
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


def _build_container_helper_block(execution: SlurmExecutionConfig) -> str:
    """Build shell helpers for Apptainer-family VLLM execution."""
    return dedent(f"""\
        # Containerized VLLM runtime helpers
        OPENJURY_CONTAINER_IMAGE="${{OPENJURY_CONTAINER_IMAGE:-{execution.container_image}}}"
        OPENJURY_CONTAINER_HOME="${{OPENJURY_CONTAINER_HOME:-{execution.container_home}}}"

        _openjury_container_bin() {{
            if command -v apptainer >/dev/null 2>&1; then
                echo apptainer
                return 0
            fi
            if command -v singularity >/dev/null 2>&1; then
                echo singularity
                return 0
            fi
            echo "ERROR: OPENJURY container runtime is enabled but neither 'apptainer' nor 'singularity' is available." >&2
            return 1
        }}

        _openjury_container_exec_script() {{
            local script_path=$1
            if [ -z "${{OPENJURY_CONTAINER_IMAGE:-}}" ]; then
                echo "ERROR: Container runtime is enabled but OPENJURY_CONTAINER_IMAGE is not set." >&2
                exit 1
            fi
            if [ ! -f "$OPENJURY_CONTAINER_IMAGE" ]; then
                echo "ERROR: Container image not found: $OPENJURY_CONTAINER_IMAGE" >&2
                exit 1
            fi
            if [ -z "${{OPENJURY_CONTAINER_HOME:-}}" ]; then
                echo "ERROR: Container runtime is enabled but OPENJURY_CONTAINER_HOME is not set." >&2
                exit 1
            fi
            mkdir -p "$OPENJURY_CONTAINER_HOME"
            if [ ! -x "$OPENJURY_CONTAINER_HOME/.local/bin/openjury-evaluate" ]; then
                echo "ERROR: OpenJury is not installed in container home $OPENJURY_CONTAINER_HOME. Run 'openjury-container-prepare' first." >&2
                exit 1
            fi

            local container_bin
            container_bin="$(_openjury_container_bin)"
            local -a bind_args
            bind_args=(
                -B "{execution.project_dir}:{execution.project_dir}"
                -B "{execution.work_dir}:{execution.work_dir}"
                -B "{execution.logs_dir}:{execution.logs_dir}"
                -B "$OPENJURY_CONTAINER_HOME:$OPENJURY_CONTAINER_HOME"
            )
            if [ -n "${{HF_HOME:-}}" ]; then
                mkdir -p "$HF_HOME"
                bind_args+=(-B "$HF_HOME:$HF_HOME")
            fi
            if [ -n "${{OPENJURY_DATA:-}}" ]; then
                mkdir -p "$OPENJURY_DATA"
                bind_args+=(-B "$OPENJURY_DATA:$OPENJURY_DATA")
            fi

            for cert_var in SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE; do
                cert_path="${{!cert_var:-}}"
                if [ -n "$cert_path" ] && [ ! -f "$cert_path" ]; then
                    unset "$cert_var"
                fi
            done
            if [ -n "${{SSL_CERT_DIR:-}}" ] && [ ! -d "$SSL_CERT_DIR" ]; then
                unset SSL_CERT_DIR
            fi

            HOME="$OPENJURY_CONTAINER_HOME" \\
            HF_HOME="${{HF_HOME:-}}" \\
            HF_TOKEN="${{HF_TOKEN:-}}" \\
            OPENJURY_DATA="${{OPENJURY_DATA:-}}" \\
            HF_HUB_OFFLINE="${{HF_HUB_OFFLINE:-1}}" \\
            TRANSFORMERS_OFFLINE="${{TRANSFORMERS_OFFLINE:-1}}" \\
            HF_DATASETS_OFFLINE="${{HF_DATASETS_OFFLINE:-1}}" \\
            VLLM_NO_USAGE_STATS="${{VLLM_NO_USAGE_STATS:-1}}" \\
            "$container_bin" exec --nv --home "$OPENJURY_CONTAINER_HOME:$OPENJURY_CONTAINER_HOME" "${{bind_args[@]}}" "$OPENJURY_CONTAINER_IMAGE" \\
                bash -lc "export PATH=\\"\\$HOME/.local/bin:\\$PATH\\"; cd \\"{execution.project_dir}\\"; bash \\"$script_path\\""
        }}
    """)


def _should_use_vllm_container(
    execution: SlurmExecutionConfig,
    model: str | None,
    *,
    is_compute: bool,
) -> bool:
    """Return True when a compute-node VLLM job should run in Apptainer."""
    return bool(
        is_compute
        and execution.container_runtime == "apptainer"
        and is_vllm_model(model)
    )


def _wrap_compute_command(
    command: str,
    *,
    use_container: bool,
    script_dir: str | None = None,
) -> str:
    """Wrap a compute command in a temp script executed inside the container."""
    if not use_container:
        return command
    if not script_dir:
        raise ValueError("script_dir is required when containerized execution is enabled.")
    return (
        f'OPENJURY_CONTAINER_SCRIPT=$(mktemp "{script_dir}/openjury_container_cmd.XXXXXX.sh")\n'
        'trap \'rm -f "$OPENJURY_CONTAINER_SCRIPT"\' EXIT\n'
        'cat > "$OPENJURY_CONTAINER_SCRIPT" <<\'__OPENJURY_CONTAINER_CMD__\'\n'
        'set -euo pipefail\n\n'
        f'{command.rstrip()}\n'
        '__OPENJURY_CONTAINER_CMD__\n'
        '_openjury_container_exec_script "$OPENJURY_CONTAINER_SCRIPT"\n'
        'trap - EXIT\n'
        'rm -f "$OPENJURY_CONTAINER_SCRIPT"'
    )


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
{container_helpers}

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


def _render_header(
    job: Any,
    execution: SlurmExecutionConfig,
    *,
    use_container: bool = False,
) -> str:
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
        env_preamble=_build_env_preamble(execution, is_compute=True, use_container=use_container),
        container_helpers=_build_container_helper_block(execution) if use_container else "",
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
    use_container = _should_use_vllm_container(execution, model, is_compute=local)
    header = (
        _render_header(job, execution, use_container=use_container)
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
    cmd = _wrap_compute_command(
        "\n".join(parts),
        use_container=use_container,
        script_dir=execution.work_dir,
    )

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

    use_container = _should_use_vllm_container(plan.execution, task.judge.model, is_compute=local)
    header = (
        _render_header(job, plan.execution, use_container=use_container)
        if local
        else _render_local_header(job.job_name, plan.execution, task.judge.model)
    )
    cmd = _wrap_compute_command(
        f'openjury-evaluate arena --config "{config_path}" --stage "{stage}"',
        use_container=use_container,
        script_dir=plan.execution.work_dir,
    )
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

    use_container = _should_use_vllm_container(plan.execution, task.judge.model, is_compute=local)
    header = (
        _render_header(job, plan.execution, use_container=use_container)
        if local
        else _render_local_header(job.job_name, plan.execution, task.judge.model)
    )
    cmd = _wrap_compute_command(
        f'openjury-evaluate agreement --config "{config_path}" --stage "{stage}"',
        use_container=use_container,
        script_dir=plan.execution.work_dir,
    )

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

    use_container = _should_use_vllm_container(plan.execution, task.judge.model, is_compute=local)
    if local:
        header = _render_header(job, plan.execution, use_container=use_container)
    else:
        api_models = [task.judge.model]
        if (
            task.challenger is not None
            and task.challenger.name
            and not task.challenger.completions
        ):
            api_models.append(task.challenger.name)
        header = _render_local_header_for_models(job.job_name, plan.execution, api_models)

    cmd = _wrap_compute_command(
        f'openjury-evaluate annotate --config "{config_path}"',
        use_container=use_container,
        script_dir=plan.execution.work_dir,
    )
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
