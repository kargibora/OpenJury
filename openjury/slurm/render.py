"""Shell script renderers for OpenJury SLURM/login-node execution."""

from __future__ import annotations

import os
from textwrap import dedent
from typing import Any

from openjury.models.utils import api_key_env_for_model


def _build_env_preamble(pipeline: Any, *, is_compute: bool) -> str:
    """Build an env-setup block for generated scripts."""
    lines: list[str] = []
    env_file = os.path.join(pipeline.project_dir, ".env")
    lines.append("# Load .env if present (API keys, cluster settings)")
    lines.append(f'if [ -f "{env_file}" ]; then')
    lines.append(f'    set -a; source "{env_file}"; set +a')
    lines.append("fi")
    lines.append("")

    if is_compute:
        lines.append("# Compute node — enable HF offline mode")
        lines.append('if [ -z "${HF_HUB_OFFLINE:-}" ]; then')
        lines.append("    export HF_HUB_OFFLINE=1")
        lines.append("    export TRANSFORMERS_OFFLINE=1")
        lines.append("    export HF_DATASETS_OFFLINE=1")
        lines.append("fi")
        lines.append("export VLLM_NO_USAGE_STATS=1")
        lines.append("export UV_NO_SYNC=1")
    else:
        lines.append("# Login node — ensure offline flags are NOT set")
        lines.append("unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE 2>/dev/null || true")

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


def _render_header(job: Any, pipeline: Any) -> str:
    extra = "\n".join(f"#SBATCH {line}" for line in job.extra_sbatch)
    gres_line = f"#SBATCH --gres=gpu:{job.n_gpus}\n" if job.n_gpus > 0 else ""
    return SBATCH_HEADER.format(
        job_name=job.job_name,
        logs_dir=pipeline.logs_dir,
        partition=job.partition or pipeline.partition,
        account=job.account or pipeline.account,
        n_nodes=job.n_nodes,
        gres_line=gres_line,
        cpus_per_task=job.cpus_per_task,
        memory=job.memory,
        time=job.time or pipeline.time_generate,
        qos=job.qos,
        extra_sbatch=extra,
        env_preamble=_build_env_preamble(pipeline, is_compute=True),
        project_dir=pipeline.project_dir,
        work_dir=pipeline.work_dir,
    )


def _render_local_header(job_name: str, pipeline: Any, model: str) -> str:
    key_env = api_key_env_for_model(model)
    if key_env:
        api_key_block = (
            f"\n# Ensure the API key is available\n"
            f"if [ -z \"${{{key_env}:-}}\" ]; then\n"
            f"    echo \"ERROR: ${key_env} is not set. Export it or add it to .env\"\n"
            f"    exit 1\n"
            f"fi\n"
        )
    else:
        api_key_block = ""

    return LOCAL_HEADER.format(
        job_name=job_name,
        env_preamble=_build_env_preamble(pipeline, is_compute=False),
        api_key_block=api_key_block,
        project_dir=pipeline.project_dir,
        work_dir=pipeline.work_dir,
        logs_dir=pipeline.logs_dir,
    )


def _generate_step_script(
    model: str,
    output_parquet: str,
    pipeline: Any,
    job: Any,
    quantization: str | None = None,
    *,
    local: bool = True,
) -> str:
    """Render a single-model generation script."""
    header = _render_header(job, pipeline) if local else _render_local_header(job.job_name, pipeline, model)

    parts: list[str] = [
        "uv run openjury-generate \\",
        f'    --model "{model}" \\',
        f'    --dataset "{pipeline.dataset}" \\',
        f'    --output "{output_parquet}" \\',
        f"    --tensor_parallel_size {job.n_gpus} \\",
        f"    --max_tokens {pipeline.generation_max_tokens} \\",
        f"    --truncate_input_chars {pipeline.truncate_input_chars} \\",
    ]
    if pipeline.n_instructions:
        parts.append(f"    --n_instructions {pipeline.n_instructions} \\")
    if getattr(pipeline, "language", None):
        parts.append(f'    --language "{pipeline.language}" \\')
    if getattr(pipeline, "seed", 42) != 42:
        parts.append(f"    --seed {pipeline.seed} \\")
    if getattr(pipeline, "balance_by", None):
        parts.append(f'    --balance_by "{pipeline.balance_by}" \\')
    if pipeline.ignore_cache:
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
    pipeline: Any,
    job: Any,
    output_dir: str,
    config_path: str,
    stage: str = "all",
    *,
    local: bool = True,
) -> str:
    """Render an arena evaluation script."""
    header = _render_header(job, pipeline) if local else _render_local_header(job.job_name, pipeline, pipeline.judge_model)
    cmd = f'uv run openjury-evaluate arena --config "{config_path}" --stage "{stage}"'
    model_list_str = "\n".join(f'echo "  [{i+1}] {m}"' for i, m in enumerate(pipeline.models))

    body = dedent(f"""\
        # ── Arena Judge (K-model evaluation) ──────────────────────
        echo "Arena evaluation: {len(pipeline.models)} models"
        {model_list_str}
        echo "Judge:      {pipeline.judge_model} (TP={job.n_gpus})"
        echo "Rubric:     {pipeline.rubric}"
        echo "Mode:       {pipeline.judge_mode}"
        echo "Stage:      {stage}"
        echo "Matchmaker: {pipeline.matchmaker}"
        echo "Config:     {config_path}"

        {cmd}

        echo ""
        echo "✅ Arena step completed: $(date)"
        echo "   Results: {output_dir}/"
    """)
    return header + body


def _agreement_step_script(
    pipeline: Any,
    job: Any,
    output_dir: str,
    config_path: str,
    stage: str = "all",
    *,
    local: bool = True,
) -> str:
    """Render an agreement evaluation script."""
    header = _render_header(job, pipeline) if local else _render_local_header(job.job_name, pipeline, pipeline.judge_model)
    cmd = f'uv run openjury-evaluate agreement --config "{config_path}" --stage "{stage}"'

    body = dedent(f"""\
        # ── Agreement Evaluation (human vs judge) ────────────────
        echo "Agreement dataset: {pipeline.dataset}"
        echo "Judge:             {pipeline.judge_model} (TP={job.n_gpus})"
        echo "Rubric:            {pipeline.rubric}"
        echo "Mode:              {pipeline.judge_mode}"
        echo "Stage:             {stage}"
        echo "Language filter:   {pipeline.language or 'all'}"
        echo "Balanced by:       {pipeline.balance_by or 'none'}"
        echo "Config:            {config_path}"

        {cmd}

        echo ""
        echo "✅ Agreement step completed: $(date)"
        echo "   Results: {output_dir}/"
    """)
    return header + body
