"""Generate SLURM sbatch scripts for K-model arena evaluation.

Supports **mixed-provider** pipelines: GPU models (VLLM, LlamaCpp) run on
compute nodes via ``sbatch``, while API models (OpenRouter, ChatOpenAI,
LiteLLM) run on the login node via plain ``bash``.  Execution targets are
auto-detected from the model provider prefix.

**Cluster settings come from environment variables or a ``.env`` file**
(see ``.env.example`` for the full list).  Variables that are required
for the current run but missing are reported together so you can fix
them all at once.

Key env vars (CLI flags override)::

    $ACCOUNT          – SLURM account          (required for sbatch jobs)
    $PARTITION        – SLURM partition         (required for sbatch jobs)
    $USER_WORK_DIR    – per-user work directory (required)
    $TIME_LIMIT       – default wall-time       (required for sbatch jobs)
    $GPUS_PER_NODE    – default GPU count per job

Usage::

    # Arena mode (default): K-model evaluation
    uv run python -m openjury.slurm.generate_slurm \\
        --dataset alpaca-eval \\
        --models VLLM/Qwen/Qwen2.5-0.5B-Instruct \\
                 VLLM/Qwen/Qwen2.5-1.5B-Instruct \\
                 VLLM/meta-llama/Llama-3.1-8B-Instruct \\
        --judge_model VLLM/Qwen/Qwen3-32B \\
        --model_gpus 1 1 4 \\
        --matchmaker round_robin

    # Or use a config file
    uv run python -m openjury.slurm.generate_slurm \\
        --config configs/arena_5model.json

    # Generate-only: cache completions for a single model
    uv run python -m openjury.slurm.generate_slurm \\
        --dataset alpaca-eval \\
        --models VLLM/Qwen/Qwen2.5-0.5B-Instruct \\
        --mode generate

    # Judge-only: run arena judge (assumes completions are cached)
    # API judges run on login node (no SLURM), GPU judges use sbatch
    uv run python -m openjury.slurm.generate_slurm \\
        --dataset alpaca-eval \\
        --models VLLM/Qwen/Qwen2.5-0.5B-Instruct \\
                 ArenaHard/gpt-4-0613 \\
        --judge_model OpenRouter/qwen/qwen3-32b \\
        --mode judge

    # Then submit
    bash slurm_scripts/<run_dir>/submit_all.sh


"""

from __future__ import annotations

import argparse
import os
import stat
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from textwrap import dedent

from openjury._logging import logger
from openjury.cli_args import add_arena_pipeline_args
from openjury.models.utils import (
    api_key_env_for_model,
    is_local_provider,
    needs_network,
    provider_from_model,
)


# ═════════════════════════════════════════════════════════════════════
#  Environment helpers
# ═════════════════════════════════════════════════════════════════════

from openjury import env as _env_mod


def _cli_or_env(cli_val: str | None, env_name: str, default: str | None = None) -> str | None:
    """CLI flag > env var > default.  Returns ``None`` if nothing set."""
    if cli_val is not None:
        return cli_val
    return _env_mod.get(env_name, default)


def _resolve_gpus(model: str, cli_gpus: int, default_gpus: int) -> int:
    """Return 0 for API models (unless explicitly overridden), else *cli_gpus*."""
    provider = provider_from_model(model)
    if not is_local_provider(provider):
        if cli_gpus != default_gpus:
            return cli_gpus
        return 0
    return cli_gpus


# ═════════════════════════════════════════════════════════════════════
#  Configuration dataclasses
# ═════════════════════════════════════════════════════════════════════


@dataclass
class SlurmJobConfig:
    """Configuration for a single SLURM job."""
    job_name: str
    n_gpus: int = 1
    n_nodes: int = 1
    partition: str = ""
    account: str = ""
    time: str = ""
    qos: str = "normal"
    memory: str = "64G"
    cpus_per_task: int = 8
    extra_sbatch: list[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    """Evaluation pipeline configuration.

    Three modes:

    - ``generate``: cache completions for one or more models.
    - ``judge``: run arena judge only (completions must be cached).
    - ``arena``: K generate jobs → 1 arena judge job (default).

    Cluster settings are read from CLI flags, environment variables, or
    a ``.env`` file (in that priority order).  Variables required for the
    current run that are missing are reported together.
    """
    dataset: str
    judge_model: str = "none"
    n_instructions: int | None = None
    rubric: str = "default"
    generation_max_tokens: int = 4096
    judge_max_tokens: int = 2048
    truncate_input_chars: int = 8192
    provide_explanation: bool = False
    judge_mode: str = "samplewise"
    no_swap: bool = False  # disable position-swap debiasing
    ignore_cache: bool = False  # force regeneration even if completions are cached
    ignore_score_cache: bool = False  # force re-scoring even if judge scores are cached

    # ── Judge ───────────────────────────────────────────────────────
    judge_gpus: int = 1
    judge_quantization: str | None = None
    judge_enable_thinking: bool | None = None
    judge_chat_template: str | None = None
    judge_chat_template_file: str | None = None
    judge_local: bool = True

    # ── Models (populated after construction from CLI / config) ─────
    models: list[str] = field(default_factory=list)
    model_gpus: list[int] = field(default_factory=list)
    model_quantizations: list[str | None] = field(default_factory=list)
    model_local: list[bool] = field(default_factory=list)

    # ── Arena matchmaker ────────────────────────────────────────────
    matchmaker: str = "round_robin"
    n_matches: int | None = None

    # ── Run tag (for unique output dirs / job names) ────────────────
    tag: str = ""

    # ── Cluster settings (from env vars) ────────────────────────────
    partition: str = ""
    account: str = ""
    time_generate: str = ""
    time_judge: str = ""
    qos: str = "normal"

    # ── Paths (from env vars) ───────────────────────────────────────
    project_dir: str = ""      # OpenJury repo root
    work_dir: str = ""         # Pipeline working directory (auto-set)
    logs_dir: str = ""         # SLURM logs directory

    @classmethod
    def from_env_and_args(cls, args: argparse.Namespace) -> PipelineConfig:
        """Construct from CLI args + environment variables.

        Priority: CLI flag > env var > ``.env`` file.
        Required variables that are missing are reported together.
        """
        # Load .env if present (env vars already set win)
        _env_mod.load_dotenv()

        # Execution target: local (GPU compute node) vs network (login node)
        judge_local = (
            not needs_network(args.judge_model)
            if args.judge_model and args.judge_model != "none"
            else False
        )
        models = args.models or []
        any_local_model = any(not needs_network(m) for m in models)
        any_slurm = any_local_model or judge_local

        # ── Validate required env vars (only those needed) ───────
        required: list[str] = ["USER_WORK_DIR"]
        if any_slurm:
            required += ["ACCOUNT", "PARTITION", "TIME_LIMIT"]
        # Only check vars not already supplied via CLI
        needed = []
        for var in required:
            cli_map = {
                "ACCOUNT": getattr(args, "account", None),
                "PARTITION": getattr(args, "partition", None),
                "TIME_LIMIT": (
                    getattr(args, "time_generate", None)
                    or getattr(args, "time_judge", None)
                ),
            }
            if var in cli_map and cli_map[var] is not None:
                continue
            needed.append(var)
        _env_mod.check_required(needed)

        # Resolve cluster settings from CLI / env
        partition = _cli_or_env(args.partition, "PARTITION") or ""
        account = _cli_or_env(args.account, "ACCOUNT") or ""
        time_generate = _cli_or_env(args.time_generate, "TIME_LIMIT") or ""
        time_judge = _cli_or_env(args.time_judge, "TIME_LIMIT") or ""

        # Resolve paths from env
        user_work = _env_mod.require("USER_WORK_DIR")
        slurm_work = _env_mod.get(
            "SLURM_WORK_DIR", os.path.join(user_work, "slurm_jobs"),
        )

        # Project dir = OpenJury repo root (auto-detect from this file)
        _auto_project = str(Path(__file__).resolve().parent.parent.parent)
        project_dir = _cli_or_env(
            args.project_dir,
            "OPENJURY_PROJECT_DIR",
            _auto_project,
        )

        # Tag for unique run identification (defaults to timestamp)
        tag = (
            getattr(args, "tag", None)
            or datetime.now().strftime("%Y%m%d_%H%M%S")
        )

        # Auto-generate work_dir under shared SLURM_WORK_DIR
        short_models = (
            "_".join(m.rsplit("/", 1)[-1][:15] for m in models[:3])
            if models else "unknown"
        )
        safe_name = f"{args.dataset}_{short_models}".replace("/", "_")
        work_dir = os.path.join(slurm_work, "openjury", safe_name, tag)

        # Judge GPU resolution
        _default_gpus = int(_env_mod.get("GPUS_PER_NODE", "1") or "1")
        if args.judge_model and args.judge_model != "none":
            judge_gpus = _resolve_gpus(
                args.judge_model, args.judge_gpus, _default_gpus,
            )
        else:
            judge_gpus = 0

        return cls(
            dataset=args.dataset,
            judge_model=args.judge_model,
            rubric=getattr(args, "rubric", "default"),
            n_instructions=args.n_instructions,
            generation_max_tokens=args.generation_max_tokens,
            judge_max_tokens=args.judge_max_tokens,
            truncate_input_chars=args.truncate_input_chars,
            provide_explanation=args.provide_explanation,
            judge_mode=getattr(args, "judge_mode", "samplewise"),
            no_swap=getattr(args, "no_swap", False),
            ignore_cache=getattr(args, "ignore_cache", False),
            ignore_score_cache=getattr(args, "ignore_score_cache", False),
            judge_gpus=judge_gpus,
            judge_quantization=args.judge_quantization,
            judge_enable_thinking=getattr(args, "enable_thinking", False) or None,
            judge_chat_template=getattr(args, "chat_template", None),
            judge_chat_template_file=getattr(args, "chat_template_file", None),
            judge_local=judge_local,
            tag=tag,
            partition=partition,
            account=account,
            time_generate=time_generate,
            time_judge=time_judge,
            qos=getattr(args, "qos", "normal"),
            project_dir=project_dir,
            work_dir=work_dir,
            logs_dir="",  # set later by generate_pipeline_scripts
        )


# ═════════════════════════════════════════════════════════════════════
#  Script templates
# ═════════════════════════════════════════════════════════════════════


def _build_env_preamble(pipeline: PipelineConfig, *, is_compute: bool) -> str:
    """Build an env-setup block for generated scripts.

    Sources the ``.env`` file (if present) and sets HF offline flags on
    compute nodes.  No external env-script mechanism — all config comes
    from env vars / ``.env``.

    Args:
        pipeline: Pipeline configuration.
        is_compute: ``True`` for sbatch scripts running on compute nodes
            (sets HF offline flags).  ``False`` for login-node scripts.
    """
    lines: list[str] = []

    # ── .env file (API keys, cluster vars) ───────────────────────
    env_file = os.path.join(pipeline.project_dir, ".env")
    lines.append("# Load .env if present (API keys, cluster settings)")
    lines.append(f'if [ -f "{env_file}" ]; then')
    lines.append(f'    set -a; source "{env_file}"; set +a')
    lines.append("fi")
    lines.append("")

    # ── HF offline flags (auto-set on compute nodes) ─────────────
    if is_compute:
        lines.append("# Compute node — enable HF offline mode")
        lines.append('if [ -z "${HF_HUB_OFFLINE:-}" ]; then')
        lines.append('    export HF_HUB_OFFLINE=1')
        lines.append('    export TRANSFORMERS_OFFLINE=1')
        lines.append('    export HF_DATASETS_OFFLINE=1')
        lines.append("fi")
        lines.append('export VLLM_NO_USAGE_STATS=1')
        lines.append('export UV_NO_SYNC=1')
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


def _render_header(
    job: SlurmJobConfig,
    pipeline: PipelineConfig,
) -> str:
    """Render the SBATCH header + environment preamble."""
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


# ── Login-node header (no SLURM, for API models) ────────────────────

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


def _render_local_header(
    job_name: str,
    pipeline: PipelineConfig,
    model: str,
) -> str:
    """Render a login-node (non-SLURM) script header for API models."""
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


# ── Generate step ────────────────────────────────────────────────────


def _generate_step_script(
    model: str,
    output_parquet: str,
    pipeline: PipelineConfig,
    job: SlurmJobConfig,
    quantization: str | None = None,
    *,
    local: bool = True,
) -> str:
    """Build a script for a generate_step job.

    Args:
        model: Model specification string.
        output_parquet: Path for the output parquet.
        pipeline: Pipeline configuration.
        job: SLURM job configuration (used for sbatch headers and TP size).
        quantization: Optional quantisation method.
        local: If ``True`` (default), emit an sbatch script for a compute
            node.  If ``False``, emit a plain bash script for a login node
            (API model that needs network access).
    """
    if local:
        header = _render_header(job, pipeline)
    else:
        header = _render_local_header(job.job_name, pipeline, model)

    # Build command parts — only include flags that are set
    parts: list[str] = [
        'uv run python -m openjury.cli.generate \\',
        f'    --model "{model}" \\',
        f'    --dataset "{pipeline.dataset}" \\',
        f'    --output "{output_parquet}" \\',
        f'    --tensor_parallel_size {job.n_gpus} \\',
        f'    --max_tokens {pipeline.generation_max_tokens} \\',
        f'    --truncate_input_chars {pipeline.truncate_input_chars} \\',
    ]
    if pipeline.n_instructions:
        parts.append(f'    --n_instructions {pipeline.n_instructions} \\')
    if pipeline.ignore_cache:
        parts.append('    --ignore_cache \\')
    if quantization:
        parts.append(f'    --quantization {quantization}')
    else:
        # Remove trailing backslash from last line
        parts[-1] = parts[-1].rstrip(' \\')

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


# ── Arena judge step (K models) ──────────────────────────────────────


def _build_arena_config_dict(
    pipeline: PipelineConfig,
    output_dir: str,
) -> dict:
    """Build the ArenaConfig JSON dict from PipelineConfig fields.

    Used by _arena_step_script to write arena_config.json alongside the
    SLURM scripts, so the arena step can be driven by ``--config``.
    """
    from openjury.arena.config import ArenaConfig, JudgeConfig, MatchmakerConfig, ModelEntry

    model_entries = []
    for i, model in enumerate(pipeline.models):
        gpus = pipeline.model_gpus[i] if i < len(pipeline.model_gpus) else 1
        quant = (
            pipeline.model_quantizations[i]
            if i < len(pipeline.model_quantizations)
            else None
        )
        model_entries.append(ModelEntry(
            name=model,
            gpus=gpus,
            quantization=quant,
        ))

    arena_cfg = ArenaConfig(
        dataset=pipeline.dataset,
        n_instructions=pipeline.n_instructions,
        models=model_entries,
        judge=JudgeConfig(
            model=pipeline.judge_model,
            gpus=pipeline.judge_gpus,
            mode=pipeline.judge_mode,
            max_tokens=pipeline.judge_max_tokens,
            quantization=pipeline.judge_quantization,
            no_swap=pipeline.no_swap,
            provide_explanation=pipeline.provide_explanation,
            enable_thinking=pipeline.judge_enable_thinking,
            chat_template=pipeline.judge_chat_template,
            chat_template_file=pipeline.judge_chat_template_file,
        ),
        rubric=pipeline.rubric,
        matchmaker=MatchmakerConfig(
            strategy=pipeline.matchmaker,
            n_matches=pipeline.n_matches,
        ),
        generation_max_tokens=pipeline.generation_max_tokens,
        truncate_input_chars=pipeline.truncate_input_chars,
        ignore_cache=pipeline.ignore_cache,
        ignore_score_cache=pipeline.ignore_score_cache,
        output_dir=output_dir,
        include_completions=False,
        include_raw_judge=False,
    )
    return arena_cfg.to_dict()


def _arena_step_script(
    pipeline: PipelineConfig,
    job: SlurmJobConfig,
    output_dir: str,
    config_path: str,
    *,
    local: bool = True,
) -> str:
    """Build a script for the arena_step judge job (K models).

    The arena step reads the arena_config.json written alongside the SLURM
    scripts, loads cached completions for all models, runs the arena judge
    (samplewise by default), computes ratings, and saves arena.json.

    Args:
        pipeline: Pipeline configuration.
        job: SLURM job parameters.
        output_dir: Results output directory.
        config_path: Path to the arena_config.json file.
        local: Whether this runs on a compute node (True) or login node.
    """
    if local:
        header = _render_header(job, pipeline)
    else:
        header = _render_local_header(job.job_name, pipeline, pipeline.judge_model)

    cmd = f'uv run python -m openjury.cli.arena --config "{config_path}"'

    model_list_str = "\n".join(
        f'echo "  [{i+1}] {m}"'
        for i, m in enumerate(pipeline.models)
    )

    body = dedent(f"""\
        # ── Arena Judge (K-model evaluation) ──────────────────────
        echo "Arena evaluation: {len(pipeline.models)} models"
        {model_list_str}
        echo "Judge:      {pipeline.judge_model} (TP={job.n_gpus})"
        echo "Rubric:     {pipeline.rubric}"
        echo "Mode:       {pipeline.judge_mode}"
        echo "Matchmaker: {pipeline.matchmaker}"
        echo "Config:     {config_path}"

        {cmd}

        echo ""
        echo "✅ Arena step completed: $(date)"
        echo "   Results: {output_dir}/"
    """)

    return header + body



# ═════════════════════════════════════════════════════════════════════
#  submit_all.sh builder (K generate jobs + 1 arena judge)
# ═════════════════════════════════════════════════════════════════════

_SLURM_WAIT_HELPER = """\
# Helper: poll squeue until a SLURM job finishes, then verify success.
_wait_slurm_job() {
    local job_id=$1 name=$2
    echo "⏳ Waiting for $name (SLURM job $job_id)..."
    while squeue -j "$job_id" -h 2>/dev/null | grep -q .; do
        sleep 30
    done
    local state
    state=$(sacct -j "$job_id" -n -o State --parsable2 | head -1)
    if [[ "$state" != "COMPLETED" ]]; then
        echo "❌ $name failed (state: $state)"
        exit 1
    fi
    echo "✅ $name completed"
}
"""


def _build_arena_submit_all(
    pipeline: PipelineConfig,
    gen_paths: list[Path],
    gen_locals: list[bool],
    path_judge: Path,
    tag_suffix: str,
    results_dir: str,
    *,
    models_cached: list[str] | None = None,
) -> str:
    """Build ``submit_all.sh`` for arena mode.

    Submits generate jobs for uncached models (if any) and one arena
    judge job that depends on all of them.  When all completions are
    cached, only the judge job is emitted.
    """
    has_gen = bool(gen_paths)
    cached_names = models_cached or []

    # ── Preamble ─────────────────────────────────────────────────
    if has_gen:
        gen_model_lines = "\n".join(
            f'echo "  [{i+1}] {m} ({"compute node" if gen_locals[i] else "login node"})"'
            for i, m in enumerate(
                # gen_paths corresponds to uncached models only;
                # derive model names from the script filenames
                [p.stem.split("_", 2)[-1] for p in gen_paths]
            )
        )
    else:
        gen_model_lines = 'echo "  (all completions loaded from cache)"'

    n_gen = len(gen_paths)
    n_cached = len(cached_names)
    summary = f"{n_gen} generate job(s)"
    if n_cached:
        summary += f" ({n_cached} cached)"
    summary += " \u2192 1 arena judge"

    preamble = dedent(f"""\
        #!/bin/bash
        # Submit arena pipeline: {summary}
        set -euo pipefail
        mkdir -p {pipeline.logs_dir} {pipeline.work_dir}

        echo "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
        echo "  OpenJury Arena Pipeline"
        echo "  Dataset:    {pipeline.dataset}"
        echo "  Models:     {len(pipeline.models)} total ({n_cached} cached, {n_gen} to generate)"
        {gen_model_lines}
        echo "  Judge:      {pipeline.judge_model} ({"compute node" if pipeline.judge_local else "login node"})"
        echo "  Matchmaker: {pipeline.matchmaker}"
        echo "  Tag:        {pipeline.tag}"
        echo "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
        echo ""
    """)

    lines: list[str] = [preamble]

    # ── Generation phase (only if there are uncached models) ─────
    if has_gen:
        any_sbatch_gen = any(gen_locals)
        if any_sbatch_gen:
            lines.append(_SLURM_WAIT_HELPER)

        lines.append("# \u2500\u2500 Generation Phase \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
        lines.append("SLURM_GEN_JOBS=()  # collect sbatch job IDs for dependency")
        lines.append("")

        for i, (path, is_local) in enumerate(zip(gen_paths, gen_locals)):
            short = path.stem.split("_", 2)[-1][:25]  # from filename
            if is_local:
                lines.append(f'JOB_{i+1}=$(sbatch --parsable {path})')
                lines.append(f'echo "\u2705 Submitted Gen [{i+1}] {short}: $JOB_{i+1}"')
                lines.append(f'SLURM_GEN_JOBS+=("$JOB_{i+1}")')
            else:
                log = f"{pipeline.logs_dir}/oj_gen{i+1}{tag_suffix}.log"
                lines.append(f'echo "\U0001f680 Starting Gen [{i+1}] {short} on login node..."')
                lines.append(f'bash {path} > {log} 2>&1 &')
                lines.append(f'PID_{i+1}=$!')
                lines.append(f'echo "   PID: $PID_{i+1}  Log: {log}"')
            lines.append("")

        # ── Synchronise ──────────────────────────────────────────
        lines.append("# \u2500\u2500 Synchronise: wait for ALL generation jobs \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")

        for i, is_local in enumerate(gen_locals):
            if not is_local:
                short = gen_paths[i].stem.split("_", 2)[-1][:25]
                lines.append(f'echo "\u23f3 Waiting for Gen [{i+1}] {short} (PID $PID_{i+1})..."')
                lines.append(f'wait $PID_{i+1} || {{ echo "\u274c Gen [{i+1}] failed"; exit 1; }}')
                lines.append(f'echo "\u2705 Gen [{i+1}] {short} completed"')

        for i, is_local in enumerate(gen_locals):
            if is_local:
                short = gen_paths[i].stem.split("_", 2)[-1][:25]
                lines.append(f'_wait_slurm_job "$JOB_{i+1}" "Gen [{i+1}] {short}"')

        lines.append("")
        lines.append('echo ""')
        lines.append('echo "All generation jobs completed."')
        lines.append('echo ""')
    else:
        lines.append('echo "\u2705 All completions loaded from cache — skipping generation"')
        lines.append('echo ""')

    # ── Arena judge phase ────────────────────────────────────────
    lines.append("")
    lines.append("# \u2500\u2500 Arena Judge Phase \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")

    if not has_gen:
        # No gen jobs → submit judge directly (no dependency)
        if pipeline.judge_local:
            lines.append(f'JOB_ARENA=$(sbatch --parsable {path_judge})')
            lines.append(f'echo "\u2705 Submitted Arena Judge: $JOB_ARENA  ({path_judge.name})"')
            lines.append(f'echo "Results: {results_dir}/"')
        else:
            log_judge = f"{pipeline.logs_dir}/oj_arena{tag_suffix}.log"
            lines.append('echo "\U0001f680 Running arena judge on login node..."')
            lines.append(f'bash {path_judge} > {log_judge} 2>&1')
            lines.append('echo "\u2705 Arena judge completed"')
            lines.append(f'echo "Results: {results_dir}/"')
            lines.append(f'echo "Log:     {log_judge}"')
    elif pipeline.judge_local and all(gen_locals):
        # Fast path: pure sbatch chain — build dependency string
        lines.append('DEP_STR=$(IFS=:; echo "${SLURM_GEN_JOBS[*]}")')
        lines.append(f'JOB_ARENA=$(sbatch --parsable --dependency=afterok:$DEP_STR {path_judge})')
        lines.append(f'echo "\u2705 Submitted Arena Judge: $JOB_ARENA  ({path_judge.name})"')
        lines.append('echo "   \u2514\u2500\u2500 depends on: $DEP_STR"')
        lines.append("")
        lines.append(f'echo "Monitor: squeue -j $JOB_ARENA"')
        lines.append(f'echo "Results: {results_dir}/"')
    elif pipeline.judge_local:
        # Mixed gen: we already waited, now submit judge
        lines.append(f'JOB_ARENA=$(sbatch --parsable {path_judge})')
        lines.append(f'echo "\u2705 Submitted Arena Judge: $JOB_ARENA  ({path_judge.name})"')
        lines.append(f'echo "Results: {results_dir}/"')
    else:
        # Login-node judge
        log_judge = f"{pipeline.logs_dir}/oj_arena{tag_suffix}.log"
        lines.append('echo "\U0001f680 Running arena judge on login node..."')
        lines.append(f'bash {path_judge} > {log_judge} 2>&1')
        lines.append('echo "\u2705 Arena judge completed"')
        lines.append(f'echo "Results: {results_dir}/"')
        lines.append(f'echo "Log:     {log_judge}"')

    lines.append("")
    return "\n".join(lines) + "\n"


# ═════════════════════════════════════════════════════════════════════
#  Pipeline script generator
# ═════════════════════════════════════════════════════════════════════


def generate_pipeline_scripts(
    pipeline: PipelineConfig,
    output_dir: Path,
    mode: str = "arena",
    on_missing_completions: str = "error",
) -> list[Path]:
    """Generate all SLURM scripts for the evaluation pipeline.

    Args:
        pipeline: Full pipeline configuration.
        output_dir: Directory to write scripts into.
        mode: ``"generate"`` = completions only for one or more models;
              ``"judge"`` = arena judge only (completions must be cached);
              ``"arena"`` = K generate jobs + 1 arena judge job (default).
        on_missing_completions: ``"error"`` = abort if any model is
            missing cached completions (judge mode only).
            ``"skip"`` = warn and let ``arena_step`` resolve on the fly.

    Returns:
        List of generated script paths.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # SLURM log files (.out / .err) go in the same directory as the scripts
    pipeline.logs_dir = str(output_dir)

    generated: list[Path] = []

    # Short tag suffix for SLURM job names (truncate to keep squeue readable)
    tag_suffix = f"_{pipeline.tag[:12]}" if pipeline.tag else ""

    def _write(path: Path, content: str) -> Path:
        path.write_text(content)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        generated.append(path)
        return path

    if mode == "generate":
        # ── Generate-only: cache completions for one or more models ──
        from openjury.cache.completions import cache as _comp_cache

        gen_paths: list[Path] = []
        gen_locals: list[bool] = []
        skipped: list[str] = []

        for idx, model in enumerate(pipeline.models):
            gpus = pipeline.model_gpus[idx] if idx < len(pipeline.model_gpus) else 1
            quant = (
                pipeline.model_quantizations[idx]
                if idx < len(pipeline.model_quantizations) else None
            )
            is_local = (
                pipeline.model_local[idx] if idx < len(pipeline.model_local) else True
            )

            # Skip if already cached
            if not pipeline.ignore_cache and _comp_cache.exists(
                model=model, dataset=pipeline.dataset, n=pipeline.n_instructions,
            ):
                logger.info("  ⏭ %s already cached — skipping", model)
                skipped.append(model)
                continue

            short = model.rsplit("/", 1)[-1].replace("/", "_")[:20]
            comp_out = f"{pipeline.work_dir}/completions_{idx+1}_{short}.parquet"

            job = SlurmJobConfig(
                job_name=f"oj_gen{idx+1}{tag_suffix}",
                n_gpus=gpus,
                partition=pipeline.partition,
                account=pipeline.account,
                time=pipeline.time_generate,
                qos=pipeline.qos,
            )
            path = _write(
                output_dir / f"{idx+1:02d}_generate_{short}.sh",
                _generate_step_script(
                    model, comp_out, pipeline, job,
                    quantization=quant,
                    local=is_local,
                ),
            )
            gen_paths.append(path)
            gen_locals.append(is_local)

        n_total = len(pipeline.models)
        n_gen = len(gen_paths)

        if not gen_paths:
            # All cached — just log and return, no scripts needed
            logger.info(
                "  ✅ All %d model(s) already cached — nothing to generate",
                n_total,
            )
            for m in pipeline.models:
                logger.info("     • %s", m)
            return generated
        else:
            # Build submit_all.sh for generation jobs
            lines: list[str] = []
            lines.append(dedent(f"""\
                #!/bin/bash
                set -euo pipefail
                mkdir -p {pipeline.logs_dir} {pipeline.work_dir}

                echo "═══════════════════════════════════════════════════"
                echo "  OpenJury — Generate Completions ({n_gen} model(s))"
                echo "  Dataset: {pipeline.dataset}"
                echo "═══════════════════════════════════════════════════"
            """))

            any_sbatch = any(gen_locals)
            if any_sbatch:
                lines.append(_SLURM_WAIT_HELPER)

            lines.append("SLURM_GEN_JOBS=()")
            lines.append("")

            for i, (path, is_local) in enumerate(zip(gen_paths, gen_locals)):
                short = path.stem.split("_", 2)[-1][:25]
                if is_local:
                    lines.append(f'JOB_{i+1}=$(sbatch --parsable {path})')
                    lines.append(f'echo "✅ Submitted Gen [{i+1}] {short}: $JOB_{i+1}"')
                    lines.append(f'SLURM_GEN_JOBS+=("$JOB_{i+1}")')
                else:
                    log = f"{pipeline.logs_dir}/oj_gen{i+1}{tag_suffix}.log"
                    lines.append(f'echo "🚀 Starting Gen [{i+1}] {short} on login node..."')
                    lines.append(f'bash {path} > {log} 2>&1 &')
                    lines.append(f'PID_{i+1}=$!')
                    lines.append(f'echo "   PID: $PID_{i+1}  Log: {log}"')
                lines.append("")

            # Wait for all
            lines.append("# Wait for all generation jobs")
            for i, is_local in enumerate(gen_locals):
                short = gen_paths[i].stem.split("_", 2)[-1][:25]
                if not is_local:
                    lines.append(f'wait $PID_{i+1} || {{ echo "❌ Gen [{i+1}] failed"; exit 1; }}')
                    lines.append(f'echo "✅ Gen [{i+1}] {short} completed"')
                else:
                    lines.append(f'_wait_slurm_job "$JOB_{i+1}" "Gen [{i+1}] {short}"')

            lines.append("")
            lines.append('echo ""')
            lines.append('echo "✅ All generation jobs completed."')

            _write(output_dir / "submit_all.sh", "\n".join(lines) + "\n")

    elif mode == "judge":
        # ── Judge-only mode: no generation, just arena judge ──
        from openjury.cache.completions import cache as _comp_cache

        # Validate that completions exist for all models
        missing: list[str] = []
        for model in pipeline.models:
            if not _comp_cache.exists(
                model=model,
                dataset=pipeline.dataset,
                n=pipeline.n_instructions,
            ):
                missing.append(model)

        if missing:
            msg = (
                f"{len(missing)} model(s) have no cached completions "
                f"for dataset={pipeline.dataset!r}:\n"
                + "\n".join(f"  ✗ {m}" for m in missing)
            )
            if on_missing_completions == "error":
                raise SystemExit(
                    f"ERROR: {msg}\n\n"
                    "Generate completions first (--mode generate) or "
                    "re-run with --on_missing_completions skip to let "
                    "arena_step resolve them on the fly."
                )
            else:
                logger.warning(
                    "⚠️  %s\n   Proceeding anyway — arena_step will "
                    "attempt to generate on the fly.", msg,
                )
        else:
            logger.info(
                "  ✅ All %d model(s) have cached completions",
                len(pipeline.models),
            )

        results = f"{pipeline.work_dir}/results"

        # Write arena_config.json
        import json as _json
        config_dict = _build_arena_config_dict(pipeline, results)
        config_path = output_dir / "arena_config.json"
        config_path.write_text(
            _json.dumps(config_dict, indent=2), encoding="utf-8",
        )
        generated.append(config_path)

        # Arena judge job
        job_judge = SlurmJobConfig(
            job_name=f"oj_arena{tag_suffix}",
            n_gpus=pipeline.judge_gpus,
            partition=pipeline.partition,
            account=pipeline.account,
            time=pipeline.time_judge,
            qos=pipeline.qos,
        )
        path_judge = _write(
            output_dir / "01_arena_judge.sh",
            _arena_step_script(
                pipeline, job_judge, results,
                config_path=str(config_path),
                local=pipeline.judge_local,
            ),
        )

        # Build submit_all.sh
        if pipeline.judge_local:
            # GPU judge → sbatch
            _write(output_dir / "submit_all.sh", dedent(f"""\
                #!/bin/bash
                set -euo pipefail
                mkdir -p {pipeline.logs_dir} {pipeline.work_dir}

                echo "═══════════════════════════════════════════════════"
                echo "  OpenJury — Judge Only"
                echo "  Dataset:  {pipeline.dataset}"
                echo "  Models:   {len(pipeline.models)}"
                echo "  Judge:    {pipeline.judge_model} (compute node, {pipeline.judge_gpus} GPU(s))"
                echo "  Mode:     {pipeline.judge_mode}"
                echo "═══════════════════════════════════════════════════"

                JOB=$(sbatch --parsable {path_judge})
                echo "✅ Submitted arena judge: $JOB  ({path_judge.name})"
                echo "Monitor: squeue -j $JOB"
                echo "Results: {results}/"
            """))
        else:
            # API judge → plain bash on login node
            log_judge = f"{pipeline.logs_dir}/oj_arena{tag_suffix}.log"
            _write(output_dir / "submit_all.sh", dedent(f"""\
                #!/bin/bash
                set -euo pipefail
                mkdir -p {pipeline.logs_dir} {pipeline.work_dir}

                echo "═══════════════════════════════════════════════════"
                echo "  OpenJury — Judge Only (login node)"
                echo "  Dataset:  {pipeline.dataset}"
                echo "  Models:   {len(pipeline.models)}"
                echo "  Judge:    {pipeline.judge_model} (API — no SLURM needed)"
                echo "  Mode:     {pipeline.judge_mode}"
                echo "═══════════════════════════════════════════════════"

                echo "🚀 Running arena judge on login node..."
                bash {path_judge} 2>&1 | tee {log_judge}
                echo "✅ Arena judge completed"
                echo "Results: {results}/"
                echo "Log:     {log_judge}"
            """))

    elif mode == "arena":
        # ── Arena mode: K generate jobs + 1 arena judge ──────────
        results = f"{pipeline.work_dir}/results"
        gen_paths: list[Path] = []
        gen_locals: list[bool] = []

        # Check which models need generation (cache-aware)
        from openjury.cache.completions import cache as _comp_cache

        models_cached: list[str] = []
        models_need_gen: list[tuple[int, str]] = []  # (original_idx, model)

        for i, model in enumerate(pipeline.models):
            if (
                not pipeline.ignore_cache
                and _comp_cache.exists(
                    model=model,
                    dataset=pipeline.dataset,
                    n=pipeline.n_instructions,
                )
            ):
                models_cached.append(model)
            else:
                models_need_gen.append((i, model))

        if models_cached:
            logger.info(
                "  \u2705 %d model(s) have cached completions — skipping generation:",
                len(models_cached),
            )
            for m in models_cached:
                logger.info("     \u2022 %s", m)

        if models_need_gen:
            logger.info(
                "  \U0001f527 %d model(s) need generation:",
                len(models_need_gen),
            )
            for _, m in models_need_gen:
                logger.info("     \u2022 %s", m)
        else:
            logger.info(
                "  \u2705 All completions cached — only judge job will be created"
            )

        # Also check score cache (all models scored → judge-only)
        all_scores_cached = False
        if not pipeline.ignore_score_cache and not models_need_gen:
            from openjury.cache.scores import score_cache as _score_cache

            all_scores_cached = all(
                _score_cache.exists(
                    judge=pipeline.judge_model,
                    rubric=pipeline.rubric,
                    model=model,
                    dataset=pipeline.dataset,
                    n=pipeline.n_instructions,
                )
                for model in pipeline.models
            )
            if all_scores_cached:
                logger.info(
                    "  \u2705 All judge scores also cached — "
                    "arena step will only compute ratings"
                )

        # Create generate scripts only for uncached models
        for seq, (orig_idx, model) in enumerate(models_need_gen):
            gpus = (
                pipeline.model_gpus[orig_idx]
                if orig_idx < len(pipeline.model_gpus) else 1
            )
            quant = (
                pipeline.model_quantizations[orig_idx]
                if orig_idx < len(pipeline.model_quantizations) else None
            )
            is_local = (
                pipeline.model_local[orig_idx]
                if orig_idx < len(pipeline.model_local) else True
            )

            short = model.rsplit("/", 1)[-1].replace("/", "_")[:20]
            comp_out = f"{pipeline.work_dir}/completions_{orig_idx+1}_{short}.parquet"

            job = SlurmJobConfig(
                job_name=f"oj_gen{seq+1}{tag_suffix}",
                n_gpus=gpus,
                partition=pipeline.partition,
                account=pipeline.account,
                time=pipeline.time_generate,
                qos=pipeline.qos,
            )
            path = _write(
                output_dir / f"{seq+1:02d}_generate_{short}.sh",
                _generate_step_script(
                    model, comp_out, pipeline, job,
                    quantization=quant,
                    local=is_local,
                ),
            )
            gen_paths.append(path)
            gen_locals.append(is_local)

        # Write arena_config.json for the arena step
        import json as _json
        config_dict = _build_arena_config_dict(pipeline, results)
        config_path = output_dir / "arena_config.json"
        config_path.write_text(
            _json.dumps(config_dict, indent=2), encoding="utf-8",
        )
        generated.append(config_path)

        # Arena judge job — depends on all generate jobs (if any)
        judge_idx = len(gen_paths) + 1
        job_judge = SlurmJobConfig(
            job_name=f"oj_arena{tag_suffix}",
            n_gpus=pipeline.judge_gpus,
            partition=pipeline.partition,
            account=pipeline.account,
            time=pipeline.time_judge,
            qos=pipeline.qos,
        )
        path_judge = _write(
            output_dir / f"{judge_idx:02d}_arena_judge.sh",
            _arena_step_script(
                pipeline, job_judge, results,
                config_path=str(config_path),
                local=pipeline.judge_local,
            ),
        )

        # Build submit_all.sh
        _write(
            output_dir / "submit_all.sh",
            _build_arena_submit_all(
                pipeline, gen_paths, gen_locals, path_judge,
                tag_suffix, results,
                models_cached=models_cached,
            ),
        )

    else:
        raise ValueError(f"Unknown pipeline mode: {mode!r}")

    return generated


# ═════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        prog="openjury-slurm",
        description=(
            "Generate SLURM sbatch scripts for LLM evaluation pipelines. "
            "Cluster settings are read from environment variables or a .env file "
            "in the project root. Missing required variables are reported together."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""\
            Environment setup:

              # Option A: .env file (recommended)
              cp .env.example .env && $EDITOR .env

              # Option B: set env vars directly
              export ACCOUNT=my_account PARTITION=my_partition
              export USER_WORK_DIR=/path/to/workdir TIME_LIMIT=02:00:00

            Examples:

              # Arena mode: K-model evaluation (default)
              uv run python -m openjury.slurm.generate_slurm \\
                --dataset alpaca-eval \\
                --models VLLM/Qwen/Qwen2.5-0.5B-Instruct \\
                         VLLM/Qwen/Qwen2.5-1.5B-Instruct \\
                         VLLM/meta-llama/Llama-3.1-8B-Instruct \\
                --judge_model VLLM/Qwen/Qwen3-32B \\
                --model_gpus 1 1 4 \\
                --matchmaker round_robin

              # Arena from config file
              uv run python -m openjury.slurm.generate_slurm \\
                --config configs/arena_5model.json

              # Generate completions for one model (cached)
              uv run python -m openjury.slurm.generate_slurm \\
                --dataset alpaca-eval \\
                --models VLLM/Qwen/Qwen2.5-0.5B-Instruct \\
                --mode generate --n_instructions 100

              # Judge-only: assumes completions exist, API judge → login node
              uv run python -m openjury.slurm.generate_slurm \\
                --dataset alpaca-eval \\
                --models VLLM/Qwen/Qwen2.5-0.5B-Instruct \\
                         ArenaHard/gpt-4-0613 \\
                --judge_model OpenRouter/qwen/qwen3-32b \\
                --mode judge --submit
        """),
    )

    # ── Pipeline args (config, models, dataset, judge, etc.) ────
    add_arena_pipeline_args(parser, output_dir_default="slurm_scripts")

    # Override --judge_gpus default with $GPUS_PER_NODE (SLURM-specific)
    _default_gpus = int(_env_mod.get("GPUS_PER_NODE", "1") or "1")
    parser.set_defaults(judge_gpus=_default_gpus)

    # ── Per-model GPU / quantization config ──────────────────────
    _default_gpus = int(_env_mod.get("GPUS_PER_NODE", "1") or "1")
    parser.add_argument(
        "--model_gpus", nargs="+", type=int, default=None,
        help="GPU counts per model (same order as --models). "
             "Default: $GPUS_PER_NODE or 1 for each.",
    )
    parser.add_argument(
        "--model_quantizations", nargs="+", default=None,
        help="Quantization per model (same order as --models). "
             "Use 'none' for no quantization.",
    )

    # ── SLURM overrides (optional, default = env vars) ───────────
    parser.add_argument(
        "--partition", default=None,
        help="Override $PARTITION env var.",
    )
    parser.add_argument(
        "--account", default=None,
        help="Override $ACCOUNT env var.",
    )
    parser.add_argument(
        "--time_generate", default=None,
        help="Wall time for generation jobs. Default: $TIME_LIMIT.",
    )
    parser.add_argument(
        "--time_judge", default=None,
        help="Wall time for judge job. Default: $TIME_LIMIT.",
    )
    parser.add_argument("--qos", default="normal")

    # ── Output ───────────────────────────────────────────────────
    parser.add_argument(
        "--tag", default=None,
        help="Run tag for unique work dirs and job names. "
             "Default: YYYYMMDD_HHMMSS timestamp.",
    )
    parser.add_argument(
        "--mode",
        choices=["generate", "arena", "judge"],
        default="arena",
        help="Pipeline mode: "
             "'arena' (default) = K generate jobs + 1 arena judge job. "
             "'generate' = generate completions only (no judge). "
             "'judge' = run arena judge only (assumes completions "
             "are already cached). API judges run on login node "
             "(no SLURM job), GPU judges use sbatch. "
    )
    
    parser.add_argument(
        "--on_missing_completions",
        choices=["error", "skip"],
        default="error",
        help="What to do in judge mode when completions are missing "
             "for a model: 'error' (default) = abort with an error. "
             "'skip' = warn and let arena_step resolve on the fly.",
    )
    parser.add_argument(
        "--project_dir", default=None,
        help="Override auto-detected OpenJury project dir.",
    )
    parser.add_argument(
        "--submit", action="store_true",
        help="Immediately run submit_all.sh after generating scripts. "
             "Without this flag, scripts are generated but not submitted "
             "(dry-run).",
    )
    parser.add_argument(
        "--remote", action="store_true",
        help="Submit to a remote cluster via slurmpilot (SSH). "
             "Requires 'uv sync --extra remote' and a configured cluster. "
             "API models run locally on your machine; GPU models are "
             "submitted via sbatch on the remote cluster. "
             "Implies --submit.",
    )
    parser.add_argument(
        "--cluster", default=None,
        help="slurmpilot cluster name for --remote (e.g. 'leonardo'). "
             "Defaults to $SP_DEFAULT_CLUSTER or slurmpilot's configured "
             "default cluster.",
    )
    parser.add_argument(
        "--remote_project_dir", default=None,
        help="Path to the OpenJury project on the remote cluster. "
             "If set, scripts 'cd' there instead of uploading sources. "
             "Use this when code is already deployed on the cluster.",
    )
    parser.add_argument(
        "--wait_timeout", type=int, default=7200,
        help="Max seconds to wait per job when using --remote (default: 7200).",
    )

    args = parser.parse_args()

    # ── Load config file (if provided) ───────────────────────────
    if args.config:
        import json as _json_cfg

        with open(args.config, "r", encoding="utf-8") as _f:
            _raw = _json_cfg.load(_f)

        
        from openjury.arena.config import ArenaConfig as _ArenaConfig
        _acfg = _ArenaConfig.load(args.config)
        # Inject into args for downstream PipelineConfig population
        if not args.models:
            args.models = _acfg.model_names
        if not args.judge_model:
            args.judge_model = _acfg.judge.model
        if not args.model_gpus:
            args.model_gpus = [m.gpus for m in _acfg.models]
        if not args.model_quantizations:
            args.model_quantizations = [
                m.quantization or "none" for m in _acfg.models
            ]
        if not args.dataset:
            args.dataset = _acfg.dataset
        # Pull judge GPU/quantization from config if not overridden on CLI
        args.judge_gpus = _acfg.judge.gpus
        if _acfg.judge.quantization and not args.judge_quantization:
            args.judge_quantization = _acfg.judge.quantization
        if _acfg.judge.mode:
            args.judge_mode = _acfg.judge.mode
        if _acfg.judge.enable_thinking is not None:
            args.enable_thinking = bool(_acfg.judge.enable_thinking)
        if _acfg.judge.chat_template and not args.chat_template:
            args.chat_template = _acfg.judge.chat_template
        if _acfg.judge.chat_template_file and not args.chat_template_file:
            args.chat_template_file = _acfg.judge.chat_template_file
        args.matchmaker = _acfg.matchmaker.strategy
        args.n_matches = _acfg.matchmaker.n_matches
        if _acfg.ignore_score_cache:
            args.ignore_score_cache = True

    # ── Mode-specific validation ─────────────────────────────────
    if args.mode == "arena":
        if not args.models or len(args.models) < 2:
            parser.error(
                "--models with at least 2 models is required for arena mode "
                "(or use --config)"
            )
        if not args.judge_model:
            parser.error(
                "--judge_model is required for arena mode "
                "(or use --config)"
            )
    elif args.mode == "generate":
        if not args.models or len(args.models) < 1:
            parser.error(
                "--models with at least 1 model is required for generate mode"
            )
        if not args.judge_model:
            args.judge_model = "none"
    elif args.mode == "judge":
        if not args.models or len(args.models) < 2:
            parser.error(
                "--models with at least 2 models is required for judge mode "
                "(or use --config)"
            )
        if not args.judge_model:
            parser.error(
                "--judge_model is required for judge mode "
                "(or use --config)"
            )
    if not args.dataset:
        parser.error(
            "--dataset is required (or supply it via --config)."
        )

    # ── Build PipelineConfig ─────────────────────────────────────
    config = PipelineConfig.from_env_and_args(args)

    # Populate model lists
    config.models = args.models
    config.model_gpus = (
        args.model_gpus if args.model_gpus
        else [_default_gpus] * len(args.models)
    )
    config.model_quantizations = [
        None if q == "none" else q
        for q in (args.model_quantizations or ["none"] * len(args.models))
    ]
    config.model_local = [not needs_network(m) for m in args.models]
    config.matchmaker = getattr(args, "matchmaker", "round_robin")
    config.n_matches = getattr(args, "n_matches", None)

    # Auto-detect judge GPU needs (API judges don't need GPUs)
    if (
        args.judge_model
        and args.judge_model != "none"
        and not is_local_provider(provider_from_model(args.judge_model))
    ):
        config.judge_gpus = 0


    # ── Build a unique per-run subfolder ─────────────────────────
    run_name = f"{config.dataset}_{args.mode}_{config.tag}".replace("/", "_")
    run_dir = (Path(args.output_dir) / run_name).resolve()

    # If output_dir was overridden by config, also redirect work_dir
    # so logs, results, and scripts all live together
    if args.output_dir != "slurm_scripts":
        config.work_dir = str(run_dir)


    scripts = generate_pipeline_scripts(
        pipeline=config,
        output_dir=run_dir,
        mode=args.mode,
        on_missing_completions=getattr(args, "on_missing_completions", "error"),
    )

    if not scripts:
        # Nothing to do (e.g. all completions cached in generate mode)
        return

    logger.info("")
    logger.info("Generated %d scripts in %s/", len(scripts), run_dir)
    for s in scripts:
        logger.info("  📄 %s", s.name)
    logger.info("")

    submit_script = run_dir / "submit_all.sh"

    if args.remote:
        # ── Remote submission via slurmpilot ──────────────────────
        from openjury.slurm.remote import submit_remote

        cluster_name = (
            args.cluster
            or os.environ.get("SP_DEFAULT_CLUSTER")
        )
        if not cluster_name:
            parser.error(
                "--cluster is required for --remote (or set $SP_DEFAULT_CLUSTER). "
                "Configure a cluster first:\n"
                "  sp-add-cluster --cluster NAME --host HOST --user USER"
            )

        # Identify gen scripts and judge script from generated files
        gen_scripts = [
            s for s in scripts
            if s.name.startswith(("0",)) and "generate" in s.name
        ]
        judge_scripts = [
            s for s in scripts
            if "arena_judge" in s.name or "judge" in s.name
        ]
        judge_script = judge_scripts[0] if judge_scripts else None

        # Match gen_locals from config
        gen_locals = config.model_local[:len(gen_scripts)]

        submit_remote(
            script_dir=run_dir,
            gen_scripts=gen_scripts,
            gen_locals=gen_locals,
            judge_script=judge_script,
            judge_local=config.judge_local,
            cluster=cluster_name,
            project_dir=config.project_dir,
            remote_project_dir=args.remote_project_dir,
            wait=True,
            wait_timeout=args.wait_timeout,
        )
    elif args.submit:
        import subprocess

        # Detect whether submit_all.sh will block (login-node jobs)
        # by checking if all models are API-based (no sbatch).
        all_api = not config.judge_local and all(
            not is_local_provider(provider_from_model(m))
            for m in config.models
        )

        if all_api:
            # API / login-node jobs would block — run in background
            log_file = run_dir / "submit_all.log"
            logger.info("Submitting jobs in background: bash %s", submit_script)
            with open(log_file, "w", encoding="utf-8") as log_fh:
                proc = subprocess.Popen(
                    ["bash", str(submit_script)],
                    cwd=str(run_dir),
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,  # detach from terminal
                )
            logger.info("✅ Background PID: %d", proc.pid)
            logger.info("Monitor:  tail -f %s", log_file)
            logger.info("Check:    ps -p %d", proc.pid)
        else:
            # SLURM jobs — sbatch returns immediately, no need to background
            logger.info("Submitting jobs via: bash %s", submit_script)
            result = subprocess.run(
                ["bash", str(submit_script)],
                cwd=str(run_dir),
                capture_output=False,
            )
            if result.returncode != 0:
                logger.error(
                    "submit_all.sh exited with code %d", result.returncode
                )
                raise SystemExit(result.returncode)
            logger.info("✅ All jobs submitted.")
    else:
        logger.info("To submit:  bash %s", submit_script)
        logger.info("Or re-run with --submit to submit immediately.")
        logger.info("Or re-run with --remote --cluster NAME to submit via SSH.")


if __name__ == "__main__":
    main()
