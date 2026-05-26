"""Generate SLURM sbatch scripts for OpenJury evaluation pipelines.

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

    # Agreement mode: human-vs-judge on datasets with inline completions
    uv run python -m openjury.slurm.generate_slurm \\
        --dataset lmsys \\
        --judge_model OpenRouter/qwen/qwen3-32b \\
        --mode agreement \\
        --n_instructions 200

    # Annotate mode: generic challenger-vs-opponent annotation
    uv run python -m openjury.slurm.generate_slurm \\
        --mode annotate \\
        --config configs/annotate/lmsys_gpt-oss20b_qwen3-32b.yaml

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
from openjury.arena.config import JudgeConfig, ModelEntry
from openjury.cli_args import add_arena_pipeline_args, parse_kwargs
from openjury.container_support import resolve_container_settings
from openjury.models.utils import (
    is_local_provider,
    needs_network,
    provider_from_model,
)
from openjury.project_paths import normalize_project_dir
from openjury.slurm import config_resolver as slurm_config_resolver
from openjury.slurm import modes as slurm_modes
from openjury.slurm.plans import (
    GenerateTaskConfig,
    SlurmExecutionConfig,
    SlurmRunPlan,
    TaskConfig,
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


def _build_pipeline_models(
    args: argparse.Namespace,
    *,
    default_gpus: int,
) -> list[ModelEntry]:
    """Build structured pipeline models from CLI args or preserved config payload."""
    if getattr(args, "model_entries", None):
        return [ModelEntry.from_raw(model.to_dict()) for model in args.model_entries]

    model_names = args.models or []
    model_gpus = (
        args.model_gpus if args.model_gpus else [default_gpus] * len(model_names)
    )
    model_quantizations = [
        None if quantization == "none" else quantization
        for quantization in (
            args.model_quantizations or ["none"] * len(model_names)
        )
    ]
    return [
        ModelEntry(
            name=model_name,
            gpus=model_gpus[idx] if idx < len(model_gpus) else default_gpus,
            quantization=(
                model_quantizations[idx]
                if idx < len(model_quantizations)
                else None
            ),
        )
        for idx, model_name in enumerate(model_names)
    ]


def _build_judge_config(
    args: argparse.Namespace,
    *,
    default_gpus: int,
) -> JudgeConfig:
    """Build a typed judge config from resolved CLI/config args."""
    if args.judge_model and args.judge_model != "none":
        judge_gpus = _resolve_gpus(
            args.judge_model,
            args.judge_gpus,
            default_gpus,
        )
    else:
        judge_gpus = 0

    return JudgeConfig(
        model=args.judge_model or "none",
        gpus=judge_gpus,
        mode=getattr(args, "judge_mode", "samplewise"),
        pairwise_prompt_style=getattr(args, "pairwise_prompt_style", "criteria"),
        max_tokens=getattr(args, "judge_max_tokens", 2048),
        quantization=getattr(args, "judge_quantization", None),
        no_swap=getattr(args, "no_swap", False),
        provide_explanation=getattr(args, "provide_explanation", False),
        enable_thinking=getattr(args, "enable_thinking", False) or None,
        chat_template=getattr(args, "chat_template", None),
        chat_template_file=getattr(args, "chat_template_file", None),
        max_model_len=(
            getattr(args, "max_model_len", None)
            if getattr(args, "max_model_len", None) is not None
            else getattr(args, "judge_max_model_len", None)
        ),
        enforce_eager=bool(
            getattr(args, "enforce_eager", False)
            or getattr(args, "judge_enforce_eager", False)
        ),
        temperature=getattr(args, "judge_temperature", 0.0),
        top_p=getattr(args, "judge_top_p", 1.0),
        n_trials=getattr(args, "judge_n_trials", 1),
        generation_kwargs=parse_kwargs(getattr(args, "gen_kwargs", None)),
    )


def _build_task_config(
    args: argparse.Namespace,
    *,
    default_gpus: int,
) -> TaskConfig:
    """Build the typed task config for the requested SLURM mode."""
    from openjury.annotate_config import AnnotateConfig
    from openjury.arena.config import AgreementConfig, ArenaConfig, MatchmakerConfig

    model_entries = _build_pipeline_models(args, default_gpus=default_gpus)
    mode = getattr(args, "mode", "arena")

    if mode == "generate":
        return GenerateTaskConfig(
            dataset=args.dataset,
            models=model_entries,
            n_instructions=args.n_instructions,
            language=getattr(args, "language", None),
            seed=getattr(args, "seed", 42),
            balance_by=getattr(args, "balance_by", None),
            generation_max_tokens=args.generation_max_tokens,
            truncate_input_chars=args.truncate_input_chars,
            ignore_cache=getattr(args, "ignore_cache", False),
        )

    if mode == "annotate":
        if not getattr(args, "config", None):
            raise SystemExit(
                "Annotate mode currently requires --config when using openjury-slurm."
            )
        return AnnotateConfig.load(args.config)

    judge = _build_judge_config(args, default_gpus=default_gpus)

    if mode == "agreement":
        return AgreementConfig(
            dataset=args.dataset,
            judge=judge,
            n_instructions=args.n_instructions,
            language=getattr(args, "language", None),
            seed=getattr(args, "seed", 42),
            balance_by=getattr(args, "balance_by", None),
            criteria=getattr(args, "criteria", "default"),
            ignore_score_cache=getattr(args, "ignore_score_cache", False),
            truncate_instruction=getattr(args, "truncate_instruction", 500),
        )

    return ArenaConfig(
        dataset=args.dataset,
        models=model_entries,
        judge=judge,
        n_instructions=args.n_instructions,
        language=getattr(args, "language", None),
        seed=getattr(args, "seed", 42),
        balance_by=getattr(args, "balance_by", None),
        criteria=getattr(args, "criteria", "default"),
        matchmaker=MatchmakerConfig(
            strategy=getattr(args, "matchmaker", "round_robin"),
            n_matches=getattr(args, "n_matches", None),
        ),
        generation_max_tokens=args.generation_max_tokens,
        truncate_input_chars=args.truncate_input_chars,
        ignore_cache=getattr(args, "ignore_cache", False),
        ignore_score_cache=getattr(args, "ignore_score_cache", False),
        bt_regularization=getattr(args, "bt_regularization", 0.01),
        elo_k=getattr(args, "elo_k", 32.0),
        include_completions=getattr(args, "include_completions", False),
        include_raw_judge=getattr(args, "include_raw_judge", False),
    )


def build_run_plan_from_env_and_args(args: argparse.Namespace) -> SlurmRunPlan:
    """Construct a typed SLURM run plan from CLI args plus environment variables."""
    _env_mod.load_dotenv()

    default_gpus = int(_env_mod.get("GPUS_PER_NODE", "1") or "1")
    task = _build_task_config(args, default_gpus=default_gpus)
    model_names = [model.name for model in getattr(task, "models", [])]
    challenger = getattr(task, "challenger", None)
    challenger_model = getattr(challenger, "name", None)
    if not model_names and challenger_model:
        model_names = [challenger_model]
    judge_model = getattr(getattr(task, "judge", None), "model", None)

    judge_local = (
        not needs_network(judge_model)
        if judge_model and judge_model != "none"
        else False
    )
    challenger_local = (
        not needs_network(challenger_model)
        if challenger_model
        else False
    )
    stage = getattr(args, "stage", "all")
    mode = getattr(args, "mode", "arena")
    any_local_model = any(not needs_network(model.name) for model in getattr(task, "models", []))
    run_annotate_phase = mode == "annotate"
    annotate_generation_needed = bool(
        run_annotate_phase
        and challenger_model
        and not getattr(challenger, "completions", None)
    )
    run_generate_phase = (
        mode in {"generate", "arena"} and not (mode == "arena" and stage == "analyze")
    ) or annotate_generation_needed
    run_judge_phase = mode in {"judge", "arena", "agreement", "annotate"}
    any_slurm_gen = (any_local_model and mode in {"generate", "arena"} and not (
        mode == "arena" and stage == "analyze"
    )) or (annotate_generation_needed and challenger_local)
    any_slurm_judge = judge_local and run_judge_phase
    any_slurm = any_slurm_gen or any_slurm_judge

    required: list[str] = ["USER_WORK_DIR"]
    if any_slurm:
        required += ["ACCOUNT", "PARTITION"]
    needed = []
    for var in required:
        cli_map = {
            "ACCOUNT": getattr(args, "account", None),
            "PARTITION": getattr(args, "partition", None),
        }
        if var in cli_map and cli_map[var] is not None:
            continue
        needed.append(var)
    _env_mod.check_required(needed)

    env_time_limit = _env_mod.get("TIME_LIMIT")
    needs_generate_time = any_slurm_gen
    needs_judge_time = any_slurm_judge
    time_missing_msgs: list[str] = []
    if needs_generate_time and getattr(args, "time_generate", None) is None and not env_time_limit:
        time_missing_msgs.append("generation jobs require --time_generate or $TIME_LIMIT")
    if needs_judge_time and getattr(args, "time_judge", None) is None and not env_time_limit:
        time_missing_msgs.append(
            "judge/agreement/annotate jobs require --time_judge or $TIME_LIMIT"
        )
    if time_missing_msgs:
        raise EnvironmentError(
            "Missing SLURM wall-time configuration:\n  - "
            + "\n  - ".join(time_missing_msgs)
        )

    partition = _cli_or_env(args.partition, "PARTITION") or ""
    account = _cli_or_env(args.account, "ACCOUNT") or ""
    time_generate = _cli_or_env(args.time_generate, "TIME_LIMIT") or ""
    time_judge = _cli_or_env(args.time_judge, "TIME_LIMIT") or ""
    container_runtime, container_image, container_home = resolve_container_settings(
        cli_runtime=getattr(args, "container_runtime", None),
        cli_image=getattr(args, "container_image", None),
        cli_home=getattr(args, "container_home", None),
        env_get=_env_mod.get,
    )

    user_work = _env_mod.require("USER_WORK_DIR")
    slurm_work = _env_mod.get(
        "SLURM_WORK_DIR", os.path.join(user_work, "slurm_jobs"),
    )

    auto_project = str(Path(__file__).resolve().parent.parent.parent)
    project_dir = _cli_or_env(
        args.project_dir,
        "OPENJURY_PROJECT_DIR",
        auto_project,
    )
    project_dir = normalize_project_dir(project_dir)

    tag = getattr(args, "tag", None) or datetime.now().strftime("%Y%m%d_%H%M%S")
    short_models = (
        "_".join(model.rsplit("/", 1)[-1][:15] for model in model_names[:3])
        if model_names
        else "unknown"
    )
    safe_name = f"{task.dataset}_{short_models}".replace("/", "_")
    work_dir = os.path.join(slurm_work, "openjury", safe_name, tag)

    execution = SlurmExecutionConfig(
        mode=mode,
        stage=stage,
        judge_local=judge_local,
        tag=tag,
        partition=partition,
        account=account,
        time_generate=time_generate,
        time_judge=time_judge,
        qos=getattr(args, "qos", "normal"),
        project_dir=project_dir,
        work_dir=work_dir,
        logs_dir="",
        container_runtime=container_runtime,
        container_image=container_image,
        container_home=container_home,
    )
    return SlurmRunPlan(task=task, execution=execution)


# ═════════════════════════════════════════════════════════════════════
#  Pipeline script generator
# ═════════════════════════════════════════════════════════════════════

# Mode planners and render/build helpers live in:
# ``openjury.slurm.modes``, ``openjury.slurm.render``,
# ``openjury.slurm.config_builders``, and ``openjury.slurm.submit_builders``.


def generate_pipeline_scripts(
    plan: SlurmRunPlan,
    output_dir: Path,
    on_missing_completions: str = "error",
) -> list[Path]:
    """Generate all SLURM scripts for the evaluation pipeline.

    Args:
        plan: Typed SLURM run plan.
        output_dir: Directory to write scripts into.
        on_missing_completions: ``"error"`` = abort if any model is
            missing cached completions (judge mode only).
            ``"skip"`` = warn and let ``arena_step`` resolve on the fly.

    Returns:
        List of generated script paths.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # SLURM log files (.out / .err) go in the same directory as the scripts
    plan.execution.logs_dir = str(output_dir)

    generated: list[Path] = []
    tag_suffix = f"_{plan.execution.tag[:12]}" if plan.execution.tag else ""

    def _write(path: Path, content: str) -> Path:
        path.write_text(content)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        generated.append(path)
        return path

    mode = plan.execution.mode

    if mode == "generate":
        slurm_modes.generate_mode_scripts(
            plan=plan,
            output_dir=output_dir,
            tag_suffix=tag_suffix,
            write_script=_write,
            job_factory=SlurmJobConfig,
        )
    elif mode == "judge":
        slurm_modes.judge_mode_scripts(
            plan=plan,
            output_dir=output_dir,
            tag_suffix=tag_suffix,
            write_script=_write,
            job_factory=SlurmJobConfig,
            on_missing_completions=on_missing_completions,
            generated_files=generated,
        )
    elif mode == "arena":
        slurm_modes.arena_mode_scripts(
            plan=plan,
            output_dir=output_dir,
            tag_suffix=tag_suffix,
            write_script=_write,
            job_factory=SlurmJobConfig,
            generated_files=generated,
        )
    elif mode == "agreement":
        slurm_modes.agreement_mode_scripts(
            plan=plan,
            output_dir=output_dir,
            tag_suffix=tag_suffix,
            write_script=_write,
            job_factory=SlurmJobConfig,
            generated_files=generated,
        )
    elif mode == "annotate":
        slurm_modes.annotate_mode_scripts(
            plan=plan,
            output_dir=output_dir,
            tag_suffix=tag_suffix,
            write_script=_write,
            job_factory=SlurmJobConfig,
            generated_files=generated,
        )
    else:
        raise ValueError(f"Unknown pipeline mode: {mode!r}")

    return generated


# ═════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═════════════════════════════════════════════════════════════════════


def main(argv: list[str] | None = None):
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

              # Agreement mode (datasets with inline completions + human prefs)
              uv run python -m openjury.slurm.generate_slurm \\
                --dataset lmsys \\
                --judge_model OpenRouter/qwen/qwen3-32b \\
                --mode agreement --n_instructions 200
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
    parser.add_argument(
        "--truncate_instruction", type=int, default=500,
        help="Truncate instruction text in saved outputs (task-specific). Default: 500.",
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
    parser.add_argument(
        "--container_runtime",
        choices=["none", "apptainer"],
        default=None,
        help=(
            "Container runtime for eligible compute-node jobs. "
            "Default: $OPENJURY_CONTAINER_RUNTIME or none."
        ),
    )
    parser.add_argument(
        "--container_image",
        default=None,
        help=(
            "Path to an Apptainer/Singularity image (.sif). "
            "Default: $OPENJURY_CONTAINER_IMAGE."
        ),
    )
    parser.add_argument(
        "--container_home",
        default=None,
        help=(
            "Persistent writable home used inside the container. "
            "Default: $OPENJURY_CONTAINER_HOME."
        ),
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
        choices=["generate", "arena", "judge", "agreement", "annotate"],
        default="arena",
        help="Pipeline mode: "
             "'arena' (default) = K generate jobs + 1 arena judge job. "
             "'generate' = generate completions only (no judge). "
             "'judge' = run arena judge only (assumes completions "
             "are already cached). "
             "'agreement' = human-vs-judge agreement on datasets with "
             "inline completions/human prefs. "
             "'annotate' = generic annotation from an annotate config. "
             "API judges run on login node (no SLURM job), GPU judges use sbatch. "
    )
    parser.add_argument(
        "--stage",
        choices=["all", "annotate", "analyze"],
        default="all",
        help=(
            "Evaluation stage for --mode arena/agreement: "
            "'all' (default), 'annotate', or 'analyze'. "
            "Not used for --mode generate/judge/annotate."
        ),
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
        "--detach",
        action="store_true",
        help=(
            "With --submit: run submit_all.sh in background and return immediately. "
            "Useful for long-running local/login-node orchestration scripts."
        ),
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

    args = parser.parse_args(argv)

    if args.detach and not args.submit:
        parser.error("--detach requires --submit.")
    if args.detach and args.remote:
        parser.error("--detach is not supported with --remote.")

    explicit_dests = slurm_config_resolver.collect_explicit_dests(parser, argv)
    slurm_config_resolver.resolve_args_from_config(
        args,
        explicit_dests=explicit_dests,
    )
    slurm_config_resolver.validate_mode_args(parser, args)

    # ── Build typed SLURM run plan ───────────────────────────────
    plan = build_run_plan_from_env_and_args(args)

    # ── Build a unique per-run subfolder ─────────────────────────
    run_name = f"{plan.dataset}_{args.mode}_{plan.execution.tag}".replace("/", "_")
    run_dir = (Path(args.output_dir) / run_name).resolve()

    # If output_dir was overridden by config, also redirect work_dir
    # so logs, results, and scripts all live together
    if args.output_dir != "slurm_scripts":
        plan.execution.work_dir = str(run_dir)


    scripts = generate_pipeline_scripts(
        plan=plan,
        output_dir=run_dir,
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
        gen_scripts = sorted(
            [
                s for s in scripts
                if s.suffix == ".sh" and "_generate_" in s.name
            ],
            key=lambda p: p.name,
        )
        judge_scripts = [
            s for s in scripts
            if "arena_judge" in s.name or "agreement" in s.name or "annotate" in s.name or "judge" in s.name
        ]
        judge_script = judge_scripts[0] if judge_scripts else None

        # Detect execution target per generated script from the rendered header.
        # This stays correct even when some generation steps were skipped due to cache.
        def _is_sbatch_script(script_path: Path) -> bool:
            try:
                with open(script_path, encoding="utf-8") as fh:
                    for _ in range(20):
                        line = fh.readline()
                        if not line:
                            break
                        if line.startswith("#SBATCH"):
                            return True
            except OSError:
                pass
            return False

        gen_locals = [_is_sbatch_script(s) for s in gen_scripts]

        submit_remote(
            script_dir=run_dir,
            gen_scripts=gen_scripts,
            gen_locals=gen_locals,
            judge_script=judge_script,
            judge_local=plan.execution.judge_local,
            cluster=cluster_name,
            project_dir=plan.execution.project_dir,
            remote_project_dir=args.remote_project_dir,
            wait=True,
            wait_timeout=args.wait_timeout,
        )
    elif args.submit:
        import subprocess

        if args.detach:
            log_file = run_dir / "submit_all.log"
            logger.info("Submitting jobs in background: bash %s", submit_script)
            with open(log_file, "w", encoding="utf-8") as log_fh:
                proc = subprocess.Popen(
                    ["bash", str(submit_script)],
                    cwd=str(run_dir),
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            logger.info("✅ Background PID: %d", proc.pid)
            logger.info("Monitor:  tail -f %s", log_file)
            logger.info("Check:    ps -p %d", proc.pid)
        else:
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
