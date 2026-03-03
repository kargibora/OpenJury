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

    # Then submit
    bash slurm_scripts/<run_dir>/submit_all.sh


"""

from __future__ import annotations

import argparse
import copy
import os
import stat
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from textwrap import dedent

from openjury._logging import logger
from openjury.cli_args import add_arena_pipeline_args
from openjury.models.utils import (
    is_local_provider,
    needs_network,
    provider_from_model,
)
from openjury.slurm import config_resolver as slurm_config_resolver
from openjury.slurm import modes as slurm_modes


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
    - ``agreement``: human-vs-judge agreement on inline-completion datasets.

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
    pairwise_prompt_style: str = "rubric"
    no_swap: bool = False  # disable position-swap debiasing
    ignore_cache: bool = False  # force regeneration even if completions are cached
    ignore_score_cache: bool = False  # force re-scoring even if judge scores are cached
    language: str | None = None  # dataset language/filter (where supported)
    seed: int = 42  # dataset sub-sampling seed (where supported)
    balance_by: str | None = None  # metadata field for balanced sub-sampling
    truncate_instruction: int = 500  # output display truncation (task-specific)
    stage: str = "all"  # evaluation stage for arena/agreement: all|annotate|analyze

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
    task_config_mode: str | None = None  # "arena"/"agreement"/"generate" when sourced from --config
    task_config_payload: dict | None = None  # original typed config payload for passthrough

    @property
    def dataset_options(self):
        """Shared dataset selection options for loaders/config builders."""
        from openjury.datasets.options import DatasetOptions
        return DatasetOptions(
            name=self.dataset,
            n_instructions=self.n_instructions,
            language=self.language,
            seed=self.seed,
            balance_by=self.balance_by,
        )

    @classmethod
    def from_env_and_args(
        cls,
        args: argparse.Namespace,
        *,
        task_payload: slurm_config_resolver.ResolvedTaskConfigPayload | None = None,
    ) -> PipelineConfig:
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
        stage = getattr(args, "stage", "all")
        mode = getattr(args, "mode", "arena")
        models = args.models or []
        any_local_model = any(not needs_network(m) for m in models)
        run_generate_phase = mode in {"generate", "arena"} and not (
            mode == "arena" and stage == "analyze"
        )
        run_judge_phase = mode in {"judge", "arena", "agreement"}
        any_slurm_gen = any_local_model and run_generate_phase
        any_slurm_judge = judge_local and run_judge_phase
        any_slurm = any_slurm_gen or any_slurm_judge

        # ── Validate required env vars (only those needed) ───────
        required: list[str] = ["USER_WORK_DIR"]
        if any_slurm:
            required += ["ACCOUNT", "PARTITION"]
        # Only check vars not already supplied via CLI
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

        # Validate per-job time sources (avoid producing scripts with empty --time=)
        env_time_limit = _env_mod.get("TIME_LIMIT")
        needs_generate_time = any_slurm_gen
        needs_judge_time = any_slurm_judge
        time_missing_msgs: list[str] = []
        if needs_generate_time and getattr(args, "time_generate", None) is None and not env_time_limit:
            time_missing_msgs.append("generation jobs require --time_generate or $TIME_LIMIT")
        if needs_judge_time and getattr(args, "time_judge", None) is None and not env_time_limit:
            time_missing_msgs.append("judge/agreement jobs require --time_judge or $TIME_LIMIT")
        if time_missing_msgs:
            raise EnvironmentError(
                "Missing SLURM wall-time configuration:\n  - "
                + "\n  - ".join(time_missing_msgs)
            )

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
            pairwise_prompt_style=getattr(args, "pairwise_prompt_style", "rubric"),
            no_swap=getattr(args, "no_swap", False),
            ignore_cache=getattr(args, "ignore_cache", False),
            ignore_score_cache=getattr(args, "ignore_score_cache", False),
            language=getattr(args, "language", None),
            seed=getattr(args, "seed", 42),
            balance_by=getattr(args, "balance_by", None),
            truncate_instruction=getattr(args, "truncate_instruction", 500),
            stage=stage,
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
            task_config_mode=(task_payload.kind if task_payload else None),
            task_config_payload=(
                copy.deepcopy(task_payload.payload)
                if task_payload and task_payload.payload is not None
                else None
            ),
        )


# ═════════════════════════════════════════════════════════════════════
#  Pipeline script generator
# ═════════════════════════════════════════════════════════════════════

# Mode planners and render/build helpers live in:
# ``openjury.slurm.modes``, ``openjury.slurm.render``,
# ``openjury.slurm.config_builders``, and ``openjury.slurm.submit_builders``.


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
              ``"arena"`` = K generate jobs + 1 arena judge job (default);
              ``"agreement"`` = one agreement evaluation job.
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
    tag_suffix = f"_{pipeline.tag[:12]}" if pipeline.tag else ""

    def _write(path: Path, content: str) -> Path:
        path.write_text(content)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        generated.append(path)
        return path

    if mode == "generate":
        slurm_modes.generate_mode_scripts(
            pipeline=pipeline,
            output_dir=output_dir,
            tag_suffix=tag_suffix,
            write_script=_write,
            job_factory=SlurmJobConfig,
        )
    elif mode == "judge":
        slurm_modes.judge_mode_scripts(
            pipeline=pipeline,
            output_dir=output_dir,
            tag_suffix=tag_suffix,
            write_script=_write,
            job_factory=SlurmJobConfig,
            on_missing_completions=on_missing_completions,
            generated_files=generated,
        )
    elif mode == "arena":
        slurm_modes.arena_mode_scripts(
            pipeline=pipeline,
            output_dir=output_dir,
            tag_suffix=tag_suffix,
            write_script=_write,
            job_factory=SlurmJobConfig,
            generated_files=generated,
        )
    elif mode == "agreement":
        slurm_modes.agreement_mode_scripts(
            pipeline=pipeline,
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
    parser.add_argument("--qos", default="normal")

    # ── Output ───────────────────────────────────────────────────
    parser.add_argument(
        "--tag", default=None,
        help="Run tag for unique work dirs and job names. "
             "Default: YYYYMMDD_HHMMSS timestamp.",
    )
    parser.add_argument(
        "--mode",
        choices=["generate", "arena", "judge", "agreement"],
        default="arena",
        help="Pipeline mode: "
             "'arena' (default) = K generate jobs + 1 arena judge job. "
             "'generate' = generate completions only (no judge). "
             "'judge' = run arena judge only (assumes completions "
             "are already cached). "
             "'agreement' = human-vs-judge agreement on datasets with "
             "inline completions/human prefs. "
             "API judges run on login node (no SLURM job), GPU judges use sbatch. "
    )
    parser.add_argument(
        "--stage",
        choices=["all", "annotate", "analyze"],
        default="all",
        help=(
            "Evaluation stage for --mode arena/agreement: "
            "'all' (default), 'annotate', or 'analyze'. "
            "Not used for --mode generate/judge."
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
    resolved_task_payload = slurm_config_resolver.resolve_args_from_config(
        args,
        explicit_dests=explicit_dests,
    )
    slurm_config_resolver.validate_mode_args(parser, args)

    # ── Build PipelineConfig ─────────────────────────────────────
    config = PipelineConfig.from_env_and_args(args, task_payload=resolved_task_payload)

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
        gen_scripts = sorted(
            [
                s for s in scripts
                if s.suffix == ".sh" and "_generate_" in s.name
            ],
            key=lambda p: p.name,
        )
        judge_scripts = [
            s for s in scripts
            if "arena_judge" in s.name or "agreement" in s.name or "judge" in s.name
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
            judge_local=config.judge_local,
            cluster=cluster_name,
            project_dir=config.project_dir,
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
