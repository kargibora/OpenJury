"""Mode-specific SLURM script planning helpers.

These functions orchestrate per-mode script generation using injected file
writers and job factories, keeping ``generate_slurm.py`` focused on CLI/env
parsing and high-level dispatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from openjury.annotate_config import AnnotateConfig
from openjury.arena.config import AgreementConfig, ArenaConfig, ModelEntry
from openjury._logging import logger
from openjury.models.utils import needs_network
from openjury.slurm import config_builders as slurm_config_builders
from openjury.slurm import render as slurm_render
from openjury.slurm import submit_builders as slurm_submit_builders
from openjury.slurm.plans import GenerateTaskConfig, SlurmRunPlan

WriteScriptFn = Callable[[Path, str], Path]
JobFactoryFn = Callable[..., Any]


def _task_config_json(
    task: ArenaConfig | AgreementConfig | AnnotateConfig,
    results_output_dir: str,
) -> dict:
    """Return the task config JSON to write for a SLURM mode.
    """
    if isinstance(task, ArenaConfig):
        return slurm_config_builders.build_arena_config_dict(task, results_output_dir)
    if isinstance(task, AgreementConfig):
        return slurm_config_builders.build_agreement_config_dict(task, results_output_dir)
    if isinstance(task, AnnotateConfig):
        return slurm_config_builders.build_annotate_config_dict(task, results_output_dir)
    raise TypeError("Only arena/agreement/annotate tasks can be serialized to config JSON.")


def generate_mode_scripts(
    *,
    plan: SlurmRunPlan,
    output_dir: Path,
    tag_suffix: str,
    write_script: WriteScriptFn,
    job_factory: JobFactoryFn,
) -> None:
    """Generate scripts for ``--mode generate``."""
    from openjury.cache.completions import cache as _comp_cache

    task = plan.task
    if not isinstance(task, GenerateTaskConfig):
        raise TypeError("Generate mode requires a GenerateTaskConfig task.")

    execution = plan.execution
    cache_dataset = task.dataset_options.cache_key()
    gen_paths: list[Path] = []
    gen_locals: list[bool] = []
    for idx, model_entry in enumerate(task.models):
        model = model_entry.name
        gpus = model_entry.gpus
        quant = model_entry.quantization
        is_local = not needs_network(model)

        if not task.ignore_cache and _comp_cache.exists(
            model=model,
            dataset=cache_dataset,
            n=task.n_instructions,
        ):
            logger.info("  ⏭ %s already cached — skipping", model)
            continue

        short = model.rsplit("/", 1)[-1].replace("/", "_")[:20]
        comp_out = f"{execution.work_dir}/completions_{idx+1}_{short}.parquet"
        job = job_factory(
            job_name=f"oj_gen{idx+1}{tag_suffix}",
            n_gpus=gpus,
            partition=execution.partition,
            account=execution.account,
            time=execution.time_generate,
            qos=execution.qos,
        )
        path = write_script(
            output_dir / f"{idx+1:02d}_generate_{short}.sh",
            slurm_render._generate_step_script(
                model,
                comp_out,
                task,
                execution,
                job,
                quantization=quant,
                local=is_local,
            ),
        )
        gen_paths.append(path)
        gen_locals.append(is_local)

    if not gen_paths:
        logger.info(
            "  ✅ All %d model(s) already cached — nothing to generate",
            len(task.models),
        )
        for model_entry in task.models:
            logger.info("     • %s", model_entry.name)
        return

    write_script(
        output_dir / "submit_all.sh",
        slurm_submit_builders.build_generate_submit_all(
            plan,
            gen_paths,
            gen_locals,
            tag_suffix,
        ),
    )


def judge_mode_scripts(
    *,
    plan: SlurmRunPlan,
    output_dir: Path,
    tag_suffix: str,
    write_script: WriteScriptFn,
    job_factory: JobFactoryFn,
    on_missing_completions: str,
    generated_files: list[Path],
) -> None:
    """Generate scripts for ``--mode judge`` (arena judge only)."""
    from openjury.cache.completions import cache as _comp_cache

    task = plan.task
    if not isinstance(task, ArenaConfig):
        raise TypeError("Judge mode requires an ArenaConfig task.")

    execution = plan.execution
    cache_dataset = task.dataset_options.cache_key()
    missing: list[str] = []
    for model_entry in task.models:
        model = model_entry.name
        if not _comp_cache.exists(
            model=model,
            dataset=cache_dataset,
            n=task.n_instructions,
        ):
            missing.append(model)

    if missing:
        msg = (
            f"{len(missing)} model(s) have no cached completions "
            f"for dataset={task.dataset!r}:\n"
            + "\n".join(f"  ✗ {m}" for m in missing)
        )
        if on_missing_completions == "error":
            raise SystemExit(
                f"ERROR: {msg}\n\n"
                "Generate completions first (--mode generate) or "
                "re-run with --on_missing_completions skip to let "
                "arena_step resolve on the fly."
            )
        logger.warning(
            "⚠️  %s\n   Proceeding anyway — arena_step will "
            "attempt to generate on the fly.",
            msg,
        )
    else:
        logger.info(
            "  ✅ All %d model(s) have cached completions",
            len(task.models),
        )

    results = f"{execution.work_dir}/results"

    import json as _json

    config_dict = _task_config_json(task, results)
    config_path = output_dir / "arena_config.json"
    config_path.write_text(_json.dumps(config_dict, indent=2), encoding="utf-8")
    generated_files.append(config_path)

    job_judge = job_factory(
        job_name=f"oj_arena{tag_suffix}",
        n_gpus=task.judge.gpus,
        partition=execution.partition,
        account=execution.account,
        time=execution.time_judge,
        qos=execution.qos,
    )
    path_judge = write_script(
        output_dir / "01_arena_judge.sh",
        slurm_render._arena_step_script(
            plan,
            job_judge,
            results,
            config_path=str(config_path),
            stage="all",
            local=execution.judge_local,
        ),
    )

    write_script(
        output_dir / "submit_all.sh",
        slurm_submit_builders.build_judge_submit_all(
            plan,
            path_judge,
            results,
            tag_suffix,
        ),
    )


def arena_mode_scripts(
    *,
    plan: SlurmRunPlan,
    output_dir: Path,
    tag_suffix: str,
    write_script: WriteScriptFn,
    job_factory: JobFactoryFn,
    generated_files: list[Path],
) -> None:
    """Generate scripts for ``--mode arena``."""
    from openjury.cache.completions import cache as _comp_cache
    from openjury.cache.scores import score_cache as _score_cache

    task = plan.task
    if not isinstance(task, ArenaConfig):
        raise TypeError("Arena mode requires an ArenaConfig task.")

    execution = plan.execution
    stage = execution.stage
    results = f"{execution.work_dir}/results"
    cache_dataset = task.dataset_options.cache_key()
    gen_paths: list[Path] = []
    gen_locals: list[bool] = []

    if stage == "analyze":
        import json as _json

        config_dict = _task_config_json(task, results)
        config_path = output_dir / "arena_config.json"
        config_path.write_text(_json.dumps(config_dict, indent=2), encoding="utf-8")
        generated_files.append(config_path)

        job_analyze = job_factory(
            job_name=f"oj_arena_analyze{tag_suffix}",
            n_gpus=0,
            partition=execution.partition,
            account=execution.account,
            time=execution.time_judge,
            qos=execution.qos,
        )
        path_analyze = write_script(
            output_dir / "01_arena_analyze.sh",
            slurm_render._arena_step_script(
                plan,
                job_analyze,
                results,
                config_path=str(config_path),
                stage=stage,
                local=execution.judge_local,
            ),
        )
        write_script(
            output_dir / "submit_all.sh",
            slurm_submit_builders.build_arena_submit_all(
                plan,
                gen_paths=[],
                gen_locals=[],
                path_judge=path_analyze,
                tag_suffix=tag_suffix,
                results_dir=results,
                models_cached=[],
                stage=stage,
            ),
        )
        return

    models_cached: list[str] = []
    models_need_gen: list[tuple[int, Any]] = []

    for i, model_entry in enumerate(task.models):
        model = model_entry.name
        if (
            not task.ignore_cache
            and _comp_cache.exists(
                model=model,
                dataset=cache_dataset,
                n=task.n_instructions,
            )
        ):
            models_cached.append(model)
        else:
            models_need_gen.append((i, model_entry))

    if models_cached:
        logger.info(
            "  ✅ %d model(s) have cached completions — skipping generation:",
            len(models_cached),
        )
        for m in models_cached:
            logger.info("     • %s", m)

    if models_need_gen:
        logger.info(
            "  🔧 %d model(s) need generation:",
            len(models_need_gen),
        )
        for _, model_entry in models_need_gen:
            logger.info("     • %s", model_entry.name)
    else:
        logger.info("  ✅ All completions cached — only judge job will be created")

    if not task.ignore_score_cache and not models_need_gen:
        if all(
            _score_cache.exists(
                judge=task.judge.model,
                criteria=task.criteria,
                model=model_entry.name,
                dataset=cache_dataset,
                n=task.n_instructions,
            )
            for model_entry in task.models
        ):
            logger.info(
                "  ✅ All judge scores also cached — arena step will only compute ratings"
            )

    for seq, (orig_idx, model_entry) in enumerate(models_need_gen):
        model = model_entry.name
        gpus = model_entry.gpus
        quant = model_entry.quantization
        is_local = not needs_network(model)

        short = model.rsplit("/", 1)[-1].replace("/", "_")[:20]
        comp_out = f"{execution.work_dir}/completions_{orig_idx+1}_{short}.parquet"

        job = job_factory(
            job_name=f"oj_gen{seq+1}{tag_suffix}",
            n_gpus=gpus,
            partition=execution.partition,
            account=execution.account,
            time=execution.time_generate,
            qos=execution.qos,
        )
        path = write_script(
            output_dir / f"{seq+1:02d}_generate_{short}.sh",
            slurm_render._generate_step_script(
                model,
                comp_out,
                task,
                execution,
                job,
                quantization=quant,
                local=is_local,
            ),
        )
        gen_paths.append(path)
        gen_locals.append(is_local)

    import json as _json

    config_dict = _task_config_json(task, results)
    config_path = output_dir / "arena_config.json"
    config_path.write_text(_json.dumps(config_dict, indent=2), encoding="utf-8")
    generated_files.append(config_path)

    judge_idx = len(gen_paths) + 1
    job_judge = job_factory(
        job_name=f"oj_arena{tag_suffix}",
        n_gpus=task.judge.gpus,
        partition=execution.partition,
        account=execution.account,
        time=execution.time_judge,
        qos=execution.qos,
    )
    path_judge = write_script(
        output_dir / f"{judge_idx:02d}_arena_judge.sh",
        slurm_render._arena_step_script(
            plan,
            job_judge,
            results,
            config_path=str(config_path),
            stage=stage,
            local=execution.judge_local,
        ),
    )

    write_script(
        output_dir / "submit_all.sh",
        slurm_submit_builders.build_arena_submit_all(
            plan,
            gen_paths,
            gen_locals,
            path_judge,
            tag_suffix,
            results,
            models_cached=models_cached,
            stage=stage,
        ),
    )


def agreement_mode_scripts(
    *,
    plan: SlurmRunPlan,
    output_dir: Path,
    tag_suffix: str,
    write_script: WriteScriptFn,
    job_factory: JobFactoryFn,
    generated_files: list[Path],
) -> None:
    """Generate scripts for ``--mode agreement``."""
    task = plan.task
    if not isinstance(task, AgreementConfig):
        raise TypeError("Agreement mode requires an AgreementConfig task.")

    execution = plan.execution
    stage = execution.stage
    results = f"{execution.work_dir}/results"

    import json as _json

    config_dict = _task_config_json(task, results)
    config_path = output_dir / "agreement_config.json"
    config_path.write_text(_json.dumps(config_dict, indent=2), encoding="utf-8")
    generated_files.append(config_path)

    job_agreement = job_factory(
        job_name=f"oj_agree{tag_suffix}",
        n_gpus=task.judge.gpus,
        partition=execution.partition,
        account=execution.account,
        time=execution.time_judge,
        qos=execution.qos,
    )
    path_agreement = write_script(
        output_dir / "01_agreement.sh",
        slurm_render._agreement_step_script(
            plan,
            job_agreement,
            results,
            config_path=str(config_path),
            stage=stage,
            local=execution.judge_local,
        ),
    )

    write_script(
        output_dir / "submit_all.sh",
        slurm_submit_builders.build_agreement_submit_all(
            plan,
            path_agreement,
            results,
            tag_suffix,
            stage=stage,
        ),
    )


def annotate_mode_scripts(
    *,
    plan: SlurmRunPlan,
    output_dir: Path,
    tag_suffix: str,
    write_script: WriteScriptFn,
    job_factory: JobFactoryFn,
    generated_files: list[Path],
) -> None:
    """Generate scripts for ``--mode annotate``."""
    task = plan.task
    if not isinstance(task, AnnotateConfig):
        raise TypeError("Annotate mode requires an AnnotateConfig task.")

    execution = plan.execution
    results = f"{execution.work_dir}/results"
    challenger = task.challenger
    challenger_requires_generation = bool(
        challenger is not None and challenger.name and not challenger.completions
    )
    challenger_local = bool(
        challenger_requires_generation
        and challenger is not None
        and not needs_network(challenger.name)
    )

    gen_paths: list[Path] = []
    gen_locals: list[bool] = []
    task_for_annotate = AnnotateConfig.from_dict(task.to_dict())

    if challenger_requires_generation and challenger is not None:
        short = challenger.name.rsplit("/", 1)[-1].replace("/", "_")[:20]
        comp_out = f"{execution.work_dir}/challenger_completions_{short}.parquet"
        generate_task = GenerateTaskConfig(
            dataset=task.dataset,
            models=[challenger],
            n_instructions=task.n_instructions,
            language=task.language,
            seed=task.seed,
            balance_by=task.balance_by,
            generation_max_tokens=task.generation.max_tokens,
            truncate_input_chars=task.generation.truncate_input_chars,
            ignore_cache=task.generation.ignore_cache,
        )
        job_generate = job_factory(
            job_name=f"oj_annotate_gen{tag_suffix}",
            n_gpus=challenger.gpus,
            partition=execution.partition,
            account=execution.account,
            time=execution.time_generate,
            qos=execution.qos,
        )
        path_generate = write_script(
            output_dir / "01_generate_challenger.sh",
            slurm_render._generate_step_script(
                challenger.name,
                comp_out,
                generate_task,
                execution,
                job_generate,
                quantization=challenger.quantization,
                local=challenger_local,
            ),
        )
        gen_paths.append(path_generate)
        gen_locals.append(challenger_local)
        task_for_annotate.challenger = ModelEntry.from_raw(challenger.to_dict())
        task_for_annotate.challenger.completions = comp_out

    import json as _json

    config_dict = _task_config_json(task_for_annotate, results)
    config_path = output_dir / "annotate_config.json"
    config_path.write_text(_json.dumps(config_dict, indent=2), encoding="utf-8")
    generated_files.append(config_path)

    job_annotate = job_factory(
        job_name=f"oj_annotate{tag_suffix}",
        n_gpus=(task.judge.gpus if execution.judge_local else 0),
        partition=execution.partition,
        account=execution.account,
        time=execution.time_judge,
        qos=execution.qos,
    )
    path_annotate = write_script(
        output_dir / f"{len(gen_paths)+1:02d}_annotate.sh",
        slurm_render._annotate_step_script(
            SlurmRunPlan(task=task_for_annotate, execution=execution),
            job_annotate,
            results,
            config_path=str(config_path),
            local=execution.judge_local,
        ),
    )

    write_script(
        output_dir / "submit_all.sh",
        slurm_submit_builders.build_annotate_submit_all(
            plan,
            gen_paths,
            gen_locals,
            path_annotate,
            results,
            tag_suffix,
            task_for_annotate=task_for_annotate,
        ),
    )
