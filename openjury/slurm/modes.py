"""Mode-specific SLURM script planning helpers.

These functions orchestrate per-mode script generation using injected file
writers and job factories, keeping ``generate_slurm.py`` focused on CLI/env
parsing and high-level dispatch.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable

from openjury._logging import logger
from openjury.slurm import config_builders as slurm_config_builders
from openjury.slurm import render as slurm_render
from openjury.slurm import submit_builders as slurm_submit_builders

WriteScriptFn = Callable[[Path, str], Path]
JobFactoryFn = Callable[..., Any]


def _task_config_json(
    pipeline: Any,
    *,
    kind: str,
    results_output_dir: str,
    fallback_builder: Callable[[Any, str], dict],
) -> dict:
    """Return the task config JSON to write for a SLURM mode.

    If a typed config payload was supplied via ``--config``, preserve it and only
    override ``output_dir`` to the SLURM-managed results directory.
    """
    payload_kind = getattr(pipeline, "task_config_mode", None)
    payload = getattr(pipeline, "task_config_payload", None)
    if payload_kind == kind and isinstance(payload, dict):
        data = copy.deepcopy(payload)
        data["output_dir"] = results_output_dir
        return data
    return fallback_builder(pipeline, results_output_dir)


def generate_mode_scripts(
    *,
    pipeline: Any,
    output_dir: Path,
    tag_suffix: str,
    write_script: WriteScriptFn,
    job_factory: JobFactoryFn,
) -> None:
    """Generate scripts for ``--mode generate``."""
    from openjury.cache.completions import cache as _comp_cache

    gen_paths: list[Path] = []
    gen_locals: list[bool] = []
    for idx, model in enumerate(pipeline.models):
        gpus = pipeline.model_gpus[idx] if idx < len(pipeline.model_gpus) else 1
        quant = (
            pipeline.model_quantizations[idx]
            if idx < len(pipeline.model_quantizations)
            else None
        )
        is_local = (
            pipeline.model_local[idx] if idx < len(pipeline.model_local) else True
        )

        if not pipeline.ignore_cache and _comp_cache.exists(
            model=model,
            dataset=pipeline.dataset,
            n=pipeline.n_instructions,
        ):
            logger.info("  ⏭ %s already cached — skipping", model)
            continue

        short = model.rsplit("/", 1)[-1].replace("/", "_")[:20]
        comp_out = f"{pipeline.work_dir}/completions_{idx+1}_{short}.parquet"
        job = job_factory(
            job_name=f"oj_gen{idx+1}{tag_suffix}",
            n_gpus=gpus,
            partition=pipeline.partition,
            account=pipeline.account,
            time=pipeline.time_generate,
            qos=pipeline.qos,
        )
        path = write_script(
            output_dir / f"{idx+1:02d}_generate_{short}.sh",
            slurm_render._generate_step_script(
                model,
                comp_out,
                pipeline,
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
            len(pipeline.models),
        )
        for m in pipeline.models:
            logger.info("     • %s", m)
        return

    write_script(
        output_dir / "submit_all.sh",
        slurm_submit_builders.build_generate_submit_all(
            pipeline,
            gen_paths,
            gen_locals,
            tag_suffix,
        ),
    )


def judge_mode_scripts(
    *,
    pipeline: Any,
    output_dir: Path,
    tag_suffix: str,
    write_script: WriteScriptFn,
    job_factory: JobFactoryFn,
    on_missing_completions: str,
    generated_files: list[Path],
) -> None:
    """Generate scripts for ``--mode judge`` (arena judge only)."""
    from openjury.cache.completions import cache as _comp_cache

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
            len(pipeline.models),
        )

    results = f"{pipeline.work_dir}/results"

    import json as _json

    config_dict = _task_config_json(
        pipeline,
        kind="arena",
        results_output_dir=results,
        fallback_builder=slurm_config_builders.build_arena_config_dict,
    )
    config_path = output_dir / "arena_config.json"
    config_path.write_text(_json.dumps(config_dict, indent=2), encoding="utf-8")
    generated_files.append(config_path)

    job_judge = job_factory(
        job_name=f"oj_arena{tag_suffix}",
        n_gpus=pipeline.judge_gpus,
        partition=pipeline.partition,
        account=pipeline.account,
        time=pipeline.time_judge,
        qos=pipeline.qos,
    )
    path_judge = write_script(
        output_dir / "01_arena_judge.sh",
        slurm_render._arena_step_script(
            pipeline,
            job_judge,
            results,
            config_path=str(config_path),
            local=pipeline.judge_local,
        ),
    )

    write_script(
        output_dir / "submit_all.sh",
        slurm_submit_builders.build_judge_submit_all(
            pipeline,
            path_judge,
            results,
            tag_suffix,
        ),
    )


def arena_mode_scripts(
    *,
    pipeline: Any,
    output_dir: Path,
    tag_suffix: str,
    write_script: WriteScriptFn,
    job_factory: JobFactoryFn,
    generated_files: list[Path],
) -> None:
    """Generate scripts for ``--mode arena``."""
    from openjury.cache.completions import cache as _comp_cache
    from openjury.cache.scores import score_cache as _score_cache

    results = f"{pipeline.work_dir}/results"
    gen_paths: list[Path] = []
    gen_locals: list[bool] = []

    models_cached: list[str] = []
    models_need_gen: list[tuple[int, str]] = []

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
        for _, m in models_need_gen:
            logger.info("     • %s", m)
    else:
        logger.info("  ✅ All completions cached — only judge job will be created")

    if not pipeline.ignore_score_cache and not models_need_gen:
        if all(
            _score_cache.exists(
                judge=pipeline.judge_model,
                rubric=pipeline.rubric,
                model=model,
                dataset=pipeline.dataset,
                n=pipeline.n_instructions,
            )
            for model in pipeline.models
        ):
            logger.info(
                "  ✅ All judge scores also cached — arena step will only compute ratings"
            )

    for seq, (orig_idx, model) in enumerate(models_need_gen):
        gpus = pipeline.model_gpus[orig_idx] if orig_idx < len(pipeline.model_gpus) else 1
        quant = (
            pipeline.model_quantizations[orig_idx]
            if orig_idx < len(pipeline.model_quantizations)
            else None
        )
        is_local = (
            pipeline.model_local[orig_idx] if orig_idx < len(pipeline.model_local) else True
        )

        short = model.rsplit("/", 1)[-1].replace("/", "_")[:20]
        comp_out = f"{pipeline.work_dir}/completions_{orig_idx+1}_{short}.parquet"

        job = job_factory(
            job_name=f"oj_gen{seq+1}{tag_suffix}",
            n_gpus=gpus,
            partition=pipeline.partition,
            account=pipeline.account,
            time=pipeline.time_generate,
            qos=pipeline.qos,
        )
        path = write_script(
            output_dir / f"{seq+1:02d}_generate_{short}.sh",
            slurm_render._generate_step_script(
                model,
                comp_out,
                pipeline,
                job,
                quantization=quant,
                local=is_local,
            ),
        )
        gen_paths.append(path)
        gen_locals.append(is_local)

    import json as _json

    config_dict = _task_config_json(
        pipeline,
        kind="arena",
        results_output_dir=results,
        fallback_builder=slurm_config_builders.build_arena_config_dict,
    )
    config_path = output_dir / "arena_config.json"
    config_path.write_text(_json.dumps(config_dict, indent=2), encoding="utf-8")
    generated_files.append(config_path)

    judge_idx = len(gen_paths) + 1
    job_judge = job_factory(
        job_name=f"oj_arena{tag_suffix}",
        n_gpus=pipeline.judge_gpus,
        partition=pipeline.partition,
        account=pipeline.account,
        time=pipeline.time_judge,
        qos=pipeline.qos,
    )
    path_judge = write_script(
        output_dir / f"{judge_idx:02d}_arena_judge.sh",
        slurm_render._arena_step_script(
            pipeline,
            job_judge,
            results,
            config_path=str(config_path),
            local=pipeline.judge_local,
        ),
    )

    write_script(
        output_dir / "submit_all.sh",
        slurm_submit_builders.build_arena_submit_all(
            pipeline,
            gen_paths,
            gen_locals,
            path_judge,
            tag_suffix,
            results,
            models_cached=models_cached,
        ),
    )


def agreement_mode_scripts(
    *,
    pipeline: Any,
    output_dir: Path,
    tag_suffix: str,
    write_script: WriteScriptFn,
    job_factory: JobFactoryFn,
    generated_files: list[Path],
) -> None:
    """Generate scripts for ``--mode agreement``."""
    results = f"{pipeline.work_dir}/results"

    import json as _json

    config_dict = _task_config_json(
        pipeline,
        kind="agreement",
        results_output_dir=results,
        fallback_builder=slurm_config_builders.build_agreement_config_dict,
    )
    config_path = output_dir / "agreement_config.json"
    config_path.write_text(_json.dumps(config_dict, indent=2), encoding="utf-8")
    generated_files.append(config_path)

    job_agreement = job_factory(
        job_name=f"oj_agree{tag_suffix}",
        n_gpus=pipeline.judge_gpus,
        partition=pipeline.partition,
        account=pipeline.account,
        time=pipeline.time_judge,
        qos=pipeline.qos,
    )
    path_agreement = write_script(
        output_dir / "01_agreement.sh",
        slurm_render._agreement_step_script(
            pipeline,
            job_agreement,
            results,
            config_path=str(config_path),
            local=pipeline.judge_local,
        ),
    )

    write_script(
        output_dir / "submit_all.sh",
        slurm_submit_builders.build_agreement_submit_all(
            pipeline,
            path_agreement,
            results,
            tag_suffix,
        ),
    )
