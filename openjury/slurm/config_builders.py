"""Config-file builders for SLURM-generated OpenJury runs."""

from __future__ import annotations

from typing import Any


def build_arena_config_dict(
    pipeline: Any,
    output_dir: str,
) -> dict:
    """Build an ArenaConfig JSON dict from SLURM pipeline fields."""
    from openjury.arena.config import ArenaConfig, JudgeConfig, MatchmakerConfig, ModelEntry

    ds_opts = pipeline.dataset_options

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
        dataset=ds_opts.name,
        n_instructions=ds_opts.n_instructions,
        language=ds_opts.language,
        seed=ds_opts.seed,
        balance_by=ds_opts.balance_by,
        models=model_entries,
        judge=JudgeConfig(
            model=pipeline.judge_model,
            gpus=pipeline.judge_gpus,
            mode=pipeline.judge_mode,
            pairwise_prompt_style=pipeline.pairwise_prompt_style,
            max_tokens=pipeline.judge_max_tokens,
            quantization=pipeline.judge_quantization,
            no_swap=pipeline.no_swap,
            provide_explanation=pipeline.provide_explanation,
            enable_thinking=pipeline.judge_enable_thinking,
            chat_template=pipeline.judge_chat_template,
            chat_template_file=pipeline.judge_chat_template_file,
        ),
        criteria=pipeline.criteria,
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


def build_agreement_config_dict(
    pipeline: Any,
    output_dir: str,
) -> dict:
    """Build an AgreementConfig JSON dict from SLURM pipeline fields."""
    from openjury.arena.config import AgreementConfig, JudgeConfig

    ds_opts = pipeline.dataset_options
    agreement_cfg = AgreementConfig(
        dataset=ds_opts.name,
        judge=JudgeConfig(
            model=pipeline.judge_model,
            gpus=pipeline.judge_gpus,
            mode=pipeline.judge_mode,
            pairwise_prompt_style=pipeline.pairwise_prompt_style,
            max_tokens=pipeline.judge_max_tokens,
            quantization=pipeline.judge_quantization,
            no_swap=pipeline.no_swap,
            provide_explanation=pipeline.provide_explanation,
            enable_thinking=pipeline.judge_enable_thinking,
            chat_template=pipeline.judge_chat_template,
            chat_template_file=pipeline.judge_chat_template_file,
        ),
        output_dir=output_dir,
        n_instructions=ds_opts.n_instructions,
        language=ds_opts.language,
        seed=ds_opts.seed,
        balance_by=ds_opts.balance_by,
        criteria=pipeline.criteria,
        ignore_score_cache=pipeline.ignore_score_cache,
        truncate_instruction=pipeline.truncate_instruction,
    )
    return agreement_cfg.to_dict()
