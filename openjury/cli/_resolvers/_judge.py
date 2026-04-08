"""Shared judge CLI helpers for arena/agreement resolvers."""

from __future__ import annotations

import argparse
from typing import Any

from openjury.arena.config import JudgeConfig
from openjury.resolution.argparse_overrides import apply_mapping_to_config

_JUDGE_ARG_TO_FIELD: tuple[tuple[str, str], ...] = (
    ("judge_model", "model"),
    ("judge_mode", "mode"),
    ("pairwise_prompt_style", "pairwise_prompt_style"),
    ("judge_max_tokens", "max_tokens"),
    ("judge_gpus", "gpus"),
    ("judge_quantization", "quantization"),
    ("chat_template", "chat_template"),
    ("chat_template_file", "chat_template_file"),
    ("max_model_len", "max_model_len"),
    ("judge_temperature", "temperature"),
    ("judge_top_p", "top_p"),
    ("judge_n_trials", "n_trials"),
)


def _arg_or_fallback(
    args: argparse.Namespace,
    primary: str,
    fallback: str | None = None,
) -> Any:
    """Read a CLI namespace attr with an optional fallback attr name."""
    value = getattr(args, primary, None)
    if value is None and fallback:
        value = getattr(args, fallback, None)
    return value


def judge_cli_override_mapping() -> list[tuple[str, Any]]:
    """Return CLI override mapping entries for judge fields."""
    return [
        (
            dest,
            lambda c, v, field_name=field_name: setattr(c.judge, field_name, v),
        )
        for dest, field_name in _JUDGE_ARG_TO_FIELD
    ]


def apply_judge_cli_overrides(
    config: Any,
    args: argparse.Namespace,
    explicit: set[str],
    *,
    gen_kwargs: dict[str, Any],
) -> None:
    """Apply explicit judge-related CLI overrides onto a typed config."""
    apply_mapping_to_config(
        config,
        args,
        explicit,
        judge_cli_override_mapping(),
    )

    if "no_swap" in explicit and args.no_swap:
        config.judge.no_swap = True
    if "provide_explanation" in explicit and args.provide_explanation:
        config.judge.provide_explanation = True
    if "enable_thinking" in explicit and args.enable_thinking:
        config.judge.enable_thinking = True
    if "enforce_eager" in explicit and args.enforce_eager:
        config.judge.enforce_eager = True
    if "gen_kwargs" in explicit and gen_kwargs:
        config.judge.generation_kwargs = gen_kwargs


def judge_from_args(
    args: argparse.Namespace,
    *,
    gen_kwargs: dict[str, Any],
) -> JudgeConfig:
    """Construct a JudgeConfig from parsed CLI args."""
    return JudgeConfig(
        model=_arg_or_fallback(args, "judge_model") or "none",
        gpus=_arg_or_fallback(args, "judge_gpus"),
        mode=_arg_or_fallback(args, "judge_mode"),
        pairwise_prompt_style=_arg_or_fallback(args, "pairwise_prompt_style"),
        max_tokens=_arg_or_fallback(args, "judge_max_tokens"),
        quantization=_arg_or_fallback(args, "judge_quantization"),
        chat_template=_arg_or_fallback(args, "chat_template"),
        chat_template_file=_arg_or_fallback(args, "chat_template_file"),
        no_swap=bool(_arg_or_fallback(args, "no_swap")),
        provide_explanation=bool(_arg_or_fallback(args, "provide_explanation")),
        enable_thinking=True if _arg_or_fallback(args, "enable_thinking") else None,
        max_model_len=_arg_or_fallback(args, "max_model_len", "judge_max_model_len"),
        enforce_eager=bool(_arg_or_fallback(args, "enforce_eager", "judge_enforce_eager")),
        temperature=_arg_or_fallback(args, "judge_temperature") or 0.0,
        top_p=_arg_or_fallback(args, "judge_top_p") or 1.0,
        n_trials=_arg_or_fallback(args, "judge_n_trials") or 1,
        generation_kwargs=gen_kwargs or {},
    )
