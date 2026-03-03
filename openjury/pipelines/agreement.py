"""Agreement pipeline — compare human preferences with LLM judge preferences.

Loads a dataset that provides inline completions and human preference labels
(e.g. ``lmsys``, ``comparia``), runs an LLM judge on the same completion
pairs, and saves per-sample results for downstream analysis.

No generation is performed in this pipeline; it only judges existing
completions embedded in the dataset.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from openjury._logging import logger
from openjury.arena.config import AgreementConfig
from openjury.common.pair_annotation import (
    PairSample,
    PairwiseCacheConfig,
    SamplewiseCacheConfig,
    score_pairs_pairwise,
    score_pairs_samplewise,
)
from openjury.datasets import load_dataset as load_eval_dataset
from openjury.models.factory import make_model
from openjury.analysis.agreement import (
    analyze_agreement_annotations as analyze_agreement_stage,
)
from openjury.pipelines.annotate import (
    load_agreement_annotations,
    save_agreement_annotations,
)
from openjury.rubrics import RubricScorer, get_rubric


def _pref_to_label(pref: float) -> str:
    """Convert numeric preference to a categorical label."""
    if pref < 0.25:
        return "A"
    if pref > 0.75:
        return "B"
    return "tie"


def _agreement_cache_dataset_key(config: AgreementConfig, *, exact: bool) -> str:
    """Build a dataset cache namespace for agreement scoring."""
    parts = [config.dataset, "agreement"]
    if exact:
        if config.language:
            parts.append(f"lang_{config.language}")
        if config.balance_by:
            parts.append(f"balance_{config.balance_by}")
        if config.n_instructions is not None:
            parts.append(f"seed_{config.seed}")
    return "__".join(parts)


def run_agreement(config: AgreementConfig, stage: str = "all") -> dict[str, Any]:
    """Run the agreement pipeline with stage control."""
    if stage not in {"all", "annotate", "analyze"}:
        raise ValueError(f"Unsupported agreement stage: {stage}")
    if stage == "analyze":
        logger.info(
            "Agreement analyze stage: loading saved annotations only (no model inference)."
        )
        annotation_data = load_agreement_annotations(config.output_dir)
        return analyze_agreement_stage(config=config, annotation_data=annotation_data)

    judge_cfg = config.judge
    dataset_name = config.dataset
    rubric_name = config.rubric
    judge_mode = judge_cfg.mode
    output_dir = config.output_dir
    ignore_score_cache = config.ignore_score_cache
    truncate_instruction = config.truncate_instruction

    ds_opts = config.dataset_options
    cache_dataset = ds_opts.cache_key()
    dataset = load_eval_dataset(
        ds_opts.name,
        n=ds_opts.n_instructions,
        **ds_opts.loader_kwargs(),
    )
    logger.info("Loaded dataset '%s': %d samples", dataset_name, len(dataset))

    valid = [
        s for s in dataset.samples
        if s.completions and len(s.completions) >= 2 and s.human_pref is not None
    ]
    if not valid:
        raise ValueError(
            f"No samples with inline completions and human_pref in '{dataset_name}'. "
            "This step requires datasets like 'lmsys' or 'comparia' that provide both."
        )
    logger.info(
        "  %d/%d samples have completions + human preferences",
        len(valid),
        len(dataset),
    )

    rubric = get_rubric(rubric_name)
    judge_model_config = judge_cfg.to_model_config()
    model = make_model(judge_cfg.model, config=judge_model_config)
    scorer = RubricScorer(
        judge_model=model,
        rubric=rubric,
        provide_explanation=judge_cfg.provide_explanation,
        pairwise_prompt_style=judge_cfg.pairwise_prompt_style,
    )
    dimension_weights = {d.name: d.weight for d in rubric.dimensions}

    logger.info(
        "Judge: %s | Mode: %s | Rubric: %s (%d dims) | Swap: %s",
        judge_cfg.model,
        judge_mode,
        rubric_name,
        rubric.k,
        "off" if judge_cfg.no_swap else "on",
    )

    pairs: list[PairSample] = []
    for i, sample in enumerate(valid):
        model_names = list(sample.completions.keys())
        ma, mb = model_names[0], model_names[1]
        pairs.append(
            PairSample(
                sample_id=sample.instruction_id,
                instruction_index=i,
                instruction=sample.instruction,
                model_a=ma,
                model_b=mb,
                completion_a=sample.completions[ma],
                completion_b=sample.completions[mb],
            )
        )

    n_total = len(pairs)
    logger.info("Judging %d pairs …", n_total)

    cache_dataset_exact = _agreement_cache_dataset_key(config, exact=True)
    cache_dataset_base = _agreement_cache_dataset_key(config, exact=False)

    if judge_mode == "pairwise":
        pair_key = (
            f"__agreement_pairwise__style_{judge_cfg.pairwise_prompt_style}"
            f"__swap_{'off' if judge_cfg.no_swap else 'on'}"
        )
        judgements = score_pairs_pairwise(
            scorer=scorer,
            pairs=pairs,
            swap_to_debias=not judge_cfg.no_swap,
            use_tqdm=True,
            cache_config=PairwiseCacheConfig(
                judge=judge_cfg.model,
                rubric=rubric_name,
                model_key=pair_key,
                dataset_exact=cache_dataset_exact,
                n_instructions=config.n_instructions,
                dataset_base=cache_dataset_base,
            ),
            ignore_cache=ignore_score_cache,
        )

    else:
        judgements = score_pairs_samplewise(
            scorer=scorer,
            pairs=pairs,
            dimension_weights=dimension_weights,
            use_tqdm=True,
            cache_config=SamplewiseCacheConfig(
                judge=judge_cfg.model,
                rubric=rubric_name,
                model_key_a="__agreement_samplewise_side_A",
                model_key_b="__agreement_samplewise_side_B",
                dataset_exact=cache_dataset_exact,
                n_instructions=config.n_instructions,
                dataset_base=cache_dataset_base,
            ),
            ignore_cache=ignore_score_cache,
        )

    def _has_nan(scores: dict[str, float]) -> bool:
        return any(isinstance(v, float) and math.isnan(v) for v in scores.values())

    def _sanitize_scores(scores: dict[str, float]) -> dict:
        return {
            k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in scores.items()
        }

    human_labels: list[str] = []
    judge_labels: list[str] = []
    per_sample: list[dict] = []
    n_parse_failures = 0

    for pair, sample, judged in zip(pairs, valid, judgements):
        h_label = _pref_to_label(sample.human_pref)

        sa, sb = judged.scores_a, judged.scores_b
        parse_failed = _has_nan(sa) or _has_nan(sb)

        if parse_failed:
            n_parse_failures += 1
            j_label = "parse_error"
            sa = _sanitize_scores(sa)
            sb = _sanitize_scores(sb)
            j_pref = None
        else:
            j_pref = judged.preference
            j_label = _pref_to_label(j_pref)

        human_labels.append(h_label)
        judge_labels.append(j_label)

        entry: dict = {
            "instruction_id": sample.instruction_id,
            "instruction": sample.instruction[:truncate_instruction],
            "model_a": pair.model_a,
            "model_b": pair.model_b,
            "human_pref": sample.human_pref,
            "human_label": h_label,
            "judge_pref": j_pref,
            "judge_label": j_label,
            "scores_a": sa,
            "scores_b": sb,
            "len_a": len(pair.completion_a),
            "len_b": len(pair.completion_b),
            "agree": h_label == j_label if not parse_failed else None,
            "parse_error": parse_failed,
        }
        if sample.metadata:
            entry["metadata"] = sample.metadata
        per_sample.append(entry)

    if n_parse_failures > 0:
        logger.warning(
            "%d/%d samples had NaN scores (parse failures) — excluded from metrics",
            n_parse_failures,
            n_total,
        )

    prompt_key = "pairwise" if judge_mode == "pairwise" else "samplewise"
    system_prompt_used = scorer.system_prompt.get(prompt_key, "")

    annotation_payload = {
        "metadata": {
            "dataset": dataset_name,
            "judge_model": judge_cfg.model,
            "judge_mode": judge_mode,
            "pairwise_prompt_style": judge_cfg.pairwise_prompt_style,
            "rubric": rubric_name,
            "rubric_k": rubric.k,
            "swap_debiasing": not judge_cfg.no_swap,
            "n_samples": n_total,
            "dataset_cache_key": cache_dataset,
            **({"language": config.language} if config.language else {}),
            **({"balance_by": config.balance_by} if config.balance_by else {}),
        },
        "system_prompt": system_prompt_used,
        "per_sample": per_sample,
    }
    annotation_path = save_agreement_annotations(
        output_dir,
        annotation_payload,
        config_snapshot=config.to_dict(),
    )
    config.save(Path(output_dir) / "agreement_config.json")
    logger.info("Saved agreement annotations: %s", annotation_path)

    if stage == "annotate":
        return {
            "annotation_path": str(annotation_path),
            "per_sample": per_sample,
        }

    return analyze_agreement_stage(config=config, annotation_data=annotation_payload)
