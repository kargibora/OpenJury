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

import numpy as np

from openjury._logging import logger
from openjury.arena.config import AgreementConfig, ModelScore
from openjury.cache.scores import score_cache
from openjury.datasets import load_dataset as load_eval_dataset
from openjury.models.factory import make_model
from openjury.pipelines.analysis import (
    compute_agreement_metrics,
    log_agreement_summary,
    save_agreement_outputs,
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


def _derive_preference_from_scores(
    scores_a: dict[str, float],
    scores_b: dict[str, float],
    weights: dict[str, float],
) -> float:
    """Derive a preference float from weighted rubric averages."""

    def _weighted_avg(scores: dict[str, float]) -> float:
        total_w, total_s = 0.0, 0.0
        for dim, w in weights.items():
            s = scores.get(dim, float("nan"))
            if s == s:  # not NaN
                total_w += w
                total_s += w * s
        return total_s / total_w if total_w > 0 else float("nan")

    avg_a = _weighted_avg(scores_a)
    avg_b = _weighted_avg(scores_b)

    if np.isnan(avg_a) or np.isnan(avg_b):
        return 0.5
    if avg_a > avg_b:
        return 0.0
    if avg_b > avg_a:
        return 1.0
    return 0.5


def _agreement_cache_dataset_key(config: AgreementConfig, *, exact: bool) -> str:
    """Build a dataset cache namespace for agreement scoring."""
    parts = [config.dataset, "agreement"]
    if exact:
        if config.language:
            parts.append(f"lang_{config.language}")
        if config.n_instructions is not None:
            parts.append(f"seed_{config.seed}")
    return "__".join(parts)


def _restore_scores_by_sample_id(
    cached_scores: list[ModelScore],
    sample_ids: list[str],
) -> list[ModelScore] | None:
    """Restore cached scores in the current sample order using stable IDs."""
    if not cached_scores:
        return []

    if all(ms.sample_id for ms in cached_scores):
        by_id: dict[str, ModelScore] = {}
        for ms in cached_scores:
            if ms.sample_id not in by_id:
                by_id[ms.sample_id] = ms
        missing = [sid for sid in sample_ids if sid not in by_id]
        if missing:
            return None
        return [by_id[sid] for sid in sample_ids]

    if len(cached_scores) != len(sample_ids):
        return None
    return sorted(cached_scores, key=lambda ms: ms.instruction_index)


def _restore_pairwise_scores_by_sample_id(
    cached_scores: list[ModelScore],
    sample_ids: list[str],
) -> list[tuple[ModelScore, ModelScore]] | None:
    """Restore interleaved pairwise cache entries (A then B) in current order."""
    if not cached_scores:
        return []
    if len(cached_scores) % 2 != 0:
        return None

    if not all(ms.sample_id for ms in cached_scores):
        if len(cached_scores) != 2 * len(sample_ids):
            return None
        ordered = sorted(cached_scores, key=lambda ms: ms.instruction_index)
        return [(ordered[i], ordered[i + 1]) for i in range(0, len(ordered), 2)]

    by_id: dict[str, tuple[ModelScore, ModelScore]] = {}
    for i in range(0, len(cached_scores), 2):
        ms_a = cached_scores[i]
        ms_b = cached_scores[i + 1]
        sid_a = ms_a.sample_id
        sid_b = ms_b.sample_id
        if not sid_a or not sid_b or sid_a != sid_b:
            return None
        if sid_a not in by_id:
            by_id[sid_a] = (ms_a, ms_b)

    missing = [sid for sid in sample_ids if sid not in by_id]
    if missing:
        return None
    return [by_id[sid] for sid in sample_ids]


def analyze_agreement_annotations(
    config: AgreementConfig,
    annotation_data: dict[str, Any],
) -> dict[str, Any]:
    """Analyze persisted agreement annotations and write final outputs."""
    meta = annotation_data["metadata"]
    per_sample = annotation_data["per_sample"]
    system_prompt = annotation_data.get("system_prompt", "")

    metrics = compute_agreement_metrics(
        per_sample=per_sample,
        dataset=meta["dataset"],
        judge_model=meta["judge_model"],
        judge_mode=meta["judge_mode"],
        rubric=meta["rubric"],
        swap_debiasing=meta["swap_debiasing"],
        language=meta.get("language"),
    )
    log_agreement_summary(metrics)
    json_path, csv_path = save_agreement_outputs(
        output_dir=config.output_dir,
        metrics=metrics,
        system_prompt=system_prompt,
        per_sample=per_sample,
    )
    logger.info("")
    logger.info("Results saved:")
    logger.info("  📄 %s", json_path)
    logger.info("  📄 %s", csv_path)
    return {"metrics": metrics, "per_sample": per_sample}


def run_agreement(config: AgreementConfig, stage: str = "all") -> dict[str, Any]:
    """Run the agreement pipeline with stage control."""
    if stage not in {"all", "annotate", "analyze"}:
        raise ValueError(f"Unsupported agreement stage: {stage}")
    if stage == "analyze":
        annotation_data = load_agreement_annotations(config.output_dir)
        return analyze_agreement_annotations(config, annotation_data)

    judge_cfg = config.judge
    dataset_name = config.dataset
    rubric_name = config.rubric
    judge_mode = judge_cfg.mode
    output_dir = config.output_dir
    ignore_score_cache = config.ignore_score_cache
    truncate_instruction = config.truncate_instruction

    ds_opts = config.dataset_options
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

    instructions: list[str] = []
    completions_a: list[str] = []
    completions_b: list[str] = []
    models_a: list[str] = []
    models_b: list[str] = []

    for sample in valid:
        model_names = list(sample.completions.keys())
        ma, mb = model_names[0], model_names[1]
        instructions.append(sample.instruction)
        completions_a.append(sample.completions[ma])
        completions_b.append(sample.completions[mb])
        models_a.append(ma)
        models_b.append(mb)

    n_total = len(instructions)
    sample_ids = [s.instruction_id for s in valid]
    logger.info("Judging %d pairs …", n_total)

    cache_dataset_exact = _agreement_cache_dataset_key(config, exact=True)
    cache_dataset_base = _agreement_cache_dataset_key(config, exact=False)

    judge_prefs: list[float] = []
    all_scores_a: list[dict[str, float]] = []
    all_scores_b: list[dict[str, float]] = []
    raw_outputs: list[str] = []
    raw_outputs_swapped: list[str | None] = []

    if judge_mode == "pairwise":
        pair_key = f"__agreement_pairwise__swap_{'off' if judge_cfg.no_swap else 'on'}"

        def _load_cached_pairwise() -> list[tuple[ModelScore, ModelScore]] | None:
            if ignore_score_cache:
                return None

            cached = score_cache.get(
                judge=judge_cfg.model,
                rubric=rubric_name,
                model=pair_key,
                dataset=cache_dataset_exact,
                n=config.n_instructions,
            )
            restored = (
                _restore_pairwise_scores_by_sample_id(cached, sample_ids)
                if cached is not None
                else None
            )
            if restored is not None:
                return restored

            if cache_dataset_exact != cache_dataset_base:
                cached_all = score_cache.get(
                    judge=judge_cfg.model,
                    rubric=rubric_name,
                    model=pair_key,
                    dataset=cache_dataset_base,
                    n=None,
                )
                restored_all = (
                    _restore_pairwise_scores_by_sample_id(cached_all, sample_ids)
                    if cached_all is not None
                    else None
                )
                if restored_all is not None:
                    logger.info(
                        "Using pairwise scores from unfiltered agreement cache (%s)",
                        cache_dataset_base,
                    )
                    return restored_all
            return None

        cached_pw_pairs = _load_cached_pairwise()

        if cached_pw_pairs is not None:
            logger.info("Using %d cached pairwise scores", len(cached_pw_pairs))
            for ms_a, ms_b in cached_pw_pairs:
                scores_a = dict(ms_a.scores)
                pref = float(scores_a.pop("__preference__", 0.5))
                all_scores_a.append(scores_a)
                all_scores_b.append(dict(ms_b.scores))
                judge_prefs.append(pref)
                raw_outputs.append(ms_a.raw_judge_output)
                raw_outputs_swapped.append(ms_b.raw_judge_output or None)
        else:
            pairwise_results = scorer.score_pairwise(
                instructions=instructions,
                completions_A=completions_a,
                completions_B=completions_b,
                swap_to_debias=not judge_cfg.no_swap,
                use_tqdm=True,
            )
            cache_entries: list[ModelScore] = []
            for i_pw, pr in enumerate(pairwise_results):
                judge_prefs.append(pr.preference)
                all_scores_a.append(pr.scores_A)
                all_scores_b.append(pr.scores_B)
                raw_outputs.append(pr.raw_judge_output)
                raw_outputs_swapped.append(pr.raw_judge_output_swapped)
                cache_entries.append(
                    ModelScore(
                        model=models_a[i_pw],
                        instruction_index=i_pw,
                        sample_id=sample_ids[i_pw],
                        scores={**pr.scores_A, "__preference__": pr.preference},
                        completion=completions_a[i_pw],
                        raw_judge_output=pr.raw_judge_output,
                    )
                )
                cache_entries.append(
                    ModelScore(
                        model=models_b[i_pw],
                        instruction_index=i_pw,
                        sample_id=sample_ids[i_pw],
                        scores=pr.scores_B,
                        completion=completions_b[i_pw],
                        raw_judge_output=pr.raw_judge_output_swapped or "",
                    )
                )
            score_cache.put(
                cache_entries,
                judge=judge_cfg.model,
                rubric=rubric_name,
                model=pair_key,
                dataset=cache_dataset_exact,
                n=config.n_instructions,
            )

    else:

        def _score_side(side_completions: list[str], side_label: str) -> list[ModelScore]:
            logger.info("Scoring %d completions (side %s) …", n_total, side_label)
            rs_list = scorer.score(
                instructions=instructions,
                completions=side_completions,
                model_name=side_label,
                use_tqdm=True,
            )
            return [
                ModelScore(
                    model=side_label,
                    instruction_index=i,
                    sample_id=sample_ids[i],
                    scores=rs.scores,
                    completion=side_completions[i],
                    raw_judge_output=rs.raw_judge_output,
                )
                for i, rs in enumerate(rs_list)
            ]

        cache_model_a = "__agreement_samplewise_side_A"
        cache_model_b = "__agreement_samplewise_side_B"

        def _load_cached_side(cache_model_key: str) -> list[ModelScore] | None:
            if ignore_score_cache:
                return None

            cached = score_cache.get(
                judge=judge_cfg.model,
                rubric=rubric_name,
                model=cache_model_key,
                dataset=cache_dataset_exact,
                n=config.n_instructions,
            )
            restored = (
                _restore_scores_by_sample_id(cached, sample_ids)
                if cached is not None
                else None
            )
            if restored is not None:
                return restored

            if cache_dataset_exact != cache_dataset_base:
                cached_all = score_cache.get(
                    judge=judge_cfg.model,
                    rubric=rubric_name,
                    model=cache_model_key,
                    dataset=cache_dataset_base,
                    n=None,
                )
                restored_all = (
                    _restore_scores_by_sample_id(cached_all, sample_ids)
                    if cached_all is not None
                    else None
                )
                if restored_all is not None:
                    logger.info(
                        "Using samplewise scores from unfiltered agreement cache (%s)",
                        cache_dataset_base,
                    )
                    return restored_all
            return None

        ms_a_list = _load_cached_side(cache_model_a)
        if ms_a_list is None:
            ms_a_list = _score_side(completions_a, "side_A")
            score_cache.put(
                ms_a_list,
                judge=judge_cfg.model,
                rubric=rubric_name,
                model=cache_model_a,
                dataset=cache_dataset_exact,
                n=config.n_instructions,
            )

        ms_b_list = _load_cached_side(cache_model_b)
        if ms_b_list is None:
            ms_b_list = _score_side(completions_b, "side_B")
            score_cache.put(
                ms_b_list,
                judge=judge_cfg.model,
                rubric=rubric_name,
                model=cache_model_b,
                dataset=cache_dataset_exact,
                n=config.n_instructions,
            )

        for ms_a, ms_b in zip(ms_a_list, ms_b_list):
            pref = _derive_preference_from_scores(ms_a.scores, ms_b.scores, dimension_weights)
            judge_prefs.append(pref)
            all_scores_a.append(ms_a.scores)
            all_scores_b.append(ms_b.scores)
            raw_outputs.append(ms_a.raw_judge_output)
            raw_outputs_swapped.append(None)

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

    for i, sample in enumerate(valid):
        h_label = _pref_to_label(sample.human_pref)

        sa, sb = all_scores_a[i], all_scores_b[i]
        parse_failed = _has_nan(sa) or _has_nan(sb)

        if parse_failed:
            n_parse_failures += 1
            j_label = "parse_error"
            sa = _sanitize_scores(sa)
            sb = _sanitize_scores(sb)
            j_pref = None
        else:
            j_pref = judge_prefs[i]
            j_label = _pref_to_label(j_pref)

        human_labels.append(h_label)
        judge_labels.append(j_label)

        entry: dict = {
            "instruction_id": sample.instruction_id,
            "instruction": sample.instruction[:truncate_instruction],
            "model_a": models_a[i],
            "model_b": models_b[i],
            "human_pref": sample.human_pref,
            "human_label": h_label,
            "judge_pref": j_pref,
            "judge_label": j_label,
            "scores_a": sa,
            "scores_b": sb,
            "len_a": len(completions_a[i]),
            "len_b": len(completions_b[i]),
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
            "rubric": rubric_name,
            "rubric_k": rubric.k,
            "swap_debiasing": not judge_cfg.no_swap,
            "n_samples": n_total,
            **({"language": config.language} if config.language else {}),
        },
        "system_prompt": system_prompt_used,
        "per_sample": per_sample,
    }
    annotation_path = save_agreement_annotations(output_dir, annotation_payload)
    config.save(Path(output_dir) / "agreement_config.json")
    logger.info("Saved agreement annotations: %s", annotation_path)

    if stage == "annotate":
        return {
            "annotation_path": str(annotation_path),
            "per_sample": per_sample,
        }

    return analyze_agreement_annotations(config, annotation_payload)
