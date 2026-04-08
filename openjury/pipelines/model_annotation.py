"""Generic annotation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

import pandas as pd

from openjury._logging import logger
from openjury.annotate_config import AnnotateConfig
from openjury.arena.config import MatchResult, ModelEntry
from openjury.arena.matchmaker import get_matchmaker
from openjury.cache.completions import cache
from openjury.common.pair_annotation import (
    PairSample,
    PairwiseCacheConfig,
    SamplewiseCacheConfig,
    score_pairs_pairwise,
    score_pairs_samplewise,
)
from openjury.criteria import CriteriaScorer, get_criteria
from openjury.datasets import EvalDataset, load_dataset
from openjury.models.factory import make_model
from openjury.pipelines.annotate import save_annotate_annotations
from openjury.pipelines.generation import generate_instructions


@dataclass(frozen=True)
class _PairContext:
    pair: PairSample
    instruction_id: str
    instruction_metadata: dict[str, Any]
    human_preference: float | None = None


def _load_criteria(criteria_spec: str):
    if Path(criteria_spec).is_file():
        from openjury.criteria.schema import Criteria, Criterion

        path = Path(criteria_spec)
        text = path.read_text(encoding="utf-8")
        if path.suffix in (".yaml", ".yml"):
            import yaml  # type: ignore[import-untyped]
            rdata = yaml.safe_load(text)
        else:
            rdata = json.loads(text)
        return Criteria(
            name=rdata.get("name", "custom"),
            description=rdata.get("description", ""),
            criteria=[Criterion(**d) for d in rdata.get("criteria", rdata.get("dimensions", []))],
        )
    return get_criteria(criteria_spec)


def _annotation_cache_dataset_key(config: AnnotateConfig, *, exact: bool) -> str:
    parts = [config.dataset, "annotate", f"pairing_{config.pairing.source}"]
    if config.cache_variant:
        parts.append(f"variant_{config.cache_variant}")
    if exact:
        if config.language:
            parts.append(f"lang_{config.language}")
        if config.balance_by:
            parts.append(f"balance_{config.balance_by}")
        if config.n_instructions is not None:
            parts.append(f"seed_{config.seed}")
    return "__".join(parts)


def _resolve_challenger_completions(
    *,
    config: AnnotateConfig,
    dataset: str,
    instructions: pd.Series,
) -> pd.DataFrame:
    if config.challenger is None:
        raise ValueError("Challenger completions requested without a challenger model.")
    entry = config.challenger
    if entry.completions:
        path = Path(entry.completions)
        if not path.exists():
            raise FileNotFoundError(
                f"Challenger completions file not found: {path} (model: {entry.name})"
            )
        df = pd.read_parquet(path)
        logger.info(
            "Loaded %d challenger completions for %s from file: %s",
            len(df),
            entry.short_name,
            path,
        )
        return df

    def _generate() -> pd.DataFrame:
        model_cfg = config.generation.to_model_config(entry)
        return generate_instructions(
            instructions=instructions,
            model=entry.name,
            max_tokens=entry.max_tokens or config.generation.max_tokens,
            config=model_cfg,
            truncate_input_chars=config.generation.truncate_input_chars,
            use_tqdm=config.generation.use_tqdm,
        )

    was_cached = (not config.generation.ignore_cache) and cache.exists(
        entry.name,
        dataset,
        config.n_instructions,
    )
    df = cache.get_or_generate(
        model=entry.name,
        dataset=dataset,
        n=config.n_instructions,
        generate_fn=_generate,
        ignore_cache=config.generation.ignore_cache,
    )
    logger.info(
        "Challenger completions for %s: %d rows (%s)",
        entry.short_name,
        len(df),
        "cache" if was_cached else "generated",
    )
    return df


def _completion_lookup(df: pd.DataFrame) -> dict[str, str]:
    required = {"instruction_index", "completion"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Completion parquet must contain columns "
            f"{sorted(required)}, missing {sorted(missing)}"
        )
    lookup: dict[str, str] = {}
    for row in df.itertuples(index=False):
        key = str(getattr(row, "instruction_index"))
        if key in lookup:
            raise ValueError(f"Completion parquet has duplicate instruction_index '{key}'.")
        lookup[key] = str(getattr(row, "completion"))
    return lookup


def _completion_for_sample(
    lookup: dict[str, str],
    *,
    instruction_index: int,
    instruction_id: str,
) -> str | None:
    """Resolve a completion by row index first, then stable instruction id."""
    return lookup.get(str(instruction_index)) or lookup.get(str(instruction_id))


def _resolve_model_completions(
    *,
    models: list[ModelEntry],
    dataset: str,
    n_instructions: int | None,
    instructions: pd.Series,
    generation_max_tokens: int,
    truncate_input_chars: int = 8192,
    ignore_cache: bool = False,
) -> dict[str, pd.DataFrame]:
    """Resolve completions for a model pool.

    Resolution order per model:
    1. use `model.completions` if explicitly provided
    2. use cache when present
    3. generate on cache miss
    """
    completions: dict[str, pd.DataFrame] = {}

    for entry in models:
        model = entry.name

        if entry.completions:
            pq = Path(entry.completions)
            if not pq.exists():
                raise FileNotFoundError(
                    f"Completions file not found: {pq} (model: {model})"
                )
            df = pd.read_parquet(pq)
            completions[model] = df
            logger.info(
                "Loaded %d completions for %s from file: %s",
                len(df),
                entry.short_name,
                pq,
            )
            continue

        max_tok = entry.max_tokens or generation_max_tokens

        def _generate(
            _model: str = model,
            _entry: ModelEntry = entry,
            _max_tok: int = max_tok,
        ) -> pd.DataFrame:
            effective_entry = _entry
            if _entry.max_tokens is None:
                from dataclasses import replace

                effective_entry = replace(_entry, max_tokens=_max_tok)
            model_config = effective_entry.to_model_config()
            return generate_instructions(
                instructions=instructions,
                model=_model,
                max_tokens=_max_tok,
                config=model_config,
                truncate_input_chars=truncate_input_chars,
            )

        was_cached = (not ignore_cache) and cache.exists(model, dataset, n_instructions)
        df = cache.get_or_generate(
            model=model,
            dataset=dataset,
            n=n_instructions,
            generate_fn=_generate,
            ignore_cache=ignore_cache,
        )
        completions[model] = df
        logger.info(
            "Completions for %s: %d rows (%s)",
            entry.short_name,
            len(df),
            "cache" if was_cached else "generated",
        )

    return completions


def _filter_inline_completions(
    inline: dict[str, Any],
    *,
    include: set[str],
    exclude: set[str],
) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for model_name, completion in inline.items():
        if include and model_name not in include:
            continue
        if model_name in exclude:
            continue
        out.append((model_name, completion))
    return out


def _build_challenger_vs_dataset_inline_pairs(
    *,
    config: AnnotateConfig,
    dataset: EvalDataset,
    challenger_lookup: dict[str, str],
) -> tuple[list[_PairContext], list[str]]:
    if config.challenger is None:
        raise ValueError("pairing.source='challenger_vs_dataset_inline' requires a challenger.")

    rng = random.Random(config.pairing.seed)
    include = set(config.pairing.include_models or [])
    exclude = set(config.pairing.exclude_models or [])

    pair_contexts: list[_PairContext] = []
    opponent_models: list[str] = []
    seen_models: set[str] = set()
    missing_challenger_ids: list[str] = []
    samples_with_inline = 0

    for idx, sample in enumerate(dataset.samples):
        inline = dict(sample.completions or {})
        if not inline:
            continue
        samples_with_inline += 1

        available = [
            (model_name, completion)
            for model_name, completion in _filter_inline_completions(
                inline,
                include=include,
                exclude=exclude,
            )
            if model_name != config.challenger.name
        ]
        if not available:
            continue

        instruction_id = str(sample.instruction_id)
        challenger_completion = challenger_lookup.get(instruction_id)
        if challenger_completion is None:
            missing_challenger_ids.append(instruction_id)
            continue

        chosen = (
            available
            if config.pairing.strategy == "all"
            else [rng.choice(available)]
        )
        for opponent_model, opponent_completion in chosen:
            if opponent_model not in seen_models:
                seen_models.add(opponent_model)
                opponent_models.append(opponent_model)
            metadata = dict(sample.metadata)
            metadata["dataset_inline_models"] = list(inline.keys())
            metadata["dataset_inline_human_pref"] = sample.human_pref
            metadata["pairing_source"] = config.pairing.source
            metadata["pairing_strategy"] = config.pairing.strategy
            pair_contexts.append(
                _PairContext(
                    pair=PairSample(
                        sample_id=(
                            f"{instruction_id}::A={config.challenger.name}::B={opponent_model}"
                        ),
                        instruction_index=idx,
                        instruction=sample.instruction,
                        model_a=config.challenger.name,
                        model_b=opponent_model,
                        completion_a=challenger_completion,
                        completion_b=str(opponent_completion),
                    ),
                    instruction_id=instruction_id,
                    instruction_metadata=metadata,
                )
            )

    if samples_with_inline == 0:
        raise ValueError(
            f"Dataset '{config.dataset}' does not provide inline completions. "
            "This annotation mode supports datasets like lmsys/comparia "
            "that expose sample.completions."
        )
    if missing_challenger_ids:
        preview = ", ".join(missing_challenger_ids[:5])
        raise ValueError(
            "Challenger completions are missing for "
            f"{len(missing_challenger_ids)} instruction IDs. First few: {preview}. "
            "Generate completions with the same dataset selection or omit "
            "challenger.completions to auto-generate them."
        )
    if not pair_contexts:
        raise ValueError(
            "No annotation pairs could be built after applying pairing filters."
        )
    return pair_contexts, opponent_models


def _build_dataset_pairs(
    *,
    config: AnnotateConfig,
    dataset: EvalDataset,
) -> tuple[list[_PairContext], list[str]]:
    include = set(config.pairing.include_models or [])
    exclude = set(config.pairing.exclude_models or [])

    pair_contexts: list[_PairContext] = []
    model_names: list[str] = []
    seen_models: set[str] = set()
    samples_with_pairs = 0

    for idx, sample in enumerate(dataset.samples):
        inline = dict(sample.completions or {})
        if not inline:
            continue

        available = _filter_inline_completions(
            inline,
            include=include,
            exclude=exclude,
        )
        if len(available) < 2:
            continue
        samples_with_pairs += 1

        (model_a, completion_a), (model_b, completion_b) = available[:2]
        for model_name in (model_a, model_b):
            if model_name not in seen_models:
                seen_models.add(model_name)
                model_names.append(model_name)

        metadata = dict(sample.metadata)
        metadata["dataset_inline_models"] = list(inline.keys())
        metadata["pairing_source"] = config.pairing.source
        metadata["pairing_strategy"] = config.pairing.strategy
        pair_contexts.append(
            _PairContext(
                pair=PairSample(
                    sample_id=f"{sample.instruction_id}::A={model_a}::B={model_b}",
                    instruction_index=idx,
                    instruction=sample.instruction,
                    model_a=model_a,
                    model_b=model_b,
                    completion_a=str(completion_a),
                    completion_b=str(completion_b),
                ),
                instruction_id=str(sample.instruction_id),
                instruction_metadata=metadata,
                human_preference=sample.human_pref,
            )
        )

    if samples_with_pairs == 0 or not pair_contexts:
        raise ValueError(
            f"Dataset '{config.dataset}' does not expose usable dataset-defined pairs."
        )
    return pair_contexts, model_names


def _build_matchmaker_pairs(
    *,
    config: AnnotateConfig,
    dataset: EvalDataset,
    completions: dict[str, pd.DataFrame],
) -> tuple[list[_PairContext], list[str]]:
    if not config.models or len(config.models) < 2:
        raise ValueError(
            "pairing.source='matchmaker' requires at least two models in annotate config."
        )

    include = set(config.pairing.include_models or [])
    exclude = set(config.pairing.exclude_models or [])
    model_entries = [
        entry
        for entry in config.models
        if (not include or entry.name in include) and entry.name not in exclude
    ]
    if len(model_entries) < 2:
        raise ValueError(
            "Matchmaker annotation requires at least two models after pairing filters."
        )

    model_names = [entry.name for entry in model_entries]
    completion_lookups = {
        name: _completion_lookup(completions[name]) for name in model_names
    }
    instruction_indices = list(range(len(dataset.samples)))
    matchmaker_fn = get_matchmaker(config.pairing.strategy or "round_robin")
    matches = matchmaker_fn(
        models=model_names,
        instruction_indices=instruction_indices,
        n_matches=config.pairing.n_matches,
        seed=config.pairing.seed,
    )

    pair_contexts: list[_PairContext] = []
    missing: list[str] = []

    for match in matches:
        idx = match.instruction_index
        sample = dataset.samples[idx]
        instruction_id = str(sample.instruction_id)
        completion_a = _completion_for_sample(
            completion_lookups[match.model_a],
            instruction_index=idx,
            instruction_id=instruction_id,
        )
        completion_b = _completion_for_sample(
            completion_lookups[match.model_b],
            instruction_index=idx,
            instruction_id=instruction_id,
        )
        if completion_a is None or completion_b is None:
            missing.append(
                f"{instruction_id}::{match.model_a if completion_a is None else ''}"
                f"{'|' if completion_a is None and completion_b is None else ''}"
                f"{match.model_b if completion_b is None else ''}"
            )
            continue
        metadata = dict(sample.metadata)
        metadata["pairing_source"] = config.pairing.source
        metadata["pairing_strategy"] = config.pairing.strategy
        pair_contexts.append(
            _PairContext(
                pair=PairSample(
                    sample_id=f"{instruction_id}::A={match.model_a}::B={match.model_b}",
                    instruction_index=idx,
                    instruction=sample.instruction,
                    model_a=match.model_a,
                    model_b=match.model_b,
                    completion_a=completion_a,
                    completion_b=completion_b,
                ),
                instruction_id=instruction_id,
                instruction_metadata=metadata,
            )
        )

    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            "Model completions are missing for one or more matchmaker pairs. "
            f"First few missing entries: {preview}"
        )
    if not pair_contexts:
        raise ValueError("Matchmaker annotation did not produce any pairs.")
    return pair_contexts, model_names


def _ordered_models(pair_contexts: list[_PairContext]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for ctx in pair_contexts:
        for model_name in (ctx.pair.model_a, ctx.pair.model_b):
            if model_name not in seen:
                seen.add(model_name)
                ordered.append(model_name)
    return ordered


def run_annotate(config: AnnotateConfig, *, persist: bool = True) -> dict[str, Any]:
    ds_opts = config.dataset_options
    cache_dataset = ds_opts.cache_key()
    dataset = load_dataset(ds_opts.name, n=ds_opts.n_instructions, **ds_opts.loader_kwargs())
    logger.info("Loaded dataset '%s': %d samples", config.dataset, len(dataset))

    pair_contexts: list[_PairContext]
    models: list[str]

    if config.pairing.source == "challenger_vs_dataset_inline":
        instructions = pd.Series(
            [sample.instruction for sample in dataset.samples],
            index=[sample.instruction_id for sample in dataset.samples],
            name="instruction",
        )
        challenger_df = _resolve_challenger_completions(
            config=config,
            dataset=cache_dataset,
            instructions=instructions,
        )
        challenger_lookup = _completion_lookup(challenger_df)
        pair_contexts, opponent_models = _build_challenger_vs_dataset_inline_pairs(
            config=config,
            dataset=dataset,
            challenger_lookup=challenger_lookup,
        )
        models = [config.challenger.name, *opponent_models] if config.challenger else opponent_models
    elif config.pairing.source == "dataset_pairs":
        pair_contexts, models = _build_dataset_pairs(config=config, dataset=dataset)
    elif config.pairing.source == "matchmaker":
        instructions = pd.Series(
            [sample.instruction for sample in dataset.samples],
            name="instruction",
        )
        completions = _resolve_model_completions(
            models=config.models or [],
            dataset=cache_dataset,
            n_instructions=config.n_instructions,
            instructions=instructions,
            generation_max_tokens=config.generation.max_tokens,
            truncate_input_chars=config.generation.truncate_input_chars,
            ignore_cache=config.generation.ignore_cache,
        )
        pair_contexts, models = _build_matchmaker_pairs(
            config=config,
            dataset=dataset,
            completions=completions,
        )
    else:
        raise ValueError(f"Unsupported pairing source '{config.pairing.source}'.")

    pairs = [ctx.pair for ctx in pair_contexts]
    logger.info(
        "Built %d annotation matches for pairing source '%s'",
        len(pairs),
        config.pairing.source,
    )

    criteria = _load_criteria(config.criteria)
    judge_cfg = config.judge
    judge_model = make_model(judge_cfg.model, config=judge_cfg.to_model_config())
    scorer = CriteriaScorer(
        judge_model=judge_model,
        criteria=criteria,
        provide_explanation=judge_cfg.provide_explanation,
        pairwise_prompt_style=judge_cfg.pairwise_prompt_style,
    )
    dimension_weights = {criterion.name: criterion.weight for criterion in criteria.criteria}

    cache_dataset_exact = _annotation_cache_dataset_key(config, exact=True)
    cache_dataset_base = _annotation_cache_dataset_key(config, exact=False)
    key_prefix = f"__annotate__source_{config.pairing.source}"
    if config.cache_variant:
        key_prefix += f"__variant_{config.cache_variant}"
    if config.challenger is not None and config.challenger.name:
        key_prefix += f"__challenger_{config.challenger.name}"
    if config.pairing.strategy:
        key_prefix += f"__strategy_{config.pairing.strategy}"

    if judge_cfg.mode == "pairwise":
        model_key = (
            f"{key_prefix}__pairwise__style_{judge_cfg.pairwise_prompt_style}"
            f"__swap_{'off' if judge_cfg.no_swap else 'on'}"
        )
        judgements = score_pairs_pairwise(
            scorer=scorer,
            pairs=pairs,
            swap_to_debias=not judge_cfg.no_swap,
            use_tqdm=True,
            cache_config=PairwiseCacheConfig(
                judge=judge_cfg.model,
                criteria=config.criteria,
                model_key=model_key,
                dataset_exact=cache_dataset_exact,
                dataset_base=cache_dataset_base,
                n_instructions=config.n_instructions,
            ),
            ignore_cache=config.ignore_score_cache,
            n_trials=judge_cfg.n_trials,
        )
    else:
        judgements = score_pairs_samplewise(
            scorer=scorer,
            pairs=pairs,
            dimension_weights=dimension_weights,
            use_tqdm=True,
            cache_config=SamplewiseCacheConfig(
                judge=judge_cfg.model,
                criteria=config.criteria,
                model_key_a=f"{key_prefix}__samplewise__A",
                model_key_b=f"{key_prefix}__samplewise__B",
                dataset_exact=cache_dataset_exact,
                dataset_base=cache_dataset_base,
                n_instructions=config.n_instructions,
            ),
            ignore_cache=config.ignore_score_cache,
            n_trials=judge_cfg.n_trials,
        )

    criteria_def: dict[str, Any] = {}
    for criterion in criteria.criteria:
        criteria_def[criterion.name] = {
            "description": criterion.description,
            "scale_min": criterion.scale_min,
            "scale_max": criterion.scale_max,
            "weight": criterion.weight,
            **(
                {"score_references": criterion.score_references}
                if criterion.score_references
                else {}
            ),
        }

    matches: list[MatchResult] = []
    for ctx, judged in zip(pair_contexts, judgements):
        matches.append(
            MatchResult(
                model_a=ctx.pair.model_a,
                model_b=ctx.pair.model_b,
                instruction_index=ctx.pair.instruction_index,
                scores_a=judged.scores_a,
                scores_b=judged.scores_b,
                preference=judged.preference,
                instruction=ctx.pair.instruction,
                instruction_id=ctx.instruction_id,
                instruction_metadata=ctx.instruction_metadata,
                human_preference=ctx.human_preference,
                completion_a=ctx.pair.completion_a if config.include_completions else "",
                completion_b=ctx.pair.completion_b if config.include_completions else "",
                raw_judge_output=judged.raw_judge_output if config.include_raw_judge else "",
                raw_judge_output_swapped=(
                    judged.raw_judge_output_swapped if config.include_raw_judge else None
                ),
                scores_a_original=judged.scores_a_original,
                scores_b_original=judged.scores_b_original,
                preference_original=judged.preference_original,
                scores_a_swapped=judged.scores_a_swapped,
                scores_b_swapped=judged.scores_b_swapped,
                preference_swapped=judged.preference_swapped,
            )
        )

    prompt_key = "pairwise" if judge_cfg.mode == "pairwise" else "samplewise"
    system_prompt_used = scorer.system_prompt.get(prompt_key, "")
    payload = {
        "metadata": {
            "models": models or _ordered_models(pair_contexts),
            "dataset": config.dataset,
            "dataset_cache_key": cache_dataset,
            "judge_model": judge_cfg.model,
            "judge_mode": judge_cfg.mode,
            "pairwise_prompt_style": judge_cfg.pairwise_prompt_style,
            "criteria": config.criteria,
            "pairing_source": config.pairing.source,
            "pairing_strategy": config.pairing.strategy,
            "matchmaker": (
                config.pairing.strategy if config.pairing.source == "matchmaker" else None
            ),
            "swap_debiasing": not judge_cfg.no_swap,
            "n_instructions": len(dataset),
            "n_matches": len(matches),
            "n_trials": judge_cfg.n_trials,
            "challenger_model": (
                config.challenger.name
                if config.challenger is not None and config.challenger.name
                else None
            ),
        },
        "criteria_definition": criteria_def,
        "criterion_names": list(criteria.criterion_names),
        "instruction_metadata": [sample.metadata for sample in dataset.samples],
        "model_scores": {},
        "matches": [match.to_dict() for match in matches],
        "system_prompt": system_prompt_used,
    }
    if persist:
        annotation_path = save_annotate_annotations(
            config.output_dir,
            payload,
            config_snapshot=config.to_dict(),
        )
        config.save(Path(config.output_dir) / "annotate_config.json")
        logger.info("Saved annotate annotations: %s", annotation_path)
    return payload
