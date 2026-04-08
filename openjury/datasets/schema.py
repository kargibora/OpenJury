"""Core schema for the unified dataset abstraction.

Every dataset — instruction-only, arena-hard with metadata, or human
preference with completions — is normalised to a list of :class:`EvalSample`
wrapped in an :class:`EvalDataset`.

This common representation lets the arena pipeline carry per-instruction
metadata (language, category, source, …) all the way through scoring,
match results, and saved ``arena.json`` output.

Example::

    from openjury.datasets import load_dataset

    ds = load_dataset("m-arena-hard-en", n=100)
    for sample in ds:
        print(sample.instruction_id, sample.metadata.get("lang"))
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any, Iterator


@dataclass(slots=True)
class EvalSample:
    """One evaluation sample in the unified schema.

    Every sample *must* have an ``instruction`` and a stable
    ``instruction_id``.  Everything else is optional and depends on the
    dataset type:

    - **Instruction datasets** (alpaca-eval, arena-hard, m-arena-hard):
      ``metadata`` may contain ``lang``, ``category``, etc.
    - **Human-preference datasets** (lmsys, comparia): additionally carry
      ``completions`` (dict mapping model name → response text),
      ``human_pref`` (0.0 = A wins, 0.5 = tie, 1.0 = B wins), and
      model identifiers in ``metadata``.
    - Any dataset may optionally provide a ``reference_answer``.

    Attributes:
        instruction: The prompt / question text.
        instruction_id: Stable unique identifier (survives re-ordering).
        metadata: Arbitrary key→value pairs (lang, category, source, …).
        reference_answer: Gold reference if available.
        completions: Pre-existing model completions (model_name → text).
        human_pref: Human preference label (only for preference datasets).
    """

    instruction: str
    instruction_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    reference_answer: str | None = None
    completions: dict[str, str] | None = None
    human_pref: float | None = None


@dataclass
class EvalDataset:
    """A named collection of :class:`EvalSample` with optional schema info.

    Attributes:
        name: Dataset identifier (e.g. ``"m-arena-hard-en"``).
        samples: Ordered list of evaluation samples.
        metadata_schema: Documents which metadata keys this dataset
            provides, e.g. ``{"lang": "ISO-639-1 language code"}``.
            Purely informational — not enforced at runtime.
    """

    name: str
    samples: list[EvalSample] = field(default_factory=list)
    metadata_schema: dict[str, str] = field(default_factory=dict)

    # ── Convenience ──────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[EvalSample]:
        return iter(self.samples)

    def __getitem__(self, idx: int) -> EvalSample:
        return self.samples[idx]

    @property
    def instructions(self) -> list[str]:
        """All instruction texts in order."""
        return [s.instruction for s in self.samples]

    @property
    def instruction_ids(self) -> list[str]:
        """All instruction IDs in order."""
        return [s.instruction_id for s in self.samples]

    @property
    def has_completions(self) -> bool:
        """Whether any sample carries pre-existing completions."""
        return any(s.completions for s in self.samples)

    @property
    def has_human_pref(self) -> bool:
        """Whether any sample carries a human preference label."""
        return any(s.human_pref is not None for s in self.samples)

    @property
    def metadata_keys(self) -> set[str]:
        """Union of all metadata keys across samples."""
        keys: set[str] = set()
        for s in self.samples:
            keys.update(s.metadata.keys())
        return keys

    def head(self, n: int) -> EvalDataset:
        """Return a new dataset with only the first *n* samples."""
        return EvalDataset(
            name=self.name,
            samples=self.samples[:n],
            metadata_schema=self.metadata_schema,
        )

    def metadata_column(self, key: str) -> list[Any]:
        """Extract one metadata field as a flat list (aligned with samples).

        Missing values are returned as ``None``.
        """
        return [s.metadata.get(key) for s in self.samples]

    def filter(self, **kwargs: Any) -> EvalDataset:
        """Return a new dataset keeping only samples whose metadata matches.

        Example::

            ds_en = ds.filter(lang="en")
        """
        filtered = [
            s for s in self.samples
            if all(s.metadata.get(k) == v for k, v in kwargs.items())
        ]
        return EvalDataset(
            name=self.name,
            samples=filtered,
            metadata_schema=self.metadata_schema,
        )

    def sample_balanced(
        self,
        *,
        by: str,
        n: int,
        seed: int = 42,
    ) -> EvalDataset:
        """Return a class-balanced subsample by a metadata/top-level field.

        Sampling is approximately uniform across classes and redistributes
        deficits automatically. For example, with ``n=2000`` across 20 classes
        it will target ~100/class; if one class has fewer examples, the
        remainder is redistributed across other classes.

        For pairwise preference datasets, ``by="model"`` or ``by="models"``
        has special handling: it balances model appearances across both
        ``model_a`` and ``model_b``. Because each selected pair usually
        contains two models, the balanced unit in that mode is model
        appearances rather than raw sample rows.

        Args:
            by: Field name to balance by. Supports metadata keys directly
                (e.g. ``"lang"``) or ``"metadata.lang"``. ``"language"``
                falls back to ``metadata["lang"]``.
            n: Requested sample size.
            seed: RNG seed for reproducible sampling.
        """
        if n <= 0:
            return EvalDataset(
                name=self.name,
                samples=[],
                metadata_schema=self.metadata_schema,
            )
        if n >= len(self.samples):
            return EvalDataset(
                name=self.name,
                samples=list(self.samples),
                metadata_schema=self.metadata_schema,
            )

        def _extract_field(sample: EvalSample, field_name: str) -> Any:
            if field_name.startswith("metadata."):
                return sample.metadata.get(field_name.split(".", 1)[1])

            # Allow balancing by a top-level sample attribute when present.
            if field_name != "metadata" and hasattr(sample, field_name):
                return getattr(sample, field_name)

            # Metadata key (common case)
            value = sample.metadata.get(field_name)
            if value is None and field_name == "language":
                value = sample.metadata.get("lang")
            elif value is None and field_name == "lang":
                value = sample.metadata.get("language")
            return value

        def _normalize_bucket(value: Any) -> str | None:
            if value is None:
                return None
            if isinstance(value, str):
                stripped = value.strip()
                return stripped or None
            try:
                return str(value)
            except Exception:
                return repr(value)

        if by in {"model", "models"}:
            sample_models: list[tuple[str, ...]] = []
            pools: dict[str, list[int]] = {}
            n_missing = 0

            for idx, sample in enumerate(self.samples):
                models: list[str] = []
                for field_name in ("model_a", "model_b"):
                    model = _normalize_bucket(_extract_field(sample, field_name))
                    if model is not None and model not in models:
                        models.append(model)

                if not models:
                    fallback = _normalize_bucket(_extract_field(sample, "model"))
                    if fallback is not None:
                        models.append(fallback)

                if not models:
                    n_missing += 1
                    models = ["__missing__"]

                sample_models.append(tuple(models))
                for model in models:
                    pools.setdefault(model, []).append(idx)

            if len(pools) == 1 and "__missing__" in pools and n_missing == len(self.samples):
                raise ValueError(
                    f"Cannot balance dataset '{self.name}' by '{by}': "
                    "pairwise model fields are not present in samples."
                )

            rng = random.Random(seed)
            for indices in pools.values():
                rng.shuffle(indices)

            model_counts = {model: 0 for model in pools}
            selected_indices: list[int] = []
            selected_set: set[int] = set()
            target = min(n, len(self.samples))

            def _pop_candidate_for_model(model: str, *, scan_limit: int = 32) -> int | None:
                pool = pools[model]
                candidates: list[int] = []
                while pool and len(candidates) < scan_limit:
                    idx = pool.pop()
                    if idx in selected_set:
                        continue
                    candidates.append(idx)
                if not candidates:
                    return None

                rng.shuffle(candidates)
                best_idx = min(
                    candidates,
                    key=lambda idx: (
                        max(model_counts.get(m, 0) for m in sample_models[idx]),
                        sum(model_counts.get(m, 0) for m in sample_models[idx]),
                        len(sample_models[idx]),
                    ),
                )
                for idx in candidates:
                    if idx != best_idx:
                        pool.append(idx)
                return best_idx

            def _has_available(model: str) -> bool:
                pool = pools[model]
                while pool and pool[-1] in selected_set:
                    pool.pop()
                return bool(pool)

            while len(selected_indices) < target:
                active_models = [model for model in pools if _has_available(model)]
                if not active_models:
                    break

                min_count = min(model_counts[model] for model in active_models)
                candidate_models = [
                    model for model in active_models
                    if model_counts[model] == min_count
                ]
                rng.shuffle(candidate_models)

                chosen_idx: int | None = None
                for model in candidate_models:
                    chosen_idx = _pop_candidate_for_model(model)
                    if chosen_idx is not None:
                        break
                if chosen_idx is None:
                    break

                selected_set.add(chosen_idx)
                selected_indices.append(chosen_idx)
                for model in sample_models[chosen_idx]:
                    model_counts[model] = model_counts.get(model, 0) + 1

            rng.shuffle(selected_indices)
            return EvalDataset(
                name=self.name,
                samples=[self.samples[idx] for idx in selected_indices],
                metadata_schema=self.metadata_schema,
            )

        groups: dict[str, list[EvalSample]] = {}
        n_missing = 0
        for sample in self.samples:
            bucket = _normalize_bucket(_extract_field(sample, by))
            if bucket is None:
                n_missing += 1
                bucket = "__missing__"
            groups.setdefault(bucket, []).append(sample)

        # If every sample is missing the target field, fail loudly.
        if len(groups) == 1 and "__missing__" in groups and n_missing == len(self.samples):
            raise ValueError(
                f"Cannot balance dataset '{self.name}' by '{by}': field not found in samples."
            )

        rng = random.Random(seed)
        pools: dict[str, list[EvalSample]] = {}
        for key, items in groups.items():
            shuffled = list(items)
            rng.shuffle(shuffled)
            pools[key] = shuffled

        target = min(n, len(self.samples))
        selected: list[EvalSample] = []
        active = list(pools.keys())
        rng.shuffle(active)

        # Round-robin over active groups yields near-equal allocations and
        # naturally redistributes deficits from smaller groups.
        while active and len(selected) < target:
            next_active: list[str] = []
            for key in active:
                pool = pools[key]
                if pool:
                    selected.append(pool.pop())
                    if len(selected) >= target:
                        break
                if pool:
                    next_active.append(key)
            active = next_active
            if active:
                rng.shuffle(active)

        rng.shuffle(selected)
        return EvalDataset(
            name=self.name,
            samples=selected,
            metadata_schema=self.metadata_schema,
        )
