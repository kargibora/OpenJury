from __future__ import annotations

from collections import Counter

from openjury.datasets.options import DatasetOptions
from openjury.datasets.schema import EvalDataset, EvalSample


def _make_dataset(group_sizes: dict[str, int]) -> EvalDataset:
    samples: list[EvalSample] = []
    for lang, count in group_sizes.items():
        for i in range(count):
            samples.append(
                EvalSample(
                    instruction=f"{lang}-{i}",
                    instruction_id=f"{lang}-{i}",
                    metadata={"lang": lang},
                )
            )
    return EvalDataset(
        name="toy",
        samples=samples,
        metadata_schema={"lang": "Language code"},
    )


def _make_pair_dataset(pairs: list[tuple[str, str]]) -> EvalDataset:
    samples: list[EvalSample] = []
    for i, (model_a, model_b) in enumerate(pairs):
        samples.append(
            EvalSample(
                instruction=f"{model_a}-{model_b}-{i}",
                instruction_id=f"pair-{i}",
                metadata={"model_a": model_a, "model_b": model_b},
            )
        )
    return EvalDataset(
        name="pairs",
        samples=samples,
        metadata_schema={
            "model_a": "Name of model A",
            "model_b": "Name of model B",
        },
    )


def test_balanced_sampling_redistributes_deficits():
    ds = _make_dataset({"a": 2, "b": 10, "c": 10, "d": 10})

    sampled = ds.sample_balanced(by="lang", n=20, seed=7)

    counts = Counter(s.metadata["lang"] for s in sampled.samples)
    assert len(sampled) == 20
    assert counts["a"] == 2  # saturated smaller class
    others = [counts[k] for k in ("b", "c", "d")]
    assert sum(others) == 18
    assert max(others) - min(others) <= 1


def test_balanced_sampling_supports_language_alias():
    ds = _make_dataset({"en": 20, "fr": 20, "de": 20})
    sampled = ds.sample_balanced(by="language", n=30, seed=1)
    counts = Counter(s.metadata["lang"] for s in sampled.samples)
    assert counts == {"en": 10, "fr": 10, "de": 10}


def test_balanced_sampling_supports_pairwise_model_appearances():
    ds = _make_pair_dataset(
        [
            ("A", "B"),
            ("A", "B"),
            ("A", "C"),
            ("A", "C"),
            ("B", "C"),
            ("B", "C"),
        ]
    )

    sampled = ds.sample_balanced(by="models", n=3, seed=7)
    counts = Counter()
    for sample in sampled.samples:
        counts[sample.metadata["model_a"]] += 1
        counts[sample.metadata["model_b"]] += 1

    assert len(sampled) == 3
    assert counts == {"A": 2, "B": 2, "C": 2}


def test_dataset_options_cache_key_includes_selection_knobs():
    opts = DatasetOptions(
        name="lmsys",
        n_instructions=2000,
        language="all",
        seed=13,
        balance_by="lang",
    )
    key = opts.cache_key()
    assert key.startswith("lmsys__sel__")
    assert "balance=lang" in key
    assert "seed=13" in key
