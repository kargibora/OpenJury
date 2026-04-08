from __future__ import annotations

from types import SimpleNamespace

from openjury.arena.config import ModelScore
from openjury.common.pair_annotation import (
    PairSample,
    PairwiseCacheConfig,
    derive_preference_from_scores,
    score_pairs_pairwise,
    score_pairs_samplewise,
)


def test_derive_preference_from_scores_prefers_higher_weighted_average():
    weights = {"a": 0.7, "b": 0.3}
    pref = derive_preference_from_scores(
        {"a": 7.0, "b": 5.0},
        {"a": 5.0, "b": 7.0},
        weights,
    )
    assert pref == 0.0


def test_score_pairs_pairwise_without_cache():
    class _FakeScorer:
        def score_pairwise(self, *, instructions, completions_A, completions_B, swap_to_debias, use_tqdm):
            assert len(instructions) == 2
            assert swap_to_debias is True
            return [
                SimpleNamespace(
                    scores_A={"quality": 7.0},
                    scores_B={"quality": 5.0},
                    preference=0.0,
                    raw_judge_output="raw-a",
                    raw_judge_output_swapped="raw-b",
                ),
                SimpleNamespace(
                    scores_A={"quality": 4.0},
                    scores_B={"quality": 6.0},
                    preference=1.0,
                    raw_judge_output="raw-a2",
                    raw_judge_output_swapped="raw-b2",
                ),
            ]

    pairs = [
        PairSample(
            sample_id="s1",
            instruction_index=0,
            instruction="i1",
            model_a="A",
            model_b="B",
            completion_a="ca1",
            completion_b="cb1",
        ),
        PairSample(
            sample_id="s2",
            instruction_index=1,
            instruction="i2",
            model_a="A",
            model_b="B",
            completion_a="ca2",
            completion_b="cb2",
        ),
    ]

    out = score_pairs_pairwise(
        scorer=_FakeScorer(),
        pairs=pairs,
        swap_to_debias=True,
        use_tqdm=False,
        cache_config=None,
        ignore_cache=True,
    )
    assert [o.preference for o in out] == [0.0, 1.0]
    assert out[0].scores_a == {"quality": 7.0}
    assert out[1].scores_b == {"quality": 6.0}


def test_score_pairs_samplewise_without_cache():
    class _FakeScorer:
        def score(self, *, instructions, completions, model_name, use_tqdm):
            assert len(instructions) == len(completions) == 2
            if model_name == "side_A":
                score = 7.0
            else:
                score = 5.0
            return [
                SimpleNamespace(scores={"quality": score}, raw_judge_output=f"raw-{model_name}-1"),
                SimpleNamespace(scores={"quality": score}, raw_judge_output=f"raw-{model_name}-2"),
            ]

    pairs = [
        PairSample(
            sample_id="s1",
            instruction_index=0,
            instruction="i1",
            model_a="A",
            model_b="B",
            completion_a="ca1",
            completion_b="cb1",
        ),
        PairSample(
            sample_id="s2",
            instruction_index=1,
            instruction="i2",
            model_a="A",
            model_b="B",
            completion_a="ca2",
            completion_b="cb2",
        ),
    ]

    out = score_pairs_samplewise(
        scorer=_FakeScorer(),
        pairs=pairs,
        dimension_weights={"quality": 1.0},
        use_tqdm=False,
        cache_config=None,
        ignore_cache=True,
    )
    assert [o.preference for o in out] == [0.0, 0.0]
    assert out[0].model_a == "A"
    assert out[1].model_b == "B"


def test_score_pairs_pairwise_reuses_swap_cache_for_no_swap(monkeypatch):
    class _FakeScorer:
        def score_pairwise(self, **_kwargs):
            raise AssertionError("Judge inference should not run on cache hit")

        def _parse_pairwise(self, raw_output: str):
            if raw_output == "raw-ab":
                return {
                    "scores_A": {"quality": 9.0},
                    "scores_B": {"quality": 2.0},
                    "preference": 0.0,
                }
            raise AssertionError("Unexpected raw output in test fixture")

    pairs = [
        PairSample(
            sample_id="s1",
            instruction_index=0,
            instruction="i1",
            model_a="A",
            model_b="B",
            completion_a="ca1",
            completion_b="cb1",
        )
    ]

    swap_on_cached = [
        ModelScore(
            model="A",
            instruction_index=0,
            sample_id="s1",
            scores={"quality": 5.0, "__preference__": 0.5},
            completion="ca1",
            raw_judge_output="raw-ab",
        ),
        ModelScore(
            model="B",
            instruction_index=0,
            sample_id="s1",
            scores={"quality": 5.0},
            completion="cb1",
            raw_judge_output="raw-ba",
        ),
    ]

    get_calls: list[str] = []
    put_models: list[str] = []

    def _fake_get(*, model: str, **_kwargs):
        get_calls.append(model)
        if model == "pairwise_swap_off":
            return None
        if model == "pairwise_swap_on":
            return swap_on_cached
        return None

    def _fake_put(model_scores, *, model: str, **_kwargs):
        put_models.append(model)
        assert len(model_scores) == 2
        return None

    monkeypatch.setattr("openjury.common.pair_annotation.score_cache.get", _fake_get)
    monkeypatch.setattr("openjury.common.pair_annotation.score_cache.put", _fake_put)

    out = score_pairs_pairwise(
        scorer=_FakeScorer(),
        pairs=pairs,
        swap_to_debias=False,
        use_tqdm=False,
        cache_config=PairwiseCacheConfig(
            judge="Judge/X",
            criteria="default",
            model_key="pairwise_swap_off",
            dataset_exact="lmsys__agreement",
            n_instructions=1,
        ),
        ignore_cache=False,
        cache_fallback_model_keys=["pairwise_swap_on"],
        prefer_raw_ab_from_cache=True,
    )

    assert get_calls == ["pairwise_swap_off", "pairwise_swap_on"]
    assert put_models == ["pairwise_swap_off"]
    assert len(out) == 1
    assert out[0].scores_a == {"quality": 9.0}
    assert out[0].scores_b == {"quality": 2.0}
    assert out[0].preference == 0.0
    assert out[0].raw_judge_output == "raw-ab"
    assert out[0].raw_judge_output_swapped is None
