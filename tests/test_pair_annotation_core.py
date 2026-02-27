from __future__ import annotations

from types import SimpleNamespace

from openjury.common.pair_annotation import (
    PairSample,
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
