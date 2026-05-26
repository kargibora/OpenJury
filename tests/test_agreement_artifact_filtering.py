import math

from openjury.arena.config import MatchResult
from openjury.pipelines.agreement import _drop_nan_agreement_rows


def test_drop_nan_agreement_rows_removes_swapped_nan_scores():
    rows = [
        {
            "human_pref": 0.5,
            "judge_pref": 0.5,
            "scores_a": {"adherence": 7.0},
            "scores_b": {"adherence": 6.0},
            "scores_a_swapped": {"adherence": float("nan")},
            "scores_b_swapped": {"adherence": float("nan")},
        },
        {
            "human_pref": 0.0,
            "judge_pref": 0.0,
            "scores_a": {"adherence": 8.0},
            "scores_b": {"adherence": 5.0},
        },
    ]

    kept, dropped = _drop_nan_agreement_rows(rows)

    assert dropped == 1
    assert len(kept) == 1
    assert kept[0]["judge_pref"] == 0.0


def test_match_result_to_dict_always_includes_completion_lengths():
    result = MatchResult(
        model_a="a",
        model_b="b",
        instruction_index=0,
        scores_a={"adherence": 7.0},
        scores_b={"adherence": 6.0},
        preference=0.0,
        completion_a="hello",
        completion_b="world!",
    )

    payload = result.to_dict()

    assert payload["len_a"] == 5
    assert payload["len_b"] == 6
    assert not math.isnan(payload["len_a"])
    assert "completion_a" not in payload
    assert "completion_b" not in payload
