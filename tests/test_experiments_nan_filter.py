import math

import pandas as pd

from experiments.common import _drop_nan_score_rows


def test_drop_nan_score_rows_removes_nan_preferences_and_nan_score_dict_rows():
    frame = pd.DataFrame(
        [
            {
                "judge_pref": 0.0,
                "human_pref": 1.0,
                "scores_a": {"clarity": 7.0, "fluency": 8.0},
                "scores_b": {"clarity": 6.0, "fluency": 7.0},
            },
            {
                "judge_pref": 0.5,
                "human_pref": 0.5,
                "scores_a": {"clarity": math.nan, "fluency": 8.0},
                "scores_b": {"clarity": 6.0, "fluency": 7.0},
            },
            {
                "judge_pref": math.nan,
                "human_pref": 0.0,
                "scores_a": {"clarity": 7.0, "fluency": 8.0},
                "scores_b": {"clarity": 6.0, "fluency": 7.0},
            },
            {
                "judge_pref": 1.0,
                "human_pref": math.nan,
                "scores_a": {"clarity": 7.0, "fluency": 8.0},
                "scores_b": {"clarity": 6.0, "fluency": 7.0},
            },
        ]
    )

    filtered, dropped = _drop_nan_score_rows(frame)

    assert dropped == 3
    assert len(filtered) == 1
    assert filtered.iloc[0]["judge_pref"] == 0.0
    assert filtered.iloc[0]["human_pref"] == 1.0
