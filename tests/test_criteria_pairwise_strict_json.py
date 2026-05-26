import math

import pytest

from openjury.criteria import CriteriaScorer, get_criteria


def _make_scorer() -> CriteriaScorer:
    return CriteriaScorer(judge_model=object(), criteria=get_criteria("default"))


def test_pairwise_parser_accepts_valid_json():
    scorer = _make_scorer()
    raw = """```json
{
  "scores_A": {
    "adherence": 7,
    "helpfulness": 8,
    "factuality": 6,
    "completeness": 9,
    "clarity": 7,
    "fluency": 8
  },
  "scores_B": {
    "adherence": 6,
    "helpfulness": 7,
    "factuality": 5,
    "completeness": 8,
    "clarity": 6,
    "fluency": 7
  },
  "preference": "A"
}
```"""

    parsed = scorer._parse_pairwise(raw)

    assert parsed["preference"] == 0.0
    assert parsed["scores_A"]["adherence"] == 7.0
    assert parsed["scores_B"]["fluency"] == 7.0


def test_pairwise_parser_accepts_json_followed_by_explanation():
    scorer = _make_scorer()
    raw = """```json
{
  "scores_A": {
    "adherence": 7,
    "helpfulness": 8,
    "factuality": 6,
    "completeness": 9,
    "clarity": 7,
    "fluency": 8
  },
  "scores_B": {
    "adherence": 6,
    "helpfulness": 7,
    "factuality": 5,
    "completeness": 8,
    "clarity": 6,
    "fluency": 7
  },
  "preference": "A"
}
```

Reasoning:
- A is slightly more complete and clearer.
- B is solid but less detailed.
"""

    parsed = scorer._parse_pairwise(raw)

    assert parsed["preference"] == 0.0
    assert parsed["scores_A"]["completeness"] == 9.0
    assert parsed["scores_B"]["clarity"] == 6.0


def test_pairwise_parser_rejects_numbered_prose_with_parenthesized_scores():
    scorer = _make_scorer()
    raw = """
Reasoning:
1) Completion A seems stronger overall.
2) Completion B is weaker.

Adherence: (1)
Helpfulness: (1)
Factuality: (1)
Completeness: (1)
Clarity: (1)
Fluency: (1)

Preference: A
"""

    parsed = scorer._parse_pairwise(raw)

    assert parsed["preference"] == 0.5
    assert all(math.isnan(v) for v in parsed["scores_A"].values())
    assert all(math.isnan(v) for v in parsed["scores_B"].values())


def test_coerce_score_value_requires_strict_numeric_strings():
    scorer = _make_scorer()

    assert scorer._coerce_score_value("1") == 1.0
    assert scorer._coerce_score_value(" 7.5 ") == 7.5

    with pytest.raises(ValueError):
        scorer._coerce_score_value("(1)")

    with pytest.raises(ValueError):
        scorer._coerce_score_value("score: 1")


def test_pairwise_prompt_requests_json_before_explanation():
    scorer = CriteriaScorer(
        judge_model=object(),
        criteria=get_criteria("default"),
        provide_explanation=True,
    )

    assert "Output the JSON scores first" in scorer.system_prompt["pairwise"]
    assert "Before providing scores" not in scorer.system_prompt["pairwise"]
    assert "MUST begin with ```json on the first line" in scorer.system_prompt["pairwise"]
