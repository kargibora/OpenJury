from __future__ import annotations

import math

import pytest

from openjury.models.factory import make_model
from openjury.rubrics import RubricDimension, RubricScorer, get_rubric


def _scorer_for_overall() -> RubricScorer:
    return RubricScorer(
        judge_model=make_model("Dummy/test-judge"),
        rubric=get_rubric("overall"),
    )


def test_rubric_dimension_normalizes_score_reference_keys_and_renders_prompt():
    dim = RubricDimension(
        name="coherence",
        description="Logical flow and structure.",
        scale_min=1,
        scale_max=7,
        score_references={"7": "Strong coherence", "1": "Incoherent"},
    )

    assert dim.score_references == {7: "Strong coherence", 1: "Incoherent"}

    block = dim.prompt_block()
    assert "Score references:" in block
    assert "- 7: Strong coherence" in block
    assert "- 1: Incoherent" in block


def test_default_rubric_prompt_includes_score_references():
    rubric = get_rubric("default")
    block = rubric.prompt_block()
    assert "Score references:" in block
    assert "Adherence" in block


def test_parse_scores_normalizes_title_cased_dimension_names():
    scorer = _scorer_for_overall()
    raw = """```json
{"Overall": 7}
```"""

    scores = scorer._parse_scores(raw)

    assert scores == {"overall": 7.0}


def test_parse_pairwise_json_and_merge_swapped_outputs():
    scorer = _scorer_for_overall()

    raw_ab = """```json
{
  "scores_A": {"Overall": 4},
  "scores_B": {"overall": 6},
  "preference": "B"
}
```"""
    raw_ba = """```json
{
  "scores_A": {"overall": 7},
  "scores_B": {"overall": 3},
  "preference": "A"
}
```"""

    parsed_ab = scorer._parse_pairwise(raw_ab)
    parsed_ba = scorer._parse_pairwise(raw_ba)
    merged = scorer._merge_swapped(parsed_ab, parsed_ba)

    assert parsed_ab["preference"] == 1.0
    assert parsed_ab["scores_A"]["overall"] == 4.0
    assert parsed_ab["scores_B"]["overall"] == 6.0

    assert merged["preference"] == pytest.approx(1.0)
    assert merged["scores_A"]["overall"] == pytest.approx(3.5)
    assert merged["scores_B"]["overall"] == pytest.approx(6.5)


def test_parse_pairwise_regex_fallback_patterns():
    scorer = _scorer_for_overall()
    raw = """
scores_A overall: 2
scores_B overall: 6
preference: B
"""

    parsed = scorer._parse_pairwise(raw)

    assert parsed["preference"] == 1.0
    assert parsed["scores_A"]["overall"] == 2.0
    assert parsed["scores_B"]["overall"] == 6.0


def test_parse_pairwise_legacy_scores_for_overall_rubric():
    scorer = RubricScorer(
        judge_model=make_model("Dummy/test-judge"),
        rubric=get_rubric("overall"),
        pairwise_prompt_style="legacy",
    )

    parsed = scorer._parse_pairwise(
        """
score_A: 8
score_B: 5
"""
    )

    assert parsed["preference"] == 0.0
    assert parsed["scores_A"] == {"overall": 8.0}
    assert parsed["scores_B"] == {"overall": 5.0}


def test_legacy_pairwise_requires_single_dimension_rubric():
    with pytest.raises(ValueError, match="single-dimension rubric"):
        RubricScorer(
            judge_model=make_model("Dummy/test-judge"),
            rubric=get_rubric("default"),
            pairwise_prompt_style="legacy",
        )


def test_parse_scores_returns_nan_when_unparseable():
    scorer = _scorer_for_overall()
    scores = scorer._parse_scores("not valid output")
    assert set(scores.keys()) == {"overall"}
    assert math.isnan(scores["overall"])
