"""Default rubric definitions and registry.

Ships with three built-in rubrics:
    - **default**: General-purpose instruction-following evaluation
    - **coding**: Code generation quality evaluation
    - **translation**: Translation quality evaluation

Custom rubrics can be registered at runtime or loaded from JSON.
"""

from __future__ import annotations

from openjury.rubrics.schema import Rubric, RubricDimension


# ── Default Dimensions ──────────────────────────────────────────────

DEFAULT_DIMENSIONS = [
    RubricDimension(
        name="adherence",
        description=(
            "Follows the user's instructions and constraints precisely: required format, scope, "
            "style constraints, and any do/don't requirements. Penalize missing requested parts "
            "or deviating from constraints."
        ),
    ),
    RubricDimension(
        name="helpfulness",
        description=(
            "Advances the user's goal with relevant, actionable content. Provides useful steps, "
            "options, or explanations tailored to the request. Penalize generic filler or "
            "non-responsive content."
        ),
    ),
    RubricDimension(
        name="factuality",
        description=(
            "Information is correct and appropriately qualified. Avoids hallucinations and "
            "unwarranted specifics. If uncertain, expresses uncertainty and does not fabricate "
            "sources, citations, or details."
        ),
    ),
    RubricDimension(
        name="completeness",
        description=(
            "Covers the key aspects of the request without major omissions. Addresses all "
            "sub-questions and important constraints. Penalize partial answers or skipped items."
        ),
    ),
    RubricDimension(
        name="clarity",
        description=(
            "Well-organized, easy to follow, and unambiguous. Uses logical structure, headings/"
            "lists when helpful, and clear references. Penalize confusing organization or ambiguity."
        ),
    ),
    RubricDimension(
        name="fluency",
        description=(
            "Language and presentation quality: fluent, readable, appropriately concise, and "
            "well-formatted. Tone is appropriate for the user/context."
        ),
    ),
]

DEFAULT_RUBRIC = Rubric(
    name="default",
    dimensions=DEFAULT_DIMENSIONS,
    description="General-purpose rubric for instruction-following evaluation.",
)

CODING_RUBRIC = Rubric(
    name="coding",
    dimensions=[
        RubricDimension(
            name="correctness",
            description="Does the code compile/run and produce the correct output?",
        ),
        RubricDimension(
            name="adherence",
            description="Does the code follow the user's specifications exactly?",
        ),
        RubricDimension(
            name="readability",
            description="Is the code well-structured, documented, and easy to understand?",
        ),
        RubricDimension(
            name="efficiency",
            description="Is the solution reasonably efficient in time and space complexity?",
        ),
        RubricDimension(
            name="explanation",
            description="Is the accompanying explanation clear and helpful?",
        ),
    ],
    description="Rubric for evaluating code generation tasks.",
)

TRANSLATION_RUBRIC = Rubric(
    name="translation",
    dimensions=[
        RubricDimension(
            name="fluency",
            description="Is the translation natural-sounding in the target language?",
        ),
        RubricDimension(
            name="accuracy",
            description="Does the translation preserve the meaning of the source text?",
        ),
        RubricDimension(
            name="terminology",
            description="Are domain-specific terms translated correctly and consistently?",
        ),
        RubricDimension(
            name="style",
            description="Does the translation preserve the tone and register of the source?",
        ),
    ],
    description="Rubric for evaluating translation quality.",
)

OVERALL_RUBRIC = Rubric(
    name="overall",
    dimensions=[
        RubricDimension(
            name="overall",
            description=(
                "Overall quality of the response, considering all relevant "
                "factors. This is a holistic judgment of how well the response "
                "meets the user's needs and expectations."
            ),
        )
    ],
    description="Single-dimension rubric for overall quality assessment.",
)


# ── Registry ────────────────────────────────────────────────────────

RUBRIC_REGISTRY: dict[str, Rubric] = {
    "default": DEFAULT_RUBRIC,
    "coding": CODING_RUBRIC,
    "translation": TRANSLATION_RUBRIC,
    "overall": OVERALL_RUBRIC,
}

# Convenience alias
DEFAULT_RUBRICS = RUBRIC_REGISTRY


def get_rubric(name: str) -> Rubric:
    """Look up a rubric by name.

    Args:
        name: Rubric identifier (e.g. "default", "coding", "translation").

    Returns:
        The Rubric object.

    Raises:
        KeyError: If the rubric name is not registered.
    """
    if name not in RUBRIC_REGISTRY:
        available = ", ".join(sorted(RUBRIC_REGISTRY.keys()))
        raise KeyError(f"Unknown rubric '{name}'. Available: {available}")
    return RUBRIC_REGISTRY[name]


def register_rubric(rubric: Rubric) -> None:
    """Register a custom rubric so it can be looked up by name.

    Args:
        rubric: Rubric object with a unique ``name`` attribute.
    """
    RUBRIC_REGISTRY[rubric.name] = rubric
