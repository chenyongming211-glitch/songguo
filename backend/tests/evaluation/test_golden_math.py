from __future__ import annotations

from songguo.backend.evaluation.golden_math import (
    build_golden_math_questions,
)
from songguo.backend.evaluation.runner import evaluate_math_provider
from songguo.backend.services.learning.ai_engine import DeterministicFallbackProvider


def test_golden_math_set_has_100_questions_and_required_categories() -> None:
    questions = build_golden_math_questions()
    categories = {item.category for item in questions}
    grades = {item.grade for item in questions}

    assert len(questions) == 100
    assert len({item.question_id for item in questions}) == 100
    assert {
        "calculation",
        "multi_digit_division",
        "word_problem",
        "unit_conversion",
        "time_money",
        "geometry",
        "area_volume",
        "remainder_division",
        "capacity_round_up",
        "fraction_decimal",
        "fraction_operations",
        "decimal_operations",
        "percent_ratio",
        "ratio_proportion",
        "average",
        "equation",
        "statistics",
        "factor_multiple",
        "angle_geometry",
        "mixed_operations",
        "distractor_condition",
    }.issubset(categories)
    assert len(categories) >= 20
    assert grades == {3, 4, 5, 6}
    assert all(item.skill_ids for item in questions)


def test_golden_math_questions_are_interleaved_for_representative_samples() -> None:
    questions = build_golden_math_questions()
    first_ten = questions[:10]

    assert len({item.category for item in first_ten}) == 10
    assert {item.grade for item in first_ten} == {3, 4, 5, 6}


def test_evaluation_runner_reports_schema_and_leakage_metrics() -> None:
    report = evaluate_math_provider(
        provider=DeterministicFallbackProvider(),
        questions=build_golden_math_questions(),
    )

    assert report.total == 100
    assert report.schema_failures == 0
    assert report.answer_leakage_count == 0
    assert report.provider_failures == 0
    assert report.skill_hit_rate > 0
    assert report.concept_card_hit_rate > 0
    assert report.controlled_generation_failures == 0
    assert report.practice_generation_failures == 0
