from __future__ import annotations

from collections import Counter, defaultdict

from songguo.backend.evaluation.submission_application_math import (
    build_submission_application_questions,
)


def test_submission_application_set_covers_100_grade_3_to_6_word_problems() -> None:
    questions = build_submission_application_questions()

    assert len(questions) == 100
    assert Counter(question.grade for question in questions) == {3: 25, 4: 25, 5: 25, 6: 25}
    assert all(question.question_text != question.correct_answer for question in questions)
    assert all(question.correct_answer != question.wrong_answer for question in questions)
    assert all("?" not in question.correct_answer for question in questions)


def test_each_grade_has_distinct_application_categories() -> None:
    questions = build_submission_application_questions()
    categories_by_grade: dict[int, set[str]] = defaultdict(set)
    for question in questions:
        categories_by_grade[question.grade].add(question.category)

    assert set(categories_by_grade) == {3, 4, 5, 6}
    for categories in categories_by_grade.values():
        assert len(categories) == 25

