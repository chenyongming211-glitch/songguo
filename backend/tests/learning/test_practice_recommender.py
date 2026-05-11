from __future__ import annotations

from songguo.backend.services.learning.practice_recommender import (
    build_similar_practice_items,
)


def test_read_conditions_arithmetic_practice_generates_real_items() -> None:
    items = build_similar_practice_items(
        knowledge_point="read_conditions",
        misconception_tag=None,
        limit=3,
        source_question="4*3+2*2=?",
    )

    assert len(items) == 3
    assert all(item.answer for item in items)
    assert all("请再输入" not in item.question for item in items)
    assert all("?" in item.question for item in items)


def test_times_five_source_question_keeps_same_multiplication_knowledge_point() -> None:
    items = build_similar_practice_items(
        knowledge_point="two_digit_times_one_digit",
        misconception_tag="treated_x5_like_x10",
        limit=3,
        source_question="4 x 5 = ?",
    )

    assert len(items) == 3
    assert all(item.knowledge_point == "two_digit_times_one_digit" for item in items)
    assert all("x 5" in item.question for item in items)
    assert all("+" not in item.question for item in items)


def test_division_with_remainder_generates_remainder_practice_items() -> None:
    items = build_similar_practice_items(
        knowledge_point="division_with_remainder",
        misconception_tag="math_division_quotient_too_large",
        limit=3,
    )

    assert len(items) == 3
    assert all(item.answer for item in items)
    assert all("还剩" in item.question for item in items)


def test_transfer_comparison_generates_same_knowledge_point_items() -> None:
    items = build_similar_practice_items(
        knowledge_point="比较问题",
        misconception_tag="misconception_1",
        limit=3,
        source_question="甲仓库有560袋米，乙仓库有420袋，甲运给乙80袋后，甲比乙少还是多多少袋？",
    )

    assert len(items) == 3
    assert all(item.answer for item in items)
    assert all("运给" in item.question or "给" in item.question or "调给" in item.question for item in items)
    assert all("比" in item.question for item in items)
    assert all(item.knowledge_point == "比较问题" for item in items)
    assert all("卡片" not in item.question and "单位" not in item.question for item in items)
