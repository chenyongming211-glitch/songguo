from __future__ import annotations

from songguo.backend.services.learning.tutor_graph.math_mistake_nodes import (
    math_problem_parse_node,
    rule_judge_node,
)
from songguo.backend.services.learning.tutor_graph.state import MathMistakeTutorGraphState


BUS_QUESTION = "学校组织三年级学生春游，一共有4个班，每班32人。如果每辆大巴车限坐45人，那么至少需要多少辆大巴车？"


def test_math_problem_parse_node_structures_capacity_round_up_question() -> None:
    state = MathMistakeTutorGraphState(
        child_id="child_001",
        session_id="pending",
        grade=3,
        question_text=BUS_QUESTION,
    )

    parsed = math_problem_parse_node(state)

    assert parsed.problem_analysis is not None
    assert parsed.problem_analysis["problem_type"] == "capacity_round_up"
    assert parsed.problem_analysis["final_answer"] == "3辆"


def test_rule_judge_node_prioritizes_deterministic_correct_answer() -> None:
    parsed = math_problem_parse_node(
        MathMistakeTutorGraphState(
            child_id="child_001",
            session_id="pending",
            grade=3,
            question_text=BUS_QUESTION,
        )
    )

    judged = rule_judge_node(parsed, child_answer="3")

    assert judged.rule_judge_result is not None
    assert judged.rule_judge_result["determined"] is True
    assert judged.rule_judge_result["correct"] is True
    assert judged.rule_judge_result["expected_answer"] == "3辆"


def test_rule_judge_node_identifies_capacity_round_up_wrong_vehicle_count() -> None:
    parsed = math_problem_parse_node(
        MathMistakeTutorGraphState(
            child_id="child_001",
            session_id="pending",
            grade=3,
            question_text=BUS_QUESTION,
        )
    )

    judged = rule_judge_node(parsed, child_answer="2")

    assert judged.rule_judge_result is not None
    assert judged.rule_judge_result["determined"] is True
    assert judged.rule_judge_result["correct"] is False
    assert judged.rule_judge_result["misconception_tag"] == "math_capacity_ignored_remainder_round_up"


def test_rule_judge_node_trusts_safe_arithmetic_expression_answer() -> None:
    parsed = math_problem_parse_node(
        MathMistakeTutorGraphState(
            child_id="child_001",
            session_id="pending",
            grade=3,
            question_text="4*3+2*2=?",
        )
    )

    judged = rule_judge_node(parsed, child_answer="16")

    assert judged.rule_judge_result is not None
    assert judged.rule_judge_result["determined"] is True
    assert judged.rule_judge_result["correct"] is True
    assert judged.rule_judge_result["expected_answer"] == "16"


def test_rule_judge_node_accepts_quotient_and_remainder_answer() -> None:
    state = MathMistakeTutorGraphState(
        child_id="child_001",
        session_id="pending",
        grade=3,
        question_text="36颗松果平均分给5只松鼠，每只最多分几颗，还剩几颗？",
        problem_analysis={
            "subject": "math",
            "grade": 3,
            "problem_type": "remainder_division",
            "target": "每只最多分几颗，还剩几颗",
            "final_answer": "每只7颗，还剩1颗",
            "confidence": 0.9,
            "source": "llm_problem_parse",
        },
    )

    judged = rule_judge_node(state, child_answer="36÷5=7余1，所以每只7颗，还剩1颗")

    assert judged.rule_judge_result is not None
    assert judged.rule_judge_result["determined"] is True
    assert judged.rule_judge_result["correct"] is True
    assert judged.rule_judge_result["expected_answer"] == "每只7颗，还剩1颗"


def test_rule_judge_node_accepts_equivalent_length_units() -> None:
    state = MathMistakeTutorGraphState(
        child_id="child_001",
        session_id="pending",
        grade=3,
        question_text="一根彩带2米35厘米，剪去80厘米，还剩多少厘米？",
        problem_analysis={
            "subject": "math",
            "grade": 3,
            "problem_type": "length_conversion",
            "target": "还剩多少厘米",
            "final_answer": "1米55厘米",
            "confidence": 0.9,
            "source": "llm_problem_parse",
        },
    )

    judged = rule_judge_node(state, child_answer="155厘米")

    assert judged.rule_judge_result is not None
    assert judged.rule_judge_result["correct"] is True


def test_rule_judge_node_accepts_yes_no_semantic_answer() -> None:
    state = MathMistakeTutorGraphState(
        child_id="child_001",
        session_id="pending",
        grade=3,
        question_text="一本故事书168页，已经读75页，剩下每天读31页，3天能读完吗？",
        problem_analysis={
            "subject": "math",
            "grade": 3,
            "problem_type": "reading_plan",
            "target": "3天能不能读完",
            "final_answer": "能，3天正好读完",
            "confidence": 0.9,
            "source": "llm_problem_parse",
        },
    )

    judged = rule_judge_node(state, child_answer="能读完")

    assert judged.rule_judge_result is not None
    assert judged.rule_judge_result["correct"] is True


def test_rule_judge_node_rejects_opposite_comparison_direction() -> None:
    state = MathMistakeTutorGraphState(
        child_id="child_001",
        session_id="pending",
        grade=4,
        question_text="甲仓库有560袋米，乙仓库有420袋，甲运给乙80袋后，甲比乙少还是多多少袋？",
        problem_analysis={
            "subject": "math",
            "grade": 4,
            "problem_type": "multi_step_total_difference",
            "target": "甲比乙少还是多多少袋",
            "final_answer": "甲少20袋",
            "confidence": 0.9,
            "source": "llm_problem_parse",
        },
    )

    judged = rule_judge_node(state, child_answer="甲多20袋")

    assert judged.rule_judge_result is not None
    assert judged.rule_judge_result["correct"] is False


def test_rule_judge_node_classifies_large_quotient_in_remainder_division() -> None:
    state = MathMistakeTutorGraphState(
        child_id="child_001",
        session_id="pending",
        grade=3,
        question_text="36颗松果平均分给5只松鼠，每只最多分几颗，还剩几颗？",
        problem_analysis={
            "subject": "math",
            "grade": 3,
            "problem_type": "division_with_remainder",
            "target": "每只最多分几颗，还剩几颗",
            "final_answer": "每只7颗，还剩1颗",
            "confidence": 0.9,
            "source": "llm_problem_parse",
        },
    )

    judged = rule_judge_node(state, child_answer="每只8颗")

    assert judged.rule_judge_result is not None
    assert judged.rule_judge_result["correct"] is False
    assert judged.rule_judge_result["misconception_tag"] == "math_division_quotient_too_large"
