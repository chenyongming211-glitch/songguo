from __future__ import annotations

from songguo.backend.services.learning.ai_engine import DeterministicFallbackProvider
from songguo.backend.services.learning.memory_profile import build_learning_memory
from songguo.backend.services.learning.reporting import (
    build_learning_deposit,
    build_session_feedback,
    build_weekly_report,
)
from songguo.backend.services.learning.service import LearningService
from songguo.backend.services.learning.store import InMemoryLearningStore


def test_weekly_report_summarizes_wrong_questions_and_misconceptions() -> None:
    store = InMemoryLearningStore()
    service = LearningService(store=store)
    created = service.create_session(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="36 x 5 = ?",
    )
    service.submit_attempt(created.session_id, child_answer="360")

    report = build_weekly_report(store, child_id="child_001")

    assert report.child_id == "child_001"
    assert report.session_count == 1
    assert report.wrong_question_count == 1
    assert report.top_knowledge_points == ["two_digit_times_one_digit"]
    assert report.common_misconceptions == ["treated_x5_like_x10"]
    assert "每天练" in report.parent_summary
    assert "two_digit_times_one_digit" not in report.parent_summary
    assert "treated_x5_like_x10" not in report.parent_summary


def test_session_feedback_uses_teaching_asset_parent_explanation_for_m4_tags() -> None:
    store = InMemoryLearningStore()
    service = LearningService(
        store=store,
        ai_provider=DeterministicFallbackProvider(),
    )
    created = service.create_session(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="学校组织三年级学生春游，一共有4个班，每班32人。如果每辆大巴车限坐45人，那么至少需要多少辆大巴车？",
    )

    service.submit_attempt(created.session_id, child_answer="128")
    feedback = build_session_feedback(
        store,
        child_id="child_001",
        session_id=created.session_id,
    )

    assert feedback.main_error_reason == "math_capacity_stopped_at_total_count"
    assert "math_capacity" not in feedback.summary
    assert "孩子算出了总人数" in feedback.summary
    assert "先圈出总数、每组最多数" in feedback.parent_suggestion


def test_learning_deposit_turns_session_into_structured_learning_asset() -> None:
    store = InMemoryLearningStore()
    service = LearningService(
        store=store,
        ai_provider=DeterministicFallbackProvider(),
    )
    created = service.create_session(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="学校组织三年级学生春游，一共有4个班，每班32人。如果每辆大巴车限坐45人，那么至少需要多少辆大巴车？",
    )

    service.submit_attempt(created.session_id, child_answer="128")
    service.submit_attempt(created.session_id, child_answer="2")
    service.submit_attempt(created.session_id, child_answer="3")
    deposit = build_learning_deposit(
        store,
        child_id="child_001",
        session_id=created.session_id,
    )

    assert deposit.child_id == "child_001"
    assert deposit.session_id == created.session_id
    assert deposit.question_record.knowledge_point == "capacity_round_up"
    assert deposit.question_record.knowledge_point_label == "限载进一问题"
    assert deposit.mistake_record is not None
    assert deposit.mistake_record.student_answer == "2"
    assert deposit.mistake_record.is_correct is False
    assert deposit.mistake_record.main_error_reason == "math_capacity_ignored_remainder_round_up"
    assert 1 <= len(deposit.practice_records) <= 3
    assert all(item.answer for item in deposit.practice_records)
    assert deposit.student_memory.top_weaknesses[0].knowledge_point == "capacity_round_up"
    assert "限载进一问题" in deposit.parent_summary
    assert "math_capacity" not in deposit.parent_summary


def test_learning_deposit_prefers_persisted_structured_asset() -> None:
    store = InMemoryLearningStore()
    service = LearningService(store=store, ai_provider=DeterministicFallbackProvider())
    created = service.create_session(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="36 x 5 = ?",
    )
    store.save_learning_deposit(
        created.session_id,
        {
            "child_id": "child_001",
            "session_id": created.session_id,
            "question_record": {
                "question_text": "36 x 5 = ?",
                "subject": "math",
                "grade": 3,
                "knowledge_point": "two_digit_times_one_digit",
                "knowledge_point_label": "两位数乘一位数",
                "question_type": "llm_structured_type",
            },
            "mistake_record": None,
            "practice_records": [],
            "student_memory": build_learning_memory(store, child_id="child_001").model_dump(mode="json"),
            "parent_summary": "这是 LLM 结构化沉淀后的家长摘要。",
        },
    )

    deposit = build_learning_deposit(
        store,
        child_id="child_001",
        session_id=created.session_id,
    )

    assert deposit.question_record.question_type == "llm_structured_type"
    assert deposit.parent_summary == "这是 LLM 结构化沉淀后的家长摘要。"
