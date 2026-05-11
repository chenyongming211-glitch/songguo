from __future__ import annotations

from songguo.backend.services.learning.ai_engine import (
    AIEngineContext,
    DeterministicFallbackProvider,
)
from songguo.backend.services.learning.models import LearningPhase
from songguo.backend.services.learning.runtime_kernel import RuntimeKernel
from songguo.backend.services.learning.store import InMemoryLearningStore


BUS_QUESTION = "学校组织三年级学生春游，一共有4个班，每班32人。如果每辆大巴车限坐45人，那么至少需要多少辆大巴车？"
TECH_MUSEUM_QUESTION = "三年级去参观科技馆，有5个班，每班28人。每辆车最多坐40人，至少需要几辆车？"


def test_runtime_kernel_creates_structured_session_and_records_key_point_release() -> None:
    store = InMemoryLearningStore()
    kernel = RuntimeKernel(store=store, provider=DeterministicFallbackProvider())

    created = kernel.create_math_session(
        child_id="child_001",
        grade=3,
        question_text=BUS_QUESTION,
        context=AIEngineContext(child_id="child_001", session_id=None),
    )
    session = store.require_session(created.session_id)
    events = store.list_events(created.session_id)

    assert created.teaching_progress.mode == "dynamic_key_points"
    assert session.current_key_point_id == "kp_total_people"
    assert "3辆" not in created.message
    assert "key_point.released" in [event.event_type for event in events]
    assert events[-1].payload["skill_ids"]
    assert events[-1].payload["concept_card_ids"]


def test_runtime_kernel_advances_only_after_current_key_point_progress() -> None:
    store = InMemoryLearningStore()
    kernel = RuntimeKernel(store=store, provider=DeterministicFallbackProvider())
    created = kernel.create_math_session(
        child_id="child_001",
        grade=3,
        question_text=BUS_QUESTION,
        context=AIEngineContext(child_id="child_001", session_id=None),
    )

    wrong = kernel.submit_math_attempt(created.session_id, child_answer="45")
    first = kernel.submit_math_attempt(created.session_id, child_answer="128")

    assert wrong.next_key_point_id is None
    assert wrong.teaching_progress.current_label == "先求总人数"
    assert first.partially_correct is True
    assert first.next_key_point_id == "kp_capacity_check"
    assert first.teaching_progress.current_label == "判断车辆容量"
    assert "3辆" not in first.message


def test_runtime_kernel_releases_concept_card_when_child_is_stuck() -> None:
    store = InMemoryLearningStore()
    kernel = RuntimeKernel(store=store, provider=DeterministicFallbackProvider())
    created = kernel.create_math_session(
        child_id="child_001",
        grade=3,
        question_text=BUS_QUESTION,
        context=AIEngineContext(child_id="child_001", session_id=None),
    )

    kernel.submit_math_attempt(created.session_id, child_answer="45")
    stuck = kernel.submit_math_attempt(created.session_id, child_answer="45")
    events = store.list_events(created.session_id)

    assert stuck.correct is False
    assert stuck.next_key_point_id is None
    assert "方法卡" in stuck.message
    assert "如果还有人或物没有被装下" in stuck.message
    assert "3辆" not in stuck.message
    assert "concept_card.released" in [event.event_type for event in events]


def test_runtime_kernel_generates_answered_practice_from_skill_and_misconception() -> None:
    store = InMemoryLearningStore()
    kernel = RuntimeKernel(store=store, provider=DeterministicFallbackProvider())
    created = kernel.create_math_session(
        child_id="child_001",
        grade=3,
        question_text=BUS_QUESTION,
        context=AIEngineContext(child_id="child_001", session_id=None),
    )

    kernel.submit_math_attempt(created.session_id, child_answer="128")
    kernel.submit_math_attempt(created.session_id, child_answer="2")
    completed = kernel.submit_math_attempt(created.session_id, child_answer="3")
    practice_event = [
        event for event in store.list_events(created.session_id) if event.event_type == "practice.generated"
    ][-1]

    assert completed.correct is True
    assert 1 <= len(completed.practice_items) <= 3
    assert all(item.answer for item in completed.practice_items)
    assert all("至少" in item.question for item in completed.practice_items)
    assert practice_event.payload["misconception_id"] == "math_capacity_ignored_remainder_round_up"
    assert "math_capacity_round_up" in practice_event.payload["skill_ids"]


def test_runtime_kernel_keeps_completed_session_in_practice_phase() -> None:
    store = InMemoryLearningStore()
    kernel = RuntimeKernel(store=store, provider=DeterministicFallbackProvider())
    created = kernel.create_math_session(
        child_id="child_001",
        grade=3,
        question_text=BUS_QUESTION,
        context=AIEngineContext(child_id="child_001", session_id=None),
    )

    kernel.submit_math_attempt(created.session_id, child_answer="128")
    kernel.submit_math_attempt(created.session_id, child_answer="2")
    completed = kernel.submit_math_attempt(created.session_id, child_answer="3")
    after_completion = kernel.submit_math_attempt(created.session_id, child_answer="继续")
    session = store.require_session(created.session_id)

    assert completed.phase == LearningPhase.SIMILAR_PRACTICE.value
    assert after_completion.phase == LearningPhase.SIMILAR_PRACTICE.value
    assert session.phase == LearningPhase.SIMILAR_PRACTICE
    assert session.completed_at is not None
    assert "先不急着回答几辆车" not in after_completion.message
    assert "同类题" in after_completion.message


def test_runtime_kernel_gives_adaptive_feedback_instead_of_repeating_prompt() -> None:
    store = InMemoryLearningStore()
    kernel = RuntimeKernel(store=store, provider=DeterministicFallbackProvider())
    created = kernel.create_math_session(
        child_id="child_001",
        grade=3,
        question_text=TECH_MUSEUM_QUESTION,
        context=AIEngineContext(child_id="child_001", session_id=None),
    )

    kernel.submit_math_attempt(created.session_id, child_answer="140")
    before = store.require_session(created.session_id).current_prompt
    response = kernel.submit_math_attempt(created.session_id, child_answer="2辆对吗")

    assert response.correct is False
    assert response.message != before
    assert "2辆" in response.message
    assert "80人" in response.message
    assert "还差" in response.message


def test_runtime_kernel_does_not_return_original_question_as_practice_item() -> None:
    store = InMemoryLearningStore()
    kernel = RuntimeKernel(store=store, provider=DeterministicFallbackProvider())
    created = kernel.create_math_session(
        child_id="child_001",
        grade=3,
        question_text=TECH_MUSEUM_QUESTION,
        context=AIEngineContext(child_id="child_001", session_id=None),
    )

    kernel.submit_math_attempt(created.session_id, child_answer="140")
    kernel.submit_math_attempt(created.session_id, child_answer="3")
    completed = kernel.submit_math_attempt(created.session_id, child_answer="4")

    assert completed.correct is True
    assert completed.practice_items
    assert all(item.question != TECH_MUSEUM_QUESTION for item in completed.practice_items)
