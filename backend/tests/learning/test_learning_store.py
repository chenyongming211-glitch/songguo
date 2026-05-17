from __future__ import annotations

from songguo.backend.services.learning.models import LearningPhase
from songguo.backend.services.learning.store import InMemoryLearningStore


def test_create_session_persists_session_created_event() -> None:
    store = InMemoryLearningStore()

    session = store.create_session(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="36 x 5 = ?",
        normalized_question="36 x 5 = ?",
        knowledge_point="two_digit_times_one_digit",
        current_prompt="What is 36 x 10?",
    )

    events = store.list_events(session.session_id)
    assert session.phase == LearningPhase.WAIT_CHILD_ATTEMPT
    assert session.hint_level == 1
    assert session.answer_unlocked is False
    assert len(events) == 1
    assert events[0].event_type == "session.created"
    assert events[0].payload["question_text"] == "36 x 5 = ?"


def test_resume_snapshot_uses_business_state_not_events_only() -> None:
    store = InMemoryLearningStore()
    session = store.create_session(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="36 x 5 = ?",
        normalized_question="36 x 5 = ?",
        knowledge_point="two_digit_times_one_digit",
        current_prompt="What is 36 x 10?",
    )

    store.update_session(
        session.session_id,
        hint_level=2,
        attempt_count=1,
        current_prompt="36 x 5 is half of 36 x 10. What is half of 360?",
        last_misconception="treated_x5_like_x10",
    )

    snapshot = store.get_resume_snapshot(session.session_id)

    assert snapshot.session_id == session.session_id
    assert snapshot.phase == LearningPhase.WAIT_CHILD_ATTEMPT
    assert snapshot.hint_level == 2
    assert snapshot.attempt_count == 1
    assert snapshot.answer_unlocked is False
    assert snapshot.current_prompt == "36 x 5 is half of 36 x 10. What is half of 360?"
    assert snapshot.last_misconception == "treated_x5_like_x10"


def test_resume_snapshot_reconstructs_recent_child_and_assistant_turns() -> None:
    store = InMemoryLearningStore()
    session = store.create_session(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="36 x 5 = ?",
        normalized_question="36 x 5 = ?",
        knowledge_point="two_digit_times_one_digit",
        current_prompt="What is 36 x 10?",
    )
    store.append_event(
        session_id=session.session_id,
        child_id="child_001",
        event_type="key_point.released",
        payload={"prompt": "What is 36 x 10?"},
    )
    store.append_event(
        session_id=session.session_id,
        child_id="child_001",
        event_type="child.attempt_submitted",
        payload={"child_answer": "360"},
    )
    store.append_event(
        session_id=session.session_id,
        child_id="child_001",
        event_type="key_point.released",
        payload={"prompt": "360 是乘以 10。乘以 5 是它的一半。"},
    )
    store.append_event(
        session_id=session.session_id,
        child_id="child_001",
        event_type="practice.followup_requested",
        payload={"child_answer": "继续"},
    )
    store.update_session(
        session.session_id,
        hint_level=2,
        attempt_count=1,
        current_prompt="360 是乘以 10。乘以 5 是它的一半。",
    )

    snapshot = store.get_resume_snapshot(session.session_id)

    assert [message["role"] for message in snapshot.last_messages] == [
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert snapshot.last_messages[1]["content"] == "360"
    assert snapshot.last_messages[3]["content"] == "继续"


def test_record_wrong_question_keeps_highest_hint_level_and_misconception() -> None:
    store = InMemoryLearningStore()
    session = store.create_session(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="36 x 5 = ?",
        normalized_question="36 x 5 = ?",
        knowledge_point="two_digit_times_one_digit",
        current_prompt="What is 36 x 10?",
    )

    wrong_question = store.record_wrong_question(
        session_id=session.session_id,
        child_id="child_001",
        normalized_question="36 x 5 = ?",
        knowledge_point="two_digit_times_one_digit",
        mistake_summary="Child treated x5 like x10.",
        last_misconception="treated_x5_like_x10",
        highest_hint_level=3,
    )

    assert wrong_question.session_id == session.session_id
    assert wrong_question.highest_hint_level == 3
    assert wrong_question.last_misconception == "treated_x5_like_x10"
    assert store.list_wrong_questions("child_001") == [wrong_question]


def test_update_wrong_question_practice_result_marks_resolved() -> None:
    store = InMemoryLearningStore()
    session = store.create_session(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="36 x 5 = ?",
        normalized_question="36 x 5 = ?",
        knowledge_point="two_digit_times_one_digit",
        current_prompt="先说说第一步。",
    )
    wrong_question = store.record_wrong_question(
        session_id=session.session_id,
        child_id="child_001",
        normalized_question="36 x 5 = ?",
        knowledge_point="two_digit_times_one_digit",
        mistake_summary="Child treated x5 like x10.",
        last_misconception="treated_x5_like_x10",
        highest_hint_level=2,
    )

    updated = store.update_wrong_question_practice_result(
        question_id=wrong_question.question_id,
        child_id="child_001",
        correct=True,
    )
    events = store.list_events(session.session_id)

    assert updated.practice_completed is True
    assert updated.resolved is True
    assert events[-1].event_type == "practice.completed"
    assert events[-1].payload["correct"] is True


def test_record_ai_call_log_tracks_basic_cost_fields() -> None:
    store = InMemoryLearningStore()

    log = store.record_ai_call(
        child_id="child_001",
        session_id="s_001",
        provider="deeptutor_adapter",
        model="deterministic",
        operation="hint",
        token_estimate=18,
        status="success",
    )

    logs = store.list_ai_call_logs(child_id="child_001")
    assert log.request_id.startswith("ai_")
    assert logs == [log]
    assert logs[0].token_estimate == 18


def test_record_ai_call_log_tracks_agent_latency_and_submission_metadata() -> None:
    store = InMemoryLearningStore()

    log = store.record_ai_call(
        child_id="child_001",
        session_id="sub_001",
        provider="deepseek",
        model="deepseek-v4-flash",
        operation="intent_router.route",
        token_estimate=0,
        status="success",
        agent="IntentRouterAgent",
        latency_ms=342,
        submission_id="sub_001",
        confidence=0.91,
        route_to="english_basic_tutor",
        metadata={
            "detected_subject": "english",
            "detected_task_type": "grammar_fix",
        },
    )

    assert log.agent == "IntentRouterAgent"
    assert log.latency_ms == 342
    assert log.submission_id == "sub_001"
    assert log.confidence == 0.91
    assert log.route_to == "english_basic_tutor"
    assert log.metadata["detected_subject"] == "english"


def test_learning_messages_are_persisted_separately_from_events() -> None:
    store = InMemoryLearningStore()
    session = store.create_session(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="36 x 5 = ?",
        normalized_question="36 x 5 = ?",
        knowledge_point="two_digit_times_one_digit",
        current_prompt="先说第一步。",
    )

    store.append_message(
        session_id=session.session_id,
        child_id="child_001",
        role="user",
        content="36 x 5 = ?",
    )
    store.append_message(
        session_id=session.session_id,
        child_id="child_001",
        role="assistant",
        content="先说第一步。",
    )

    messages = store.list_messages(session.session_id)
    snapshot = store.get_resume_snapshot(session.session_id)

    assert [message.role for message in messages] == ["user", "assistant"]
    assert snapshot.last_messages[-2]["content"] == "36 x 5 = ?"
    assert snapshot.last_messages[-1]["content"] == "先说第一步。"
