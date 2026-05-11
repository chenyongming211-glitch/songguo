from __future__ import annotations

from songguo.backend.services.learning.models import LearningPhase
from songguo.backend.services.learning.state_machine import (
    AttemptOutcome,
    apply_attempt_result,
    can_unlock_explanation,
    initialize_session_state,
)


def test_initial_state_starts_with_locked_answer_and_first_hint() -> None:
    state = initialize_session_state()

    assert state.phase == LearningPhase.WAIT_CHILD_ATTEMPT
    assert state.hint_level == 1
    assert state.attempt_count == 0
    assert state.answer_unlocked is False


def test_wrong_known_misconception_advances_hint_and_records_event() -> None:
    state = initialize_session_state()

    result = apply_attempt_result(
        state,
        AttemptOutcome.WRONG_KNOWN_MISCONCEPTION,
        misconception="treated_x5_like_x10",
    )

    assert result.state.hint_level == 2
    assert result.state.attempt_count == 1
    assert result.state.answer_unlocked is False
    assert result.state.last_misconception == "treated_x5_like_x10"
    assert result.event_types == [
        "child.attempt_submitted",
        "attempt.evaluated",
        "hint_level.upgraded",
    ]


def test_hint_level_never_decreases_after_multiple_wrong_attempts() -> None:
    state = initialize_session_state(hint_level=3, attempt_count=2)

    result = apply_attempt_result(state, AttemptOutcome.WRONG_UNKNOWN)

    assert result.state.hint_level == 4
    assert result.state.hint_level >= state.hint_level
    assert result.state.answer_unlocked is False


def test_answer_unlock_requires_policy_conditions() -> None:
    early_state = initialize_session_state(hint_level=2, attempt_count=1)
    ready_state = initialize_session_state(hint_level=5, attempt_count=3)

    assert can_unlock_explanation(early_state) is False
    assert can_unlock_explanation(ready_state) is True


def test_correct_attempt_moves_to_summary_without_revealing_answer_by_default() -> None:
    state = initialize_session_state(hint_level=2, attempt_count=1)

    result = apply_attempt_result(state, AttemptOutcome.CORRECT)

    assert result.state.phase == LearningPhase.SIMILAR_PRACTICE
    assert result.state.answer_unlocked is False
    assert result.event_types == [
        "child.attempt_submitted",
        "attempt.evaluated",
        "practice.generated",
    ]

