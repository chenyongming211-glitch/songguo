from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from songguo.backend.services.learning.models import LearningPhase


MAX_HINT_LEVEL = 5
MIN_ATTEMPTS_BEFORE_EXPLANATION = 3


class AttemptOutcome(StrEnum):
    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    WRONG_KNOWN_MISCONCEPTION = "wrong_known_misconception"
    WRONG_UNKNOWN = "wrong_unknown"
    EMPTY_OR_OFF_TASK = "empty_or_off_task"


@dataclass(frozen=True)
class LearningState:
    phase: LearningPhase
    hint_level: int
    attempt_count: int
    answer_unlocked: bool
    last_misconception: str | None = None


@dataclass(frozen=True)
class TransitionResult:
    state: LearningState
    event_types: list[str]


def initialize_session_state(
    *,
    phase: LearningPhase = LearningPhase.WAIT_CHILD_ATTEMPT,
    hint_level: int = 1,
    attempt_count: int = 0,
    answer_unlocked: bool = False,
    last_misconception: str | None = None,
) -> LearningState:
    return LearningState(
        phase=phase,
        hint_level=max(1, min(hint_level, MAX_HINT_LEVEL)),
        attempt_count=max(0, attempt_count),
        answer_unlocked=answer_unlocked,
        last_misconception=last_misconception,
    )


def can_unlock_explanation(state: LearningState) -> bool:
    return state.hint_level >= MAX_HINT_LEVEL and state.attempt_count >= MIN_ATTEMPTS_BEFORE_EXPLANATION


def apply_attempt_result(
    state: LearningState,
    outcome: AttemptOutcome,
    *,
    misconception: str | None = None,
) -> TransitionResult:
    next_attempt_count = state.attempt_count + 1
    base_events = ["child.attempt_submitted", "attempt.evaluated"]

    if outcome == AttemptOutcome.CORRECT:
        return TransitionResult(
            state=replace(
                state,
                phase=LearningPhase.SIMILAR_PRACTICE,
                attempt_count=next_attempt_count,
            ),
            event_types=[*base_events, "practice.generated"],
        )

    if outcome == AttemptOutcome.EMPTY_OR_OFF_TASK:
        return TransitionResult(
            state=replace(
                state,
                phase=LearningPhase.WAIT_CHILD_ATTEMPT,
                attempt_count=next_attempt_count,
            ),
            event_types=[*base_events, "hint.generated"],
        )

    next_hint_level = min(MAX_HINT_LEVEL, max(state.hint_level, state.hint_level + 1))
    next_state = replace(
        state,
        phase=LearningPhase.WAIT_CHILD_ATTEMPT,
        hint_level=next_hint_level,
        attempt_count=next_attempt_count,
        last_misconception=misconception or state.last_misconception,
    )
    events = [*base_events, "hint_level.upgraded"]

    if can_unlock_explanation(next_state):
        next_state = replace(
            next_state,
            phase=LearningPhase.FULL_EXPLANATION_UNLOCKED,
            answer_unlocked=True,
        )
        events.append("answer.unlocked")

    return TransitionResult(state=next_state, event_types=events)


def advance_to_similar_practice(state: LearningState) -> TransitionResult:
    return TransitionResult(
        state=replace(state, phase=LearningPhase.SIMILAR_PRACTICE),
        event_types=["practice.generated"],
    )

