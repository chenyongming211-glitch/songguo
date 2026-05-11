from __future__ import annotations

import pytest

from songguo.backend.services.learning.deeptutor_adapter import (
    DeepTutorLearningAdapter,
    TeachingDraft,
)
from songguo.backend.services.learning.prompt_registry import PromptRegistry, UnknownPromptError


def test_prompt_registry_rejects_unknown_prompt_id() -> None:
    registry = PromptRegistry()

    with pytest.raises(UnknownPromptError):
        registry.require("unknown_prompt")


def test_generate_hint_returns_structured_draft_from_generator() -> None:
    def generator(payload: dict) -> TeachingDraft:
        return TeachingDraft(
            action="hint",
            hint_level=payload["hint_level"],
            exposes_final_answer=False,
            misconception_tag="treated_x5_like_x10",
            text="What is 36 x 10?",
            trace_id="trace_001",
        )

    adapter = DeepTutorLearningAdapter(draft_generator=generator)

    draft = adapter.generate_hint(
        question_text="36 x 5 = ?",
        grade=3,
        knowledge_point="two_digit_times_one_digit",
        hint_level=1,
    )

    assert draft.action == "hint"
    assert draft.hint_level == 1
    assert draft.exposes_final_answer is False
    assert draft.trace_id == "trace_001"


def test_explanation_rejected_while_answer_locked() -> None:
    adapter = DeepTutorLearningAdapter()

    with pytest.raises(PermissionError):
        adapter.generate_explanation(
            question_text="36 x 5 = ?",
            answer_unlocked=False,
        )


def test_generate_similar_practice_uses_history_context() -> None:
    adapter = DeepTutorLearningAdapter()

    draft = adapter.generate_similar_practice(
        knowledge_point="two_digit_times_one_digit",
        misconception_tag="treated_x5_like_x10",
        difficulty=1,
    )

    assert draft.action == "similar_practice"
    assert draft.exposes_final_answer is False
    assert "two_digit_times_one_digit" in draft.metadata["knowledge_point"]


def test_generator_failure_marks_fallback_metadata() -> None:
    def failing_generator(_payload: dict) -> TeachingDraft:
        raise RuntimeError("provider unavailable")

    adapter = DeepTutorLearningAdapter(draft_generator=failing_generator)

    draft = adapter.generate_hint(
        question_text="36 x 5 = ?",
        grade=3,
        knowledge_point="two_digit_times_one_digit",
        hint_level=1,
    )

    assert draft.action == "hint"
    assert draft.metadata["fallback_reason"] == "draft_generator_error"
    assert draft.metadata["fallback_error"] == "RuntimeError"


def test_default_first_hint_for_times_five_uses_question_numbers_without_final_answer() -> None:
    adapter = DeepTutorLearningAdapter()

    draft = adapter.generate_hint(
        question_text="48 x 5 = ?",
        grade=3,
        knowledge_point="two_digit_times_one_digit",
        hint_level=1,
    )

    assert "48 × 10" in draft.text
    assert "36" not in draft.text
    assert "240" not in draft.text
