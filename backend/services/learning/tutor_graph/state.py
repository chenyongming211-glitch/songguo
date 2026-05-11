from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from songguo.backend.services.learning.models import TeachingProgress
from songguo.backend.services.learning.practice_recommender import PracticeItem


class RuleJudgeResult(BaseModel):
    determined: bool = False
    correct: bool | None = None
    partially_correct: bool = False
    expected_answer: str | None = None
    misconception_tag: str | None = None
    evidence: str = ""
    matched_key_point_id: str | None = None
    next_key_point_id: str | None = None


class MathMistakeTutorGraphState(BaseModel):
    child_id: str
    session_id: str = "pending"
    grade: int
    question_text: str
    subject: str = "math"
    input_type: str = "text"
    ocr_result: dict[str, Any] | None = None
    problem_analysis: dict[str, Any] | None = None
    rule_judge_result: dict[str, Any] | None = None
    messages: list[dict[str, str]] = Field(default_factory=list)
    context_pack: dict[str, Any] | None = None
    parent_settings: dict[str, Any] = Field(default_factory=dict)
    tutor_reply: str = ""
    structured_state: dict[str, Any] = Field(default_factory=dict)
    learning_deposit_delta: dict[str, Any] = Field(default_factory=dict)
    practice_items: list[PracticeItem] = Field(default_factory=list)
    guardrail_result: dict[str, Any] = Field(default_factory=dict)
    provider_trace: dict[str, Any] = Field(default_factory=dict)
    child_answer: str = ""
    graph_create_result: dict[str, Any] | None = None
    graph_submit_result: dict[str, Any] | None = None
    should_end_session: bool = False

    @field_validator("practice_items")
    @classmethod
    def _clamp_practice_items(cls, value: list[PracticeItem]) -> list[PracticeItem]:
        return list(value[:3])


class GraphCreateResult(BaseModel):
    session_id: str
    question_text: str
    subject: str = "math"
    grade: int = 3
    phase: str
    hint_level: int
    message: str
    answer_unlocked: bool
    teaching_progress: TeachingProgress


class GraphSubmitResult(BaseModel):
    correct: bool
    partially_correct: bool = False
    phase: str
    hint_level: int
    message: str
    answer_unlocked: bool
    misconception_tag: str | None = None
    matched_key_point_id: str | None = None
    next_key_point_id: str | None = None
    teaching_progress: TeachingProgress
    practice_items: list[PracticeItem] = Field(default_factory=list)
