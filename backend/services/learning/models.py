from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class LearningPhase(StrEnum):
    NEW_QUESTION = "NEW_QUESTION"
    CLASSIFIED = "CLASSIFIED"
    WAIT_CHILD_ATTEMPT = "WAIT_CHILD_ATTEMPT"
    LEARNING_PAUSED = "LEARNING_PAUSED"
    FULL_EXPLANATION_UNLOCKED = "FULL_EXPLANATION_UNLOCKED"
    SIMILAR_PRACTICE = "SIMILAR_PRACTICE"
    PRACTICE_PAUSED = "PRACTICE_PAUSED"
    SESSION_SUMMARY = "SESSION_SUMMARY"


class LearningSession(BaseModel):
    session_id: str = Field(default_factory=lambda: f"s_{uuid4().hex}")
    child_id: str
    family_id: str = ""
    question_id: str = Field(default_factory=lambda: f"q_{uuid4().hex}")
    subject: str
    grade: int
    question_text: str
    normalized_question: str
    knowledge_point: str
    phase: LearningPhase = LearningPhase.WAIT_CHILD_ATTEMPT
    hint_level: int = 1
    attempt_count: int = 0
    answer_unlocked: bool = False
    unlock_reason: str | None = None
    last_misconception: str | None = None
    current_prompt: str
    problem_analysis: dict[str, Any] | None = None
    current_key_point_id: str | None = None
    released_key_point_ids: list[str] = Field(default_factory=list)
    mastered_key_point_ids: list[str] = Field(default_factory=list)
    runner_mode: str = "kernel"
    policy_version: str = "guided_math_policy@v1.0.0"
    strategy_version: str = "grade3_math_hint@v1.0.0"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None


class TeachingProgress(BaseModel):
    mode: str = "hint_policy"
    phase_label: str = "分步引导"
    current_label: str = "等待孩子尝试"
    current_index: int | None = None
    total_count: int | None = None
    hint_level: int = 0
    answer_policy_label: str = "答案锁定中"
    hint_policy_label: str = "按孩子回答动态推进"


class ChildProfile(BaseModel):
    child_id: str = Field(default_factory=lambda: f"child_{uuid4().hex}")
    family_id: str = ""
    name: str
    grade: int
    term_label: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ReminderSubscription(BaseModel):
    subscription_id: str = Field(default_factory=lambda: f"sub_{uuid4().hex}")
    openid: str
    child_id: str
    template_id: str
    enabled: bool = True
    scope: str = "weekly"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ReminderJob(BaseModel):
    job_id: str = Field(default_factory=lambda: f"rem_{uuid4().hex}")
    openid: str
    child_id: str
    template_id: str
    scope: str
    status: str = "pending"
    created_at: datetime = Field(default_factory=utc_now)


class LearningEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: f"evt_{uuid4().hex}")
    session_id: str
    child_id: str
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    policy_version: str = "guided_math_policy@v1.0.0"
    strategy_version: str = "grade3_math_hint@v1.0.0"
    deeptutor_trace_id: str | None = None
    leakage_check_result: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=utc_now)


class LearningMessage(BaseModel):
    message_id: str = Field(default_factory=lambda: f"msg_{uuid4().hex}")
    session_id: str
    child_id: str
    role: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class AICallLog(BaseModel):
    request_id: str = Field(default_factory=lambda: f"ai_{uuid4().hex}")
    child_id: str
    session_id: str
    provider: str
    model: str
    operation: str
    agent: str = ""
    submission_id: str = ""
    item_id: str = ""
    latency_ms: int = 0
    confidence: float = 0.0
    route_to: str = ""
    failure_reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    token_estimate: int = 0
    status: str = "success"
    created_at: datetime = Field(default_factory=utc_now)


class LearningSummary(BaseModel):
    summary_id: str = Field(default_factory=lambda: f"sum_{uuid4().hex}")
    child_id: str
    scope: str
    session_count: int = 0
    wrong_question_count: int = 0
    completed_session_count: int = 0
    top_knowledge_points: list[str] = Field(default_factory=list)
    common_misconceptions: list[str] = Field(default_factory=list)
    parent_summary: str
    next_actions: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class WrongQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: f"wq_{uuid4().hex}")
    child_id: str
    family_id: str = ""
    session_id: str
    normalized_question: str
    knowledge_point: str
    mistake_summary: str
    last_misconception: str | None = None
    highest_hint_level: int = 1
    explanation_unlocked: bool = False
    practice_completed: bool = False
    resolved: bool = False
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class SafetyEvent(BaseModel):
    safety_event_id: str = Field(default_factory=lambda: f"safe_{uuid4().hex}")
    session_id: str
    child_id: str
    event_type: str
    input_text: str | None = None
    blocked_text: str | None = None
    reason: str
    policy_version: str = "guided_math_policy@v1.0.0"
    created_at: datetime = Field(default_factory=utc_now)


class ResumeSnapshot(BaseModel):
    session_id: str
    question_text: str
    subject: str = "math"
    grade: int = 3
    phase: LearningPhase
    hint_level: int
    attempt_count: int
    answer_unlocked: bool
    current_prompt: str
    teaching_progress: TeachingProgress = Field(default_factory=TeachingProgress)
    last_misconception: str | None = None
    last_messages: list[dict[str, Any]] = Field(default_factory=list)
