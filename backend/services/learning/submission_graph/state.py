from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from songguo.backend.services.learning.submission_models import (
    LearningSubmissionStatus,
    SourceType,
)


class LearningSubmissionGraphState(BaseModel):
    submission_id: str | None = None
    child_id: str
    subject: str = "math"
    grade: int = 3
    source_type: SourceType = SourceType.TEXT
    raw_text: str = ""
    item_count: int = 0
    correct_count: int = 0
    wrong_count: int = 0
    active_queue_item_id: str | None = None
    active_tutor_session_id: str | None = None
    status: LearningSubmissionStatus = LearningSubmissionStatus.INTAKE_PENDING
    summary: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class LearningSubmissionGraphResult(BaseModel):
    submission_id: str
    status: LearningSubmissionStatus
    item_count: int = 0
    correct_count: int = 0
    wrong_count: int = 0
    active_queue_item_id: str | None = None
    active_tutor_session_id: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)
