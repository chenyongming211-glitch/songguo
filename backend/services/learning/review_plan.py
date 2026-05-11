from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, Field

from songguo.backend.services.learning.labels import knowledge_point_label, misconception_label
from songguo.backend.services.learning.store import InMemoryLearningStore

ReviewScope = Literal["weekly", "monthly", "quarterly", "term", "yearly"]


class ReviewPlanItem(BaseModel):
    question_id: str
    session_id: str
    normalized_question: str
    knowledge_point: str
    knowledge_point_label: str
    misconception_label: str
    highest_hint_level: int
    reason: str


class ReviewPlan(BaseModel):
    child_id: str
    scope: ReviewScope
    summary: str
    items: list[ReviewPlanItem] = Field(default_factory=list)


def build_review_plan(
    store: InMemoryLearningStore,
    *,
    child_id: str,
    scope: ReviewScope = "weekly",
    now: datetime | None = None,
) -> ReviewPlan:
    reference_time = now or datetime.now().astimezone()
    cutoff = reference_time - _scope_window(scope)
    wrong_questions = [
        item for item in store.list_wrong_questions(child_id) if item.updated_at >= cutoff
    ]
    items = [
        ReviewPlanItem(
            question_id=item.question_id,
            session_id=item.session_id,
            normalized_question=item.normalized_question,
            knowledge_point=item.knowledge_point,
            knowledge_point_label=knowledge_point_label(item.knowledge_point),
            misconception_label=misconception_label(item.last_misconception),
            highest_hint_level=item.highest_hint_level,
            reason=(
                f"这题进入{_scope_label(scope)}复习，因为错因是"
                f"{misconception_label(item.last_misconception)}，最高用到 {item.highest_hint_level} 级提示。"
            ),
        )
        for item in wrong_questions
    ]
    return ReviewPlan(
        child_id=child_id,
        scope=scope,
        summary=_summary(scope, len(items)),
        items=items,
    )


def _scope_label(scope: str) -> str:
    return {
        "weekly": "本周",
        "monthly": "本月",
        "quarterly": "本季度",
        "term": "本学期",
        "yearly": "本年度",
    }.get(scope, "当前周期")


def _scope_window(scope: str) -> timedelta:
    return {
        "weekly": timedelta(days=7),
        "monthly": timedelta(days=31),
        "quarterly": timedelta(days=92),
        "term": timedelta(days=190),
        "yearly": timedelta(days=366),
    }.get(scope, timedelta(days=7))


def _summary(scope: str, count: int) -> str:
    label = _scope_label(scope)
    if count == 0:
        return f"{label}暂时没有需要复习的错题。"
    return f"{label}建议复习 {count} 道错题，优先处理反复出现的错因。"
