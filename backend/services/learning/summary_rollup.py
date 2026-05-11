from __future__ import annotations

from songguo.backend.services.learning.models import LearningSummary
from songguo.backend.services.learning.reporting import build_summary_draft
from songguo.backend.services.learning.store import InMemoryLearningStore


def build_daily_summary(store: InMemoryLearningStore, *, child_id: str) -> LearningSummary:
    return _build_summary(store, child_id=child_id, scope="daily")


def build_weekly_summary(store: InMemoryLearningStore, *, child_id: str) -> LearningSummary:
    return _build_summary(store, child_id=child_id, scope="weekly")


def build_monthly_summary(store: InMemoryLearningStore, *, child_id: str) -> LearningSummary:
    return _build_summary(store, child_id=child_id, scope="monthly")


def _build_summary(
    store: InMemoryLearningStore,
    *,
    child_id: str,
    scope: str,
) -> LearningSummary:
    draft = build_summary_draft(store, child_id=child_id, scope=scope)
    completed = sum(
        1
        for session in store.list_sessions(child_id=child_id)
        if session.completed_at is not None
    )
    summary = LearningSummary(
        child_id=child_id,
        scope=scope,
        session_count=draft.session_count,
        wrong_question_count=draft.wrong_question_count,
        completed_session_count=completed,
        top_knowledge_points=draft.top_knowledge_points,
        common_misconceptions=draft.common_misconceptions,
        parent_summary=draft.parent_summary,
        next_actions=draft.next_actions,
    )
    store.save_learning_summary(summary)
    return summary
