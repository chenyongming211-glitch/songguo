from __future__ import annotations

from songguo.backend.services.learning.service import LearningService
from songguo.backend.services.learning.store import InMemoryLearningStore
from songguo.backend.services.learning.summary_rollup import (
    build_daily_summary,
    build_monthly_summary,
    build_weekly_summary,
)


def test_summary_rollup_builds_daily_weekly_and_monthly_layers() -> None:
    store = InMemoryLearningStore()
    service = LearningService(store=store)
    created = service.create_session(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="36 x 5 = ?",
    )
    service.submit_attempt(created.session_id, child_answer="360")

    daily = build_daily_summary(store, child_id="child_001")
    weekly = build_weekly_summary(store, child_id="child_001")
    monthly = build_monthly_summary(store, child_id="child_001")

    assert daily.scope == "daily"
    assert weekly.scope == "weekly"
    assert monthly.scope == "monthly"
    assert daily.wrong_question_count == 1
    assert weekly.common_misconceptions == ["treated_x5_like_x10"]
    assert monthly.top_knowledge_points == ["two_digit_times_one_digit"]
