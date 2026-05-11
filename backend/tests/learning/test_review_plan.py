from __future__ import annotations

from datetime import datetime, timedelta, timezone

from songguo.backend.services.learning.review_plan import build_review_plan
from songguo.backend.services.learning.service import LearningService
from songguo.backend.services.learning.store import InMemoryLearningStore


def test_review_plan_groups_wrong_questions_by_scope() -> None:
    store = InMemoryLearningStore()
    service = LearningService(store=store)
    created = service.create_session(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="36 x 5 = ?",
    )
    service.submit_attempt(created.session_id, child_answer="360")

    plan = build_review_plan(store, child_id="child_001", scope="weekly")

    assert plan.child_id == "child_001"
    assert plan.scope == "weekly"
    assert len(plan.items) == 1
    assert plan.items[0].knowledge_point_label == "两位数乘一位数"
    assert "把乘以 5 当成乘以 10" in plan.items[0].reason
    assert "本周" in plan.summary


def test_review_plan_filters_by_scope_window() -> None:
    now = datetime(2026, 5, 1, tzinfo=timezone.utc)
    store = InMemoryLearningStore()
    service = LearningService(store=store)

    recent = service.create_session(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="36 x 5 = ?",
    )
    service.submit_attempt(recent.session_id, child_answer="360")

    older = service.create_session(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="24 x 5 = ?",
    )
    service.submit_attempt(older.session_id, child_answer="240")
    store.wrong_questions["child_001"][1] = store.wrong_questions["child_001"][1].model_copy(
        update={
            "created_at": now - timedelta(days=40),
            "updated_at": now - timedelta(days=40),
        }
    )

    weekly = build_review_plan(store, child_id="child_001", scope="weekly", now=now)
    term = build_review_plan(store, child_id="child_001", scope="term", now=now)

    assert [item.normalized_question for item in weekly.items] == ["36 x 5 = ?"]
    assert {item.normalized_question for item in term.items} == {"36 x 5 = ?", "24 x 5 = ?"}


def test_review_plan_supports_yearly_scope() -> None:
    now = datetime(2026, 5, 1, tzinfo=timezone.utc)
    store = InMemoryLearningStore()
    service = LearningService(store=store)
    created = service.create_session(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="36 x 5 = ?",
    )
    service.submit_attempt(created.session_id, child_answer="360")
    store.wrong_questions["child_001"][0] = store.wrong_questions["child_001"][0].model_copy(
        update={
            "created_at": now - timedelta(days=250),
            "updated_at": now - timedelta(days=250),
        }
    )

    term = build_review_plan(store, child_id="child_001", scope="term", now=now)
    yearly = build_review_plan(store, child_id="child_001", scope="yearly", now=now)

    assert term.items == []
    assert len(yearly.items) == 1
    assert "本年度" in yearly.summary
