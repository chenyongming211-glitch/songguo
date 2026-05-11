from __future__ import annotations

from datetime import datetime, timedelta, timezone

from songguo.backend.services.learning.memory_profile import build_learning_memory
from songguo.backend.services.learning.practice_recommender import recommend_targeted_practice
from songguo.backend.services.learning.service import LearningService
from songguo.backend.services.learning.store import InMemoryLearningStore


def _store_with_repeated_x5_mistake() -> InMemoryLearningStore:
    store = InMemoryLearningStore()
    service = LearningService(store=store)
    first = service.create_session(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="36 x 5 = ?",
    )
    service.submit_attempt(first.session_id, child_answer="360")
    second = service.create_session(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="24 x 5 = ?",
    )
    service.submit_attempt(second.session_id, child_answer="240")
    return store


def test_learning_memory_ranks_repeated_weaknesses() -> None:
    memory = build_learning_memory(_store_with_repeated_x5_mistake(), child_id="child_001")

    assert memory.child_id == "child_001"
    assert memory.scope == "weekly"
    assert memory.top_weaknesses[0].knowledge_point == "two_digit_times_one_digit"
    assert memory.top_weaknesses[0].knowledge_point_label == "两位数乘一位数"
    assert memory.top_weaknesses[0].wrong_count == 2
    assert memory.top_weaknesses[0].common_misconceptions == ["treated_x5_like_x10"]
    assert memory.top_weaknesses[0].common_misconception_labels == ["把乘以 5 当成乘以 10"]
    assert memory.top_weaknesses[0].mastery_score < 80
    assert memory.top_weaknesses[0].risk_level in {"medium", "high"}
    assert "两位数乘一位数" in memory.summary
    assert "two_digit_times_one_digit" not in memory.summary
    assert "treated_x5_like_x10" not in memory.summary


def test_targeted_practice_uses_memory_not_only_current_question() -> None:
    recommendation = recommend_targeted_practice(
        _store_with_repeated_x5_mistake(),
        child_id="child_001",
        limit=3,
    )

    assert recommendation.child_id == "child_001"
    assert recommendation.knowledge_point == "two_digit_times_one_digit"
    assert recommendation.misconception_tag == "treated_x5_like_x10"
    assert len(recommendation.items) == 3
    assert all("x 5" in item.question for item in recommendation.items)
    assert all(item.answer for item in recommendation.items)


def test_targeted_practice_clamps_mvp_practice_count_to_three() -> None:
    recommendation = recommend_targeted_practice(
        _store_with_repeated_x5_mistake(),
        child_id="child_001",
        limit=5,
    )

    assert len(recommendation.items) == 3


def test_learning_memory_filters_by_scope_window() -> None:
    now = datetime(2026, 5, 1, tzinfo=timezone.utc)
    store = _store_with_repeated_x5_mistake()
    store.wrong_questions["child_001"][1] = store.wrong_questions["child_001"][1].model_copy(
        update={
            "created_at": now - timedelta(days=40),
            "updated_at": now - timedelta(days=40),
        }
    )

    weekly = build_learning_memory(store, child_id="child_001", scope="weekly", now=now)
    term = build_learning_memory(store, child_id="child_001", scope="term", now=now)

    assert weekly.top_weaknesses[0].wrong_count == 1
    assert term.top_weaknesses[0].wrong_count == 2


def test_targeted_practice_accepts_long_term_memory_scope() -> None:
    now = datetime(2026, 5, 1, tzinfo=timezone.utc)
    store = _store_with_repeated_x5_mistake()
    store.wrong_questions["child_001"][0] = store.wrong_questions["child_001"][0].model_copy(
        update={
            "created_at": now - timedelta(days=40),
            "updated_at": now - timedelta(days=40),
        }
    )
    store.wrong_questions["child_001"][1] = store.wrong_questions["child_001"][1].model_copy(
        update={
            "created_at": now - timedelta(days=40),
            "updated_at": now - timedelta(days=40),
        }
    )

    weekly = recommend_targeted_practice(store, child_id="child_001", scope="weekly", now=now)
    term = recommend_targeted_practice(store, child_id="child_001", scope="term", now=now)

    assert weekly.items == []
    assert len(term.items) == 3
