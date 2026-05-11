from __future__ import annotations

from songguo.backend.services.learning.context_pack import build_student_context_pack
from songguo.backend.services.learning.store import InMemoryLearningStore


def test_context_pack_injects_summary_and_at_most_three_related_evidence() -> None:
    store = InMemoryLearningStore()
    for index in range(5):
        session = store.create_session(
            child_id="child_001",
            subject="math",
            grade=3,
            question_text=f"错题 {index}",
            normalized_question=f"错题 {index}",
            knowledge_point="capacity_round_up",
            current_prompt="先说第一步。",
        )
        store.record_wrong_question(
            session_id=session.session_id,
            child_id="child_001",
            normalized_question=f"错题 {index}",
            knowledge_point="capacity_round_up",
            mistake_summary=f"证据 {index}",
            last_misconception="math_capacity_ignored_remainder_round_up",
            highest_hint_level=2,
        )
    other_session = store.create_session(
        child_id="child_002",
        subject="math",
        grade=3,
        question_text="其他孩子错题",
        normalized_question="其他孩子错题",
        knowledge_point="two_digit_times_one_digit",
        current_prompt="先说第一步。",
    )
    store.record_wrong_question(
        session_id=other_session.session_id,
        child_id="child_002",
        normalized_question="其他孩子错题",
        knowledge_point="two_digit_times_one_digit",
        mistake_summary="不能泄露",
        last_misconception="treated_x5_like_x10",
        highest_hint_level=3,
    )

    pack = build_student_context_pack(
        store,
        child_id="child_001",
        grade=3,
        current_question="新题：每辆最多坐40人，至少需要几辆车？",
    )

    assert pack.child_id == "child_001"
    assert len(pack.related_evidence) == 3
    assert all("child_002" not in item.session_id for item in pack.related_evidence)
    serialized = pack.model_dump_json()
    assert "限载" in serialized or "capacity_round_up" in serialized
    assert "不能泄露" not in serialized


def test_context_pack_allows_empty_history() -> None:
    store = InMemoryLearningStore()

    pack = build_student_context_pack(
        store,
        child_id="child_empty",
        grade=3,
        current_question="36 x 5 = ?",
    )

    assert pack.child_id == "child_empty"
    assert pack.grade == 3
    assert pack.related_evidence == []
    assert "还没有" in pack.recent_week_summary.summary
