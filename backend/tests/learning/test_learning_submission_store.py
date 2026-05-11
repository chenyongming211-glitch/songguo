from __future__ import annotations

from pathlib import Path

from songguo.backend.services.learning.submission_models import (
    EvidenceType,
    JudgeResult,
    LearningItem,
    LearningSubmissionStatus,
    MasteryState,
    SourceType,
)
from songguo.backend.services.learning.store import InMemoryLearningStore, SQLiteLearningStore


def test_submission_store_records_items_evidence_and_tutor_queue() -> None:
    store = InMemoryLearningStore()

    submission = store.create_submission(
        child_id="child_001",
        subject="math",
        grade=3,
        source_type=SourceType.TEXT,
        raw_text="36 平均分给 5 只小松鼠，每只几颗，还剩几颗？\n孩子答案：每只 6 颗还剩 6 颗",
    )
    item = store.add_submission_item(
        LearningItem(
            submission_id=submission.submission_id,
            child_id="child_001",
            item_index=1,
            question_text="36 平均分给 5 只小松鼠，每只几颗，还剩几颗？",
            child_answer="每只 6 颗还剩 6 颗",
            judge_result=JudgeResult.WRONG,
            knowledge_point="division_with_remainder",
        )
    )
    evidence = store.save_mastery_evidence(
        item_id=item.item_id,
        child_id="child_001",
        question_type_id="division_with_remainder",
        evidence_type=EvidenceType.WRONG_UNRESOLVED,
        is_correct=False,
        mastery_state_after=MasteryState.NEEDS_REVIEW,
        review_due=True,
    )
    queue_item = store.enqueue_tutor_item(
        submission_id=submission.submission_id,
        item_id=item.item_id,
        child_id="child_001",
        question_type_id="division_with_remainder",
        priority=10,
    )

    snapshot = store.get_submission_snapshot(submission.submission_id)

    assert snapshot.submission.item_count == 1
    assert snapshot.submission.wrong_count == 1
    assert snapshot.items == [item]
    assert snapshot.mastery_evidence == [evidence]
    assert snapshot.tutor_queue == [queue_item]
    assert store.get_active_tutor_item(submission.submission_id) == queue_item


def test_submission_store_marks_submission_complete_when_queue_done() -> None:
    store = InMemoryLearningStore()
    submission = store.create_submission(
        child_id="child_001",
        subject="math",
        grade=3,
        source_type=SourceType.TEXT,
        raw_text="18÷3=？ 孩子答案：6",
    )
    item = store.add_submission_item(
        LearningItem(
            submission_id=submission.submission_id,
            child_id="child_001",
            item_index=1,
            question_text="18÷3=？",
            child_answer="6",
            judge_result=JudgeResult.CORRECT,
        )
    )

    store.save_mastery_evidence(
        item_id=item.item_id,
        child_id="child_001",
        question_type_id="division",
        evidence_type=EvidenceType.SUBMISSION_CORRECT,
        is_correct=True,
        mastery_state_after=MasteryState.OBSERVED,
        review_due=False,
    )
    completed = store.complete_submission_if_queue_done(submission.submission_id)

    assert completed.status == LearningSubmissionStatus.COMPLETED
    assert completed.completed_at is not None


def test_sqlite_submission_store_persists_submission_snapshot(tmp_path: Path) -> None:
    store = SQLiteLearningStore(tmp_path / "learning.db")
    submission = store.create_submission(
        child_id="child_001",
        subject="math",
        grade=3,
        source_type=SourceType.TEXT,
        raw_text="48 ÷ 6 = ? 孩子答案：8",
    )
    item = store.add_submission_item(
        LearningItem(
            submission_id=submission.submission_id,
            child_id="child_001",
            item_index=1,
            question_text="48 ÷ 6 = ?",
            child_answer="8",
            judge_result=JudgeResult.CORRECT,
            question_type_id="division",
        )
    )
    store.save_mastery_evidence(
        item_id=item.item_id,
        child_id="child_001",
        question_type_id="division",
        evidence_type=EvidenceType.SUBMISSION_CORRECT,
        is_correct=True,
        mastery_state_after=MasteryState.OBSERVED,
    )

    reloaded = SQLiteLearningStore(tmp_path / "learning.db")
    snapshot = reloaded.get_submission_snapshot(submission.submission_id)

    assert snapshot.submission.item_count == 1
    assert snapshot.items[0].question_text == "48 ÷ 6 = ?"
    assert snapshot.mastery_evidence[0].evidence_type == EvidenceType.SUBMISSION_CORRECT
