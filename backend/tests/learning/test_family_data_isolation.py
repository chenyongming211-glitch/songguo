from __future__ import annotations

from songguo.backend.services.learning.submission_models import (
    EvidenceType,
    JudgeResult,
    LearningItem,
    MasteryState,
    SourceType,
)
from songguo.backend.services.learning.store import InMemoryLearningStore, SQLiteLearningStore
from songguo.backend.services.session_auth import authorize_child_access
from songguo.backend.services.wechat import WechatService


def test_store_stamps_family_id_on_learning_assets() -> None:
    store = InMemoryLearningStore()
    token = WechatService()._sign_session("openid_001")
    child_id = "child_openid_001"

    authorize_child_access(store, child_id=child_id, session_token=token)
    session = store.create_session(
        child_id=child_id,
        subject="math",
        grade=3,
        question_text="36 x 5 = ?",
        normalized_question="36 x 5 = ?",
        knowledge_point="two_digit_times_one_digit",
        current_prompt="先说说 36 x 10 是多少？",
    )
    wrong = store.record_wrong_question(
        session_id=session.session_id,
        child_id=child_id,
        normalized_question="36 x 5 = ?",
        knowledge_point="two_digit_times_one_digit",
        mistake_summary="把乘以 5 当成乘以 10。",
        last_misconception="treated_x5_like_x10",
        highest_hint_level=1,
    )
    submission = store.create_submission(
        child_id=child_id,
        subject="math",
        grade=3,
        source_type=SourceType.TEXT,
        raw_text="36 x 5 = ?\n孩子答案：360",
    )
    item = store.add_submission_item(
        LearningItem(
            submission_id=submission.submission_id,
            child_id=child_id,
            item_index=1,
            question_text="36 x 5 = ?",
            child_answer="360",
            judge_result=JudgeResult.WRONG,
            question_type_id="two_digit_times_one_digit",
        )
    )
    evidence = store.save_mastery_evidence(
        item_id=item.item_id,
        child_id=child_id,
        question_type_id="two_digit_times_one_digit",
        evidence_type=EvidenceType.WRONG_UNRESOLVED,
        is_correct=False,
        mastery_state_after=MasteryState.NEEDS_REVIEW,
        review_due=True,
    )
    queue_item = store.enqueue_tutor_item(
        submission_id=submission.submission_id,
        item_id=item.item_id,
        child_id=child_id,
        question_type_id="two_digit_times_one_digit",
    )

    assert store.get_child(child_id).family_id == "family_openid_001"
    assert session.family_id == "family_openid_001"
    assert wrong.family_id == "family_openid_001"
    assert store.get_submission(submission.submission_id).family_id == "family_openid_001"
    assert store.list_submission_items(submission.submission_id)[0].family_id == "family_openid_001"
    assert evidence.family_id == "family_openid_001"
    assert queue_item.family_id == "family_openid_001"


def test_sqlite_persists_family_binding_and_child_family(tmp_path) -> None:
    db_path = tmp_path / "learning.db"
    store = SQLiteLearningStore(db_path)
    token = WechatService()._sign_session("openid_002")
    child_id = "child_openid_002"

    authorize_child_access(store, child_id=child_id, session_token=token)
    store.upsert_child(child_id=child_id, name="孩子二", grade=4)

    reloaded = SQLiteLearningStore(db_path)

    assert reloaded.is_child_bound_to_openid(openid="openid_002", child_id=child_id)
    assert reloaded.get_child(child_id).family_id == "family_openid_002"
