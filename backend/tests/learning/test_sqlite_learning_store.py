from __future__ import annotations

import sqlite3

from songguo.backend.services.learning.memory_profile import build_learning_memory
from songguo.backend.services.learning.service import get_global_learning_store
from songguo.backend.services.learning.store import SQLiteLearningStore, _postgres_sql


def test_sqlite_learning_store_persists_session_events_and_resume(tmp_path) -> None:
    db_path = tmp_path / "learning.db"
    store = SQLiteLearningStore(db_path=db_path)
    session = store.create_session(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="36 x 5 = ?",
        normalized_question="36 x 5 = ?",
        knowledge_point="two_digit_times_one_digit",
        current_prompt="What is 36 x 10?",
    )
    store.update_session(
        session.session_id,
        hint_level=2,
        attempt_count=1,
        current_prompt="Try taking half of 360.",
        last_misconception="treated_x5_like_x10",
    )

    reopened = SQLiteLearningStore(db_path=db_path)
    snapshot = reopened.get_resume_snapshot(session.session_id)
    events = reopened.list_events(session.session_id)

    assert snapshot.hint_level == 2
    assert snapshot.attempt_count == 1
    assert snapshot.current_prompt == "Try taking half of 360."
    assert events[0].event_type == "session.created"


def test_sqlite_learning_store_resume_reconstructs_recent_turns(tmp_path) -> None:
    db_path = tmp_path / "learning.db"
    store = SQLiteLearningStore(db_path=db_path)
    session = store.create_session(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="36 x 5 = ?",
        normalized_question="36 x 5 = ?",
        knowledge_point="two_digit_times_one_digit",
        current_prompt="What is 36 x 10?",
    )
    store.append_event(
        session_id=session.session_id,
        child_id="child_001",
        event_type="key_point.released",
        payload={"prompt": "What is 36 x 10?"},
    )
    store.append_event(
        session_id=session.session_id,
        child_id="child_001",
        event_type="child.attempt_submitted",
        payload={"child_answer": "360"},
    )
    store.append_event(
        session_id=session.session_id,
        child_id="child_001",
        event_type="key_point.released",
        payload={"prompt": "360 是乘以 10。乘以 5 是它的一半。"},
    )
    store.append_event(
        session_id=session.session_id,
        child_id="child_001",
        event_type="practice.followup_requested",
        payload={"child_answer": "继续"},
    )
    store.update_session(
        session.session_id,
        hint_level=2,
        attempt_count=1,
        current_prompt="360 是乘以 10。乘以 5 是它的一半。",
    )

    reopened = SQLiteLearningStore(db_path=db_path)
    snapshot = reopened.get_resume_snapshot(session.session_id)

    assert [message["role"] for message in snapshot.last_messages] == [
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert snapshot.last_messages[1]["content"] == "360"
    assert snapshot.last_messages[3]["content"] == "继续"


def test_sqlite_learning_store_persists_wrong_questions_for_memory(tmp_path) -> None:
    db_path = tmp_path / "learning.db"
    store = SQLiteLearningStore(db_path=db_path)
    session = store.create_session(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="36 x 5 = ?",
        normalized_question="36 x 5 = ?",
        knowledge_point="two_digit_times_one_digit",
        current_prompt="What is 36 x 10?",
    )
    store.record_wrong_question(
        session_id=session.session_id,
        child_id="child_001",
        normalized_question="36 x 5 = ?",
        knowledge_point="two_digit_times_one_digit",
        mistake_summary="Child treated x5 like x10.",
        last_misconception="treated_x5_like_x10",
        highest_hint_level=2,
    )

    reopened = SQLiteLearningStore(db_path=db_path)
    wrong_questions = reopened.list_wrong_questions("child_001")
    memory = build_learning_memory(reopened, child_id="child_001")

    assert len(wrong_questions) == 1
    assert wrong_questions[0].last_misconception == "treated_x5_like_x10"
    assert memory.top_weaknesses[0].knowledge_point == "two_digit_times_one_digit"


def test_sqlite_learning_store_exposes_core_learning_asset_columns(tmp_path) -> None:
    db_path = tmp_path / "learning.db"
    store = SQLiteLearningStore(db_path=db_path)
    session = store.create_session(
        child_id="child_001",
        subject="math",
        grade=3,
        question_text="36 x 5 = ?",
        normalized_question="36 x 5 = ?",
        knowledge_point="two_digit_times_one_digit",
        current_prompt="先想一想。",
    )
    wrong = store.record_wrong_question(
        session_id=session.session_id,
        child_id="child_001",
        normalized_question="36 x 5 = ?",
        knowledge_point="two_digit_times_one_digit",
        mistake_summary="Child treated x5 like x10.",
        last_misconception="treated_x5_like_x10",
        highest_hint_level=2,
    )
    store.save_learning_deposit(
        session.session_id,
        {
            "child_id": "child_001",
            "question_record": {
                "knowledge_point": "two_digit_times_one_digit",
                "question_type": "calculation",
            },
            "mistake_record": {
                "main_error_reason": "treated_x5_like_x10",
                "need_review": True,
            },
        },
    )

    with sqlite3.connect(db_path) as conn:
        wrong_row = conn.execute(
            """
            SELECT knowledge_point, last_misconception, resolved
            FROM wrong_questions
            WHERE question_id = ?
            """,
            (wrong.question_id,),
        ).fetchone()
        deposit_row = conn.execute(
            """
            SELECT knowledge_point, question_type, main_misconception, need_review
            FROM learning_deposits
            WHERE session_id = ?
            """,
            (session.session_id,),
        ).fetchone()

    assert wrong_row == ("two_digit_times_one_digit", "treated_x5_like_x10", 0)
    assert deposit_row == (
        "two_digit_times_one_digit",
        "calculation",
        "treated_x5_like_x10",
        1,
    )


def test_global_learning_store_uses_sqlite_persistence() -> None:
    assert isinstance(get_global_learning_store(), SQLiteLearningStore)


def test_postgres_sql_translates_sqlite_placeholders_and_insert_ignore() -> None:
    translated = _postgres_sql(
        """
        INSERT OR IGNORE INTO child_bindings (openid, child_id, created_at)
        VALUES (?, ?, ?)
        """
    )

    assert "INSERT INTO child_bindings" in translated
    assert "VALUES (%s, %s, %s)" in translated
    assert "ON CONFLICT (openid, child_id) DO NOTHING" in translated


def test_sqlite_learning_store_persists_child_profiles(tmp_path) -> None:
    db_path = tmp_path / "learning.db"
    store = SQLiteLearningStore(db_path=db_path)

    created = store.upsert_child(
        child_id="child_001",
        name="小明",
        grade=3,
        term_label="2026春季",
    )
    reopened = SQLiteLearningStore(db_path=db_path)
    children = reopened.list_children()

    assert created.child_id == "child_001"
    assert children[0].name == "小明"
    assert children[0].grade == 3
    assert children[0].term_label == "2026春季"


def test_sqlite_learning_store_persists_safety_events(tmp_path) -> None:
    db_path = tmp_path / "learning.db"
    store = SQLiteLearningStore(db_path=db_path)
    event = store.record_safety_event(
        session_id="s_001",
        child_id="child_001",
        event_type="safety.blocked",
        input_text="答案是 180。",
        blocked_text="答案是 180。",
        reason="exact_answer_leak",
    )

    reopened = SQLiteLearningStore(db_path=db_path)
    events = reopened.list_safety_events("child_001")

    assert event.safety_event_id.startswith("safe_")
    assert len(events) == 1
    assert events[0].reason == "exact_answer_leak"
