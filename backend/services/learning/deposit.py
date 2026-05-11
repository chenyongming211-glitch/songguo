from __future__ import annotations

from songguo.backend.services.learning.labels import knowledge_point_label, misconception_label
from songguo.backend.services.learning.llm_session_runner import LLMSessionOutput
from songguo.backend.services.learning.memory_profile import build_learning_memory
from songguo.backend.services.learning.reporting import (
    LearningDeposit,
    MistakeRecord,
    PracticeRecord,
    QuestionRecord,
)
from songguo.backend.services.learning.store import InMemoryLearningStore


def save_learning_deposit_from_llm_output(
    store: InMemoryLearningStore,
    *,
    session_id: str,
    output: LLMSessionOutput,
) -> LearningDeposit:
    session = store.require_session(session_id)
    delta = output.learning_deposit_delta
    knowledge_point = _normalize_knowledge_point(delta.knowledge_point or session.knowledge_point)
    misconception = delta.main_misconception or output.structured_state.main_misconception
    question_record = QuestionRecord(
        question_text=session.question_text,
        subject=session.subject,
        grade=session.grade,
        knowledge_point=knowledge_point,
        knowledge_point_label=knowledge_point_label(knowledge_point),
        question_type=delta.question_type or knowledge_point,
    )
    mistake_record = (
        MistakeRecord(
            student_answer=_latest_mistake_answer(
                store,
                session_id=session_id,
                misconception=misconception,
            ),
            is_correct=False,
            main_error_reason=misconception,
            main_error_reason_label=misconception_label(misconception),
            evidence=_latest_mistake_evidence(
                store,
                session_id=session_id,
                misconception=misconception,
            )
            or delta.evidence
            or "本次会话已完成，系统记录了学习表现。",
            hint_level_used=session.hint_level,
            need_review=delta.need_review,
        )
        if misconception
        else None
    )
    practice_records = [
        PracticeRecord(
            question=item.question,
            knowledge_point=item.knowledge_point,
            difficulty=item.difficulty,
            answer=item.answer,
        )
        for item in output.practice_items
    ]
    deposit = LearningDeposit(
        child_id=session.child_id,
        session_id=session.session_id,
        question_record=question_record,
        mistake_record=mistake_record,
        practice_records=practice_records,
        student_memory=build_learning_memory(store, child_id=session.child_id),
        parent_summary=(
            delta.parent_summary
            or f"本次主要练习{knowledge_point_label(knowledge_point)}，建议继续完成同类练习。"
        ),
    )
    store.save_learning_deposit(session_id, deposit.model_dump(mode="json"))
    return deposit


def _latest_user_answer(store: InMemoryLearningStore, *, session_id: str) -> str:
    for message in reversed(store.list_messages(session_id)):
        if message.role == "user":
            return message.content
    return ""


def _latest_mistake_answer(
    store: InMemoryLearningStore,
    *,
    session_id: str,
    misconception: str | None,
) -> str:
    latest_child_answer = ""
    latest_wrong_answer = ""
    for event in store.list_events(session_id):
        if event.event_type == "child.attempt_submitted":
            latest_child_answer = str(event.payload.get("child_answer") or "")
            continue
        if event.event_type != "wrong_question.recorded":
            continue
        payload = event.payload or {}
        event_misconception = payload.get("last_misconception")
        if misconception and event_misconception != misconception:
            continue
        latest_wrong_answer = latest_child_answer
    return latest_wrong_answer or _latest_user_answer(store, session_id=session_id)


def _latest_mistake_evidence(
    store: InMemoryLearningStore,
    *,
    session_id: str,
    misconception: str | None,
) -> str:
    latest_child_answer = ""
    latest_evidence = ""
    for event in store.list_events(session_id):
        if event.event_type == "child.attempt_submitted":
            latest_child_answer = str(event.payload.get("child_answer") or "")
            continue
        if event.event_type != "wrong_question.recorded":
            continue
        payload = event.payload or {}
        event_misconception = payload.get("last_misconception")
        if misconception and event_misconception != misconception:
            continue
        if latest_child_answer:
            latest_evidence = (
                f"孩子本次提交了 {latest_child_answer}，"
                f"系统判断为{misconception_label(event_misconception)}。"
            )
        else:
            latest_evidence = str(payload.get("mistake_summary") or "")
    return latest_evidence


def _normalize_knowledge_point(value: str) -> str:
    mapping = {
        "限载进一应用题": "capacity_round_up",
        "限载进一": "capacity_round_up",
        "两位数乘一位数": "two_digit_times_one_digit",
    }
    return mapping.get(value, value)
