from __future__ import annotations

from collections import Counter
from datetime import date

from pydantic import BaseModel, Field

from songguo.backend.services.learning.labels import knowledge_point_label, misconception_label
from songguo.backend.services.learning.memory_profile import LearningMemory, build_learning_memory
from songguo.backend.services.learning.store import InMemoryLearningStore
from songguo.backend.services.learning.teaching_assets import DEFAULT_MATH_ASSET_LIBRARY


class QuestionRecord(BaseModel):
    question_text: str
    subject: str
    grade: int
    knowledge_point: str
    knowledge_point_label: str
    question_type: str


class MistakeRecord(BaseModel):
    student_answer: str
    is_correct: bool
    main_error_reason: str | None = None
    main_error_reason_label: str
    evidence: str
    hint_level_used: int
    need_review: bool = True


class PracticeRecord(BaseModel):
    question: str
    knowledge_point: str
    difficulty: int
    answer: str | None = None


class LearningDeposit(BaseModel):
    child_id: str
    session_id: str
    question_record: QuestionRecord
    mistake_record: MistakeRecord | None = None
    practice_records: list[PracticeRecord] = Field(default_factory=list)
    student_memory: LearningMemory
    parent_summary: str


class WeeklyReport(BaseModel):
    child_id: str
    week_start: str | None = None
    week_end: str | None = None
    session_count: int
    wrong_question_count: int
    completed_session_count: int
    top_knowledge_points: list[str] = Field(default_factory=list)
    common_misconceptions: list[str] = Field(default_factory=list)
    average_hint_level: float
    practice_generated_count: int
    parent_summary: str


class SessionFeedback(BaseModel):
    child_id: str
    session_id: str
    question_text: str
    knowledge_point: str
    knowledge_point_label: str
    main_error_reason: str | None = None
    main_error_reason_label: str
    evidence: str
    parent_suggestion: str
    next_practice_count: int = 3
    summary: str
    hint_level: int
    answer_unlocked: bool


class SummaryDraft(BaseModel):
    child_id: str
    scope: str
    session_count: int
    wrong_question_count: int
    top_knowledge_points: list[str] = Field(default_factory=list)
    common_misconceptions: list[str] = Field(default_factory=list)
    parent_summary: str
    next_actions: list[str] = Field(default_factory=list)


class WrongQuestionSummary(BaseModel):
    question_id: str
    session_id: str
    normalized_question: str
    knowledge_point: str
    knowledge_point_label: str
    last_misconception: str | None
    misconception_label: str
    highest_hint_level: int
    explanation_unlocked: bool
    practice_completed: bool


class WrongQuestionList(BaseModel):
    child_id: str
    items: list[WrongQuestionSummary]


def build_weekly_report(
    store: InMemoryLearningStore,
    *,
    child_id: str,
    week_start: date | None = None,
    week_end: date | None = None,
) -> WeeklyReport:
    saved_weekly_summaries = store.list_learning_summaries(child_id=child_id, scope="weekly")
    latest_summary = saved_weekly_summaries[-1] if saved_weekly_summaries else None
    sessions = [session for session in store.sessions.values() if session.child_id == child_id]
    wrong_questions = store.list_wrong_questions(child_id)
    knowledge_counts = Counter(item.knowledge_point for item in wrong_questions)
    misconception_counts = Counter(
        item.last_misconception for item in wrong_questions if item.last_misconception
    )
    hint_levels = [item.highest_hint_level for item in wrong_questions]
    practice_generated_count = sum(
        1
        for session in sessions
        for event in store.list_events(session.session_id)
        if event.event_type == "practice.generated"
    )

    top_knowledge_points = (
        latest_summary.top_knowledge_points
        if latest_summary
        else [item for item, _ in knowledge_counts.most_common(3)]
    )
    common_misconceptions = (
        latest_summary.common_misconceptions
        if latest_summary
        else [item for item, _ in misconception_counts.most_common(3)]
    )
    parent_summary = (
        latest_summary.parent_summary
        if latest_summary
        else _parent_summary(top_knowledge_points, common_misconceptions)
    )

    return WeeklyReport(
        child_id=child_id,
        week_start=week_start.isoformat() if week_start else None,
        week_end=week_end.isoformat() if week_end else None,
        session_count=latest_summary.session_count if latest_summary else len(sessions),
        wrong_question_count=latest_summary.wrong_question_count if latest_summary else len(wrong_questions),
        completed_session_count=(
            latest_summary.completed_session_count
            if latest_summary
            else sum(1 for session in sessions if session.completed_at is not None)
        ),
        top_knowledge_points=top_knowledge_points,
        common_misconceptions=common_misconceptions,
        average_hint_level=round(sum(hint_levels) / len(hint_levels), 2) if hint_levels else 0.0,
        practice_generated_count=practice_generated_count,
        parent_summary=parent_summary,
    )


def build_session_feedback(
    store: InMemoryLearningStore,
    *,
    child_id: str,
    session_id: str,
) -> SessionFeedback:
    session = store.get_session(session_id)
    if session is None or session.child_id != child_id:
        raise KeyError(session_id)

    submission_context = _submission_context_for_tutor_session(
        store,
        child_id=child_id,
        tutor_session_id=session_id,
    )
    wrong_question = _latest_wrong_question_for_session(
        store,
        child_id=child_id,
        session_id=session_id,
    )
    item_misconception = (
        str(submission_context["item"].misconception_tag)
        if submission_context and submission_context["item"].misconception_tag
        else None
    )
    main_error_reason = (
        wrong_question.last_misconception
        if wrong_question
        else item_misconception or session.last_misconception
    )
    reason_label = misconception_label(main_error_reason)
    point_label = knowledge_point_label(session.knowledge_point)
    evidence = _session_feedback_evidence(
        store,
        session_id=session_id,
        main_error_reason=main_error_reason,
        submission_context=submission_context,
    )
    parent_suggestion = _parent_suggestion(session.knowledge_point, main_error_reason)
    if main_error_reason:
        summary = f"本次主要练习{point_label}，当前卡点是{reason_label}。{parent_suggestion}"
    else:
        summary = f"本次主要练习{point_label}，孩子已完成本题。{parent_suggestion}"

    return SessionFeedback(
        child_id=child_id,
        session_id=session.session_id,
        question_text=session.question_text,
        knowledge_point=session.knowledge_point,
        knowledge_point_label=point_label,
        main_error_reason=main_error_reason,
        main_error_reason_label=reason_label,
        evidence=evidence,
        parent_suggestion=parent_suggestion,
        next_practice_count=3,
        summary=summary,
        hint_level=session.hint_level,
        answer_unlocked=session.answer_unlocked,
    )


def build_summary_draft(
    store: InMemoryLearningStore,
    *,
    child_id: str,
    scope: str = "weekly",
) -> SummaryDraft:
    report = build_weekly_report(store, child_id=child_id)
    return SummaryDraft(
        child_id=child_id,
        scope=scope,
        session_count=report.session_count,
        wrong_question_count=report.wrong_question_count,
        top_knowledge_points=report.top_knowledge_points,
        common_misconceptions=report.common_misconceptions,
        parent_summary=report.parent_summary,
        next_actions=_summary_next_actions(
            report.top_knowledge_points,
            report.common_misconceptions,
        ),
    )


def build_wrong_question_list(
    store: InMemoryLearningStore,
    *,
    child_id: str,
) -> WrongQuestionList:
    return WrongQuestionList(
        child_id=child_id,
        items=[
            WrongQuestionSummary(
                question_id=item.question_id,
                session_id=item.session_id,
                normalized_question=item.normalized_question,
                knowledge_point=item.knowledge_point,
                knowledge_point_label=knowledge_point_label(item.knowledge_point),
                last_misconception=item.last_misconception,
                misconception_label=misconception_label(item.last_misconception),
                highest_hint_level=item.highest_hint_level,
                explanation_unlocked=item.explanation_unlocked,
                practice_completed=item.practice_completed,
            )
            for item in store.list_wrong_questions(child_id)
        ],
    )


def build_learning_deposit(
    store: InMemoryLearningStore,
    *,
    child_id: str,
    session_id: str,
    scope: str = "weekly",
) -> LearningDeposit:
    session = store.get_session(session_id)
    if session is None or session.child_id != child_id:
        raise KeyError(session_id)
    persisted = store.get_learning_deposit(session_id)
    if persisted and persisted.get("child_id") == child_id:
        return LearningDeposit.model_validate(persisted)

    question_record = QuestionRecord(
        question_text=session.question_text,
        subject=session.subject,
        grade=session.grade,
        knowledge_point=session.knowledge_point,
        knowledge_point_label=knowledge_point_label(session.knowledge_point),
        question_type=session.knowledge_point,
    )
    mistake_record = _build_mistake_record(store, child_id=child_id, session_id=session_id)
    practice_records = _practice_records_from_events(store, session_id=session_id)
    student_memory = build_learning_memory(store, child_id=child_id, scope=scope)
    feedback = build_session_feedback(store, child_id=child_id, session_id=session_id)

    return LearningDeposit(
        child_id=child_id,
        session_id=session_id,
        question_record=question_record,
        mistake_record=mistake_record,
        practice_records=practice_records,
        student_memory=student_memory,
        parent_summary=feedback.summary,
    )


def _parent_summary(top_knowledge_points: list[str], common_misconceptions: list[str]) -> str:
    if not top_knowledge_points:
        return "本周还没有足够的错题记录。先完成 1-2 次错题引导，再生成建议。"
    point = knowledge_point_label(top_knowledge_points[0])
    misconception = misconception_label(common_misconceptions[0] if common_misconceptions else None)
    return f"这周主要薄弱点是{point}，常见错因是{misconception}。建议每天练 3 道同类小题。"


def _build_mistake_record(
    store: InMemoryLearningStore,
    *,
    child_id: str,
    session_id: str,
) -> MistakeRecord | None:
    wrong_question = _latest_wrong_question_for_session(
        store,
        child_id=child_id,
        session_id=session_id,
    )
    if wrong_question is None:
        return None

    attempt = _latest_wrong_attempt_for_session(
        store,
        session_id=session_id,
        misconception=wrong_question.last_misconception,
    )
    student_answer = attempt.get("child_answer") or ""
    main_error_reason = wrong_question.last_misconception
    return MistakeRecord(
        student_answer=student_answer,
        is_correct=False,
        main_error_reason=main_error_reason,
        main_error_reason_label=misconception_label(main_error_reason),
        evidence=attempt.get("evidence") or wrong_question.mistake_summary,
        hint_level_used=wrong_question.highest_hint_level,
        need_review=not wrong_question.resolved,
    )


def _latest_wrong_attempt_for_session(
    store: InMemoryLearningStore,
    *,
    session_id: str,
    misconception: str | None,
) -> dict[str, str | None]:
    latest_child_answer: str | None = None
    latest_attempt: dict[str, str | None] = {}
    for event in store.list_events(session_id):
        if event.event_type == "child.attempt_submitted":
            latest_child_answer = str(event.payload.get("child_answer") or "")
            continue
        if event.event_type != "attempt.evaluated":
            continue
        payload = event.payload or {}
        if payload.get("correct") is True:
            continue
        current_misconception = (
            payload.get("misconception_tag")
            or payload.get("misconception")
        )
        if misconception and current_misconception != misconception:
            continue
        latest_attempt = {
            "child_answer": latest_child_answer,
            "evidence": str(payload.get("evidence") or ""),
            "misconception": str(current_misconception or ""),
        }
    return latest_attempt


def _latest_wrong_question_for_session(
    store: InMemoryLearningStore,
    *,
    child_id: str,
    session_id: str,
):
    wrong_questions = [
        item
        for item in store.list_wrong_questions(child_id)
        if item.session_id == session_id
    ]
    if not wrong_questions:
        return None
    return max(wrong_questions, key=lambda item: item.updated_at)


def _submission_context_for_tutor_session(
    store: InMemoryLearningStore,
    *,
    child_id: str,
    tutor_session_id: str,
):
    for submission in store.list_submissions(child_id=child_id):
        for queue_item in store.list_tutor_queue_items(submission.submission_id):
            if queue_item.tutor_session_id != tutor_session_id:
                continue
            for item in store.list_submission_items(submission.submission_id):
                if item.item_id == queue_item.item_id:
                    return {
                        "submission": submission,
                        "queue_item": queue_item,
                        "item": item,
                    }
    return None


def _practice_records_from_events(
    store: InMemoryLearningStore,
    *,
    session_id: str,
) -> list[PracticeRecord]:
    records: list[PracticeRecord] = []
    for event in reversed(store.list_events(session_id)):
        if event.event_type != "practice.generated":
            continue
        items = event.payload.get("items")
        if not isinstance(items, list):
            return []
        for item in items:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question") or "").strip()
            if not question:
                continue
            records.append(
                PracticeRecord(
                    question=question,
                    knowledge_point=str(item.get("knowledge_point") or ""),
                    difficulty=int(item.get("difficulty") or 1),
                    answer=item.get("answer"),
                )
            )
        return records
    return records


def _session_feedback_evidence(
    store: InMemoryLearningStore,
    *,
    session_id: str,
    main_error_reason: str | None,
    submission_context=None,
) -> str:
    if submission_context is not None:
        item = submission_context["item"]
        final_answer = _latest_correct_answer_for_session(store, session_id=session_id)
        pieces = [
            f"提交时孩子答“{item.child_answer or '未识别到答案'}”，系统判为{_judge_result_label(item.judge_result)}。"
        ]
        if item.correct_answer:
            pieces.append(f"参考答案是“{item.correct_answer}”。")
        if item.misconception_tag:
            pieces.append(f"主要错因是{misconception_label(item.misconception_tag)}。")
        if final_answer:
            pieces.append(f"陪练后孩子改答“{final_answer}”，本题已完成。")
        return "".join(pieces)

    for event in reversed(store.list_events(session_id)):
        child_answer = event.payload.get("child_answer")
        misconception = event.payload.get("misconception_tag") or event.payload.get("misconception")
        if child_answer and (not main_error_reason or misconception == main_error_reason):
            return f"孩子本次提交了 {child_answer}，系统判断为{misconception_label(misconception)}。"
    wrong_answer = _latest_wrong_answer_for_session(
        store,
        session_id=session_id,
        misconception=main_error_reason,
    )
    if wrong_answer:
        return f"孩子本次提交了 {wrong_answer}，系统判断为{misconception_label(main_error_reason)}。"
    if main_error_reason:
        return f"系统在本次引导中记录了{misconception_label(main_error_reason)}。"
    for event in reversed(store.list_events(session_id)):
        if event.event_type == "attempt.evaluated" and event.payload.get("correct") is True:
            child_answer = event.payload.get("child_answer")
            if child_answer:
                return f"孩子本次提交了 {child_answer}，系统判断已完成本题。"
    return "本次还没有足够的尝试记录，需要继续观察。"


def _latest_correct_answer_for_session(
    store: InMemoryLearningStore,
    *,
    session_id: str,
) -> str:
    latest_child_answer = ""
    latest_correct_answer = ""
    for event in store.list_events(session_id):
        if event.event_type == "child.attempt_submitted":
            latest_child_answer = str(event.payload.get("child_answer") or "")
            continue
        if event.event_type == "attempt.evaluated" and event.payload.get("correct") is True:
            latest_correct_answer = latest_child_answer
    return latest_correct_answer


def _judge_result_label(value) -> str:
    result = str(value)
    if result == "correct":
        return "正确"
    if result == "wrong":
        return "错误"
    if result == "needs_manual_confirm":
        return "待确认"
    return "未判题"


def _latest_wrong_answer_for_session(
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
    return latest_wrong_answer


def _parent_suggestion(knowledge_point: str, main_error_reason: str | None) -> str:
    asset_suggestion = _asset_parent_suggestion(main_error_reason)
    if asset_suggestion:
        return asset_suggestion
    if (
        knowledge_point == "two_digit_times_one_digit"
        and main_error_reason == "treated_x5_like_x10"
    ):
        return "建议先问孩子乘以 10 是多少，再让孩子说出乘以 5 是乘以 10 的一半。"
    if main_error_reason == "unknown_misconception":
        return "建议让孩子先说一遍题意，再只做第一步，不急着算最终答案。"
    if main_error_reason is None:
        return "建议继续完成同类练习，确认孩子能独立迁移。"
    return "建议让孩子先复述题目，再完成 1-3 道同类练习。"


def _asset_parent_suggestion(main_error_reason: str | None) -> str | None:
    if not main_error_reason:
        return None
    try:
        misconception = DEFAULT_MATH_ASSET_LIBRARY.require_misconception(main_error_reason)
    except KeyError:
        return None
    tips: list[str] = [misconception.parent_explanation]
    for skill_id in misconception.skill_ids:
        try:
            card = DEFAULT_MATH_ASSET_LIBRARY.require_concept_card_for_skill(skill_id)
        except KeyError:
            continue
        tips.append(card.parent_tip)
        break
    return " ".join(tips)


def _summary_next_actions(
    top_knowledge_points: list[str],
    common_misconceptions: list[str],
) -> list[str]:
    if not top_knowledge_points:
        return ["先完成一次错题引导，积累可分析的学习记录。"]
    point = knowledge_point_label(top_knowledge_points[0])
    misconception = misconception_label(common_misconceptions[0] if common_misconceptions else None)
    return [
        f"每天练 3 道{point}同类题。",
        f"辅导时重点观察是否还会出现“{misconception}”。",
        "先让孩子说思路，再决定是否看完整讲解。",
    ]
