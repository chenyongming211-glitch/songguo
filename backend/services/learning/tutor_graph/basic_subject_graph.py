from __future__ import annotations

from songguo.backend.services.learning.deeptutor_adapter import (
    DeepTutorLearningAdapter,
    TeachingDraft,
)
from songguo.backend.services.learning.models import LearningPhase
from songguo.backend.services.learning.progress import build_teaching_progress
from songguo.backend.services.learning.store import InMemoryLearningStore
from songguo.backend.services.learning.tutor_graph.state import (
    GraphCreateResult,
    GraphSubmitResult,
)


class BasicSubjectTutorGraph:
    """Minimal guided tutor for non-math routes.

    Submission-level rubric grading decides whether a non-math item enters this
    graph. Once here, the graph asks for the child's reasoning and closes the
    private-test loop after a meaningful reply without revealing the answer.
    """

    def __init__(
        self,
        *,
        store: InMemoryLearningStore,
        adapter: DeepTutorLearningAdapter | None = None,
    ) -> None:
        self.store = store
        self.adapter = adapter or DeepTutorLearningAdapter()

    def start(
        self,
        *,
        child_id: str,
        subject: str,
        grade: int,
        question_text: str,
    ) -> GraphCreateResult:
        normalized_subject = _normalize_basic_subject(subject)
        knowledge_point = knowledge_point_for_basic_subject(normalized_subject)
        draft = self.adapter.generate_hint(
            question_text=question_text,
            grade=grade,
            knowledge_point=knowledge_point,
            hint_level=1,
        )
        message = _safe_start_message(
            subject=normalized_subject,
            question_text=question_text,
            fallback=draft.text,
        )
        session = self.store.create_session(
            child_id=child_id,
            subject=normalized_subject,
            grade=grade,
            question_text=question_text,
            normalized_question=question_text.strip(),
            knowledge_point=knowledge_point,
            current_prompt=message,
            runner_mode="basic_subject",
        )
        self.store.append_message(
            session_id=session.session_id,
            child_id=child_id,
            role="user",
            content=question_text,
            metadata={"graph": "BasicSubjectTutorGraph"},
        )
        self.store.append_message(
            session_id=session.session_id,
            child_id=child_id,
            role="assistant",
            content=message,
            metadata={"graph": "BasicSubjectTutorGraph"},
        )
        self.store.append_event(
            session_id=session.session_id,
            child_id=child_id,
            event_type="basic_subject_tutor.started",
            payload={
                "subject": normalized_subject,
                "knowledge_point": knowledge_point,
                "runner_mode": "basic_subject",
            },
            deeptutor_trace_id=draft.trace_id,
        )
        self._record_ai_call(child_id=child_id, session_id=session.session_id, draft=draft)
        return GraphCreateResult(
            session_id=session.session_id,
            question_text=session.question_text,
            subject=session.subject,
            grade=session.grade,
            phase=session.phase.value,
            hint_level=session.hint_level,
            message=message,
            answer_unlocked=session.answer_unlocked,
            teaching_progress=build_teaching_progress(session),
        )

    def submit_attempt(self, session_id: str, *, child_answer: str) -> GraphSubmitResult:
        session = self.store.require_session(session_id)
        answer = child_answer.strip()
        self.store.append_message(
            session_id=session.session_id,
            child_id=session.child_id,
            role="user",
            content=child_answer,
            metadata={"graph": "BasicSubjectTutorGraph"},
        )
        if not answer:
            message = _empty_answer_prompt(session.subject)
            updated = self.store.update_session(
                session.session_id,
                hint_level=min(3, session.hint_level + 1),
                attempt_count=session.attempt_count + 1,
                current_prompt=message,
            )
            self.store.append_event(
                session_id=session.session_id,
                child_id=session.child_id,
                event_type="basic_subject_tutor.needs_child_reply",
                payload={"subject": session.subject},
            )
            return GraphSubmitResult(
                correct=False,
                phase=updated.phase.value,
                hint_level=updated.hint_level,
                message=message,
                answer_unlocked=updated.answer_unlocked,
                misconception_tag="empty_or_off_task",
                teaching_progress=build_teaching_progress(updated),
            )

        message = _completion_message(session.subject)
        updated = self.store.update_session(
            session.session_id,
            phase=LearningPhase.SIMILAR_PRACTICE,
            attempt_count=session.attempt_count + 1,
            current_prompt=message,
        )
        self.store.append_message(
            session_id=session.session_id,
            child_id=session.child_id,
            role="assistant",
            content=message,
            metadata={"graph": "BasicSubjectTutorGraph"},
        )
        self.store.append_event(
            session_id=session.session_id,
            child_id=session.child_id,
            event_type="basic_subject_tutor.completed",
            payload={"subject": session.subject, "answer_unlocked": False},
        )
        self.store.record_ai_call(
            child_id=session.child_id,
            session_id=session.session_id,
            provider="basic_subject_tutor",
            model="deterministic",
            operation="basic_subject_tutor.submit",
            token_estimate=max(1, len(message) // 4),
            status="success",
        )
        return GraphSubmitResult(
            correct=True,
            phase=updated.phase.value,
            hint_level=updated.hint_level,
            message=message,
            answer_unlocked=False,
            teaching_progress=build_teaching_progress(updated),
        )

    def _record_ai_call(
        self,
        *,
        child_id: str,
        session_id: str,
        draft: TeachingDraft,
    ) -> None:
        self.store.record_ai_call(
            child_id=child_id,
            session_id=session_id,
            provider="basic_subject_tutor",
            model="deterministic",
            operation=draft.action,
            token_estimate=max(1, len(draft.text) // 4),
            status="success",
        )


def knowledge_point_for_basic_subject(subject: str) -> str:
    normalized = _normalize_basic_subject(subject)
    if normalized == "english":
        return "english_sentence_pattern"
    if normalized == "chinese":
        return "chinese_reading_summary"
    return "general_learning_strategy"


def is_basic_subject_route(route_to: str | None) -> bool:
    return (route_to or "") in {
        "chinese_basic_tutor",
        "english_basic_tutor",
        "general_basic_tutor",
    }


def _normalize_basic_subject(subject: str | None) -> str:
    normalized = (subject or "").strip().lower()
    if normalized in {"english", "chinese"}:
        return normalized
    return "general"


def _safe_start_message(*, subject: str, question_text: str, fallback: str) -> str:
    if subject == "english":
        return "这是一道英语题。先不直接给答案，我们先看句子里的时间线索和动作形式。你先说说：这句话是在说现在、过去，还是将来？"
    if subject == "chinese":
        return "这是一道语文题。先不直接改答案，我们先抓题目要求。你先说说：题目让你概括、理解词句，还是回答原因？"
    return fallback or "先不急着给答案。你先说说题目问什么，以及你的答案是怎么想出来的。"


def _empty_answer_prompt(subject: str) -> str:
    if subject == "english":
        return "先用一句话说说你看到的时间线索，或者把你想改的句子再写一遍。"
    if subject == "chinese":
        return "先用一句话说说题目问什么，或者把你准备保留的关键词写出来。"
    return "先把你的想法写一句话，系统再继续陪你往下看。"


def _completion_message(subject: str) -> str:
    if subject == "english":
        return "已记录你的修正思路。本轮先完成：你已经抓住了时间线索和动作形式，后面可以继续用同样方法核对句子。"
    if subject == "chinese":
        return "已记录你的理解思路。本轮先完成：你已经先对齐题目要求，再整理答案，后面可以继续按这个方法检查。"
    return "已记录你的思路。本轮先完成：先确认题目要求，再说明自己的判断依据。"
