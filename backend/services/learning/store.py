from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import sqlite3
from typing import Any

from songguo.backend.services.learning.models import (
    AICallLog,
    ChildProfile,
    LearningEvent,
    LearningMessage,
    LearningSession,
    LearningSummary,
    ReminderSubscription,
    ResumeSnapshot,
    SafetyEvent,
    WrongQuestion,
    utc_now,
)
from songguo.backend.services.learning.progress import build_teaching_progress
from songguo.backend.services.learning.real_model_client import songguo_data_root
from songguo.backend.services.learning.submission_models import (
    EvidenceType,
    JudgeResult,
    LearningItem,
    LearningItemStatus,
    LearningSubmission,
    LearningSubmissionSnapshot,
    LearningSubmissionStatus,
    MasteryEvidence,
    MasteryState,
    SourceType,
    TutorQueueItem,
    TutorQueueStatus,
)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _sqlite_schema_migrations() -> list[str]:
    return [
        "ALTER TABLE children ADD COLUMN family_id TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE child_bindings ADD COLUMN family_id TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE learning_sessions ADD COLUMN family_id TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE wrong_questions ADD COLUMN knowledge_point TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE wrong_questions ADD COLUMN family_id TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE wrong_questions ADD COLUMN last_misconception TEXT",
        "ALTER TABLE wrong_questions ADD COLUMN highest_hint_level INTEGER NOT NULL DEFAULT 1",
        "ALTER TABLE wrong_questions ADD COLUMN resolved INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE wrong_questions ADD COLUMN practice_completed INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE learning_deposits ADD COLUMN knowledge_point TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE learning_deposits ADD COLUMN question_type TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE learning_deposits ADD COLUMN main_misconception TEXT",
        "ALTER TABLE learning_deposits ADD COLUMN need_review INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE learning_submissions ADD COLUMN family_id TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE learning_submission_items ADD COLUMN family_id TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE mastery_evidence ADD COLUMN family_id TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE tutor_queue_items ADD COLUMN family_id TEXT NOT NULL DEFAULT ''",
    ]


def _build_resume_messages(
    events: list[LearningEvent],
    *,
    current_prompt: str,
    hint_level: int,
    limit: int = 20,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for event in events:
        message = _message_from_event(event)
        if not message:
            continue
        if (
            messages
            and messages[-1]["role"] == message["role"]
            and messages[-1]["content"] == message["content"]
        ):
            continue
        messages.append(message)

    if current_prompt and (
        not messages
        or messages[-1]["role"] != "assistant"
        or messages[-1]["content"] != current_prompt
    ):
        messages.append(
            {
                "role": "assistant",
                "content": current_prompt,
                "hint_level": hint_level,
            }
        )
    return messages[-limit:]


def _message_from_event(event: LearningEvent) -> dict[str, Any] | None:
    payload = event.payload or {}
    if event.event_type in {"key_point.released", "hint.generated"}:
        content = str(payload.get("prompt") or payload.get("message") or "").strip()
        if not content:
            return None
        return {
            "role": "assistant",
            "content": content,
            "hint_level": payload.get("hint_level"),
        }
    if event.event_type in {"child.attempt_submitted", "practice.followup_requested"}:
        content = str(payload.get("child_answer") or "").strip()
        if not content:
            return None
        return {"role": "user", "content": content}
    if event.event_type == "practice.generated":
        content = str(payload.get("message") or "").strip()
        if not content:
            items = payload.get("items")
            if isinstance(items, list) and items:
                lines = ["你已经找到方法了。我们再练习 1-3 道同类题，确认真的掌握："]
                for index, item in enumerate(items, start=1):
                    if isinstance(item, dict) and item.get("question"):
                        lines.append(f"{index}. {item['question']}")
                content = "\n".join(lines)
        if not content:
            return None
        return {"role": "assistant", "content": content}
    return None


def _messages_for_snapshot(messages: list[LearningMessage], *, limit: int = 20) -> list[dict[str, Any]]:
    return [
        {
            "role": message.role,
            "content": message.content,
            **({"metadata": message.metadata} if message.metadata else {}),
        }
        for message in messages[-limit:]
    ]


@dataclass
class InMemoryLearningStore:
    """Small local store for P0 tests and development.

    This keeps product state isolated behind methods that can later be backed by
    SQLite or PostgreSQL without changing the state machine.
    """

    sessions: dict[str, LearningSession] = field(default_factory=dict)
    events: dict[str, list[LearningEvent]] = field(default_factory=dict)
    messages: dict[str, list[LearningMessage]] = field(default_factory=dict)
    wrong_questions: dict[str, list[WrongQuestion]] = field(default_factory=dict)
    learning_deposits: dict[str, dict[str, Any]] = field(default_factory=dict)
    learning_summaries: dict[str, list[LearningSummary]] = field(default_factory=dict)
    photo_reviews: dict[str, dict[str, Any]] = field(default_factory=dict)
    children: dict[str, ChildProfile] = field(default_factory=dict)
    child_bindings: dict[str, set[str]] = field(default_factory=dict)
    child_families: dict[str, str] = field(default_factory=dict)
    openid_families: dict[str, str] = field(default_factory=dict)
    ai_call_logs: list[AICallLog] = field(default_factory=list)
    safety_events: list[SafetyEvent] = field(default_factory=list)
    reminder_subscriptions: dict[str, ReminderSubscription] = field(default_factory=dict)
    submissions: dict[str, LearningSubmission] = field(default_factory=dict)
    submission_items: dict[str, list[LearningItem]] = field(default_factory=dict)
    mastery_evidence: dict[str, list[MasteryEvidence]] = field(default_factory=dict)
    tutor_queue_items: dict[str, list[TutorQueueItem]] = field(default_factory=dict)

    def create_session(
        self,
        *,
        child_id: str,
        subject: str,
        grade: int,
        question_text: str,
        normalized_question: str,
        knowledge_point: str,
        current_prompt: str,
        runner_mode: str = "kernel",
        problem_analysis: dict[str, Any] | None = None,
        current_key_point_id: str | None = None,
        released_key_point_ids: list[str] | None = None,
        mastered_key_point_ids: list[str] | None = None,
    ) -> LearningSession:
        session = LearningSession(
            child_id=child_id,
            family_id=self.get_family_id_for_child(child_id),
            subject=subject,
            grade=grade,
            question_text=question_text,
            normalized_question=normalized_question,
            knowledge_point=knowledge_point,
            current_prompt=current_prompt,
            runner_mode=runner_mode,
            problem_analysis=problem_analysis,
            current_key_point_id=current_key_point_id,
            released_key_point_ids=released_key_point_ids or [],
            mastered_key_point_ids=mastered_key_point_ids or [],
        )
        self.sessions[session.session_id] = session
        self.events[session.session_id] = []
        self.messages[session.session_id] = []
        self.append_event(
            session_id=session.session_id,
            child_id=child_id,
            event_type="session.created",
            payload={
                "question_text": question_text,
                "normalized_question": normalized_question,
                "knowledge_point": knowledge_point,
                "hint_level": session.hint_level,
                "answer_unlocked": session.answer_unlocked,
                "runner_mode": session.runner_mode,
                "current_key_point_id": session.current_key_point_id,
            },
        )
        return session

    def get_session(self, session_id: str) -> LearningSession | None:
        return self.sessions.get(session_id)

    def require_session(self, session_id: str) -> LearningSession:
        session = self.get_session(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def update_session(self, session_id: str, **updates: Any) -> LearningSession:
        session = self.require_session(session_id)
        updated = session.model_copy(update={**updates, "updated_at": utc_now()})
        self.sessions[session_id] = updated
        return updated

    def append_event(
        self,
        *,
        session_id: str,
        child_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        deeptutor_trace_id: str | None = None,
        leakage_check_result: dict[str, Any] | None = None,
    ) -> LearningEvent:
        event = LearningEvent(
            session_id=session_id,
            child_id=child_id,
            event_type=event_type,
            payload=payload or {},
            deeptutor_trace_id=deeptutor_trace_id,
            leakage_check_result=leakage_check_result,
        )
        self.events.setdefault(session_id, []).append(event)
        return event

    def list_events(self, session_id: str) -> list[LearningEvent]:
        return list(self.events.get(session_id, []))

    def append_message(
        self,
        *,
        session_id: str,
        child_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> LearningMessage:
        message = LearningMessage(
            session_id=session_id,
            child_id=child_id,
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self.messages.setdefault(session_id, []).append(message)
        return message

    def list_messages(self, session_id: str) -> list[LearningMessage]:
        return list(self.messages.get(session_id, []))

    def list_sessions(self, child_id: str | None = None) -> list[LearningSession]:
        sessions = list(self.sessions.values())
        if child_id:
            sessions = [session for session in sessions if session.child_id == child_id]
        return sorted(sessions, key=lambda session: session.updated_at, reverse=True)

    def get_resume_snapshot(self, session_id: str) -> ResumeSnapshot:
        session = self.require_session(session_id)
        stored_messages = self.list_messages(session_id)
        last_messages = (
            _messages_for_snapshot(stored_messages)
            if stored_messages
            else _build_resume_messages(
                self.list_events(session_id),
                current_prompt=session.current_prompt,
                hint_level=session.hint_level,
            )
        )
        return ResumeSnapshot(
            session_id=session.session_id,
            question_text=session.question_text,
            subject=session.subject,
            grade=session.grade,
            phase=session.phase,
            hint_level=session.hint_level,
            attempt_count=session.attempt_count,
            answer_unlocked=session.answer_unlocked,
            current_prompt=session.current_prompt,
            teaching_progress=build_teaching_progress(session),
            last_misconception=session.last_misconception,
            last_messages=last_messages,
        )

    def record_wrong_question(
        self,
        *,
        session_id: str,
        child_id: str,
        normalized_question: str,
        knowledge_point: str,
        mistake_summary: str,
        last_misconception: str | None,
        highest_hint_level: int,
        explanation_unlocked: bool = False,
        practice_completed: bool = False,
    ) -> WrongQuestion:
        wrong_question = WrongQuestion(
            child_id=child_id,
            family_id=self.get_family_id_for_child(child_id),
            session_id=session_id,
            normalized_question=normalized_question,
            knowledge_point=knowledge_point,
            mistake_summary=mistake_summary,
            last_misconception=last_misconception,
            highest_hint_level=highest_hint_level,
            explanation_unlocked=explanation_unlocked,
            practice_completed=practice_completed,
        )
        self.wrong_questions.setdefault(child_id, []).append(wrong_question)
        self.append_event(
            session_id=session_id,
            child_id=child_id,
            event_type="wrong_question.recorded",
            payload=wrong_question.model_dump(mode="json"),
        )
        return wrong_question

    def list_wrong_questions(self, child_id: str) -> list[WrongQuestion]:
        return list(self.wrong_questions.get(child_id, []))

    def update_wrong_question_practice_result(
        self,
        *,
        question_id: str,
        child_id: str,
        correct: bool,
    ) -> WrongQuestion:
        questions = self.wrong_questions.get(child_id, [])
        for index, question in enumerate(questions):
            if question.question_id == question_id:
                updated = question.model_copy(
                    update={
                        "practice_completed": True,
                        "resolved": bool(correct),
                        "updated_at": utc_now(),
                    }
                )
                questions[index] = updated
                self.append_event(
                    session_id=updated.session_id,
                    child_id=child_id,
                    event_type="practice.completed",
                    payload={
                        "question_id": updated.question_id,
                        "correct": bool(correct),
                        "resolved": updated.resolved,
                    },
                )
                return updated
        raise KeyError(question_id)

    def save_photo_review(self, review: dict[str, Any]) -> None:
        self.photo_reviews[str(review["review_id"])] = dict(review)

    def get_photo_review(self, review_id: str) -> dict[str, Any] | None:
        review = self.photo_reviews.get(review_id)
        return dict(review) if review else None

    def upsert_child(
        self,
        *,
        child_id: str,
        name: str,
        grade: int,
        term_label: str | None = None,
        family_id: str | None = None,
    ) -> ChildProfile:
        existing = self.children.get(child_id)
        resolved_family_id = (
            family_id
            or (existing.family_id if existing else "")
            or self.child_families.get(child_id, "")
        )
        profile = ChildProfile(
            child_id=child_id,
            family_id=resolved_family_id,
            name=name,
            grade=grade,
            term_label=term_label,
            created_at=existing.created_at if existing else utc_now(),
        )
        self.children[child_id] = profile
        if resolved_family_id:
            self.child_families[child_id] = resolved_family_id
        return profile

    def get_child(self, child_id: str) -> ChildProfile | None:
        child = self.children.get(child_id)
        if child is None and child_id in self.child_families:
            return ChildProfile(
                child_id=child_id,
                family_id=self.child_families[child_id],
                name="",
                grade=0,
            )
        return child

    def list_children(self) -> list[ChildProfile]:
        return sorted(self.children.values(), key=lambda child: child.created_at)

    def bind_child_to_openid(self, *, openid: str, child_id: str) -> None:
        family_id = self.get_family_id_for_child(child_id) or self.get_family_id_for_openid(openid)
        self.openid_families[openid] = family_id
        self.child_families[child_id] = family_id
        self.child_bindings.setdefault(openid, set()).add(child_id)
        if child_id in self.children:
            child = self.children[child_id]
            self.children[child_id] = child.model_copy(
                update={"family_id": child.family_id or family_id, "updated_at": utc_now()}
            )

    def is_child_bound_to_openid(self, *, openid: str, child_id: str) -> bool:
        return child_id in self.child_bindings.get(openid, set())

    def list_children_for_openid(self, openid: str) -> list[ChildProfile]:
        child_ids = self.child_bindings.get(openid, set())
        return sorted(
            [
                child
                for child in self.children.values()
                if child.child_id in child_ids
            ],
            key=lambda child: child.created_at,
        )

    def get_family_id_for_openid(self, openid: str) -> str:
        if openid not in self.openid_families:
            self.openid_families[openid] = f"family_{openid}"
        return self.openid_families[openid]

    def get_family_id_for_child(self, child_id: str) -> str:
        if child_id in self.child_families:
            return self.child_families[child_id]
        child = self.children.get(child_id)
        if child and child.family_id:
            self.child_families[child_id] = child.family_id
            return child.family_id
        for openid, child_ids in self.child_bindings.items():
            if child_id in child_ids:
                family_id = self.get_family_id_for_openid(openid)
                self.child_families[child_id] = family_id
                return family_id
        return ""

    def record_ai_call(
        self,
        *,
        child_id: str,
        session_id: str,
        provider: str,
        model: str,
        operation: str,
        token_estimate: int,
        status: str,
    ) -> AICallLog:
        log = AICallLog(
            child_id=child_id,
            session_id=session_id,
            provider=provider,
            model=model,
            operation=operation,
            token_estimate=max(0, int(token_estimate)),
            status=status,
        )
        self.ai_call_logs.append(log)
        return log

    def list_ai_call_logs(self, child_id: str | None = None) -> list[AICallLog]:
        if child_id is None:
            return list(self.ai_call_logs)
        return [log for log in self.ai_call_logs if log.child_id == child_id]

    def save_learning_deposit(self, session_id: str, deposit: dict[str, Any]) -> dict[str, Any]:
        self.learning_deposits[session_id] = dict(deposit)
        return self.learning_deposits[session_id]

    def get_learning_deposit(self, session_id: str) -> dict[str, Any] | None:
        deposit = self.learning_deposits.get(session_id)
        return dict(deposit) if deposit else None

    def save_learning_summary(self, summary: LearningSummary) -> LearningSummary:
        self.learning_summaries.setdefault(summary.child_id, []).append(summary)
        return summary

    def list_learning_summaries(
        self,
        *,
        child_id: str,
        scope: str | None = None,
    ) -> list[LearningSummary]:
        summaries = list(self.learning_summaries.get(child_id, []))
        if scope:
            summaries = [summary for summary in summaries if summary.scope == scope]
        return summaries

    def record_safety_event(
        self,
        *,
        session_id: str,
        child_id: str,
        event_type: str,
        input_text: str | None = None,
        blocked_text: str | None = None,
        reason: str,
    ) -> SafetyEvent:
        event = SafetyEvent(
            session_id=session_id,
            child_id=child_id,
            event_type=event_type,
            input_text=input_text,
            blocked_text=blocked_text,
            reason=reason,
        )
        self.safety_events.append(event)
        return event

    def list_safety_events(self, child_id: str | None = None) -> list[SafetyEvent]:
        if child_id is None:
            return list(self.safety_events)
        return [event for event in self.safety_events if event.child_id == child_id]

    def create_submission(
        self,
        *,
        child_id: str,
        subject: str,
        grade: int,
        source_type: SourceType | str,
        raw_text: str = "",
        image_refs: list[str] | None = None,
        audio_refs: list[str] | None = None,
        data_json: dict[str, Any] | None = None,
    ) -> LearningSubmission:
        submission = LearningSubmission(
            child_id=child_id,
            family_id=self.get_family_id_for_child(child_id),
            subject=subject,
            grade=grade,
            source_type=source_type,
            raw_text=raw_text,
            image_refs=image_refs or [],
            audio_refs=audio_refs or [],
            data_json=data_json or {},
        )
        self.submissions[submission.submission_id] = submission
        self.submission_items[submission.submission_id] = []
        self.tutor_queue_items[submission.submission_id] = []
        return submission

    def get_submission(self, submission_id: str) -> LearningSubmission | None:
        return self.submissions.get(submission_id)

    def require_submission(self, submission_id: str) -> LearningSubmission:
        submission = self.get_submission(submission_id)
        if submission is None:
            raise KeyError(submission_id)
        return submission

    def update_submission(self, submission_id: str, **updates: Any) -> LearningSubmission:
        submission = self.require_submission(submission_id)
        updated = submission.model_copy(update={**updates, "updated_at": utc_now()})
        self.submissions[submission_id] = updated
        return updated

    def list_submissions(self, child_id: str | None = None) -> list[LearningSubmission]:
        submissions = list(self.submissions.values())
        if child_id:
            submissions = [submission for submission in submissions if submission.child_id == child_id]
        return sorted(submissions, key=lambda submission: submission.updated_at, reverse=True)

    def add_submission_item(self, item: LearningItem) -> LearningItem:
        self.require_submission(item.submission_id)
        submission = self.require_submission(item.submission_id)
        stored = item.model_copy(
            update={
                "family_id": item.family_id or submission.family_id or self.get_family_id_for_child(item.child_id)
            }
        )
        self.submission_items.setdefault(item.submission_id, []).append(stored)
        self._refresh_submission_counts(item.submission_id)
        return stored

    def update_submission_item(self, item_id: str, **updates: Any) -> LearningItem:
        for submission_id, items in self.submission_items.items():
            for index, item in enumerate(items):
                if item.item_id == item_id:
                    updated = item.model_copy(update={**updates, "updated_at": utc_now()})
                    items[index] = updated
                    self._refresh_submission_counts(submission_id)
                    return updated
        raise KeyError(item_id)

    def list_submission_items(self, submission_id: str) -> list[LearningItem]:
        self.require_submission(submission_id)
        return sorted(
            list(self.submission_items.get(submission_id, [])),
            key=lambda item: item.item_index,
        )

    def save_mastery_evidence(
        self,
        *,
        item_id: str,
        child_id: str,
        question_type_id: str,
        evidence_type: EvidenceType | str,
        is_correct: bool,
        mastery_state_after: MasteryState | str,
        review_due: bool = False,
        solved_without_help: bool = False,
        misconception_tag: str | None = None,
        confidence: float = 0.0,
        data_json: dict[str, Any] | None = None,
    ) -> MasteryEvidence:
        evidence = MasteryEvidence(
            item_id=item_id,
            child_id=child_id,
            family_id=self.get_family_id_for_child(child_id) or self._family_id_for_item(item_id),
            question_type_id=question_type_id,
            evidence_type=evidence_type,
            is_correct=is_correct,
            mastery_state_after=mastery_state_after,
            review_due=review_due,
            solved_without_help=solved_without_help,
            misconception_tag=misconception_tag,
            confidence=confidence,
            data_json=data_json or {},
        )
        self.mastery_evidence.setdefault(item_id, []).append(evidence)
        return evidence

    def list_mastery_evidence(self, item_id: str | None = None, child_id: str | None = None) -> list[MasteryEvidence]:
        if item_id:
            items = list(self.mastery_evidence.get(item_id, []))
        else:
            items = [evidence for values in self.mastery_evidence.values() for evidence in values]
        if child_id:
            items = [evidence for evidence in items if evidence.child_id == child_id]
        return sorted(items, key=lambda evidence: evidence.created_at)

    def enqueue_tutor_item(
        self,
        *,
        submission_id: str,
        item_id: str,
        child_id: str,
        question_type_id: str,
        priority: int = 0,
    ) -> TutorQueueItem:
        self.require_submission(submission_id)
        queue_item = TutorQueueItem(
            submission_id=submission_id,
            item_id=item_id,
            child_id=child_id,
            family_id=self.require_submission(submission_id).family_id or self.get_family_id_for_child(child_id),
            question_type_id=question_type_id,
            priority=priority,
        )
        self.tutor_queue_items.setdefault(submission_id, []).append(queue_item)
        if not self.get_active_tutor_item(submission_id):
            queue_item = self.mark_tutor_item_active(queue_item.queue_item_id)
        self.update_submission(submission_id, status=LearningSubmissionStatus.TUTORING)
        return queue_item

    def list_tutor_queue_items(self, submission_id: str) -> list[TutorQueueItem]:
        self.require_submission(submission_id)
        return sorted(
            list(self.tutor_queue_items.get(submission_id, [])),
            key=lambda item: (item.status != TutorQueueStatus.ACTIVE, -item.priority, item.created_at),
        )

    def get_active_tutor_item(self, submission_id: str) -> TutorQueueItem | None:
        for item in self.tutor_queue_items.get(submission_id, []):
            if item.status in {TutorQueueStatus.ACTIVE, TutorQueueStatus.PENDING}:
                return item
        return None

    def mark_tutor_item_active(
        self,
        queue_item_id: str,
        *,
        tutor_session_id: str | None = None,
    ) -> TutorQueueItem:
        updates: dict[str, Any] = {"status": TutorQueueStatus.ACTIVE}
        if tutor_session_id is not None:
            updates["tutor_session_id"] = tutor_session_id
        return self._update_tutor_queue_item(queue_item_id, **updates)

    def mark_tutor_item_completed(
        self,
        queue_item_id: str,
        *,
        tutor_session_id: str | None = None,
    ) -> TutorQueueItem:
        return self._update_tutor_queue_item(
            queue_item_id,
            status=TutorQueueStatus.COMPLETED,
            tutor_session_id=tutor_session_id,
            completed_at=utc_now(),
        )

    def complete_submission_if_queue_done(self, submission_id: str) -> LearningSubmission:
        submission = self.require_submission(submission_id)
        queue = self.tutor_queue_items.get(submission_id, [])
        has_open_queue = any(item.status in {TutorQueueStatus.PENDING, TutorQueueStatus.ACTIVE} for item in queue)
        if has_open_queue:
            return self.update_submission(submission_id, status=LearningSubmissionStatus.TUTORING)
        if submission.needs_manual_confirm_count:
            return self.update_submission(submission_id, status=LearningSubmissionStatus.NEEDS_MANUAL_CONFIRM)
        return self.update_submission(
            submission_id,
            status=LearningSubmissionStatus.COMPLETED,
            completed_at=utc_now(),
        )

    def get_submission_snapshot(self, submission_id: str) -> LearningSubmissionSnapshot:
        submission = self.require_submission(submission_id)
        items = self.list_submission_items(submission_id)
        item_ids = {item.item_id for item in items}
        return LearningSubmissionSnapshot(
            submission=submission,
            items=items,
            mastery_evidence=[
                evidence
                for values in self.mastery_evidence.values()
                for evidence in values
                if evidence.item_id in item_ids
            ],
            tutor_queue=list(self.tutor_queue_items.get(submission_id, [])),
        )

    def _refresh_submission_counts(self, submission_id: str) -> None:
        submission = self.require_submission(submission_id)
        items = self.submission_items.get(submission_id, [])
        updated = submission.model_copy(
            update={
                "item_count": len(items),
                "correct_count": sum(1 for item in items if item.judge_result == JudgeResult.CORRECT),
                "wrong_count": sum(1 for item in items if item.judge_result == JudgeResult.WRONG),
                "needs_manual_confirm_count": sum(
                    1 for item in items if item.judge_result == JudgeResult.NEEDS_MANUAL_CONFIRM
                ),
                "updated_at": utc_now(),
            }
        )
        self.submissions[submission_id] = updated

    def _update_tutor_queue_item(self, queue_item_id: str, **updates: Any) -> TutorQueueItem:
        for submission_id, queue in self.tutor_queue_items.items():
            for index, item in enumerate(queue):
                if item.queue_item_id == queue_item_id:
                    updated = item.model_copy(update={**updates, "updated_at": utc_now()})
                    queue[index] = updated
                    active_id = updated.queue_item_id if updated.status == TutorQueueStatus.ACTIVE else None
                    self.update_submission(submission_id, active_queue_item_id=active_id)
                    return updated
        raise KeyError(queue_item_id)

    def _family_id_for_item(self, item_id: str) -> str:
        for items in self.submission_items.values():
            for item in items:
                if item.item_id == item_id:
                    return item.family_id
        return ""

    def save_reminder_subscription(
        self,
        *,
        openid: str,
        child_id: str,
        template_id: str,
        enabled: bool = True,
        scope: str = "weekly",
    ) -> ReminderSubscription:
        key = f"{openid}:{child_id}:{template_id}:{scope}"
        existing = self.reminder_subscriptions.get(key)
        subscription = ReminderSubscription(
            subscription_id=existing.subscription_id if existing else f"sub_{len(self.reminder_subscriptions) + 1}",
            openid=openid,
            child_id=child_id,
            template_id=template_id,
            enabled=enabled,
            scope=scope,
            created_at=existing.created_at if existing else utc_now(),
        )
        self.reminder_subscriptions[key] = subscription
        return subscription

    def list_reminder_subscriptions(self, child_id: str | None = None) -> list[ReminderSubscription]:
        items = list(self.reminder_subscriptions.values())
        if child_id:
            items = [item for item in items if item.child_id == child_id]
        return [item for item in items if item.enabled]


class SQLiteLearningStore:
    """SQLite-backed learning store for local and single-server deployments."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or (songguo_data_root() / "learning.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @property
    def sessions(self) -> dict[str, LearningSession]:
        with self._connect() as conn:
            rows = conn.execute("SELECT data_json FROM learning_sessions").fetchall()
        sessions = [LearningSession.model_validate_json(row["data_json"]) for row in rows]
        return {session.session_id: session for session in sessions}

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS learning_sessions (
                    session_id TEXT PRIMARY KEY,
                    child_id TEXT NOT NULL,
                    family_id TEXT NOT NULL DEFAULT '',
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_learning_sessions_child
                    ON learning_sessions(child_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS learning_events (
                    event_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    child_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_learning_events_session_created
                    ON learning_events(session_id, created_at, event_id);

                CREATE INDEX IF NOT EXISTS idx_learning_events_child_type
                    ON learning_events(child_id, event_type, created_at);

                CREATE TABLE IF NOT EXISTS learning_messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    child_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_learning_messages_session_created
                    ON learning_messages(session_id, created_at, message_id);

                CREATE TABLE IF NOT EXISTS ai_call_logs (
                    request_id TEXT PRIMARY KEY,
                    child_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_ai_call_logs_child_created
                    ON ai_call_logs(child_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS wrong_questions (
                    question_id TEXT PRIMARY KEY,
                    child_id TEXT NOT NULL,
                    family_id TEXT NOT NULL DEFAULT '',
                    session_id TEXT NOT NULL,
                    knowledge_point TEXT NOT NULL DEFAULT '',
                    last_misconception TEXT,
                    highest_hint_level INTEGER NOT NULL DEFAULT 1,
                    resolved INTEGER NOT NULL DEFAULT 0,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_wrong_questions_child_updated
                    ON wrong_questions(child_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS learning_deposits (
                    session_id TEXT PRIMARY KEY,
                    child_id TEXT NOT NULL,
                    knowledge_point TEXT NOT NULL DEFAULT '',
                    question_type TEXT NOT NULL DEFAULT '',
                    main_misconception TEXT,
                    need_review INTEGER NOT NULL DEFAULT 0,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_learning_deposits_child
                    ON learning_deposits(child_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS learning_summaries (
                    summary_id TEXT PRIMARY KEY,
                    child_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_learning_summaries_child_scope
                    ON learning_summaries(child_id, scope, created_at DESC);

                CREATE TABLE IF NOT EXISTS photo_reviews (
                    review_id TEXT PRIMARY KEY,
                    child_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_photo_reviews_child_updated
                    ON photo_reviews(child_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS children (
                    child_id TEXT PRIMARY KEY,
                    family_id TEXT NOT NULL DEFAULT '',
                    name TEXT NOT NULL,
                    grade INTEGER NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS child_bindings (
                    openid TEXT NOT NULL,
                    child_id TEXT NOT NULL,
                    family_id TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (openid, child_id)
                );

                CREATE INDEX IF NOT EXISTS idx_child_bindings_openid
                    ON child_bindings(openid, child_id);

                CREATE TABLE IF NOT EXISTS safety_events (
                    safety_event_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    child_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_safety_events_child_created
                    ON safety_events(child_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS reminder_subscriptions (
                    subscription_id TEXT PRIMARY KEY,
                    subscription_key TEXT UNIQUE NOT NULL,
                    child_id TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    enabled INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_reminder_subscriptions_child
                    ON reminder_subscriptions(child_id, enabled);

                CREATE TABLE IF NOT EXISTS learning_submissions (
                    submission_id TEXT PRIMARY KEY,
                    child_id TEXT NOT NULL,
                    family_id TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_learning_submissions_child_updated
                    ON learning_submissions(child_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS learning_submission_items (
                    item_id TEXT PRIMARY KEY,
                    submission_id TEXT NOT NULL,
                    child_id TEXT NOT NULL,
                    family_id TEXT NOT NULL DEFAULT '',
                    item_index INTEGER NOT NULL,
                    judge_result TEXT NOT NULL,
                    question_type_id TEXT NOT NULL DEFAULT '',
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_learning_submission_items_submission
                    ON learning_submission_items(submission_id, item_index);

                CREATE TABLE IF NOT EXISTS mastery_evidence (
                    evidence_id TEXT PRIMARY KEY,
                    item_id TEXT NOT NULL,
                    child_id TEXT NOT NULL,
                    family_id TEXT NOT NULL DEFAULT '',
                    question_type_id TEXT NOT NULL,
                    evidence_type TEXT NOT NULL,
                    review_due INTEGER NOT NULL DEFAULT 0,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_mastery_evidence_child_type
                    ON mastery_evidence(child_id, question_type_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS tutor_queue_items (
                    queue_item_id TEXT PRIMARY KEY,
                    submission_id TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    child_id TEXT NOT NULL,
                    family_id TEXT NOT NULL DEFAULT '',
                    question_type_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 0,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_tutor_queue_items_submission_status
                    ON tutor_queue_items(submission_id, status, priority DESC, created_at);
                """
            )
            for statement in _sqlite_schema_migrations():
                try:
                    conn.execute(statement)
                except sqlite3.OperationalError as exc:
                    if "duplicate column name" not in str(exc).lower():
                        raise
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def create_session(
        self,
        *,
        child_id: str,
        subject: str,
        grade: int,
        question_text: str,
        normalized_question: str,
        knowledge_point: str,
        current_prompt: str,
        runner_mode: str = "kernel",
        problem_analysis: dict[str, Any] | None = None,
        current_key_point_id: str | None = None,
        released_key_point_ids: list[str] | None = None,
        mastered_key_point_ids: list[str] | None = None,
    ) -> LearningSession:
        session = LearningSession(
            child_id=child_id,
            family_id=self.get_family_id_for_child(child_id),
            subject=subject,
            grade=grade,
            question_text=question_text,
            normalized_question=normalized_question,
            knowledge_point=knowledge_point,
            current_prompt=current_prompt,
            runner_mode=runner_mode,
            problem_analysis=problem_analysis,
            current_key_point_id=current_key_point_id,
            released_key_point_ids=released_key_point_ids or [],
            mastered_key_point_ids=mastered_key_point_ids or [],
        )
        self._save_session(session)
        self.append_event(
            session_id=session.session_id,
            child_id=child_id,
            event_type="session.created",
            payload={
                "question_text": question_text,
                "normalized_question": normalized_question,
                "knowledge_point": knowledge_point,
                "hint_level": session.hint_level,
                "answer_unlocked": session.answer_unlocked,
                "runner_mode": session.runner_mode,
                "current_key_point_id": session.current_key_point_id,
            },
        )
        return session

    def get_session(self, session_id: str) -> LearningSession | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data_json FROM learning_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return LearningSession.model_validate_json(row["data_json"])

    def require_session(self, session_id: str) -> LearningSession:
        session = self.get_session(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def update_session(self, session_id: str, **updates: Any) -> LearningSession:
        session = self.require_session(session_id)
        updated = session.model_copy(update={**updates, "updated_at": utc_now()})
        self._save_session(updated)
        return updated

    def _save_session(self, session: LearningSession) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO learning_sessions
                    (session_id, child_id, family_id, data_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    child_id = excluded.child_id,
                    family_id = excluded.family_id,
                    data_json = excluded.data_json,
                    updated_at = excluded.updated_at
                """,
                (
                    session.session_id,
                    session.child_id,
                    session.family_id,
                    session.model_dump_json(),
                    session.created_at.isoformat(),
                    session.updated_at.isoformat(),
                ),
            )
            conn.commit()

    def append_event(
        self,
        *,
        session_id: str,
        child_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        deeptutor_trace_id: str | None = None,
        leakage_check_result: dict[str, Any] | None = None,
    ) -> LearningEvent:
        event = LearningEvent(
            session_id=session_id,
            child_id=child_id,
            event_type=event_type,
            payload=payload or {},
            deeptutor_trace_id=deeptutor_trace_id,
            leakage_check_result=leakage_check_result,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO learning_events
                    (event_id, session_id, child_id, event_type, data_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.session_id,
                    event.child_id,
                    event.event_type,
                    event.model_dump_json(),
                    event.created_at.isoformat(),
                ),
            )
            conn.commit()
        return event

    def list_events(self, session_id: str) -> list[LearningEvent]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT data_json
                FROM learning_events
                WHERE session_id = ?
                ORDER BY created_at, event_id
                """,
                (session_id,),
            ).fetchall()
        return [LearningEvent.model_validate_json(row["data_json"]) for row in rows]

    def append_message(
        self,
        *,
        session_id: str,
        child_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> LearningMessage:
        message = LearningMessage(
            session_id=session_id,
            child_id=child_id,
            role=role,
            content=content,
            metadata=metadata or {},
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO learning_messages
                    (message_id, session_id, child_id, role, data_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    message.message_id,
                    message.session_id,
                    message.child_id,
                    message.role,
                    message.model_dump_json(),
                    message.created_at.isoformat(),
                ),
            )
            conn.commit()
        return message

    def list_messages(self, session_id: str) -> list[LearningMessage]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT data_json
                FROM learning_messages
                WHERE session_id = ?
                ORDER BY created_at, message_id
                """,
                (session_id,),
            ).fetchall()
        return [LearningMessage.model_validate_json(row["data_json"]) for row in rows]

    def list_sessions(self, child_id: str | None = None) -> list[LearningSession]:
        query = "SELECT data_json FROM learning_sessions"
        params: tuple[Any, ...] = ()
        if child_id:
            query += " WHERE child_id = ?"
            params = (child_id,)
        query += " ORDER BY updated_at DESC, session_id"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [LearningSession.model_validate_json(row["data_json"]) for row in rows]

    def get_resume_snapshot(self, session_id: str) -> ResumeSnapshot:
        session = self.require_session(session_id)
        stored_messages = self.list_messages(session_id)
        last_messages = (
            _messages_for_snapshot(stored_messages)
            if stored_messages
            else _build_resume_messages(
                self.list_events(session_id),
                current_prompt=session.current_prompt,
                hint_level=session.hint_level,
            )
        )
        return ResumeSnapshot(
            session_id=session.session_id,
            question_text=session.question_text,
            subject=session.subject,
            grade=session.grade,
            phase=session.phase,
            hint_level=session.hint_level,
            attempt_count=session.attempt_count,
            answer_unlocked=session.answer_unlocked,
            current_prompt=session.current_prompt,
            teaching_progress=build_teaching_progress(session),
            last_misconception=session.last_misconception,
            last_messages=last_messages,
        )

    def record_wrong_question(
        self,
        *,
        session_id: str,
        child_id: str,
        normalized_question: str,
        knowledge_point: str,
        mistake_summary: str,
        last_misconception: str | None,
        highest_hint_level: int,
        explanation_unlocked: bool = False,
        practice_completed: bool = False,
    ) -> WrongQuestion:
        wrong_question = WrongQuestion(
            child_id=child_id,
            family_id=self.get_family_id_for_child(child_id),
            session_id=session_id,
            normalized_question=normalized_question,
            knowledge_point=knowledge_point,
            mistake_summary=mistake_summary,
            last_misconception=last_misconception,
            highest_hint_level=highest_hint_level,
            explanation_unlocked=explanation_unlocked,
            practice_completed=practice_completed,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO wrong_questions
                    (
                        question_id, child_id, family_id, session_id, knowledge_point,
                        last_misconception, highest_hint_level, resolved,
                        practice_completed, data_json, created_at, updated_at
                    )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    wrong_question.question_id,
                    wrong_question.child_id,
                    wrong_question.family_id,
                    wrong_question.session_id,
                    wrong_question.knowledge_point,
                    wrong_question.last_misconception,
                    wrong_question.highest_hint_level,
                    1 if wrong_question.resolved else 0,
                    1 if wrong_question.practice_completed else 0,
                    wrong_question.model_dump_json(),
                    wrong_question.created_at.isoformat(),
                    wrong_question.updated_at.isoformat(),
                ),
            )
            conn.commit()
        self.append_event(
            session_id=session_id,
            child_id=child_id,
            event_type="wrong_question.recorded",
            payload=wrong_question.model_dump(mode="json"),
        )
        return wrong_question

    def list_wrong_questions(self, child_id: str) -> list[WrongQuestion]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT data_json
                FROM wrong_questions
                WHERE child_id = ?
                ORDER BY updated_at DESC, question_id
                """,
                (child_id,),
            ).fetchall()
        return [WrongQuestion.model_validate_json(row["data_json"]) for row in rows]

    def update_wrong_question_practice_result(
        self,
        *,
        question_id: str,
        child_id: str,
        correct: bool,
    ) -> WrongQuestion:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT data_json
                FROM wrong_questions
                WHERE question_id = ? AND child_id = ?
                """,
                (question_id, child_id),
            ).fetchone()
        if row is None:
            raise KeyError(question_id)

        question = WrongQuestion.model_validate_json(row["data_json"])
        updated = question.model_copy(
            update={
                "practice_completed": True,
                "resolved": bool(correct),
                "updated_at": utc_now(),
            }
        )
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE wrong_questions
                SET data_json = ?, updated_at = ?, resolved = ?, practice_completed = ?
                WHERE question_id = ? AND child_id = ?
                """,
                (
                    updated.model_dump_json(),
                    updated.updated_at.isoformat(),
                    1 if updated.resolved else 0,
                    1 if updated.practice_completed else 0,
                    question_id,
                    child_id,
                ),
            )
            conn.commit()
        self.append_event(
            session_id=updated.session_id,
            child_id=child_id,
            event_type="practice.completed",
            payload={
                "question_id": updated.question_id,
                "correct": bool(correct),
                "resolved": updated.resolved,
            },
        )
        return updated

    def _family_id_for_item(self, item_id: str) -> str:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data_json FROM learning_submission_items WHERE item_id = ?",
                (item_id,),
            ).fetchone()
        if row is None:
            return ""
        return LearningItem.model_validate_json(row["data_json"]).family_id

    def save_photo_review(self, review: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO photo_reviews
                    (review_id, child_id, status, data_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(review_id) DO UPDATE SET
                    child_id = excluded.child_id,
                    status = excluded.status,
                    data_json = excluded.data_json,
                    updated_at = excluded.updated_at
                """,
                (
                    review["review_id"],
                    review["child_id"],
                    review["status"],
                    _json_dumps(review),
                    review["created_at"],
                    review["updated_at"],
                ),
            )
            conn.commit()

    def get_photo_review(self, review_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data_json FROM photo_reviews WHERE review_id = ?",
                (review_id,),
            ).fetchone()
        if row is None:
            return None
        return _json_loads(row["data_json"], None)

    def upsert_child(
        self,
        *,
        child_id: str,
        name: str,
        grade: int,
        term_label: str | None = None,
        family_id: str | None = None,
    ) -> ChildProfile:
        existing = self.get_child(child_id)
        resolved_family_id = (
            family_id
            or (existing.family_id if existing else "")
            or self.get_family_id_for_child(child_id)
        )
        profile = ChildProfile(
            child_id=child_id,
            family_id=resolved_family_id,
            name=name,
            grade=grade,
            term_label=term_label,
            created_at=existing.created_at if existing else utc_now(),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO children
                    (child_id, family_id, name, grade, data_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(child_id) DO UPDATE SET
                    family_id = excluded.family_id,
                    name = excluded.name,
                    grade = excluded.grade,
                    data_json = excluded.data_json,
                    updated_at = excluded.updated_at
                """,
                (
                    profile.child_id,
                    profile.family_id,
                    profile.name,
                    profile.grade,
                    profile.model_dump_json(),
                    profile.created_at.isoformat(),
                    profile.updated_at.isoformat(),
                ),
            )
            conn.commit()
        return profile

    def get_child(self, child_id: str) -> ChildProfile | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data_json FROM children WHERE child_id = ?",
                (child_id,),
            ).fetchone()
        if row is None:
            return None
        return ChildProfile.model_validate_json(row["data_json"])

    def list_children(self) -> list[ChildProfile]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT data_json FROM children ORDER BY created_at, child_id",
            ).fetchall()
        return [ChildProfile.model_validate_json(row["data_json"]) for row in rows]

    def bind_child_to_openid(self, *, openid: str, child_id: str) -> None:
        family_id = self.get_family_id_for_child(child_id) or self.get_family_id_for_openid(openid)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO child_bindings (openid, child_id, family_id, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (openid, child_id, family_id, utc_now().isoformat()),
            )
            conn.execute(
                """
                UPDATE child_bindings
                SET family_id = ?
                WHERE openid = ? AND child_id = ? AND family_id = ''
                """,
                (family_id, openid, child_id),
            )
            conn.commit()
        child = self.get_child(child_id)
        if child is not None and not child.family_id:
            self.upsert_child(
                child_id=child.child_id,
                name=child.name,
                grade=child.grade,
                term_label=child.term_label,
                family_id=family_id,
            )

    def is_child_bound_to_openid(self, *, openid: str, child_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM child_bindings
                WHERE openid = ? AND child_id = ?
                """,
                (openid, child_id),
            ).fetchone()
        return row is not None

    def list_children_for_openid(self, openid: str) -> list[ChildProfile]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.data_json
                FROM children c
                INNER JOIN child_bindings b ON b.child_id = c.child_id
                WHERE b.openid = ?
                ORDER BY c.created_at, c.child_id
                """,
                (openid,),
            ).fetchall()
        return [ChildProfile.model_validate_json(row["data_json"]) for row in rows]

    def get_family_id_for_openid(self, openid: str) -> str:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT family_id
                FROM child_bindings
                WHERE openid = ? AND family_id != ''
                ORDER BY created_at, child_id
                LIMIT 1
                """,
                (openid,),
            ).fetchone()
        if row is not None and row["family_id"]:
            return str(row["family_id"])
        return f"family_{openid}"

    def get_family_id_for_child(self, child_id: str) -> str:
        child = self.get_child(child_id)
        if child is not None and child.family_id:
            return child.family_id
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT family_id, openid
                FROM child_bindings
                WHERE child_id = ?
                ORDER BY created_at, openid
                LIMIT 1
                """,
                (child_id,),
            ).fetchone()
        if row is None:
            return ""
        if row["family_id"]:
            return str(row["family_id"])
        return self.get_family_id_for_openid(str(row["openid"]))

    def record_ai_call(
        self,
        *,
        child_id: str,
        session_id: str,
        provider: str,
        model: str,
        operation: str,
        token_estimate: int,
        status: str,
    ) -> AICallLog:
        log = AICallLog(
            child_id=child_id,
            session_id=session_id,
            provider=provider,
            model=model,
            operation=operation,
            token_estimate=max(0, int(token_estimate)),
            status=status,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ai_call_logs
                    (request_id, child_id, session_id, operation, data_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    log.request_id,
                    log.child_id,
                    log.session_id,
                    log.operation,
                    log.model_dump_json(),
                    log.created_at.isoformat(),
                ),
            )
            conn.commit()
        return log

    def list_ai_call_logs(self, child_id: str | None = None) -> list[AICallLog]:
        query = "SELECT data_json FROM ai_call_logs"
        params: tuple[Any, ...] = ()
        if child_id:
            query += " WHERE child_id = ?"
            params = (child_id,)
        query += " ORDER BY created_at, request_id"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [AICallLog.model_validate_json(row["data_json"]) for row in rows]

    def save_learning_deposit(self, session_id: str, deposit: dict[str, Any]) -> dict[str, Any]:
        child_id = str(deposit.get("child_id") or "")
        question_record = deposit.get("question_record") or {}
        mistake_record = deposit.get("mistake_record") or {}
        knowledge_point = str(question_record.get("knowledge_point") or "")
        question_type = str(question_record.get("question_type") or "")
        main_misconception = mistake_record.get("main_error_reason")
        need_review = bool(mistake_record.get("need_review")) if mistake_record else False
        now = utc_now().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO learning_deposits
                    (
                        session_id, child_id, knowledge_point, question_type,
                        main_misconception, need_review, data_json, created_at, updated_at
                    )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    child_id = excluded.child_id,
                    knowledge_point = excluded.knowledge_point,
                    question_type = excluded.question_type,
                    main_misconception = excluded.main_misconception,
                    need_review = excluded.need_review,
                    data_json = excluded.data_json,
                    updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    child_id,
                    knowledge_point,
                    question_type,
                    main_misconception,
                    1 if need_review else 0,
                    _json_dumps(deposit),
                    now,
                    now,
                ),
            )
            conn.commit()
        return dict(deposit)

    def get_learning_deposit(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data_json FROM learning_deposits WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return _json_loads(row["data_json"], None)

    def save_learning_summary(self, summary: LearningSummary) -> LearningSummary:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO learning_summaries
                    (summary_id, child_id, scope, data_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    summary.summary_id,
                    summary.child_id,
                    summary.scope,
                    summary.model_dump_json(),
                    summary.created_at.isoformat(),
                ),
            )
            conn.commit()
        return summary

    def list_learning_summaries(
        self,
        *,
        child_id: str,
        scope: str | None = None,
    ) -> list[LearningSummary]:
        query = "SELECT data_json FROM learning_summaries WHERE child_id = ?"
        params: tuple[Any, ...] = (child_id,)
        if scope:
            query += " AND scope = ?"
            params = (child_id, scope)
        query += " ORDER BY created_at DESC, summary_id"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [LearningSummary.model_validate_json(row["data_json"]) for row in rows]

    def record_safety_event(
        self,
        *,
        session_id: str,
        child_id: str,
        event_type: str,
        input_text: str | None = None,
        blocked_text: str | None = None,
        reason: str,
    ) -> SafetyEvent:
        event = SafetyEvent(
            session_id=session_id,
            child_id=child_id,
            event_type=event_type,
            input_text=input_text,
            blocked_text=blocked_text,
            reason=reason,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO safety_events
                    (safety_event_id, session_id, child_id, event_type, data_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event.safety_event_id,
                    event.session_id,
                    event.child_id,
                    event.event_type,
                    event.model_dump_json(),
                    event.created_at.isoformat(),
                ),
            )
            conn.commit()
        return event

    def list_safety_events(self, child_id: str | None = None) -> list[SafetyEvent]:
        query = "SELECT data_json FROM safety_events"
        params: tuple[Any, ...] = ()
        if child_id:
            query += " WHERE child_id = ?"
            params = (child_id,)
        query += " ORDER BY created_at DESC, safety_event_id"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [SafetyEvent.model_validate_json(row["data_json"]) for row in rows]

    def create_submission(
        self,
        *,
        child_id: str,
        subject: str,
        grade: int,
        source_type: SourceType | str,
        raw_text: str = "",
        image_refs: list[str] | None = None,
        audio_refs: list[str] | None = None,
        data_json: dict[str, Any] | None = None,
    ) -> LearningSubmission:
        submission = LearningSubmission(
            child_id=child_id,
            family_id=self.get_family_id_for_child(child_id),
            subject=subject,
            grade=grade,
            source_type=source_type,
            raw_text=raw_text,
            image_refs=image_refs or [],
            audio_refs=audio_refs or [],
            data_json=data_json or {},
        )
        self._save_submission(submission)
        return submission

    def get_submission(self, submission_id: str) -> LearningSubmission | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data_json FROM learning_submissions WHERE submission_id = ?",
                (submission_id,),
            ).fetchone()
        return LearningSubmission.model_validate_json(row["data_json"]) if row else None

    def require_submission(self, submission_id: str) -> LearningSubmission:
        submission = self.get_submission(submission_id)
        if submission is None:
            raise KeyError(submission_id)
        return submission

    def update_submission(self, submission_id: str, **updates: Any) -> LearningSubmission:
        submission = self.require_submission(submission_id)
        updated = submission.model_copy(update={**updates, "updated_at": utc_now()})
        self._save_submission(updated)
        return updated

    def list_submissions(self, child_id: str | None = None) -> list[LearningSubmission]:
        query = "SELECT data_json FROM learning_submissions"
        params: tuple[Any, ...] = ()
        if child_id:
            query += " WHERE child_id = ?"
            params = (child_id,)
        query += " ORDER BY updated_at DESC, submission_id"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [LearningSubmission.model_validate_json(row["data_json"]) for row in rows]

    def add_submission_item(self, item: LearningItem) -> LearningItem:
        submission = self.require_submission(item.submission_id)
        stored = item.model_copy(
            update={
                "family_id": item.family_id or submission.family_id or self.get_family_id_for_child(item.child_id)
            }
        )
        self._save_submission_item(stored)
        self._refresh_submission_counts(stored.submission_id)
        return stored

    def update_submission_item(self, item_id: str, **updates: Any) -> LearningItem:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data_json FROM learning_submission_items WHERE item_id = ?",
                (item_id,),
            ).fetchone()
        if row is None:
            raise KeyError(item_id)
        item = LearningItem.model_validate_json(row["data_json"])
        updated = item.model_copy(update={**updates, "updated_at": utc_now()})
        self._save_submission_item(updated)
        self._refresh_submission_counts(updated.submission_id)
        return updated

    def list_submission_items(self, submission_id: str) -> list[LearningItem]:
        self.require_submission(submission_id)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT data_json FROM learning_submission_items
                WHERE submission_id = ?
                ORDER BY item_index, item_id
                """,
                (submission_id,),
            ).fetchall()
        return [LearningItem.model_validate_json(row["data_json"]) for row in rows]

    def save_mastery_evidence(
        self,
        *,
        item_id: str,
        child_id: str,
        question_type_id: str,
        evidence_type: EvidenceType | str,
        is_correct: bool,
        mastery_state_after: MasteryState | str,
        review_due: bool = False,
        solved_without_help: bool = False,
        misconception_tag: str | None = None,
        confidence: float = 0.0,
        data_json: dict[str, Any] | None = None,
    ) -> MasteryEvidence:
        evidence = MasteryEvidence(
            item_id=item_id,
            child_id=child_id,
            family_id=self.get_family_id_for_child(child_id) or self._family_id_for_item(item_id),
            question_type_id=question_type_id,
            evidence_type=evidence_type,
            is_correct=is_correct,
            mastery_state_after=mastery_state_after,
            review_due=review_due,
            solved_without_help=solved_without_help,
            misconception_tag=misconception_tag,
            confidence=confidence,
            data_json=data_json or {},
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO mastery_evidence
                    (evidence_id, item_id, child_id, family_id, question_type_id, evidence_type, review_due, data_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evidence.evidence_id,
                    evidence.item_id,
                    evidence.child_id,
                    evidence.family_id,
                    evidence.question_type_id,
                    evidence.evidence_type,
                    1 if evidence.review_due else 0,
                    evidence.model_dump_json(),
                    evidence.created_at.isoformat(),
                ),
            )
            conn.commit()
        return evidence

    def list_mastery_evidence(self, item_id: str | None = None, child_id: str | None = None) -> list[MasteryEvidence]:
        query = "SELECT data_json FROM mastery_evidence"
        filters: list[str] = []
        params: list[Any] = []
        if item_id:
            filters.append("item_id = ?")
            params.append(item_id)
        if child_id:
            filters.append("child_id = ?")
            params.append(child_id)
        if filters:
            query += " WHERE " + " AND ".join(filters)
        query += " ORDER BY created_at, evidence_id"
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [MasteryEvidence.model_validate_json(row["data_json"]) for row in rows]

    def enqueue_tutor_item(
        self,
        *,
        submission_id: str,
        item_id: str,
        child_id: str,
        question_type_id: str,
        priority: int = 0,
    ) -> TutorQueueItem:
        submission = self.require_submission(submission_id)
        queue_item = TutorQueueItem(
            submission_id=submission_id,
            item_id=item_id,
            child_id=child_id,
            family_id=submission.family_id or self.get_family_id_for_child(child_id),
            question_type_id=question_type_id,
            priority=priority,
        )
        self._save_tutor_queue_item(queue_item)
        if not self.get_active_tutor_item(submission_id):
            queue_item = self.mark_tutor_item_active(queue_item.queue_item_id)
        self.update_submission(submission_id, status=LearningSubmissionStatus.TUTORING)
        return queue_item

    def list_tutor_queue_items(self, submission_id: str) -> list[TutorQueueItem]:
        self.require_submission(submission_id)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT data_json FROM tutor_queue_items
                WHERE submission_id = ?
                ORDER BY CASE WHEN status = 'active' THEN 0 ELSE 1 END, priority DESC, created_at, queue_item_id
                """,
                (submission_id,),
            ).fetchall()
        return [TutorQueueItem.model_validate_json(row["data_json"]) for row in rows]

    def get_active_tutor_item(self, submission_id: str) -> TutorQueueItem | None:
        for item in self.list_tutor_queue_items(submission_id):
            if item.status in {TutorQueueStatus.ACTIVE, TutorQueueStatus.PENDING}:
                return item
        return None

    def mark_tutor_item_active(
        self,
        queue_item_id: str,
        *,
        tutor_session_id: str | None = None,
    ) -> TutorQueueItem:
        updates: dict[str, Any] = {"status": TutorQueueStatus.ACTIVE}
        if tutor_session_id is not None:
            updates["tutor_session_id"] = tutor_session_id
        return self._update_tutor_queue_item(queue_item_id, **updates)

    def mark_tutor_item_completed(
        self,
        queue_item_id: str,
        *,
        tutor_session_id: str | None = None,
    ) -> TutorQueueItem:
        return self._update_tutor_queue_item(
            queue_item_id,
            status=TutorQueueStatus.COMPLETED,
            tutor_session_id=tutor_session_id,
            completed_at=utc_now(),
        )

    def complete_submission_if_queue_done(self, submission_id: str) -> LearningSubmission:
        submission = self.require_submission(submission_id)
        has_open_queue = any(
            item.status in {TutorQueueStatus.PENDING, TutorQueueStatus.ACTIVE}
            for item in self.list_tutor_queue_items(submission_id)
        )
        if has_open_queue:
            return self.update_submission(submission_id, status=LearningSubmissionStatus.TUTORING)
        if submission.needs_manual_confirm_count:
            return self.update_submission(submission_id, status=LearningSubmissionStatus.NEEDS_MANUAL_CONFIRM)
        return self.update_submission(submission_id, status=LearningSubmissionStatus.COMPLETED, completed_at=utc_now())

    def get_submission_snapshot(self, submission_id: str) -> LearningSubmissionSnapshot:
        submission = self.require_submission(submission_id)
        items = self.list_submission_items(submission_id)
        evidence: list[MasteryEvidence] = []
        for item in items:
            evidence.extend(self.list_mastery_evidence(item_id=item.item_id))
        return LearningSubmissionSnapshot(
            submission=submission,
            items=items,
            mastery_evidence=evidence,
            tutor_queue=self.list_tutor_queue_items(submission_id),
        )

    def _save_submission(self, submission: LearningSubmission) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO learning_submissions
                    (submission_id, child_id, family_id, status, source_type, data_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(submission_id) DO UPDATE SET
                    child_id = excluded.child_id,
                    family_id = excluded.family_id,
                    status = excluded.status,
                    source_type = excluded.source_type,
                    data_json = excluded.data_json,
                    updated_at = excluded.updated_at
                """,
                (
                    submission.submission_id,
                    submission.child_id,
                    submission.family_id,
                    submission.status,
                    submission.source_type,
                    submission.model_dump_json(),
                    submission.created_at.isoformat(),
                    submission.updated_at.isoformat(),
                ),
            )
            conn.commit()

    def _save_submission_item(self, item: LearningItem) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO learning_submission_items
                    (
                        item_id, submission_id, child_id, family_id, item_index, judge_result,
                        question_type_id, data_json, created_at, updated_at
                    )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(item_id) DO UPDATE SET
                    submission_id = excluded.submission_id,
                    child_id = excluded.child_id,
                    family_id = excluded.family_id,
                    item_index = excluded.item_index,
                    judge_result = excluded.judge_result,
                    question_type_id = excluded.question_type_id,
                    data_json = excluded.data_json,
                    updated_at = excluded.updated_at
                """,
                (
                    item.item_id,
                    item.submission_id,
                    item.child_id,
                    item.family_id,
                    item.item_index,
                    item.judge_result,
                    item.question_type_id,
                    item.model_dump_json(),
                    item.created_at.isoformat(),
                    item.updated_at.isoformat(),
                ),
            )
            conn.commit()

    def _save_tutor_queue_item(self, item: TutorQueueItem) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tutor_queue_items
                    (
                        queue_item_id, submission_id, item_id, child_id, family_id, question_type_id,
                        status, priority, data_json, created_at, updated_at
                    )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(queue_item_id) DO UPDATE SET
                    submission_id = excluded.submission_id,
                    item_id = excluded.item_id,
                    child_id = excluded.child_id,
                    family_id = excluded.family_id,
                    question_type_id = excluded.question_type_id,
                    status = excluded.status,
                    priority = excluded.priority,
                    data_json = excluded.data_json,
                    updated_at = excluded.updated_at
                """,
                (
                    item.queue_item_id,
                    item.submission_id,
                    item.item_id,
                    item.child_id,
                    item.family_id,
                    item.question_type_id,
                    item.status,
                    item.priority,
                    item.model_dump_json(),
                    item.created_at.isoformat(),
                    item.updated_at.isoformat(),
                ),
            )
            conn.commit()

    def _refresh_submission_counts(self, submission_id: str) -> None:
        submission = self.require_submission(submission_id)
        items = self.list_submission_items(submission_id)
        self._save_submission(
            submission.model_copy(
                update={
                    "item_count": len(items),
                    "correct_count": sum(1 for item in items if item.judge_result == JudgeResult.CORRECT),
                    "wrong_count": sum(1 for item in items if item.judge_result == JudgeResult.WRONG),
                    "needs_manual_confirm_count": sum(
                        1 for item in items if item.judge_result == JudgeResult.NEEDS_MANUAL_CONFIRM
                    ),
                    "updated_at": utc_now(),
                }
            )
        )

    def _update_tutor_queue_item(self, queue_item_id: str, **updates: Any) -> TutorQueueItem:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data_json FROM tutor_queue_items WHERE queue_item_id = ?",
                (queue_item_id,),
            ).fetchone()
        if row is None:
            raise KeyError(queue_item_id)
        item = TutorQueueItem.model_validate_json(row["data_json"])
        updated = item.model_copy(update={**updates, "updated_at": utc_now()})
        self._save_tutor_queue_item(updated)
        self.update_submission(
            updated.submission_id,
            active_queue_item_id=updated.queue_item_id if updated.status == TutorQueueStatus.ACTIVE else None,
        )
        return updated

    def save_reminder_subscription(
        self,
        *,
        openid: str,
        child_id: str,
        template_id: str,
        enabled: bool = True,
        scope: str = "weekly",
    ) -> ReminderSubscription:
        key = f"{openid}:{child_id}:{template_id}:{scope}"
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data_json FROM reminder_subscriptions WHERE subscription_key = ?",
                (key,),
            ).fetchone()
        existing = ReminderSubscription.model_validate_json(row["data_json"]) if row else None
        subscription_data = {
            "openid": openid,
            "child_id": child_id,
            "template_id": template_id,
            "enabled": enabled,
            "scope": scope,
            "created_at": existing.created_at if existing else utc_now(),
        }
        if existing:
            subscription_data["subscription_id"] = existing.subscription_id
        subscription = ReminderSubscription(**subscription_data)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO reminder_subscriptions
                    (subscription_id, subscription_key, child_id, data_json, enabled, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(subscription_key) DO UPDATE SET
                    child_id = excluded.child_id,
                    data_json = excluded.data_json,
                    enabled = excluded.enabled,
                    updated_at = excluded.updated_at
                """,
                (
                    subscription.subscription_id,
                    key,
                    child_id,
                    subscription.model_dump_json(),
                    1 if enabled else 0,
                    subscription.updated_at.isoformat(),
                ),
            )
            conn.commit()
        return subscription

    def list_reminder_subscriptions(self, child_id: str | None = None) -> list[ReminderSubscription]:
        query = "SELECT data_json FROM reminder_subscriptions WHERE enabled = 1"
        params: tuple[Any, ...] = ()
        if child_id:
            query += " AND child_id = ?"
            params = (child_id,)
        query += " ORDER BY updated_at DESC, subscription_id"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [ReminderSubscription.model_validate_json(row["data_json"]) for row in rows]


class PostgresLearningStore(SQLiteLearningStore):
    """PostgreSQL-backed learning store for real mini-program trials.

    The first PostgreSQL implementation intentionally preserves the existing
    store contract and JSON payload columns so SQLite/InMemory tests and the
    current product code can keep moving. Core fields are still indexed as
    normal columns; richer relational schema splits come next.
    """

    def __init__(self, database_url: str) -> None:
        if not database_url:
            raise ValueError("database_url is required")
        self.database_url = database_url
        self._initialize()

    def _connect(self):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError(
                "PostgreSQL storage requires psycopg. Install requirements-postgres.txt."
            ) from exc

        return _PostgresConnectionAdapter(
            psycopg.connect(self.database_url, row_factory=dict_row)
        )

    def _initialize(self) -> None:
        statements = [
            """
            CREATE TABLE IF NOT EXISTS learning_sessions (
                session_id TEXT PRIMARY KEY,
                child_id TEXT NOT NULL,
                family_id TEXT NOT NULL DEFAULT '',
                data_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_learning_sessions_child
                ON learning_sessions(child_id, updated_at DESC)
            """,
            """
            CREATE TABLE IF NOT EXISTS learning_events (
                event_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                child_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                data_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_learning_events_session_created
                ON learning_events(session_id, created_at, event_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_learning_events_child_type
                ON learning_events(child_id, event_type, created_at)
            """,
            """
            CREATE TABLE IF NOT EXISTS learning_messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                child_id TEXT NOT NULL,
                role TEXT NOT NULL,
                data_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_learning_messages_session_created
                ON learning_messages(session_id, created_at, message_id)
            """,
            """
            CREATE TABLE IF NOT EXISTS ai_call_logs (
                request_id TEXT PRIMARY KEY,
                child_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                operation TEXT NOT NULL,
                data_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_ai_call_logs_child_created
                ON ai_call_logs(child_id, created_at DESC)
            """,
            """
            CREATE TABLE IF NOT EXISTS wrong_questions (
                question_id TEXT PRIMARY KEY,
                child_id TEXT NOT NULL,
                family_id TEXT NOT NULL DEFAULT '',
                session_id TEXT NOT NULL,
                knowledge_point TEXT NOT NULL DEFAULT '',
                last_misconception TEXT,
                highest_hint_level INTEGER NOT NULL DEFAULT 1,
                resolved INTEGER NOT NULL DEFAULT 0,
                practice_completed INTEGER NOT NULL DEFAULT 0,
                data_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_wrong_questions_child_updated
                ON wrong_questions(child_id, updated_at DESC)
            """,
            """
            CREATE TABLE IF NOT EXISTS learning_deposits (
                session_id TEXT PRIMARY KEY,
                child_id TEXT NOT NULL,
                knowledge_point TEXT NOT NULL DEFAULT '',
                question_type TEXT NOT NULL DEFAULT '',
                main_misconception TEXT,
                need_review INTEGER NOT NULL DEFAULT 0,
                data_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_learning_deposits_child
                ON learning_deposits(child_id, updated_at DESC)
            """,
            """
            CREATE TABLE IF NOT EXISTS learning_summaries (
                summary_id TEXT PRIMARY KEY,
                child_id TEXT NOT NULL,
                scope TEXT NOT NULL,
                data_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_learning_summaries_child_scope
                ON learning_summaries(child_id, scope, created_at DESC)
            """,
            """
            CREATE TABLE IF NOT EXISTS photo_reviews (
                review_id TEXT PRIMARY KEY,
                child_id TEXT NOT NULL,
                status TEXT NOT NULL,
                data_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_photo_reviews_child_updated
                ON photo_reviews(child_id, updated_at DESC)
            """,
            """
            CREATE TABLE IF NOT EXISTS children (
                child_id TEXT PRIMARY KEY,
                family_id TEXT NOT NULL DEFAULT '',
                name TEXT NOT NULL,
                grade INTEGER NOT NULL,
                data_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS child_bindings (
                openid TEXT NOT NULL,
                child_id TEXT NOT NULL,
                family_id TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                PRIMARY KEY (openid, child_id)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_child_bindings_openid
                ON child_bindings(openid, child_id)
            """,
            """
            CREATE TABLE IF NOT EXISTS safety_events (
                safety_event_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                child_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                data_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_safety_events_child_created
                ON safety_events(child_id, created_at DESC)
            """,
            """
            CREATE TABLE IF NOT EXISTS reminder_subscriptions (
                subscription_id TEXT PRIMARY KEY,
                subscription_key TEXT UNIQUE NOT NULL,
                child_id TEXT NOT NULL,
                data_json TEXT NOT NULL,
                enabled INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_reminder_subscriptions_child
                ON reminder_subscriptions(child_id, enabled)
            """,
            """
            CREATE TABLE IF NOT EXISTS learning_submissions (
                submission_id TEXT PRIMARY KEY,
                child_id TEXT NOT NULL,
                family_id TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                source_type TEXT NOT NULL,
                data_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_learning_submissions_child_updated
                ON learning_submissions(child_id, updated_at DESC)
            """,
            """
            CREATE TABLE IF NOT EXISTS learning_submission_items (
                item_id TEXT PRIMARY KEY,
                submission_id TEXT NOT NULL,
                child_id TEXT NOT NULL,
                family_id TEXT NOT NULL DEFAULT '',
                item_index INTEGER NOT NULL,
                judge_result TEXT NOT NULL,
                question_type_id TEXT NOT NULL DEFAULT '',
                data_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_learning_submission_items_submission
                ON learning_submission_items(submission_id, item_index)
            """,
            """
            CREATE TABLE IF NOT EXISTS mastery_evidence (
                evidence_id TEXT PRIMARY KEY,
                item_id TEXT NOT NULL,
                child_id TEXT NOT NULL,
                family_id TEXT NOT NULL DEFAULT '',
                question_type_id TEXT NOT NULL,
                evidence_type TEXT NOT NULL,
                review_due INTEGER NOT NULL DEFAULT 0,
                data_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_mastery_evidence_child_type
                ON mastery_evidence(child_id, question_type_id, created_at DESC)
            """,
            """
            CREATE TABLE IF NOT EXISTS tutor_queue_items (
                queue_item_id TEXT PRIMARY KEY,
                submission_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                child_id TEXT NOT NULL,
                family_id TEXT NOT NULL DEFAULT '',
                question_type_id TEXT NOT NULL,
                status TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 0,
                data_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_tutor_queue_items_submission_status
                ON tutor_queue_items(submission_id, status, priority DESC, created_at)
            """,
            "ALTER TABLE wrong_questions ADD COLUMN IF NOT EXISTS knowledge_point TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE wrong_questions ADD COLUMN IF NOT EXISTS last_misconception TEXT",
            "ALTER TABLE wrong_questions ADD COLUMN IF NOT EXISTS highest_hint_level INTEGER NOT NULL DEFAULT 1",
            "ALTER TABLE wrong_questions ADD COLUMN IF NOT EXISTS resolved INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE wrong_questions ADD COLUMN IF NOT EXISTS practice_completed INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE children ADD COLUMN IF NOT EXISTS family_id TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE child_bindings ADD COLUMN IF NOT EXISTS family_id TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE learning_sessions ADD COLUMN IF NOT EXISTS family_id TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE wrong_questions ADD COLUMN IF NOT EXISTS family_id TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE learning_submissions ADD COLUMN IF NOT EXISTS family_id TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE learning_submission_items ADD COLUMN IF NOT EXISTS family_id TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE mastery_evidence ADD COLUMN IF NOT EXISTS family_id TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE tutor_queue_items ADD COLUMN IF NOT EXISTS family_id TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE learning_deposits ADD COLUMN IF NOT EXISTS knowledge_point TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE learning_deposits ADD COLUMN IF NOT EXISTS question_type TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE learning_deposits ADD COLUMN IF NOT EXISTS main_misconception TEXT",
            "ALTER TABLE learning_deposits ADD COLUMN IF NOT EXISTS need_review INTEGER NOT NULL DEFAULT 0",
        ]
        with self._connect() as conn:
            for statement in statements:
                conn.execute(statement)
            conn.commit()


class _PostgresConnectionAdapter:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def __enter__(self) -> "_PostgresConnectionAdapter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if exc_type is None:
                self._conn.commit()
            else:
                self._conn.rollback()
        finally:
            self._conn.close()

    def execute(self, sql: str, params: tuple[Any, ...] | None = None):
        return self._conn.execute(_postgres_sql(sql), params or ())

    def commit(self) -> None:
        self._conn.commit()


def _postgres_sql(sql: str) -> str:
    stripped = " ".join(sql.split())
    if stripped.startswith("INSERT OR IGNORE INTO child_bindings"):
        sql = sql.replace("INSERT OR IGNORE INTO", "INSERT INTO")
        sql = f"{sql} ON CONFLICT (openid, child_id) DO NOTHING"
    return sql.replace("?", "%s")
