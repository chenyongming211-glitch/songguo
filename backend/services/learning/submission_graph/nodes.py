from __future__ import annotations

from typing import Any

from songguo.backend.services.learning.submission_evaluator import evaluate_submission_items
from songguo.backend.services.learning.submission_graph.state import (
    LearningSubmissionGraphState,
)
from songguo.backend.services.learning.submission_intake import parse_text_submission
from songguo.backend.services.learning.submission_models import (
    LearningItem,
    LearningSubmissionStatus,
    SourceType,
    TutorQueueStatus,
)


def input_normalize_node(state: LearningSubmissionGraphState) -> LearningSubmissionGraphState:
    return state.model_copy(
        update={
            "subject": state.subject or "math",
            "source_type": SourceType(state.source_type),
            "raw_text": (state.raw_text or "").strip(),
        }
    )


class LearningSubmissionGraphNodes:
    def __init__(
        self,
        *,
        store: Any,
        math_gateway: Any,
        tutor_graph: Any,
    ) -> None:
        self.store = store
        self.math_gateway = math_gateway
        self.tutor_graph = tutor_graph

    def intake_parse_node(self, state: LearningSubmissionGraphState) -> LearningSubmissionGraphState:
        draft = parse_text_submission(
            child_id=state.child_id,
            subject=state.subject,
            grade=state.grade,
            raw_text=state.raw_text,
        )
        submission = self.store.create_submission(
            child_id=state.child_id,
            subject=state.subject,
            grade=state.grade,
            source_type=state.source_type,
            raw_text=state.raw_text,
            data_json={"needs_manual_confirm": draft.needs_manual_confirm},
        )
        for draft_item in draft.items:
            self.store.add_submission_item(
                LearningItem(
                    submission_id=submission.submission_id,
                    child_id=state.child_id,
                    item_index=draft_item.item_index,
                    question_text=draft_item.question_text,
                    child_answer=draft_item.child_answer,
                    confidence=draft_item.confidence,
                )
            )
        if draft.needs_manual_confirm:
            submission = self.store.update_submission(
                submission.submission_id,
                status=LearningSubmissionStatus.NEEDS_MANUAL_CONFIRM,
            )
        return _state_from_submission(state, submission)

    def structure_and_judge_items_node(
        self,
        state: LearningSubmissionGraphState,
    ) -> LearningSubmissionGraphState:
        if not state.submission_id:
            return state.model_copy(update={"error": "missing_submission_id"})
        submission = self.store.require_submission(state.submission_id)
        if submission.status == LearningSubmissionStatus.NEEDS_MANUAL_CONFIRM:
            return _state_from_submission(state, submission)
        snapshot = evaluate_submission_items(
            store=self.store,
            submission_id=state.submission_id,
            math_gateway=self.math_gateway,
        )
        return _state_from_submission(state, snapshot.submission)

    def start_or_resume_active_wrong_item_node(
        self,
        state: LearningSubmissionGraphState,
    ) -> LearningSubmissionGraphState:
        if not state.submission_id:
            return state.model_copy(update={"error": "missing_submission_id"})
        queue_item = self.store.get_active_tutor_item(state.submission_id)
        if queue_item is None:
            completed = self.store.complete_submission_if_queue_done(state.submission_id)
            return _state_from_submission(state, completed)
        if queue_item.status == TutorQueueStatus.PENDING:
            queue_item = self.store.mark_tutor_item_active(queue_item.queue_item_id)
        if not queue_item.tutor_session_id:
            item = _find_submission_item(self.store, state.submission_id, queue_item.item_id)
            created = self.tutor_graph.start(
                child_id=state.child_id,
                subject=state.subject,
                grade=state.grade,
                question_text=item.question_text,
            )
            queue_item = self.store.mark_tutor_item_active(
                queue_item.queue_item_id,
                tutor_session_id=created.session_id,
            )
        submission = self.store.update_submission(
            state.submission_id,
            status=LearningSubmissionStatus.TUTORING,
            active_queue_item_id=queue_item.queue_item_id,
        )
        return _state_from_submission(
            state,
            submission,
            active_queue_item_id=queue_item.queue_item_id,
            active_tutor_session_id=queue_item.tutor_session_id,
        )

    def build_submission_summary_node(
        self,
        state: LearningSubmissionGraphState,
    ) -> LearningSubmissionGraphState:
        if not state.submission_id:
            return state
        snapshot = self.store.get_submission_snapshot(state.submission_id)
        summary = {
            "item_count": snapshot.submission.item_count,
            "correct_count": snapshot.submission.correct_count,
            "wrong_count": snapshot.submission.wrong_count,
            "tutor_queue_count": len(snapshot.tutor_queue),
        }
        return _state_from_submission(state, snapshot.submission, summary=summary)


def _state_from_submission(
    state: LearningSubmissionGraphState,
    submission: Any,
    *,
    active_queue_item_id: str | None = None,
    active_tutor_session_id: str | None = None,
    summary: dict[str, Any] | None = None,
) -> LearningSubmissionGraphState:
    return state.model_copy(
        update={
            "submission_id": submission.submission_id,
            "status": submission.status,
            "item_count": submission.item_count,
            "correct_count": submission.correct_count,
            "wrong_count": submission.wrong_count,
            "active_queue_item_id": active_queue_item_id or submission.active_queue_item_id,
            "active_tutor_session_id": active_tutor_session_id or state.active_tutor_session_id,
            "summary": summary or state.summary,
        }
    )


def _find_submission_item(store: Any, submission_id: str, item_id: str) -> LearningItem:
    for item in store.list_submission_items(submission_id):
        if item.item_id == item_id:
            return item
    raise KeyError(item_id)
