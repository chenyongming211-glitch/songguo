from __future__ import annotations

from time import perf_counter
from typing import Any

from songguo.backend.services.learning.intent_router import (
    IntentRouterAgent,
    IntentRouterContext,
    IntentRoutingDecision,
    RouterGuard,
)
from songguo.backend.services.learning.submission_evaluator import (
    evaluate_basic_subject_submission_items,
    evaluate_submission_items,
)
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
from songguo.backend.services.learning.tutor_graph.basic_subject_graph import (
    is_basic_subject_route,
)


def input_normalize_node(state: LearningSubmissionGraphState) -> LearningSubmissionGraphState:
    return state.model_copy(
        update={
            "subject": state.subject or "auto",
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
        basic_tutor_graph: Any | None = None,
        basic_rubric_evaluator: Any | None = None,
        intent_router: Any | None = None,
        router_guard: RouterGuard | None = None,
    ) -> None:
        self.store = store
        self.math_gateway = math_gateway
        self.tutor_graph = tutor_graph
        self.basic_tutor_graph = basic_tutor_graph
        self.basic_rubric_evaluator = basic_rubric_evaluator
        self.intent_router = intent_router or IntentRouterAgent()
        self.router_guard = router_guard or RouterGuard()

    def intake_parse_node(self, state: LearningSubmissionGraphState) -> LearningSubmissionGraphState:
        draft = parse_text_submission(
            child_id=state.child_id,
            subject=state.subject,
            grade=state.grade,
            raw_text=state.raw_text,
        )
        route_started_at = perf_counter()
        route = self._route_intent(state, draft)
        route_latency_ms = int((perf_counter() - route_started_at) * 1000)
        detected_subject = route.subject
        subject = detected_subject if _is_auto_subject(state.subject) else state.subject
        data_json = {
            "needs_manual_confirm": draft.needs_manual_confirm,
            "needs_clarification": route.needs_clarification,
            "route_to": route.route_to,
        }
        submission = self.store.create_submission(
            child_id=state.child_id,
            subject=subject,
            grade=state.grade,
            source_type=state.source_type,
            raw_text=state.raw_text,
            image_refs=state.image_refs,
            detected_subject=detected_subject,
            detected_task_type=route.task_type,
            detected_intent=route.user_intent,
            subject_confidence=route.confidence,
            route_to=route.route_to,
            routing_evidence=route.evidence,
            guard_reason=route.guard_reason,
            router_version=route.router_version,
            needs_clarification=route.needs_clarification,
            data_json=data_json,
        )
        self._record_intent_router_observation(
            submission=submission,
            route=route,
            latency_ms=route_latency_ms,
        )
        for draft_item in draft.items:
            item_data_json = dict(state.item_metadata.get(draft_item.item_index, {}))
            if draft_item.work_steps:
                item_data_json["answer_extraction"] = {
                    "work_steps": draft_item.work_steps,
                }
            self.store.add_submission_item(
                LearningItem(
                    submission_id=submission.submission_id,
                    child_id=state.child_id,
                    item_index=draft_item.item_index,
                    question_text=draft_item.question_text,
                    child_answer=draft_item.child_answer,
                    detected_subject=detected_subject,
                    detected_task_type=route.task_type,
                    evaluation_mode=_evaluation_mode_for_route(route),
                    confidence=draft_item.confidence,
                    bbox_json=state.item_bboxes.get(draft_item.item_index),
                    data_json=item_data_json,
                )
            )
        if draft.needs_manual_confirm or route.needs_clarification:
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
        if submission.route_to and submission.route_to != "math_mistake_tutor":
            if is_basic_subject_route(submission.route_to):
                snapshot = evaluate_basic_subject_submission_items(
                    store=self.store,
                    submission_id=state.submission_id,
                    rubric_evaluator=self.basic_rubric_evaluator,
                )
                return _state_from_submission(state, snapshot.submission)
            submission = self.store.update_submission(
                submission.submission_id,
                status=LearningSubmissionStatus.NEEDS_MANUAL_CONFIRM,
                needs_clarification=True,
                guard_reason=submission.guard_reason or "non_math_tutor_not_enabled",
            )
            return _state_from_submission(state, submission)
        snapshot = evaluate_submission_items(
            store=self.store,
            submission_id=state.submission_id,
            math_gateway=self.math_gateway,
        )
        return _state_from_submission(state, snapshot.submission)

    def _route_intent(
        self,
        state: LearningSubmissionGraphState,
        draft: Any,
    ) -> IntentRoutingDecision:
        if not _is_auto_subject(state.subject):
            subject = (state.subject or "math").lower()
            return IntentRoutingDecision(
                subject=subject,
                task_type="unknown",
                user_intent="check_answer",
                confidence=1.0,
                evidence=["前端显式传入学科"],
                needs_clarification=False,
                route_to={
                    "math": "math_mistake_tutor",
                    "chinese": "chinese_basic_tutor",
                    "english": "english_basic_tutor",
                }.get(subject, "clarification_tutor"),
                router_version="explicit_subject",
            )
        first_item = draft.items[0] if draft.items else None
        context = IntentRouterContext(
            child_id=state.child_id,
            subject_hint=state.subject,
            grade=state.grade,
            source_type=str(state.source_type),
            raw_text=state.raw_text,
            question_text=first_item.question_text if first_item else "",
            child_answer=first_item.child_answer if first_item else None,
            input_confidence=first_item.confidence if first_item else 0.0,
        )
        return self.router_guard.apply(self.intent_router.route(context), raw_text=state.raw_text)

    def _record_intent_router_observation(
        self,
        *,
        submission: Any,
        route: IntentRoutingDecision,
        latency_ms: int,
    ) -> None:
        self.store.record_ai_call(
            child_id=submission.child_id,
            session_id=submission.submission_id,
            provider=route.provider,
            model=route.model,
            operation="intent_router.route",
            token_estimate=0,
            status="clarification" if route.needs_clarification else "success",
            agent="IntentRouterAgent",
            submission_id=submission.submission_id,
            latency_ms=latency_ms,
            confidence=route.confidence,
            route_to=route.route_to,
            failure_reason=route.guard_reason,
            metadata={
                "detected_subject": route.subject,
                "detected_task_type": route.task_type,
                "detected_intent": route.user_intent,
                "router_version": route.router_version,
                "source": route.source,
                "needs_clarification": route.needs_clarification,
                "evidence_count": len(route.evidence),
            },
        )

    def start_or_resume_active_wrong_item_node(
        self,
        state: LearningSubmissionGraphState,
    ) -> LearningSubmissionGraphState:
        if not state.submission_id:
            return state.model_copy(update={"error": "missing_submission_id"})
        queue_item = self.store.get_active_tutor_item(state.submission_id)
        if queue_item is None:
            submission = self.store.require_submission(state.submission_id)
            if (
                submission.status == LearningSubmissionStatus.NEEDS_MANUAL_CONFIRM
                or submission.needs_clarification
            ):
                return _state_from_submission(state, submission)
            completed = self.store.complete_submission_if_queue_done(state.submission_id)
            return _state_from_submission(state, completed)
        if queue_item.status == TutorQueueStatus.PENDING:
            queue_item = self.store.mark_tutor_item_active(queue_item.queue_item_id)
        if not queue_item.tutor_session_id:
            item = _find_submission_item(self.store, state.submission_id, queue_item.item_id)
            submission = self.store.require_submission(state.submission_id)
            tutor_graph = (
                self.basic_tutor_graph
                if is_basic_subject_route(submission.route_to)
                else self.tutor_graph
            )
            created = tutor_graph.start(
                child_id=state.child_id,
                subject=submission.subject or state.subject,
                grade=submission.grade or state.grade,
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


def _is_auto_subject(subject: str) -> bool:
    return (subject or "").strip().lower() in {"", "auto"}


def _evaluation_mode_for_route(route: IntentRoutingDecision) -> str:
    if route.needs_clarification or route.route_to == "clarification_tutor":
        return "clarification"
    if route.subject == "math":
        return "exact_answer"
    return "rubric_feedback"
