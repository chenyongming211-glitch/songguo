from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from time import perf_counter
from typing import Any

from songguo.backend.services.learning.basic_subject_rubric import (
    BasicSubjectRubricContext,
    BasicSubjectRubricEvaluator,
)
from songguo.backend.services.learning.math_structuring import (
    MathProblemStructuringGateway,
    ProblemAnalysis,
    reliable_final_answer,
)
from songguo.backend.services.learning.submission_models import (
    EvidenceType,
    JudgeResult,
    LearningItemStatus,
    LearningSubmissionStatus,
    MasteryState,
)
from songguo.backend.services.learning.tutor_graph.math_mistake_nodes import rule_judge_node
from songguo.backend.services.learning.tutor_graph.state import (
    MathMistakeTutorGraphState,
    RuleJudgeResult,
)

VISUAL_FALLBACK_KEY = "visual_fallback"


def evaluate_submission_items(
    *,
    store: Any,
    submission_id: str,
    math_gateway: MathProblemStructuringGateway | None = None,
):
    gateway = math_gateway or MathProblemStructuringGateway()
    submission = store.require_submission(submission_id)
    items_to_evaluate = _items_for_sync_math_evaluation(
        store=store,
        submission=submission,
        items=store.list_submission_items(submission_id),
    )

    for analysis_result in _analyze_math_submission_items(
        items=items_to_evaluate,
        submission=submission,
        gateway=gateway,
    ):
        item = analysis_result.item
        if analysis_result.error is not None:
            exc = analysis_result.error
            store.record_ai_call(
                child_id=item.child_id,
                session_id=submission.submission_id,
                provider=str(getattr(gateway, "provider", "") or "math_gateway"),
                model=str(getattr(gateway, "model", "") or getattr(gateway, "model_name", "") or "configured"),
                operation="math_gateway.analyze",
                token_estimate=0,
                status="error",
                agent="MathProblemStructuringGateway",
                submission_id=submission.submission_id,
                item_id=item.item_id,
                latency_ms=analysis_result.latency_ms,
                failure_reason=exc.__class__.__name__,
                metadata={
                    "detected_subject": submission.detected_subject or submission.subject,
                    "question_type_id": item.question_type_id,
                },
            )
            store.update_submission_item(
                item.item_id,
                judge_result=JudgeResult.NEEDS_MANUAL_CONFIRM,
                status=LearningItemStatus.NEEDS_MANUAL_CONFIRM,
                data_json={
                    **item.data_json,
                    "reason": "math_gateway_error",
                    "failure_reason": exc.__class__.__name__,
                },
            )
            continue
        analysis = analysis_result.analysis
        _record_math_gateway_observation(
            store=store,
            gateway=gateway,
            submission=submission,
            item=item,
            analysis=analysis,
            latency_ms=analysis_result.latency_ms,
        )
        if analysis is None or not item.child_answer:
            store.update_submission_item(
                item.item_id,
                judge_result=JudgeResult.NEEDS_MANUAL_CONFIRM,
                status=LearningItemStatus.NEEDS_MANUAL_CONFIRM,
                data_json={**item.data_json, "reason": "missing_analysis_or_answer"},
            )
            continue

        judged = _judge_item(analysis=analysis, child_answer=item.child_answer)
        question_type_id = analysis.problem_type
        knowledge_point = analysis.knowledge_point
        correct_answer = reliable_final_answer(analysis)

        if judged.correct:
            updated_item = store.update_submission_item(
                item.item_id,
                judge_result=JudgeResult.CORRECT,
                status=LearningItemStatus.JUDGED,
                correct_answer=correct_answer,
                question_type_id=question_type_id,
                knowledge_point=knowledge_point,
                confidence=analysis.confidence,
                data_json={**item.data_json, "problem_analysis": analysis.model_dump(mode="json")},
            )
            store.save_mastery_evidence(
                item_id=updated_item.item_id,
                child_id=updated_item.child_id,
                question_type_id=question_type_id,
                evidence_type=EvidenceType.SUBMISSION_CORRECT,
                is_correct=True,
                mastery_state_after=MasteryState.OBSERVED,
                review_due=False,
                solved_without_help=True,
                confidence=analysis.confidence,
            )
            continue

        misconception_tag = judged.misconception_tag or _fallback_misconception_tag(analysis)
        updated_item = store.update_submission_item(
            item.item_id,
            judge_result=JudgeResult.WRONG,
            status=LearningItemStatus.QUEUED_FOR_TUTORING,
            correct_answer=correct_answer,
            question_type_id=question_type_id,
            knowledge_point=knowledge_point,
            misconception_tag=misconception_tag,
            confidence=analysis.confidence,
            data_json={
                **item.data_json,
                "problem_analysis": analysis.model_dump(mode="json"),
                "judge_evidence": judged.evidence,
            },
        )
        store.record_wrong_question(
            session_id=submission.submission_id,
            child_id=updated_item.child_id,
            normalized_question=updated_item.question_text,
            knowledge_point=knowledge_point,
            mistake_summary=judged.evidence or "孩子提交的答案与正确结果不一致，需要进入错题陪练。",
            last_misconception=misconception_tag,
            highest_hint_level=1,
        )
        store.save_mastery_evidence(
            item_id=updated_item.item_id,
            child_id=updated_item.child_id,
            question_type_id=question_type_id,
            evidence_type=EvidenceType.WRONG_UNRESOLVED,
            is_correct=False,
            mastery_state_after=MasteryState.NEEDS_REVIEW,
            review_due=True,
            misconception_tag=misconception_tag,
            confidence=analysis.confidence,
        )
        store.enqueue_tutor_item(
            submission_id=submission.submission_id,
            item_id=updated_item.item_id,
            child_id=updated_item.child_id,
            question_type_id=question_type_id,
            priority=10,
        )

    snapshot = store.get_submission_snapshot(submission_id)
    if snapshot.tutor_queue:
        store.update_submission(submission_id, status=LearningSubmissionStatus.TUTORING)
    elif snapshot.submission.needs_manual_confirm_count:
        store.update_submission(submission_id, status=LearningSubmissionStatus.NEEDS_MANUAL_CONFIRM)
    elif _has_active_deferred_or_visual_fallback(snapshot.items):
        store.update_submission(submission_id, status=LearningSubmissionStatus.JUDGED)
    else:
        store.update_submission(submission_id, status=LearningSubmissionStatus.JUDGED)
        store.complete_submission_if_queue_done(submission_id)
    return store.get_submission_snapshot(submission_id)


@dataclass
class _MathAnalysisResult:
    item: Any
    analysis: ProblemAnalysis | None = None
    latency_ms: int = 0
    error: Exception | None = None


def _analyze_math_submission_items(
    *,
    items: list[Any],
    submission: Any,
    gateway: MathProblemStructuringGateway,
) -> list[_MathAnalysisResult]:
    if len(items) <= 1:
        return [_analyze_one_math_submission_item(item=item, submission=submission, gateway=gateway) for item in items]
    max_workers = _submission_eval_max_workers(default=min(4, len(items)))
    if max_workers <= 1:
        return [_analyze_one_math_submission_item(item=item, submission=submission, gateway=gateway) for item in items]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as executor:
        return list(
            executor.map(
                lambda item: _analyze_one_math_submission_item(
                    item=item,
                    submission=submission,
                    gateway=gateway,
                ),
                items,
            )
        )


def _analyze_one_math_submission_item(
    *,
    item: Any,
    submission: Any,
    gateway: MathProblemStructuringGateway,
) -> _MathAnalysisResult:
    gateway_started_at = perf_counter()
    try:
        analysis = gateway.analyze(
            question_text=item.question_text,
            grade=submission.grade,
            subject=submission.subject,
        )
        return _MathAnalysisResult(
            item=item,
            analysis=analysis,
            latency_ms=int((perf_counter() - gateway_started_at) * 1000),
        )
    except Exception as exc:
        return _MathAnalysisResult(
            item=item,
            latency_ms=int((perf_counter() - gateway_started_at) * 1000),
            error=exc,
        )


def _submission_eval_max_workers(*, default: int = 4) -> int:
    raw = os.getenv("SONGGUO_SUBMISSION_EVAL_MAX_WORKERS", "")
    try:
        return max(1, int(raw or default))
    except ValueError:
        return max(1, default)


def _items_for_sync_math_evaluation(
    *,
    store: Any,
    submission: Any,
    items: list[Any],
) -> list[Any]:
    if str(submission.source_type) != "photo":
        return items
    sync_limit = _photo_sync_judgement_limit()
    if len(items) <= sync_limit:
        return items
    for item in items[sync_limit:]:
        state = item.data_json.get(VISUAL_FALLBACK_KEY) if isinstance(item.data_json, dict) else None
        if isinstance(state, dict) and state.get("status") in {"pending", "running", "done"}:
            continue
        store.update_submission_item(
            item.item_id,
            data_json={
                **(item.data_json if isinstance(item.data_json, dict) else {}),
                VISUAL_FALLBACK_KEY: {
                    "status": "pending",
                    "reason": "deferred_judgement",
                    "message": "这道题已识别，正在排队判题，其他结果先显示。",
                    "attempts": 0,
                },
            },
        )
    return items[:sync_limit]


def _photo_sync_judgement_limit() -> int:
    raw = os.getenv("SONGGUO_PHOTO_SYNC_JUDGEMENT_LIMIT", "")
    try:
        return max(1, int(raw or 10))
    except ValueError:
        return 10


def _has_active_deferred_or_visual_fallback(items: list[Any]) -> bool:
    for item in items:
        data_json = item.data_json if isinstance(item.data_json, dict) else {}
        state = data_json.get(VISUAL_FALLBACK_KEY)
        if isinstance(state, dict) and state.get("status") in {"pending", "running"}:
            return True
    return False


def _record_math_gateway_observation(
    *,
    store: Any,
    gateway: Any,
    submission: Any,
    item: Any,
    analysis: ProblemAnalysis | None,
    latency_ms: int,
) -> None:
    metadata = {
        "detected_subject": submission.detected_subject or submission.subject,
        "question_type_id": analysis.problem_type if analysis is not None else item.question_type_id,
        "knowledge_point": analysis.knowledge_point if analysis is not None else item.knowledge_point,
        "source": analysis.source if analysis is not None else "",
    }
    store.record_ai_call(
        child_id=item.child_id,
        session_id=submission.submission_id,
        provider=str(getattr(gateway, "provider", "") or "math_gateway"),
        model=str(getattr(gateway, "model", "") or getattr(gateway, "model_name", "") or "configured"),
        operation="math_gateway.analyze",
        token_estimate=0,
        status="success" if analysis is not None else "empty",
        agent="MathProblemStructuringGateway",
        submission_id=submission.submission_id,
        item_id=item.item_id,
        latency_ms=latency_ms,
        confidence=analysis.confidence if analysis is not None else 0.0,
        metadata=metadata,
    )


def evaluate_basic_subject_submission_items(
    *,
    store: Any,
    submission_id: str,
    rubric_evaluator: BasicSubjectRubricEvaluator | None = None,
):
    submission = store.require_submission(submission_id)
    subject = (submission.detected_subject or submission.subject or "general").lower()
    evaluator = rubric_evaluator or BasicSubjectRubricEvaluator()

    for item in store.list_submission_items(submission_id):
        if not item.question_text.strip() or not (item.child_answer or "").strip():
            store.update_submission_item(
                item.item_id,
                judge_result=JudgeResult.NEEDS_MANUAL_CONFIRM,
                status=LearningItemStatus.NEEDS_MANUAL_CONFIRM,
                data_json={**item.data_json, "reason": "missing_question_or_answer"},
            )
            continue

        rubric_started_at = perf_counter()
        rubric = evaluator.evaluate(
            BasicSubjectRubricContext(
                subject=subject,
                task_type=submission.detected_task_type or item.detected_task_type or "unknown",
                grade=submission.grade,
                question_text=item.question_text,
                child_answer=item.child_answer or "",
                route_to=submission.route_to,
            )
        )
        rubric_latency_ms = int((perf_counter() - rubric_started_at) * 1000)
        store.record_ai_call(
            child_id=item.child_id,
            session_id=submission.submission_id,
            provider=rubric.provider,
            model=rubric.model,
            operation="basic_subject_rubric.evaluate",
            token_estimate=0,
            status=rubric.outcome,
            agent="BasicSubjectRubricEvaluator",
            submission_id=submission.submission_id,
            item_id=item.item_id,
            latency_ms=rubric_latency_ms,
            confidence=rubric.confidence,
            route_to=submission.route_to,
            metadata={
                "rubric_outcome": rubric.outcome,
                "question_type_id": rubric.question_type_id,
                "knowledge_point": rubric.knowledge_point,
                "misconception_tag": rubric.misconception_tag,
                "source": rubric.source,
                "rubric_version": rubric.rubric_version,
            },
        )
        rubric_data = rubric.model_dump(mode="json")
        if rubric.outcome == "needs_manual_confirm":
            store.update_submission_item(
                item.item_id,
                judge_result=JudgeResult.NEEDS_MANUAL_CONFIRM,
                status=LearningItemStatus.NEEDS_MANUAL_CONFIRM,
                question_type_id=rubric.question_type_id,
                knowledge_point=rubric.knowledge_point,
                misconception_tag=rubric.misconception_tag,
                confidence=max(item.confidence, rubric.confidence),
                data_json={**item.data_json, "basic_subject_rubric": rubric_data},
            )
            continue

        if rubric.is_correct:
            updated_item = store.update_submission_item(
                item.item_id,
                judge_result=JudgeResult.CORRECT,
                status=LearningItemStatus.JUDGED,
                question_type_id=rubric.question_type_id,
                knowledge_point=rubric.knowledge_point,
                misconception_tag=None,
                confidence=max(item.confidence, rubric.confidence),
                data_json={
                    **item.data_json,
                    "basic_subject_tutor": True,
                    "route_to": submission.route_to,
                    "basic_subject_rubric": rubric_data,
                },
            )
            store.save_mastery_evidence(
                item_id=updated_item.item_id,
                child_id=updated_item.child_id,
                question_type_id=rubric.question_type_id,
                evidence_type=EvidenceType.SUBMISSION_CORRECT,
                is_correct=True,
                mastery_state_after=MasteryState.OBSERVED,
                review_due=False,
                solved_without_help=True,
                confidence=updated_item.confidence,
                data_json={"basic_subject_rubric": rubric_data},
            )
            continue

        updated_item = store.update_submission_item(
            item.item_id,
            judge_result=JudgeResult.WRONG,
            status=LearningItemStatus.QUEUED_FOR_TUTORING,
            question_type_id=rubric.question_type_id,
            knowledge_point=rubric.knowledge_point,
            misconception_tag=rubric.misconception_tag,
            confidence=max(item.confidence, rubric.confidence, submission.subject_confidence),
            data_json={
                **item.data_json,
                "basic_subject_tutor": True,
                "route_to": submission.route_to,
                "basic_subject_rubric": rubric_data,
            },
        )
        store.record_wrong_question(
            session_id=submission.submission_id,
            child_id=updated_item.child_id,
            normalized_question=updated_item.question_text,
            knowledge_point=rubric.knowledge_point,
            mistake_summary=rubric.feedback_summary
            or "非数学题需要进入基础陪练，确认孩子的理解过程和表达依据。",
            last_misconception=updated_item.misconception_tag,
            highest_hint_level=1,
        )
        store.save_mastery_evidence(
            item_id=updated_item.item_id,
            child_id=updated_item.child_id,
            question_type_id=rubric.question_type_id,
            evidence_type=EvidenceType.WRONG_UNRESOLVED,
            is_correct=False,
            mastery_state_after=MasteryState.NEEDS_REVIEW,
            review_due=True,
            misconception_tag=updated_item.misconception_tag,
            confidence=updated_item.confidence,
            data_json={"basic_subject_rubric": rubric_data},
        )
        store.enqueue_tutor_item(
            submission_id=submission.submission_id,
            item_id=updated_item.item_id,
            child_id=updated_item.child_id,
            question_type_id=rubric.question_type_id,
            priority=6 if rubric.outcome == "partial" else 5,
        )

    snapshot = store.get_submission_snapshot(submission_id)
    if snapshot.tutor_queue:
        store.update_submission(submission_id, status=LearningSubmissionStatus.TUTORING)
    elif snapshot.submission.needs_manual_confirm_count:
        store.update_submission(submission_id, status=LearningSubmissionStatus.NEEDS_MANUAL_CONFIRM)
    else:
        store.update_submission(submission_id, status=LearningSubmissionStatus.JUDGED)
        store.complete_submission_if_queue_done(submission_id)
    return store.get_submission_snapshot(submission_id)


def _judge_item(*, analysis: ProblemAnalysis, child_answer: str) -> RuleJudgeResult:
    state = MathMistakeTutorGraphState(
        child_id="submission_evaluator",
        subject=analysis.subject,
        grade=analysis.grade,
        question_text="",
        child_answer=child_answer,
        problem_analysis=analysis.model_dump(mode="json"),
    )
    judged_state = rule_judge_node(state, child_answer=child_answer)
    return RuleJudgeResult.model_validate(judged_state.rule_judge_result or {})


def _fallback_misconception_tag(analysis: ProblemAnalysis) -> str:
    if analysis.common_misconceptions:
        return analysis.common_misconceptions[0].tag
    return "needs_tutor_review"
