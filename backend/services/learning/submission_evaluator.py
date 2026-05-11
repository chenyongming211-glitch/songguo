from __future__ import annotations

from typing import Any

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


def evaluate_submission_items(
    *,
    store: Any,
    submission_id: str,
    math_gateway: MathProblemStructuringGateway | None = None,
):
    gateway = math_gateway or MathProblemStructuringGateway()
    submission = store.require_submission(submission_id)

    for item in store.list_submission_items(submission_id):
        analysis = gateway.analyze(
            question_text=item.question_text,
            grade=submission.grade,
            subject=submission.subject,
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
