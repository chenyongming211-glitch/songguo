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
from songguo.backend.services.learning.objective_judging import (
    ObjectiveJudgeResult,
    judge_objective_math_item,
)
from songguo.backend.services.learning.question_type_routing import (
    QuestionTypeRoute,
    route_math_question_type,
)
from songguo.backend.services.learning.question_splitter import split_grouped_math_item
from songguo.backend.services.learning.submission_models import (
    EvidenceType,
    JudgeResult,
    LearningItem,
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
    items_to_evaluate = _expand_item_split_required_items(
        store=store,
        items=store.list_submission_items(submission_id),
    )
    items_needing_gateway: list[Any] = []
    for item in items_to_evaluate:
        route = route_math_question_type(
            question_text=item.question_text,
            child_answer=item.child_answer,
            ocr_action=_item_ocr_action(item),
        )
        item = _record_question_type_route(store=store, item=item, route=route)
        if route.evaluation_strategy == "manual_confirm":
            store.update_submission_item(
                item.item_id,
                judge_result=JudgeResult.NEEDS_MANUAL_CONFIRM,
                status=LearningItemStatus.NEEDS_MANUAL_CONFIRM,
                question_type_id=route.question_type_id,
                data_json=_append_evidence_trace(
                    {
                        **(item.data_json if isinstance(item.data_json, dict) else {}),
                        "question_type_route": route.model_dump(mode="json"),
                        "reason": "question_type_requires_manual_confirm",
                    },
                    _evidence_entry(
                        stage="question_type_route",
                        label="题型路由",
                        text="题型需要先做题区或坐标复核，暂不进入自动判分。",
                        confidence=route.confidence,
                        source=route.question_type_id,
                    ),
                ),
            )
            continue
        if route.evaluation_strategy == "item_split_required":
            store.update_submission_item(
                item.item_id,
                judge_result=JudgeResult.NEEDS_MANUAL_CONFIRM,
                status=LearningItemStatus.NEEDS_MANUAL_CONFIRM,
                question_type_id=route.question_type_id,
                data_json=_append_evidence_trace(
                    {
                        **(item.data_json if isinstance(item.data_json, dict) else {}),
                        "question_type_route": route.model_dump(mode="json"),
                        "reason": "question_type_requires_item_split",
                    },
                    _evidence_entry(
                        stage="question_type_route",
                        label="题型路由",
                        text="题型需要先拆成独立小题后再判分。",
                        confidence=route.confidence,
                        source=route.question_type_id,
                    ),
                ),
            )
            continue
        native_ocr_result = _native_ocr_judge_result(item=item, route=route)
        if native_ocr_result is not None:
            _apply_objective_judge_result(
                store=store,
                submission=submission,
                item=item,
                result=native_ocr_result,
            )
            continue
        objective_result = judge_objective_math_item(
            question_text=item.question_text,
            child_answer=item.child_answer,
            ocr_action=_item_ocr_action(item),
        )
        if objective_result is None:
            items_needing_gateway.append(item)
            continue
        _apply_objective_judge_result(
            store=store,
            submission=submission,
            item=item,
            result=objective_result,
        )

    items_needing_gateway = _items_for_sync_math_evaluation(
        store=store,
        submission=submission,
        items=items_needing_gateway,
    )

    for analysis_result in _analyze_math_submission_items(
        items=items_needing_gateway,
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
                data_json=_append_evidence_trace(
                    {
                        **item.data_json,
                        "reason": "math_gateway_error",
                        "failure_reason": exc.__class__.__name__,
                    },
                    _evidence_entry(
                        stage="math_gateway",
                        label="数学分析",
                        text=f"数学结构化分析失败：{exc.__class__.__name__}，需要人工确认。",
                        source="math_gateway_error",
                    ),
                ),
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
                data_json=_append_evidence_trace(
                    {**item.data_json, "reason": "missing_analysis_or_answer"},
                    _evidence_entry(
                        stage="math_gateway",
                        label="数学分析",
                        text="题目结构或孩子答案不足，暂不能确定判分。",
                        source="missing_analysis_or_answer",
                    ),
                ),
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
                data_json=_append_evidence_trace(
                    {**item.data_json, "problem_analysis": analysis.model_dump(mode="json")},
                    _evidence_entry(
                        stage="math_gateway",
                        label="数学分析",
                        text=judged.evidence or f"结构化识别为 {question_type_id}。",
                        confidence=analysis.confidence,
                        source=analysis.source,
                        outcome="correct",
                    ),
                ),
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
                **_append_evidence_trace(
                    item.data_json,
                    _evidence_entry(
                        stage="math_gateway",
                        label="数学分析",
                        text=judged.evidence or "孩子答案与结构化结果不一致。",
                        confidence=analysis.confidence,
                        source=analysis.source,
                        outcome="wrong",
                    ),
                ),
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


def _apply_objective_judge_result(
    *,
    store: Any,
    submission: Any,
    item: Any,
    result: ObjectiveJudgeResult,
) -> None:
    result_data = result.model_dump(mode="json")
    if result.correct:
        updated_item = store.update_submission_item(
            item.item_id,
            judge_result=JudgeResult.CORRECT,
            status=LearningItemStatus.JUDGED,
            correct_answer=result.correct_answer,
            question_type_id=result.question_type_id,
            knowledge_point=result.knowledge_point,
            misconception_tag=None,
            confidence=max(item.confidence, result.confidence),
            data_json={
                **_append_evidence_trace(
                    item.data_json,
                    _evidence_entry(
                        stage="objective_judge",
                        label="客观题判分",
                        text=result.evidence,
                        confidence=result.confidence,
                        source=result.question_type_id,
                        outcome="correct",
                    ),
                ),
                "objective_judge": result_data,
            },
        )
        store.save_mastery_evidence(
            item_id=updated_item.item_id,
            child_id=updated_item.child_id,
            question_type_id=result.question_type_id,
            evidence_type=EvidenceType.SUBMISSION_CORRECT,
            is_correct=True,
            mastery_state_after=MasteryState.OBSERVED,
            review_due=False,
            solved_without_help=True,
            confidence=updated_item.confidence,
            data_json={"objective_judge": result_data},
        )
        return

    misconception_tag = result.misconception_tag or "objective_answer_mismatch"
    updated_item = store.update_submission_item(
        item.item_id,
        judge_result=JudgeResult.WRONG,
        status=LearningItemStatus.QUEUED_FOR_TUTORING,
        correct_answer=result.correct_answer,
        question_type_id=result.question_type_id,
        knowledge_point=result.knowledge_point,
        misconception_tag=misconception_tag,
        confidence=max(item.confidence, result.confidence),
        data_json={
            **_append_evidence_trace(
                item.data_json,
                _evidence_entry(
                    stage="objective_judge",
                    label="客观题判分",
                    text=result.evidence,
                    confidence=result.confidence,
                    source=result.question_type_id,
                    outcome="wrong",
                ),
            ),
            "objective_judge": result_data,
        },
    )
    store.record_wrong_question(
        session_id=submission.submission_id,
        child_id=updated_item.child_id,
        normalized_question=updated_item.question_text,
        knowledge_point=result.knowledge_point,
        mistake_summary=result.evidence or "孩子提交的客观题答案与正确结果不一致，需要进入错题陪练。",
        last_misconception=misconception_tag,
        highest_hint_level=1,
    )
    store.save_mastery_evidence(
        item_id=updated_item.item_id,
        child_id=updated_item.child_id,
        question_type_id=result.question_type_id,
        evidence_type=EvidenceType.WRONG_UNRESOLVED,
        is_correct=False,
        mastery_state_after=MasteryState.NEEDS_REVIEW,
        review_due=True,
        misconception_tag=misconception_tag,
        confidence=updated_item.confidence,
        data_json={"objective_judge": result_data},
    )
    store.enqueue_tutor_item(
        submission_id=submission.submission_id,
        item_id=updated_item.item_id,
        child_id=updated_item.child_id,
        question_type_id=result.question_type_id,
        priority=10,
    )


def _expand_item_split_required_items(*, store: Any, items: list[Any]) -> list[Any]:
    expanded: list[Any] = []
    changed = False
    for item in items:
        data_json = item.data_json if isinstance(item.data_json, dict) else {}
        if data_json.get("split_from"):
            expanded.append(item)
            continue
        routing_question_text = _item_question_text_for_split(item)
        route = route_math_question_type(
            question_text=routing_question_text,
            child_answer=item.child_answer,
            ocr_action=_item_ocr_action(item),
        )
        if route.evaluation_strategy != "item_split_required":
            expanded.append(item)
            continue
        split_items = split_grouped_math_item(
            question_text=routing_question_text,
            child_answer=item.child_answer,
            question_type_id=route.question_type_id,
        )
        if len(split_items) < 2:
            expanded.append(item)
            continue
        changed = True
        split_count = len(split_items)
        first_split = split_items[0]
        first_item = store.update_submission_item(
            item.item_id,
            question_text=first_split.question_text,
            child_answer=first_split.child_answer or None,
            judge_result=JudgeResult.UNKNOWN,
            status=LearningItemStatus.INTAKE_PENDING,
            correct_answer=None,
            question_type_id="",
            knowledge_point="",
            misconception_tag=None,
            data_json=_split_child_data_json(
                item=item,
                route=route,
                split_index=1,
                split_count=split_count,
            ),
        )
        expanded.append(first_item)
        for split_index, split_item in enumerate(split_items[1:], start=2):
            expanded.append(
                store.add_submission_item(
                    LearningItem(
                        submission_id=item.submission_id,
                        child_id=item.child_id,
                        family_id=item.family_id,
                        item_index=item.item_index,
                        question_text=split_item.question_text,
                        child_answer=split_item.child_answer or None,
                        detected_subject=item.detected_subject,
                        detected_task_type=item.detected_task_type,
                        evaluation_mode=item.evaluation_mode,
                        confidence=item.confidence,
                        bbox_json=item.bbox_json,
                        data_json=_split_child_data_json(
                            item=item,
                            route=route,
                            split_index=split_index,
                            split_count=split_count,
                        ),
                    )
                )
            )
    if not changed:
        return items
    renumbered: list[Any] = []
    for index, item in enumerate(expanded, start=1):
        if item.item_index == index:
            renumbered.append(item)
        else:
            renumbered.append(store.update_submission_item(item.item_id, item_index=index))
    return renumbered


def _split_child_data_json(
    *,
    item: Any,
    route: QuestionTypeRoute,
    split_index: int,
    split_count: int,
) -> dict[str, Any]:
    data_json = item.data_json if isinstance(item.data_json, dict) else {}
    return _append_evidence_trace(
        {
            **data_json,
            "split_from": {
                "item_id": item.item_id,
                "question_text": item.question_text,
                "child_answer": item.child_answer or "",
                "question_type_id": route.question_type_id,
                "split_index": split_index,
                "split_count": split_count,
            },
            "question_type_route": route.model_dump(mode="json"),
        },
        _evidence_entry(
            stage="question_type_route",
            label="题型路由",
            text=f"从组合题中拆出第 {split_index}/{split_count} 个小题。",
            confidence=route.confidence,
            source=route.question_type_id,
        ),
    )


def _record_question_type_route(*, store: Any, item: Any, route: QuestionTypeRoute):
    data_json = item.data_json if isinstance(item.data_json, dict) else {}
    if route.question_type_id == "math_unknown" and "question_type_route" in data_json:
        return item
    return store.update_submission_item(
        item.item_id,
        question_type_id=item.question_type_id or route.question_type_id,
        data_json={
            **_append_evidence_trace(
                data_json,
                _evidence_entry(
                    stage="question_type_route",
                    label="题型路由",
                    text="；".join(route.evidence) or route.question_type_id,
                    confidence=route.confidence,
                    source=route.question_type_id,
                ),
            ),
            "question_type_route": route.model_dump(mode="json"),
        },
    )


def _native_ocr_judge_result(*, item: Any, route: QuestionTypeRoute) -> ObjectiveJudgeResult | None:
    data_json = item.data_json if isinstance(item.data_json, dict) else {}
    if str(data_json.get("ocr_action") or "") != "RecognizeEduOralCalculation":
        return None
    judgement = str(data_json.get("ocr_judgement") or "").strip()
    if judgement not in {"correct", "wrong"}:
        return None
    correct_answer = str(data_json.get("correct_answer") or "").strip()
    if not correct_answer and judgement == "correct":
        correct_answer = str(item.child_answer or "").strip()
    if not correct_answer:
        return None
    evidence_points = data_json.get("evidence_points")
    evidence = ""
    if isinstance(evidence_points, list):
        evidence = "；".join(str(point).strip() for point in evidence_points if str(point).strip())
    if not evidence:
        evidence = "教育OCR口算判题结果作为客观题证据。"
    return ObjectiveJudgeResult(
        correct=judgement == "correct",
        correct_answer=correct_answer,
        question_type_id=route.question_type_id or "math_oral_calculation",
        knowledge_point="口算",
        question_kind=route.kind.value,
        evaluation_strategy="aliyun_edu_oral_calculation",
        misconception_tag=None if judgement == "correct" else "ocr_oral_answer_mismatch",
        evidence=evidence,
        confidence=max(0.8, min(0.98, float(getattr(item, "confidence", 0.0) or 0.0))),
    )


def _evidence_entry(
    *,
    stage: str,
    label: str,
    text: str,
    confidence: float = 0.0,
    source: str = "",
    outcome: str = "",
) -> dict[str, Any]:
    return {
        "stage": stage,
        "label": label,
        "text": str(text or "").strip(),
        "confidence": max(0.0, min(1.0, float(confidence or 0.0))),
        "source": str(source or "").strip(),
        "outcome": str(outcome or "").strip(),
    }


def _append_evidence_trace(data_json: Any, *entries: dict[str, Any]) -> dict[str, Any]:
    base = data_json if isinstance(data_json, dict) else {}
    trace = [entry for entry in base.get("evidence_trace", []) if isinstance(entry, dict)]
    seen = {
        (
            str(entry.get("stage") or ""),
            str(entry.get("label") or ""),
            str(entry.get("text") or ""),
        )
        for entry in trace
    }
    for entry in entries:
        normalized = _evidence_entry(
            stage=str(entry.get("stage") or ""),
            label=str(entry.get("label") or ""),
            text=str(entry.get("text") or ""),
            confidence=float(entry.get("confidence") or 0.0),
            source=str(entry.get("source") or ""),
            outcome=str(entry.get("outcome") or ""),
        )
        if not normalized["text"]:
            continue
        key = (normalized["stage"], normalized["label"], normalized["text"])
        if key in seen:
            continue
        seen.add(key)
        trace.append(normalized)
    return {**base, "evidence_trace": trace[-6:]}


def _basic_subject_evidence_entry(rubric_data: dict[str, Any]) -> dict[str, Any]:
    criteria = rubric_data.get("criteria_evidence")
    if isinstance(criteria, list):
        text = "；".join(str(item).strip() for item in criteria if str(item).strip())
    else:
        text = ""
    if not text:
        text = str(rubric_data.get("feedback_summary") or "").strip()
    return _evidence_entry(
        stage="basic_subject_rubric",
        label="基础学科判分",
        text=text or "已根据题干要求和孩子作答完成 rubric 判分。",
        confidence=float(rubric_data.get("confidence") or 0.0),
        source=str(rubric_data.get("question_type_id") or ""),
        outcome=str(rubric_data.get("outcome") or ""),
    )


def _item_ocr_action(item: Any) -> str:
    data_json = item.data_json if isinstance(item.data_json, dict) else {}
    return str(data_json.get("ocr_action") or "")


def _item_question_text_for_split(item: Any) -> str:
    question_text = str(getattr(item, "question_text", "") or "")
    data_json = item.data_json if isinstance(item.data_json, dict) else {}
    answer_extraction = data_json.get("answer_extraction") if isinstance(data_json, dict) else None
    work_steps = ""
    if isinstance(answer_extraction, dict):
        work_steps = str(answer_extraction.get("work_steps") or "")
    if not work_steps:
        return question_text
    if work_steps in question_text:
        return question_text
    return f"{question_text} {work_steps}".strip()


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
                **_append_evidence_trace(
                    item.data_json if isinstance(item.data_json, dict) else {},
                    _evidence_entry(
                        stage="visual_fallback",
                        label="视觉复核",
                        text="这道题已识别，正在排队判题，其他结果先显示。",
                        source="deferred_judgement",
                        outcome="pending",
                    ),
                ),
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
                data_json=_append_evidence_trace(
                    {**item.data_json, "reason": "missing_question_or_answer"},
                    _evidence_entry(
                        stage="basic_subject_rubric",
                        label="基础学科判分",
                        text="题目或孩子答案缺失，暂不能完成语文/英语判分。",
                        source="missing_question_or_answer",
                    ),
                ),
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
                data_json={
                    **_append_evidence_trace(
                        item.data_json,
                        _basic_subject_evidence_entry(rubric_data),
                    ),
                    "basic_subject_rubric": rubric_data,
                },
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
                    **_append_evidence_trace(
                        item.data_json,
                        _basic_subject_evidence_entry(rubric_data),
                    ),
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
                **_append_evidence_trace(
                    item.data_json,
                    _basic_subject_evidence_entry(rubric_data),
                ),
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
