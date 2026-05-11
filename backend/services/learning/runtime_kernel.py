from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from songguo.backend.services.learning.ai_engine import (
    AIEngineContext,
    AIEngineProvider,
    DeterministicFallbackProvider,
)
from songguo.backend.services.learning.leakage_checker import (
    LeakageAction,
    check_answer_leakage,
)
from songguo.backend.services.learning.math_structuring import (
    AttemptEvaluation,
    ProblemAnalysis,
    expected_answer_for_leakage,
)
from songguo.backend.services.learning.models import LearningPhase, TeachingProgress, utc_now
from songguo.backend.services.learning.practice_recommender import PracticeItem
from songguo.backend.services.learning.progress import build_teaching_progress
from songguo.backend.services.learning.store import InMemoryLearningStore
from songguo.backend.services.learning.teaching_assets import (
    DEFAULT_MATH_ASSET_LIBRARY,
    ConceptCard,
)


class KernelCreateResult(BaseModel):
    session_id: str
    question_text: str
    subject: str = "math"
    grade: int = 3
    phase: str
    hint_level: int
    message: str
    answer_unlocked: bool
    teaching_progress: TeachingProgress


class KernelAttemptResult(BaseModel):
    correct: bool
    partially_correct: bool = False
    phase: str
    hint_level: int
    message: str
    answer_unlocked: bool
    misconception_tag: str | None = None
    matched_key_point_id: str | None = None
    next_key_point_id: str | None = None
    teaching_progress: TeachingProgress
    practice_items: list[PracticeItem] = Field(default_factory=list)


class RuntimeKernel:
    """Runtime-only teaching control boundary.

    The kernel consumes structured `ProblemAnalysis` and controls what key point
    is released next. It does not own model prompting, asset authoring, accounts,
    or parent-facing reports.
    """

    def __init__(
        self,
        *,
        store: InMemoryLearningStore,
        provider: AIEngineProvider | None = None,
    ) -> None:
        self.store = store
        self.provider = provider or DeterministicFallbackProvider()

    def create_math_session(
        self,
        *,
        child_id: str,
        grade: int,
        question_text: str,
        context: AIEngineContext,
    ) -> KernelCreateResult:
        analysis = self.provider.structure_math_problem(
            question_text=question_text,
            grade=grade,
            context=context,
        )
        first_key_point = analysis.first_key_point
        draft = self.provider.generate_hint(
            problem_analysis=analysis,
            key_point=first_key_point,
            student_profile={},
            context=context,
        )
        message, leakage_result = _safe_message(
            draft_text=draft.text,
            answer_unlocked=False,
            expected_answer=expected_answer_for_leakage(analysis),
            hint_level=draft.hint_level or 1,
        )
        session = self.store.create_session(
            child_id=child_id,
            subject="math",
            grade=grade,
            question_text=question_text,
            normalized_question=_normalize_question(question_text),
            knowledge_point=analysis.problem_type,
            current_prompt=message,
            problem_analysis=analysis.model_dump(mode="json"),
            current_key_point_id=first_key_point.id,
            released_key_point_ids=[first_key_point.id],
        )
        self.store.append_event(
            session_id=session.session_id,
            child_id=child_id,
            event_type="question.structured",
            payload={
                "problem_type": analysis.problem_type,
                "skill_ids": analysis.skill_ids,
                "misconception_ids": analysis.misconception_ids,
                "concept_card_ids": analysis.concept_card_ids,
                "source": analysis.source,
                "confidence": analysis.confidence,
            },
        )
        self.store.append_event(
            session_id=session.session_id,
            child_id=child_id,
            event_type="key_point.released",
            payload={
                "key_point_id": first_key_point.id,
                "release_stage": first_key_point.release_stage,
                "skill_ids": analysis.skill_ids,
                "concept_card_ids": analysis.concept_card_ids,
                "prompt": message,
            },
            deeptutor_trace_id=draft.trace_id,
            leakage_check_result=leakage_result.model_dump(mode="json"),
        )
        if leakage_result.action == LeakageAction.BLOCK:
            self.store.record_safety_event(
                session_id=session.session_id,
                child_id=child_id,
                event_type="safety.blocked",
                input_text=question_text,
                blocked_text=draft.text,
                reason=leakage_result.reason,
            )
        _record_ai_call(self.store, child_id=child_id, session_id=session.session_id, draft=draft)
        return KernelCreateResult(
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

    def submit_math_attempt(
        self,
        session_id: str,
        *,
        child_answer: str,
    ) -> KernelAttemptResult:
        session = self.store.require_session(session_id)
        if session.phase == LearningPhase.SIMILAR_PRACTICE:
            return self._respond_inside_practice_phase(session, child_answer)
        analysis = ProblemAnalysis.model_validate(session.problem_analysis)
        evaluation = _evaluate_attempt(
            analysis=analysis,
            child_answer=child_answer,
            current_key_point_id=session.current_key_point_id,
        )
        self.store.append_event(
            session_id=session.session_id,
            child_id=session.child_id,
            event_type="child.attempt_submitted",
            payload={
                "child_answer": child_answer,
                "current_key_point_id": session.current_key_point_id,
            },
        )
        self.store.append_event(
            session_id=session.session_id,
            child_id=session.child_id,
            event_type="attempt.evaluated",
            payload={
                **evaluation.model_dump(mode="json"),
                "skill_ids": analysis.skill_ids,
                "misconception_ids": analysis.misconception_ids,
            },
        )
        for key_point_id in evaluation.mastered_key_point_ids:
            self.store.append_event(
                session_id=session.session_id,
                child_id=session.child_id,
                event_type="key_point.mastered",
                payload={
                    "key_point_id": key_point_id,
                    "child_answer": child_answer,
                    "evidence": evaluation.evidence,
                },
            )

        if evaluation.correct:
            return self._complete_with_practice(session_id, analysis, evaluation, child_answer)

        next_key_point = (
            analysis.get_key_point(evaluation.next_key_point_id)
            if evaluation.next_key_point_id
            else analysis.get_key_point(session.current_key_point_id)
        )
        next_hint_level = min(5, session.hint_level + 1)
        adaptive_text = _adaptive_key_point_feedback(
            analysis=analysis,
            child_answer=child_answer,
            current_key_point_id=session.current_key_point_id,
            next_key_point_id=next_key_point.id,
        )
        concept_card = (
            _concept_card_for_analysis(analysis)
            if adaptive_text is None
            and _should_release_concept_card(session.attempt_count, evaluation)
            else None
        )
        if adaptive_text is not None:
            draft = _feedback_draft(
                text=adaptive_text,
                hint_level=next_hint_level,
                key_point_id=next_key_point.id,
                analysis=analysis,
            )
        elif concept_card:
            draft = _concept_card_draft(
                analysis=analysis,
                key_point=next_key_point,
                concept_card=concept_card,
            )
        else:
            draft = self.provider.generate_hint(
                problem_analysis=analysis,
                key_point=next_key_point,
                student_profile={},
                context=AIEngineContext(child_id=session.child_id, session_id=session.session_id),
            )
        message, leakage_result = _safe_message(
            draft_text=draft.text,
            answer_unlocked=False,
            expected_answer=expected_answer_for_leakage(analysis),
            hint_level=draft.hint_level or next_hint_level,
        )
        released_key_point_ids = _merge_ids(
            session.released_key_point_ids,
            [next_key_point.id],
        )
        mastered_key_point_ids = _merge_ids(
            session.mastered_key_point_ids,
            evaluation.mastered_key_point_ids,
        )
        updated = self.store.update_session(
            session.session_id,
            phase=LearningPhase.WAIT_CHILD_ATTEMPT,
            attempt_count=session.attempt_count + 1,
            hint_level=next_hint_level,
            answer_unlocked=False,
            last_misconception=evaluation.misconception_tag or session.last_misconception,
            current_prompt=message,
            current_key_point_id=next_key_point.id,
            released_key_point_ids=released_key_point_ids,
            mastered_key_point_ids=mastered_key_point_ids,
        )
        self.store.append_event(
            session_id=session.session_id,
            child_id=session.child_id,
            event_type="key_point.released",
            payload={
                "key_point_id": next_key_point.id,
                "release_stage": next_key_point.release_stage,
                "skill_ids": analysis.skill_ids,
                "misconception_id": evaluation.misconception_tag,
                "concept_card_ids": analysis.concept_card_ids,
                "prompt": message,
            },
            deeptutor_trace_id=draft.trace_id,
            leakage_check_result=leakage_result.model_dump(mode="json"),
        )
        if concept_card:
            self.store.append_event(
                session_id=session.session_id,
                child_id=session.child_id,
                event_type="concept_card.released",
                payload={
                    "card_id": concept_card.card_id,
                    "skill_id": concept_card.skill_id,
                    "key_point_id": next_key_point.id,
                    "misconception_id": evaluation.misconception_tag,
                    "title": concept_card.title,
                },
                deeptutor_trace_id=draft.trace_id,
                leakage_check_result=leakage_result.model_dump(mode="json"),
            )
        if evaluation.misconception_tag:
            self.store.record_wrong_question(
                session_id=session.session_id,
                child_id=session.child_id,
                normalized_question=session.normalized_question,
                knowledge_point=session.knowledge_point,
                mistake_summary=evaluation.evidence,
                last_misconception=evaluation.misconception_tag,
                highest_hint_level=updated.hint_level,
                explanation_unlocked=False,
            )
        _record_ai_call(self.store, child_id=session.child_id, session_id=session.session_id, draft=draft)
        return KernelAttemptResult(
            correct=False,
            partially_correct=evaluation.partially_correct,
            phase=updated.phase.value,
            hint_level=updated.hint_level,
            message=updated.current_prompt,
            answer_unlocked=updated.answer_unlocked,
            misconception_tag=evaluation.misconception_tag,
            matched_key_point_id=evaluation.matched_key_point_id,
            next_key_point_id=(
                next_key_point.id
                if next_key_point.id != session.current_key_point_id
                else None
            ),
            teaching_progress=build_teaching_progress(updated),
        )

    def _complete_with_practice(
        self,
        session_id: str,
        analysis: ProblemAnalysis,
        evaluation: AttemptEvaluation,
        child_answer: str,
    ) -> KernelAttemptResult:
        session = self.store.require_session(session_id)
        practice_items = self.provider.generate_similar_practice(
            problem_analysis=analysis,
            misconception=session.last_misconception,
            limit=3,
            context=AIEngineContext(child_id=session.child_id, session_id=session.session_id),
        )
        practice_items = _remove_original_question_from_practice(
            practice_items,
            original_question=session.question_text,
        )
        message = _similar_practice_message(practice_items)
        updated = self.store.update_session(
            session.session_id,
            phase=LearningPhase.SIMILAR_PRACTICE,
            attempt_count=session.attempt_count + 1,
            current_prompt=message,
            current_key_point_id=None,
            mastered_key_point_ids=_merge_ids(
                session.mastered_key_point_ids,
                evaluation.mastered_key_point_ids,
            ),
            completed_at=utc_now(),
        )
        self.store.append_event(
            session_id=session.session_id,
            child_id=session.child_id,
            event_type="practice.generated",
            payload={
                "items": [item.model_dump(mode="json") for item in practice_items],
                "child_answer": child_answer,
                "skill_ids": analysis.skill_ids,
                "misconception_id": session.last_misconception,
                "concept_card_ids": analysis.concept_card_ids,
            },
        )
        return KernelAttemptResult(
            correct=True,
            partially_correct=False,
            phase=updated.phase.value,
            hint_level=updated.hint_level,
            message=updated.current_prompt,
            answer_unlocked=updated.answer_unlocked,
            misconception_tag=None,
            matched_key_point_id=evaluation.matched_key_point_id,
            next_key_point_id=None,
            teaching_progress=build_teaching_progress(updated),
            practice_items=practice_items,
        )

    def _respond_inside_practice_phase(
        self,
        session,
        child_answer: str,
    ) -> KernelAttemptResult:
        self.store.append_event(
            session_id=session.session_id,
            child_id=session.child_id,
            event_type="practice.followup_requested",
            payload={"child_answer": child_answer},
        )
        message = _practice_phase_message(session.current_prompt)
        updated = self.store.update_session(
            session.session_id,
            phase=LearningPhase.SIMILAR_PRACTICE,
            current_prompt=message,
        )
        return KernelAttemptResult(
            correct=False,
            partially_correct=False,
            phase=updated.phase.value,
            hint_level=updated.hint_level,
            message=updated.current_prompt,
            answer_unlocked=updated.answer_unlocked,
            misconception_tag=updated.last_misconception,
            matched_key_point_id=None,
            next_key_point_id=None,
            teaching_progress=build_teaching_progress(updated),
            practice_items=[],
        )


def _evaluate_attempt(
    *,
    analysis: ProblemAnalysis,
    child_answer: str,
    current_key_point_id: str | None,
) -> AttemptEvaluation:
    from songguo.backend.services.learning.math_structuring import evaluate_attempt_by_key_points

    return evaluate_attempt_by_key_points(
        analysis,
        child_answer=child_answer,
        current_key_point_id=current_key_point_id,
    )


def _safe_message(
    *,
    draft_text: str,
    answer_unlocked: bool,
    expected_answer: str | None,
    hint_level: int,
):
    verdict = check_answer_leakage(
        draft_text=draft_text,
        answer_unlocked=answer_unlocked,
        expected_answer=expected_answer,
        hint_level=hint_level,
        draft_hint_level=hint_level,
    )
    return (draft_text if verdict.action == LeakageAction.ALLOW else verdict.safe_text), verdict


def _record_ai_call(
    store: InMemoryLearningStore,
    *,
    child_id: str,
    session_id: str,
    draft: object,
) -> None:
    metadata = getattr(draft, "metadata", {}) or {}
    text = str(getattr(draft, "text", "") or "")
    store.record_ai_call(
        child_id=child_id,
        session_id=session_id,
        provider=str(metadata.get("provider") or "runtime_kernel"),
        model=str(metadata.get("model") or "deterministic"),
        operation=str(getattr(draft, "action", "unknown") or "unknown"),
        token_estimate=max(1, len(text) // 4) if text.strip() else 0,
        status="success",
    )


def _feedback_draft(
    *,
    text: str,
    hint_level: int,
    key_point_id: str,
    analysis: ProblemAnalysis,
):
    from songguo.backend.services.learning.deeptutor_adapter import TeachingDraft

    return TeachingDraft(
        action="adaptive_key_point_feedback",
        hint_level=hint_level,
        exposes_final_answer=False,
        text=text,
        metadata={
            "provider": "runtime_kernel",
            "model": "adaptive_feedback_v0.1",
            "problem_type": analysis.problem_type,
            "key_point_id": key_point_id,
            "skill_ids": analysis.skill_ids,
            "misconception_ids": analysis.misconception_ids,
        },
    )


def _should_release_concept_card(
    previous_attempt_count: int,
    evaluation: AttemptEvaluation,
) -> bool:
    return (
        previous_attempt_count >= 1
        and not evaluation.correct
        and not evaluation.partially_correct
        and evaluation.next_action == "stay_on_current_key_point"
    )


def _concept_card_for_analysis(analysis: ProblemAnalysis) -> ConceptCard | None:
    preferred_skill_id = f"math_{analysis.problem_type}"
    if preferred_skill_id in analysis.skill_ids:
        try:
            return DEFAULT_MATH_ASSET_LIBRARY.require_concept_card_for_skill(
                preferred_skill_id
            )
        except KeyError:
            pass
    for card_id in analysis.concept_card_ids:
        try:
            return DEFAULT_MATH_ASSET_LIBRARY.require_concept_card(card_id)
        except KeyError:
            continue
    for skill_id in analysis.skill_ids:
        try:
            return DEFAULT_MATH_ASSET_LIBRARY.require_concept_card_for_skill(skill_id)
        except KeyError:
            continue
    return None


def _concept_card_draft(
    *,
    analysis: ProblemAnalysis,
    key_point,
    concept_card: ConceptCard,
):
    from songguo.backend.services.learning.deeptutor_adapter import TeachingDraft

    return TeachingDraft(
        action="concept_card_hint",
        hint_level=_hint_level_from_stage_name(key_point.release_stage),
        exposes_final_answer=False,
        text=(
            f"方法卡：{concept_card.title}\n"
            f"{concept_card.concept_explanation}\n"
            f"先看一个更简单的例子：{concept_card.simple_example}\n"
            f"常见卡点：{concept_card.common_mistake}\n"
            f"回到这道题：{key_point.child_prompt}"
        ),
        metadata={
            "provider": "teaching_asset_library",
            "model": concept_card.version,
            "problem_type": analysis.problem_type,
            "key_point_id": key_point.id,
            "skill_ids": analysis.skill_ids,
            "misconception_ids": analysis.misconception_ids,
            "concept_card_id": concept_card.card_id,
        },
    )


def _hint_level_from_stage_name(stage: str) -> int:
    if stage.endswith("_1"):
        return 1
    if stage.endswith("_2"):
        return 2
    if stage.endswith("_3"):
        return 3
    if stage.endswith("_4"):
        return 4
    return 1


def _normalize_question(question_text: str) -> str:
    return " ".join(question_text.strip().replace("×", "x").split())


def _adaptive_key_point_feedback(
    *,
    analysis: ProblemAnalysis,
    child_answer: str,
    current_key_point_id: str | None,
    next_key_point_id: str,
) -> str | None:
    if analysis.problem_type != "capacity_round_up":
        return None
    if current_key_point_id not in {"kp_capacity_check", "kp_round_up"}:
        return None

    total = _step_result_number(analysis, "step_total_people")
    capacity = _condition_number(analysis, "capacity")
    if total is None or capacity is None:
        return None

    quotient, remainder = divmod(total, capacity)
    answer_number = _first_int(child_answer)
    if answer_number is None:
        return (
            f"我们现在只看限载这一步。已经有{total}人，每辆最多{capacity}人。"
            f"先算一算：{quotient}辆车最多能坐多少人？"
        )

    seats = answer_number * capacity
    if seats < total:
        remaining = total - seats
        if answer_number == quotient and remainder:
            return (
                f"{answer_number}辆最多坐{seats}人，还剩{remaining}人。"
                "剩下的人也要坐车，所以还要不要再加一辆？"
            )
        return (
            f"{answer_number}辆最多坐{seats}人，还差{remaining}人没有坐上。"
            f"我们再往上试一试：{answer_number + 1}辆最多能坐多少人？"
        )
    if seats == total:
        return "这个数量刚好能坐下。再看题目问的是“至少”，能坐下又不浪费太多，就是我们要找的方向。"
    return (
        f"{answer_number}辆能坐{seats}人，已经能坐下{total}人。"
        "不过题目问“至少”，我们还要想想能不能少一辆也坐下。"
    )


def _condition_number(analysis: ProblemAnalysis, condition_id: str) -> int | None:
    for condition in analysis.conditions:
        if condition.id != condition_id:
            continue
        if isinstance(condition.value, int):
            return condition.value
        if isinstance(condition.value, float):
            return int(condition.value)
        return _first_int(str(condition.value or condition.text))
    return None


def _step_result_number(analysis: ProblemAnalysis, step_id: str) -> int | None:
    for step in analysis.solution_steps:
        if step.id == step_id:
            return _first_int(step.result)
    return None


def _first_int(text: str) -> int | None:
    match = re.search(r"-?\d+", str(text or ""))
    return int(match.group(0)) if match else None


def _remove_original_question_from_practice(
    items: list[PracticeItem],
    *,
    original_question: str,
) -> list[PracticeItem]:
    normalized_original = _normalize_question(original_question)
    return [
        item
        for item in items
        if _normalize_question(item.question) != normalized_original
    ]


def _practice_phase_message(current_prompt: str) -> str:
    if current_prompt.strip():
        return "这道题已经完成。现在先做下面的同类题；如果要重新开始，请新建一道题。\n" + current_prompt
    return "这道题已经完成。现在进入同类题巩固；如果要重新开始，请新建一道题。"


def _merge_ids(existing: list[str], incoming: list[str]) -> list[str]:
    result = list(existing)
    for item in incoming:
        if item and item not in result:
            result.append(item)
    return result


def _similar_practice_message(items: list[PracticeItem]) -> str:
    if not items:
        return "你已经找到方法了。下一步可以练一道同类题。"
    lines = ["你已经找到方法了。我们再练习 1-3 道同类题，确认真的掌握："]
    lines.extend(f"{index}. {item.question}" for index, item in enumerate(items, start=1))
    return "\n".join(lines)
