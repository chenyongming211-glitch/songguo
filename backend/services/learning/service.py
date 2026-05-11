from __future__ import annotations

import os
from pathlib import Path
import re

from pydantic import BaseModel, Field

from songguo.backend.services.learning.answer_matching import answers_match
from songguo.backend.services.learning.ai_engine import (
    AIEngineContext,
    AIEngineProvider,
    DeepSeekProvider,
    DeepTutorProvider,
    DeterministicFallbackProvider,
    OllamaProvider,
    ProviderChain,
)
from songguo.backend.services.learning.deeptutor_adapter import DeepTutorLearningAdapter
from songguo.backend.services.learning.deeptutor_provider import OrchestratorDraftProvider
from songguo.backend.services.learning.context_pack import build_student_context_pack
from songguo.backend.services.learning.deposit import save_learning_deposit_from_llm_output
from songguo.backend.services.learning.input_safety import check_learning_input
from songguo.backend.services.learning.leakage_checker import (
    LeakageAction,
    check_answer_leakage,
    detect_answer_leakage,
)
from songguo.backend.services.learning.llm_session_runner import LLMSessionRunner
from songguo.backend.services.learning.math_structuring import (
    AttemptEvaluation,
    LLMMathStructurer,
    MathProblemStructuringGateway,
    ProblemAnalysis,
    expected_answer_for_leakage,
)
from songguo.backend.services.learning.models import LearningPhase, TeachingProgress, utc_now
from songguo.backend.services.learning.progress import build_teaching_progress
from songguo.backend.services.learning.practice_recommender import (
    PracticeItem,
    build_similar_practice_items,
)
from songguo.backend.services.learning.runtime_config import resolve_agent_runtime
from songguo.backend.services.learning.runtime_kernel import RuntimeKernel
from songguo.backend.services.learning.state_machine import (
    AttemptOutcome,
    apply_attempt_result,
    initialize_session_state,
)
from songguo.backend.services.learning.store import (
    InMemoryLearningStore,
    PostgresLearningStore,
    SQLiteLearningStore,
)
from songguo.backend.services.learning.submission_graph import (
    LearningSubmissionGraph,
    LearningSubmissionGraphResult,
)
from songguo.backend.services.learning.submission_models import (
    LearningSubmissionSnapshot,
    SourceType,
)
from songguo.backend.services.learning.tutor_graph.math_mistake_graph import (
    MathMistakeTutorGraph,
)


class CreateLearningSessionResult(BaseModel):
    session_id: str
    question_text: str
    subject: str = "math"
    grade: int = 3
    phase: str
    hint_level: int
    message: str
    answer_unlocked: bool
    teaching_progress: TeachingProgress


class SubmitAttemptResult(BaseModel):
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


class SubmissionTutorAttemptResult(BaseModel):
    attempt: SubmitAttemptResult
    submission: LearningSubmissionGraphResult


class LearningService:
    def __init__(
        self,
        *,
        store: InMemoryLearningStore,
        adapter: DeepTutorLearningAdapter | None = None,
        math_gateway: MathProblemStructuringGateway | None = None,
        ai_provider: AIEngineProvider | None = None,
        session_runner: LLMSessionRunner | None = None,
        tutor_graph: MathMistakeTutorGraph | None = None,
        agent_runtime: str | None = None,
    ) -> None:
        self.store = store
        self._adapter_explicit = adapter is not None
        self._math_gateway_explicit = math_gateway is not None
        self.adapter = adapter or DeepTutorLearningAdapter()
        self.math_gateway = math_gateway or MathProblemStructuringGateway()
        self.ai_provider = ai_provider
        self.session_runner = session_runner
        self.agent_runtime = resolve_agent_runtime(
            {"SONGGUO_AGENT_RUNTIME": agent_runtime} if agent_runtime else None,
            langgraph_available=True if agent_runtime == "langgraph" else None,
            default="llm" if session_runner is not None else "kernel",
        )
        self.tutor_graph = tutor_graph
        if self.tutor_graph is None and self.agent_runtime == "langgraph":
            self.tutor_graph = MathMistakeTutorGraph(
                store=store,
                session_runner=session_runner,
                math_gateway=self.math_gateway,
            )
        self.submission_graph = LearningSubmissionGraph(
            store=store,
            session_runner=session_runner,
            math_gateway=self.math_gateway,
            tutor_graph=self.tutor_graph,
        )
        self.runtime_kernel = (
            RuntimeKernel(store=store, provider=ai_provider) if ai_provider is not None else None
        )

    def create_submission(
        self,
        *,
        child_id: str,
        subject: str,
        grade: int,
        source_type: SourceType | str,
        raw_text: str,
    ) -> LearningSubmissionGraphResult:
        return self.submission_graph.start(
            child_id=child_id,
            subject=subject,
            grade=grade,
            source_type=str(source_type),
            raw_text=raw_text,
        )

    def get_submission_snapshot(self, submission_id: str) -> LearningSubmissionSnapshot:
        return self.store.get_submission_snapshot(submission_id)

    def confirm_submission(self, submission_id: str) -> LearningSubmissionGraphResult:
        return self.submission_graph.confirm(submission_id)

    def start_next_tutor_item(self, submission_id: str) -> LearningSubmissionGraphResult:
        return self.submission_graph.start_next_tutor_item(submission_id)

    def submit_submission_tutor_attempt(
        self,
        submission_id: str,
        *,
        child_answer: str,
    ) -> SubmissionTutorAttemptResult:
        active = self.store.get_active_tutor_item(submission_id)
        if active is None or not active.tutor_session_id:
            raise KeyError("Active tutor item not found")
        attempt = self.submit_attempt(active.tutor_session_id, child_answer=child_answer)
        if attempt.correct:
            submission = self.submission_graph.complete_active_tutor_item(
                submission_id=submission_id,
                tutor_session_id=active.tutor_session_id,
            )
        else:
            submission = self.submission_graph.start_next_tutor_item(submission_id)
        return SubmissionTutorAttemptResult(attempt=attempt, submission=submission)

    def create_session(
        self,
        *,
        child_id: str,
        subject: str,
        grade: int,
        question_text: str,
    ) -> CreateLearningSessionResult:
        input_verdict = check_learning_input(question_text)
        if not input_verdict.allowed:
            raise ValueError(input_verdict.message)

        normalized_question = _normalize_question(question_text)
        if self.tutor_graph is not None and (subject or "").lower() == "math":
            result = self.tutor_graph.start(
                child_id=child_id,
                subject=subject,
                grade=grade,
                question_text=question_text,
            )
            return CreateLearningSessionResult.model_validate(result.model_dump(mode="json"))
        if self.session_runner is not None:
            return self._create_llm_session(
                child_id=child_id,
                subject=subject,
                grade=grade,
                question_text=question_text,
                normalized_question=normalized_question,
            )
        if self.runtime_kernel is not None and (subject or "").lower() == "math":
            from songguo.backend.services.learning.ai_engine import AIEngineContext

            result = self.runtime_kernel.create_math_session(
                child_id=child_id,
                grade=grade,
                question_text=question_text,
                context=AIEngineContext(child_id=child_id, session_id=None),
            )
            return CreateLearningSessionResult.model_validate(
                result.model_dump(mode="json")
            )
        analysis = self._analyze_math_problem(
            question_text=question_text,
            grade=grade,
            subject=subject,
        )
        if analysis is not None:
            return self._create_structured_math_session(
                child_id=child_id,
                subject=subject,
                grade=grade,
                question_text=question_text,
                normalized_question=normalized_question,
                analysis=analysis,
            )

        knowledge_point = _classify_knowledge_point(
            normalized_question,
            grade,
            subject=subject,
        )
        draft = self.adapter.generate_hint(
            question_text=question_text,
            grade=grade,
            knowledge_point=knowledge_point,
            hint_level=1,
        )
        verdict = check_answer_leakage(
            draft_text=draft.text,
            answer_unlocked=False,
            expected_answer=_expected_answer(normalized_question),
            hint_level=1,
            draft_hint_level=draft.hint_level,
        )
        message = draft.text if verdict.action == LeakageAction.ALLOW else verdict.safe_text
        session = self.store.create_session(
            child_id=child_id,
            subject=subject,
            grade=grade,
            question_text=question_text,
            normalized_question=normalized_question,
            knowledge_point=knowledge_point,
            current_prompt=message,
        )
        if verdict.action == LeakageAction.BLOCK:
            self.store.record_safety_event(
                session_id=session.session_id,
                child_id=child_id,
                event_type="safety.blocked",
                input_text=question_text,
                blocked_text=draft.text,
                reason=verdict.reason,
            )
        self._record_ai_call(
            child_id=child_id,
            session_id=session.session_id,
            draft=draft,
        )
        self.store.append_event(
            session_id=session.session_id,
            child_id=child_id,
            event_type="question.classified",
            payload={
                "knowledge_point": knowledge_point,
                "expected_answer_known": _expected_answer(normalized_question) is not None,
            },
        )
        self.store.append_event(
            session_id=session.session_id,
            child_id=child_id,
            event_type="hint.generated",
            payload={"hint_level": 1, "message": message},
            deeptutor_trace_id=draft.trace_id,
            leakage_check_result=verdict.model_dump(mode="json"),
        )
        return CreateLearningSessionResult(
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

    def submit_attempt(self, session_id: str, *, child_answer: str) -> SubmitAttemptResult:
        session = self.store.require_session(session_id)
        input_verdict = check_learning_input(child_answer)
        if not input_verdict.allowed:
            self.store.record_safety_event(
                session_id=session.session_id,
                child_id=session.child_id,
                event_type="safety.blocked",
                input_text=child_answer,
                blocked_text=child_answer,
                reason=input_verdict.reason,
            )
            raise ValueError(input_verdict.message)

        if session.runner_mode == "langgraph" and self.tutor_graph is not None:
            result = self.tutor_graph.submit_attempt(
                session_id=session_id,
                child_answer=child_answer,
            )
            return SubmitAttemptResult.model_validate(result.model_dump(mode="json"))

        if session.runner_mode == "llm" and self.session_runner is not None:
            return self._submit_llm_attempt(session_id=session_id, child_answer=child_answer)

        if session.problem_analysis:
            if self.runtime_kernel is not None:
                result = self.runtime_kernel.submit_math_attempt(
                    session_id,
                    child_answer=child_answer,
                )
                return SubmitAttemptResult.model_validate(
                    result.model_dump(mode="json")
                )
            return self._submit_structured_math_attempt(
                session_id=session_id,
                child_answer=child_answer,
            )

        expected_answer = _expected_answer(session.normalized_question)
        outcome, misconception = _evaluate_attempt(
            question_text=session.normalized_question,
            child_answer=child_answer,
            expected_answer=expected_answer,
        )
        transition = apply_attempt_result(
            initialize_session_state(
                phase=session.phase,
                hint_level=session.hint_level,
                attempt_count=session.attempt_count,
                answer_unlocked=session.answer_unlocked,
                last_misconception=session.last_misconception,
            ),
            outcome,
            misconception=misconception,
        )

        correct = outcome == AttemptOutcome.CORRECT
        verdict = None
        practice_items: list[PracticeItem] = []
        if correct:
            practice_items = build_similar_practice_items(
                knowledge_point=session.knowledge_point,
                misconception_tag=session.last_misconception,
                limit=3,
                source_question=session.question_text,
            )
            draft = self.adapter.generate_similar_practice(
                knowledge_point=session.knowledge_point,
                misconception_tag=session.last_misconception,
                difficulty=1,
            )
            message = _similar_practice_message(practice_items, fallback=draft.text)
        else:
            if transition.state.answer_unlocked:
                draft = self.adapter.generate_explanation(
                    question_text=session.question_text,
                    answer_unlocked=True,
                )
                message = draft.text
            else:
                draft = self.adapter.generate_hint(
                    question_text=session.question_text,
                    grade=session.grade,
                    knowledge_point=session.knowledge_point,
                    hint_level=transition.state.hint_level,
                    misconception_tag=misconception,
                )
                verdict = check_answer_leakage(
                    draft_text=draft.text,
                    answer_unlocked=transition.state.answer_unlocked,
                    expected_answer=expected_answer,
                    hint_level=transition.state.hint_level,
                    draft_hint_level=draft.hint_level,
                )
                message = draft.text if verdict.action == LeakageAction.ALLOW else verdict.safe_text
                if verdict.action == LeakageAction.BLOCK:
                    self.store.record_safety_event(
                        session_id=session.session_id,
                        child_id=session.child_id,
                        event_type="safety.blocked",
                        input_text=child_answer,
                        blocked_text=draft.text,
                        reason=verdict.reason,
                    )
            self.store.record_wrong_question(
                session_id=session.session_id,
                child_id=session.child_id,
                normalized_question=session.normalized_question,
                knowledge_point=session.knowledge_point,
                mistake_summary=_mistake_summary(misconception),
                last_misconception=misconception,
                highest_hint_level=transition.state.hint_level,
                explanation_unlocked=transition.state.answer_unlocked,
            )

        updated = self.store.update_session(
            session.session_id,
            phase=transition.state.phase,
            hint_level=transition.state.hint_level,
            attempt_count=transition.state.attempt_count,
            answer_unlocked=transition.state.answer_unlocked,
            last_misconception=transition.state.last_misconception,
            current_prompt=message,
        )
        for event_type in transition.event_types:
            self.store.append_event(
                session_id=session.session_id,
                child_id=session.child_id,
                event_type=event_type,
                payload={
                    "child_answer": child_answer,
                    "outcome": outcome.value,
                    "hint_level": updated.hint_level,
                    "misconception": misconception,
                },
                deeptutor_trace_id=draft.trace_id,
            )
        self._record_ai_call(
            child_id=session.child_id,
            session_id=session.session_id,
            draft=draft,
        )

        return SubmitAttemptResult(
            correct=correct,
            partially_correct=False,
            phase=updated.phase.value,
            hint_level=updated.hint_level,
            message=message,
            answer_unlocked=updated.answer_unlocked,
            misconception_tag=misconception,
            teaching_progress=build_teaching_progress(updated),
            practice_items=practice_items,
        )

    def _analyze_math_problem(
        self,
        *,
        question_text: str,
        grade: int,
        subject: str,
    ) -> ProblemAnalysis | None:
        if (subject or "").lower() != "math":
            return None
        if self._adapter_explicit and not self._math_gateway_explicit:
            return None
        try:
            return self.math_gateway.analyze(
                question_text=question_text,
                grade=grade,
                subject=subject,
            )
        except ValueError:
            return None

    def _create_structured_math_session(
        self,
        *,
        child_id: str,
        subject: str,
        grade: int,
        question_text: str,
        normalized_question: str,
        analysis: ProblemAnalysis,
    ) -> CreateLearningSessionResult:
        first_key_point = analysis.first_key_point
        expected_answer = expected_answer_for_leakage(analysis)
        draft = _key_point_draft(
            text=first_key_point.child_prompt,
            hint_level=1,
            key_point_id=first_key_point.id,
            analysis=analysis,
        )
        verdict = check_answer_leakage(
            draft_text=draft.text,
            answer_unlocked=False,
            expected_answer=expected_answer,
            hint_level=1,
            draft_hint_level=draft.hint_level,
        )
        message = draft.text if verdict.action == LeakageAction.ALLOW else verdict.safe_text
        session = self.store.create_session(
            child_id=child_id,
            subject=subject,
            grade=grade,
            question_text=question_text,
            normalized_question=normalized_question,
            knowledge_point=analysis.knowledge_point,
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
                "source": analysis.source,
                "confidence": analysis.confidence,
                "key_point_count": len(analysis.key_points),
            },
        )
        self.store.append_event(
            session_id=session.session_id,
            child_id=child_id,
            event_type="key_point.released",
            payload={
                "key_point_id": first_key_point.id,
                "release_stage": first_key_point.release_stage,
                "hint_level": 1,
                "prompt": message,
            },
            deeptutor_trace_id=draft.trace_id,
            leakage_check_result=verdict.model_dump(mode="json"),
        )
        if verdict.action == LeakageAction.BLOCK:
            self.store.record_safety_event(
                session_id=session.session_id,
                child_id=child_id,
                event_type="safety.blocked",
                input_text=question_text,
                blocked_text=draft.text,
                reason=verdict.reason,
            )
        self._record_ai_call(
            child_id=child_id,
            session_id=session.session_id,
            draft=draft,
        )
        return CreateLearningSessionResult(
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

    def _create_llm_session(
        self,
        *,
        child_id: str,
        subject: str,
        grade: int,
        question_text: str,
        normalized_question: str,
    ) -> CreateLearningSessionResult:
        context_pack = build_student_context_pack(
            self.store,
            child_id=child_id,
            grade=grade,
            current_question=question_text,
        )
        output = self.session_runner.start(
            child_id=child_id,
            grade=grade,
            question_text=question_text,
            context_pack=context_pack,
        )
        raw_message = output.child_message
        message = self._safe_llm_child_message(
            raw_message,
            answer_unlocked=output.structured_state.answer_unlocked,
            expected_answer=_expected_answer(normalized_question),
            hint_level=1,
            child_id=child_id,
            session_id="pending",
        )
        knowledge_point = _normalize_llm_knowledge_point(
            output.learning_deposit_delta.knowledge_point
        )
        session = self.store.create_session(
            child_id=child_id,
            subject=subject,
            grade=grade,
            question_text=question_text,
            normalized_question=normalized_question,
            knowledge_point=knowledge_point,
            current_prompt=message,
            current_key_point_id=output.structured_state.current_key_point,
            runner_mode="llm",
        )
        if message != raw_message:
            safety_verdict = check_answer_leakage(
                draft_text=raw_message,
                answer_unlocked=output.structured_state.answer_unlocked,
                expected_answer=_expected_answer(normalized_question),
                hint_level=1,
                draft_hint_level=1,
            )
            self.store.record_safety_event(
                session_id=session.session_id,
                child_id=child_id,
                event_type="safety.blocked",
                blocked_text=raw_message,
                reason=safety_verdict.reason,
            )
        self._record_teaching_quality_if_needed(
            session_id=session.session_id,
            child_id=child_id,
            message=raw_message,
            answer_unlocked=output.structured_state.answer_unlocked,
            expected_answer=_expected_answer(normalized_question),
        )
        self.store.append_message(
            session_id=session.session_id,
            child_id=child_id,
            role="user",
            content=question_text,
        )
        self.store.append_message(
            session_id=session.session_id,
            child_id=child_id,
            role="assistant",
            content=message,
            metadata={"structured_state": output.structured_state.model_dump(mode="json")},
        )
        self.store.append_event(
            session_id=session.session_id,
            child_id=child_id,
            event_type="message.created",
            payload={"role": "assistant", "message": message, "runner_mode": "llm"},
        )
        self.store.append_event(
            session_id=session.session_id,
            child_id=child_id,
            event_type="hint.generated",
            payload={
                "hint_level": 1,
                "message": message,
                "current_key_point": output.structured_state.current_key_point,
            },
        )
        self.store.record_ai_call(
            child_id=child_id,
            session_id=session.session_id,
            provider=output.provider,
            model=output.model,
            operation="llm_session.start",
            token_estimate=output.token_estimate,
            status=output.status,
        )
        return CreateLearningSessionResult(
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

    def _submit_llm_attempt(
        self,
        *,
        session_id: str,
        child_answer: str,
    ) -> SubmitAttemptResult:
        session = self.store.require_session(session_id)
        self.store.append_message(
            session_id=session.session_id,
            child_id=session.child_id,
            role="user",
            content=child_answer,
        )
        self.store.append_event(
            session_id=session.session_id,
            child_id=session.child_id,
            event_type="child.attempt_submitted",
            payload={
                "child_answer": child_answer,
                "runner_mode": "llm",
            },
        )
        context_pack = build_student_context_pack(
            self.store,
            child_id=session.child_id,
            grade=session.grade,
            current_question=session.question_text,
        )
        output = self.session_runner.run(
            child_id=session.child_id,
            grade=session.grade,
            context_pack=context_pack,
            messages=self.store.list_messages(session.session_id),
        )
        expected_answer = self._expected_answer_for_session(session)
        backend_answer_is_correct = _answer_matches_expected(child_answer, expected_answer)
        next_attempt_count = session.attempt_count + 1
        next_hint_level = min(5, session.hint_level + 1)
        is_done = (
            output.structured_state.should_end_session
            or output.structured_state.phase == LearningPhase.SIMILAR_PRACTICE.value
            or backend_answer_is_correct
        )
        phase = LearningPhase.SIMILAR_PRACTICE if is_done else LearningPhase.WAIT_CHILD_ATTEMPT
        answer_unlocked = bool(output.structured_state.answer_unlocked and is_done)
        message = self._safe_llm_child_message(
            output.child_message,
            answer_unlocked=answer_unlocked,
            expected_answer=expected_answer,
            hint_level=next_hint_level,
            child_id=session.child_id,
            session_id=session.session_id,
        )
        practice_items = list(output.practice_items[:3])
        if is_done and not practice_items:
            practice_items = build_similar_practice_items(
                knowledge_point=_normalize_llm_knowledge_point(
                    output.learning_deposit_delta.knowledge_point
                    or session.knowledge_point
                ),
                misconception_tag=(
                    output.learning_deposit_delta.main_misconception
                    or output.structured_state.main_misconception
                    or session.last_misconception
                ),
                limit=3,
                source_question=session.question_text,
            )
        if is_done and practice_items and "1." not in message:
            message = _similar_practice_message(practice_items, fallback=message)

        updated = self.store.update_session(
            session.session_id,
            phase=phase,
            attempt_count=next_attempt_count,
            hint_level=next_hint_level,
            answer_unlocked=answer_unlocked,
            last_misconception=(
                output.learning_deposit_delta.main_misconception
                or output.structured_state.main_misconception
                or session.last_misconception
            ),
            current_prompt=message,
            current_key_point_id=output.structured_state.current_key_point,
            completed_at=utc_now() if is_done else session.completed_at,
        )
        self.store.append_message(
            session_id=session.session_id,
            child_id=session.child_id,
            role="assistant",
            content=message,
            metadata={"structured_state": output.structured_state.model_dump(mode="json")},
        )
        self.store.append_event(
            session_id=session.session_id,
            child_id=session.child_id,
            event_type="assistant.message_created",
            payload={
                "message": message,
                "runner_mode": "llm",
                "structured_state": output.structured_state.model_dump(mode="json"),
            },
        )
        self._record_teaching_quality_if_needed(
            session_id=session.session_id,
            child_id=session.child_id,
            message=output.child_message,
            answer_unlocked=answer_unlocked,
            expected_answer=expected_answer,
        )
        self.store.append_event(
            session_id=session.session_id,
            child_id=session.child_id,
            event_type="attempt.evaluated",
            payload={
                "correct": is_done,
                "misconception_tag": updated.last_misconception,
                "evidence": output.learning_deposit_delta.evidence,
                "runner_mode": "llm",
            },
        )
        if updated.last_misconception:
            self.store.record_wrong_question(
                session_id=session.session_id,
                child_id=session.child_id,
                normalized_question=session.normalized_question,
                knowledge_point=_normalize_llm_knowledge_point(
                    output.learning_deposit_delta.knowledge_point
                    or session.knowledge_point
                ),
                mistake_summary=(
                    output.learning_deposit_delta.evidence
                    or _mistake_summary(updated.last_misconception)
                ),
                last_misconception=updated.last_misconception,
                highest_hint_level=updated.hint_level,
                explanation_unlocked=answer_unlocked,
                practice_completed=is_done,
            )
        if is_done:
            self.store.append_event(
                session_id=session.session_id,
                child_id=session.child_id,
                event_type="practice.generated",
                payload={
                    "items": [item.model_dump(mode="json") for item in practice_items],
                    "message": message,
                    "runner_mode": "llm",
                },
            )
            output.practice_items = practice_items
            save_learning_deposit_from_llm_output(
                self.store,
                session_id=session.session_id,
                output=output,
            )
        self.store.record_ai_call(
            child_id=session.child_id,
            session_id=session.session_id,
            provider=output.provider,
            model=output.model,
            operation="llm_session.submit_attempt",
            token_estimate=output.token_estimate,
            status=output.status,
        )
        return SubmitAttemptResult(
            correct=is_done,
            partially_correct=bool(output.structured_state.mastered_key_points) and not is_done,
            phase=updated.phase.value,
            hint_level=updated.hint_level,
            message=updated.current_prompt,
            answer_unlocked=updated.answer_unlocked,
            misconception_tag=updated.last_misconception,
            matched_key_point_id=(
                output.structured_state.mastered_key_points[-1]
                if output.structured_state.mastered_key_points
                else None
            ),
            next_key_point_id=output.structured_state.current_key_point,
            teaching_progress=build_teaching_progress(updated),
            practice_items=practice_items,
        )

    def _expected_answer_for_session(self, session) -> str | None:
        expected = _expected_answer(session.normalized_question)
        if expected:
            return expected
        if session.problem_analysis:
            return expected_answer_for_leakage(
                ProblemAnalysis.model_validate(session.problem_analysis)
            )
        analysis = self._analyze_math_problem(
            question_text=session.question_text,
            grade=session.grade,
            subject=session.subject,
        )
        if analysis is not None:
            return expected_answer_for_leakage(analysis)
        if (session.subject or "").lower() != "math":
            return None
        try:
            fallback_analysis = DeterministicFallbackProvider().structure_math_problem(
                question_text=session.question_text,
                grade=session.grade,
                context=AIEngineContext(child_id=session.child_id, session_id=session.session_id),
            )
        except Exception:
            return None
        return expected_answer_for_leakage(fallback_analysis)

    def _safe_llm_child_message(
        self,
        text: str,
        *,
        answer_unlocked: bool,
        expected_answer: str | None,
        hint_level: int,
        child_id: str,
        session_id: str,
    ) -> str:
        verdict = check_answer_leakage(
            draft_text=text,
            answer_unlocked=answer_unlocked,
            expected_answer=expected_answer,
            hint_level=hint_level,
            draft_hint_level=hint_level,
        )
        if verdict.action == LeakageAction.BLOCK and session_id != "pending":
            self.store.record_safety_event(
                session_id=session_id,
                child_id=child_id,
                event_type="safety.blocked",
                blocked_text=text,
                reason=verdict.reason,
            )
        return text if verdict.action == LeakageAction.ALLOW else verdict.safe_text

    def _record_teaching_quality_if_needed(
        self,
        *,
        session_id: str,
        child_id: str,
        message: str,
        answer_unlocked: bool,
        expected_answer: str | None,
    ) -> None:
        signal = detect_answer_leakage(
            draft_text=message,
            answer_unlocked=answer_unlocked,
            expected_answer=expected_answer,
        )
        if not signal.detected:
            return
        self.store.append_event(
            session_id=session_id,
            child_id=child_id,
            event_type="teaching_quality.direct_answer",
            payload={
                "reason": signal.reason,
                "message": message,
                "expected_answer": expected_answer,
                "blocked": False,
            },
        )

    def _submit_structured_math_attempt(
        self,
        *,
        session_id: str,
        child_answer: str,
    ) -> SubmitAttemptResult:
        session = self.store.require_session(session_id)
        analysis = ProblemAnalysis.model_validate(session.problem_analysis)
        evaluation = self.math_gateway.evaluate_attempt(
            analysis,
            child_answer=child_answer,
            current_key_point_id=session.current_key_point_id,
        )
        mastered_key_point_ids = _merge_ids(
            session.mastered_key_point_ids,
            evaluation.mastered_key_point_ids,
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
            payload=evaluation.model_dump(mode="json"),
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

        next_attempt_count = session.attempt_count + 1
        if evaluation.correct:
            practice_items = build_similar_practice_items(
                knowledge_point=analysis.knowledge_point,
                misconception_tag=session.last_misconception,
                limit=3,
                source_question=session.question_text,
            )
            draft = self.adapter.generate_similar_practice(
                knowledge_point=analysis.knowledge_point,
                misconception_tag=session.last_misconception,
                difficulty=1,
            )
            message = _similar_practice_message(practice_items, fallback=draft.text)
            updated = self.store.update_session(
                session.session_id,
                phase=LearningPhase.SIMILAR_PRACTICE,
                attempt_count=next_attempt_count,
                hint_level=session.hint_level,
                current_prompt=message,
                mastered_key_point_ids=mastered_key_point_ids,
            )
            self.store.append_event(
                session_id=session.session_id,
                child_id=session.child_id,
                event_type="practice.generated",
                payload={
                    "knowledge_point": analysis.knowledge_point,
                    "message": message,
                    "items": [item.model_dump(mode="json") for item in practice_items],
                },
                deeptutor_trace_id=draft.trace_id,
            )
            self._record_ai_call(
                child_id=session.child_id,
                session_id=session.session_id,
                draft=draft,
            )
            return SubmitAttemptResult(
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

        outcome = _structured_evaluation_to_outcome(
            evaluation,
            child_answer=child_answer,
        )
        transition = apply_attempt_result(
            initialize_session_state(
                phase=session.phase,
                hint_level=session.hint_level,
                attempt_count=session.attempt_count,
                answer_unlocked=session.answer_unlocked,
                last_misconception=session.last_misconception,
            ),
            outcome,
            misconception=evaluation.misconception_tag,
        )

        if transition.state.answer_unlocked:
            draft = self.adapter.generate_explanation(
                question_text=session.question_text,
                answer_unlocked=True,
            )
            updated = self.store.update_session(
                session.session_id,
                phase=transition.state.phase,
                attempt_count=transition.state.attempt_count,
                hint_level=transition.state.hint_level,
                answer_unlocked=True,
                last_misconception=evaluation.misconception_tag
                or session.last_misconception,
                current_prompt=draft.text,
                mastered_key_point_ids=mastered_key_point_ids,
            )
            if evaluation.misconception_tag:
                self.store.record_wrong_question(
                    session_id=session.session_id,
                    child_id=session.child_id,
                    normalized_question=session.normalized_question,
                    knowledge_point=session.knowledge_point,
                    mistake_summary=_mistake_summary(evaluation.misconception_tag),
                    last_misconception=evaluation.misconception_tag,
                    highest_hint_level=updated.hint_level,
                    explanation_unlocked=True,
                )
            self.store.append_event(
                session_id=session.session_id,
                child_id=session.child_id,
                event_type="answer.unlocked",
                payload={
                    "child_answer": child_answer,
                    "misconception": evaluation.misconception_tag,
                    "hint_level": updated.hint_level,
                },
                deeptutor_trace_id=draft.trace_id,
            )
            self._record_ai_call(
                child_id=session.child_id,
                session_id=session.session_id,
                draft=draft,
            )
            return SubmitAttemptResult(
                correct=False,
                partially_correct=evaluation.partially_correct,
                phase=updated.phase.value,
                hint_level=updated.hint_level,
                message=updated.current_prompt,
                answer_unlocked=updated.answer_unlocked,
                misconception_tag=evaluation.misconception_tag,
                matched_key_point_id=evaluation.matched_key_point_id,
                next_key_point_id=evaluation.next_key_point_id,
                teaching_progress=build_teaching_progress(updated),
            )

        next_hint_level = transition.state.hint_level
        next_key_point = (
            analysis.get_key_point(evaluation.next_key_point_id)
            if evaluation.next_key_point_id
            else analysis.get_key_point(session.current_key_point_id)
        )
        expected_answer = expected_answer_for_leakage(analysis)
        draft = _key_point_draft(
            text=next_key_point.child_prompt,
            hint_level=next_hint_level,
            key_point_id=next_key_point.id,
            analysis=analysis,
        )
        verdict = check_answer_leakage(
            draft_text=draft.text,
            answer_unlocked=False,
            expected_answer=expected_answer,
            hint_level=next_hint_level,
            draft_hint_level=draft.hint_level,
        )
        message = draft.text if verdict.action == LeakageAction.ALLOW else verdict.safe_text
        released_key_point_ids = _merge_ids(
            session.released_key_point_ids,
            [next_key_point.id],
        )
        updated = self.store.update_session(
            session.session_id,
            phase=LearningPhase.WAIT_CHILD_ATTEMPT,
            attempt_count=transition.state.attempt_count,
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
                "hint_level": updated.hint_level,
                "prompt": message,
            },
            deeptutor_trace_id=draft.trace_id,
            leakage_check_result=verdict.model_dump(mode="json"),
        )
        if evaluation.misconception_tag:
            self.store.record_wrong_question(
                session_id=session.session_id,
                child_id=session.child_id,
                normalized_question=session.normalized_question,
                knowledge_point=session.knowledge_point,
                mistake_summary=_mistake_summary(evaluation.misconception_tag),
                last_misconception=evaluation.misconception_tag,
                highest_hint_level=updated.hint_level,
                explanation_unlocked=False,
            )
        if verdict.action == LeakageAction.BLOCK:
            self.store.record_safety_event(
                session_id=session.session_id,
                child_id=session.child_id,
                event_type="safety.blocked",
                input_text=child_answer,
                blocked_text=draft.text,
                reason=verdict.reason,
            )
        self._record_ai_call(
            child_id=session.child_id,
            session_id=session.session_id,
            draft=draft,
        )
        return SubmitAttemptResult(
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

    def _record_ai_call(
        self,
        *,
        child_id: str,
        session_id: str,
        draft: object,
    ) -> None:
        metadata = getattr(draft, "metadata", {}) or {}
        text = str(getattr(draft, "text", "") or "")
        self.store.record_ai_call(
            child_id=child_id,
            session_id=session_id,
            provider=str(metadata.get("provider") or "deeptutor_adapter"),
            model=str(metadata.get("model") or "deterministic"),
            operation=str(getattr(draft, "action", "unknown") or "unknown"),
            token_estimate=_estimate_tokens(text),
            status="fallback" if metadata.get("fallback_reason") else "success",
        )


_MUL_PATTERN = re.compile(r"(?P<a>\d+)\s*(?:x|×|\*)\s*(?P<b>\d+)", re.IGNORECASE)


def _normalize_question(question_text: str) -> str:
    return " ".join(question_text.strip().replace("×", "x").split())


def _expected_answer(question_text: str) -> str | None:
    match = _MUL_PATTERN.search(question_text)
    if not match:
        return None
    return str(int(match.group("a")) * int(match.group("b")))


def _classify_knowledge_point(question_text: str, grade: int, *, subject: str = "math") -> str:
    normalized_subject = (subject or "math").lower()
    if normalized_subject == "english":
        return "english_sentence_pattern"
    if normalized_subject == "chinese":
        return "chinese_reading_summary"

    match = _MUL_PATTERN.search(question_text)
    if grade == 3 and match:
        a = int(match.group("a"))
        b = int(match.group("b"))
        if (10 <= a <= 99 and 1 <= b <= 9) or (10 <= b <= 99 and 1 <= a <= 9):
            return "two_digit_times_one_digit"
    return "grade_math_unknown"


def _evaluate_attempt(
    *,
    question_text: str,
    child_answer: str,
    expected_answer: str | None,
) -> tuple[AttemptOutcome, str | None]:
    normalized_answer = child_answer.strip()
    if not normalized_answer:
        return AttemptOutcome.EMPTY_OR_OFF_TASK, None
    if expected_answer is not None and normalized_answer == expected_answer:
        return AttemptOutcome.CORRECT, None
    if _looks_like_times_ten_misconception(question_text, normalized_answer):
        return AttemptOutcome.WRONG_KNOWN_MISCONCEPTION, "treated_x5_like_x10"
    return AttemptOutcome.WRONG_UNKNOWN, "unknown_misconception"


def _answer_matches_expected(child_answer: str, expected_answer: str | None) -> bool:
    return answers_match(child_answer, expected_answer)


def _looks_like_times_ten_misconception(question_text: str, child_answer: str) -> bool:
    match = _MUL_PATTERN.search(question_text)
    if not match:
        return False
    a = int(match.group("a"))
    b = int(match.group("b"))
    if 5 not in (a, b):
        return False
    other = b if a == 5 else a
    return child_answer.strip() == str(other * 10)


def _mistake_summary(misconception: str | None) -> str:
    summaries = {
        "treated_x5_like_x10": "Child treated multiplication by 5 like multiplication by 10.",
        "stopped_at_total_count": "Child stopped after calculating the intermediate total.",
        "ignored_remainder_round_up": "Child ignored that a remainder requires one more group.",
        "used_group_count_as_answer": "Child used the number of groups from the question as the answer.",
        "copied_capacity": "Child copied the capacity number instead of solving the target.",
        "unknown_misconception": "Child submitted an incorrect answer that needs follow-up.",
    }
    return summaries.get(misconception or "", "Child needs another guided attempt.")


def _structured_evaluation_to_outcome(
    evaluation: AttemptEvaluation,
    *,
    child_answer: str,
) -> AttemptOutcome:
    if evaluation.correct:
        return AttemptOutcome.CORRECT
    if not child_answer.strip():
        return AttemptOutcome.EMPTY_OR_OFF_TASK
    if evaluation.misconception_tag:
        return AttemptOutcome.WRONG_KNOWN_MISCONCEPTION
    if evaluation.partially_correct:
        return AttemptOutcome.PARTIALLY_CORRECT
    return AttemptOutcome.WRONG_UNKNOWN


def _estimate_tokens(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)


def _similar_practice_message(items: list[PracticeItem], *, fallback: str) -> str:
    if not items:
        return fallback
    lines = ["你已经找到方法了。我们再练习 1-3 道同类题，确认真的掌握："]
    lines.extend(f"{index}. {item.question}" for index, item in enumerate(items, start=1))
    return "\n".join(lines)


def _normalize_llm_knowledge_point(value: str | None) -> str:
    mapping = {
        "限载进一应用题": "capacity_round_up",
        "限载进一": "capacity_round_up",
        "两位数乘一位数": "two_digit_times_one_digit",
    }
    text = (value or "").strip()
    return mapping.get(text, text or "grade_math_unknown")


def _merge_ids(existing: list[str], incoming: list[str]) -> list[str]:
    result = list(existing)
    for item in incoming:
        if item and item not in result:
            result.append(item)
    return result


def _key_point_draft(
    *,
    text: str,
    hint_level: int,
    key_point_id: str,
    analysis: ProblemAnalysis,
):
    from songguo.backend.services.learning.deeptutor_adapter import TeachingDraft

    return TeachingDraft(
        action="hint",
        hint_level=hint_level,
        exposes_final_answer=False,
        text=text,
        metadata={
            "provider": "math_structuring_gateway",
            "model": analysis.source,
            "problem_type": analysis.problem_type,
            "key_point_id": key_point_id,
        },
    )


def build_default_learning_service() -> LearningService:
    store = _build_learning_store_from_env()
    math_gateway = _build_math_gateway_from_env()
    ai_provider = _build_ai_provider_from_env()
    session_runner = _build_session_runner_from_env()
    agent_runtime = (
        os.getenv("SONGGUO_AGENT_RUNTIME")
        or _read_local_dotenv_value("SONGGUO_AGENT_RUNTIME")
        or None
    )
    if (
        os.getenv("SONGGUO_LEARNING_USE_EXTERNAL_ORCHESTRATOR")
        or os.getenv("DEEPTUTOR_LEARNING_USE_ORCHESTRATOR")
    ) == "1":
        provider = OrchestratorDraftProvider()
        return LearningService(
            store=store,
            adapter=DeepTutorLearningAdapter(draft_generator=provider.generate),
            math_gateway=math_gateway,
            ai_provider=ai_provider,
            session_runner=session_runner,
            agent_runtime=agent_runtime,
        )
    return LearningService(
        store=store,
        math_gateway=math_gateway,
        ai_provider=ai_provider,
        session_runner=session_runner,
        agent_runtime=agent_runtime,
    )


def _build_learning_store_from_env():
    database_url = (
        os.getenv("SONGGUO_DATABASE_URL")
        or _read_local_dotenv_value("SONGGUO_DATABASE_URL")
    )
    if database_url:
        return PostgresLearningStore(database_url=database_url)
    return SQLiteLearningStore()


def _build_session_runner_from_env() -> LLMSessionRunner | None:
    runner = (
        os.getenv("SONGGUO_SESSION_RUNNER")
        or _read_local_dotenv_value("SONGGUO_SESSION_RUNNER")
    ).strip().lower()
    if runner != "llm":
        return None
    provider = (
        os.getenv("SONGGUO_AI_ENGINE_PROVIDER")
        or _read_local_dotenv_value("SONGGUO_AI_ENGINE_PROVIDER")
        or os.getenv("LLM_BINDING")
        or _read_local_dotenv_value("LLM_BINDING")
        or "direct_llm"
    )
    model = (
        os.getenv("SONGGUO_LLM_MODEL")
        or _read_local_dotenv_value("SONGGUO_LLM_MODEL")
        or os.getenv("DEEPTUTOR_LLM_MODEL")
        or _read_local_dotenv_value("DEEPTUTOR_LLM_MODEL")
        or os.getenv("DEEPSEEK_MODEL")
        or _read_local_dotenv_value("DEEPSEEK_MODEL")
        or os.getenv("LLM_MODEL")
        or _read_local_dotenv_value("LLM_MODEL")
        or "configured"
    )
    timeout_raw = (
        os.getenv("SONGGUO_LLM_SESSION_TIMEOUT_SECONDS")
        or _read_local_dotenv_value("SONGGUO_LLM_SESSION_TIMEOUT_SECONDS")
        or "20"
    )
    try:
        timeout_seconds = max(0.1, float(timeout_raw))
    except ValueError:
        timeout_seconds = 20.0
    return LLMSessionRunner(provider=provider, model=model, timeout_seconds=timeout_seconds)


def _build_ai_provider_from_env() -> AIEngineProvider | None:
    provider = (
        os.getenv("SONGGUO_AI_ENGINE_PROVIDER")
        or _read_local_dotenv_value("SONGGUO_AI_ENGINE_PROVIDER")
    ).strip().lower()
    if not provider:
        return None
    if provider == "fallback":
        return DeterministicFallbackProvider()
    if provider == "deeptutor":
        return ProviderChain([DeepTutorProvider(), DeterministicFallbackProvider()])
    if provider == "deepseek":
        return ProviderChain([DeepSeekProvider(), DeterministicFallbackProvider()])
    if provider == "ollama":
        return ProviderChain([OllamaProvider(), DeterministicFallbackProvider()])
    if provider == "auto":
        return ProviderChain([DeepTutorProvider(), DeterministicFallbackProvider()])
    return DeterministicFallbackProvider()


def _build_math_gateway_from_env() -> MathProblemStructuringGateway:
    provider = (
        os.getenv("SONGGUO_MATH_STRUCTURING_PROVIDER")
        or _read_local_dotenv_value("SONGGUO_MATH_STRUCTURING_PROVIDER")
    ).strip().lower()
    if provider in {"llm", "deeptutor"}:
        return MathProblemStructuringGateway(structurer=LLMMathStructurer())
    return MathProblemStructuringGateway()


def _read_local_dotenv_value(key: str) -> str:
    root = Path(__file__).resolve().parents[4]
    for env_path in (root / "songguo" / ".env", root / ".env"):
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            current_key, value = line.split("=", 1)
            if current_key.strip() == key:
                return value.strip().strip("\"'")
    return ""


_GLOBAL_SERVICE = build_default_learning_service()
_GLOBAL_STORE = _GLOBAL_SERVICE.store


def get_global_learning_store() -> InMemoryLearningStore:
    return _GLOBAL_STORE


def get_global_learning_service() -> LearningService:
    return _GLOBAL_SERVICE
