from __future__ import annotations

from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph

from songguo.backend.services.learning.child_safety import (
    ChildSafetyAction,
    check_child_safety,
)
from songguo.backend.services.learning.context_pack import build_student_context_pack
from songguo.backend.services.learning.deposit import save_learning_deposit_from_llm_output
from songguo.backend.services.learning.leakage_checker import detect_answer_leakage
from songguo.backend.services.learning.llm_session_runner import (
    LLMSessionOutput,
    LearningDepositDelta,
    LLMSessionRunner,
    StructuredState,
)
from songguo.backend.services.learning.math_structuring import (
    MathProblemStructuringGateway,
    ProblemAnalysis,
    expected_answer_for_leakage,
)
from songguo.backend.services.learning.models import LearningPhase, utc_now
from songguo.backend.services.learning.practice_recommender import (
    PRACTICE_GENERATOR_VERSION,
    build_similar_practice_items,
)
from songguo.backend.services.learning.progress import build_teaching_progress
from songguo.backend.services.learning.store import InMemoryLearningStore
from songguo.backend.services.learning.tutor_graph.math_mistake_nodes import (
    input_normalize_node,
    math_problem_parse_node,
    ocr_or_text_confirm_node,
    rule_judge_node,
)
from songguo.backend.services.learning.tutor_graph.state import (
    GraphCreateResult,
    GraphSubmitResult,
    MathMistakeTutorGraphState,
    RuleJudgeResult,
)
from songguo.backend.services.learning.tutor_contract import (
    SafetyContract,
    SafetyResponseSource,
    build_contextual_fallback_message,
)


@dataclass(frozen=True)
class _SafeMessage:
    text: str
    event_type: str | None = None
    event_payload: dict | None = None


class MathMistakeTutorGraph:
    def __init__(
        self,
        *,
        store: InMemoryLearningStore,
        session_runner: LLMSessionRunner | None = None,
        math_gateway: MathProblemStructuringGateway | None = None,
    ) -> None:
        self.store = store
        self.session_runner = session_runner or LLMSessionRunner()
        self.math_gateway = math_gateway or MathProblemStructuringGateway()
        self.start_graph = self._build_start_graph()
        self.submit_graph = self._build_submit_graph()

    def _build_start_graph(self):
        graph = StateGraph(MathMistakeTutorGraphState)
        graph.add_node("input_normalize", input_normalize_node)
        graph.add_node("ocr_or_text_confirm", ocr_or_text_confirm_node)
        graph.add_node("math_problem_parse", self._math_problem_parse_node)
        graph.add_node("start_session", self._start_session_node)
        graph.add_edge(START, "input_normalize")
        graph.add_edge("input_normalize", "ocr_or_text_confirm")
        graph.add_edge("ocr_or_text_confirm", "math_problem_parse")
        graph.add_edge("math_problem_parse", "start_session")
        graph.add_edge("start_session", END)
        return graph.compile()

    def _build_submit_graph(self):
        graph = StateGraph(MathMistakeTutorGraphState)
        graph.add_node("record_child_attempt", self._record_child_attempt_node)
        graph.add_node("rule_judge", self._rule_judge_node)
        graph.add_node("complete_by_rule_judge", self._complete_by_rule_judge_node)
        graph.add_node("continue_with_model", self._continue_with_model_node)
        graph.add_edge(START, "record_child_attempt")
        graph.add_edge("record_child_attempt", "rule_judge")
        graph.add_conditional_edges(
            "rule_judge",
            self._route_after_rule_judge,
            {
                "complete": "complete_by_rule_judge",
                "continue": "continue_with_model",
            },
        )
        graph.add_edge("complete_by_rule_judge", END)
        graph.add_edge("continue_with_model", END)
        return graph.compile()

    def start(
        self,
        *,
        child_id: str,
        subject: str,
        grade: int,
        question_text: str,
    ) -> GraphCreateResult:
        state = MathMistakeTutorGraphState(
            child_id=child_id,
            grade=grade,
            subject=subject or "math",
            question_text=question_text,
            messages=[{"role": "user", "content": question_text}],
        )
        state = MathMistakeTutorGraphState.model_validate(self.start_graph.invoke(state))
        if not state.graph_create_result:
            raise RuntimeError("MathMistakeTutorGraph start graph did not produce a result")
        return GraphCreateResult.model_validate(state.graph_create_result)

    def _math_problem_parse_node(
        self,
        state: MathMistakeTutorGraphState,
    ) -> MathMistakeTutorGraphState:
        if self.math_gateway is not None:
            try:
                analysis = self.math_gateway.analyze(
                    question_text=state.question_text,
                    grade=state.grade,
                    subject=state.subject or "math",
                )
            except ValueError:
                analysis = None
            if analysis is not None:
                return state.model_copy(
                    update={"problem_analysis": analysis.model_dump(mode="json")}
                )
        return math_problem_parse_node(state)

    def _start_session_node(
        self,
        state: MathMistakeTutorGraphState,
    ) -> MathMistakeTutorGraphState:
        analysis = ProblemAnalysis.model_validate(state.problem_analysis)
        context_pack = build_student_context_pack(
            self.store,
            child_id=state.child_id,
            grade=state.grade,
            current_question=state.question_text,
        )
        output = self.session_runner.start(
            child_id=state.child_id,
            grade=state.grade,
            question_text=state.question_text,
            context_pack=context_pack,
        )
        expected_answer = expected_answer_for_leakage(analysis)
        safe_message = self._safe_child_message(
            output.child_message,
            answer_unlocked=False,
            expected_answer=expected_answer,
            hint_level=1,
            child_id=state.child_id,
            session_id="pending",
            question_text=state.question_text,
            grade=state.grade,
            analysis=analysis,
        )
        message = safe_message.text
        session = self.store.create_session(
            child_id=state.child_id,
            subject=state.subject or "math",
            grade=state.grade,
            question_text=state.question_text,
            normalized_question=state.question_text,
            knowledge_point=analysis.problem_type,
            current_prompt=message,
            runner_mode="langgraph",
            problem_analysis=analysis.model_dump(mode="json"),
            current_key_point_id=(
                output.structured_state.current_key_point or analysis.first_key_point.id
            ),
            released_key_point_ids=[analysis.first_key_point.id],
        )
        self.store.append_message(
            session_id=session.session_id,
            child_id=state.child_id,
            role="user",
            content=state.question_text,
            metadata={"graph": "MathMistakeTutorGraph"},
        )
        self.store.append_message(
            session_id=session.session_id,
            child_id=state.child_id,
            role="assistant",
            content=message,
            metadata={
                "graph": "MathMistakeTutorGraph",
                "structured_state": output.structured_state.model_dump(mode="json"),
            },
        )
        self.store.append_event(
            session_id=session.session_id,
            child_id=state.child_id,
            event_type="math_mistake_graph.started",
            payload={
                "problem_type": analysis.problem_type,
                "source": analysis.source,
                "runner_mode": "langgraph",
            },
        )
        if safe_message.event_type:
            self.store.append_event(
                session_id=session.session_id,
                child_id=state.child_id,
                event_type=safe_message.event_type,
                payload=safe_message.event_payload or {},
            )
            if safe_message.event_type == "safety.blocked":
                self.store.record_safety_event(
                    session_id=session.session_id,
                    child_id=state.child_id,
                    event_type="safety.blocked",
                    blocked_text=output.child_message,
                    reason=(safe_message.event_payload or {}).get("reason", "child_safety_risk"),
                )
        self._record_output_quality_signals(
            session_id=session.session_id,
            child_id=state.child_id,
            output=output,
        )
        self.store.record_ai_call(
            child_id=state.child_id,
            session_id=session.session_id,
            provider=output.provider,
            model=output.model,
            operation="math_mistake_graph.start",
            token_estimate=output.token_estimate,
            status=output.status,
        )
        result = GraphCreateResult(
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
        return state.model_copy(
            update={
                "session_id": session.session_id,
                "tutor_reply": message,
                "structured_state": output.structured_state.model_dump(mode="json"),
                "learning_deposit_delta": output.learning_deposit_delta.model_dump(mode="json"),
                "practice_items": output.practice_items,
                "provider_trace": {
                    "provider": output.provider,
                    "model": output.model,
                    "status": output.status,
                    "token_estimate": output.token_estimate,
                },
                "graph_create_result": result.model_dump(mode="json"),
            }
        )

    def submit_attempt(self, session_id: str, *, child_answer: str) -> GraphSubmitResult:
        session = self.store.require_session(session_id)
        if session.phase in {LearningPhase.SIMILAR_PRACTICE, LearningPhase.PRACTICE_PAUSED}:
            return self._continue_practice_with_model(
                session_id=session_id,
                child_answer=child_answer,
            )
        analysis = ProblemAnalysis.model_validate(session.problem_analysis)
        state = MathMistakeTutorGraphState(
            child_id=session.child_id,
            session_id=session.session_id,
            grade=session.grade,
            subject=session.subject,
            question_text=session.question_text,
            problem_analysis=analysis.model_dump(mode="json"),
            messages=[
                {"role": message.role, "content": message.content}
                for message in self.store.list_messages(session.session_id)
            ],
            child_answer=child_answer,
        )
        state = MathMistakeTutorGraphState.model_validate(self.submit_graph.invoke(state))
        if not state.graph_submit_result:
            raise RuntimeError("MathMistakeTutorGraph submit graph did not produce a result")
        return GraphSubmitResult.model_validate(state.graph_submit_result)

    def _continue_practice_with_model(self, *, session_id: str, child_answer: str) -> GraphSubmitResult:
        session = self.store.require_session(session_id)
        analysis = ProblemAnalysis.model_validate(session.problem_analysis or {})
        practice_items = _current_practice_items_for_session(self.store, session, analysis)
        context_pack = build_student_context_pack(
            self.store,
            child_id=session.child_id,
            grade=session.grade,
            current_question=session.question_text,
        )
        messages = [
            *self.store.list_messages(session.session_id),
            {"role": "user", "content": child_answer},
        ]
        output = self.session_runner.run_practice_control(
            child_id=session.child_id,
            grade=session.grade,
            context_pack=context_pack,
            messages=messages,
            practice_items=practice_items,
            current_phase=session.phase.value,
        )
        target_phase = _practice_phase_from_output(output)
        safe_message = self._safe_child_message(
            output.child_message,
            answer_unlocked=True,
            expected_answer=None,
            hint_level=session.hint_level,
            child_id=session.child_id,
            session_id=session.session_id,
            question_text=session.question_text,
            grade=session.grade,
            analysis=analysis,
        )
        message = safe_message.text
        self.store.append_message(
            session_id=session.session_id,
            child_id=session.child_id,
            role="user",
            content=child_answer,
            metadata={"graph": "MathMistakeTutorGraph", "phase": "practice_control"},
        )
        self.store.append_message(
            session_id=session.session_id,
            child_id=session.child_id,
            role="assistant",
            content=message,
            metadata={
                "graph": "MathMistakeTutorGraph",
                "phase": "practice_control",
                "teaching_intent": output.teaching_intent.model_dump(mode="json"),
                "structured_state": output.structured_state.model_dump(mode="json"),
            },
        )
        updated = self.store.update_session(
            session.session_id,
            phase=target_phase,
            answer_unlocked=True,
            current_prompt=message,
            current_key_point_id=(
                output.structured_state.current_key_point or session.current_key_point_id
            ),
        )
        self.store.append_event(
            session_id=session.session_id,
            child_id=session.child_id,
            event_type="practice.control_intent_evaluated",
            payload={
                "child_answer": child_answer,
                "runner_mode": "langgraph",
                "target_phase": target_phase.value,
                "teaching_intent": output.teaching_intent.model_dump(mode="json"),
                "structured_state": output.structured_state.model_dump(mode="json"),
            },
        )
        if safe_message.event_type:
            self.store.append_event(
                session_id=session.session_id,
                child_id=session.child_id,
                event_type=safe_message.event_type,
                payload=safe_message.event_payload or {},
            )
        self.store.record_ai_call(
            child_id=session.child_id,
            session_id=session.session_id,
            provider=output.provider,
            model=output.model,
            operation="math_mistake_graph.practice_control",
            token_estimate=output.token_estimate,
            status=output.status,
        )
        move = output.teaching_intent.teacher_move
        correct = (
            output.learning_deposit_delta.is_correct
            if output.learning_deposit_delta.is_correct is not None
            else move
            in {
                "resume_practice",
                "pause_practice",
                "start_new_question",
                "small_talk_or_other",
                "ask_clarification",
            }
        )
        return GraphSubmitResult(
            correct=bool(correct),
            phase=updated.phase.value,
            hint_level=updated.hint_level,
            message=updated.current_prompt,
            answer_unlocked=updated.answer_unlocked,
            teaching_progress=build_teaching_progress(updated),
            practice_items=list(output.practice_items[:3]),
        )

    def _record_child_attempt_node(
        self,
        state: MathMistakeTutorGraphState,
    ) -> MathMistakeTutorGraphState:
        session = self.store.require_session(state.session_id)
        self.store.append_message(
            session_id=session.session_id,
            child_id=session.child_id,
            role="user",
            content=state.child_answer,
            metadata={"graph": "MathMistakeTutorGraph"},
        )
        self.store.append_event(
            session_id=session.session_id,
            child_id=session.child_id,
            event_type="child.attempt_submitted",
            payload={"child_answer": state.child_answer, "runner_mode": "langgraph"},
        )
        return state.model_copy(
            update={
                "messages": [
                    {"role": message.role, "content": message.content}
                    for message in self.store.list_messages(session.session_id)
                ]
            }
        )

    def _rule_judge_node(
        self,
        state: MathMistakeTutorGraphState,
    ) -> MathMistakeTutorGraphState:
        return rule_judge_node(state, child_answer=state.child_answer)

    def _route_after_rule_judge(self, state: MathMistakeTutorGraphState) -> str:
        rule_result = RuleJudgeResult.model_validate(state.rule_judge_result or {})
        if rule_result.correct is True:
            return "complete"
        return "continue"

    def _complete_by_rule_judge_node(
        self,
        state: MathMistakeTutorGraphState,
    ) -> MathMistakeTutorGraphState:
        result = self._complete_by_rule_judge(
            session_id=state.session_id,
            child_answer=state.child_answer,
            analysis=ProblemAnalysis.model_validate(state.problem_analysis),
            rule_result=RuleJudgeResult.model_validate(state.rule_judge_result or {}),
        )
        return state.model_copy(
            update={"graph_submit_result": result.model_dump(mode="json")}
        )

    def _continue_with_model_node(
        self,
        state: MathMistakeTutorGraphState,
    ) -> MathMistakeTutorGraphState:
        result = self._continue_with_model(
            session_id=state.session_id,
            child_answer=state.child_answer,
            analysis=ProblemAnalysis.model_validate(state.problem_analysis),
            rule_result=RuleJudgeResult.model_validate(state.rule_judge_result or {}),
        )
        return state.model_copy(
            update={"graph_submit_result": result.model_dump(mode="json")}
        )

    def _complete_by_rule_judge(
        self,
        *,
        session_id: str,
        child_answer: str,
        analysis: ProblemAnalysis,
        rule_result: RuleJudgeResult,
    ) -> GraphSubmitResult:
        session = self.store.require_session(session_id)
        practice_items = build_similar_practice_items(
            knowledge_point=analysis.problem_type,
            misconception_tag=session.last_misconception,
            limit=3,
            source_question=session.question_text,
        )
        message = _similar_practice_message(practice_items)
        updated = self.store.update_session(
            session.session_id,
            phase=LearningPhase.SIMILAR_PRACTICE,
            attempt_count=session.attempt_count + 1,
            hint_level=session.hint_level,
            answer_unlocked=True,
            current_prompt=message,
            completed_at=utc_now(),
        )
        self.store.append_message(
            session_id=session.session_id,
            child_id=session.child_id,
            role="assistant",
            content=message,
            metadata={
                "graph": "MathMistakeTutorGraph",
                "rule_judge_result": rule_result.model_dump(mode="json"),
            },
        )
        self.store.append_event(
            session_id=session.session_id,
            child_id=session.child_id,
            event_type="attempt.evaluated",
            payload={
                "correct": True,
                "child_answer": child_answer,
                "runner_mode": "langgraph",
                "rule_judge_result": rule_result.model_dump(mode="json"),
            },
        )
        self.store.append_event(
            session_id=session.session_id,
            child_id=session.child_id,
            event_type="practice.generated",
            payload={
                "items": [item.model_dump(mode="json") for item in practice_items],
                "message": message,
                "runner_mode": "langgraph",
                "practice_generator_version": PRACTICE_GENERATOR_VERSION,
            },
        )
        output = LLMSessionOutput(
            child_message=message,
            structured_state=StructuredState(
                phase=LearningPhase.SIMILAR_PRACTICE.value,
                answer_unlocked=True,
                should_end_session=True,
            ),
            learning_deposit_delta=LearningDepositDelta(
                knowledge_point=analysis.problem_type,
                question_type=analysis.problem_type,
                is_correct=True,
                main_misconception=session.last_misconception,
                evidence=rule_result.evidence,
                need_review=False,
                parent_summary="孩子已经完成本题，可以继续做同类题巩固。",
            ),
            practice_items=practice_items,
            provider="math_rule_judge",
            model="local_rules_v0.1",
            status="success",
        )
        save_learning_deposit_from_llm_output(
            self.store,
            session_id=session.session_id,
            output=output,
        )
        self.store.record_ai_call(
            child_id=session.child_id,
            session_id=session.session_id,
            provider="math_rule_judge",
            model="local_rules_v0.1",
            operation="math_mistake_graph.rule_judge",
            token_estimate=0,
            status="success",
        )
        return GraphSubmitResult(
            correct=True,
            phase=updated.phase.value,
            hint_level=updated.hint_level,
            message=updated.current_prompt,
            answer_unlocked=updated.answer_unlocked,
            teaching_progress=build_teaching_progress(updated),
            practice_items=practice_items,
        )

    def _continue_with_model(
        self,
        *,
        session_id: str,
        child_answer: str,
        analysis: ProblemAnalysis,
        rule_result: RuleJudgeResult,
    ) -> GraphSubmitResult:
        session = self.store.require_session(session_id)
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
        next_attempt_count = session.attempt_count + 1
        next_hint_level = min(5, session.hint_level + 1)
        expected_answer = expected_answer_for_leakage(analysis)
        is_done = (
            output.structured_state.should_end_session
            or output.structured_state.phase == LearningPhase.SIMILAR_PRACTICE.value
            or output.learning_deposit_delta.is_correct is True
            or bool(output.practice_items)
        )
        next_phase = _normal_tutoring_phase_from_output(output, is_done=is_done)
        answer_unlocked = bool(is_done or output.structured_state.answer_unlocked)
        safe_message = self._safe_child_message(
            output.child_message,
            answer_unlocked=answer_unlocked,
            expected_answer=expected_answer,
            hint_level=next_hint_level,
            child_id=session.child_id,
            session_id=session.session_id,
            question_text=session.question_text,
            grade=session.grade,
            analysis=analysis,
        )
        message = safe_message.text
        misconception = (
            rule_result.misconception_tag
            or output.learning_deposit_delta.main_misconception
            or output.structured_state.main_misconception
            or session.last_misconception
        )
        practice_items = list(output.practice_items[:3])
        if is_done and not practice_items:
            practice_items = build_similar_practice_items(
                knowledge_point=analysis.problem_type,
                misconception_tag=misconception,
                limit=3,
                source_question=session.question_text,
            )
        if is_done and practice_items and "1." not in message:
            message = _similar_practice_message(practice_items)

        updated = self.store.update_session(
            session.session_id,
            phase=next_phase,
            attempt_count=next_attempt_count,
            hint_level=next_hint_level,
            answer_unlocked=answer_unlocked,
            last_misconception=misconception,
            current_prompt=message,
            current_key_point_id=(
                rule_result.next_key_point_id
                or output.structured_state.current_key_point
                or session.current_key_point_id
            ),
            completed_at=utc_now() if is_done else session.completed_at,
        )
        self.store.append_message(
            session_id=session.session_id,
            child_id=session.child_id,
            role="assistant",
            content=message,
            metadata={
                "graph": "MathMistakeTutorGraph",
                "structured_state": output.structured_state.model_dump(mode="json"),
                "rule_judge_result": rule_result.model_dump(mode="json"),
            },
        )
        self.store.append_event(
            session_id=session.session_id,
            child_id=session.child_id,
            event_type="assistant.message_created",
            payload={
                "message": message,
                "runner_mode": "langgraph",
                "rule_judge_result": rule_result.model_dump(mode="json"),
                "structured_state": output.structured_state.model_dump(mode="json"),
            },
        )
        if safe_message.event_type:
            self.store.append_event(
                session_id=session.session_id,
                child_id=session.child_id,
                event_type=safe_message.event_type,
                payload=safe_message.event_payload or {},
            )
        self._record_output_quality_signals(
            session_id=session.session_id,
            child_id=session.child_id,
            output=output,
        )
        if misconception:
            self.store.record_wrong_question(
                session_id=session.session_id,
                child_id=session.child_id,
                normalized_question=session.normalized_question,
                knowledge_point=analysis.problem_type,
                mistake_summary=rule_result.evidence or output.learning_deposit_delta.evidence,
                last_misconception=misconception,
                highest_hint_level=updated.hint_level,
                explanation_unlocked=answer_unlocked,
                practice_completed=is_done,
            )
        if is_done:
            self.store.append_event(
                session_id=session.session_id,
                child_id=session.child_id,
                event_type="attempt.evaluated",
                payload={
                    "correct": True,
                    "child_answer": child_answer,
                    "runner_mode": "langgraph",
                    "evidence": output.learning_deposit_delta.evidence,
                    "structured_state": output.structured_state.model_dump(mode="json"),
                    "rule_judge_result": rule_result.model_dump(mode="json"),
                },
            )
            self.store.append_event(
                session_id=session.session_id,
                child_id=session.child_id,
                event_type="practice.generated",
                payload={
                    "items": [item.model_dump(mode="json") for item in practice_items],
                    "message": message,
                    "runner_mode": "langgraph",
                    "practice_generator_version": PRACTICE_GENERATOR_VERSION,
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
            operation="math_mistake_graph.continue",
            token_estimate=output.token_estimate,
            status=output.status,
        )
        return GraphSubmitResult(
            correct=is_done,
            partially_correct=rule_result.partially_correct and not is_done,
            phase=updated.phase.value,
            hint_level=updated.hint_level,
            message=updated.current_prompt,
            answer_unlocked=updated.answer_unlocked,
            misconception_tag=misconception,
            matched_key_point_id=rule_result.matched_key_point_id,
            next_key_point_id=updated.current_key_point_id,
            teaching_progress=build_teaching_progress(updated),
            practice_items=practice_items,
        )

    def _safe_child_message(
        self,
        text: str,
        *,
        answer_unlocked: bool,
        expected_answer: str | None,
        hint_level: int,
        child_id: str,
        session_id: str,
        question_text: str,
        grade: int,
        analysis: ProblemAnalysis,
    ) -> _SafeMessage:
        quality_signal = detect_answer_leakage(
            draft_text=text,
            answer_unlocked=answer_unlocked,
            expected_answer=expected_answer,
        )
        quality_event_type = None
        quality_event_payload = None
        if quality_signal.detected:
            quality_event_type = "teaching_quality.direct_answer"
            quality_event_payload = {
                "reason": quality_signal.reason,
                "message": text,
                "expected_answer": expected_answer,
                "blocked": False,
            }

        verdict = check_child_safety(text)
        if verdict.action == ChildSafetyAction.ALLOW:
            return _SafeMessage(
                text=text,
                event_type=quality_event_type,
                event_payload=quality_event_payload,
            )

        if session_id != "pending":
            self.store.record_safety_event(
                session_id=session_id,
                child_id=child_id,
                event_type="safety.blocked",
                blocked_text=text,
                reason=verdict.reason,
            )

        return _SafeMessage(
            text=verdict.safe_text,
            event_type="safety.blocked",
            event_payload={
                **SafetyContract(
                    action="hard_fallback",
                    reason=verdict.reason,
                    final_response_source=SafetyResponseSource.HARD_FALLBACK,
                    repair_attempted=False,
                ).model_dump(mode="json"),
                "blocked_text": text,
            },
        )

    def _record_output_quality_signals(
        self,
        *,
        session_id: str,
        child_id: str,
        output: LLMSessionOutput,
    ) -> None:
        failed_checks = [
            name
            for name, failed in {
                "praised_wrong_answer": output.self_check.praised_wrong_answer,
                "revealed_final_answer": output.self_check.revealed_final_answer,
                "one_question_only": not output.self_check.one_question_only,
                "directly_solved_multiple_steps": output.self_check.directly_solved_multiple_steps,
            }.items()
            if failed
        ]
        if not failed_checks:
            return
        self.store.append_event(
            session_id=session_id,
            child_id=child_id,
            event_type="teaching_quality.self_check_failed",
            payload={
                "failed_checks": failed_checks,
                "child_answer_status": output.teaching_intent.child_answer_status,
                "teacher_move": output.teaching_intent.teacher_move,
                "message": output.child_message,
                "blocked": False,
            },
        )


def _similar_practice_message(practice_items) -> str:
    lines = ["你已经找到方法了。我们再练习 1-3 道同类题，确认真的掌握："]
    for index, item in enumerate(practice_items[:3], start=1):
        lines.append(f"{index}. {item.question}")
    return "\n".join(lines)


def _normal_tutoring_phase_from_output(
    output: LLMSessionOutput, *, is_done: bool
) -> LearningPhase:
    requested_phase = str(output.structured_state.phase or "").strip()
    teacher_move = str(output.teaching_intent.teacher_move or "").strip()
    learner_readiness = str(output.teaching_intent.learner_readiness or "").strip()
    if requested_phase == LearningPhase.LEARNING_PAUSED.value or teacher_move == "pause_learning":
        return LearningPhase.LEARNING_PAUSED
    if learner_readiness == "not_ready":
        return LearningPhase.LEARNING_PAUSED
    if requested_phase == LearningPhase.SESSION_SUMMARY.value:
        return LearningPhase.SESSION_SUMMARY
    if is_done:
        return LearningPhase.SIMILAR_PRACTICE
    return LearningPhase.WAIT_CHILD_ATTEMPT


def _practice_phase_from_output(output: LLMSessionOutput) -> LearningPhase:
    requested_phase = str(output.structured_state.phase or "").strip()
    teacher_move = str(output.teaching_intent.teacher_move or "").strip()
    if requested_phase == LearningPhase.SESSION_SUMMARY.value:
        return LearningPhase.SESSION_SUMMARY
    if requested_phase == LearningPhase.PRACTICE_PAUSED.value or teacher_move in {
        "pause_practice",
        "start_new_question",
        "small_talk_or_other",
        "ask_clarification",
    }:
        return LearningPhase.PRACTICE_PAUSED
    return LearningPhase.SIMILAR_PRACTICE


def _latest_practice_items(store: InMemoryLearningStore, session_id: str) -> list[dict]:
    for event in reversed(store.list_events(session_id)):
        if event.event_type != "practice.generated":
            continue
        if event.payload.get("practice_generator_version") != PRACTICE_GENERATOR_VERSION:
            continue
        items = event.payload.get("items")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


def _current_practice_items_for_session(
    store: InMemoryLearningStore,
    session,
    analysis: ProblemAnalysis,
) -> list[dict]:
    existing = _latest_practice_items(store, session.session_id)
    if existing:
        return existing
    practice_items = build_similar_practice_items(
        knowledge_point=analysis.problem_type,
        misconception_tag=session.last_misconception,
        limit=3,
        source_question=session.question_text,
    )
    message = _similar_practice_message(practice_items)
    store.append_event(
        session_id=session.session_id,
        child_id=session.child_id,
        event_type="practice.generated",
        payload={
            "items": [item.model_dump(mode="json") for item in practice_items],
            "message": message,
            "runner_mode": "langgraph",
            "practice_generator_version": PRACTICE_GENERATOR_VERSION,
            "regenerated_for_practice_control": True,
        },
    )
    return [item.model_dump(mode="json") for item in practice_items]
