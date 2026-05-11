from __future__ import annotations

from pydantic import BaseModel, Field

from songguo.backend.services.learning.llm_session_runner import LLMSessionRunner
from songguo.backend.services.learning.reporting import build_session_feedback
from songguo.backend.services.learning.store import InMemoryLearningStore
from songguo.backend.services.learning.tutor_graph.math_mistake_graph import (
    MathMistakeTutorGraph,
)


class GraphEvaluationCase(BaseModel):
    question_id: str
    grade: int
    question_text: str
    final_answer: str
    wrong_answers: list[str] = Field(default_factory=list)
    correct_answer: str


class GraphEvaluationReport(BaseModel):
    total: int
    completed_session_count: int = 0
    learning_deposit_count: int = 0
    parent_feedback_count: int = 0
    practice_generated_count: int = 0
    practice_quality_failure_count: int = 0
    answer_leakage_count: int = 0
    model_fallback_count: int = 0
    failed_case_ids: list[str] = Field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.completed_session_count / self.total


def evaluate_math_tutor_graph(
    *,
    cases: list[GraphEvaluationCase],
    session_runner: LLMSessionRunner | None = None,
) -> GraphEvaluationReport:
    completed_session_count = 0
    learning_deposit_count = 0
    parent_feedback_count = 0
    practice_generated_count = 0
    practice_quality_failure_count = 0
    answer_leakage_count = 0
    model_fallback_count = 0
    failed_case_ids: list[str] = []

    for case in cases:
        store = InMemoryLearningStore()
        graph = MathMistakeTutorGraph(store=store, session_runner=session_runner)
        child_id = f"eval_{case.question_id}"
        try:
            created = graph.start(
                child_id=child_id,
                subject="math",
                grade=case.grade,
                question_text=case.question_text,
            )
            if _leaks_answer(created.message, case.final_answer):
                answer_leakage_count += 1

            last_result = None
            for answer in case.wrong_answers:
                last_result = graph.submit_attempt(created.session_id, child_answer=answer)
                if _leaks_answer(last_result.message, case.final_answer):
                    answer_leakage_count += 1

            last_result = graph.submit_attempt(
                created.session_id,
                child_answer=case.correct_answer,
            )
            if last_result.correct:
                completed_session_count += 1
            else:
                failed_case_ids.append(case.question_id)
            if last_result.practice_items:
                practice_generated_count += 1
            if not _practice_items_are_usable(last_result.practice_items):
                practice_quality_failure_count += 1
                failed_case_ids.append(f"{case.question_id}:practice_quality")
            if store.get_learning_deposit(created.session_id) is not None:
                learning_deposit_count += 1
            try:
                build_session_feedback(
                    store,
                    child_id=child_id,
                    session_id=created.session_id,
                )
                parent_feedback_count += 1
            except Exception:
                failed_case_ids.append(f"{case.question_id}:parent_feedback")
            model_fallback_count += sum(
                1
                for call in store.list_ai_call_logs(child_id=child_id)
                if call.status == "fallback"
            )
        except Exception:
            failed_case_ids.append(case.question_id)

    return GraphEvaluationReport(
        total=len(cases),
        completed_session_count=completed_session_count,
        learning_deposit_count=learning_deposit_count,
        parent_feedback_count=parent_feedback_count,
        practice_generated_count=practice_generated_count,
        practice_quality_failure_count=practice_quality_failure_count,
        answer_leakage_count=answer_leakage_count,
        model_fallback_count=model_fallback_count,
        failed_case_ids=failed_case_ids,
    )


def _leaks_answer(text: str, final_answer: str) -> bool:
    answer = (final_answer or "").strip()
    if not answer:
        return False
    return answer in (text or "")


def _practice_items_are_usable(items: list) -> bool:
    if not 1 <= len(items) <= 3:
        return False
    for item in items:
        question = str(getattr(item, "question", "")).strip()
        answer = getattr(item, "answer", None)
        if not question or not answer:
            return False
        if "请再输入" in question or "同类型数学题" in question:
            return False
        if not any(marker in question for marker in ["?", "？", "多少", "几"]):
            return False
    return True
