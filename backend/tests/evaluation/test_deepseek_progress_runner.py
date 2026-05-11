from __future__ import annotations

from songguo.backend.evaluation.deepseek_progress_runner import (
    DeepSeekProgressOptions,
    run_deepseek_progress_evaluation,
)
from songguo.backend.evaluation.golden_math import GoldenMathQuestion
from songguo.backend.services.learning.ai_engine import AIEngineContext, ProviderError
from songguo.backend.services.learning.math_structuring import ProblemAnalysis


QUESTION = GoldenMathQuestion(
    question_id="sample_001",
    grade=3,
    category="calculation",
    question_text="21 × 3 = ?",
    skill_ids=["math_two_digit_times_one_digit"],
    misconception_ids=["math_multiplication_carry_missing"],
    final_answer="63",
)


class SuccessfulProvider:
    provider_name = "deepseek"
    model_name = "deepseek-chat"

    def structure_math_problem(
        self,
        *,
        question_text: str,
        grade: int,
        context: AIEngineContext,
    ) -> ProblemAnalysis:
        return _analysis(source="deepseek")


class TimeoutProvider:
    provider_name = "deepseek"
    model_name = "deepseek-chat"

    def structure_math_problem(
        self,
        *,
        question_text: str,
        grade: int,
        context: AIEngineContext,
    ) -> ProblemAnalysis:
        raise ProviderError("deepseek.structure_math_problem timed out after 20.0s")


class BadSchemaProvider:
    provider_name = "deepseek"
    model_name = "deepseek-chat"

    def structure_math_problem(
        self,
        *,
        question_text: str,
        grade: int,
        context: AIEngineContext,
    ) -> ProblemAnalysis:
        return _analysis(source="deepseek", key_point_prompt="最终答案是63。")


class MissingAnswerProvider:
    provider_name = "deepseek"
    model_name = "deepseek-chat"

    def structure_math_problem(
        self,
        *,
        question_text: str,
        grade: int,
        context: AIEngineContext,
    ) -> ProblemAnalysis:
        raise ProviderError("final_answer is required")


class FallbackProvider:
    provider_name = "deterministic_fallback"
    model_name = "local_rules_v0.1"

    def structure_math_problem(
        self,
        *,
        question_text: str,
        grade: int,
        context: AIEngineContext,
    ) -> ProblemAnalysis:
        return _analysis(source="deterministic_fallback")


def test_progress_runner_reports_direct_deepseek_success() -> None:
    lines: list[str] = []

    report = run_deepseek_progress_evaluation(
        questions=[QUESTION],
        deepseek_provider=SuccessfulProvider(),
        fallback_provider=FallbackProvider(),
        options=DeepSeekProgressOptions(limit=1),
        emit=lines.append,
    )

    assert report.total == 1
    assert report.deepseek_success == 1
    assert report.fallback_success == 0
    assert report.failure_counts == {}
    assert "[1/1] sample_001 start" in lines[0]
    assert "provider=deepseek" in lines[1]
    assert "summary total=1 deepseek_success=1 fallback_success=0" in lines[-1]


def test_progress_runner_classifies_timeout_and_uses_fallback() -> None:
    lines: list[str] = []

    report = run_deepseek_progress_evaluation(
        questions=[QUESTION],
        deepseek_provider=TimeoutProvider(),
        fallback_provider=FallbackProvider(),
        options=DeepSeekProgressOptions(limit=1),
        emit=lines.append,
    )

    assert report.total == 1
    assert report.deepseek_success == 0
    assert report.fallback_success == 1
    assert report.failure_counts == {"timeout": 1}
    assert report.failed_case_ids == ["sample_001:timeout"]
    assert "failure=timeout fallback=success" in lines[1]


def test_progress_runner_classifies_schema_failure_before_fallback() -> None:
    lines: list[str] = []

    report = run_deepseek_progress_evaluation(
        questions=[QUESTION],
        deepseek_provider=BadSchemaProvider(),
        fallback_provider=FallbackProvider(),
        options=DeepSeekProgressOptions(limit=1),
        emit=lines.append,
    )

    assert report.deepseek_success == 0
    assert report.fallback_success == 1
    assert report.failure_counts == {"schema": 1}
    assert report.failed_case_ids == ["sample_001:schema"]
    assert "failure=schema fallback=success" in lines[1]


def test_progress_runner_classifies_missing_required_field_as_schema() -> None:
    report = run_deepseek_progress_evaluation(
        questions=[QUESTION],
        deepseek_provider=MissingAnswerProvider(),
        fallback_provider=FallbackProvider(),
        options=DeepSeekProgressOptions(limit=1),
        emit=lambda _line: None,
    )

    assert report.failure_counts == {"schema": 1}
    assert report.failed_case_ids == ["sample_001:schema"]


def _analysis(*, source: str, key_point_prompt: str = "先想21×3可以怎么算？") -> ProblemAnalysis:
    return ProblemAnalysis(
        subject="math",
        grade=3,
        problem_type="multiplication",
        skill_ids=["math_two_digit_times_one_digit"],
        misconception_ids=["math_multiplication_carry_missing"],
        concept_card_ids=["card_math_two_digit_times_one_digit_v01"],
        knowledge_points=["两位数乘一位数"],
        conditions=[{"id": "condition_1", "text": "21 × 3", "value": None, "unit": ""}],
        target="求乘积",
        solution_steps=[
            {"id": "step_1", "goal": "计算乘积", "expression": "21 × 3", "result": "63"}
        ],
        final_answer="63",
        common_misconceptions=[
            {"tag": "math_multiplication_carry_missing", "description": "进位遗漏"}
        ],
        key_points=[
            {
                "id": "kp_1",
                "name": "先拆分",
                "teaching_goal": "先拆成20和1",
                "release_stage": "HINT_STEP_1",
                "unlock_condition": "question_started",
                "child_prompt": key_point_prompt,
                "expected_child_response": ["60", "3"],
                "forbidden_content": ["63"],
            }
        ],
        confidence=0.9,
        source=source,
    )
