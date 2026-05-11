from __future__ import annotations

import time
from collections.abc import Callable

from pydantic import BaseModel, Field

from songguo.backend.evaluation.golden_math import GoldenMathQuestion
from songguo.backend.services.learning.ai_engine import AIEngineContext, AIEngineProvider
from songguo.backend.services.learning.math_structuring import validate_problem_analysis


class DeepSeekProgressOptions(BaseModel):
    limit: int = 100
    tenant_id: str = "eval_tenant"
    child_id: str = "eval_child"
    stop_on_first_failure: bool = False


class DeepSeekProgressReport(BaseModel):
    total: int
    deepseek_success: int = 0
    fallback_success: int = 0
    fallback_failures: int = 0
    failure_counts: dict[str, int] = Field(default_factory=dict)
    failed_case_ids: list[str] = Field(default_factory=list)

    @property
    def deepseek_success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.deepseek_success / self.total


def run_deepseek_progress_evaluation(
    *,
    questions: list[GoldenMathQuestion],
    deepseek_provider: AIEngineProvider,
    fallback_provider: AIEngineProvider,
    options: DeepSeekProgressOptions | None = None,
    emit: Callable[[str], None] = print,
) -> DeepSeekProgressReport:
    options = options or DeepSeekProgressOptions()
    selected_questions = questions[: max(0, options.limit)]
    report = DeepSeekProgressReport(total=len(selected_questions))

    for index, question in enumerate(selected_questions, start=1):
        prefix = f"[{index}/{report.total}] {question.question_id}"
        emit(f"{prefix} start category={question.category}")
        started = time.perf_counter()
        try:
            analysis = deepseek_provider.structure_math_problem(
                question_text=question.question_text,
                grade=question.grade,
                context=AIEngineContext(
                    child_id=options.child_id,
                    session_id=None,
                    tenant_id=options.tenant_id,
                    request_id=question.question_id,
                ),
            )
            verdict = validate_problem_analysis(analysis)
            latency_ms = _latency_ms(started)
            if not verdict.valid:
                failure_type = "schema"
                _record_failure(report, question.question_id, failure_type)
                fallback_status = _run_fallback(
                    fallback_provider=fallback_provider,
                    question=question,
                    options=options,
                    report=report,
                )
                emit(
                    f"{prefix} failure={failure_type} fallback={fallback_status} "
                    f"latency_ms={latency_ms} errors={_compact_errors(verdict.errors)}"
                )
                if options.stop_on_first_failure:
                    break
                continue
            report.deepseek_success += 1
            provider_name = getattr(deepseek_provider, "provider_name", "deepseek")
            emit(
                f"{prefix} provider={provider_name} status=success "
                f"valid=True latency_ms={latency_ms}"
            )
        except Exception as exc:
            latency_ms = _latency_ms(started)
            failure_type = _classify_failure(exc)
            _record_failure(report, question.question_id, failure_type)
            fallback_status = _run_fallback(
                fallback_provider=fallback_provider,
                question=question,
                options=options,
                report=report,
            )
            emit(
                f"{prefix} failure={failure_type} fallback={fallback_status} "
                f"latency_ms={latency_ms} error={exc.__class__.__name__}: {_compact_error(exc)}"
            )
            if options.stop_on_first_failure:
                break

    emit(
        "summary "
        f"total={report.total} "
        f"deepseek_success={report.deepseek_success} "
        f"fallback_success={report.fallback_success} "
        f"fallback_failures={report.fallback_failures} "
        f"failure_counts={dict(sorted(report.failure_counts.items()))}"
    )
    return report


def _run_fallback(
    *,
    fallback_provider: AIEngineProvider,
    question: GoldenMathQuestion,
    options: DeepSeekProgressOptions,
    report: DeepSeekProgressReport,
) -> str:
    try:
        analysis = fallback_provider.structure_math_problem(
            question_text=question.question_text,
            grade=question.grade,
            context=AIEngineContext(
                child_id=options.child_id,
                session_id=None,
                tenant_id=options.tenant_id,
                request_id=f"{question.question_id}:fallback",
            ),
        )
        verdict = validate_problem_analysis(analysis)
        if not verdict.valid:
            report.fallback_failures += 1
            return "schema_failed"
        report.fallback_success += 1
        return "success"
    except Exception:
        report.fallback_failures += 1
        return "failed"


def _record_failure(
    report: DeepSeekProgressReport,
    question_id: str,
    failure_type: str,
) -> None:
    report.failure_counts[failure_type] = report.failure_counts.get(failure_type, 0) + 1
    report.failed_case_ids.append(f"{question_id}:{failure_type}")


def _classify_failure(exc: Exception) -> str:
    text = f"{exc.__class__.__name__}: {exc}".lower()
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if (
        "validation error" in text
        or "schema" in text
        or "leaks forbidden content" in text
        or " is required" in text
        or "must include" in text
    ):
        return "schema"
    if "http" in text or "request" in text or "provider" in text:
        return "provider"
    return "unknown"


def _latency_ms(started: float) -> int:
    return max(0, int((time.perf_counter() - started) * 1000))


def _compact_errors(errors: list[str]) -> str:
    if not errors:
        return ""
    return "; ".join(errors[:2]).replace("\n", " ")[:240]


def _compact_error(exc: Exception) -> str:
    return str(exc).replace("\n", " ")[:240]
