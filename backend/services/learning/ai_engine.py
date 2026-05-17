from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import os
import re
import time
from typing import Any, Protocol

from pydantic import BaseModel, Field

from songguo.backend.services.learning.deeptutor_adapter import TeachingDraft
from songguo.backend.services.learning.math_structuring import (
    LLMMathStructurer,
    KeyPoint,
    ProblemAnalysis,
    validate_problem_analysis,
)
from songguo.backend.services.learning.practice_recommender import (
    PracticeItem,
    build_similar_practice_items,
)
from songguo.backend.services.learning.real_model_client import DEFAULT_DEEPSEEK_MODEL
from songguo.backend.services.learning.teaching_assets import (
    DEFAULT_MATH_ASSET_LIBRARY,
    TeachingAssetLibrary,
)


_PROVIDER_EXECUTOR = ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="songguo-ai-provider",
)


class ProviderError(RuntimeError):
    pass


class AIEngineContext(BaseModel):
    child_id: str
    session_id: str | None = None
    tenant_id: str = "local_tenant"
    request_id: str | None = None
    knowledge_scope: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderCall(BaseModel):
    provider: str
    model: str
    operation: str
    status: str
    latency_ms: int
    token_estimate: int = 0
    cost_estimate: float = 0.0
    error: str | None = None


class AIEngineProvider(Protocol):
    provider_name: str
    model_name: str

    def structure_math_problem(
        self,
        *,
        question_text: str,
        grade: int,
        context: AIEngineContext,
    ) -> ProblemAnalysis:
        ...

    def generate_hint(
        self,
        *,
        problem_analysis: ProblemAnalysis,
        key_point: KeyPoint,
        student_profile: dict[str, Any],
        context: AIEngineContext,
    ) -> TeachingDraft:
        ...

    def generate_explanation(
        self,
        *,
        problem_analysis: ProblemAnalysis,
        unlock_context: dict[str, Any],
        context: AIEngineContext,
    ) -> TeachingDraft:
        ...

    def generate_similar_practice(
        self,
        *,
        problem_analysis: ProblemAnalysis,
        misconception: str | None,
        limit: int,
        context: AIEngineContext,
    ) -> list[PracticeItem]:
        ...

    def summarize_session(
        self,
        *,
        events: list[dict[str, Any]],
        profile_delta: dict[str, Any],
        context: AIEngineContext,
    ) -> dict[str, Any]:
        ...


class ProviderChain:
    def __init__(
        self,
        providers: list[AIEngineProvider],
        *,
        provider_timeout_seconds: float | None = None,
        provider_retry_attempts: int | None = None,
        provider_retry_backoff_seconds: float | None = None,
    ) -> None:
        if not providers:
            raise ValueError("ProviderChain requires at least one provider")
        self.providers = providers
        self.provider_timeout_seconds = (
            provider_timeout_seconds
            if provider_timeout_seconds is not None
            else _provider_timeout_from_env()
        )
        self.provider_retry_attempts = (
            max(1, provider_retry_attempts)
            if provider_retry_attempts is not None
            else _provider_retry_attempts_from_env()
        )
        self.provider_retry_backoff_seconds = (
            max(0.0, provider_retry_backoff_seconds)
            if provider_retry_backoff_seconds is not None
            else _provider_retry_backoff_from_env()
        )
        self.last_call = ProviderCall(
            provider="none",
            model="none",
            operation="none",
            status="not_started",
            latency_ms=0,
        )

    def structure_math_problem(
        self,
        *,
        question_text: str,
        grade: int,
        context: AIEngineContext,
    ) -> ProblemAnalysis:
        last_error: Exception | None = None
        for provider in self.providers:
            for attempt_index in range(self.provider_retry_attempts):
                started = time.perf_counter()
                try:
                    analysis = self._invoke_provider(
                        provider,
                        "structure_math_problem",
                        question_text=question_text,
                        grade=grade,
                        context=context,
                    )
                    verdict = validate_problem_analysis(analysis)
                    if not verdict.valid:
                        raise ProviderError("; ".join(verdict.errors))
                    self.last_call = _provider_call(
                        provider,
                        operation="structure_math_problem",
                        status="success",
                        started=started,
                        token_estimate=_estimate_tokens(question_text),
                    )
                    return analysis
                except Exception as exc:  # provider chain must isolate provider failures
                    last_error = exc
                    self.last_call = _provider_call(
                        provider,
                        operation="structure_math_problem",
                        status="failed",
                        started=started,
                        error=exc.__class__.__name__,
                    )
                    if attempt_index + 1 < self.provider_retry_attempts:
                        self._sleep_before_provider_retry(attempt_index)
        raise ProviderError(str(last_error or "all providers failed"))

    def generate_hint(
        self,
        *,
        problem_analysis: ProblemAnalysis,
        key_point: KeyPoint,
        student_profile: dict[str, Any],
        context: AIEngineContext,
    ) -> TeachingDraft:
        return self._call_provider_method(
            "generate_hint",
            problem_analysis=problem_analysis,
            key_point=key_point,
            student_profile=student_profile,
            context=context,
        )

    def generate_explanation(
        self,
        *,
        problem_analysis: ProblemAnalysis,
        unlock_context: dict[str, Any],
        context: AIEngineContext,
    ) -> TeachingDraft:
        return self._call_provider_method(
            "generate_explanation",
            problem_analysis=problem_analysis,
            unlock_context=unlock_context,
            context=context,
        )

    def generate_similar_practice(
        self,
        *,
        problem_analysis: ProblemAnalysis,
        misconception: str | None,
        limit: int,
        context: AIEngineContext,
    ) -> list[PracticeItem]:
        return self._call_provider_method(
            "generate_similar_practice",
            problem_analysis=problem_analysis,
            misconception=misconception,
            limit=limit,
            context=context,
        )

    def summarize_session(
        self,
        *,
        events: list[dict[str, Any]],
        profile_delta: dict[str, Any],
        context: AIEngineContext,
    ) -> dict[str, Any]:
        return self._call_provider_method(
            "summarize_session",
            events=events,
            profile_delta=profile_delta,
            context=context,
        )

    def _call_provider_method(self, method_name: str, **kwargs: Any):
        last_error: Exception | None = None
        for provider in self.providers:
            for attempt_index in range(self.provider_retry_attempts):
                started = time.perf_counter()
                try:
                    result = self._invoke_provider(provider, method_name, **kwargs)
                    self.last_call = _provider_call(
                        provider,
                        operation=method_name,
                        status="success",
                        started=started,
                    )
                    return result
                except Exception as exc:
                    last_error = exc
                    self.last_call = _provider_call(
                        provider,
                        operation=method_name,
                        status="failed",
                        started=started,
                        error=exc.__class__.__name__,
                    )
                    if attempt_index + 1 < self.provider_retry_attempts:
                        self._sleep_before_provider_retry(attempt_index)
        raise ProviderError(str(last_error or f"{method_name} failed"))

    def _invoke_provider(
        self,
        provider: AIEngineProvider,
        method_name: str,
        **kwargs: Any,
    ):
        method = getattr(provider, method_name)
        if self.provider_timeout_seconds is None or self.provider_timeout_seconds <= 0:
            return method(**kwargs)

        future = _PROVIDER_EXECUTOR.submit(method, **kwargs)
        try:
            return future.result(timeout=self.provider_timeout_seconds)
        except FutureTimeoutError as exc:
            future.cancel()
            provider_name = getattr(provider, "provider_name", provider.__class__.__name__)
            raise ProviderError(
                f"{provider_name}.{method_name} timed out after "
                f"{self.provider_timeout_seconds:.1f}s"
            ) from exc

    def _sleep_before_provider_retry(self, attempt_index: int) -> None:
        if self.provider_retry_backoff_seconds <= 0:
            return
        time.sleep(self.provider_retry_backoff_seconds * (2**attempt_index))


class DeterministicFallbackProvider:
    provider_name = "deterministic_fallback"
    model_name = "local_rules_v0.1"

    def __init__(self, *, assets: TeachingAssetLibrary | None = None) -> None:
        self.assets = assets or DEFAULT_MATH_ASSET_LIBRARY

    def structure_math_problem(
        self,
        *,
        question_text: str,
        grade: int,
        context: AIEngineContext,
    ) -> ProblemAnalysis:
        if _looks_like_capacity_round_up(question_text):
            return self._structure_capacity_round_up(question_text, grade)
        if _looks_like_times_five(question_text):
            return self._structure_times_five(question_text, grade)
        return self._structure_generic_math_problem(question_text, grade)

    def generate_hint(
        self,
        *,
        problem_analysis: ProblemAnalysis,
        key_point: KeyPoint,
        student_profile: dict[str, Any],
        context: AIEngineContext,
    ) -> TeachingDraft:
        return TeachingDraft(
            action="key_point_hint",
            hint_level=_hint_level_from_stage(key_point.release_stage),
            exposes_final_answer=False,
            text=key_point.child_prompt,
            metadata={
                "provider": self.provider_name,
                "model": self.model_name,
                "problem_type": problem_analysis.problem_type,
                "key_point_id": key_point.id,
                "skill_ids": problem_analysis.skill_ids,
                "misconception_ids": problem_analysis.misconception_ids,
            },
        )

    def generate_explanation(
        self,
        *,
        problem_analysis: ProblemAnalysis,
        unlock_context: dict[str, Any],
        context: AIEngineContext,
    ) -> TeachingDraft:
        return TeachingDraft(
            action="explanation",
            exposes_final_answer=True,
            text=f"现在可以完整讲解：先看关键条件，再按步骤解决。最终结果是{problem_analysis.final_answer}。",
            metadata={"provider": self.provider_name, "model": self.model_name},
        )

    def generate_similar_practice(
        self,
        *,
        problem_analysis: ProblemAnalysis,
        misconception: str | None,
        limit: int,
        context: AIEngineContext,
    ) -> list[PracticeItem]:
        return build_similar_practice_items(
            knowledge_point=problem_analysis.problem_type,
            misconception_tag=misconception,
            limit=limit,
            source_question=_source_question_from_analysis(problem_analysis),
        )

    def summarize_session(
        self,
        *,
        events: list[dict[str, Any]],
        profile_delta: dict[str, Any],
        context: AIEngineContext,
    ) -> dict[str, Any]:
        return {
            "summary": "本次学习已完成分步引导。",
            "profile_delta": profile_delta,
            "event_count": len(events),
        }

    def _structure_capacity_round_up(self, question_text: str, grade: int) -> ProblemAnalysis:
        numbers = [int(value) for value in re.findall(r"\d+", question_text)]
        class_count, per_class, capacity = _pick_capacity_numbers(numbers)
        total = class_count * per_class
        quotient, remainder = divmod(total, capacity)
        needed = quotient + (1 if remainder else 0)
        final_answer = f"{needed}辆"
        skill_ids = [
            "math_multiplication_total_count",
            "math_remainder_division",
            "math_capacity_round_up",
        ]
        misconception_ids = [
            "math_capacity_stopped_at_total_count",
            "math_capacity_ignored_remainder_round_up",
            "math_capacity_copied_capacity",
            "math_capacity_misread_at_least",
        ]
        return ProblemAnalysis(
            subject="math",
            grade=grade,
            problem_type="capacity_round_up",
            skill_ids=skill_ids,
            misconception_ids=misconception_ids,
            concept_card_ids=self.assets.concept_card_ids_for_skills(skill_ids),
            knowledge_points=["乘法求总数", "有余数除法", "限载进一"],
            conditions=[
                {"id": "group_count", "text": f"一共有{class_count}组", "value": class_count, "unit": "组"},
                {"id": "per_group", "text": f"每组{per_class}人", "value": per_class, "unit": "人/组"},
                {"id": "capacity", "text": f"每辆最多{capacity}人", "value": capacity, "unit": "人/辆"},
            ],
            target="至少需要多少辆车或几组",
            solution_steps=[
                {"id": "step_total_people", "goal": "先求总数", "expression": f"{class_count} × {per_class}", "result": str(total)},
                {"id": "step_divide_capacity", "goal": "按限载分组", "expression": f"{total} ÷ {capacity}", "result": f"{quotient}余{remainder}"},
                {"id": "step_round_up", "goal": "有余数进一", "expression": f"{quotient} + 1", "result": str(needed)},
            ],
            final_answer=final_answer,
            common_misconceptions=[
                {"tag": "math_capacity_stopped_at_total_count", "description": "只算出总人数就停止"},
                {"tag": "math_capacity_ignored_remainder_round_up", "description": "有余数但没有进一"},
                {"tag": "math_capacity_copied_capacity", "description": "把限载人数直接当答案"},
            ],
            key_points=[
                {
                    "id": "kp_total_people",
                    "name": "先求总人数",
                    "teaching_goal": "先把几组人数合起来",
                    "release_stage": "HINT_STEP_1",
                    "unlock_condition": "question_started",
                    "child_prompt": f"先不急着回答几辆车。{class_count}组，每组{per_class}人，一共有多少人？",
                    "expected_child_response": [str(total), f"{total}人", f"{class_count}×{per_class}={total}"],
                    "forbidden_content": [final_answer],
                    "partial_misconception_tag": "math_capacity_stopped_at_total_count",
                },
                {
                    "id": "kp_capacity_check",
                    "name": "判断车辆容量",
                    "teaching_goal": "用总数除以每辆最多人数",
                    "release_stage": "HINT_STEP_2",
                    "unlock_condition": "total_people_mastered",
                    "child_prompt": f"你已经知道一共有{total}人。每辆最多{capacity}人，先想想{quotient}辆最多能坐多少人？还剩人吗？",
                    "expected_child_response": [str(quotient * capacity), f"{quotient}余{remainder}", f"剩{remainder}人"],
                    "misconception_responses": {
                        "math_capacity_ignored_remainder_round_up": [str(quotient), f"{quotient}辆"]
                    },
                    "forbidden_content": [final_answer],
                    "partial_misconception_tag": "math_capacity_ignored_remainder_round_up",
                },
                {
                    "id": "kp_round_up",
                    "name": "有剩余也要加一辆",
                    "teaching_goal": "理解至少需要时，有剩余就要再加一组",
                    "release_stage": "HINT_STEP_3",
                    "unlock_condition": "capacity_checked",
                    "child_prompt": "如果还有人没坐上，这些人是不是也需要一辆车？",
                    "expected_child_response": ["需要", "还要一辆", str(needed), final_answer],
                    "forbidden_content": [],
                },
            ],
            confidence=0.82,
            source=self.provider_name,
        )

    def _structure_times_five(self, question_text: str, grade: int) -> ProblemAnalysis:
        factor = _times_five_factor(question_text)
        answer = factor * 5
        skill_ids = ["math_two_digit_times_one_digit"]
        misconception_ids = ["math_multiplication_treated_x5_like_x10"]
        return ProblemAnalysis(
            subject="math",
            grade=grade,
            problem_type="two_digit_times_one_digit",
            skill_ids=skill_ids,
            misconception_ids=misconception_ids,
            concept_card_ids=self.assets.concept_card_ids_for_skills(skill_ids),
            knowledge_points=["两位数乘一位数"],
            conditions=[
                {"id": "factor", "text": f"{factor}", "value": factor, "unit": ""},
                {"id": "multiplier", "text": "乘以5", "value": 5, "unit": ""},
            ],
            target="求乘积",
            solution_steps=[
                {
                    "id": "step_times_ten",
                    "goal": "先想乘以10",
                    "expression": f"{factor} × 10",
                    "result": str(factor * 10),
                },
                {
                    "id": "step_half",
                    "goal": "乘以5是乘以10的一半",
                    "expression": f"{factor * 10} ÷ 2",
                    "result": str(answer),
                },
            ],
            final_answer=str(answer),
            common_misconceptions=[
                {
                    "tag": "math_multiplication_treated_x5_like_x10",
                    "description": "把乘以5当成乘以10",
                }
            ],
            key_points=[
                {
                    "id": "kp_times_ten_relation",
                    "name": "先想乘以10",
                    "teaching_goal": "让孩子用乘以10搭桥理解乘以5",
                    "release_stage": "HINT_STEP_1",
                    "unlock_condition": "question_started",
                    "child_prompt": f"先不急着算最后答案。你先想一想：{factor} × 10 会是多少？乘以 5 和乘以 10 有什么关系？",
                    "expected_child_response": [str(factor * 10), f"{factor * 10}的一半"],
                    "misconception_responses": {
                        "math_multiplication_treated_x5_like_x10": [str(factor * 10)]
                    },
                    "forbidden_content": [str(answer)],
                    "partial_misconception_tag": "math_multiplication_treated_x5_like_x10",
                }
            ],
            confidence=0.9,
            source=self.provider_name,
        )

    def _structure_generic_math_problem(self, question_text: str, grade: int) -> ProblemAnalysis:
        skill_ids = self.assets.match_skill_ids([question_text]) or ["math_read_conditions"]
        concept_card_ids = self.assets.concept_card_ids_for_skills(skill_ids)
        arithmetic_answer = _safe_arithmetic_expression_answer(question_text)
        final_answer = arithmetic_answer or _best_effort_final_answer(question_text)
        forbidden_content = [] if final_answer == "待确认" else [final_answer]
        return ProblemAnalysis(
            subject="math",
            grade=grade,
            problem_type=skill_ids[0].replace("math_", ""),
            skill_ids=skill_ids,
            misconception_ids=["math_unknown"],
            concept_card_ids=concept_card_ids,
            knowledge_points=[self.assets.require_skill(skill_ids[0]).name],
            conditions=[{"id": "question_text", "text": question_text, "value": None, "unit": ""}],
            target="解决题目要求的问题",
            solution_steps=[
                {"id": "step_read", "goal": "读题找条件", "expression": "", "result": ""},
            ],
            final_answer=final_answer,
            common_misconceptions=[
                {"tag": "math_unknown", "description": "需要继续观察孩子的尝试"}
            ],
            key_points=[
                {
                    "id": "kp_read_conditions",
                    "name": "先读题找条件",
                    "teaching_goal": "让孩子先说出已知条件和问题",
                    "release_stage": "HINT_STEP_1",
                    "unlock_condition": "question_started",
                    "child_prompt": "先不急着算答案。你能先说说题目告诉了我们什么，要求什么吗？",
                    "expected_child_response": [],
                    "forbidden_content": forbidden_content,
                }
            ],
            confidence=0.72 if arithmetic_answer is not None else 0.45,
            source=self.provider_name,
        )


class DeepTutorProvider:
    provider_name = "deeptutor"
    model_name = "deeptutor_orchestrator"

    def __init__(self, *, structurer: LLMMathStructurer | None = None) -> None:
        self.structurer = structurer or LLMMathStructurer()
        self.fallback = DeterministicFallbackProvider()

    def structure_math_problem(
        self,
        *,
        question_text: str,
        grade: int,
        context: AIEngineContext,
    ) -> ProblemAnalysis:
        return self.structurer.analyze(
            question_text=question_text,
            grade=grade,
            subject="math",
        )

    def generate_hint(self, **kwargs: Any) -> TeachingDraft:
        return self.fallback.generate_hint(**kwargs)

    def generate_explanation(self, **kwargs: Any) -> TeachingDraft:
        return self.fallback.generate_explanation(**kwargs)

    def generate_similar_practice(self, **kwargs: Any) -> list[PracticeItem]:
        return self.fallback.generate_similar_practice(**kwargs)

    def summarize_session(self, **kwargs: Any) -> dict[str, Any]:
        return self.fallback.summarize_session(**kwargs)


class DeepSeekProvider(DeepTutorProvider):
    provider_name = "deepseek"

    def __init__(self, *, api_key: str | None = None, model_name: str = DEFAULT_DEEPSEEK_MODEL) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model_name = model_name
        super().__init__()


class OllamaProvider(DeepTutorProvider):
    provider_name = "ollama"

    def __init__(self, *, model_name: str = "qwen2.5:7b") -> None:
        self.model_name = model_name
        super().__init__()


def _provider_call(
    provider: AIEngineProvider,
    *,
    operation: str,
    status: str,
    started: float,
    token_estimate: int = 0,
    error: str | None = None,
) -> ProviderCall:
    return ProviderCall(
        provider=getattr(provider, "provider_name", provider.__class__.__name__),
        model=getattr(provider, "model_name", "unknown"),
        operation=operation,
        status=status,
        latency_ms=max(0, int((time.perf_counter() - started) * 1000)),
        token_estimate=token_estimate,
        error=error,
    )


def _provider_timeout_from_env() -> float:
    raw = os.getenv("SONGGUO_AI_PROVIDER_TIMEOUT_SECONDS", "20").strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 20.0


def _provider_retry_attempts_from_env() -> int:
    raw = os.getenv("SONGGUO_AI_PROVIDER_RETRY_ATTEMPTS", "2").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 2


def _provider_retry_backoff_from_env() -> float:
    raw = os.getenv("SONGGUO_AI_PROVIDER_RETRY_BACKOFF_SECONDS", "0.4").strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.4


def _source_question_from_analysis(problem_analysis: ProblemAnalysis) -> str | None:
    for condition in problem_analysis.conditions:
        condition_id = getattr(condition, "id", None)
        condition_text = getattr(condition, "text", None)
        if condition_id == "question_text" and condition_text:
            return str(condition_text)
    return None


def _looks_like_capacity_round_up(question_text: str) -> bool:
    return (
        "至少" in question_text
        and any(word in question_text for word in ["限坐", "最多", "限载", "每辆", "每组"])
        and len(re.findall(r"\d+", question_text)) >= 3
    )


def _looks_like_times_five(question_text: str) -> bool:
    return _times_five_match(question_text) is not None


def _times_five_factor(question_text: str) -> int:
    match = _times_five_match(question_text)
    if not match:
        return 1
    left = int(match.group("left"))
    right = int(match.group("right"))
    return right if left == 5 else left


def _times_five_match(question_text: str):
    match = re.search(
        r"(?P<left>\d+)\s*(?:x|×|\*)\s*(?P<right>\d+)",
        question_text,
        re.IGNORECASE,
    )
    if not match:
        return None
    return match if 5 in {int(match.group("left")), int(match.group("right"))} else None


def _pick_capacity_numbers(numbers: list[int]) -> tuple[int, int, int]:
    if len(numbers) < 3:
        return 1, max(numbers[0], 1) if numbers else 1, max(numbers[-1], 1) if numbers else 1
    return numbers[-3], numbers[-2], numbers[-1]


def _best_effort_final_answer(question_text: str) -> str:
    expression_answer = _safe_arithmetic_expression_answer(question_text)
    if expression_answer is not None:
        return expression_answer
    numbers = [int(value) for value in re.findall(r"\d+", question_text)]
    if len(numbers) >= 2 and any(op in question_text for op in ["×", "x", "*"]):
        return str(numbers[0] * numbers[1])
    return "待确认"


def _safe_arithmetic_expression_answer(question_text: str) -> str | None:
    expression = _extract_arithmetic_expression(question_text)
    if not expression:
        return None
    try:
        parsed = ast.parse(expression, mode="eval")
        value = _eval_arithmetic_ast(parsed.body)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return None
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.6g}"
    return str(value)


def _extract_arithmetic_expression(question_text: str) -> str | None:
    text = question_text.strip()
    if not text:
        return None
    text = text.replace("×", "*").replace("÷", "/").replace("x", "*").replace("X", "*")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"=[?？]?$", "", text)
    text = text.rstrip("?？")
    if not re.fullmatch(r"[0-9+\-*/().]+", text):
        return None
    if not re.search(r"[+\-*/]", text):
        return None
    return text


def _eval_arithmetic_ast(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _eval_arithmetic_ast(node.operand)
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
    ):
        left = _eval_arithmetic_ast(node.left)
        right = _eval_arithmetic_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        return left / right
    raise ValueError("unsupported arithmetic expression")


def _hint_level_from_stage(stage: str) -> int:
    match = re.search(r"(\d+)", stage)
    return int(match.group(1)) if match else 1


def _estimate_tokens(text: str) -> int:
    stripped = text.strip()
    return max(1, len(stripped) // 4) if stripped else 0
