from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel, Field

from songguo.backend.services.learning.answer_matching import answers_match
from songguo.backend.services.learning.teaching_assets import DEFAULT_MATH_ASSET_LIBRARY


class Condition(BaseModel):
    id: str
    text: str
    value: int | float | str | None = None
    unit: str = ""


class SolutionStep(BaseModel):
    id: str
    goal: str
    expression: str = ""
    result: str = ""
    explanation_for_backend: str = ""


class CommonMisconception(BaseModel):
    tag: str
    description: str


class KeyPoint(BaseModel):
    id: str
    name: str
    teaching_goal: str
    release_stage: str
    unlock_condition: str
    child_prompt: str
    expected_child_response: list[str] = Field(default_factory=list)
    misconception_responses: dict[str, list[str]] = Field(default_factory=dict)
    forbidden_content: list[str] = Field(default_factory=list)
    partial_misconception_tag: str | None = None
    required_before_next: bool = True


class ProblemAnalysis(BaseModel):
    subject: str
    grade: int
    problem_type: str
    skill_ids: list[str] = Field(default_factory=list)
    misconception_ids: list[str] = Field(default_factory=list)
    concept_card_ids: list[str] = Field(default_factory=list)
    knowledge_points: list[str] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)
    target: str
    solution_steps: list[SolutionStep] = Field(default_factory=list)
    final_answer: str
    common_misconceptions: list[CommonMisconception] = Field(default_factory=list)
    key_points: list[KeyPoint] = Field(default_factory=list)
    confidence: float = 0.0
    source: str = "unknown"

    @property
    def knowledge_point(self) -> str:
        return self.problem_type

    @property
    def first_key_point(self) -> KeyPoint:
        if not self.key_points:
            raise ValueError("ProblemAnalysis must include at least one key point")
        return self.key_points[0]

    def get_key_point(self, key_point_id: str | None) -> KeyPoint:
        if not key_point_id:
            return self.first_key_point
        for key_point in self.key_points:
            if key_point.id == key_point_id:
                return key_point
        raise KeyError(key_point_id)

    def next_key_point_after(self, key_point_id: str | None) -> KeyPoint | None:
        if not self.key_points:
            return None
        if not key_point_id:
            return self.first_key_point
        for index, key_point in enumerate(self.key_points):
            if key_point.id == key_point_id:
                if index + 1 < len(self.key_points):
                    return self.key_points[index + 1]
                return None
        return None


class LLMProblemParse(BaseModel):
    subject: str = "math"
    grade: int
    problem_type: str
    skill_ids: list[str] = Field(default_factory=list)
    misconception_ids: list[str] = Field(default_factory=list)
    concept_card_ids: list[str] = Field(default_factory=list)
    knowledge_points: list[str] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)
    target: str
    solution_steps: list[SolutionStep] = Field(default_factory=list)
    final_answer: str
    common_misconceptions: list[CommonMisconception] = Field(default_factory=list)
    confidence: float = 0.0
    source: str = "llm_problem_parse"


class AttemptEvaluation(BaseModel):
    correct: bool = False
    partially_correct: bool = False
    matched_key_point_id: str | None = None
    mastered_key_point_ids: list[str] = Field(default_factory=list)
    next_key_point_id: str | None = None
    misconception_tag: str | None = None
    evidence: str = ""
    next_action: str = "stay_on_current_key_point"


class ValidationResult(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)


class MathStructurer(Protocol):
    def analyze(self, *, question_text: str, grade: int, subject: str) -> ProblemAnalysis | None:
        ...


class AttemptEvaluator(Protocol):
    def evaluate_attempt(
        self,
        *,
        analysis: ProblemAnalysis,
        child_answer: str,
        current_key_point_id: str | None,
    ) -> AttemptEvaluation:
        ...


class MathFastPathRegistry:
    """Parameterized high-frequency math structuring before LLM fallback.

    Fast paths may compute verifiable facts, but they must not contain fixed
    questions. Every analysis below is derived from numbers, units and relation
    words in the current question text.
    """

    source = "math_fast_path_v0.1"

    def analyze(
        self,
        *,
        question_text: str,
        grade: int,
        subject: str,
    ) -> ProblemAnalysis | None:
        if (subject or "").lower() != "math":
            return None
        for analyzer in (
            self._analyze_remainder_division,
            self._analyze_unit_conversion,
            self._analyze_perimeter,
            self._analyze_average,
            self._analyze_arithmetic_expression,
        ):
            analysis = analyzer(question_text=question_text, grade=grade, subject=subject)
            if analysis is not None:
                return analysis
        return None

    def _analysis_from_parse(
        self,
        parse: LLMProblemParse,
        *,
        question_text: str,
    ) -> ProblemAnalysis:
        return build_problem_analysis_from_parse(parse, question_text=question_text)

    def _analyze_arithmetic_expression(
        self,
        *,
        question_text: str,
        grade: int,
        subject: str,
    ) -> ProblemAnalysis | None:
        expression = _extract_safe_arithmetic_expression(question_text)
        if not expression:
            return None
        value = _safe_arithmetic_value(expression)
        if value is None:
            return None
        if "/" in expression and isinstance(value, float) and not value.is_integer():
            return None
        numbers = [int(item) for item in re.findall(r"\d+", expression)]
        if _is_two_digit_times_five(expression, numbers):
            return self._analyze_times_five_expression(
                expression=expression,
                numbers=numbers,
                grade=grade,
                subject=subject,
                question_text=question_text,
            )
        answer = _format_number(value)
        problem_type = (
            "two_digit_times_one_digit"
            if _is_two_digit_times_one_digit(expression, numbers)
            else "mixed_operations"
        )
        skill_ids = (
            ["math_two_digit_times_one_digit"]
            if problem_type == "two_digit_times_one_digit"
            else ["math_mixed_operations"]
        )
        misconceptions = (
            [
                {
                    "tag": "math_multiplication_carry_missing",
                    "description": "乘法计算时可能漏进位或漏算某一位。",
                }
            ]
            if problem_type == "two_digit_times_one_digit"
            else [
                {
                    "tag": "math_mixed_order_wrong",
                    "description": "四则混合运算时可能没有按运算顺序计算。",
                }
            ]
        )
        return self._analysis_from_parse(
            LLMProblemParse(
                subject=subject,
                grade=grade,
                problem_type=problem_type,
                skill_ids=skill_ids,
                knowledge_points=["两位数乘一位数" if problem_type == "two_digit_times_one_digit" else "四则混合运算"],
                conditions=[{"id": "expression", "text": expression, "value": expression, "unit": ""}],
                target="计算算式结果",
                solution_steps=[
                    {
                        "id": "step_calculate",
                        "goal": "按正确运算顺序计算",
                        "expression": expression,
                        "result": answer,
                    }
                ],
                final_answer=answer,
                common_misconceptions=misconceptions,
                confidence=0.95,
                source=self.source,
            ),
            question_text=question_text,
        )

    def _analyze_times_five_expression(
        self,
        *,
        expression: str,
        numbers: list[int],
        grade: int,
        subject: str,
        question_text: str,
    ) -> ProblemAnalysis:
        factor = numbers[1] if numbers[0] == 5 else numbers[0]
        times_ten = factor * 10
        answer = factor * 5
        return self._analysis_from_parse(
            LLMProblemParse(
                subject=subject,
                grade=grade,
                problem_type="two_digit_times_one_digit",
                skill_ids=["math_two_digit_times_one_digit"],
                misconception_ids=["math_multiplication_treated_x5_like_x10"],
                knowledge_points=["两位数乘一位数"],
                conditions=[
                    {"id": "factor", "text": str(factor), "value": factor, "unit": ""},
                    {"id": "multiplier", "text": "乘以5", "value": 5, "unit": ""},
                ],
                target="求乘积",
                solution_steps=[
                    {
                        "id": "step_times_ten",
                        "goal": "先想乘以10",
                        "expression": f"{factor} × 10",
                        "result": str(times_ten),
                    },
                    {
                        "id": "step_half",
                        "goal": "乘以5是乘以10的一半",
                        "expression": "",
                        "result": str(answer),
                    },
                ],
                final_answer=str(answer),
                common_misconceptions=[
                    {
                        "tag": "treated_x5_like_x10",
                        "description": "把乘以5当成乘以10。",
                    }
                ],
                confidence=0.95,
                source=self.source,
            ),
            question_text=question_text or expression,
        )

    def _analyze_remainder_division(
        self,
        *,
        question_text: str,
        grade: int,
        subject: str,
    ) -> ProblemAnalysis | None:
        if "平均" not in question_text or not any(token in question_text for token in ["还剩", "剩几", "余"]):
            return None
        numbers = [int(item) for item in re.findall(r"\d+", question_text)]
        if len(numbers) < 2:
            return None
        total, divisor = numbers[0], numbers[1]
        if divisor <= 0:
            return None
        quotient, remainder = divmod(total, divisor)
        item_unit = _unit_after_first_number(question_text) or ""
        quotient_unit = _quotient_unit_for_remainder_question(question_text)
        expression = f"{total} ÷ {divisor}"
        result = f"{quotient}余{remainder}"
        if quotient_unit.startswith("每"):
            final_answer = f"{quotient_unit}{quotient}{item_unit}，还剩{remainder}{item_unit}"
        else:
            final_answer = f"{quotient}{quotient_unit}，还剩{remainder}{item_unit}"
        return self._analysis_from_parse(
            LLMProblemParse(
                subject=subject,
                grade=grade,
                problem_type="division_with_remainder",
                skill_ids=["math_remainder_division", "math_equal_grouping"],
                misconception_ids=[
                    "math_division_quotient_remainder_swap",
                    "math_division_equal_group_direction",
                ],
                knowledge_points=["有余数除法", "平均分问题"],
                conditions=[
                    {"id": "total", "text": f"总数是{total}{item_unit}", "value": total, "unit": item_unit},
                    {"id": "divisor", "text": f"按{divisor}分", "value": divisor, "unit": ""},
                ],
                target="求商和余数",
                solution_steps=[
                    {
                        "id": "step_divide_with_remainder",
                        "goal": "求商和余数",
                        "expression": expression,
                        "result": result,
                    }
                ],
                final_answer=final_answer,
                common_misconceptions=[
                    {
                        "tag": "math_division_quotient_remainder_swap",
                        "description": "把商和余数的位置写反。",
                    },
                    {
                        "tag": "math_division_equal_group_direction",
                        "description": "没有分清题目要求每份数还是份数。",
                    },
                ],
                confidence=0.93,
                source=self.source,
            ),
            question_text=question_text,
        )

    def _analyze_unit_conversion(
        self,
        *,
        question_text: str,
        grade: int,
        subject: str,
    ) -> ProblemAnalysis | None:
        conversions = [
            (r"(\d+)米(\d+)厘米", "厘米", 100, "米", "厘米"),
            (r"(\d+)千克(\d+)克", "克", 1000, "千克", "克"),
            (r"(\d+)元(\d+)角", "角", 10, "元", "角"),
        ]
        for pattern, final_unit, rate, big_unit, small_unit in conversions:
            match = re.search(pattern, question_text)
            if not match:
                continue
            big = int(match.group(1))
            small = int(match.group(2))
            converted = big * rate + small
            operation = _unit_conversion_operation_after_match(
                question_text[match.end() :],
                unit=small_unit,
            )
            if operation is not None:
                op_word, op_value, op_sign = operation
                answer = converted + op_sign * op_value
                op_symbol = "+" if op_sign > 0 else "-"
                problem_type = "unit_conversion_arithmetic"
                knowledge_points = ["单位换算", "单位换算应用题"]
                target = f"先换算成{final_unit}，再根据题意计算结果"
                solution_steps = [
                    {
                        "id": "step_convert_compound_unit",
                        "goal": f"先把{big_unit}{small_unit}复合单位换成{final_unit}",
                        "expression": f"{big} × {rate} + {small}",
                        "result": str(converted),
                    },
                    {
                        "id": "step_apply_operation",
                        "goal": f"根据“{op_word}{op_value}{small_unit}”继续计算",
                        "expression": f"{converted} {op_symbol} {op_value}",
                        "result": str(answer),
                    },
                ]
                misconceptions = [
                    {
                        "tag": "math_unit_conversion_ignored_operation",
                        "description": "只完成单位换算，没有继续处理题目中的加减变化。",
                    },
                    {
                        "tag": "math_unit_missing_conversion",
                        "description": "不同单位直接加减，没有先换算。",
                    },
                ]
            else:
                answer = converted
                problem_type = "unit_conversion"
                knowledge_points = ["单位换算"]
                target = f"换算成{final_unit}"
                solution_steps = [
                    {
                        "id": "step_convert_big_unit",
                        "goal": f"先把{big_unit}换成{final_unit}",
                        "expression": f"{big} × {rate}",
                        "result": str(big * rate),
                    },
                    {
                        "id": "step_add_small_unit",
                        "goal": f"再加上已有的{small_unit}",
                        "expression": f"{big * rate} + {small}",
                        "result": str(answer),
                    },
                ]
                misconceptions = [
                    {"tag": "math_unit_missing_conversion", "description": "不同单位直接相加，没有先换算。"},
                ]
            return self._analysis_from_parse(
                LLMProblemParse(
                    subject=subject,
                    grade=grade,
                    problem_type=problem_type,
                    skill_ids=["math_unit_conversion"],
                    misconception_ids=[
                        misconception["tag"] for misconception in misconceptions
                    ]
                    + ["math_unit_wrong_rate"],
                    knowledge_points=knowledge_points,
                    conditions=[
                        {"id": "big_unit", "text": f"{big}{big_unit}", "value": big, "unit": big_unit},
                        {"id": "small_unit", "text": f"{small}{small_unit}", "value": small, "unit": small_unit},
                    ],
                    target=target,
                    solution_steps=solution_steps,
                    final_answer=f"{answer}{final_unit}",
                    common_misconceptions=misconceptions,
                    confidence=0.94,
                    source=self.source,
                ),
                question_text=question_text,
            )
        return None

    def _analyze_perimeter(
        self,
        *,
        question_text: str,
        grade: int,
        subject: str,
    ) -> ProblemAnalysis | None:
        if "周长" not in question_text:
            return None
        rectangle = re.search(r"长(\d+)(?P<unit>厘米|米|分米).*?宽(\d+)(?:厘米|米|分米)", question_text)
        if rectangle:
            length = int(rectangle.group(1))
            width = int(rectangle.group(3))
            unit = rectangle.group("unit")
            answer = 2 * (length + width)
            expression = f"({length} + {width}) × 2"
            return self._analysis_from_parse(
                LLMProblemParse(
                    subject=subject,
                    grade=grade,
                    problem_type="rectangle_square_perimeter",
                    skill_ids=["math_rectangle_square_perimeter", "math_length_perimeter"],
                    misconception_ids=["math_perimeter_area_confusion", "math_perimeter_missing_sides"],
                    knowledge_points=["长方形正方形周长"],
                    conditions=[
                        {"id": "length", "text": f"长{length}{unit}", "value": length, "unit": unit},
                        {"id": "width", "text": f"宽{width}{unit}", "value": width, "unit": unit},
                    ],
                    target="求长方形周长",
                    solution_steps=[
                        {
                            "id": "step_perimeter",
                            "goal": "长方形周长等于长宽和乘2",
                            "expression": expression,
                            "result": str(answer),
                        }
                    ],
                    final_answer=f"{answer}{unit}",
                    common_misconceptions=[
                        {"tag": "math_perimeter_area_confusion", "description": "把周长和面积公式混淆。"},
                    ],
                    confidence=0.94,
                    source=self.source,
                ),
                question_text=question_text,
            )
        square = re.search(r"正方形.*?边长(\d+)(?P<unit>厘米|米|分米)", question_text)
        if not square:
            return None
        side = int(square.group(1))
        unit = square.group("unit")
        answer = side * 4
        return self._analysis_from_parse(
            LLMProblemParse(
                subject=subject,
                grade=grade,
                problem_type="rectangle_square_perimeter",
                skill_ids=["math_rectangle_square_perimeter", "math_length_perimeter"],
                misconception_ids=["math_perimeter_area_confusion", "math_perimeter_missing_sides"],
                knowledge_points=["长方形正方形周长"],
                conditions=[{"id": "side", "text": f"边长{side}{unit}", "value": side, "unit": unit}],
                target="求正方形周长",
                solution_steps=[
                    {
                        "id": "step_square_perimeter",
                        "goal": "正方形周长等于边长乘4",
                        "expression": f"{side} × 4",
                        "result": str(answer),
                    }
                ],
                final_answer=f"{answer}{unit}",
                common_misconceptions=[
                    {"tag": "math_perimeter_missing_sides", "description": "只加了部分边长。"},
                ],
                confidence=0.94,
                source=self.source,
            ),
            question_text=question_text,
        )

    def _analyze_average(
        self,
        *,
        question_text: str,
        grade: int,
        subject: str,
    ) -> ProblemAnalysis | None:
        if "平均" not in question_text:
            return None
        scores = [int(item) for item in re.findall(r"(\d+)分", question_text)]
        if len(scores) < 2:
            return None
        total = sum(scores)
        average = total / len(scores)
        answer = _format_number(average)
        return self._analysis_from_parse(
            LLMProblemParse(
                subject=subject,
                grade=grade,
                problem_type="average",
                skill_ids=["math_average"],
                misconception_ids=["math_average_total_missing"],
                knowledge_points=["平均数"],
                conditions=[
                    {
                        "id": "scores",
                        "text": "、".join(f"{score}分" for score in scores),
                        "value": "、".join(str(score) for score in scores),
                        "unit": "分",
                    }
                ],
                target="求平均分",
                solution_steps=[
                    {
                        "id": "step_sum_scores",
                        "goal": "先求总分",
                        "expression": " + ".join(str(score) for score in scores),
                        "result": str(total),
                    },
                    {
                        "id": "step_divide_count",
                        "goal": "用总分除以次数",
                        "expression": f"{total} ÷ {len(scores)}",
                        "result": answer,
                    },
                ],
                final_answer=f"{answer}分",
                common_misconceptions=[
                    {"tag": "math_average_total_missing", "description": "求平均数时总量或份数用错。"},
                ],
                confidence=0.92,
                source=self.source,
            ),
            question_text=question_text,
        )


class MathProblemStructuringGateway:
    """Boundary for LLM-produced math structure plus deterministic validation.

    The gateway does not solve every math type by local rules. It accepts a
    structurer that may be backed by DeepTutor/LLM and enforces a common schema
    before the teaching state machine consumes the result.
    """

    def __init__(
        self,
        *,
        structurer: MathStructurer | None = None,
        attempt_evaluator: AttemptEvaluator | None = None,
        fast_path_registry: MathFastPathRegistry | None = None,
    ) -> None:
        self.structurer = structurer
        self.attempt_evaluator = attempt_evaluator
        self.fast_path_registry = fast_path_registry or MathFastPathRegistry()

    def analyze(
        self,
        *,
        question_text: str,
        grade: int,
        subject: str,
    ) -> ProblemAnalysis | None:
        fast_path_analysis = self.fast_path_registry.analyze(
            question_text=question_text,
            grade=grade,
            subject=subject,
        )
        if fast_path_analysis is not None:
            verdict = validate_problem_analysis(fast_path_analysis)
            if not verdict.valid:
                raise ValueError("; ".join(verdict.errors))
            return fast_path_analysis
        if not self.structurer:
            return None
        analysis = self.structurer.analyze(
            question_text=question_text,
            grade=grade,
            subject=subject,
        )
        if analysis is None:
            return None
        verdict = validate_problem_analysis(analysis)
        if not verdict.valid:
            raise ValueError("; ".join(verdict.errors))
        return analysis

    def evaluate_attempt(
        self,
        analysis: ProblemAnalysis,
        *,
        child_answer: str,
        current_key_point_id: str | None = None,
    ) -> AttemptEvaluation:
        if self.attempt_evaluator:
            return self.attempt_evaluator.evaluate_attempt(
                analysis=analysis,
                child_answer=child_answer,
                current_key_point_id=current_key_point_id,
            )
        return evaluate_attempt_by_key_points(
            analysis,
            child_answer=child_answer,
            current_key_point_id=current_key_point_id,
        )


class LLMMathStructurer:
    """LLM-backed structurer that is constrained to return ProblemAnalysis JSON."""

    def __init__(self, *, llm_func: Callable[[str], str] | None = None) -> None:
        self.llm_func = llm_func

    def analyze(self, *, question_text: str, grade: int, subject: str) -> ProblemAnalysis:
        response = self._complete(_build_structuring_prompt(question_text, grade, subject))
        payload = _extract_json_object(response)
        payload = _normalize_problem_payload(
            payload,
            subject=subject,
            grade=grade,
            question_text=question_text,
        )
        parse = LLMProblemParse.model_validate(payload)
        return build_problem_analysis_from_parse(parse, question_text=question_text)

    def _complete(self, prompt: str) -> str:
        if self.llm_func:
            return self.llm_func(prompt)
        from songguo.backend.services.learning.langchain_model_client import (
            get_langchain_llm_client,
        )

        return get_langchain_llm_client().complete_sync(
            prompt,
            system_prompt=(
                "你是小学数学题目结构化引擎。只输出 JSON object，不要输出 Markdown。"
                "输出必须符合 LLMProblemParse，不要输出 key_points、child_prompt、forbidden_content。"
            ),
            temperature=0,
        )


def validate_problem_analysis(analysis: ProblemAnalysis) -> ValidationResult:
    errors: list[str] = []
    if analysis.subject != "math":
        errors.append("subject must be math")
    if not analysis.problem_type:
        errors.append("problem_type is required")
    if not analysis.target:
        errors.append("target is required")
    if not analysis.final_answer:
        errors.append("final_answer is required")
    if not analysis.solution_steps:
        errors.append("solution_steps are required")
    if not analysis.key_points:
        errors.append("key_points are required")
    key_point_ids: set[str] = set()
    for key_point in analysis.key_points:
        if key_point.id in key_point_ids:
            errors.append(f"duplicate key_point id: {key_point.id}")
        key_point_ids.add(key_point.id)
        if not key_point.release_stage:
            errors.append(f"key_point {key_point.id} release_stage is required")
        if not key_point.child_prompt:
            errors.append(f"key_point {key_point.id} child_prompt is required")
        forbidden_items = [analysis.final_answer, *key_point.forbidden_content]
        if _contains_any_forbidden(key_point.child_prompt, forbidden_items):
            errors.append(f"key_point {key_point.id} leaks forbidden content")
    return ValidationResult(valid=not errors, errors=errors)


def build_problem_analysis_from_parse(
    parse: LLMProblemParse,
    *,
    question_text: str,
) -> ProblemAnalysis:
    normalized = parse.model_dump(mode="json")
    _bind_teaching_assets(normalized, question_text=question_text)
    normalized["common_misconceptions"] = _ensure_common_misconceptions(
        normalized.get("common_misconceptions"),
        normalized.get("misconception_ids", []),
    )
    normalized["key_points"] = _build_backend_key_points(
        solution_steps=normalized.get("solution_steps", []),
        final_answer=str(normalized.get("final_answer") or ""),
        common_misconceptions=normalized["common_misconceptions"],
    )
    return ProblemAnalysis.model_validate(normalized)


def _build_structuring_prompt(question_text: str, grade: int, subject: str) -> str:
    return f"""请把下面的小学数学题结构化为 LLMProblemParse。只输出 JSON。

要求：
1. 顶层必须是一个 JSON object。
2. 不要输出 ```json 代码块、Markdown 或解释文字。
3. 字段缺失时使用空字符串、空数组、false 或 null，不要省略必需字段。
4. 输出字段必须匹配 LLMProblemParse。
5. 必须包含 problem_type、knowledge_points、conditions、target、solution_steps、final_answer、common_misconceptions、confidence、source。
6. final_answer 必须是 string，例如 "28支"、"36"、"3辆"。
7. target 必须是 string。
8. solution_steps 每项包含 id、goal、expression、result。
9. 不要输出 key_points。
10. 不要输出 child_prompt。
11. 不要输出 forbidden_content。
12. 算式和结果必须自洽。

subject: {subject}
grade: {grade}
question: {question_text}
"""


def _extract_json_object(text: str) -> dict:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("LLM math structurer must return a JSON object")
    return value


def _normalize_problem_payload(
    payload: dict[str, Any],
    *,
    subject: str,
    grade: int,
    question_text: str,
) -> dict[str, Any]:
    """Convert useful but loose LLM JSON into the strict ProblemAnalysis schema."""
    normalized = dict(payload)
    normalized["subject"] = str(normalized.get("subject") or subject)
    normalized["grade"] = _normalize_int(normalized.get("grade"), default=grade)
    normalized["problem_type"] = _normalize_text(normalized.get("problem_type"))
    normalized["target"] = _normalize_text(normalized.get("target"))
    normalized["final_answer"] = _normalize_text(normalized.get("final_answer"))
    normalized["source"] = _normalize_text(normalized.get("source") or "llm_structured")
    normalized["skill_ids"] = _normalize_string_list(normalized.get("skill_ids"))
    normalized["misconception_ids"] = _normalize_string_list(
        normalized.get("misconception_ids")
    )
    normalized["concept_card_ids"] = _normalize_string_list(
        normalized.get("concept_card_ids")
    )
    normalized["conditions"] = _normalize_conditions(normalized.get("conditions"))
    normalized["solution_steps"] = _normalize_solution_steps(normalized.get("solution_steps"))
    normalized["common_misconceptions"] = _normalize_common_misconceptions(
        normalized.get("common_misconceptions")
    )
    normalized["key_points"] = _normalize_key_points(normalized.get("key_points"))
    normalized["confidence"] = _normalize_confidence(normalized.get("confidence"))
    _bind_teaching_assets(normalized, question_text=question_text)
    return normalized


def _bind_teaching_assets(normalized: dict[str, Any], *, question_text: str) -> None:
    """Attach stable teaching-asset ids to loose LLM output.

    LLMs often return natural-language labels such as "进一法" or temporary ids
    such as "misconception_2". The runtime and parent reports need our stable
    library ids instead.
    """

    labels = _asset_match_labels(normalized, question_text=question_text)
    skill_ids = [
        *normalized.get("skill_ids", []),
        *DEFAULT_MATH_ASSET_LIBRARY.match_skill_ids(labels),
    ]
    misconception_ids = [
        *normalized.get("misconception_ids", []),
        *DEFAULT_MATH_ASSET_LIBRARY.match_misconception_ids(labels),
    ]
    if _looks_like_capacity_round_up(labels):
        skill_ids.extend(
            [
                "math_capacity_round_up",
                "math_remainder_division",
                "math_multiplication_total_count",
            ]
        )
        misconception_ids.extend(
            [
                "math_capacity_stopped_at_total_count",
                "math_capacity_ignored_remainder_round_up",
            ]
        )

    normalized["skill_ids"] = _dedupe(skill_ids)
    normalized["misconception_ids"] = _dedupe(misconception_ids)
    normalized["common_misconceptions"] = _normalize_misconception_tags(
        normalized.get("common_misconceptions", []),
        labels=labels,
    )
    normalized["concept_card_ids"] = _dedupe(
        [
            *normalized.get("concept_card_ids", []),
            *DEFAULT_MATH_ASSET_LIBRARY.concept_card_ids_for_skills(
                normalized["skill_ids"]
            ),
        ]
    )


def _asset_match_labels(normalized: dict[str, Any], *, question_text: str) -> list[str]:
    labels: list[str] = [
        question_text,
        str(normalized.get("problem_type") or ""),
        str(normalized.get("target") or ""),
    ]
    labels.extend(_normalize_string_list(normalized.get("knowledge_points")))
    for step in normalized.get("solution_steps", []):
        if isinstance(step, dict):
            labels.extend(
                [
                    str(step.get("goal") or ""),
                    str(step.get("expression") or ""),
                    str(step.get("result") or ""),
                    str(step.get("explanation_for_backend") or ""),
                ]
            )
    for misconception in normalized.get("common_misconceptions", []):
        if isinstance(misconception, dict):
            labels.extend(
                [
                    str(misconception.get("tag") or ""),
                    str(misconception.get("description") or ""),
                ]
            )
    return labels


def _looks_like_capacity_round_up(labels: list[str]) -> bool:
    text = " ".join(labels)
    has_capacity_limit = any(token in text for token in ["每辆", "限坐", "最多坐", "最多放", "每组最多"])
    has_round_up_target = any(token in text for token in ["至少", "进一", "余数", "剩下", "余下"])
    return has_capacity_limit and has_round_up_target


def _normalize_misconception_tags(
    misconceptions: list[dict[str, str]],
    *,
    labels: list[str],
) -> list[dict[str, str]]:
    if not _looks_like_capacity_round_up(labels):
        return misconceptions
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in misconceptions:
        text = f"{item.get('tag', '')} {item.get('description', '')}"
        tag = item.get("tag") or ""
        if any(token in text for token in ["总人数", "总数", "先求人数"]):
            tag = "math_capacity_stopped_at_total_count"
        elif any(token in text for token in ["进一", "余数", "剩下", "余下", "只写2"]):
            tag = "math_capacity_ignored_remainder_round_up"
        normalized = {**item, "tag": tag}
        if normalized["tag"] not in seen:
            result.append(normalized)
            seen.add(normalized["tag"])
    if "math_capacity_stopped_at_total_count" not in seen:
        result.append(
            {
                "tag": "math_capacity_stopped_at_total_count",
                "description": "只算出总人数，没有继续判断限载分组。",
            }
        )
    if "math_capacity_ignored_remainder_round_up" not in seen:
        result.append(
            {
                "tag": "math_capacity_ignored_remainder_round_up",
                "description": "算出商后忽略剩余的人也需要再安排一组。",
            }
        )
    return result


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _normalize_conditions(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    conditions: list[dict[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if isinstance(item, str):
            conditions.append(
                {
                    "id": f"condition_{index}",
                    "text": item,
                    "value": _first_number(item),
                    "unit": "",
                }
            )
            continue
        if isinstance(item, dict):
            condition = dict(item)
            condition.setdefault("id", f"condition_{index}")
            condition["id"] = str(condition.get("id") or f"condition_{index}")
            condition.setdefault(
                "text",
                str(condition.get("name") or condition.get("description") or ""),
            )
            condition["text"] = _normalize_text(condition.get("text"))
            condition["value"] = _normalize_condition_value(condition.get("value"))
            condition.setdefault("unit", "")
            condition["unit"] = _normalize_text(condition.get("unit"))
            conditions.append(condition)
    return conditions


def _normalize_solution_steps(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    steps: list[dict[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if isinstance(item, str):
            goal, expression, result = _parse_step_text(item)
            steps.append(
                {
                    "id": f"step_{index}",
                    "goal": goal or f"步骤{index}",
                    "expression": expression,
                    "result": result,
                    "explanation_for_backend": item,
                }
            )
            continue
        if isinstance(item, dict):
            step = dict(item)
            step.setdefault("id", f"step_{index}")
            step["id"] = str(step.get("id") or f"step_{index}")
            step.setdefault("goal", str(step.get("name") or f"步骤{index}"))
            step.setdefault("expression", "")
            step.setdefault("result", "")
            step.setdefault("explanation_for_backend", "")
            step["goal"] = _normalize_text(step.get("goal"))
            step["expression"] = _normalize_text(step.get("expression"))
            step["result"] = _normalize_text(step.get("result"))
            step["explanation_for_backend"] = _normalize_text(
                step.get("explanation_for_backend")
            )
            steps.append(step)
    return steps


def _normalize_common_misconceptions(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    misconceptions: list[dict[str, str]] = []
    for index, item in enumerate(value, start=1):
        if isinstance(item, str):
            misconceptions.append(
                {
                    "tag": f"misconception_{index}",
                    "description": item,
                }
            )
            continue
        if isinstance(item, dict):
            misconception = dict(item)
            misconception.setdefault("tag", f"misconception_{index}")
            misconception.setdefault(
                "description",
                str(misconception.get("name") or misconception.get("text") or ""),
            )
            misconceptions.append(
                {
                    "tag": str(misconception.get("tag") or f"misconception_{index}"),
                    "description": str(misconception.get("description") or ""),
                }
            )
    return misconceptions


def _normalize_key_points(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    key_points: list[dict[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            continue
        key_point = dict(item)
        key_point.setdefault("id", f"key_point_{index}")
        key_point["id"] = _normalize_key_point_id(key_point.get("id"), index)
        key_point.setdefault("name", f"关键点{index}")
        key_point["name"] = _normalize_text(key_point.get("name"))
        key_point.setdefault("teaching_goal", str(key_point.get("name") or ""))
        key_point["teaching_goal"] = _normalize_text(key_point.get("teaching_goal"))
        key_point.setdefault("release_stage", f"HINT_STEP_{index}")
        key_point["release_stage"] = _normalize_text(key_point.get("release_stage"))
        key_point.setdefault("unlock_condition", "previous_key_point_mastered")
        key_point["unlock_condition"] = _normalize_text(key_point.get("unlock_condition"))
        key_point.setdefault("child_prompt", str(key_point.get("prompt") or ""))
        key_point["child_prompt"] = _normalize_text(key_point.get("child_prompt"))
        key_point["expected_child_response"] = _normalize_string_list(
            key_point.get("expected_child_response")
        )
        key_point["forbidden_content"] = _normalize_string_list(
            key_point.get("forbidden_content")
        )
        misconception_responses = key_point.get("misconception_responses")
        if isinstance(misconception_responses, dict):
            key_point["misconception_responses"] = {
                str(tag): _normalize_string_list(responses)
                for tag, responses in misconception_responses.items()
            }
        else:
            key_point["misconception_responses"] = {}
        key_points.append(key_point)
    return key_points


def _build_backend_key_points(
    *,
    solution_steps: list[dict[str, Any]],
    final_answer: str,
    common_misconceptions: list[dict[str, str]],
) -> list[dict[str, Any]]:
    if not solution_steps:
        return [
            {
                "id": "kp_read_conditions",
                "name": "先读题找条件",
                "teaching_goal": "让孩子先说出题目条件和目标",
                "release_stage": "HINT_STEP_1",
                "unlock_condition": "question_started",
                "child_prompt": "先不急着说最终答案。你能先说说题目告诉了哪些条件，要求什么吗？",
                "expected_child_response": [],
                "forbidden_content": [final_answer] if final_answer else [],
                "partial_misconception_tag": _first_common_misconception(common_misconceptions),
            }
        ]

    key_points: list[dict[str, Any]] = []
    for index, step in enumerate(solution_steps[:3], start=1):
        step_id = _normalize_key_point_id(step.get("id"), index)
        result = _normalize_text(step.get("result"))
        expected = [result] if result else []
        key_points.append(
            {
                "id": f"kp_{step_id}",
                "name": _normalize_text(step.get("goal")) or f"关键步骤{index}",
                "teaching_goal": _normalize_text(step.get("goal")) or f"完成第{index}步",
                "release_stage": f"HINT_STEP_{index}",
                "unlock_condition": "question_started" if index == 1 else "previous_key_point_mastered",
                "child_prompt": _backend_child_prompt_for_step(
                    step,
                    index=index,
                    final_answer=final_answer,
                ),
                "expected_child_response": expected,
                "forbidden_content": [final_answer] if final_answer else [],
                "partial_misconception_tag": _first_common_misconception(common_misconceptions),
            }
        )
    return key_points


def _backend_child_prompt_for_step(
    step: dict[str, Any],
    *,
    index: int,
    final_answer: str,
) -> str:
    goal = _normalize_text(step.get("goal"))
    expression = _expression_prompt_text(_normalize_text(step.get("expression")))
    if index == 1:
        if expression:
            prompt = f"先不急着说最终答案。你觉得第一步可以怎样列式？可以先想：{expression}。"
            return _safe_backend_prompt(prompt, final_answer=final_answer, index=index, goal=goal)
        prompt = f"先不急着说最终答案。我们先看第一步：{goal or '找出题目条件'}，你会怎么做？"
        return _safe_backend_prompt(prompt, final_answer=final_answer, index=index, goal=goal)
    if expression:
        prompt = f"上一步想清楚后，下一步可以看这个关系：{expression}。你觉得这一步表示什么？"
        return _safe_backend_prompt(prompt, final_answer=final_answer, index=index, goal=goal)
    prompt = f"接着看：{goal or f'第{index}步'}。你觉得该怎么继续？"
    return _safe_backend_prompt(prompt, final_answer=final_answer, index=index, goal=goal)


def _safe_backend_prompt(
    prompt: str,
    *,
    final_answer: str,
    index: int,
    goal: str,
) -> str:
    if final_answer and _contains_any_forbidden(prompt, [final_answer]):
        safe_goal = "" if _contains_any_forbidden(goal, [final_answer]) else goal
        if index == 1:
            return "先不急着说最终答案。你能先说说题目里哪些条件最关键吗？"
        return f"接着看第{index}步：{safe_goal or '继续判断关系'}。你能用自己的话说说这一步要解决什么吗？"
    return prompt


def _expression_prompt_text(expression: str) -> str:
    text = expression.strip()
    if not text:
        return ""
    for delimiter in ("=", "＝"):
        if delimiter in text:
            return text.split(delimiter, 1)[0].strip()
    return text


def _extract_safe_arithmetic_expression(question_text: str) -> str | None:
    text = question_text.strip()
    text = text.replace("×", "*").replace("÷", "/").replace("Ｘ", "*")
    text = text.replace("x", "*").replace("X", "*")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"(=|＝)[?？]?$", "", text)
    text = text.rstrip("?？")
    if not re.fullmatch(r"[0-9+\-*/().]+", text):
        return None
    if not re.search(r"[+\-*/]", text):
        return None
    return text


def _safe_arithmetic_value(expression: str) -> int | float | None:
    try:
        tree = ast.parse(expression, mode="eval")
        return _eval_arithmetic_node(tree.body)
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError):
        return None


def _eval_arithmetic_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_arithmetic_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(
        node.op,
        (ast.Add, ast.Sub, ast.Mult, ast.Div),
    ):
        left = _eval_arithmetic_node(node.left)
        right = _eval_arithmetic_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        return left / right
    raise ValueError("unsupported arithmetic expression")


def _format_number(value: int | float) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.6g}"
    return str(value)


def _is_two_digit_times_one_digit(expression: str, numbers: list[int]) -> bool:
    if not re.fullmatch(r"\d+\*\d+", expression):
        return False
    if len(numbers) != 2:
        return False
    return any(10 <= number <= 99 for number in numbers) and any(
        1 <= number <= 9 for number in numbers
    )


def _is_two_digit_times_five(expression: str, numbers: list[int]) -> bool:
    if not re.fullmatch(r"\d+\*\d+", expression):
        return False
    if len(numbers) != 2:
        return False
    return 5 in numbers and any(number > 0 for number in numbers)


def _unit_after_first_number(question_text: str) -> str | None:
    match = re.search(
        r"\d+(?P<unit>个|颗|本|支|张|页|人|朵|只|盒|袋|辆|米|厘米|分米|千克|克|元|角|分)",
        question_text,
    )
    if not match:
        return None
    return match.group("unit")


def _unit_conversion_operation_after_match(
    text_after_compound_unit: str,
    *,
    unit: str,
) -> tuple[str, int, int] | None:
    pattern = (
        r"(?P<word>剪去|用去|减去|花去|少了|减少|增加|加上|又加|又接上|接上|添上)"
        rf"\s*(?P<value>\d+){re.escape(unit)}"
    )
    match = re.search(pattern, text_after_compound_unit)
    if not match:
        return None
    word = match.group("word")
    value = int(match.group("value"))
    sign = -1 if word in {"剪去", "用去", "减去", "花去", "少了", "减少"} else 1
    return word, value, sign


def _quotient_unit_for_remainder_question(question_text: str) -> str:
    if "每只" in question_text or re.search(r"分给\d+只", question_text):
        return "每只"
    if "每人" in question_text or re.search(r"分给\d+人", question_text):
        return "每人"
    match = re.search(r"每(?P<container>袋|盒|层|排|组|车|辆|页|本|盘|篮|箱)\d+", question_text)
    if match:
        return match.group("container")
    target = re.search(r"(?:几|多少)(?P<container>袋|盒|层|排|组|车|辆|页|本|盘|篮|箱)", question_text)
    if target:
        return target.group("container")
    return "份"


def _ensure_common_misconceptions(
    misconceptions: Any,
    misconception_ids: list[str],
) -> list[dict[str, str]]:
    if misconceptions:
        return misconceptions
    if misconception_ids:
        return [
            {
                "tag": misconception_ids[0],
                "description": "需要结合孩子作答继续确认错因。",
            }
        ]
    return [{"tag": "math_unknown", "description": "需要继续观察孩子的尝试。"}]


def _first_common_misconception(misconceptions: list[dict[str, str]]) -> str | None:
    if not misconceptions:
        return None
    return misconceptions[0].get("tag") or None


def _normalize_key_point_id(value: Any, index: int) -> str:
    text = str(value or "").strip()
    if not text:
        return f"key_point_{index}"
    if re.fullmatch(r"\d+", text):
        return f"kp_{text}"
    return text


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        if "value" in value:
            answer_value = value.get("value")
            unit = value.get("unit")
            if answer_value is not None:
                return f"{answer_value}{unit or ''}".strip()
        for key in ("text", "description", "name", "target", "question"):
            current = value.get(key)
            if current is not None and str(current).strip():
                return str(current).strip()
        return ""
    return str(value).strip()


def _normalize_condition_value(value: Any) -> int | float | str | None:
    if value is None or isinstance(value, (int, float, str)):
        return value
    if isinstance(value, list):
        return "、".join(str(item) for item in value)
    return _normalize_text(value)


def _normalize_int(value: Any, *, default: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _normalize_confidence(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value or "").strip().lower()
    if not text:
        return 0.0
    confidence_map = {
        "高": 0.85,
        "high": 0.85,
        "中": 0.6,
        "medium": 0.6,
        "低": 0.35,
        "low": 0.35,
    }
    if text in confidence_map:
        return confidence_map[text]
    try:
        return float(text)
    except ValueError:
        return 0.0


def _parse_step_text(text: str) -> tuple[str, str, str]:
    goal = text
    body = text
    if "：" in text:
        goal, body = text.split("：", 1)
    elif ":" in text:
        goal, body = text.split(":", 1)
    expression = ""
    result = ""
    if "=" in body:
        expression, result_text = body.split("=", 1)
        expression = expression.strip()
        result = _format_step_result(result_text)
    return goal.strip(), expression, result


def _format_step_result(text: str) -> str:
    numbers = _number_tokens(text)
    if "……" in text or "余" in text:
        if len(numbers) >= 2:
            return f"{numbers[0]}余{numbers[1]}"
    return numbers[-1] if numbers else text.strip()


def evaluate_attempt_by_key_points(
    analysis: ProblemAnalysis,
    *,
    child_answer: str,
    current_key_point_id: str | None = None,
) -> AttemptEvaluation:
    answer = child_answer.strip()
    current = analysis.get_key_point(current_key_point_id)
    next_key_point = analysis.next_key_point_after(current.id)

    if _answer_matches(answer, analysis.final_answer):
        return AttemptEvaluation(
            correct=True,
            partially_correct=False,
            matched_key_point_id=current.id,
            mastered_key_point_ids=[current.id],
            misconception_tag=None,
            evidence="孩子给出了整题最终答案。",
            next_action="complete_session",
        )

    if _matches_any(answer, current.expected_child_response):
        misconception = current.partial_misconception_tag
        if not misconception and next_key_point is not None:
            misconception = _first_misconception_tag(analysis)
        return AttemptEvaluation(
            correct=False,
            partially_correct=True,
            matched_key_point_id=current.id,
            mastered_key_point_ids=[current.id],
            next_key_point_id=next_key_point.id if next_key_point else None,
            misconception_tag=misconception,
            evidence=f"孩子回答 {answer}，说明当前关键点已有进展。",
            next_action="release_next_key_point" if next_key_point else "complete_session",
        )

    if _matches_non_condition_step_number(answer, current.expected_child_response, analysis):
        misconception = current.partial_misconception_tag or _round_up_misconception_tag(analysis)
        return AttemptEvaluation(
            correct=False,
            partially_correct=True,
            matched_key_point_id=current.id,
            mastered_key_point_ids=[],
            next_key_point_id=next_key_point.id if next_key_point else None,
            misconception_tag=misconception,
            evidence=f"孩子回答 {answer}，命中了当前步骤里的中间结果，但还没有完成关键判断。",
            next_action="release_next_key_point" if next_key_point else "stay_on_current_key_point",
        )

    for tag, responses in current.misconception_responses.items():
        if _matches_any(answer, responses):
            return AttemptEvaluation(
                correct=False,
                partially_correct=True,
                matched_key_point_id=current.id,
                mastered_key_point_ids=[],
                next_key_point_id=next_key_point.id if next_key_point else None,
                misconception_tag=tag,
                evidence=f"孩子回答 {answer}，命中了关键点相关错因。",
                next_action="release_next_key_point" if next_key_point else "stay_on_current_key_point",
            )

    return AttemptEvaluation(
        correct=False,
        partially_correct=False,
        matched_key_point_id=None,
        mastered_key_point_ids=[],
        next_key_point_id=current.id,
        misconception_tag="unknown_misconception",
        evidence=f"孩子回答 {answer}，暂未匹配当前关键点。",
        next_action="stay_on_current_key_point",
    )


def expected_answer_for_leakage(analysis: ProblemAnalysis) -> str | None:
    final_answer = reliable_final_answer(analysis)
    if not final_answer:
        return None
    numeric = _first_number(final_answer)
    return numeric or final_answer


def reliable_final_answer(analysis: ProblemAnalysis) -> str | None:
    final_answer = str(analysis.final_answer or "").strip()
    if not final_answer or final_answer == "待确认":
        return None
    if (
        analysis.source == "deterministic_fallback"
        and analysis.problem_type == "read_conditions"
        and analysis.confidence < 0.6
    ):
        return None
    return final_answer


def _first_misconception_tag(analysis: ProblemAnalysis) -> str | None:
    if not analysis.common_misconceptions:
        return None
    return analysis.common_misconceptions[0].tag


def _round_up_misconception_tag(analysis: ProblemAnalysis) -> str | None:
    for misconception in analysis.common_misconceptions:
        tag = misconception.tag
        text = f"{misconception.tag} {misconception.description}"
        if "remainder" in text or "余" in text or "进一" in text or "只写" in text:
            return tag
    return _first_misconception_tag(analysis)


def _matches_any(answer: str, candidates: list[str]) -> bool:
    return any(_answer_matches(answer, candidate) for candidate in candidates)


def _matches_non_condition_step_number(
    answer: str,
    candidates: list[str],
    analysis: ProblemAnalysis,
) -> bool:
    answer_numbers = _number_tokens(_normalize_answer(answer))
    if len(answer_numbers) != 1:
        return False
    answer_number = answer_numbers[0]
    if answer_number == _first_number(analysis.final_answer or ""):
        return False
    condition_numbers = {
        number
        for condition in analysis.conditions
        for number in _number_tokens(str(condition.value if condition.value is not None else condition.text))
    }
    if answer_number in condition_numbers:
        return False
    for candidate in candidates:
        if answer_number in _number_tokens(candidate):
            return True
    return False


def _answer_matches(answer: str, expected: str | int | float | None) -> bool:
    return answers_match(answer, expected)


def _normalize_answer(value: str) -> str:
    return re.sub(r"[\s，。,.？?！!：:；;、]", "", value.strip().lower())


def _first_number(value: str) -> str | None:
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    return match.group(0) if match else None


def _number_tokens(value: str) -> list[str]:
    return re.findall(r"-?\d+(?:\.\d+)?", value)


def _contains_any_forbidden(text: str, forbidden_items: list[str | None]) -> bool:
    normalized_text = _normalize_answer(text)
    for item in forbidden_items:
        if not item:
            continue
        normalized_item = _normalize_answer(item)
        if normalized_item and normalized_item in normalized_text:
            return True
        number = _first_number(str(item))
        if number and _bare_number_leaks_answer(text, str(item), number):
            return True
    return False


def _bare_number_leaks_answer(text: str, forbidden_item: str, number: str) -> bool:
    if not re.search(rf"(?<!\d){re.escape(number)}(?!\d)", text):
        return False
    normalized_item = _normalize_answer(forbidden_item)
    if normalized_item == number:
        return True
    answer_cues = r"(答案|结果|等于|一共是|至少需要|需要|就是|应该是)"
    return bool(
        re.search(rf"{answer_cues}\D{{0,4}}{re.escape(number)}(?!\d)", text)
        or re.search(rf"(?<!\d){re.escape(number)}\D{{0,4}}{answer_cues}", text)
    )
