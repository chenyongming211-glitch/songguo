from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, Field


class MathQuestionKind(StrEnum):
    ORAL_CALCULATION = "oral_calculation"
    GROUPED_ORAL_CALCULATION = "grouped_oral_calculation"
    VERTICAL_CALCULATION_BLOCK = "vertical_calculation_block"
    COMPARISON_SIGN = "comparison_sign"
    GROUPED_COMPARISON_SIGN = "grouped_comparison_sign"
    TRUE_FALSE = "true_false"
    GROUPED_TRUE_FALSE = "grouped_true_false"
    CHOICE = "choice"
    FILL_BLANK = "fill_blank"
    GROUPED_FILL_BLANK = "grouped_fill_blank"
    WORD_PROBLEM = "word_problem"
    UNKNOWN = "unknown"


class QuestionTypeRoute(BaseModel):
    subject: str = "math"
    kind: MathQuestionKind = MathQuestionKind.UNKNOWN
    question_type_id: str = "math_unknown"
    evaluation_strategy: str = "math_gateway"
    confidence: float = 0.0
    evidence: list[str] = Field(default_factory=list)


def route_math_question_type(
    *,
    question_text: str,
    child_answer: str | None = None,
    ocr_action: str = "",
) -> QuestionTypeRoute:
    question = _normalize_question(question_text)
    answer = _normalize_answer(child_answer)
    ocr_action = (ocr_action or "").strip()
    ocr_evidence = f"ocr_action:{ocr_action}" if ocr_action else ""

    if not question:
        return _route(
            MathQuestionKind.UNKNOWN,
            "math_unknown",
            "manual_confirm",
            0.0,
            _evidence("empty_question", ocr_evidence),
        )
    if _has_choice_options(question) and answer and not re.fullmatch(r"[A-Da-d]", answer):
        return _route(
            MathQuestionKind.CHOICE,
            "math_choice",
            "manual_confirm",
            0.82,
            _evidence("choice_answer_not_option_letter", ocr_evidence),
        )
    if _looks_like_choice_item(question, answer):
        return _route(
            MathQuestionKind.CHOICE,
            "math_choice",
            "deterministic",
            0.88,
            _evidence("choice_options_detected", ocr_evidence),
        )
    if answer and "解题过程" in question and _looks_like_word_problem(question):
        return _route(
            MathQuestionKind.WORD_PROBLEM,
            "math_word_problem",
            "math_gateway",
            0.78,
            _evidence("word_problem_language_detected", ocr_evidence),
        )
    if _looks_like_vertical_calculation_block(question):
        strategy = (
            "item_split_required"
            if _vertical_process_equation_count(question) >= 2
            else "manual_confirm"
        )
        return _route(
            MathQuestionKind.VERTICAL_CALCULATION_BLOCK,
            "math_vertical_calculation_block",
            strategy,
            0.9,
            _evidence("vertical_calculation_block_needs_coordinate_review", ocr_evidence),
        )
    if _looks_like_grouped_oral_calculation_block(question):
        return _route(
            MathQuestionKind.GROUPED_ORAL_CALCULATION,
            "math_grouped_oral_calculation",
            "item_split_required",
            0.9,
            _evidence("grouped_oral_calculation_block", ocr_evidence),
        )
    if _looks_like_grouped_true_false_block(question, answer):
        return _route(
            MathQuestionKind.GROUPED_TRUE_FALSE,
            "math_grouped_true_false",
            "item_split_required",
            0.88,
            _evidence("grouped_true_false_block", ocr_evidence),
        )
    if _looks_like_grouped_comparison_sign_item(question, answer):
        return _route(
            MathQuestionKind.GROUPED_COMPARISON_SIGN,
            "math_grouped_comparison_sign",
            "item_split_required",
            0.88,
            _evidence("grouped_comparison_block", ocr_evidence),
        )
    if _looks_like_comparison_sign_item(question, answer):
        return _route(
            MathQuestionKind.COMPARISON_SIGN,
            "math_comparison_sign",
            "deterministic",
            0.92,
            _evidence("comparison_blank_between_expressions", ocr_evidence),
        )
    if _looks_like_true_false_item(question, answer):
        return _route(
            MathQuestionKind.TRUE_FALSE,
            "math_true_false_fact",
            "deterministic",
            0.9,
            _evidence("true_false_answer_mark", ocr_evidence),
        )
    if _looks_like_calendar_fill_blank_noise(question, answer):
        return _route(
            MathQuestionKind.FILL_BLANK,
            "math_fill_blank",
            "manual_confirm",
            0.76,
            _evidence("calendar_fill_blank_ocr_noise", ocr_evidence),
        )
    if _looks_like_uncertain_grouped_fill_blank(question, answer):
        return _route(
            MathQuestionKind.GROUPED_FILL_BLANK,
            "math_grouped_fill_blank",
            "item_split_required",
            0.86,
            _evidence("multi_blank_answer_needs_alignment", ocr_evidence),
        )
    if _looks_like_direct_arithmetic(question, answer) or ocr_action == "RecognizeEduOralCalculation":
        return _route(
            MathQuestionKind.ORAL_CALCULATION,
            "math_oral_calculation",
            "deterministic",
            0.9,
            _evidence("direct_arithmetic_expression", ocr_evidence),
        )
    if _looks_like_embedded_arithmetic_answer(question):
        return _route(
            MathQuestionKind.FILL_BLANK,
            "math_fill_blank",
            "deterministic",
            0.84,
            _evidence("embedded_arithmetic_answer_detected", ocr_evidence),
        )
    if _looks_like_simple_word_blank(question, answer):
        return _route(
            MathQuestionKind.WORD_PROBLEM,
            "math_fill_blank_word_problem",
            "deterministic",
            0.84,
            _evidence("simple_word_problem_blank", ocr_evidence),
        )
    if _looks_like_fill_blank_item(question, answer):
        return _route(
            MathQuestionKind.FILL_BLANK,
            "math_fill_blank",
            "deterministic",
            0.82,
            _evidence("blank_answer_detected", ocr_evidence),
        )
    if _looks_like_word_problem(question):
        return _route(
            MathQuestionKind.WORD_PROBLEM,
            "math_word_problem",
            "math_gateway",
            0.78,
            _evidence("word_problem_language_detected", ocr_evidence),
        )
    return _route(
        MathQuestionKind.UNKNOWN,
        "math_unknown",
        "math_gateway",
        0.3,
        _evidence("no_specific_question_type_matched", ocr_evidence),
    )


def _route(
    kind: MathQuestionKind,
    question_type_id: str,
    evaluation_strategy: str,
    confidence: float,
    evidence: str | list[str],
) -> QuestionTypeRoute:
    return QuestionTypeRoute(
        kind=kind,
        question_type_id=question_type_id,
        evaluation_strategy=evaluation_strategy,
        confidence=confidence,
        evidence=[evidence] if isinstance(evidence, str) else evidence,
    )


def _evidence(*values: str) -> list[str]:
    return [value for value in values if value]


def _looks_like_grouped_comparison_sign_item(question: str, answer: str) -> bool:
    if _comparison_pair_count(question) < 2:
        return False
    return bool(_comparison_answer_parts(answer) or _has_comparison_prompt(question))


def _looks_like_comparison_sign_item(question: str, answer: str) -> bool:
    if answer and re.fullmatch(r"[<>＝=≤≥]+", answer):
        return True
    if _has_comparison_prompt(question):
        return True
    return False


def _has_comparison_prompt(question: str) -> bool:
    return any(token in question for token in ("填上“>”“<”或“=”", "填上><或=", "比较大小"))


def _looks_like_grouped_true_false_block(question: str, answer: str) -> bool:
    if "判断" not in question:
        return False
    mark_count = len(re.findall(r"[(（]\s*[√✓Vv×xX对错]\s*[)）]?", question))
    numbered_count = len(re.findall(r"(?<!\d)\d+[.．、)]", question))
    answer_mark_count = len(re.findall(r"[√✓Vv×xX对错]", answer))
    return mark_count >= 2 or numbered_count >= 2 or answer_mark_count >= 2


def _looks_like_calendar_fill_blank_noise(question: str, answer: str) -> bool:
    if _blank_count(question) < 2:
        return False
    if len(_split_answer_parts(answer)) < 2:
        return False
    if not any(token in question for token in ("月历", "日历", "霜降", "立冬", "重阳", "相差", "日期")):
        return False
    return _blank_count(question) != len(_split_answer_parts(answer)) or _comparison_pair_count(question) >= 1


def _looks_like_grouped_oral_calculation_block(question: str) -> bool:
    equation_count = len(_arithmetic_equation_parts(question))
    if equation_count < 2:
        return False
    if any(token in question for token in ("直接写得数", "口算", "脱式计算", "计算下面")):
        return True
    return equation_count >= 5 and _looks_like_clean_equation_list(question)


def _looks_like_vertical_calculation_block(question: str) -> bool:
    if "竖式" in question:
        return True
    if "错因" in question and "改正" in question:
        return True
    if "解题过程" in question and len(_arithmetic_equation_parts(question)) >= 2:
        return not _looks_like_clean_equation_list(question)
    return False


def _vertical_process_equation_count(question: str) -> int:
    if "解题过程" not in question:
        return 0
    process = question.split("解题过程", 1)[1]
    return len(_arithmetic_equation_parts(process))


def _looks_like_clean_equation_list(question: str) -> bool:
    remainder = re.sub(
        r"\d+(?:\s*[+\-＋－×xX*÷/]\s*\d+)+\s*[=＝]\s*-?\d+(?:\.\d+)?(?:余\d+)?",
        " ",
        question,
    )
    remainder = re.sub(r"[，,。；;、\s()（）:：]+", "", remainder)
    if not remainder:
        return True
    return len(remainder) <= max(12, int(len(question) * 0.18))


def _looks_like_true_false_item(question: str, answer: str) -> bool:
    if answer in {"√", "✓", "V", "v", "对", "×", "x", "X", "错"}:
        return True
    if _embedded_true_false_mark(question):
        return True
    return question.startswith("判断") or "判断。" in question or "二、判断" in question


def _looks_like_choice_item(question: str, answer: str) -> bool:
    return _has_choice_options(question) and (not answer or re.fullmatch(r"[A-Da-d]", answer))


def _has_choice_options(question: str) -> bool:
    return bool(re.search(r"[A-D]\.", question))


def _looks_like_uncertain_grouped_fill_blank(question: str, answer: str) -> bool:
    blank_count = _blank_count(question)
    if blank_count < 2:
        return False
    if not answer:
        return True
    return len(_split_answer_parts(answer)) != blank_count


def _looks_like_direct_arithmetic(question: str, answer: str) -> bool:
    if not answer or not re.fullmatch(r"-?\d+(?:\.\d+)?(?:余\d+)?", answer):
        return False
    return bool(
        re.fullmatch(
            r"\d+(?:\s*[+\-×xX*÷/]\s*\d+)+\s*[=＝]\s*(?:\?|\(\s*\)|-?\d+(?:\.\d+)?(?:余\d+)?)?",
            question,
        )
    )


def _looks_like_embedded_arithmetic_answer(question: str) -> bool:
    return bool(re.search(r"\d+\s*[+\-×xX*÷/]\s*\d+\s*[=＝]\s*[（(]\s*-?\d+", question))


def _looks_like_simple_word_blank(question: str, answer: str) -> bool:
    if not answer or not re.fullmatch(r"\d+(?:\.\d+)?", answer):
        return False
    if not _has_blank(question):
        return False
    return _looks_like_word_problem(question)


def _looks_like_fill_blank_item(question: str, answer: str) -> bool:
    return bool(answer and _has_blank(question))


def _looks_like_word_problem(question: str) -> bool:
    if len(re.findall(r"\d+", question)) < 2:
        return False
    return any(token in question for token in ("一共", "平均", "每", "多少", "几", "剩", "买", "打", "分给", "可以"))


def _has_blank(question: str) -> bool:
    return _blank_count(question) > 0


def _blank_count(question: str) -> int:
    return len(re.findall(r"\(\s*\)|□|_{3,}", question))


def _split_answer_parts(answer: str) -> list[str]:
    return [part for part in re.split(r"[；;，,、/|]+", answer) if part]


def _arithmetic_equation_parts(question: str) -> list[tuple[str, str]]:
    return [
        (match.group("expr"), match.group("answer"))
        for match in re.finditer(
            r"(?P<expr>\d+(?:\s*[+\-＋－×xX*÷/]\s*\d+)+)\s*[=＝]\s*"
            r"(?P<answer>-?\d+(?:\.\d+)?(?:余\d+)?)",
            question,
        )
    ]


def _comparison_pair_re() -> re.Pattern[str]:
    expression = r"\d+(?:\s*[+\-×xX*÷/]\s*\d+)*"
    return re.compile(rf"{expression}\s*(?:\(\s*\)|[○Oo]|[(（])\s*{expression}")


def _comparison_pair_count(question: str) -> int:
    return len(_comparison_pair_re().findall(question))


def _comparison_answer_parts(child_answer: str) -> list[str]:
    return re.findall(r"[<>＝=≤≥]", child_answer)


def _normalize_question(value: str) -> str:
    text = str(value or "").strip()
    replacements = {
        "（": "(",
        "）": ")",
        "？": "?",
        "：": ":",
        "＝": "=",
        "✕": "×",
        "✖": "×",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_answer(value: str | None) -> str:
    text = str(value or "").strip()
    text = text.replace("✓", "√").replace("对", "√").replace("错", "×")
    return re.sub(r"\s+", "", text)


def _embedded_true_false_mark(question: str) -> str:
    match = re.search(r"[（(]\s*([√✓Vv×xX对错])\s*[）)]?\s*$", question)
    return match.group(1) if match else ""
