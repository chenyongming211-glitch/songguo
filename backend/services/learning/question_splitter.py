from __future__ import annotations

from dataclasses import dataclass
import re

from songguo.backend.services.learning.answer_binding import bind_grouped_comparison_answers


@dataclass(frozen=True)
class SplitQuestionItem:
    question_text: str
    child_answer: str


def split_grouped_math_item(
    *,
    question_text: str,
    child_answer: str | None,
    question_type_id: str,
) -> list[SplitQuestionItem]:
    question = _normalize_text(question_text)
    answer = _normalize_answer(child_answer)
    if question_type_id == "math_grouped_comparison_sign":
        return _split_grouped_comparison(question, answer)
    if question_type_id == "math_grouped_oral_calculation":
        return _split_grouped_oral_calculation(question)
    if question_type_id == "math_vertical_calculation_block":
        return _split_vertical_process_calculation(question)
    if question_type_id == "math_grouped_true_false":
        return _split_grouped_true_false(question, answer)
    if question_type_id == "math_grouped_fill_blank":
        return _split_grouped_fill_blank(question, answer)
    return []


def _split_grouped_comparison(question: str, answer: str) -> list[SplitQuestionItem]:
    bindings = bind_grouped_comparison_answers(question_text=question, child_answer=answer)
    return [
        SplitQuestionItem(question_text=binding.question_text, child_answer=binding.child_answer)
        for binding in bindings
    ]


def _split_grouped_oral_calculation(question: str) -> list[SplitQuestionItem]:
    items: list[SplitQuestionItem] = []
    for match in re.finditer(
        r"(?P<expr>\d+(?:\s*[+\-＋－×xX*÷/]\s*\d+)+)\s*[=＝]\s*"
        r"(?P<answer>-?\d+(?:\.\d+)?(?:余\d+)?)",
        question,
    ):
        expression = _normalize_expression(match.group("expr"))
        answer = match.group("answer").strip()
        items.append(SplitQuestionItem(question_text=f"{expression}=", child_answer=answer))
    return items if len(items) >= 2 else []


def _split_vertical_process_calculation(question: str) -> list[SplitQuestionItem]:
    if "解题过程" not in question:
        return []
    process = question.split("解题过程", 1)[1]
    return _split_grouped_oral_calculation(process)


def _split_grouped_true_false(question: str, answer: str) -> list[SplitQuestionItem]:
    segments = _numbered_segments(question)
    if len(segments) < 2:
        return []
    answer_marks = _true_false_answer_parts(answer)
    items: list[SplitQuestionItem] = []
    fallback_answer_index = 0
    for index, segment in enumerate(segments):
        segment_answer = _last_true_false_mark(segment)
        question_text = _strip_true_false_mark(segment)
        if not segment_answer and fallback_answer_index < len(answer_marks):
            segment_answer = answer_marks[fallback_answer_index]
            fallback_answer_index += 1
        if question_text:
            items.append(SplitQuestionItem(question_text=question_text, child_answer=segment_answer))
    return items if len(items) >= 2 else []


def _split_grouped_fill_blank(question: str, answer: str) -> list[SplitQuestionItem]:
    if _blank_count(question) < 2:
        return []
    expression_match = re.search(
        r"(?P<expr>\d+\s*[+\-＋－×xX*÷/]\s*\d+)\s*[=＝]\s*\(\s*\)",
        question,
    )
    if not expression_match:
        return _split_fill_blank_by_separators(question, answer)

    expression = _normalize_expression(expression_match.group("expr"))
    first_answer = _simple_arithmetic_answer(expression)
    if not first_answer:
        return _split_fill_blank_by_separators(question, answer)
    compact_answer = re.sub(r"\s+", "", answer)
    if compact_answer.startswith(first_answer):
        second_answer = compact_answer[len(first_answer):]
    else:
        second_answer = ""
    context = _leading_calculation_context(question)
    second_question = f"{context}，再在积的后面添上( )个0。" if context else "再在积的后面添上( )个0。"
    return [
        SplitQuestionItem(question_text=f"{expression}=( )", child_answer=first_answer),
        SplitQuestionItem(question_text=second_question, child_answer=second_answer),
    ]


def _split_fill_blank_by_separators(question: str, answer: str) -> list[SplitQuestionItem]:
    parts = _split_answer_parts(answer)
    blanks = list(re.finditer(r"\(\s*\)|□|_{3,}", question))
    if len(blanks) < 2:
        return []
    items: list[SplitQuestionItem] = []
    for index, blank in enumerate(blanks):
        start = _previous_clause_start(question, blank.start())
        end = _next_clause_end(question, blank.end())
        question_part = question[start:end].strip(" ，,；;。")
        if question_part:
            items.append(
                SplitQuestionItem(
                    question_text=question_part,
                    child_answer=parts[index] if index < len(parts) else "",
                )
            )
    return items if len(items) >= 2 else []


def _numbered_segments(question: str) -> list[str]:
    matches = list(re.finditer(r"(?<!\d)\d+[.．、)]\s*", question))
    segments: list[str] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(question)
        segment = question[start:end].strip(" ，,；;")
        if segment:
            segments.append(segment)
    return segments


def _last_true_false_mark(segment: str) -> str:
    marks = re.findall(r"[（(]\s*([√✓Vv×xX对错])\s*[）)]?", segment)
    return marks[-1] if marks else ""


def _strip_true_false_mark(segment: str) -> str:
    text = re.sub(r"[（(]\s*[√✓Vv×xX对错]\s*[）)]?", "", segment)
    text = re.sub(r"[（(]\s*[）)]", "", text)
    return text.strip(" ，,；;。") + "。"


def _true_false_answer_parts(answer: str) -> list[str]:
    return re.findall(r"[√✓Vv×xX对错]", answer)


def _split_answer_parts(answer: str) -> list[str]:
    return [part for part in re.split(r"[；;，,、/|]+", answer) if part]


def _blank_count(question: str) -> int:
    return len(re.findall(r"\(\s*\)|□|_{3,}", question))


def _simple_arithmetic_answer(expression: str) -> str:
    match = re.fullmatch(r"(\d+)\s*([+\-＋－×xX*÷/])\s*(\d+)", expression)
    if not match:
        return ""
    left = int(match.group(1))
    op = match.group(2)
    right = int(match.group(3))
    if op in {"×", "x", "X", "*"}:
        return str(left * right)
    if op in {"+", "＋"}:
        return str(left + right)
    if op in {"-", "－"}:
        return str(left - right)
    if op in {"÷", "/"} and right:
        if left % right == 0:
            return str(left // right)
    return ""


def _leading_calculation_context(question: str) -> str:
    match = re.search(r"口算[^，,。；;]*?时", question)
    return match.group(0).strip() if match else ""


def _previous_clause_start(question: str, position: int) -> int:
    candidates = [question.rfind(token, 0, position) for token in ("，", ",", "；", ";", "。")]
    previous = max(candidates)
    return 0 if previous < 0 else previous + 1


def _next_clause_end(question: str, position: int) -> int:
    candidates = [question.find(token, position) for token in ("，", ",", "；", ";", "。")]
    positive = [candidate for candidate in candidates if candidate >= 0]
    return min(positive) if positive else len(question)


def _normalize_expression(value: str) -> str:
    return re.sub(r"\s+", "", value).replace("（", "(").replace("）", ")").replace("＝", "=")


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).replace("（", "(").replace("）", ")")


def _normalize_answer(value: str | None) -> str:
    return re.sub(r"\s+", "", str(value or "").strip())
