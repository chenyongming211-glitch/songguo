from __future__ import annotations

import ast
import re

from pydantic import BaseModel

from songguo.backend.services.learning.answer_matching import answers_match, normalize_answer
from songguo.backend.services.learning.question_type_routing import (
    MathQuestionKind,
    route_math_question_type,
)


class ObjectiveJudgeResult(BaseModel):
    correct: bool
    correct_answer: str
    question_type_id: str
    knowledge_point: str
    question_kind: str = ""
    evaluation_strategy: str = "deterministic"
    misconception_tag: str | None = None
    evidence: str = ""
    confidence: float = 0.9


def judge_objective_math_item(
    *,
    question_text: str,
    child_answer: str | None,
    ocr_action: str = "",
) -> ObjectiveJudgeResult | None:
    question = _normalize_question(question_text)
    answer = _normalize_child_answer(child_answer)
    if question and not answer:
        embedded_answer = _embedded_true_false_answer(question)
        if embedded_answer:
            answer = embedded_answer
            question = _strip_embedded_true_false_answer(question)
        else:
            answer = _embedded_direct_arithmetic_answer(question)
        if not answer:
            answer = _embedded_time_format_answer(question)
    if not question or not answer:
        return None

    route = route_math_question_type(
        question_text=question_text,
        child_answer=child_answer,
        ocr_action=ocr_action,
    )
    if route.evaluation_strategy != "deterministic":
        for judge in (
            _judge_cross_month_daily_fee_word_answer,
            _judge_same_day_time_interval_word_answer,
            _judge_twenty_four_hour_to_twelve_hour_conversion,
            _judge_can_enter_before_closing_time_answer,
            _judge_rectangular_queue_people_count,
            _judge_simple_multiplication_word_answer,
        ):
            result = judge(question, answer)
            if result is not None:
                result.question_kind = route.kind.value
                return result
        return None
    routed_judges = {
        MathQuestionKind.ORAL_CALCULATION: (_judge_direct_arithmetic_item,),
        MathQuestionKind.COMPARISON_SIGN: (_judge_comparison_sign_item,),
        MathQuestionKind.TRUE_FALSE: (_judge_true_false_fact,),
        MathQuestionKind.CHOICE: (_judge_choice_item,),
        MathQuestionKind.FILL_BLANK: (
            _judge_embedded_direct_arithmetic_fill,
            _judge_fill_blank_calculation,
            _judge_unit_conversion_fill_blank,
        ),
        MathQuestionKind.WORD_PROBLEM: (
            _judge_same_month_daily_count_word_fill,
            _judge_simple_multiplication_word_fill,
        ),
    }.get(route.kind)
    if routed_judges:
        for judge in routed_judges:
            result = judge(question, answer)
            if result is not None:
                result.question_kind = route.kind.value
                result.evaluation_strategy = route.evaluation_strategy
                return result

    for judge in (
        _judge_direct_arithmetic_item,
        _judge_comparison_sign_item,
        _judge_true_false_fact,
        _judge_choice_item,
        _judge_embedded_direct_arithmetic_fill,
        _judge_fill_blank_calculation,
        _judge_unit_conversion_fill_blank,
        _judge_same_month_daily_count_word_fill,
        _judge_cross_month_daily_fee_word_answer,
        _judge_same_day_time_interval_word_answer,
        _judge_twenty_four_hour_to_twelve_hour_conversion,
        _judge_can_enter_before_closing_time_answer,
        _judge_rectangular_queue_people_count,
        _judge_simple_multiplication_word_fill,
        _judge_simple_multiplication_word_answer,
    ):
        result = judge(question, answer)
        if result is not None:
            return result
    return None


def _judge_direct_arithmetic_item(question: str, child_answer: str) -> ObjectiveJudgeResult | None:
    expression = _direct_arithmetic_expression(question)
    if not expression:
        return None
    value = _safe_arithmetic_value(expression)
    if not isinstance(value, int | float):
        return None
    correct_answer = _format_number(value)
    return _build_result(
        child_answer=child_answer,
        correct_answer=correct_answer,
        question_type_id="math_oral_calculation",
        knowledge_point="口算",
        evidence=f"{expression}={correct_answer}。",
    )


def _judge_comparison_sign_item(question: str, child_answer: str) -> ObjectiveJudgeResult | None:
    pairs = _comparison_expression_pairs(question)
    if not pairs:
        return None
    expected: list[str] = []
    evidence_parts: list[str] = []
    for left_expr, right_expr in pairs:
        left_value = _safe_arithmetic_value(left_expr)
        right_value = _safe_arithmetic_value(right_expr)
        if not isinstance(left_value, int | float) or not isinstance(right_value, int | float):
            return None
        sign = ">" if left_value > right_value else "<" if left_value < right_value else "="
        expected.append(sign)
        evidence_parts.append(f"{left_expr}={_format_number(left_value)}，{right_expr}={_format_number(right_value)}")
    correct_answer = "；".join(expected)
    child_signs = _comparison_answer_parts(child_answer)
    if len(child_signs) != len(expected):
        return None
    comparable_child_answer = "；".join(child_signs) if len(child_signs) > 1 else (child_signs[0] if child_signs else "")
    if not comparable_child_answer:
        return None
    return _build_result(
        child_answer=comparable_child_answer,
        correct_answer=correct_answer,
        question_type_id="math_comparison_sign",
        knowledge_point="比较大小",
        evidence="；".join(evidence_parts),
    )


def _judge_choice_item(question: str, child_answer: str) -> ObjectiveJudgeResult | None:
    child_choice = _normalize_choice_answer(child_answer)
    if not child_choice:
        return None
    options = _parse_options(question)
    if not options:
        return None

    result = (
        _choice_by_blank_digit_product(question, options)
        or _choice_by_direct_arithmetic(question, options)
        or _choice_by_estimated_product(question, options)
        or _choice_by_product_place_value(question, options)
    )
    if result is None:
        return None

    correct_answer, question_type_id, knowledge_point, evidence = result
    return _build_result(
        child_answer=child_choice,
        correct_answer=correct_answer,
        question_type_id=question_type_id,
        knowledge_point=knowledge_point,
        evidence=evidence,
    )


def _choice_by_estimated_product(
    question: str,
    options: dict[str, str],
) -> tuple[str, str, str, str] | None:
    if "大约" not in question or "积" not in question:
        return None
    target_match = re.search(r"大约(?:是|为)?\s*(\d+)", question)
    if not target_match:
        return None
    target = int(target_match.group(1))
    candidates: list[tuple[int, str, int, str]] = []
    for key, option in options.items():
        expression = _extract_option_expression(option)
        if not expression:
            continue
        value = _safe_arithmetic_value(expression)
        if isinstance(value, int | float):
            candidates.append((abs(int(value) - target), key, int(value), expression))
    if not candidates:
        return None
    _, correct_answer, value, expression = min(candidates, key=lambda item: item[0])
    return (
        correct_answer,
        "math_choice_estimation",
        "乘法估算",
        f"{expression}={value}，最接近{target}。",
    )


def _choice_by_product_place_value(
    question: str,
    options: dict[str, str],
) -> tuple[str, str, str, str] | None:
    if "最高位是百位" not in question or "积" not in question:
        return None
    matches: list[tuple[str, int, str]] = []
    for key, option in options.items():
        expression = _extract_option_expression(option)
        if not expression:
            continue
        value = _safe_arithmetic_value(expression)
        if isinstance(value, int | float) and 100 <= abs(int(value)) <= 999:
            matches.append((key, int(value), expression))
    if len(matches) != 1:
        return None
    correct_answer, value, expression = matches[0]
    return (
        correct_answer,
        "math_choice_place_value",
        "乘积位数判断",
        f"{expression}={value}，最高位是百位。",
    )


def _choice_by_blank_digit_product(
    question: str,
    options: dict[str, str],
) -> tuple[str, str, str, str] | None:
    expression_match = re.search(r"(\d*□\d*\s*[×xX*]\s*\d*□?\d*|\d*□?\d*\s*[×xX*]\s*\d*□\d*)", question)
    if not expression_match:
        return None
    expression = expression_match.group(1)
    option_values = {key: _option_int_value(option) for key, option in options.items()}
    matches: list[tuple[str, int, int]] = []
    for digit in range(10):
        value = _safe_arithmetic_value(expression.replace("□", str(digit)))
        if not isinstance(value, int | float):
            continue
        for key, option_value in option_values.items():
            if option_value is not None and int(value) == option_value:
                matches.append((key, digit, int(value)))
    if len(matches) != 1:
        return None
    correct_answer, digit, value = matches[0]
    return (
        correct_answer,
        "math_choice_blank_digit_product",
        "乘法数位推断",
        f"方框填{digit}时乘积为{value}。",
    )


def _choice_by_direct_arithmetic(
    question: str,
    options: dict[str, str],
) -> tuple[str, str, str, str] | None:
    expression = _last_expression_before_blank(question)
    if not expression:
        return None
    value = _safe_arithmetic_value(expression)
    if not isinstance(value, int | float):
        return None
    formatted = _format_number(value)
    matches = []
    for key, option in options.items():
        option_value = _option_int_value(option)
        numeric_match = isinstance(value, int) and option_value == value
        if numeric_match or answers_match(option, formatted):
            matches.append(key)
    if len(matches) != 1:
        return None
    return (
        matches[0],
        "math_choice_direct_calculation",
        "四则计算",
        f"{expression}={formatted}。",
    )


def _judge_fill_blank_calculation(question: str, child_answer: str) -> ObjectiveJudgeResult | None:
    expected: list[str] = []
    for match in re.finditer(r"(\d+)个(\d+)的和是\s*[（(]", question):
        expected.append(str(int(match.group(1)) * int(match.group(2))))
    for match in re.finditer(r"(\d+)的(\d+)倍是\s*[（(]", question):
        expected.append(str(int(match.group(1)) * int(match.group(2))))
    for match in re.finditer(r"口算\d+\s*[×xX*]\s*(\d+0+)\s*时.*添上\s*[（(]\s*[）)]\s*个0", question):
        expected.append(str(_trailing_zero_count(int(match.group(1)))))
    for match in re.finditer(r"(\d+)\s*[×xX*]\s*(\d+)的积是\s*[（(]\s*[）)]\s*位数", question):
        value = int(match.group(1)) * int(match.group(2))
        expected.append(_digit_to_chinese(len(str(abs(value)))))
    for match in re.finditer(r"(\d+)\s*[×xX*]\s*(\d+)的积的末尾有\s*[（(]\s*[）)]\s*个0", question):
        value = int(match.group(1)) * int(match.group(2))
        expected.append(str(_trailing_zero_count(value)))
    if not expected:
        return None
    correct_answer = "；".join(expected)
    child_answer = _merge_embedded_fill_blank_answers(child_answer, question, expected_count=len(expected))
    return _build_result(
        child_answer=child_answer,
        correct_answer=correct_answer,
        question_type_id="math_fill_blank_calculation",
        knowledge_point="乘法计算与数位",
        evidence=f"按题干算式得到：{correct_answer}。",
    )


def _judge_embedded_direct_arithmetic_fill(question: str, child_answer: str) -> ObjectiveJudgeResult | None:
    match = re.search(
        r"(?P<expression>\d+\s*[+\-＋－×xX*÷/]\s*\d+)\s*[=＝]\s*[（(]\s*(?P<answer>-?\d+(?:\.\d+)?)",
        question,
    )
    if not match:
        return None
    expression = match.group("expression")
    embedded_answer = child_answer or match.group("answer")
    value = _safe_arithmetic_value(expression)
    if not isinstance(value, int | float):
        return None
    correct_answer = _format_number(value)
    return _build_result(
        child_answer=embedded_answer,
        correct_answer=correct_answer,
        question_type_id="math_fill_blank_calculation",
        knowledge_point="乘法计算与数位",
        evidence=f"{expression}={correct_answer}。",
    )


def _merge_embedded_fill_blank_answers(child_answer: str, question: str, *, expected_count: int) -> str:
    parts = _split_answer_parts(child_answer)
    if len(parts) >= expected_count:
        return child_answer
    embedded = [
        value
        for value in re.findall(r"[（(]\s*[（(]?\s*([一二三四五六七八九十百千万\d]+)\s*[）)]?", question)
        if value and value not in {")", "）"}
    ]
    for value in embedded:
        if value not in parts:
            parts.append(value)
        if len(parts) >= expected_count:
            break
    return "；".join(parts) if parts else child_answer


def _judge_unit_conversion_fill_blank(question: str, child_answer: str) -> ObjectiveJudgeResult | None:
    expected_with_positions: list[tuple[int, str, str]] = []
    conversion_patterns: list[tuple[str, int, str]] = [
        (r"(\d+)\s*日\s*=\s*[（(]\s*[）)]\s*时", 24, "乘24"),
        (r"(\d+)\s*周\s*=\s*[（(]\s*[）)]\s*天", 7, "乘7"),
        (r"(\d+)\s*年\s*=\s*[（(]\s*[）)]\s*个?月", 12, "乘12"),
        (r"(\d+)\s*时\s*=\s*[（(]\s*[）)]\s*分", 60, "乘60"),
        (r"(\d+)\s*世纪\s*=\s*[（(]\s*[）)]\s*年", 100, "乘100"),
    ]
    for pattern, multiplier, evidence in conversion_patterns:
        for match in re.finditer(pattern, question):
            value = int(match.group(1)) * multiplier
            expected_with_positions.append((match.start(), str(value), f"{match.group(1)}{evidence}={value}"))
    for match in re.finditer(r"(\d+)\s*时\s*=\s*[（(]\s*[）)]\s*日", question):
        hours = int(match.group(1))
        if hours % 24 != 0:
            return None
        value = hours // 24
        expected_with_positions.append((match.start(), str(value), f"{hours}÷24={value}"))
    if not expected_with_positions:
        return None
    expected_with_positions.sort(key=lambda item: item[0])
    expected = [item[1] for item in expected_with_positions]
    evidence_parts = [item[2] for item in expected_with_positions]
    return _build_result(
        child_answer=child_answer,
        correct_answer="；".join(expected),
        question_type_id="math_unit_conversion",
        knowledge_point="单位换算",
        evidence="；".join(evidence_parts),
    )


def _judge_simple_multiplication_word_fill(question: str, child_answer: str) -> ObjectiveJudgeResult | None:
    if not _has_blank(question):
        return None
    if not any(token in question for token in ("一共", "共", "可以", "总共", "合计")):
        return None
    if not any(token in question for token in ("每", "平均每", "盒", "分钟", "套", "箱", "袋")):
        return None
    numbers = [int(value) for value in re.findall(r"\d+", question)]
    if len(numbers) < 2:
        return None
    factors = numbers[-2:]
    correct_answer = str(factors[0] * factors[1])
    return _build_result(
        child_answer=child_answer,
        correct_answer=correct_answer,
        question_type_id="math_fill_blank_word_problem",
        knowledge_point="乘法应用题",
        evidence=f"{factors[0]}×{factors[1]}={correct_answer}。",
    )


def _judge_same_month_daily_count_word_fill(question: str, child_answer: str) -> ObjectiveJudgeResult | None:
    if not _has_blank(question):
        return None
    if "每天" not in question or not re.search(r"(?:1|一)\s*集", question) or "一共播放" not in question:
        return None
    match = re.search(
        r"(?P<month>\d+)\s*月\s*(?P<start>\d+)\s*日开始.*?"
        r"(?P=month)\s*月\s*(?P<end>\d+)\s*日",
        question,
    )
    if not match:
        return None
    start_day = int(match.group("start"))
    end_day = int(match.group("end"))
    if end_day < start_day:
        return None
    correct_answer = str(end_day - start_day + 1)
    return _build_result(
        child_answer=child_answer,
        correct_answer=correct_answer,
        question_type_id="math_date_count_word_problem",
        knowledge_point="日期区间",
        evidence=f"{end_day}-{start_day}+1={correct_answer}。",
    )


def _judge_simple_multiplication_word_answer(question: str, child_answer: str) -> ObjectiveJudgeResult | None:
    if not child_answer:
        return None
    if not any(token in question for token in ("多少", "一共", "最多", "需要", "可生产")):
        return None
    factors = _multiplication_word_problem_factors(question)
    if factors is None:
        return None
    left, right = factors
    correct_answer = str(left * right)
    return _build_result(
        child_answer=child_answer,
        correct_answer=correct_answer,
        question_type_id="math_multiplication_word_problem",
        knowledge_point="乘法应用题",
        evidence=f"{left}×{right}={correct_answer}。",
    )


def _judge_cross_month_daily_fee_word_answer(question: str, child_answer: str) -> ObjectiveJudgeResult | None:
    if not child_answer:
        return None
    match = re.search(
        r"从\s*(?P<start_month>\d+)\s*月\s*(?P<start_day>\d+)\s*日开始.*?"
        r"到\s*(?P<end_month>\d+)\s*月\s*(?P<end_day>\d+)\s*日结束.*?"
        r"每天[^，。；;?？]*?经费为\s*(?P<fee>\d+)\s*元",
        question,
    )
    if not match:
        return None
    day_count = _inclusive_month_day_count(
        int(match.group("start_month")),
        int(match.group("start_day")),
        int(match.group("end_month")),
        int(match.group("end_day")),
    )
    if day_count is None:
        return None
    fee = int(match.group("fee"))
    correct_answer = str(day_count * fee)
    return _build_result(
        child_answer=child_answer,
        correct_answer=correct_answer,
        question_type_id="math_date_fee_word_problem",
        knowledge_point="日期区间费用",
        evidence=f"{day_count}天×{fee}元={correct_answer}元。",
    )


def _judge_same_day_time_interval_word_answer(question: str, child_answer: str) -> ObjectiveJudgeResult | None:
    if not child_answer:
        return None
    match = re.search(r"(\d{1,2})\s*[:：]\s*(\d{2})\s*[-—－]\s*(\d{1,2})\s*[:：]\s*(\d{2})", question)
    if not match:
        return None
    first_minutes = int(match.group(1)) * 60 + int(match.group(2))
    second_minutes = int(match.group(3)) * 60 + int(match.group(4))
    diff = abs(first_minutes - second_minutes)
    if diff <= 0:
        return None
    if diff % 60 == 0:
        correct_answer = f"{diff // 60}小时"
    else:
        correct_answer = f"{diff // 60}小时{diff % 60}分"
    return _build_result(
        child_answer=child_answer,
        correct_answer=correct_answer,
        question_type_id="math_time_interval_word_problem",
        knowledge_point="经过时间",
        evidence=f"两个时刻相差{correct_answer}。",
    )


def _judge_twenty_four_hour_to_twelve_hour_conversion(question: str, child_answer: str) -> ObjectiveJudgeResult | None:
    match = re.search(r"(?P<hour>\d{1,2})\s*[:：]\s*(?P<minute>\d{2})\s*用12时记时法表示是", question)
    if not match:
        return None
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    period = "上午" if hour < 12 else "下午"
    display_hour = hour % 12
    if display_hour == 0:
        display_hour = 12
    correct_answer = f"{period}{display_hour}:{minute:02d}"
    return _build_result(
        child_answer=child_answer,
        correct_answer=correct_answer,
        question_type_id="math_time_format_conversion",
        knowledge_point="24时记时法",
        evidence=f"{hour:02d}:{minute:02d}是{correct_answer}。",
    )


def _judge_can_enter_before_closing_time_answer(question: str, child_answer: str) -> ObjectiveJudgeResult | None:
    if not child_answer or not any(token in question for token in ("还能进去", "还能去", "能进去")):
        return None
    arrival = re.search(r"下午\s*(\d{1,2})\s*时", question)
    closing = re.search(r"(\d{1,2})\s*[:：]\s*(\d{2})\s*=\s*(\d{1,2})\s*[:：]\s*(\d{2})", question)
    if not arrival or not closing:
        return None
    arrival_hour = int(arrival.group(1))
    if arrival_hour < 12:
        arrival_hour += 12
    arrival_minutes = arrival_hour * 60
    closing_hour = int(closing.group(3))
    closing_minute = int(closing.group(4))
    if closing_hour < 12:
        closing_hour += 12
    closing_minutes = closing_hour * 60 + closing_minute
    correct_answer = "能" if arrival_minutes < closing_minutes else "不能"
    return _build_result(
        child_answer=child_answer,
        correct_answer=correct_answer,
        question_type_id="math_time_comparison_word_problem",
        knowledge_point="时间比较",
        evidence=f"到达时间早于闭馆时间。" if correct_answer == "能" else "到达时间不早于闭馆时间。",
    )


def _judge_rectangular_queue_people_count(question: str, child_answer: str) -> ObjectiveJudgeResult | None:
    if not child_answer or not any(token in question for token in ("方队", "排队")):
        return None
    match = re.search(
        r"左起第\s*(?P<left>\d+)\s*列.*?右起第\s*(?P<right>\d+)\s*列.*?"
        r"前面有\s*(?P<front>\d+)\s*人.*?后面有\s*(?P<behind>\d+)\s*人",
        question,
    )
    if not match:
        return None
    columns = int(match.group("left")) + int(match.group("right")) - 1
    rows = int(match.group("front")) + int(match.group("behind")) + 1
    correct_answer = str(columns * rows)
    return _build_result(
        child_answer=child_answer,
        correct_answer=correct_answer,
        question_type_id="math_rectangular_queue_word_problem",
        knowledge_point="方队问题",
        evidence=f"列数{columns}，行数{rows}，{columns}×{rows}={correct_answer}。",
    )


def _inclusive_month_day_count(
    start_month: int,
    start_day: int,
    end_month: int,
    end_day: int,
) -> int | None:
    month_days = {
        1: 31,
        2: 28,
        3: 31,
        4: 30,
        5: 31,
        6: 30,
        7: 31,
        8: 31,
        9: 30,
        10: 31,
        11: 30,
        12: 31,
    }
    if start_month not in month_days or end_month not in month_days:
        return None
    if not (1 <= start_day <= month_days[start_month]) or not (1 <= end_day <= month_days[end_month]):
        return None
    if start_month == end_month:
        return end_day - start_day + 1 if end_day >= start_day else None
    if end_month < start_month:
        return None
    days = month_days[start_month] - start_day + 1 + end_day
    for month in range(start_month + 1, end_month):
        days += month_days[month]
    return days


def _multiplication_word_problem_factors(question: str) -> tuple[int, int] | None:
    patterns = [
        r"每套\s*(\d+)\s*元.*?买了?\s*(\d+)\s*套",
        r"每天[^，。；;?？]*?(\d+)\s*千克.*?(?:投放了?|有|使用)\s*(\d+)\s*台",
        r"1\s*吨[^，。；;?？]*?(?:可|能)?生产\s*(\d+)\s*千克.*?(\d+)\s*吨",
        r"每(?:盒|箱|袋|套|支|本|张|条|辆|台)\s*(\d+)\s*(?:个|元|千克|克|米|厘米)?.*?(?:买了?|有|用了?|需要)\s*(\d+)",
        r"一(?:盒|箱|袋|套|支|本|张|条|辆|台)[^，。；;?？]*?(\d+)\s*(?:个|元|千克|克|米|厘米).*?(?:买了?|有|用了?|需要)\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def _judge_true_false_fact(question: str, child_answer: str) -> ObjectiveJudgeResult | None:
    child_mark = _normalize_true_false_answer(child_answer)
    if not child_mark:
        return None
    if _looks_like_grouped_true_false_block(question):
        return None
    match = re.search(r"时针转([一二两三四五六七八九十\d]+)圈是(\d+)小时", question)
    if match:
        turns = _parse_small_int(match.group(1))
        hours = int(match.group(2))
        if turns is None:
            return None
        expected = "√" if turns * 12 == hours else "×"
        return _build_result(
            child_answer=child_mark,
            correct_answer=expected,
            question_type_id="math_true_false_fact",
            knowledge_point="时间常识",
            evidence=f"时针转1圈是12小时，{turns}圈是{turns * 12}小时。",
        )
    fact = _common_true_false_fact(question)
    if fact is None:
        return None
    expected, knowledge_point, evidence = fact
    return _build_result(
        child_answer=child_mark,
        correct_answer=expected,
        question_type_id="math_true_false_fact",
        knowledge_point=knowledge_point,
        evidence=evidence,
    )


def _build_result(
    *,
    child_answer: str,
    correct_answer: str,
    question_type_id: str,
    knowledge_point: str,
    evidence: str,
) -> ObjectiveJudgeResult:
    correct = _answers_equivalent(child_answer, correct_answer)
    return ObjectiveJudgeResult(
        correct=correct,
        correct_answer=correct_answer,
        question_type_id=question_type_id,
        knowledge_point=knowledge_point,
        misconception_tag=None if correct else "objective_answer_mismatch",
        evidence=evidence if correct else f"正确答案是{correct_answer}；孩子答案是{child_answer}。",
        confidence=0.93 if correct else 0.88,
    )


def _parse_options(question: str) -> dict[str, str]:
    text = question.replace("．", ".").replace("、", ".")
    matches = list(re.finditer(r"([A-Da-d])\s*[.]\s*(.+?)(?=\s*[A-Da-d]\s*[.]|$)", text))
    options: dict[str, str] = {}
    for match in matches:
        key = match.group(1).upper()
        value = match.group(2).strip(" 。；;,，")
        if value:
            options[key] = value
    return options if len(options) >= 2 else {}


def _normalize_question(value: str) -> str:
    text = str(value or "").strip()
    replacements = {
        "（": "(",
        "）": ")",
        "？": "?",
        "：": ":",
        "；": "；",
        "＝": "=",
        "×": "×",
        "✕": "×",
        "✖": "×",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_child_answer(value: str | None) -> str:
    text = str(value or "").strip()
    text = text.replace("✓", "√").replace("对", "√").replace("错", "×")
    return re.sub(r"\s+", "", text)


def _normalize_choice_answer(value: str) -> str:
    normalized = normalize_answer(value).upper()
    return normalized if re.fullmatch(r"[A-D]", normalized) else ""


def _normalize_true_false_answer(value: str) -> str:
    normalized = _normalize_child_answer(value)
    if normalized in {"√", "✓", "对", "V", "v"}:
        return "√"
    if normalized in {"×", "x", "X", "错"}:
        return "×"
    return ""


def _embedded_true_false_answer(question: str) -> str:
    match = re.search(r"[（(]\s*([√✓Vv×xX对错])\s*[）)]?\s*$", question)
    return _normalize_true_false_answer(match.group(1)) if match else ""


def _strip_embedded_true_false_answer(question: str) -> str:
    return re.sub(r"[（(]\s*[√✓Vv×xX对错]\s*[）)]?\s*$", "", question).strip()


def _embedded_direct_arithmetic_answer(question: str) -> str:
    match = re.search(
        r"\d+\s*[+\-＋－×xX*÷/]\s*\d+\s*[=＝]\s*[（(]\s*(-?\d+(?:\.\d+)?)",
        question,
    )
    return match.group(1) if match else ""


def _embedded_time_format_answer(question: str) -> str:
    match = re.search(r"[（(]\s*((?:上午|下午)?\s*\d{1,2}\s*[:：]\s*\d{2})", question)
    return re.sub(r"\s+", "", match.group(1)).replace("：", ":") if match else ""


def _answers_equivalent(child_answer: str, correct_answer: str) -> bool:
    child_parts = _split_answer_parts(child_answer)
    correct_parts = _split_answer_parts(correct_answer)
    if len(child_parts) == len(correct_parts) and len(correct_parts) > 1:
        return all(answers_match(child, expected) for child, expected in zip(child_parts, correct_parts, strict=True))
    return answers_match(child_answer, correct_answer)


def _split_answer_parts(value: str) -> list[str]:
    return [part for part in re.split(r"[；;,，、/]", str(value or "")) if part]


def _extract_option_expression(option: str) -> str:
    match = re.search(r"\d+\s*[×xX*]\s*\d+", option)
    return match.group(0) if match else ""


def _option_int_value(option: str) -> int | None:
    match = re.search(r"\d+", option)
    return int(match.group(0)) if match else None


def _direct_arithmetic_expression(question: str) -> str:
    expression = r"\d+(?:\s*[+\-＋－×xX*÷/]\s*\d+)+"
    match = re.fullmatch(rf"\s*({expression})\s*[=＝]\s*(?:\?|\(\s*\)|-?\d+(?:\.\d+)?(?:余\d+)?)?\s*", question)
    return match.group(1) if match else ""


def _comparison_expression_pairs(question: str) -> list[tuple[str, str]]:
    atom = r"(?:\d+|[（(]\s*\d+(?:\s*[+\-＋－×xX*÷/]\s*\d+)+\s*[）)])"
    expression = rf"{atom}(?:\s*[+\-＋－×xX*÷/]\s*{atom})*"
    pairs: list[tuple[str, str]] = []
    for match in re.finditer(rf"({expression})\s*(?:\(\s*\)|[○Oo]|[(（])\s*({expression})", question):
        pairs.append((_normalize_math_expression(match.group(1)), _normalize_math_expression(match.group(2))))
    return pairs


def _normalize_math_expression(expression: str) -> str:
    return expression.replace("（", "(").replace("）", ")")


def _comparison_answer_parts(child_answer: str) -> list[str]:
    parts = []
    for sign in re.findall(r"[<>＝=≤≥]", child_answer):
        if sign in {"＝", "="}:
            parts.append("=")
        elif sign == "≤":
            parts.append("<=")
        elif sign == "≥":
            parts.append(">=")
        else:
            parts.append(sign)
    return parts


def _common_true_false_fact(question: str) -> tuple[str, str, str] | None:
    if "两位数乘两位数" in question and "三位数" in question and "四位数" in question:
        return "√", "两位数乘两位数", "10×10=100，99×99=9801，积可能是三位数也可能是四位数。"
    if "乘数末尾共有2个0" in question and "积的末尾也一定有2个0" in question:
        return "×", "乘法末尾0", "20×50=1000，积的末尾可能多于2个0。"
    if "积一定大于" in question and "和" in question:
        return "×", "乘法意义", "1×1=1，小于1+1=2，所以不是一定大于。"
    match = re.search(r"最大的两位数与最小的三位数的乘积是(\d+)", question)
    if match:
        product = 99 * 100
        expected = "√" if int(match.group(1)) == product else "×"
        return expected, "两位数乘三位数", f"最大的两位数是99，最小的三位数是100，99×100={product}。"
    return None


def _looks_like_grouped_true_false_block(question: str) -> bool:
    numbered_count = len(re.findall(r"(?:^|\s)\d+[.．、)]", question))
    mark_count = len(re.findall(r"[（(]\s*[√✓Vv×xX对错]\s*[）)]?", question))
    return numbered_count >= 2 or mark_count >= 2 or ("二、判断" in question and mark_count >= 1)


def _last_expression_before_blank(question: str) -> str:
    matches = list(
        re.finditer(
            r"(\d+(?:\s*[+\-＋－×xX*÷/]\s*\d+)+)\s*=\s*(?:[（(]\s*[）)]|\?)",
            question,
        )
    )
    return matches[-1].group(1) if matches else ""


def _safe_arithmetic_value(expression: str) -> int | float | None:
    normalized = (
        expression.replace("×", "*")
        .replace("x", "*")
        .replace("X", "*")
        .replace("÷", "/")
        .replace("＋", "+")
        .replace("－", "-")
    )
    if not re.fullmatch(r"[0-9+\-*/().\s]+", normalized):
        return None
    try:
        tree = ast.parse(normalized, mode="eval")
        value = _eval_ast(tree.body)
    except (SyntaxError, ValueError, ZeroDivisionError):
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _eval_ast(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_ast(node.operand)
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
    raise ValueError("unsafe arithmetic expression")


def _format_number(value: int | float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _digit_to_chinese(value: int) -> str:
    mapping = {
        0: "零",
        1: "一",
        2: "二",
        3: "三",
        4: "四",
        5: "五",
        6: "六",
        7: "七",
        8: "八",
        9: "九",
        10: "十",
    }
    return mapping.get(value, str(value))


def _trailing_zero_count(value: int) -> int:
    if value == 0:
        return 1
    count = 0
    while value % 10 == 0:
        count += 1
        value //= 10
    return count


def _has_blank(question: str) -> bool:
    return "( )" in question or "()" in question or "□" in question


def _parse_small_int(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    mapping = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    return mapping.get(value)
