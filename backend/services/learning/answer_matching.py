from __future__ import annotations

import re


def answers_match(child_answer: str | int | float | None, expected_answer: str | int | float | None) -> bool:
    normalized_child = normalize_answer(child_answer)
    normalized_expected = normalize_answer(expected_answer)
    if not normalized_child or not normalized_expected:
        return False
    if normalized_child == normalized_expected:
        return True

    child_yes_no = _yes_no_polarity(normalized_child)
    expected_yes_no = _yes_no_polarity(normalized_expected)
    if child_yes_no and expected_yes_no:
        if child_yes_no != expected_yes_no:
            return False
        if not _number_tokens(normalized_child):
            return True

    child_direction = _comparison_direction(normalized_child)
    expected_direction = _comparison_direction(normalized_expected)
    if expected_direction:
        if child_direction != expected_direction:
            return False
    elif child_direction and expected_direction and child_direction != expected_direction:
        return False

    child_length = _length_centimeters(normalized_child)
    expected_length = _length_centimeters(normalized_expected)
    if child_length is not None and expected_length is not None:
        return abs(child_length - expected_length) < 1e-9

    child_numbers = _number_tokens(normalized_child)
    expected_numbers = _number_tokens(normalized_expected)
    if not child_numbers or not expected_numbers:
        return False
    if len(child_numbers) > len(expected_numbers):
        return child_numbers[-len(expected_numbers) :] == expected_numbers
    return child_numbers == expected_numbers[-len(child_numbers) :]


def normalize_answer(value: str | int | float | None) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[\s，。,.？?！!：:；;、]", "", text)
    text = text.replace("×", "x").replace("公分", "厘米")
    return text


def _number_tokens(value: str) -> list[str]:
    return re.findall(r"-?\d+(?:\.\d+)?", value)


def _yes_no_polarity(value: str) -> str | None:
    if any(token in value for token in ("不能", "不可以", "不够", "无法", "没法", "不是", "否")):
        return "no"
    if any(token in value for token in ("能", "可以", "够", "是")):
        return "yes"
    return None


def _comparison_direction(value: str) -> str | None:
    has_more = any(token in value for token in ("多", "增加", "高出", "超过"))
    has_less = any(token in value for token in ("少", "减少", "低于", "不足"))
    if has_more and not has_less:
        return "more"
    if has_less and not has_more:
        return "less"
    return None


def _length_centimeters(value: str) -> float | None:
    if not any(unit in value for unit in ("米", "厘米")):
        return None
    if any(unit in value for unit in ("平方米", "平方厘米", "立方米", "立方厘米")):
        return None

    total = 0.0
    matched = False
    for match in re.finditer(r"(-?\d+(?:\.\d+)?)米(?!米|方)", value):
        total += float(match.group(1)) * 100
        matched = True
    for match in re.finditer(r"(-?\d+(?:\.\d+)?)厘米", value):
        total += float(match.group(1))
        matched = True
    return total if matched else None
