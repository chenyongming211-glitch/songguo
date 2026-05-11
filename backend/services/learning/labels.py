from __future__ import annotations

from songguo.backend.services.learning.teaching_assets import DEFAULT_MATH_ASSET_LIBRARY


KNOWLEDGE_POINT_LABELS = {
    "two_digit_times_one_digit": "两位数乘一位数",
    "capacity_round_up": "限载进一问题",
    "division_with_remainder": "有余数除法",
    "remainder_division": "有余数除法",
    "grade_math_unknown": "当前数学知识点",
    "english_sentence_pattern": "英语句型表达",
    "chinese_reading_summary": "语文阅读概括",
}

MISCONCEPTION_LABELS = {
    "treated_x5_like_x10": "把乘以 5 当成乘以 10",
    "stopped_at_total_count": "只算出总数就停止",
    "ignored_remainder_round_up": "有余数但没有进一",
    "math_division_quotient_too_large": "商估大了，缺少回乘核对",
    "math_division_quotient_too_small": "商估小了，缺少回乘核对",
    "math_division_missing_remainder": "只写商，漏掉余数",
    "used_group_count_as_answer": "把题目中的组数当成答案",
    "copied_capacity": "只抄限载数量，没有理解问题",
    "unknown_misconception": "还没有找到稳定错因",
}


def knowledge_point_label(value: str | None) -> str:
    if not value:
        return "当前知识点"
    return KNOWLEDGE_POINT_LABELS.get(value, value)


def misconception_label(value: str | None) -> str:
    if not value:
        return "未知错因"
    try:
        return DEFAULT_MATH_ASSET_LIBRARY.require_misconception(value).name
    except KeyError:
        pass
    return MISCONCEPTION_LABELS.get(value, value)
