from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel

from songguo.backend.services.learning.math_structuring import ProblemAnalysis


class SafetyResponseSource(StrEnum):
    MODEL = "model"
    REPAIRED_MODEL = "repaired_model"
    CONTEXTUAL_FALLBACK = "contextual_fallback"
    HARD_FALLBACK = "hard_fallback"


class SafetyContract(BaseModel):
    action: str
    reason: str
    final_response_source: SafetyResponseSource
    repair_attempted: bool = False


def build_contextual_fallback_message(
    *,
    question_text: str,
    analysis: ProblemAnalysis,
    hint_level: int,
) -> str:
    position_hint = _position_reasoning_hint(question_text)
    if position_hint:
        return position_hint
    if analysis.key_points:
        prompt = analysis.first_key_point.child_prompt.strip()
        if prompt and "答案" not in prompt:
            return prompt
    if hint_level <= 2:
        return "我们先只看题目条件。你能先说说题目告诉了哪些数量，最后要求什么吗？"
    return "把这道题拆成两小步，先告诉我你准备先算哪一步。"


def _position_reasoning_hint(question_text: str) -> str:
    total_match = re.search(r"(\d+)\s*个小朋友", question_text)
    right_match = re.search(r"从右往左数[，,。；;、\s]*(?P<name>[^，,。；;、\s]{1,6})排在第(?P<rank>\d+)个", question_text)
    if not total_match or not right_match:
        return ""
    total = total_match.group(1)
    name = right_match.group("name")
    rank = right_match.group("rank")
    return (
        f"这题先抓住位置方向。{name}从右往左第{rank}个，"
        f"我们先把它换成从左边数的位置：一共有{total}个小朋友，"
        f"你觉得{name}从左边数是第几个？"
    )
