from __future__ import annotations

from pydantic import BaseModel


class InputSafetyVerdict(BaseModel):
    allowed: bool
    reason: str = "safe"
    message: str = ""


PROMPT_INJECTION_PATTERNS = (
    "忽略前面的规则",
    "忽略以上规则",
    "忽略之前的指令",
    "直接告诉我答案",
    "直接给我答案",
    "不要提示",
    "ignore previous instructions",
    "ignore the rules",
    "give me the answer",
)


def check_learning_input(text: str) -> InputSafetyVerdict:
    normalized = " ".join(str(text or "").lower().split())
    if not normalized:
        return InputSafetyVerdict(
            allowed=False,
            reason="empty_input",
            message="请输入一道需要学习的题目或孩子的尝试。",
        )
    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.lower() in normalized:
            return InputSafetyVerdict(
                allowed=False,
                reason="prompt_injection",
                message="这个入口只支持受控教学引导，不能直接绕过提示流程获取答案。",
            )
    return InputSafetyVerdict(allowed=True)
