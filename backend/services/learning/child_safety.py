from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class ChildSafetyAction(StrEnum):
    ALLOW = "allow"
    BLOCK = "block"


class ChildSafetyVerdict(BaseModel):
    action: ChildSafetyAction
    reason: str
    safe_text: str


_SAFE_FALLBACK_TEXT = "这个内容不适合在学习陪练里继续。我们回到题目本身，一步一步想。"

_PRIVACY_TERMS = (
    "手机号",
    "电话号码",
    "家庭住址",
    "住址",
    "地址发给我",
    "身份证",
    "银行卡",
    "密码",
    "验证码",
    "微信号",
    "qq号",
)

_SEXUAL_TERMS = (
    "裸照",
    "裸体",
    "露点",
    "色情",
    "黄色图片",
    "生殖器",
)

_SELF_HARM_OR_VIOLENCE_TERMS = (
    "自杀",
    "自残",
    "割腕",
    "杀人",
    "打死",
    "砍人",
)

_ILLEGAL_OR_ABUSE_TERMS = (
    "偷钱",
    "诈骗",
    "吸毒",
    "毒品",
    "骂你",
    "蠢货",
    "笨死",
)


def check_child_safety(text: str) -> ChildSafetyVerdict:
    content = str(text or "")
    normalized = content.lower()
    reason = _risk_reason(normalized)
    if reason:
        return ChildSafetyVerdict(
            action=ChildSafetyAction.BLOCK,
            reason=reason,
            safe_text=_SAFE_FALLBACK_TEXT,
        )
    return ChildSafetyVerdict(
        action=ChildSafetyAction.ALLOW,
        reason="safe",
        safe_text=content,
    )


def _risk_reason(text: str) -> str:
    if _contains_any(text, _SEXUAL_TERMS):
        return "sexual_content"
    if _contains_any(text, _SELF_HARM_OR_VIOLENCE_TERMS):
        return "self_harm_or_violence"
    if _contains_any(text, _PRIVACY_TERMS):
        return "child_privacy_risk"
    if _contains_any(text, _ILLEGAL_OR_ABUSE_TERMS):
        return "illegal_or_abuse_risk"
    return ""


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)
