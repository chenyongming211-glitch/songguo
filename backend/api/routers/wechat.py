from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from songguo.backend.services.wechat import ContentSafetyResult, WechatLoginResult, WechatService

router = APIRouter()


class WechatLoginRequest(BaseModel):
    code: str


class TextSafetyRequest(BaseModel):
    openid: str | None = None
    text: str


def get_wechat_service() -> WechatService:
    return WechatService()


@router.post("/login", response_model=WechatLoginResult)
def login(request: WechatLoginRequest) -> WechatLoginResult:
    try:
        return get_wechat_service().login(request.code)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


@router.post("/content-safety/text", response_model=ContentSafetyResult)
def check_text_safety(request: TextSafetyRequest) -> ContentSafetyResult:
    return get_wechat_service().check_text(request.text, openid=request.openid)
