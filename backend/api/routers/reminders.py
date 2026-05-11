from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from songguo.backend.services.learning.models import ReminderSubscription
from songguo.backend.services.learning.reminders import (
    DueReminderList,
    SendReminderResult,
    build_due_reminders,
    send_due_reminders,
)
from songguo.backend.services.session_auth import authorize_child_access, strict_auth_enabled
from songguo.backend.services.learning.service import get_global_learning_store
from songguo.backend.services.learning.store import InMemoryLearningStore

router = APIRouter()


class SubscriptionRequest(BaseModel):
    openid: str
    child_id: str
    template_id: str
    enabled: bool = True
    scope: str = "weekly"


class SendDueRequest(BaseModel):
    child_id: str | None = None


def get_learning_store() -> InMemoryLearningStore:
    return get_global_learning_store()


def _authorize_child(child_id: str, session_token: str | None) -> str | None:
    try:
        return authorize_child_access(
            get_learning_store(),
            child_id=child_id,
            session_token=session_token,
        )
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.post("/subscriptions", response_model=ReminderSubscription)
def save_subscription(
    request: SubscriptionRequest,
    x_session_token: str | None = Header(None, alias="X-Session-Token"),
) -> ReminderSubscription:
    openid = _authorize_child(request.child_id, x_session_token)
    return get_learning_store().save_reminder_subscription(
        openid=openid or request.openid,
        child_id=request.child_id,
        template_id=request.template_id,
        enabled=request.enabled,
        scope=request.scope,
    )


@router.get("/due", response_model=DueReminderList)
def list_due_reminders(
    child_id: str | None = None,
    x_session_token: str | None = Header(None, alias="X-Session-Token"),
) -> DueReminderList:
    if strict_auth_enabled() and not child_id:
        raise HTTPException(status_code=400, detail="child_id is required in strict auth mode")
    if child_id:
        _authorize_child(child_id, x_session_token)
    return build_due_reminders(get_learning_store(), child_id=child_id)


@router.post("/send-due", response_model=SendReminderResult)
def send_due(
    request: SendDueRequest,
    x_session_token: str | None = Header(None, alias="X-Session-Token"),
) -> SendReminderResult:
    if strict_auth_enabled() and not request.child_id:
        raise HTTPException(status_code=400, detail="child_id is required in strict auth mode")
    if request.child_id:
        _authorize_child(request.child_id, x_session_token)
    return send_due_reminders(get_learning_store(), child_id=request.child_id)
