from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

from songguo.backend.services.learning.models import ReminderJob, ReminderSubscription
from songguo.backend.services.learning.store import InMemoryLearningStore


class DueReminderList(BaseModel):
    child_id: str | None = None
    items: list[ReminderJob] = Field(default_factory=list)


class SendReminderResult(BaseModel):
    sent_count: int
    failed_count: int = 0
    items: list[ReminderJob] = Field(default_factory=list)


def build_due_reminders(
    store: InMemoryLearningStore,
    *,
    child_id: str | None = None,
) -> DueReminderList:
    subscriptions = store.list_reminder_subscriptions(child_id=child_id)
    return DueReminderList(
        child_id=child_id,
        items=[_job_from_subscription(item) for item in subscriptions],
    )


def send_due_reminders(
    store: InMemoryLearningStore,
    *,
    child_id: str | None = None,
    wechat_service: Any | None = None,
) -> SendReminderResult:
    due = build_due_reminders(store, child_id=child_id)
    should_send_wechat = (
        wechat_service is not None
        or os.getenv("WECHAT_SEND_SUBSCRIBE_MESSAGES", "").lower() in {"1", "true", "yes"}
    )
    if should_send_wechat and wechat_service is None:
        from songguo.backend.services.wechat import WechatService

        wechat_service = WechatService()

    sent_items: list[ReminderJob] = []
    failed_items: list[ReminderJob] = []
    for item in due.items:
        if should_send_wechat:
            try:
                wechat_service.send_subscribe_message(
                    openid=item.openid,
                    template_id=item.template_id,
                    page="pages/parent-report/index",
                    data=_wechat_template_data(item),
                )
            except Exception:
                failed_items.append(item.model_copy(update={"status": "failed"}))
                continue
        sent_items.append(item.model_copy(update={"status": "sent"}))
    return SendReminderResult(
        sent_count=len(sent_items),
        failed_count=len(failed_items),
        items=[*sent_items, *failed_items],
    )


def _job_from_subscription(subscription: ReminderSubscription) -> ReminderJob:
    return ReminderJob(
        openid=subscription.openid,
        child_id=subscription.child_id,
        template_id=subscription.template_id,
        scope=subscription.scope,
    )


def _wechat_template_data(job: ReminderJob) -> dict[str, dict[str, str]]:
    return {
        "thing1": {"value": "错题复习提醒"},
        "thing2": {"value": _scope_label(job.scope)},
    }


def _scope_label(scope: str) -> str:
    return {
        "weekly": "本周复习",
        "monthly": "本月复习",
        "quarterly": "季度复习",
        "term": "本学期复习",
        "yearly": "本年度复习",
    }.get(scope, scope)
