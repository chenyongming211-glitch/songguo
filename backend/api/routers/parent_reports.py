from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from songguo.backend.services.learning.models import ChildProfile, LearningSummary, SafetyEvent
from songguo.backend.services.learning.memory_profile import LearningMemory, build_learning_memory
from songguo.backend.services.learning.reporting import (
    LearningDeposit,
    SessionFeedback,
    SummaryDraft,
    WeeklyReport,
    WrongQuestionList,
    build_learning_deposit,
    build_session_feedback,
    build_summary_draft,
    build_weekly_report,
    build_wrong_question_list,
)
from songguo.backend.services.learning.review_plan import ReviewPlan, build_review_plan
from songguo.backend.services.learning.service import get_global_learning_store
from songguo.backend.services.learning.store import InMemoryLearningStore
from songguo.backend.services.learning.summary_rollup import (
    build_daily_summary,
    build_monthly_summary,
    build_weekly_summary,
)
from songguo.backend.services.session_auth import (
    authorize_child_access,
    openid_from_session_token,
    strict_auth_enabled,
)

router = APIRouter()


class UpsertChildRequest(BaseModel):
    child_id: str
    name: str
    grade: int
    term_label: str | None = None


class ChildListResponse(BaseModel):
    children: list[ChildProfile]


class SafetyEventListResponse(BaseModel):
    child_id: str
    items: list[SafetyEvent]


def get_learning_store() -> InMemoryLearningStore:
    return get_global_learning_store()


def _authorize_child(child_id: str, session_token: str | None) -> None:
    try:
        authorize_child_access(
            get_learning_store(),
            child_id=child_id,
            session_token=session_token,
        )
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def _openid_from_header(session_token: str | None) -> str | None:
    if not session_token:
        if strict_auth_enabled():
            raise HTTPException(status_code=401, detail="X-Session-Token is required")
        return None
    try:
        return openid_from_session_token(session_token)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@router.post("/children", response_model=ChildProfile)
def upsert_child(
    request: UpsertChildRequest,
    x_session_token: str | None = Header(None, alias="X-Session-Token"),
) -> ChildProfile:
    openid = _openid_from_header(x_session_token)
    store = get_learning_store()
    if strict_auth_enabled() and request.child_id != f"child_{openid}":
        try:
            authorize_child_access(
                store,
                child_id=request.child_id,
                session_token=x_session_token,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
    child = store.upsert_child(
        child_id=request.child_id,
        name=request.name,
        grade=request.grade,
        term_label=request.term_label,
    )
    if openid:
        store.bind_child_to_openid(openid=openid, child_id=child.child_id)
    return child


@router.get("/children", response_model=ChildListResponse)
def list_children(
    x_session_token: str | None = Header(None, alias="X-Session-Token"),
) -> ChildListResponse:
    store = get_learning_store()
    openid = _openid_from_header(x_session_token)
    if openid:
        return ChildListResponse(children=store.list_children_for_openid(openid))
    return ChildListResponse(children=store.list_children())


@router.get("/children/{child_id}/weekly-report", response_model=WeeklyReport)
def get_weekly_report(
    child_id: str,
    x_session_token: str | None = Header(None, alias="X-Session-Token"),
) -> WeeklyReport:
    _authorize_child(child_id, x_session_token)
    return build_weekly_report(get_learning_store(), child_id=child_id)


@router.get("/children/{child_id}/session-feedback", response_model=SessionFeedback)
def get_session_feedback(
    child_id: str,
    session_id: str,
    x_session_token: str | None = Header(None, alias="X-Session-Token"),
) -> SessionFeedback:
    _authorize_child(child_id, x_session_token)
    try:
        return build_session_feedback(
            get_learning_store(),
            child_id=child_id,
            session_id=session_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Learning session not found") from exc


@router.get("/children/{child_id}/learning-deposit", response_model=LearningDeposit)
def get_learning_deposit(
    child_id: str,
    session_id: str,
    scope: str = "weekly",
    x_session_token: str | None = Header(None, alias="X-Session-Token"),
) -> LearningDeposit:
    _authorize_child(child_id, x_session_token)
    if scope not in {"weekly", "monthly", "quarterly", "term", "yearly"}:
        raise HTTPException(status_code=400, detail="Unsupported learning deposit scope")
    try:
        return build_learning_deposit(
            get_learning_store(),
            child_id=child_id,
            session_id=session_id,
            scope=scope,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Learning session not found") from exc


@router.get("/children/{child_id}/summary-draft", response_model=SummaryDraft)
def get_summary_draft(
    child_id: str,
    scope: str = "weekly",
    x_session_token: str | None = Header(None, alias="X-Session-Token"),
) -> SummaryDraft:
    _authorize_child(child_id, x_session_token)
    if scope not in {"weekly", "monthly", "quarterly", "term", "yearly"}:
        raise HTTPException(status_code=400, detail="Unsupported summary scope")
    return build_summary_draft(get_learning_store(), child_id=child_id, scope=scope)


@router.post("/children/{child_id}/summary-rollup", response_model=LearningSummary)
def rollup_learning_summary(
    child_id: str,
    scope: str = "weekly",
    x_session_token: str | None = Header(None, alias="X-Session-Token"),
) -> LearningSummary:
    _authorize_child(child_id, x_session_token)
    store = get_learning_store()
    if scope == "daily":
        return build_daily_summary(store, child_id=child_id)
    if scope == "weekly":
        return build_weekly_summary(store, child_id=child_id)
    if scope == "monthly":
        return build_monthly_summary(store, child_id=child_id)
    raise HTTPException(status_code=400, detail="Unsupported summary rollup scope")


@router.get("/children/{child_id}/wrong-questions", response_model=WrongQuestionList)
def get_wrong_questions(
    child_id: str,
    x_session_token: str | None = Header(None, alias="X-Session-Token"),
) -> WrongQuestionList:
    _authorize_child(child_id, x_session_token)
    return build_wrong_question_list(get_learning_store(), child_id=child_id)


@router.get("/children/{child_id}/safety-events", response_model=SafetyEventListResponse)
def get_safety_events(
    child_id: str,
    x_session_token: str | None = Header(None, alias="X-Session-Token"),
) -> SafetyEventListResponse:
    _authorize_child(child_id, x_session_token)
    return SafetyEventListResponse(
        child_id=child_id,
        items=get_learning_store().list_safety_events(child_id),
    )


@router.get("/children/{child_id}/learning-memory", response_model=LearningMemory)
def get_learning_memory(
    child_id: str,
    scope: str = "weekly",
    x_session_token: str | None = Header(None, alias="X-Session-Token"),
) -> LearningMemory:
    _authorize_child(child_id, x_session_token)
    if scope not in {"weekly", "monthly", "quarterly", "term", "yearly"}:
        raise HTTPException(status_code=400, detail="Unsupported learning memory scope")
    return build_learning_memory(get_learning_store(), child_id=child_id, scope=scope)


@router.get("/children/{child_id}/review-plan", response_model=ReviewPlan)
def get_review_plan(
    child_id: str,
    scope: str = "weekly",
    x_session_token: str | None = Header(None, alias="X-Session-Token"),
) -> ReviewPlan:
    _authorize_child(child_id, x_session_token)
    if scope not in {"weekly", "monthly", "quarterly", "term", "yearly"}:
        raise HTTPException(status_code=400, detail="Unsupported review scope")
    return build_review_plan(get_learning_store(), child_id=child_id, scope=scope)  # type: ignore[arg-type]
