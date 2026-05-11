from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from songguo.backend.services.learning.artifacts import (
    LearningArtifact,
    generate_static_svg_diagram,
    request_animation_artifact,
)
from songguo.backend.services.learning.service import get_global_learning_store
from songguo.backend.services.learning.store import InMemoryLearningStore
from songguo.backend.services.session_auth import authorize_child_access

router = APIRouter()


class DiagramRequest(BaseModel):
    child_id: str
    question_text: str
    knowledge_point: str


def get_artifact_root() -> Path:
    return Path("data/user/learning_artifacts")


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


@router.post("/diagram", response_model=LearningArtifact)
def generate_diagram(
    request: DiagramRequest,
    x_session_token: str | None = Header(None, alias="X-Session-Token"),
) -> LearningArtifact:
    _authorize_child(request.child_id, x_session_token)
    return generate_static_svg_diagram(
        artifact_root=get_artifact_root(),
        child_id=request.child_id,
        question_text=request.question_text,
        knowledge_point=request.knowledge_point,
    )


@router.post("/animation", response_model=LearningArtifact)
def request_animation(
    request: DiagramRequest,
    x_session_token: str | None = Header(None, alias="X-Session-Token"),
) -> LearningArtifact:
    _authorize_child(request.child_id, x_session_token)
    return request_animation_artifact(
        artifact_root=get_artifact_root(),
        child_id=request.child_id,
        question_text=request.question_text,
        knowledge_point=request.knowledge_point,
    )
