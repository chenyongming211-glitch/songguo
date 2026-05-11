from __future__ import annotations

import os

from fastapi import APIRouter
from pydantic import BaseModel, Field

from songguo.backend.services.learning.real_model_client import load_real_model_config
from songguo.backend.services.learning.service import get_global_learning_service
from songguo.backend.services.production_readiness import (
    ProductionReadiness,
    check_production_readiness,
)

router = APIRouter()


class ComponentStatus(BaseModel):
    status: str
    model: str | None = None
    provider: str | None = None


class SystemStatus(BaseModel):
    backend: ComponentStatus
    llm: ComponentStatus
    embeddings: ComponentStatus
    search: ComponentStatus
    runtime: str
    production_readiness: ProductionReadiness
    warnings: list[str] = Field(default_factory=list)


@router.get("/status", response_model=SystemStatus)
def get_system_status() -> SystemStatus:
    service = get_global_learning_service()
    model_config = load_real_model_config()
    readiness = check_production_readiness()
    llm_status = "configured" if model_config.api_key else "missing_api_key"
    return SystemStatus(
        backend=ComponentStatus(status="ok", provider="songguo"),
        llm=ComponentStatus(
            status=llm_status,
            model=model_config.model,
            provider=model_config.binding,
        ),
        embeddings=ComponentStatus(
            status="not_configured",
            model=os.getenv("SONGGUO_EMBEDDING_MODEL") or None,
            provider=os.getenv("SONGGUO_EMBEDDING_PROVIDER") or None,
        ),
        search=ComponentStatus(
            status="not_configured",
            provider=os.getenv("SONGGUO_SEARCH_PROVIDER") or "none",
        ),
        runtime=service.agent_runtime,
        production_readiness=readiness,
        warnings=readiness.warnings,
    )
