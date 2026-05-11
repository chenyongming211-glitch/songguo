from __future__ import annotations

import os

from fastapi import FastAPI

from songguo.backend.api.routers import (
    learning,
    learning_artifacts,
    parent_reports,
    reminders,
    system,
    wechat,
)


def experimental_features_enabled() -> bool:
    return os.getenv("SONGGUO_ENABLE_EXPERIMENTAL_FEATURES", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def create_app() -> FastAPI:
    app = FastAPI(title="Songguo AI Backend", version="0.1.0")
    app.include_router(system.router, prefix="/api/v1/system", tags=["system"])
    app.include_router(learning.router, prefix="/api/v1/learning", tags=["learning"])
    app.include_router(parent_reports.router, prefix="/api/v1/parent", tags=["parent"])
    app.include_router(wechat.router, prefix="/api/v1/wechat", tags=["wechat"])
    if experimental_features_enabled():
        app.include_router(reminders.router, prefix="/api/v1/reminders", tags=["reminders"])
        app.include_router(
            learning_artifacts.router,
            prefix="/api/v1/learning-artifacts",
            tags=["learning-artifacts"],
        )
    return app


app = create_app()
