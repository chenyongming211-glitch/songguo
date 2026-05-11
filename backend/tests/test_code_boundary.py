from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_songguo_backend_code_is_not_stored_under_deeptutor_package() -> None:
    old_paths = [
        ROOT / "deeptutor/services/learning",
        ROOT / "deeptutor/services/wechat.py",
        ROOT / "deeptutor/services/session_auth.py",
        ROOT / "deeptutor/services/production_readiness.py",
        ROOT / "deeptutor/api/routers/learning.py",
        ROOT / "deeptutor/api/routers/parent_reports.py",
        ROOT / "deeptutor/api/routers/wechat.py",
        ROOT / "deeptutor/api/routers/reminders.py",
        ROOT / "deeptutor/api/routers/learning_artifacts.py",
    ]

    assert [str(path.relative_to(ROOT)) for path in old_paths if path.exists()] == []


def test_songguo_backend_import_roots_exist() -> None:
    modules = [
        "songguo.backend.services.learning.service",
        "songguo.backend.services.wechat",
        "songguo.backend.services.session_auth",
        "songguo.backend.services.production_readiness",
        "songguo.backend.api.routers.learning",
        "songguo.backend.api.routers.parent_reports",
        "songguo.backend.api.routers.wechat",
        "songguo.backend.api.routers.reminders",
        "songguo.backend.api.routers.learning_artifacts",
    ]

    missing = [module for module in modules if importlib.util.find_spec(module) is None]

    assert missing == []
