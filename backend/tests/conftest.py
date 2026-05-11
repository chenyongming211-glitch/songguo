from __future__ import annotations

import os


if os.getenv("SONGGUO_ALLOW_REAL_MODEL_TESTS") != "1":
    os.environ["SONGGUO_AGENT_RUNTIME"] = "kernel"
    os.environ["SONGGUO_SESSION_RUNNER"] = "kernel"
    os.environ["SONGGUO_AI_ENGINE_PROVIDER"] = "fallback"
