from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_e2e_script():
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "songguo_run_photo_submission_e2e_eval.py"
    spec = importlib.util.spec_from_file_location("songguo_run_photo_submission_e2e_eval", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load script module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_env_agent_service_uses_math_gateway_from_env(monkeypatch) -> None:
    e2e_script = _load_e2e_script()

    class SentinelMathGateway:
        pass

    sentinel = SentinelMathGateway()
    monkeypatch.setattr(e2e_script, "_build_intent_router_from_env", lambda: None)
    monkeypatch.setattr(e2e_script, "_build_basic_subject_rubric_evaluator_from_env", lambda: None)
    monkeypatch.setattr(e2e_script, "_build_math_gateway_from_env", lambda: sentinel, raising=False)

    service = e2e_script._build_service("env")

    assert service.math_gateway is sentinel
