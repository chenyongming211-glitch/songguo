from __future__ import annotations

from collections.abc import Mapping
import importlib.util
import os


VALID_AGENT_RUNTIMES = {"langgraph", "llm", "kernel"}


def resolve_agent_runtime(
    environ: Mapping[str, str] | None = None,
    *,
    langgraph_available: bool | None = None,
    default: str = "kernel",
) -> str:
    env = environ if environ is not None else os.environ
    raw = (
        env.get("SONGGUO_AGENT_RUNTIME")
        or env.get("SONGGUO_SESSION_RUNNER")
        or default
    )
    runtime = raw.strip().lower()
    if runtime not in VALID_AGENT_RUNTIMES:
        runtime = default if default in VALID_AGENT_RUNTIMES else "kernel"

    available = _langgraph_available() if langgraph_available is None else langgraph_available
    if runtime == "langgraph" and not available:
        return "llm"
    return runtime


def _langgraph_available() -> bool:
    return importlib.util.find_spec("langgraph") is not None
