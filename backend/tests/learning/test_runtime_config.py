from __future__ import annotations

from songguo.backend.services.learning import service
from songguo.backend.services.learning.store import InMemoryLearningStore
from songguo.backend.services.learning.runtime_config import resolve_agent_runtime


def test_runtime_config_prefers_explicit_langgraph() -> None:
    runtime = resolve_agent_runtime(
        {"SONGGUO_AGENT_RUNTIME": "langgraph"},
        langgraph_available=True,
    )

    assert runtime == "langgraph"


def test_runtime_config_falls_back_from_missing_langgraph_to_llm() -> None:
    runtime = resolve_agent_runtime(
        {"SONGGUO_AGENT_RUNTIME": "langgraph"},
        langgraph_available=False,
    )

    assert runtime == "llm"


def test_runtime_config_keeps_legacy_session_runner_mode() -> None:
    runtime = resolve_agent_runtime(
        {"SONGGUO_SESSION_RUNNER": "kernel"},
        langgraph_available=True,
    )

    assert runtime == "kernel"


def test_session_runner_from_env_uses_deepseek_llm_runtime(monkeypatch) -> None:
    monkeypatch.setenv("SONGGUO_SESSION_RUNNER", "llm")
    monkeypatch.setenv("LLM_BINDING", "deepseek")
    monkeypatch.setenv("LLM_MODEL", "deepseek-chat")
    monkeypatch.setenv("SONGGUO_LLM_SESSION_TIMEOUT_SECONDS", "6")
    monkeypatch.delenv("SONGGUO_AI_ENGINE_PROVIDER", raising=False)
    monkeypatch.delenv("DEEPTUTOR_LLM_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)

    runner = service._build_session_runner_from_env()

    assert runner is not None
    assert runner.provider == "deepseek"
    assert runner.model == "deepseek-chat"
    assert runner.timeout_seconds == 6


def test_default_learning_service_uses_dotenv_agent_runtime(monkeypatch) -> None:
    values = {
        "SONGGUO_AGENT_RUNTIME": "langgraph",
        "SONGGUO_SESSION_RUNNER": "llm",
        "SONGGUO_AI_ENGINE_PROVIDER": "deepseek",
        "LLM_BINDING": "deepseek",
        "LLM_MODEL": "deepseek-chat",
    }
    for key in values:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(service, "_read_local_dotenv_value", lambda key: values.get(key, ""))
    monkeypatch.setattr(service, "SQLiteLearningStore", InMemoryLearningStore)

    learning_service = service.build_default_learning_service()

    assert learning_service.agent_runtime == "langgraph"
    assert learning_service.session_runner is not None
    assert learning_service.session_runner.provider == "deepseek"
    assert learning_service.tutor_graph is not None


def test_default_learning_service_uses_postgres_store_when_database_url_is_set(
    monkeypatch,
) -> None:
    class FakePostgresLearningStore:
        def __init__(self, database_url: str) -> None:
            self.database_url = database_url

    monkeypatch.setenv("SONGGUO_DATABASE_URL", "postgresql://songguo:test@localhost/songguo")
    monkeypatch.setattr(service, "PostgresLearningStore", FakePostgresLearningStore, raising=False)

    learning_service = service.build_default_learning_service()

    assert isinstance(learning_service.store, FakePostgresLearningStore)
    assert learning_service.store.database_url == "postgresql://songguo:test@localhost/songguo"


def test_default_learning_service_keeps_sqlite_store_without_database_url(
    monkeypatch,
) -> None:
    class FakeSQLiteLearningStore(InMemoryLearningStore):
        pass

    monkeypatch.delenv("SONGGUO_DATABASE_URL", raising=False)
    monkeypatch.setattr(service, "_read_local_dotenv_value", lambda key: "")
    monkeypatch.setattr(service, "SQLiteLearningStore", FakeSQLiteLearningStore)

    learning_service = service.build_default_learning_service()

    assert isinstance(learning_service.store, FakeSQLiteLearningStore)
