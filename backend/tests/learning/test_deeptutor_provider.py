from __future__ import annotations

from songguo.backend.services.learning.deeptutor_provider import OrchestratorDraftProvider
from songguo.backend.services.learning.math_structuring import LLMMathStructurer
from songguo.backend.services.learning.service import build_default_learning_service


class _StreamEvent:
    def __init__(self, *, type: str, source: str, content: str = "") -> None:
        self.type = type
        self.source = source
        self.content = content


class _FakeOrchestrator:
    async def handle(self, context):
        yield _StreamEvent(type="session", source="orchestrator")
        yield _StreamEvent(
            type="content",
            source="chat",
            content=f"提示：{context.user_message}",
        )
        yield _StreamEvent(type="done", source="chat")


def test_orchestrator_provider_collects_content_into_teaching_draft() -> None:
    provider = OrchestratorDraftProvider(orchestrator=_FakeOrchestrator())

    draft = provider.generate(
        {
            "prompt_id": "grade3_math_hint",
            "question_text": "36 x 5 = ?",
            "grade": 3,
            "knowledge_point": "two_digit_times_one_digit",
            "hint_level": 1,
            "misconception_tag": "treated_x5_like_x10",
        }
    )

    assert draft.action == "hint"
    assert draft.hint_level == 1
    assert draft.exposes_final_answer is False
    assert "36 x 5" in draft.text
    assert draft.trace_id.startswith("deeptutor_")


class _ErrorOrchestrator:
    async def handle(self, context):
        yield _StreamEvent(type="error", source="chat", content="model failed")
        yield _StreamEvent(type="done", source="chat")


def test_orchestrator_provider_returns_safe_fallback_on_error() -> None:
    provider = OrchestratorDraftProvider(orchestrator=_ErrorOrchestrator())

    draft = provider.generate(
        {
            "prompt_id": "grade3_math_hint",
            "question_text": "36 x 5 = ?",
            "hint_level": 2,
        }
    )

    assert draft.action == "hint"
    assert draft.hint_level == 2
    assert draft.exposes_final_answer is False
    assert "模型暂时不可用" in draft.text


def test_default_learning_service_uses_orchestrator_provider_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("DEEPTUTOR_LEARNING_USE_ORCHESTRATOR", "1")

    service = build_default_learning_service()

    assert service.adapter.draft_generator is not None


def test_default_learning_service_wires_llm_math_structurer_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("SONGGUO_MATH_STRUCTURING_PROVIDER", "llm")

    service = build_default_learning_service()

    assert isinstance(service.math_gateway.structurer, LLMMathStructurer)


def test_default_learning_service_reads_math_structurer_from_dotenv_when_env_not_preloaded(
    monkeypatch,
) -> None:
    import songguo.backend.services.learning.service as service_module

    monkeypatch.delenv("SONGGUO_MATH_STRUCTURING_PROVIDER", raising=False)
    monkeypatch.setattr(
        service_module,
        "_read_local_dotenv_value",
        lambda key: "llm" if key == "SONGGUO_MATH_STRUCTURING_PROVIDER" else "",
        raising=False,
    )

    service = build_default_learning_service()

    assert isinstance(service.math_gateway.structurer, LLMMathStructurer)
