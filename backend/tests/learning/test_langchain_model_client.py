from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from songguo.backend.services.learning.langchain_model_client import (
    LangChainLLMClient,
    OpenAICompatibleChatModel,
)
from songguo.backend.services.learning.real_model_client import RealModelConfig


def test_langchain_llm_client_invokes_chat_model_with_system_and_user_messages() -> None:
    seen = {}

    class FakeChatModel:
        provider = "deepseek"
        model = "deepseek-chat"

        def invoke(self, messages, **kwargs):
            seen["messages"] = messages
            seen["kwargs"] = kwargs

            class Result:
                content = '{"child_message":"你好"}'

            return Result()

    client = LangChainLLMClient(chat_model=FakeChatModel())

    output = client.complete_sync(
        "prompt-json",
        system_prompt="system-json-only",
        temperature=0,
    )

    assert output == '{"child_message":"你好"}'
    assert isinstance(seen["messages"][0], SystemMessage)
    assert isinstance(seen["messages"][1], HumanMessage)
    assert seen["messages"][0].content == "system-json-only"
    assert seen["messages"][1].content == "prompt-json"
    assert seen["kwargs"]["temperature"] == 0
    assert client.provider == "deepseek"
    assert client.model == "deepseek-chat"


def test_openai_compatible_chat_model_requests_json_response(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": '{"child_message":"你好"}'}}],
                "usage": {"total_tokens": 12},
            }

    def fake_post(endpoint, *, headers, json, timeout):
        captured["endpoint"] = endpoint
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)
    model = OpenAICompatibleChatModel(
        config=RealModelConfig(
            binding="deepseek",
            model="deepseek-chat",
            api_key="key",
            base_url="https://api.deepseek.com",
        )
    )

    result = model.invoke(
        [
            SystemMessage(content="只输出 JSON"),
            HumanMessage(content="题目"),
        ],
        temperature=0,
    )

    assert result.content == '{"child_message":"你好"}'
    assert captured["endpoint"] == "https://api.deepseek.com/chat/completions"
    assert captured["json"]["model"] == "deepseek-chat"
    assert captured["json"]["response_format"] == {"type": "json_object"}
    assert captured["json"]["messages"] == [
        {"role": "system", "content": "只输出 JSON"},
        {"role": "user", "content": "题目"},
    ]
