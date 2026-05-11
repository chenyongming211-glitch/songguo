from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from songguo.backend.services.learning.real_model_client import (
    RealModelConfig,
    load_real_model_config,
    request_chat_completion,
)


class OpenAICompatibleChatModel(BaseChatModel):
    config: RealModelConfig
    temperature: float = 0

    @property
    def _llm_type(self) -> str:
        return f"songguo-{self.config.binding}-chat"

    @property
    def provider(self) -> str:
        return self.config.binding

    @property
    def model(self) -> str:
        return self.config.model

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if stop:
            raise ValueError("Stop sequences are not supported by Songguo chat model")
        if not self.config.api_key:
            raise RuntimeError("LLM_API_KEY is not configured")

        payload = request_chat_completion(
            self.config.model_copy(
                update={"timeout_seconds": float(kwargs.get("timeout", self.config.timeout_seconds))}
            ),
            json_payload={
                "model": self.config.model,
                "messages": [_message_to_openai_payload(message) for message in messages],
                "temperature": kwargs.get("temperature", self.temperature),
                "response_format": {"type": "json_object"},
            },
        )
        content = str(payload["choices"][0]["message"]["content"])
        usage = payload.get("usage") if isinstance(payload, dict) else None
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=content,
                        response_metadata={"usage": usage or {}},
                    )
                )
            ],
            llm_output={"usage": usage or {}},
        )


class LangChainLLMClient:
    def __init__(self, chat_model: BaseChatModel | None = None) -> None:
        self.chat_model = chat_model or OpenAICompatibleChatModel(
            config=load_real_model_config()
        )

    @property
    def provider(self) -> str:
        return str(getattr(self.chat_model, "provider", "langchain"))

    @property
    def model(self) -> str:
        return str(getattr(self.chat_model, "model", "configured"))

    def complete_sync(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0,
    ) -> str:
        messages: list[BaseMessage] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        result = self.chat_model.invoke(messages, temperature=temperature)
        content = result.content
        if isinstance(content, str):
            return content
        return "".join(str(item) for item in content)


def get_langchain_llm_client() -> LangChainLLMClient:
    return LangChainLLMClient()


def _message_to_openai_payload(message: BaseMessage) -> dict[str, str]:
    if isinstance(message, SystemMessage):
        role = "system"
    elif isinstance(message, AIMessage):
        role = "assistant"
    else:
        role = "user"
    content = message.content
    if not isinstance(content, str):
        content = "".join(str(item) for item in content)
    return {"role": role, "content": content}
