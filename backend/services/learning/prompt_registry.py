from __future__ import annotations

from pydantic import BaseModel


class UnknownPromptError(ValueError):
    pass


class PromptTemplate(BaseModel):
    prompt_id: str
    version: str
    purpose: str


class PromptRegistry:
    def __init__(self) -> None:
        self._templates = {
            "grade3_math_hint": PromptTemplate(
                prompt_id="grade3_math_hint",
                version="v1.0.0",
                purpose="Generate a low-exposure math hint.",
            ),
            "grade3_math_explanation": PromptTemplate(
                prompt_id="grade3_math_explanation",
                version="v1.0.0",
                purpose="Generate a full explanation after unlock.",
            ),
            "grade3_math_similar_practice": PromptTemplate(
                prompt_id="grade3_math_similar_practice",
                version="v1.0.0",
                purpose="Generate targeted similar-practice items.",
            ),
            "learning_memory_summary": PromptTemplate(
                prompt_id="learning_memory_summary",
                version="v1.0.0",
                purpose="Summarize learning evidence into a memory draft.",
            ),
        }

    def require(self, prompt_id: str) -> PromptTemplate:
        template = self._templates.get(prompt_id)
        if template is None:
            raise UnknownPromptError(f"Unknown learning prompt: {prompt_id}")
        return template

