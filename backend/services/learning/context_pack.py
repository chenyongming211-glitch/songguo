from __future__ import annotations

from pydantic import BaseModel, Field

from songguo.backend.services.learning.labels import knowledge_point_label, misconception_label
from songguo.backend.services.learning.memory_profile import LearningMemory, build_learning_memory
from songguo.backend.services.learning.store import InMemoryLearningStore


class RelatedEvidence(BaseModel):
    session_id: str
    question_text: str
    knowledge_point: str
    knowledge_point_label: str
    misconception: str | None = None
    misconception_label: str
    evidence: str


class ParentSettingsContext(BaseModel):
    answer_mode: str = "strict_guided"
    focus: list[str] = Field(default_factory=list)


class StudentContextPack(BaseModel):
    child_id: str
    grade: int
    current_question: str
    recent_week_summary: LearningMemory
    parent_settings: ParentSettingsContext = Field(default_factory=ParentSettingsContext)
    short_term_goal: str
    related_evidence: list[RelatedEvidence] = Field(default_factory=list)


def build_student_context_pack(
    store: InMemoryLearningStore,
    *,
    child_id: str,
    grade: int,
    current_question: str,
    evidence_limit: int = 3,
) -> StudentContextPack:
    memory = build_learning_memory(store, child_id=child_id, scope="weekly")
    related = _related_evidence(
        store,
        child_id=child_id,
        current_question=current_question,
        limit=max(0, min(3, evidence_limit)),
    )
    return StudentContextPack(
        child_id=child_id,
        grade=grade,
        current_question=current_question,
        recent_week_summary=memory,
        parent_settings=ParentSettingsContext(
            focus=[
                weakness.knowledge_point_label
                for weakness in memory.top_weaknesses[:2]
            ],
        ),
        short_term_goal=_short_term_goal(memory),
        related_evidence=related,
    )


def _related_evidence(
    store: InMemoryLearningStore,
    *,
    child_id: str,
    current_question: str,
    limit: int,
) -> list[RelatedEvidence]:
    if limit <= 0:
        return []
    current_text = current_question.strip()
    scored: list[tuple[int, object]] = []
    for item in store.list_wrong_questions(child_id):
        score = 0
        if item.knowledge_point and item.knowledge_point in current_text:
            score += 5
        if any(token in current_text for token in ["至少", "最多", "限坐", "每辆"]):
            if item.knowledge_point == "capacity_round_up":
                score += 10
        if item.last_misconception:
            score += 1
        scored.append((score, item))
    scored.sort(key=lambda pair: (pair[0], pair[1].updated_at), reverse=True)
    result: list[RelatedEvidence] = []
    for _, item in scored[:limit]:
        result.append(
            RelatedEvidence(
                session_id=item.session_id,
                question_text=item.normalized_question,
                knowledge_point=item.knowledge_point,
                knowledge_point_label=knowledge_point_label(item.knowledge_point),
                misconception=item.last_misconception,
                misconception_label=misconception_label(item.last_misconception),
                evidence=item.mistake_summary,
            )
        )
    return result


def _short_term_goal(memory: LearningMemory) -> str:
    if not memory.top_weaknesses:
        return "先完成一道题，观察孩子读题、尝试和提示依赖情况。"
    top = memory.top_weaknesses[0]
    return f"优先巩固{top.knowledge_point_label}，观察是否还需要高等级提示。"
