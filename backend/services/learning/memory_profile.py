from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, Field

from songguo.backend.services.learning.labels import knowledge_point_label, misconception_label
from songguo.backend.services.learning.store import InMemoryLearningStore


class WeaknessItem(BaseModel):
    knowledge_point: str
    knowledge_point_label: str
    wrong_count: int
    average_hint_level: float
    common_misconceptions: list[str] = Field(default_factory=list)
    common_misconception_labels: list[str] = Field(default_factory=list)
    priority_score: float
    mastery_score: int
    risk_level: str


class LearningMemory(BaseModel):
    child_id: str
    scope: str
    top_weaknesses: list[WeaknessItem] = Field(default_factory=list)
    summary: str
    next_actions: list[str] = Field(default_factory=list)


def build_learning_memory(
    store: InMemoryLearningStore,
    *,
    child_id: str,
    scope: str = "weekly",
    now: datetime | None = None,
) -> LearningMemory:
    reference_time = now or datetime.now(timezone.utc)
    cutoff = reference_time - _scope_window(scope)
    wrong_questions = [
        item for item in store.list_wrong_questions(child_id) if item.updated_at >= cutoff
    ]
    by_knowledge_point: dict[str, list] = defaultdict(list)
    for item in wrong_questions:
        by_knowledge_point[item.knowledge_point].append(item)

    weaknesses: list[WeaknessItem] = []
    for knowledge_point, items in by_knowledge_point.items():
        misconception_counts = Counter(
            item.last_misconception for item in items if item.last_misconception
        )
        average_hint_level = round(
            sum(item.highest_hint_level for item in items) / len(items),
            2,
        )
        priority_score = len(items) * 10 + average_hint_level
        mastery_score = _mastery_score(
            wrong_count=len(items),
            average_hint_level=average_hint_level,
        )
        weaknesses.append(
            WeaknessItem(
                knowledge_point=knowledge_point,
                knowledge_point_label=knowledge_point_label(knowledge_point),
                wrong_count=len(items),
                average_hint_level=average_hint_level,
                common_misconceptions=[
                    misconception for misconception, _ in misconception_counts.most_common(3)
                ],
                common_misconception_labels=[
                    misconception_label(misconception)
                    for misconception, _ in misconception_counts.most_common(3)
                ],
                priority_score=priority_score,
                mastery_score=mastery_score,
                risk_level=_risk_level(mastery_score),
            )
        )

    weaknesses.sort(key=lambda item: item.priority_score, reverse=True)
    return LearningMemory(
        child_id=child_id,
        scope=scope,
        top_weaknesses=weaknesses[:5],
        summary=_summary(weaknesses),
        next_actions=_next_actions(weaknesses),
    )


def _summary(weaknesses: list[WeaknessItem]) -> str:
    if not weaknesses:
        return "还没有足够学习记录形成薄弱项画像。"
    top = weaknesses[0]
    point = knowledge_point_label(top.knowledge_point)
    misconception = misconception_label(
        top.common_misconceptions[0] if top.common_misconceptions else None
    )
    return (
        f"当前最需要关注的是{point}，出现 {top.wrong_count} 次错误，"
        f"常见错因是 {misconception}。"
    )


def _next_actions(weaknesses: list[WeaknessItem]) -> list[str]:
    if not weaknesses:
        return ["先完成 1-2 次错题引导，积累学习证据。"]
    top = weaknesses[0]
    return [
        f"围绕{knowledge_point_label(top.knowledge_point)}每天练 3 道小题。",
        "练习后继续记录是否仍需要高等级提示。",
    ]


def _scope_window(scope: str) -> timedelta:
    return {
        "weekly": timedelta(days=7),
        "monthly": timedelta(days=31),
        "quarterly": timedelta(days=92),
        "term": timedelta(days=190),
        "yearly": timedelta(days=366),
    }.get(scope, timedelta(days=7))


def _mastery_score(*, wrong_count: int, average_hint_level: float) -> int:
    penalty = wrong_count * 12 + int(round(average_hint_level * 8))
    return max(0, min(100, 100 - penalty))


def _risk_level(mastery_score: int) -> str:
    if mastery_score < 60:
        return "high"
    if mastery_score < 80:
        return "medium"
    return "low"
