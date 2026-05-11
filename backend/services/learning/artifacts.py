from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field


class LearningArtifact(BaseModel):
    artifact_id: str = Field(default_factory=lambda: f"art_{uuid4().hex}")
    child_id: str
    artifact_type: str
    status: str
    path: str
    title: str


def generate_static_svg_diagram(
    *,
    artifact_root: Path,
    child_id: str,
    question_text: str,
    knowledge_point: str,
) -> LearningArtifact:
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_id = f"art_{uuid4().hex}"
    filename = f"{artifact_id}.svg"
    target = artifact_root / filename
    title = "错题图解卡"
    svg = _build_svg(title=title, question_text=question_text, knowledge_point=knowledge_point)
    target.write_text(svg, encoding="utf-8")
    return LearningArtifact(
        artifact_id=artifact_id,
        child_id=child_id,
        artifact_type="svg",
        status="ready",
        path=filename,
        title=title,
    )


def request_animation_artifact(
    *,
    artifact_root: Path,
    child_id: str,
    question_text: str,
    knowledge_point: str,
) -> LearningArtifact:
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_id = f"art_{uuid4().hex}"
    filename = f"{artifact_id}.json"
    title = "动态图解任务"
    job = {
        "artifact_id": artifact_id,
        "child_id": child_id,
        "artifact_type": "animation",
        "status": "queued",
        "question_text": question_text,
        "knowledge_point": knowledge_point,
        "title": title,
        "worker": "manim",
    }
    (artifact_root / filename).write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    return LearningArtifact(
        artifact_id=artifact_id,
        child_id=child_id,
        artifact_type="animation",
        status="queued",
        path=filename,
        title=title,
    )


def _build_svg(*, title: str, question_text: str, knowledge_point: str) -> str:
    safe_question = _escape_svg(question_text)
    safe_point = _escape_svg(knowledge_point)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="720" height="420" viewBox="0 0 720 420">
  <rect width="720" height="420" rx="32" fill="#fffaf3"/>
  <rect x="36" y="36" width="648" height="348" rx="28" fill="#f0fdfa" stroke="#0f766e" stroke-width="4"/>
  <text x="64" y="96" font-size="34" font-weight="700" fill="#0f172a">{_escape_svg(title)}</text>
  <text x="64" y="158" font-size="30" fill="#134e4a">题目：{safe_question}</text>
  <text x="64" y="218" font-size="26" fill="#334155">知识点：{safe_point}</text>
  <text x="64" y="278" font-size="26" fill="#334155">先拆成更简单的一步，再让孩子自己补最后一步。</text>
  <text x="64" y="334" font-size="24" fill="#64748b">松果AI 受控教学图解卡</text>
</svg>
"""


def _escape_svg(value: str) -> str:
    return (
        str(value or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
