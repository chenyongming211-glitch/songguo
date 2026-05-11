from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from songguo.backend.services.learning.service import build_default_learning_service


DEFAULT_QUESTION = (
    "学校组织三年级学生春游，一共有4个班，每班32人。"
    "如果每辆大巴车限坐45人，那么至少需要多少辆大巴车？"
)
DEFAULT_ANSWERS = ["128", "90", "不够", "135", "3"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Songguo MathMistakeTutorGraph demo.")
    parser.add_argument("--child-id", default="child_m8_demo_script")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--answers",
        default="|".join(DEFAULT_ANSWERS),
        help="Pipe-separated child answers, for example: 128|90|不够|135|3",
    )
    args = parser.parse_args()

    service = build_default_learning_service()
    print(f"runtime={service.agent_runtime}")
    print(f"runner_provider={service.session_runner.provider if service.session_runner else None}")
    print(f"runner_model={service.session_runner.model if service.session_runner else None}")
    print(f"tutor_graph_enabled={service.tutor_graph is not None}")
    print(f"question={args.question}")

    created = service.create_session(
        child_id=args.child_id,
        subject="math",
        grade=3,
        question_text=args.question,
    )
    print(f"session_id={created.session_id}")
    print(f"assistant={created.message}")

    last_result = None
    for answer in [item.strip() for item in args.answers.split("|") if item.strip()]:
        last_result = service.submit_attempt(created.session_id, child_answer=answer)
        print(f"child={answer}")
        print(
            "result="
            f"correct:{last_result.correct},"
            f"phase:{last_result.phase},"
            f"practice_count:{len(last_result.practice_items)}"
        )
        print(f"assistant={last_result.message}")

    print(f"deposit_saved={service.store.get_learning_deposit(created.session_id) is not None}")
    logs = service.store.list_ai_call_logs(child_id=args.child_id)
    print(f"ai_call_count={len(logs)}")
    for call in logs[-8:]:
        print(
            "ai_call="
            f"{call.provider},"
            f"{call.model},"
            f"{call.operation},"
            f"{call.status},"
            f"tokens:{call.token_estimate}"
        )
    return 0 if last_result is None or last_result.correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
