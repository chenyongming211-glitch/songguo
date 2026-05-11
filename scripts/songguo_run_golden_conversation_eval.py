from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from songguo.backend.evaluation.golden_conversations import (
    build_golden_conversation_cases,
    evaluate_golden_conversations,
)
from songguo.backend.services.learning.llm_session_runner import LLMSessionRunner


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Songguo golden conversation evaluation for Tutor State Contract."
    )
    parser.add_argument("--json", action="store_true", help="Print report JSON.")
    args = parser.parse_args()

    report = evaluate_golden_conversations(
        cases=build_golden_conversation_cases(),
        session_runner=LLMSessionRunner(),
    )
    print(
        "summary "
        f"total={report.total} "
        f"passed={report.passed_count} "
        f"success_rate={report.success_rate:.2%} "
        f"readiness_failures={report.readiness_failures} "
        f"status_failures={report.child_answer_status_failures} "
        f"teacher_move_failures={report.teacher_move_failures} "
        f"phase_failures={report.phase_failures} "
        f"message_contract_failures={report.message_contract_failures} "
        f"provider_failures={report.provider_failures}",
        flush=True,
    )
    if report.failed_case_ids:
        print("failed_case_ids=" + ",".join(report.failed_case_ids), flush=True)
    if args.json:
        print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False), flush=True)
    return 0 if report.passed_count == report.total else 1


if __name__ == "__main__":
    raise SystemExit(main())
