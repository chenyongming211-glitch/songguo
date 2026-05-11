from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from songguo.backend.services.learning.service import build_default_learning_service
from songguo.backend.services.learning.real_model_client import get_llm_client


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Songguo real-model runtime config.")
    parser.add_argument(
        "--call",
        action="store_true",
        help="Also run a minimal real LLM JSON call.",
    )
    args = parser.parse_args()

    service = build_default_learning_service()
    print(f"runtime={service.agent_runtime}")
    print(f"runner_provider={service.session_runner.provider if service.session_runner else None}")
    print(f"runner_model={service.session_runner.model if service.session_runner else None}")
    print(f"tutor_graph_enabled={service.tutor_graph is not None}")

    if not args.call:
        return 0

    text = get_llm_client().complete_sync(
        '请只输出JSON：{"child_message":"你好","structured_state":{"phase":"WAIT_CHILD_ATTEMPT","answer_unlocked":false},"learning_deposit_delta":{"knowledge_point":"test","question_type":"test"},"practice_items":[]}',
        system_prompt="你只输出 JSON，不输出 Markdown。",
        temperature=0,
    )
    print(f"real_model_call_ok={bool(text.strip())}")
    print(f"response_prefix={text.strip()[:160].replace(chr(10), ' ')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
