from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from songguo.backend.evaluation.deepseek_progress_runner import (
    DeepSeekProgressOptions,
    run_deepseek_progress_evaluation,
)
from songguo.backend.evaluation.golden_math import build_golden_math_questions
from songguo.backend.services.learning.ai_engine import (
    DeepSeekProvider,
    DeterministicFallbackProvider,
    ProviderChain,
)
from songguo.backend.services.learning.real_model_client import load_real_model_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Songguo golden math evaluation with DeepSeek-first progress output."
    )
    parser.add_argument("--limit", type=int, default=100, help="Number of golden questions to run.")
    parser.add_argument(
        "--provider-timeout-seconds",
        type=float,
        default=20.0,
        help="Outer timeout for one DeepSeek structure call.",
    )
    parser.add_argument(
        "--provider-retry-attempts",
        type=int,
        default=1,
        help="Outer ProviderChain retry attempts. Real HTTP retry is controlled by SONGGUO_REAL_MODEL_RETRY_ATTEMPTS.",
    )
    parser.add_argument(
        "--provider-retry-backoff-seconds",
        type=float,
        default=0.0,
        help="Outer ProviderChain retry backoff.",
    )
    parser.add_argument(
        "--stop-on-first-failure",
        action="store_true",
        help="Stop after the first DeepSeek direct failure.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print final report JSON after the human-readable summary.",
    )
    args = parser.parse_args()

    config = load_real_model_config()
    print(
        "config "
        f"provider={config.binding} "
        f"model={config.model} "
        f"has_key={bool(config.api_key)} "
        f"real_timeout={config.timeout_seconds} "
        f"real_retries={config.retry_attempts}",
        flush=True,
    )
    if not config.api_key:
        print("error=missing LLM_API_KEY or DEEPSEEK_API_KEY", flush=True)
        return 2

    deepseek_provider = ProviderChain(
        [DeepSeekProvider(model_name=config.model)],
        provider_timeout_seconds=args.provider_timeout_seconds,
        provider_retry_attempts=args.provider_retry_attempts,
        provider_retry_backoff_seconds=args.provider_retry_backoff_seconds,
    )
    report = run_deepseek_progress_evaluation(
        questions=build_golden_math_questions(),
        deepseek_provider=deepseek_provider,
        fallback_provider=DeterministicFallbackProvider(),
        options=DeepSeekProgressOptions(
            limit=args.limit,
            stop_on_first_failure=args.stop_on_first_failure,
        ),
        emit=lambda line: print(line, flush=True),
    )
    if args.json:
        print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False), flush=True)
    return 0 if report.fallback_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
