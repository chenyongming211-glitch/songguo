# Golden Math Evaluation Report

Date: 2026-05-08

## Scope

This report records the current v0.1 math evaluation after fixing the generic
arithmetic similar-practice gap.

Evaluated paths:

1. 100-question deterministic golden math evaluation.
2. 10-question DeepSeek-first sampled evaluation with deterministic fallback.
3. Full MathMistakeTutorGraph regression for completion, LearningDeposit,
   parent feedback and similar-practice quality.

## Deterministic 100-Question Golden Set

Current golden set shape:

- 100 questions.
- Grades 3-6 only.
- 21 elementary math categories.
- Interleaved order, so early samples are representative instead of grouped by
  one category.

Command:

```bash
PYTHONPATH=/Users/chen/code python - <<'PY'
from songguo.backend.evaluation.golden_math import build_golden_math_questions
from songguo.backend.evaluation.runner import evaluate_math_provider
from songguo.backend.services.learning.ai_engine import DeterministicFallbackProvider

report = evaluate_math_provider(
    provider=DeterministicFallbackProvider(),
    questions=build_golden_math_questions(),
)
print(report.model_dump())
print("skill_hit_rate=", report.skill_hit_rate)
print("concept_card_hit_rate=", report.concept_card_hit_rate)
PY
```

Result:

```text
total: 100
schema_failures: 0
answer_leakage_count: 0
provider_failures: 0
skill_hit_count: 100
concept_card_hit_count: 100
controlled_generation_failures: 0
practice_generation_failures: 0
skill_hit_rate: 1.0
concept_card_hit_rate: 1.0
```

## DeepSeek-First 100-Question Run

Command:

```bash
PYTHONPATH=/Users/chen/code \
SONGGUO_REAL_MODEL_TIMEOUT_SECONDS=8 \
SONGGUO_REAL_MODEL_RETRY_ATTEMPTS=2 \
SONGGUO_REAL_MODEL_RETRY_BACKOFF_SECONDS=0.2 \
python -u /Users/chen/code/songguo/scripts/songguo_run_golden_math_eval.py \
  --limit 100 \
  --provider-timeout-seconds 20 \
  --provider-retry-attempts 1 \
  --json
```

Result on the interleaved 3-6 grade set:

```text
total: 100
deepseek_success: 47
fallback_success: 53
fallback_failures: 0
failure_counts:
  timeout: 34
  schema: 19
  provider: 0
```

Note: the raw run was completed before the failure classifier was corrected, so
17 `final_answer is required` cases were initially printed as `provider`. They
are schema/structured-output failures and will be classified as `schema` in the
next run.

Interpretation:

- The product fallback path did not break: every failed DeepSeek direct call was
  covered by deterministic fallback.
- The direct DeepSeek path is not yet good enough as the primary quality bar:
  47/100 direct success is too low.
- The biggest issue is latency under the current 8-second real HTTP timeout.
- The second issue is structured output quality, especially missing
  `final_answer` or child prompts that leak forbidden content.

## DeepSeek-First 100-Question Run After Tutor Structure Contract

The first Tutor Structure Contract optimization changed the DeepSeek
responsibility from full `ProblemAnalysis` generation to minimal
`LLMProblemParse` generation. Songguo backend now builds key points,
forbidden-content rules and child prompts.

Result with the same command and same timeout/retry settings:

```text
total: 100
deepseek_success: 97
fallback_success: 3
fallback_failures: 0
failure_counts:
  schema: 3
```

Compared with the previous run:

```text
deepseek_success: 47 -> 97
fallback_success: 53 -> 3
timeout: 34 -> 0
schema: 19 -> 3
```

Remaining failures were backend normalization/builder issues, not provider
timeouts:

- Statistics conditions may return `condition.value` as a list, for example
  `[12, 14, 11, 16, 17]`.
- Fraction-operation builder prompts may include a final-answer fraction inside
  an intermediate expression or model-provided step goal.

Both remaining issues were fixed after the 97/100 run:

- list-valued condition values are normalized to a string such as
  `12、14、11、16、17`;
- builder-generated prompts now re-check the final text and replace unsafe
  prompts with a generic step prompt if the expression or goal includes the
  final answer.

Targeted real DeepSeek retest passed for the failed cases:

```text
g_statistics_002 valid=True final=70页
g_fraction_ops_003 valid=True final=1/2
g_statistics_003 valid=True final=75页
```

Full rerun after those normalization fixes:

```text
total: 100
deepseek_success: 100
fallback_success: 0
fallback_failures: 0
failure_counts: {}
```

Second-layer model-call optimization then moved `LLMMathStructurer` from direct
HTTP to the LangChain ChatModel JSON path. Smoke run after the switch:

```text
total: 5
deepseek_success: 5
fallback_success: 0
fallback_failures: 0
failure_counts: {}
```

## DeepSeek-First Sample

Full 100-question DeepSeek evaluation was attempted first, but the command had
no progress output and was stopped after waiting more than 4 minutes. The
follow-up sample used a provider chain with a 15-second DeepSeek timeout and
deterministic fallback.

Result for the first 10 golden questions:

```text
total: 10
deepseek_success: 4
fallback_success: 6
schema_failures: 0
provider_failures: 0
```

The same 10-question provider-chain run through the standard evaluation runner:

```text
total: 10
schema_failures: 0
answer_leakage_count: 0
provider_failures: 0
skill_hit_count: 10
concept_card_hit_count: 10
controlled_generation_failures: 0
practice_generation_failures: 0
skill_hit_rate: 1.0
concept_card_hit_rate: 1.0
```

## DeepSeek Structured Prompt And Timeout Retry Fix

The 4/10 DeepSeek direct success rate was traced to two issues:

1. The real HTTP client still used a hard-coded 60-second request timeout and
   did not retry transient provider failures.
2. DeepSeek often returned useful JSON that was looser than the strict
   `ProblemAnalysis` schema, for example numeric `final_answer` values or an
   object-shaped `target`, so Pydantic validation rejected otherwise usable
   model output.

Changes made:

- `RealModelConfig` now carries `timeout_seconds`, `retry_attempts` and
  `retry_backoff_seconds`.
- `OpenAICompatibleModelClient` and `OpenAICompatibleChatModel` now share the
  same retrying OpenAI-compatible request path.
- `ProviderChain` can retry the same provider before falling back.
- The math structuring and session prompts now explicitly require a top-level
  JSON object, no Markdown/code fences, and stable defaults for missing fields.
- `LLMMathStructurer` now normalizes common DeepSeek loose JSON shapes before
  validating `ProblemAnalysis`.

Verification command:

```bash
PYTHONPATH=/Users/chen/code \
SONGGUO_REAL_MODEL_TIMEOUT_SECONDS=8 \
SONGGUO_REAL_MODEL_RETRY_ATTEMPTS=2 \
SONGGUO_REAL_MODEL_RETRY_BACKOFF_SECONDS=0.2 \
SONGGUO_AI_PROVIDER_TIMEOUT_SECONDS=20 \
SONGGUO_AI_PROVIDER_RETRY_ATTEMPTS=1 \
python -u - <<'PY'
from songguo.backend.evaluation.golden_math import build_golden_math_questions
from songguo.backend.services.learning.ai_engine import AIEngineContext, DeepSeekProvider, DeterministicFallbackProvider, ProviderChain
from songguo.backend.services.learning.math_structuring import validate_problem_analysis

chain = ProviderChain(
    [DeepSeekProvider(), DeterministicFallbackProvider()],
    provider_timeout_seconds=20,
    provider_retry_attempts=1,
    provider_retry_backoff_seconds=0,
)
deepseek_success = fallback_success = schema_failures = failures = 0
for question in build_golden_math_questions()[:10]:
    try:
        analysis = chain.structure_math_problem(
            question_text=question.question_text,
            grade=question.grade,
            context=AIEngineContext(child_id="eval_child", request_id=question.question_id),
        )
        verdict = validate_problem_analysis(analysis)
        if not verdict.valid:
            schema_failures += 1
        if chain.last_call.provider == "deepseek" and verdict.valid:
            deepseek_success += 1
        elif chain.last_call.provider == "deterministic_fallback" and verdict.valid:
            fallback_success += 1
    except Exception:
        failures += 1
print(
    f"deepseek_success={deepseek_success} fallback_success={fallback_success} "
    f"schema_failures={schema_failures} failures={failures}"
)
PY
```

Result for the same first 10 golden questions:

```text
deepseek_success: 10
fallback_success: 0
schema_failures: 0
failures: 0
```

## Fix Recorded

The previous weak behavior was:

```text
请再输入一道同类型数学题。
```

That placeholder appeared when generic arithmetic was categorized as
`read_conditions`, because `build_similar_practice_items(...)` had no original
question context and fell through to a placeholder seed.

Current behavior:

- `build_similar_practice_items(...)` accepts `source_question`.
- Pure arithmetic expressions generate concrete 1-3 similar practice items with
  answers.
- Common v0.1 math categories now have concrete template-backed practice items.
- Evaluation now records `practice_generation_failures`.
- Full graph evaluation now records `practice_quality_failure_count`.

## Remaining Follow-Up

The first 10-question DeepSeek sample is now healthy. The next evaluation step
is a progress-reporting 100-question DeepSeek-first run with timeout/retry
settings fixed at the command level, plus per-question failure categories so
slow provider calls and schema normalization misses are visible immediately.

Use the fixed progress runner for that full run:

```bash
PYTHONPATH=/Users/chen/code \
SONGGUO_REAL_MODEL_TIMEOUT_SECONDS=8 \
SONGGUO_REAL_MODEL_RETRY_ATTEMPTS=2 \
SONGGUO_REAL_MODEL_RETRY_BACKOFF_SECONDS=0.2 \
python -u /Users/chen/code/songguo/scripts/songguo_run_golden_math_eval.py \
  --limit 100 \
  --provider-timeout-seconds 20 \
  --provider-retry-attempts 1 \
  --json
```

The script prints one line before every question, one result line after every
question, and a final summary with:

- `deepseek_success`
- `fallback_success`
- `fallback_failures`
- `failure_counts`, grouped as `timeout`, `schema`, `provider`, or `unknown`
- `failed_case_ids`

## 2026-05-10 DeepSeek-First 100-Question Run

Command:

```bash
PYTHONPATH=/Users/chen/code \
SONGGUO_REAL_MODEL_TIMEOUT_SECONDS=8 \
SONGGUO_REAL_MODEL_RETRY_ATTEMPTS=2 \
SONGGUO_REAL_MODEL_RETRY_BACKOFF_SECONDS=0.2 \
python -u /Users/chen/code/songguo/scripts/songguo_run_golden_math_eval.py \
  --limit 100 \
  --provider-timeout-seconds 20 \
  --provider-retry-attempts 1 \
  --json
```

Runtime config printed by the script:

- `provider=deepseek`
- `model=deepseek-chat`
- `has_key=True`
- `real_timeout=8.0`
- `real_retries=2`

Final result:

```json
{
  "total": 100,
  "deepseek_success": 100,
  "fallback_success": 0,
  "fallback_failures": 0,
  "failure_counts": {},
  "failed_case_ids": []
}
```

Observed result:

- DeepSeek-first structure path reached `100/100`.
- No fallback was needed.
- No timeout, schema, provider, or unknown failure was reported by the runner.
- Most successful calls completed in roughly 2-5 seconds.
- Two slow but successful calls were observed: `g_average_002` at about 11.5s and `g_distractor_003` at about 12.1s.

Conclusion:

This run confirms that the first two layers plus the conservative fast path are
stable enough for a full golden-set DeepSeek-first pass. The next product check
should move back to the real mini program one-question flow, because this report
only proves structure-call reliability, not the full child-facing tutoring
experience, similar-practice quality, LearningDeposit content, or parent report
readability.
