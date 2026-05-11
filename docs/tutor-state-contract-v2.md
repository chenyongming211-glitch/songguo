# Tutor State Contract v2

本文档定义松果AI教学协议层第一期目标：让大模型自然教学、让后端执行可校验状态契约，并让安全拦截后的回复仍然贴合当前题。

## 目标

Tutor State Contract v2 不是 LangChain 或 LangGraph 自带能力，而是松果AI自己的产品协议。LangChain 负责模型调用和结构化输出，LangGraph 负责节点编排；松果负责定义教学状态、状态迁移、安全修复、学习沉淀和评测标准。

第一期优先解决两个问题：

- 大模型已经给出完成态时，后端不能覆盖成未完成态。
- 安全拦截不能直接返回通用固定话术，必须优先修复成贴合题目的追问。

## Contract 字段

每一轮模型输出最终要归一成一个 Tutor Turn Contract：

- `phase`: `WAIT_CHILD_ATTEMPT`、`LEARNING_PAUSED`、`SIMILAR_PRACTICE`、`PRACTICE_PAUSED`、`SESSION_SUMMARY`
- `learner_readiness`: `ready`、`needs_support`、`not_ready`、`safety_risk`
- `intent`: `math_attempt`、`confused`、`answer_seeking`、`learning_resistance`、`wellbeing_not_ready`、`off_task`、`safety_risk`、`unknown`
- `correctness`: `correct`、`partial`、`wrong`、`unknown`
- `next_action`: `continue_tutoring`、`simplify`、`refuse_direct_answer`、`pause_learning`、`ask_hint`、`ask_check`、`generate_practice`、`summarize`、`clarify`、`safety_response`
- `child_message`: 给孩子看的自然语言
- `learning_deposit_delta`: 本轮学习沉淀增量
- `practice_items`: 同类题
- `safety`: 安全处理结果

第一期代码中可以先保持兼容 `LLMSessionOutput`，新增 validator 和 safety 字段，不一次性迁移所有调用方。

## Learner Readiness Contract

这次“孩子说身体不舒服，松鼠博士仍继续讲题”的问题不能靠关键词补丁解决。Tutor State Contract 必须要求大模型每轮先做抽象判断：孩子当前是否适合继续学习。

判断优先级：

```text
儿童安全
-> 学习准备度
-> 数学/学科作答判断
-> 教学推进
```

学习准备度不是关键词列表，而是语义类别：

- `ready`: 孩子正在答题、确认、提问，或愿意继续。
- `needs_support`: 孩子卡住、不会、没思路，但没有表达要停止；松鼠博士可以降低难度继续启发。
- `not_ready`: 孩子表达当前身体、情绪、注意力或意愿状态不适合继续学习；松鼠博士应尊重暂停，不推进题目。
- `safety_risk`: 涉及儿童安全、隐私、暴力自伤、色情裸露、违法诱导、陌生人风险等，交给 Safety Contract。

对应意图：

- `math_attempt`: 孩子在尝试作答。
- `confused`: 孩子表示不会、看不懂、没思路，但仍处于可继续学习状态。
- `answer_seeking`: 孩子想直接要答案。
- `learning_resistance`: 孩子表达不想继续学、今天不练、明天再说。
- `wellbeing_not_ready`: 孩子表达身体、情绪、注意力状态不适合继续。
- `off_task`: 与当前学习无关但无安全风险。
- `safety_risk`: 儿童安全风险。

状态映射：

- `learner_readiness=ready` 且 `intent=math_attempt/confused/answer_seeking`: 进入正常教学策略。
- `learner_readiness=needs_support`: 降低难度、一次只问一个更小的问题。
- `learner_readiness=not_ready` 且 `intent=learning_resistance/wellbeing_not_ready`: `phase=LEARNING_PAUSED` 或同类题阶段 `phase=PRACTICE_PAUSED`，`teacher_move=pause_learning`。
- `learner_readiness=safety_risk`: 进入 Safety Contract。

一致性约束：

- 当 `teacher_move=pause_learning` 或 `phase=LEARNING_PAUSED/PRACTICE_PAUSED` 时，`child_message` 不能继续推进题目、不能要求孩子继续算、不能进入同类题。
- 当孩子只是 `confused` 且仍愿意学习时，不能直接暂停，应降低难度继续启发。
- 当孩子要答案时，不能直接给最终答案，应转成提示或核对问题。
- 后端不靠关键词判断“头疼”等具体表达；后端只检查模型输出是否自洽。若模型声明暂停却继续讲题，属于合同违规，应记录质量事件并优先模型修复。

## Safety Contract

Tutor Contract 不能把正常教学内容当成安全内容过滤器处理。正常的数字、条件复述、阶段性追问、模型自然表达，默认应该允许。Safety Contract 只处理儿童安全风险，例如隐私暴露、裸露/色情、暴力自伤、违法诱导、辱骂恐吓、陌生人风险等。

“是否直接给答案”属于 Teaching Policy 和教学质量问题，不属于 Safety Contract。它可以被记录成 `teaching_quality.direct_answer` 等评测事件，用于 prompt、结构化输出、黄金题集和回放优化，但不应该覆盖正常模型回复。

安全层输出必须说明来源：

- `action`: `allow`、`repair`、`contextual_fallback`、`hard_fallback`
- `reason`: `safe`、`child_privacy_risk`、`sexual_content`、`self_harm_or_violence`、`illegal_or_abuse_risk`
- `final_response_source`: `model`、`repaired_model`、`contextual_fallback`、`hard_fallback`
- `repair_attempted`: 是否尝试模型改写

处理边界：

- 内容安全风险、隐私风险、越权请求：硬拦截或安全兜底。
- 裸露/色情、暴力自伤、违法诱导、辱骂恐吓、陌生人风险：硬拦截或安全兜底。
- 普通条件数字、中间量追问、贴题引导：不因包含数字而拦截。
- 直接给答案、话术生硬、追问质量差：只进入 Teaching Policy、评测和回放，不进入 Safety hard block。

处理顺序：

1. 原始模型回复通过安全检查，直接返回。
2. 命中儿童安全风险时，返回安全兜底，并记录 `safety.blocked`。
3. 命中教学质量风险时，记录 `teaching_quality.*`，不覆盖 `child_message`。
4. 题目无法结构化时，仍优先让模型自然追问，不能用固定话术替代正常内容。

## 状态迁移原则

后端执行状态迁移，但不应随意覆盖模型结构化状态。

允许：

- `WAIT_CHILD_ATTEMPT -> WAIT_CHILD_ATTEMPT`
- `WAIT_CHILD_ATTEMPT -> LEARNING_PAUSED`
- `WAIT_CHILD_ATTEMPT -> SIMILAR_PRACTICE`
- `LEARNING_PAUSED -> WAIT_CHILD_ATTEMPT`
- `LEARNING_PAUSED -> SESSION_SUMMARY`
- `SIMILAR_PRACTICE -> SIMILAR_PRACTICE`
- `SIMILAR_PRACTICE -> PRACTICE_PAUSED`
- `PRACTICE_PAUSED -> SIMILAR_PRACTICE`
- `SIMILAR_PRACTICE -> SESSION_SUMMARY`

默认禁止：

- `SIMILAR_PRACTICE -> WAIT_CHILD_ATTEMPT`
- `SESSION_SUMMARY -> WAIT_CHILD_ATTEMPT`

除非孩子明确开启新题或重新开始。

## 模型教学协议

大模型负责松鼠博士的教学表达。后端不把正常回答改写成固定话术，而是通过 prompt、结构化输出和评测约束模型。

每次模型调用必须突出本轮任务：

- `latest_child_answer`: 孩子最新回答。
- `answer_unlocked`: 当前是否允许完整讲解或最终答案。
- `task`: `judge_latest_child_answer_then_ask_one_diagnostic_question`。
- `learner_readiness`: 先判断孩子当前是否适合继续学习。
- `teacher_move_priority`: 先判断学习准备度，再判断孩子回答是 `correct`、`partial`、`wrong` 还是 `unclear`，最后决定下一问。

模型提示词必须明确：

- 每轮先判断孩子是否适合继续学习；如果孩子表达当前身体、情绪、注意力或意愿状态不适合继续，应先尊重暂停，不继续追问题。
- 不能肯定错误答案。
- 孩子答错或不确定时，只问一个诊断问题，不要直接完整讲解。
- 每轮只推进一个认知动作。
- `answer_unlocked=false` 时不输出最终答案。
- 如果孩子的回答可能是错的，先让孩子解释来源或核对关系，不说“很棒”“算得很好”。

模型结构化输出包含教学自检：

- `teaching_intent.learner_readiness`: `ready | needs_support | not_ready | safety_risk`
- `teaching_intent.child_answer_status`: `correct | partial | wrong | unclear | math_attempt | confused | answer_seeking | learning_resistance | wellbeing_not_ready | off_task | safety_risk | unknown`
- `teaching_intent.teacher_move`: `ask_diagnostic_question | ask_next_step | simplify | refuse_direct_answer | pause_learning | summarize | generate_practice | safety_response`
- `teaching_intent.should_reveal_final_answer`
- `teaching_intent.next_question`
- `self_check.praised_wrong_answer`
- `self_check.revealed_final_answer`
- `self_check.one_question_only`
- `self_check.directly_solved_multiple_steps`

如果自检失败，后端记录 `teaching_quality.self_check_failed`，用于评测和 prompt 优化，但不覆盖 `child_message`。

## Tutor Structure Contract

100 题 DeepSeek-first 评测暴露的问题不是单纯 timeout，而是题目结构化契约过重：

- DeepSeek 同时承担题目理解、最终答案、解题步骤、错因、key_points、child_prompt 和防泄露规则。
- 结构化链路没有强制 JSON schema，模型会返回合理但不稳定的结构，例如 `final_answer: {"value": 28, "unit": "支"}`。
- `child_prompt` 和 `forbidden_content` 由模型生成，容易泄露答案或把规则写成自然语言。

第一层优化把题目结构化拆成两层：

```text
LLMProblemParse
  subject
  grade
  problem_type
  knowledge_points
  conditions
  target
  solution_steps
  final_answer
  common_misconceptions
  confidence
  source

ProblemAnalysisBuilder
  skill_ids
  misconception_ids
  concept_card_ids
  key_points
  forbidden_content
  child_prompt
```

规则：

- 大模型只输出 `LLMProblemParse`，不输出 `key_points`、`child_prompt`、`forbidden_content`。
- `final_answer` 必须归一成字符串；如果模型返回 `{value, unit}`，后端归一成 `"value+unit"`。
- `target` 必须归一成字符串；如果模型返回 `{description, value}`，后端取 `description`。
- key points、禁答内容、知识点资产和概念卡由后端 builder 生成。
- 后端 builder 生成的 `child_prompt` 不能包含 `final_answer`。
- 后续第二层再把模型调用从 direct HTTP 切到 LangChain JSON mode / structured output。

第二层优化已把结构化模型调用切到 LangChain OpenAI-compatible ChatModel，并要求 JSON mode / structured output。DeepSeek 仍负责理解题目、生成自然教学表达和处理泛化题型，后端只接收可校验结构。

第三层优化是 `MathFastPathRegistry`，但它不是固定题库，也不是写死某几道题的答案。它只做高频、低歧义、小学数学结构的参数化识别：

- 从当前题目文本提取数字、单位和关系词。
- 只在能稳定计算并能通过 `ProblemAnalysis` 校验时返回结构化结果。
- 识别不到、题型复杂、语义有歧义或结果不稳时返回 `None`，继续交给 DeepSeek。
- 生成的 `key_points`、`child_prompt` 和 `forbidden_content` 仍由后端 builder 统一生成，避免规则层输出固定教学话术。

第一批 fast path 覆盖范围：

- 纯算式：两位数乘一位数、低歧义四则表达式；非整除除法不本地兜底。
- 单位换算：米厘米、千克克、元角。
- 几何周长：长方形、正方形周长。
- 有余数平均分：从题目参数计算商和余数，单位从当前题目提取。
- 平均数：从当前题目分数列表计算平均分。

明确不做：

- 不把黄金题集里的具体题目写成特例。
- 不把“小松鼠”“苹果”等场景写成固定答案。
- 不用 fast path 代替大模型处理容量进一、分数、小数、百分数、比例、方程、统计、干扰条件等容易出现语义歧义的应用题。
- 不在后端把正常模型回答改写成固定话术。

当前 100 题黄金集静态覆盖检查：fast path 命中 28/100，类别包括 `calculation`、`multi_digit_division`、`unit_conversion`、`geometry`、`remainder_division`、`average`。这个比例是刻意保守的，目标是降低结构化失败和成本，而不是让规则层接管教学。

## 事件记录

每次 contract 执行至少记录关键事件：

- `contract.validated`
- `safety.blocked`
- `teaching_quality.direct_answer`
- `teaching_quality.repeated_prompt`
- `teaching_quality.low_context_fit`
- `transition.applied`
- `transition.rejected`

这些事件用于复盘、家长报告证据和 replay evaluation。

## 流式输出原则

松鼠博士的学生端回复应该有流式体验，但不能把原始大模型 token 直接流给孩子。

v0.1 采用安全后的分段流式：

1. 大模型完整生成结构化输出。
2. 后端完成 Tutor Contract 校验、状态迁移判断、安全检查、repair/contextual fallback。
3. 后端得到最终安全 `child_message`。
4. 小程序将最终安全文本做打字机式分段展示。

这样可以保留“现场老师正在说话”的体验，同时避免半截 token 在安全检查前泄露答案。

后续 v0.2 可以升级为 SSE 或 WebSocket，但仍必须流“已校验片段”，不能流原始模型 token。

## 第一期验收

- 首轮模型泄露答案时，不再出现“我们先不急着看答案。先找一个更简单的相关问题想一想。”这类通用话术。
- 排队位置题的安全兜底能贴合题目，引导孩子先转换方向位置。
- 学生端展示松鼠博士回复时采用安全后的分段流式，不流原始模型 token。
- 直接给答案、话术生硬、重复开场等问题不触发 `safety.blocked`，只记录教学质量事件。
- 隐私暴露、裸露/色情、暴力自伤、违法诱导、辱骂恐吓等儿童安全风险必须记录 `safety.blocked`。
- safety fallback 必须记录明确 reason/source。
- 后端全量测试通过。
