# 松果AI产品文档索引

本目录记录松果AI作为独立产品的产品规划文档，属于根目录 `songguo/` 产品工作区。

松果AI的定位是面向小学阶段的 AI 学习陪练产品层。v0.1 不重新做大模型，也不强依赖 DeepTutor；默认采用“松果AI业务后端 + LangGraph MathMistakeTutorGraph + 工具模型层 + 松果AI业务记忆层”的轻量路线。DeepTutor 在后续版本中作为可选 AI 学习引擎、工具 Provider 或教学资产来源使用，不作为账号、租户、权限、错题、学习画像、周报和计费的业务主系统。

## 文档

- `current-product-requirements.md`: 当前产品需求基准，作为后续开发和进度检查的优先对齐口径。
- `current-architecture-gap-fix-plan.md`: 当前架构不合理点与修复计划，记录 8 个需要按优先级整改的问题。
- `00-product-master-plan.md`: 松果AI PRD 产品总纲。
- `01-ai-teaching-engine-strategy.md`: 长期 AI 教学引擎方案，定义 Oppia 教学结构参考、DeepTutor AI 引擎和松果AI Teaching Kernel 的边界。
- `02-ai-teaching-short-term-plan.md`: AI 教学体系短期落地计划，定义 Runtime Kernel、AIEngineProvider、数学种子库和黄金评测集的近期任务。
- `03-ai-teaching-long-term-roadmap.md`: AI 教学体系长期路线图，定义从 v0.1 到 v1.0 的 Teaching Kernel、教研资产库、评测治理和商业化演进。
- `v0.1-mvp-prd.md`: 松果AI v0.1 MVP 核心闭环验证版 PRD，采用一个 v0.1 版本、PRD 基础里程碑为 M1/M2/M3，工程计划已扩展到 M8。
- `v0.1-technical-plan.md`: 松果AI v0.1 技术实施计划，按 M1-M8 拆分工程任务、文件范围和验证命令。
- `v0.1-lightweight-product-layer-plan.md`: v0.1 轻量产品层方案，定义“大模型负责当次生成，松果AI负责长期记录、错题、复习、画像、周报和数据隔离”的新主线。
- `v0.1-llm-session-runner-plan.md`: v0.1 一题一会话 LLM Runner 方案与开发计划，定义当前题完整 messages、长期历史摘要化注入、题后结构化沉淀的 M6 基础能力。
- `v0.1-langgraph-tutor-agent-plan.md`: v0.1 LangGraph Tutor Agent 方案与开发计划，定义 M7 主运行时 MathMistakeTutorGraph、后续 TutorGraph 流程库、护栏、降级和小程序体验调整。
- `v0.1-real-model-m8-plan.md`: v0.1 M8 真实模型试用闭环计划，定义 DeepSeek 接入、真实模型直连、小程序真实对话、黄金题集、家长反馈和成本日志。
- `v0.1-math-structured-teaching-gateway.md`: 小学数学题目结构化网关与关键点释放状态机方案，用于把 v0.1 从计算题演示闭环升级为自然语言小学数学题的受控教学闭环。
- `tutor-state-contract-v2.md`: Tutor State Contract v2，定义松果自有的教学状态协议、学习准备度判断、模型结构化意图、安全边界、状态迁移和流式输出原则。
- `v0.1-unified-learning-submission-plan.md`: 统一 Learning Submission 方案与开发计划，定义“开始做题”统一入口、1 到 N 道题提交、正确题和错题学习资产沉淀、错题队列逐题陪练、MasteryEvidence 和家长反馈升级。
- `v0.1-unified-learning-submission-development-plan.md`: 统一 Learning Submission 实施版开发计划，按 S0-S10 拆分 UI、Store、Intake、SubmissionGraph、API、小程序、掌握度、拍照语音和真实评测任务。
- `v0.2-auto-subject-routing-plan.md`: v0.2 自动学科识别与智能体路由整改方案，定义前端统一做题入口、后端自动识别数学/语文/英语并路由到对应 TutorGraph 的短中长期计划。
- `v0.2-product-architecture-optimization-plan.md`: v0.2 产品架构优化方案，定义真实手机私测安全、拍照可靠性、Graph 幂等、统一鉴权、可观测性、评测门禁和长期平台化演进。

## 后续文档规划

- `04-version-roadmap.md`: v0.1 到 v1.0 的完整产品版本路线图。
- `v0.2-family-review-outline.md`: 家庭多孩子、错题复习、学习画像增强概要。
- `v0.3-subject-expansion-outline.md`: 作文、阅读、英语等学科增强概要。
- `v0.4-institution-outline.md`: 老师端、机构租户和班级场景概要。
- `v1.0-commercial-platform-outline.md`: 商业化正式版平台概要。
