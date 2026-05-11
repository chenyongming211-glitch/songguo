# 松果AI当前架构不合理点与修复计划

本文档记录 2026-05-04 基于当前产品需求对代码和方案复盘后确认的 8 个不合理点。后续开发优先按本文档整改，避免继续扩展新概念。

2026-05-10 追加：当前 8 项中的多项基础整改已经完成，下一阶段产品架构优化详见 `v0.2-product-architecture-optimization-plan.md`。该方案将后续工作拆为短期真实手机私测安全、中期稳定性和可观测性、长期多学科学习平台化三个层级。

## 1. 数据库策略不适合真实试点

当前问题：

SQLite 适合本地开发和测试，但不适合作为真实小程序试点的主数据库。松果AI需要长期学习档案、错题、消息、AI 调用日志、家长报告、数据隔离和后续审计，SQLite 的并发、迁移、备份和运维能力不够。

修复口径：

- 本地开发和单元测试继续支持 SQLite / InMemory。
- 真实试点和部署切 PostgreSQL。
- 保持 Store 接口稳定，新增 PostgreSQL 实现。
- 引入可迁移 schema，后续使用 Alembic 或等价迁移工具管理表结构。

优先级：P1。

## 2. 现在不是真正的 LangChain + LangGraph 主链路

当前问题：

`MathMistakeTutorGraph` 当前是手写 Python 编排类，虽然运行时名称叫 `langgraph`，但没有真正使用 LangGraph `StateGraph`、node/edge、compiled graph，也没有把 DeepSeek 调用接入 LangChain ChatModel / structured output 主链路。

修复口径：

- `MathMistakeTutorGraph` 改为真正 LangGraph `StateGraph`。
- LLM 调用层接入 LangChain 风格 ChatModel 或兼容封装。
- 保留当前规则判题、答案泄露检查、LearningDeposit 落库逻辑。
- `kernel` 和 `llm` 只作为 fallback，不再伪装成主运行时。

优先级：P1。

## 3. 学习数据模型偏 JSON blob，不够产品级

当前问题：

当前 SQLite 表能保存对象，但大量业务对象以 JSON blob 形式存储。长期看不利于错因趋势、复习计划、家长报告、学习画像和审计查询。

修复口径：

- 保留 JSON 快速落库能力用于 MVP。
- PostgreSQL schema 中逐步拆清核心实体：children、learning_sessions、learning_messages、learning_events、wrong_questions、learning_deposits、learning_summaries、review_plan_items、ai_call_logs、safety_events。
- LearningDeposit 和 WrongQuestion 的核心字段必须可查询，不只存在 JSON 中。

优先级：P2。

## 4. Songguo 与 DeepTutor 边界仍不干净

当前问题：

DeepTutor 已降级为可选 AI Engine、工具或教学资产来源，但代码和小程序仍有 `deeptutor_*` storage key、`DEEPTUTOR_*` 环境变量、project name、debug WebSocket、trace 字段等残留。

修复口径：

- 小程序本地 storage 和配置命名切到 `songguo_*`。
- 后端新增 `SONGGUO_*` 环境变量，旧 `DEEPTUTOR_*` 仅保留兼容读取。
- DeepTutor 相关 provider 保留为可选适配层，不进入默认主链路。
- 前端 UI、README、项目配置不暴露 DeepTutor 品牌。

优先级：P1。

## 5. 小程序与后端 API 契约不完整

当前问题：

小程序设置页调用 `/api/v1/system/status`，但当前 Songguo 后端没有独立 FastAPI app 入口和 system router。小程序里也还残留旧 `/api/v1/sessions/*` 和 `/api/v1/ws` 调用。

修复口径：

- 新增 Songguo 独立 FastAPI app。
- 默认挂载 learning、parent、wechat、system；reminders、learning-artifacts 仅在实验开关开启时挂载。
- 新增 `/api/v1/system/status`。
- 小程序删除或隔离旧 DeepTutor session / WebSocket 主链路。

优先级：P0。

## 6. v0.1 功能边界有扩散风险

当前问题：

代码里已经出现 reminders、animation、English/Chinese copy、learning artifacts 等能力。它们可以保留，但不能抢占 v0.1 主线。当前最重要的是数学错题、真实 DeepSeek、小程序体验、家长反馈、长期档案和复习计划。

修复口径：

- v0.1 默认入口只突出数学错题陪练。
- 非主线能力保持隐藏、弱入口或实验状态。
- 新开发不再扩展英语、作文、老师端、机构 SaaS、支付、复杂多租户后台、动画视频。

优先级：P1。

## 7. 权限和数据隔离还只是雏形

当前问题：

当前已有 `X-Session-Token` 和 openid-child binding，但部分 list API 在无 token 时仍可返回全量数据。这适合本地开发，不适合真实试点。

修复口径：

- 真实环境下 child 维度 API 必须强制授权。
- 本地开发可通过显式 dev mode 放宽。
- 所有家长端、学习端、错题、报告、复习计划 API 都必须按 child_id 隔离。
- AI 调用日志和安全事件也必须按 child_id 查询授权。

优先级：P1。

## 8. 黄金评测还没有覆盖完整松鼠博士体验

当前问题：

当前黄金数学题集和 runner 已存在，但评测更偏向题目结构化、schema 和答案泄露检查，还没有系统评测完整 `MathMistakeTutorGraph` 体验。

修复口径：

- 评测完整 graph：创建会话、孩子多轮错误回答、追问质量、防答案泄露、答对后同类题、LearningDeposit、家长摘要。
- 指标至少包括：答案泄露数、schema 失败数、追问轮次、LearningDeposit 完成率、同类题生成率、家长摘要可读性。
- DeepSeek 真实模型评测和 deterministic fallback 评测分开运行。

优先级：P1。

## 修复顺序

1. P0：补独立 FastAPI app 和 `/api/v1/system/status`，修小程序契约缺口。
2. P1：清理小程序 `deeptutor_*` 命名和旧 session/ws 主链路。
3. P1：为真实试点引入 PostgreSQL 方案和依赖边界。
4. P1：把 `MathMistakeTutorGraph` 改为真正 LangGraph 主运行时。
5. P1：收紧真实环境下的数据隔离。
6. P1：扩展完整松鼠博士 graph 评测。
7. P2：重构核心学习数据 schema，减少 JSON blob 依赖。
8. P1/P2：冻结非主线入口，保留但不扩展实验能力。

## 当前整改进度

已修复：

- 第 1 项：已实现 `PostgresLearningStore`，并支持通过 `SONGGUO_DATABASE_URL` 将默认后端存储切到 PostgreSQL；已用真实 PostgreSQL 容器验证学习会话和 `LearningDeposit` 可以落库。
- 第 5 项：已新增 Songguo 独立 FastAPI app，并挂载 `/api/v1/system/status`；默认主 app 只暴露 v0.1 主链路 API。
- 第 4 项的一部分：小程序主链路已切到 `songguo_*` storage key，项目名改为 `songguo-mini-program`，旧 key 仅作为兼容读取。
- 第 4 项的一部分：小程序主链路已移除旧 `/api/v1/sessions/*`、`/api/v1/ws` 和 debug WebSocket 配置入口。
- 第 4 项的一部分：后端新增 `SONGGUO_SESSION_SECRET`、`SONGGUO_PHOTO_OCR_PROVIDER`、`SONGGUO_VISION_MODEL`、`SONGGUO_LLM_MODEL` 等主环境变量，旧 `DEEPTUTOR_*` 仅作为兼容 fallback。
- 第 1 项：已补 `requirements-postgres.txt`，明确真实试点 PostgreSQL 依赖边界。当前 PostgreSQL 第一版沿用现有 Store 接口和 JSON payload，核心字段保留关系索引。
- 第 2 项：`MathMistakeTutorGraph` 已改为真正 LangGraph `StateGraph`，start 和 submit 路径均通过 compiled graph 执行。模型调用层已从直接 HTTP 默认路径升级为 LangChain ChatModel 适配器，并在 prompt 中注入 `LLMSessionOutput` JSON schema 做结构化输出约束。
- 第 2 项的一部分：已补 `requirements-langgraph.txt`，明确真实 LangGraph 主链路依赖边界。
- 第 3 项：`wrong_questions` 和 `learning_deposits` 已拆出 `knowledge_point`、错因、复习需求、练习完成状态等核心可查询列，SQLite 和 PostgreSQL 初始化/迁移都覆盖这些字段，JSON payload 仅作为完整对象快照保留。
- 第 6 项：默认后端不再挂载 reminders、learning-artifacts；小程序家长页的提醒、图解、动画入口已挂在 `songguo_experimental_features` 开关后，v0.1 默认体验回到数学错题闭环。
- 第 7 项：已新增 `SONGGUO_AUTH_MODE=strict`，严格模式下 child 维度 API 必须带 `X-Session-Token`，无 token 返回 401，跨 child 返回 403；reminders 和 learning-artifacts 在实验开启时也执行 child 授权。
- 第 8 项：已新增完整 `MathMistakeTutorGraph` 评测 runner，覆盖创建会话、多轮错误回答、防答案泄露、答对后同类题、`LearningDeposit`、家长反馈和模型 fallback 统计。

待继续修复：

- 第 1 项后续优化：补正式迁移工具或 schema 版本表，替代当前 Store 初始化内置迁移。
- 第 2 项后续优化：补更细的 token usage 统计、LangChain tracing 和模型级重试策略。
- 第 3 项后续优化：继续拆 `review_plan_items`、`parent_report_items` 等二级查询表；当前核心错题和学习沉淀已先完成可查询化。
- 第 4 项后续优化：可选 provider 内部仍保留 `deeptutor` 兼容命名，用于后续作为 AI Engine 或教学资产来源；默认产品链路不暴露该品牌。
- 第 7 项后续优化：未来新增后台 API 时必须补严格模式边界测试。
