# 松果AI AI 教学体系长期路线图

## 1. 路线图定位

本文档定义松果AI AI 教学体系从 v0.1 到 v1.0 的长期演进。

长期方向是：

```text
Oppia 的教学结构思想
+ DeepTutor / LLM 的 AI 引擎能力
+ 松果AI自己的 Teaching Kernel
+ 学生长期画像和家长反馈闭环
```

长期目标不是做一个普通 AI 问答工具，而是形成一个可持续运营、可评测、可扩展的专属 AI 教师系统。

## 2. 长期核心资产

松果AI长期应沉淀四类核心资产。

### 2.1 Runtime Kernel

运行时教学内核。

负责：

```text
教学状态
关键点推进
答案解锁
错因记录
学习事件
画像增量
复习触发
```

### 2.2 Teaching Asset Library

教研资产库。

负责：

```text
Subject
Grade
Topic
Skill
Misconception
ConceptCard
TeachingPath
FeedbackTemplate
PracticeTemplate
ParentTipTemplate
```

### 2.3 AI Engine Layer

AI 引擎层。

负责：

```text
DeepTutorProvider
DeepSeekProvider
OpenAIProvider
OllamaProvider
OCRProvider
RAGProvider
QuizProvider
DiagramProvider
```

DeepTutor 是其中一个重要 Provider，不是完整业务系统。

### 2.4 Evaluation & Governance

评测和治理体系。

负责：

```text
黄金评测集
Prompt 版本管理
教学资产版本管理
结构化质量评测
答案泄露评测
儿童话术评测
成本监控
版本回滚
```

## 3. 版本路线

### V0.1：数学错题闭环验证

目标：

```text
验证“输入/拍题 -> 结构化 -> 动态引导 -> 错因记录 -> 同类题 -> 家长反馈”是否成立。
```

重点能力：

```text
Runtime Kernel 最小闭环
MathProblemStructuringGateway
KeyPoint State Machine
Answer Leakage Checker
LearningEvent
WrongQuestion
SessionFeedback
AIEngineProvider 初版
数学 Skill / Misconception 种子库
100 道黄金评测集
```

不做：

```text
完整教研后台
完整多学科资产库
完整 Oppia 式 Exploration
完整机构 SaaS
复杂支付和计费
```

成功标准：

```text
1. 数学题能按关键点动态引导。
2. 首次提示不直接泄露最终答案。
3. 错因能进入错题和家长反馈。
4. 结构化质量有评测集守住。
5. DeepTutor 可替换为其他 Provider。
```

### V0.2：家庭复习和学习画像增强

目标：

```text
让松鼠博士开始真正“记住这个孩子”。
```

重点能力：

```text
家庭多孩子管理
学生级学习画像
错题复习计划
掌握度记录
今日复习
家长周报升级
Skill / Misconception 统计
ConceptCard 补救提示
```

Teaching Asset Library 进入轻量使用阶段：

```text
Skill Seed -> Skill Library v0.2
Misconception Seed -> Misconception Library v0.2
ConceptCard 极简版 -> ConceptCard 可复用版
```

成功标准：

```text
1. 每个孩子有独立画像。
2. 家长能看到高频 Skill 和 Misconception。
3. 错题能自动进入复习计划。
4. AI 提示能根据孩子历史错因调整。
```

### V0.3：语文阅读和作文表达扩展

目标：

```text
把 Teaching Kernel 从数学扩展到语文阅读和作文表达。
```

重点能力：

```text
阅读 Skill / Misconception
作文表达 Skill / Misconception
阅读 ConceptCard
作文表达 ConceptCard
不代写作文安全策略
阅读完整回答训练
作文细节补充训练
```

需要新增评测：

```text
阅读理解回答完整性评测
作文代写风险评测
语文年级适配评测
```

成功标准：

```text
1. AI 不直接代写作文。
2. 阅读陪练能引导孩子完整回答。
3. 语文错因能进入画像和家长反馈。
4. 数学和语文共用同一套 Runtime Kernel 思路。
```

### V0.4：英语和多模态增强

目标：

```text
扩展英语句型、单词、跟读和多模态输入能力。
```

重点能力：

```text
英语句型 Skill
单词拼写 Misconception
自然拼读基础画像
ASR / TTS Provider
英语跟读评分 Provider
图片 OCR 质量评测
语音隐私和儿童安全策略
```

成功标准：

```text
1. 英语陪练不只是翻译，而是句型和表达训练。
2. 语音数据有最小化和授权边界。
3. 英语错因能进入学生画像。
```

### V0.5：教研资产后台

目标：

```text
让教学资产可以被维护、审核、版本化和回滚。
```

重点能力：

```text
Skill 管理
Misconception 管理
ConceptCard 管理
TeachingPath 管理
FeedbackTemplate 管理
PracticeTemplate 管理
Prompt 版本管理
审核流
灰度发布
回滚机制
```

成功标准：

```text
1. 教研人员能维护知识点、错因和概念卡。
2. 每次教学策略更新都有版本。
3. 出现错误策略时能回滚。
4. 评测集能作为发布前门禁。
```

### V0.6：机构和老师场景

目标：

```text
把家庭场景沉淀的能力扩展到老师、班级和机构。
```

重点能力：

```text
老师角色
班级管理
班级 Skill 统计
班级 Misconception 统计
布置练习
租户知识库
机构后台
班级报告
```

成功标准：

```text
1. 老师看到的是班级共性卡点，不是单纯聊天记录。
2. 机构能维护自己的资料和练习。
3. 学生数据仍然按 tenant_id 和 student_id 隔离。
```

### V1.0：商业化正式平台

目标：

```text
形成家庭订阅 + 机构 SaaS 的完整商业闭环。
```

重点能力：

```text
完整多租户权限
家庭会员
机构套餐
额度和成本控制
学习画像成熟版
家长周报成熟版
教研资产库成熟版
评测治理成熟版
内容安全闭环
运营后台
```

成功标准：

```text
1. 一个孩子有稳定专属 AI 老师体验。
2. 一个家庭可管理多个孩子。
3. 一个机构可管理多个班级。
4. AI 成本可控。
5. 教学质量可评测。
6. 教学策略可版本化和回滚。
```

## 4. 长期数据模型演进

### 4.1 V0.1 必须有

```text
LearningSession
LearningEvent
WrongQuestion
TeachingProgress
ProblemAnalysis
KeyPoint
AttemptEvaluation
SessionFeedback
```

### 4.2 V0.2 开始引入

```text
Skill
Misconception
ConceptCard
StudentProfile
MasteryRecord
ReviewPlan
ReviewTask
WeeklyReport
```

### 4.3 V0.5 开始完善

```text
TeachingPath
FeedbackTemplate
PracticeTemplate
PromptVersion
AssetVersion
EvaluationRun
EvaluationCase
ReleaseAudit
```

### 4.4 V1.0 平台化

```text
Tenant
TenantKnowledgeBase
Class
TeacherStudentBinding
BillingPlan
UsageQuota
CostRecord
AuditLog
```

## 5. 长期技术原则

```text
1. 小程序永远不直接调用 DeepTutor。
2. DeepTutor 永远只是 AI Engine Provider，不是业务主系统。
3. Oppia 永远只是教学结构参考，不作为运行时依赖。
4. Teaching Kernel 必须可评测、可回滚、可版本化。
5. 学生学习数据必须由松果AI主库维护。
6. AI 生成内容必须经过结构化、校验、泄露检查和状态机。
7. 教学资产必须支持人工审核，不能完全由模型自动入库。
8. 任何新学科扩展前必须先补 Skill、Misconception、ConceptCard 和评测集。
```

## 6. 长期风险和应对

| 风险 | 表现 | 应对 |
| --- | --- | --- |
| Teaching Kernel 过大 | 模块边界混乱，开发慢 | 拆成 Runtime Kernel、Teaching Asset Library、Evaluation & Governance |
| 教研资产质量差 | 错因标签混乱，画像不可信 | 建立命名规范、审核流、版本管理 |
| 大模型结构化不稳定 | 题型、关键点、答案错误 | 黄金评测集 + Schema 校验 + 回归测试 |
| 答案泄露 | 孩子直接看到答案 | 输出后检查 + 状态机拦截 + 评测门禁 |
| DeepTutor 耦合 | 后续替换困难 | AIEngineProvider 抽象 |
| 成本失控 | 每题多次调用强模型 | 模型路由、缓存、模板化、异步生成 |
| 多学科扩展失控 | 语文英语各做一套逻辑 | 统一 Runtime Kernel，学科差异放入 Asset Library |
| 教研后台过早复杂化 | MVP 变慢 | v0.1/v0.2 只做种子库，后台放到 v0.5 |

## 7. 长期完成定义

松果AI长期成功不是功能数量多，而是形成以下闭环：

```text
孩子学习
-> AI 动态引导
-> 错因归因
-> Skill / Misconception 沉淀
-> 复习计划
-> 学生画像
-> 家长周报
-> 教研资产优化
-> 评测集回归
-> 策略版本迭代
```

最终产品能力应表现为：

```text
每个孩子都有专属 AI 老师。
AI 老师知道孩子在哪里卡住。
每次提示都受控，不直接变成答案机器。
家长看得懂孩子为什么不会。
教研资产越用越完善。
教学质量可以被评测，而不是只靠感觉。
```

