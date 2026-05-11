# 松果AI长期方案：Oppia 教学结构 + DeepTutor AI 引擎 + 松果AI Teaching Kernel

## 1. 方案结论

松果AI不应该在 Oppia 和 DeepTutor 之间二选一。

长期最优方案是：

```text
Oppia 提供教学结构参考
DeepTutor 提供 AI 引擎能力
松果AI自研 Teaching Kernel
```

也就是说：

```text
Oppia 教我们如何把教学过程结构化。
DeepTutor 帮我们调用大模型完成理解、生成、检索和工具调用。
松果AI自己掌控孩子学习节奏、教学状态、错因、画像和家长反馈。
```

松果AI的核心壁垒不应该是 DeepTutor 本身，也不应该是 Oppia 本身，而应该是：

```text
松果AI Teaching Kernel
+ 学生长期学习画像
+ 小学知识点和错因图谱
+ 家长可执行反馈闭环
```

## 2. Oppia 什么地方好

Oppia 的核心优势不是生成式 AI，而是教学内容和互动学习结构。

Oppia 官方将核心学习单元称为 `exploration`，从学习者视角看，它像学生和导师之间的一段对话。它的基础结构包括：

```text
Topic
Story
Skill
Exploration
Card
Content
Interaction
Response
Misconception
Concept Card
```

这些设计对松果AI有长期参考价值。

### 2.1 Exploration：把学习设计成探索过程

Oppia 的 `exploration` 不是单向课程，而是一段预先设计好的互动学习路径。

这和松果AI“不做答案机器，做学习陪练”的方向一致。

松果AI可以借鉴这个思想，把一道题、一类错因、一个知识点补救过程都设计成可追踪的教学探索。

### 2.2 Card / Interaction / Response：每一步都有反馈

Oppia 的一节 exploration 由多个 card 组成。每个 card 包含：

```text
老师提出内容
学生进行交互
系统根据学生输入给 response
```

这正好对应松果AI的关键点释放状态机：

```text
当前关键点
-> 孩子尝试
-> 系统判断掌握情况
-> 给下一步提示或补救反馈
```

### 2.3 Skill + Misconception：把错因变成教研资产

Oppia 不是只维护题目，而是维护 `skill` 和 `misconception`。

松果AI应该重点借鉴这一点。

未来松果AI不应该只记录：

```text
这道题错了
```

而应该记录：

```text
这个孩子在哪个 skill 上卡住
对应的 misconception 是什么
需要哪个 concept card 补救
应该安排哪类同类题复习
```

### 2.4 Concept Card：孩子卡住时回到概念

Oppia 的 Concept Card 用来帮助学习者理解某个 skill 的概念、方法和例子。

这对松果AI很重要。

当孩子答错时，松果AI不应该只继续追问，也不应该直接讲答案，而应该在合适阶段回到：

```text
概念卡
方法卡
易错提醒
一个更简单的例题
```

### 2.5 教研内容可运营

Oppia 的长期价值来自结构化内容运营。

松果AI也需要把“提示、错因、概念卡、同类题、家长建议”沉淀成可维护资产，而不是每次都让大模型自由发挥。

## 3. DeepTutor 什么地方好

DeepTutor 的优势是 AI 工程底座。

它适合承担：

```text
题目理解
Deep Solve
RAG 检索
Quiz 生成
TutorBot
Memory
多模型 Provider 适配
工具调用
多模态能力
```

### 3.1 多模型和本地模型适配

DeepTutor 已经有多模型 Provider 和本地模型接入能力。

这对松果AI控制成本、支持 DeepSeek、Ollama、本地模型和后续模型路由有价值。

### 3.2 RAG 和知识库

松果AI后续需要接入：

```text
教材
教辅
老师讲义
机构资料
孩子私有错题
作文和阅读记录
```

DeepTutor 的 RAG 能力可以作为底层引擎，但知识库权限必须由松果AI控制。

### 3.3 Deep Solve 和 Quiz

DeepTutor 的 Deep Solve 适合复杂题推理，Quiz 能力适合同类题生成。

但它生成的内容只能作为候选结果，不能直接成为孩子端最终输出。

### 3.4 Memory 和 TutorBot

DeepTutor 的 Memory 和 TutorBot 可以辅助形成“专属 AI 老师”的体验。

但学生学习画像、错题、复习计划和家长周报必须由松果AI主库维护，不能完全依赖 DeepTutor 内部 memory。

### 3.5 Agent / Tool 架构

DeepTutor 的 Tools + Capabilities 架构适合后续扩展：

```text
OCR
ASR / TTS
数学图解
动画
作文批改
阅读陪练
英语跟读
知识库检索
```

## 4. 松果AI Teaching Kernel

松果AI必须自研 Teaching Kernel。

Teaching Kernel 是松果AI的核心产品和技术资产，不是 DeepTutor 或 Oppia 的替代品。

它负责：

```text
知识点体系
错因体系
题目结构化 Schema
关键点释放状态机
答案泄露检查
学生画像更新
复习计划
家长反馈生成
教学策略版本管理
```

但 Teaching Kernel 不能在第一阶段做成一个过大的抽象层。

为了便于落地，必须拆成两个边界清晰的部分：

```text
Runtime Kernel
Teaching Asset Library
```

Runtime Kernel 负责运行时教学控制：

```text
会话状态
关键点推进
答案解锁
错因记录
学习事件
画像增量
```

Teaching Asset Library 负责教研资产：

```text
Skill
Misconception
ConceptCard
TeachingPath
FeedbackTemplate
PracticeTemplate
```

第一阶段优先做 Runtime Kernel，Teaching Asset Library 只做种子库，不做完整后台。

### 4.1 Teaching Kernel 的职责

Teaching Kernel 决定：

```text
当前孩子应该看到哪一个关键点
什么时候允许完整讲解
什么时候进入同类练习
这次错误属于什么错因
这次学习如何更新画像
家长应该看到什么建议
```

Teaching Kernel 不直接承担：

```text
大模型推理
RAG 检索
OCR 识别
语音识别
动画生成
```

这些交给 AI Engine Adapter 和底层引擎处理。

### 4.2 Teaching Kernel 的核心数据模型

建议长期沉淀以下模型：

```text
Subject 学科
Grade 年级
Topic 单元主题
Skill 能力点
ConceptCard 概念卡
Misconception 错因
TeachingPath 教学路径
KeyPoint 关键点
Interaction 互动
FeedbackTemplate 反馈模板
PracticeTemplate 同类题模板
MasteryRecord 掌握记录
ReviewPlan 复习计划
```

这些模型借鉴 Oppia，但必须服务松果AI自己的产品闭环。

### 4.3 例子：大巴限载应用题

题目：

```text
学校组织三年级学生春游，一共有4个班，每班32人。如果每辆大巴车限坐45人，那么至少需要多少辆大巴车？
```

Teaching Kernel 应该把它归到：

```text
Topic：乘除法应用题
Skill：乘法求总数、按容量分组、有余数进一
Misconception：只算总人数、不理解“至少”、忘记余数进一
ConceptCard：什么叫“至少需要几辆车”
TeachingPath：
  1. 先算总人数
  2. 找每辆车最多坐多少人
  3. 用除法分组
  4. 有余数时多一辆
```

DeepTutor 或 LLM 可以帮助输出这个结构，但最终是否采用、是否释放给孩子，由松果AI校验和状态机决定。

## 5. 方案不足与优化原则

当前长期方案方向成立，但不能直接作为开发计划落地。主要不足如下。

### 5.1 Teaching Kernel 容易过大

如果把知识点、错因、画像、复习、家长报告、内容后台和模型调用都塞进 Teaching Kernel，它会变成一个边界模糊的大模块。

优化原则：

```text
Runtime Kernel 只管运行时教学决策。
Teaching Asset Library 只管教研资产。
AI Engine Layer 只管模型和工具调用。
Evaluation & Governance 只管质量、成本和版本治理。
```

### 5.2 Oppia 不能直接照搬

Oppia 更适合预设课程和 exploration，而松果AI的高频场景是孩子临时拍题或输入题目。

如果直接照搬 Oppia 的课程流，会让每道题都像走课件，体验变慢。

优化原则：

```text
借鉴 Oppia 的 Skill / Misconception / ConceptCard / Response 思想。
不在 v0.1 引入完整 Exploration 运行时。
不要求每道题都进入课程化路径。
```

### 5.3 大模型结构化质量必须评测

大模型负责输出 ProblemAnalysis、KeyPoint、Misconception，但这些输出可能错分知识点、漏掉关键点、算错答案或提前泄露答案。

优化原则：

```text
必须建立黄金评测集。
必须评测题目结构化准确率。
必须评测关键点覆盖率。
必须评测答案泄露率。
必须评测错因识别一致性。
```

没有评测集，不应扩大题型覆盖。

### 5.4 DeepTutor 不能被绑定死

DeepTutor 是当前 AI Engine 的核心实现之一，但松果AI不能在业务代码里直接依赖 DeepTutor 内部接口。

优化原则：

```text
抽象 AIEngineProvider。
DeepTutorProvider 只是其中一个 Provider。
未来可以并联 DeepSeekProvider、OpenAIProvider、OllamaProvider、自研结构化模型。
```

### 5.5 教研资产库建设成本不能低估

Skill、Misconception、ConceptCard 不会自动长出来。

它需要：

```text
人工定义框架
AI 辅助生成候选
教研人员校验
真实错题数据反向修正
版本化管理
```

优化原则：

```text
第一版只做种子库。
先覆盖高频小学数学。
不追求完整知识图谱。
```

### 5.6 儿童安全不只是答案泄露

长期还需要检查：

```text
答案泄露
提示是否超龄
语气是否打击孩子
是否诱导隐私
是否代写作业
是否给出危险内容
```

优化原则：

```text
答案泄露检查是 P0。
儿童话术检查、隐私检查、代写检查进入 P1。
作文、阅读、英语扩展前必须补安全策略。
```

### 5.7 成本和延迟必须分层

如果每道题都调用强模型、DeepTutor、RAG 和多轮校验，会导致成本高、响应慢。

优化原则：

```text
简单题走规则或小模型。
常见题走模板和缓存。
复杂题走强模型和 DeepTutor。
周报、复习计划异步生成。
```

## 6. 优化后的长期架构

优化后的长期架构拆成四个核心模块。

```text
微信小程序
  ↓
松果AI业务后端
  ↓
Songguo Runtime Kernel
  ↓
Teaching Asset Library
  ↓
AI Engine Adapter
  ↓
DeepTutor / LLM / RAG / OCR / ASR / TTS
  ↓
Evaluation & Governance
```

### 6.1 各层职责

微信小程序：

```text
学生端学习交互
家长端报告查看
拍照、输入、语音、练习展示
```

松果AI业务后端：

```text
账号
多租户
学生绑定
权限
计费
审计
业务数据主库
```

Songguo Teaching Kernel：

```text
教学状态
关键点释放
错因分析
画像更新
复习计划
家长反馈
教学策略版本
```

其中应进一步拆分：

```text
Runtime Kernel：
  会话状态、关键点推进、答案解锁、事件落库、画像增量。

Teaching Asset Library：
  Skill、Misconception、ConceptCard、TeachingPath、模板和版本。
```

AI Engine Adapter：

```text
统一调用 DeepTutor、外部 LLM、本地模型、OCR、RAG、ASR、TTS
结构化输入输出
错误降级
调用日志
成本统计
```

DeepTutor / LLM 层：

```text
题目理解
候选关键点生成
候选提示生成
复杂题推理
同类题生成
RAG 检索
工具调用
```

Evaluation & Governance：

```text
结构化评测集
答案泄露评测
教学话术评测
错因一致性评测
成本监控
版本回滚
```

## 7. 运行时边界

### 7.1 DeepTutor 不能直接控制教学状态

DeepTutor 可以生成：

```text
ProblemAnalysis
KeyPoint
SolutionStep
候选提示
候选讲解
同类题
```

DeepTutor 不能决定：

```text
是否进入下一关键点
是否解锁最终答案
是否记录错题
是否更新画像
家长报告如何归因
学生是否真正掌握
```

### 7.2 Oppia 不作为运行时依赖

Oppia 不建议直接接入松果AI运行时。

原因：

```text
1. Oppia 是完整课程平台，集成成本重。
2. Oppia 的核心不是生成式 AI 引擎。
3. 松果AI需要微信小程序、家长端、多租户、学生画像和 AI 调用成本控制。
4. 直接引入 Oppia 会让架构变重，反而弱化松果AI自己的 Teaching Kernel。
```

因此，Oppia 的角色是：

```text
教学结构参考
教研模型参考
内容后台设计参考
```

不是：

```text
松果AI运行时引擎
松果AI业务后端
松果AI小程序后端
```

## 8. Skill / Misconception 种子库建设方法

松果AI需要拥有自己的 Skill 和 Misconception 体系，但不需要完全手工从零整理。

推荐方法：

```text
人工定义骨架
+ AI 辅助扩展
+ 教研/家长/真实错题校验
+ 使用数据持续修正
```

### 8.1 第一版规模

第一版建议先做：

```text
小学数学 Skill Seed Library：约 30 个
小学数学 Misconception Seed Library：约 50 个
ConceptCard：每个高频 Skill 配一个极简版
```

这足够支撑 v0.1/v0.2 的数学错题闭环，不需要一开始覆盖完整小学数学知识图谱。

### 8.2 Skill 示例

第一版高频小学数学 Skill 可以包括：

```text
两位数乘一位数
三位数乘一位数
有余数除法
乘法求总数
平均分问题
限载进一问题
倍数关系
和差问题
归一问题
单位换算
时间计算
人民币计算
周长计算
面积计算
分数初步
小数初步
四则混合运算
读题找条件
应用题列式
检查和估算
```

### 8.3 Misconception 示例

以“限载进一问题”为例：

```text
只算总人数，不知道要分车
忘记除以每辆车人数
有余数但没有多加一辆
把“最多坐45人”理解成“一共45人”
不理解“至少需要”的含义
```

### 8.4 入库原则

AI 可以生成候选 Skill 和 Misconception，但不能直接入库。

入库前必须做：

```text
统一命名
去重合并
年级适配检查
是否可观测
是否能驱动教学反馈
是否能进入家长报告
```

### 8.5 真实数据修正

使用中如果孩子频繁出现新错因，应进入候选池。

候选池经过人工确认后再进入正式 Misconception 库。

## 9. 评测与治理

为了让方案可落地，必须建立 Evaluation & Governance。

### 9.1 黄金评测集

第一版至少准备 100 道小学数学题，覆盖：

```text
计算题
自然语言应用题
单位换算
时间和人民币
图形周长面积
有余数除法
限载进一
分数小数初步
四则混合运算
带干扰条件题目
```

每道题人工标注：

```text
年级
题型
Skill
Misconception 候选
条件
目标
解题步骤
关键点
最终答案
禁止泄露内容
```

### 9.2 核心评测指标

```text
题目结构化准确率
关键点覆盖率
最终答案正确率
答案泄露率
错因识别一致性
提示年级适配率
同类题可用率
平均响应时延
单题模型成本
```

### 9.3 版本治理

所有教学资产和提示策略必须版本化。

```text
Skill version
Misconception version
ConceptCard version
Prompt version
StateMachine version
AIEngineProvider version
```

一旦发现某个版本导致错误提示或答案泄露，必须能够回滚。

## 10. 与既有方案对比

### 10.1 与产品总纲的关系

既有产品总纲已经明确：

```text
DeepTutor 是 AI 学习引擎，不是完整业务后端。
松果AI数据库是业务主库。
学生画像、错题、复习计划、周报由松果AI维护。
```

本方案不推翻这些结论。

本方案需要补充的是：

```text
松果AI需要明确提出 Teaching Kernel。
Oppia 作为教学结构参考进入长期设计。
DeepTutor 不再被表述为唯一“教学引擎”，而应表述为 AI Engine 的核心实现之一。
```

### 10.2 与 v0.1 MVP PRD 的关系

v0.1 MVP 的目标仍然不变：

```text
拍照/输入错题
-> AI分步引导
-> 错因记录
-> 1-3道同类练习
-> 本次学习反馈
```

本方案不扩大 v0.1 范围。

v0.1 只需要继续做：

```text
MathProblemStructuringGateway
KeyPoint State Machine
答案泄露检查
学习事件记录
家长反馈草稿
```

不应该在 v0.1 引入完整 Oppia 式课程平台。

### 10.3 与 v0.1 技术计划的关系

既有技术计划已经写明：

```text
大模型负责题目结构化和关键点提取。
规则层负责校验。
状态机负责关键点释放。
```

这与本方案完全一致。

需要更新的是术语：

```text
把这套“结构化网关 + 状态机 + 泄露检查 + 画像更新”的组合，正式命名为 Songguo Teaching Kernel。
```

### 10.4 与小学数学结构化网关方案的关系

小学数学结构化网关是 Teaching Kernel 的第一个落地模块。

它对应：

```text
ProblemAnalysis
SolutionStep
KeyPoint
AttemptEvaluation
答案泄露校验
关键点释放
```

后续语文、英语、阅读、作文都应该沿用同一套 Teaching Kernel 思路，而不是各做一套散乱逻辑。

## 11. 需要更新的地方

### 11.1 产品总纲需要更新

需要在产品总纲中补充：

```text
松果AI核心技术资产是 Teaching Kernel。
DeepTutor 是 AI Engine，不是完整教学内核。
Oppia 是教学结构和教研资产模型参考。
```

### 11.2 文档索引需要更新

需要把本文档加入 `songguo/docs/README.md`。

### 11.3 后续技术计划需要更新

后续技术文档中应统一使用：

```text
Songguo Teaching Kernel
AI Engine Adapter
DeepTutor Provider
Teaching Asset Library
```

避免继续把所有 AI 和教学控制能力都笼统叫做 DeepTutor。

### 11.4 数据模型需要新增长期规划

后续 `v0.2` 或 `v0.3` 应新增：

```text
Skill
ConceptCard
Misconception
TeachingPath
FeedbackTemplate
PracticeTemplate
```

v0.1 不强制建完整表，但接口和事件模型应预留这些字段。

## 12. 落地路线

短期执行计划见：

```text
songguo/docs/02-ai-teaching-short-term-plan.md
```

长期路线图见：

```text
songguo/docs/03-ai-teaching-long-term-roadmap.md
```

### 阶段一：继续做稳 v0.1 数学闭环

```text
MathProblemStructuringGateway
KeyPoint State Machine
Answer Leakage Checker
WrongQuestion
LearningEvent
SessionFeedback
```

同时补齐：

```text
AIEngineProvider 抽象
100 道数学黄金评测集
30 个 Skill 种子库
50 个 Misconception 种子库
ConceptCard 极简模板
```

### 阶段二：建立小学数学 Teaching Asset Library

先做小学数学：

```text
Topic
Skill
Misconception
ConceptCard
TeachingPath
FeedbackTemplate
PracticeTemplate
```

### 阶段三：让大模型绑定教学资产

大模型不再自由回答，而是基于：

```text
当前 Skill
当前 ConceptCard
当前 Misconception
当前 TeachingPath
当前 KeyPoint
当前 StudentProfile
```

生成受控提示。

### 阶段四：建设教研后台

让教研人员可以维护：

```text
知识点
错因
概念卡
提示模板
同类题模板
家长建议模板
```

### 阶段五：形成真正的专属 AI 老师

每个孩子的 TutorInstance 等于：

```text
基础人格：松鼠博士
+ 学生画像
+ 错因历史
+ Teaching Asset Library
+ DeepTutor / LLM 生成能力
+ 家长设置
+ 安全策略
```

## 13. 资料参考

- Oppia GitHub README: https://github.com/oppia/oppia
- Oppia Key Terms: https://oppia-documentation.readthedocs.io/en/latest/keyconcepts.html
- Oppia Codebase Overview: https://github.com/oppia/oppia/wiki/Overview-of-the-Oppia-codebase
- DeepTutor 本地架构说明: `/Users/chen/code/deeptutor/AGENTS.md`
