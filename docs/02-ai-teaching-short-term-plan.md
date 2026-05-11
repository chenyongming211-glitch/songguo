# 松果AI AI 教学体系短期落地计划

## 1. 计划定位

本文档把《松果AI长期方案：Oppia 教学结构 + DeepTutor AI 引擎 + 松果AI Teaching Kernel》拆成短期可执行任务。

短期目标不是做完整教学平台，而是把 v0.1/v0.2 阶段最关键的教学内核能力做稳：

```text
Runtime Kernel 最小闭环
+ AIEngineProvider 抽象
+ 小学数学 Skill / Misconception 种子库
+ 100 道黄金评测集
+ 答案泄露和结构化质量评测
```

短期计划服务于一个目标：

```text
孩子输入或拍一道小学数学题后，系统能稳定识别关键点，按孩子回答动态引导，记录错因，并生成可被家长看懂的反馈。
```

## 2. 短期边界

### 2.1 必须做

```text
1. Runtime Kernel 边界清晰。
2. DeepTutor 降级为 AIEngineProvider 之一。
3. 小学数学题目结构化网关继续增强。
4. 状态机按关键点推进，而不是固定步骤数。
5. 建立 Skill / Misconception 种子库。
6. 建立数学黄金评测集。
7. 建立答案泄露、结构化质量和错因一致性评测。
8. 学习事件、错题、家长反馈继续作为松果AI主库数据。
```

### 2.2 暂不做

```text
1. 不接入 Oppia 运行时。
2. 不做完整 Oppia 式 Exploration 平台。
3. 不做完整教研后台。
4. 不一次性覆盖小学全部知识图谱。
5. 不把 DeepTutor Memory 当成业务画像主库。
6. 不让大模型直接决定是否给答案。
7. 不让小程序直接调用 DeepTutor 原始 WebSocket。
```

## 3. 短期架构

短期架构应收敛为：

```text
微信小程序
  ↓
Songguo Learning API
  ↓
Runtime Kernel
  ├── LearningSession
  ├── KeyPoint State Machine
  ├── Answer Leakage Checker
  ├── LearningEvent
  └── WrongQuestion / SessionFeedback
  ↓
AIEngineProvider
  ├── DeepTutorProvider
  ├── DeepSeekProvider
  ├── OllamaProvider
  └── DeterministicFallbackProvider
  ↓
Teaching Seed Library
  ├── Skill Seed
  ├── Misconception Seed
  └── ConceptCard Template
  ↓
Evaluation Harness
  ├── Math Golden Set
  ├── Structuring Eval
  ├── Leakage Eval
  └── Misconception Eval
```

## 4. 工作流 A：Runtime Kernel 最小闭环

### 4.1 目标

把当前已有的结构化网关、状态机、泄露检查和学习事件统一归入 Runtime Kernel。

Runtime Kernel 只负责运行时教学控制，不负责大模型推理和教研资产维护。

### 4.2 主要任务

```text
1. 明确 Runtime Kernel 输入输出。
2. 统一 ProblemAnalysis / KeyPoint / AttemptEvaluation / TeachingProgress。
3. 状态机只消费结构化结果，不直接信任模型自然语言。
4. 每一次关键点推进都写 LearningEvent。
5. answer_unlocked=false 时，输出不得包含最终答案。
6. 未结构化题目显示“动态推进”，不暴露内部 hint_level。
```

### 4.3 验收标准

```text
1. 新建学习会话时能返回 teaching_progress。
2. 结构化题目能显示动态关键点进度。
3. 未结构化题目不显示固定 /5 步骤。
4. 孩子答对某个关键点后，状态机进入下一个关键点。
5. 孩子答错时，状态机不跳过当前关键点。
6. 学习事件能记录关键点推进和错因。
```

## 5. 工作流 B：AIEngineProvider 抽象

### 5.1 目标

避免松果AI业务代码绑定 DeepTutor 内部实现。

DeepTutor 应该是 AIEngineProvider 的一个实现，而不是唯一入口。

### 5.2 Provider 设计

短期至少定义这些能力：

```text
structure_math_problem(question_text, grade, context) -> ProblemAnalysis
generate_hint(problem_analysis, key_point, student_profile) -> TeachingDraft
generate_explanation(problem_analysis, unlock_context) -> TeachingDraft
generate_similar_practice(problem_analysis, misconception, limit) -> PracticeItems
summarize_session(events, profile_delta) -> SessionFeedback
```

### 5.3 Provider 类型

```text
DeepTutorProvider：调用 DeepTutor / ChatOrchestrator / RAG / Quiz。
DeepSeekProvider：调用 DeepSeek API 做结构化和候选生成。
OllamaProvider：调用本地模型，适合低成本试验。
DeterministicFallbackProvider：用于测试、降级和无模型环境。
```

### 5.4 验收标准

```text
1. LearningService 不直接依赖 DeepTutor 内部接口。
2. Provider 失败时可以降级到确定性安全草稿。
3. Provider 输出必须经过 Schema 校验和泄露检查。
4. Provider 调用有 request_id、student_id、tenant_id、provider、model、latency、token/cost 记录。
```

## 6. 工作流 C：小学数学 Skill 种子库

### 6.1 目标

先建立一套可用的 Skill Seed Library，不追求完整知识图谱。

第一版规模：

```text
30 个左右小学数学高频 Skill
```

### 6.2 建设方法

```text
人工定义骨架
+ AI 辅助扩展候选
+ 人工统一命名
+ 黄金评测集验证
+ 真实错题持续修正
```

### 6.3 第一批 Skill 建议

```text
两位数乘一位数
三位数乘一位数
多位数乘一位数
有余数除法
乘法求总数
平均分问题
限载进一问题
倍数关系
和差问题
归一问题
归总问题
单位换算
时间计算
人民币计算
长度周长
面积计算
长方形正方形周长
长方形正方形面积
分数初步
小数初步
四则混合运算
括号优先级
估算
验算
读题找条件
应用题列式
图形条件提取
表格信息读取
统计图读取
检查习惯
```

### 6.4 验收标准

```text
1. 每个 Skill 有唯一 id、名称、年级范围、学科、简短定义。
2. 每个 Skill 至少关联 1 个 ConceptCard 极简模板。
3. 每个 Skill 至少关联 1-3 个常见 Misconception。
4. ProblemAnalysis 能引用 Skill id，而不是只返回自由文本。
```

## 7. 工作流 D：小学数学 Misconception 种子库

### 7.1 目标

第一版建立 50 个左右常见错因，支撑错题记录、画像和家长反馈。

### 7.2 错因入库原则

AI 可以生成候选错因，但不能直接入库。

正式入库前必须检查：

```text
命名是否统一
是否可被孩子答案观测
是否能驱动下一步提示
是否能进入家长报告
是否和已有错因重复
是否适合对应年级
```

### 7.3 Misconception 示例

限载进一问题：

```text
只算总人数，不知道要分组
忘记除以每辆车人数
有余数但没有多加一辆
把“最多坐45人”理解成“一共45人”
不理解“至少需要”的含义
```

乘法计算：

```text
把乘以 5 当成乘以 10
漏加个位贡献
进位遗漏
把加法和乘法混淆
只算一部分数位
```

应用题读题：

```text
漏看关键条件
把每份数量当成总数
把问题问的目标看错
没有区分已知条件和要求问题
被无关条件干扰
```

### 7.4 验收标准

```text
1. 每个 Misconception 有唯一 id、名称、解释、证据样例。
2. 每个 Misconception 绑定一个或多个 Skill。
3. 学习事件能记录 misconception_id。
4. 家长反馈能显示错因的儿童学习解释，而不是内部标签。
```

## 8. 工作流 E：ConceptCard 极简模板

### 8.1 目标

孩子卡住时，系统能回到概念和方法，而不是直接给答案。

第一版每个高频 Skill 只需要一个极简 ConceptCard。

### 8.2 ConceptCard 字段

```text
id
skill_id
title
grade_range
concept_explanation
simple_example
common_mistake
parent_tip
version
```

### 8.3 示例

```text
Skill：限载进一问题
ConceptCard：什么叫“至少需要几辆车”
概念：如果还有人没坐上车，就需要再加一辆车。
例子：91 人坐车，每辆 45 人。45+45=90，还剩 1 人，所以需要第 3 辆车。
易错：只算 91 ÷ 45 = 2，就忘了剩下的人。
家长提示：让孩子先画圈分组，看到“还剩下”。
```

### 8.4 验收标准

```text
1. 每个 ConceptCard 不直接替孩子做当前题。
2. 示例题不能和当前题完全相同。
3. 文案适合小学阶段。
4. 可用于家长端解释和孩子端补救提示。
```

## 9. 工作流 F：100 道数学黄金评测集

### 9.1 目标

建立结构化网关和状态机质量基线。

没有评测集，不扩大题型覆盖。

### 9.2 覆盖范围

第一版至少 100 道题，覆盖：

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

### 9.3 每题标注字段

```text
question_id
grade
subject
question_text
skill_ids
misconception_ids
conditions
target
solution_steps
key_points
final_answer
forbidden_content
expected_first_prompt
acceptable_child_attempts
```

### 9.4 评测指标

```text
题目结构化准确率
Skill 命中率
关键点覆盖率
最终答案正确率
答案泄露率
错因识别一致性
提示年级适配率
同类题可用率
平均响应时延
单题模型成本
```

### 9.5 验收标准

```text
1. 评测集可在本地自动运行。
2. 每次 Provider 或 Prompt 调整后能跑回归。
3. 答案泄露率必须作为阻断指标。
4. 结构化准确率和关键点覆盖率必须可量化。
```

## 10. 短期里程碑

### M4-1：Runtime Kernel 边界收敛

```text
目标：所有教学状态、关键点推进和答案解锁逻辑归入 Runtime Kernel。
验收：小程序不显示固定步骤，结构化题目按关键点推进。
```

### M4-2：AIEngineProvider 抽象

```text
目标：DeepTutorProvider、DeepSeekProvider、OllamaProvider、FallbackProvider 走统一接口。
验收：LearningService 不直接依赖 DeepTutor 内部接口。
```

### M4-3：数学 Skill / Misconception 种子库

```text
目标：落地 30 个 Skill、50 个 Misconception 和 ConceptCard 极简模板。
验收：ProblemAnalysis 能引用 skill_id 和 misconception_id。
```

### M4-4：黄金评测集和评测脚手架

```text
目标：建立 100 道小学数学黄金评测集。
验收：结构化、泄露、关键点覆盖、错因一致性能自动评测。
```

### M4-5：受控生成闭环

```text
目标：大模型输出绑定 Skill、Misconception、KeyPoint 和 ConceptCard。
验收：孩子端只看到当前应释放的关键点，家长端看到错因解释和建议。
状态：已完成第一版。Runtime Kernel 已控制 ConceptCard 释放、同类题生成和家长端错因解释。
```

## 11. 短期完成定义

短期计划完成的标准不是“功能多”，而是：

```text
1. 松果AI有清晰 Runtime Kernel。
2. DeepTutor 只是 AIEngineProvider 之一。
3. 小学数学有第一版 Skill / Misconception 种子库。
4. 有 100 道黄金评测集守住质量。
5. 孩子端引导是动态关键点，不是固定步骤。
6. 家长端反馈能解释孩子为什么不会。
7. 任何答案泄露都能被测试发现。
```
