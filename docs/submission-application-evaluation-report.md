# LearningSubmission 100 题应用题评测报告

日期：2026-05-10

## 评测目的

验证 S7 统一提交链路在三到六年级应用题型上的稳定性：

```text
题目提交 -> 文本拆题 -> 判题 -> 正确题沉淀 -> 错题入队 -> PostgreSQL 数据隔离字段写入
```

本报告不替代微信开发者工具视觉验收，也不等同于完整错题陪练质量评测。

## 题集覆盖

题集文件：

```text
backend/evaluation/submission_application_math.py
```

覆盖规模：

```text
总题数：100
三年级：25 题
四年级：25 题
五年级：25 题
六年级：25 题
```

题型覆盖：

```text
三年级：加减法、乘除法、有余数除法、两步应用题、周长、面积入门、时间、人民币、长度/质量/容量单位、统计图、平均数、规律、比较、路线、购物、阅读计划、无关条件。
四年级：多位数乘除、速度路程、组合图形面积、复合周长、角度、小数、估算、面积单位、车辆取整、平均数、折线/表格、逆运算、方程、反求面积、时间计划、植树问题、多步差量、路线、无关条件。
五年级：小数乘除、分数意义、同分母分数加减、三角形/平行四边形/梯形面积、长方体体积和表面积、平均数、方程、因数倍数、最大公因数、最小公倍数、可能性、统计、反向应用题、组合面积、倍数关系、行程、分数应用。
六年级：百分数、折扣、增长率、比、比例、比例尺、圆面积/周长、圆柱体积/侧面积、负数、速度、工程问题、服务费、利息、概率、中位数、平均数、百分率、正方体表面积、反求体积、按比分配、无关条件。
```

覆盖检查命令：

```bash
PYTHONPATH=/Users/chen/code python -m pytest backend/tests/evaluation/test_submission_application_math.py
PYTHONPATH=/Users/chen/code python scripts/songguo_run_submission_eval.py --dry-run --limit 100
```

结果：

```text
2 passed
total=100
grade=3 count=25
grade=4 count=25
grade=5 count=25
grade=6 count=25
```

## Live 评测命令

第一轮包含少量错题陪练抽样：

```bash
PYTHONPATH=/Users/chen/code python -u scripts/songguo_run_submission_eval.py \
  --limit 100 \
  --batch-size 5 \
  --answer-mode mixed \
  --tutor-wrong-limit 8 \
  --timeout 180 \
  --json
```

第二轮修正评测集口径后重跑判题主链路：

```bash
PYTHONPATH=/Users/chen/code python -u scripts/songguo_run_submission_eval.py \
  --limit 100 \
  --batch-size 5 \
  --answer-mode mixed \
  --tutor-wrong-limit 0 \
  --timeout 180 \
  --json
```

第三轮修复判题质量缺口后重跑判题主链路：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  NO_PROXY=127.0.0.1,localhost \
  PYTHONPATH=/Users/chen/code \
  python -u scripts/songguo_run_submission_eval.py \
  --limit 100 \
  --batch-size 5 \
  --answer-mode mixed \
  --tutor-wrong-limit 0 \
  --timeout 180 \
  --json
```

## 第三轮结果

```text
total: 100
batch_count: 20
item_count: 100
expected_correct: 40
expected_wrong: 60
judge_match: 100
judge_mismatch: 0
judge_match_rate: 1.0
wrong_queue_count: 60
failures: 0
elapsed_seconds: 601.99
```

覆盖结果：

```text
三年级：25/25
四年级：25/25
五年级：25/25
六年级：25/25
题型类别：100 个 category 覆盖项
```

## 第二轮历史结果

```text
total: 100
batch_count: 20
item_count: 100
expected_correct: 40
expected_wrong: 60
judge_match: 97
judge_mismatch: 3
judge_match_rate: 0.97
wrong_queue_count: 61
failures: 0
elapsed_seconds: 615.81
```

PostgreSQL 数据隔离抽查：

```text
submission_family_id=family_openid_local_dev
item_family_ids=['family_openid_local_dev']
queue_family_ids=['family_openid_local_dev']
postgres_latest_submission_family_check=passed
```

## 已修复问题

1. 长度单位等价答案没有被判对。

```text
question_id: app_g3_12_length_conversion
题目：一根彩带2米35厘米，剪去80厘米，还剩多少厘米？
孩子答案：155厘米
期望：correct
实际：wrong
```

修复：抽出共享 `answers_match`，支持长度单位等价；同时修复 `MathFastPathRegistry._analyze_unit_conversion`，避免把“2米35厘米，剪去80厘米”误判成单纯单位换算 235 厘米。

2. 是/否型阅读计划答案没有被判对。

```text
question_id: app_g3_24_reading_plan
题目：一本故事书168页，小华已经读了75页，剩下每天读31页，3天能读完吗？
孩子答案：能读完
期望：correct
实际：wrong
```

修复：共享 `answers_match` 支持是/否极性归一化，`能读完` 和 `能，3天正好读完` 视为一致。

3. 差量方向判断错误。

```text
question_id: app_g4_23_multi_step_total_difference
题目：甲仓库有560袋米，乙仓库有420袋，甲运给乙80袋后，甲比乙少还是多多少袋？
孩子答案：甲多20袋
期望：wrong
实际：correct
```

修复：共享 `answers_match` 支持比较方向极性，`多/增加/超过` 与 `少/减少/不足` 不再只按数字尾部匹配。

live API 回归：

```text
app_g3_12_length_conversion: expected=True actual=True correct_answer=155厘米
app_g3_24_reading_plan: expected=True actual=True correct_answer=能
app_g4_23_multi_step_total_difference: expected=False actual=False correct_answer=甲比乙少20袋
```

## 结论

S7 submission 主链路能稳定处理 100 道三到六年级应用题：

```text
提交成功率：100/100
拆题成功率：100/100
判题匹配率：100/100
错题入队：正常
PostgreSQL family_id：正常
```

下一步不应扩新功能，应回到完整产品体验验收：

```text
1. 用小程序真实跑一次“一次提交 -> 判题概览 -> 多错题逐题陪练 -> 本次总结”。
2. 抽查 LearningDeposit、MasteryEvidence、WrongQuestion、TutorQueue 和家长反馈证据。
3. 再评测同类题生成和松鼠博士自然追问质量。
```

## 产品闭环验收补充

2026-05-10 继续做小程序主链路对应的 live API 产品验收，使用 3 道三四年级应用题模拟“一次提交中有正确题和错题”的场景。

验收链路：

```text
开始做题
-> 创建 LearningSubmission
-> 确认题目
-> 判题概览
-> 错题进入 TutorQueue
-> 松鼠博士逐题陪练
-> submission completed
-> 家长反馈 / 错题 / 学习记忆 / 复习计划可查询
```

本轮发现并修复一个体验不一致：

```text
修复前：createLearningSubmission 已经完成判题、沉淀和启动错题陪练，和小程序“确认题目后再判题”的体验不一致。
修复后：create 只做 intake parse，所有 item 仍是 unknown；confirm 后才执行判题、MasteryEvidence、WrongQuestion、TutorQueue 和首个错题 session。
```

live 验收结果：

```text
draft: status=intake_pending, judges=[unknown, unknown, unknown], evidence=0, queue=0, active=false
confirmed: status=tutoring, judges=[correct, correct, wrong], correct=2, wrong=1, evidence=3, queue=1, active=true
final: status=completed, active=false, queue=[completed]
wrong_questions: 1
learning_memory.top_weakness: 比较问题
review_plan.items: 1
```

家长反馈证据也已补齐全过程：

```text
提交时孩子答“甲多20袋”，系统判为错误。参考答案是“甲比乙少20袋”。主要错因是misconception_1。陪练后孩子改答“甲少20袋”，本题已完成。
```

本轮仍未替代微信开发者工具的视觉验收；下一步需要在小程序真机/开发者工具里检查页面层级、按钮状态、跳转和聊天内容呈现。
