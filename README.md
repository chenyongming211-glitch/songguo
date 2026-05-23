# 松果AI

松果AI是独立产品层，定位为面向小学阶段学生的多租户 AI 学习陪练平台。

## 边界

- `songguo/`: 松果AI产品层，包含小程序、产品文档、业务后端、脚本和测试。
- `others/deeptutor/`: 开源 DeepTutor AI 学习引擎归档目录，后续只作为可选引擎或资产来源。
- 松果AI业务后端负责账号、租户、学生、家长、权限、错题、学习画像、周报、计费和合规。
- DeepTutor 不直接暴露给小程序；小程序只调用松果AI业务 API。

## 当前目录

- `docs/`: 松果AI产品总纲和后续 PRD/路线图。
- `miniprogram/`: 微信小程序端。
- `backend/`: 松果AI业务后端，包含学习状态机、错题、家长报告、微信登录 seam、提醒和图解等产品 API。
- `backend/tests/`: 松果AI后端和小程序合同测试。
- `scripts/`: 松果AI本地验证和演示脚本。
- `.env`: 松果AI本地模型和运行时配置。
- `requirements.txt`: Songguo 后端运行依赖。
- `requirements-langgraph.txt`: 后续真实 LangGraph 主链路依赖。
- `requirements-postgres.txt`: 后续真实试点 PostgreSQL 依赖。

## 本地启动

从 `/Users/chen/code/songguo` 运行：

```bash
PYTHONPATH=/Users/chen/code uvicorn songguo.backend.api.app:app --host 127.0.0.1 --port 8001 --reload
```

本地启用真实意图识别智能体时，先确保 `.env` 已配置 `LLM_API_KEY` 或 `DEEPSEEK_API_KEY`，再启动：

```bash
export SONGGUO_INTENT_ROUTER_PROVIDER=llm
export SONGGUO_BASIC_RUBRIC_PROVIDER=llm
export LLM_MODEL=deepseek-v4-flash
export DEEPSEEK_MODEL=deepseek-v4-flash
PYTHONPATH=/Users/chen/code uvicorn songguo.backend.api.app:app --host 127.0.0.1 --port 8001 --reload
```

本地启用真实拍照 OCR 时，优先使用阿里云读光教育 OCR。它和阿里云 MaaS OpenAI-compatible 视觉模型不是同一个服务，配置也分开：

```bash
SONGGUO_PHOTO_OCR_PROVIDER=aliyun_edu
SONGGUO_ALIYUN_EDU_OCR_ACCESS_KEY_ID=your_access_key_id
SONGGUO_ALIYUN_EDU_OCR_ACCESS_KEY_SECRET=your_access_key_secret
SONGGUO_ALIYUN_EDU_OCR_ENDPOINT=https://ocr-api.cn-hangzhou.aliyuncs.com
SONGGUO_ALIYUN_EDU_OCR_REGION=cn-hangzhou
SONGGUO_ALIYUN_EDU_OCR_SCENE=auto
SONGGUO_ALIYUN_EDU_OCR_ROUTER_MODE=auto
SONGGUO_ALIYUN_EDU_OCR_MAX_SECONDARY_ACTIONS=1
SONGGUO_ALIYUN_EDU_OCR_CUT_TYPE=question
SONGGUO_ALIYUN_EDU_OCR_IMAGE_TYPE=photo
SONGGUO_ALIYUN_EDU_OCR_SUBJECT=default
SONGGUO_ALIYUN_EDU_OCR_OUTPUT_ORICOORD=true
SONGGUO_ALIYUN_EDU_OCR_NEED_ROTATE=true
SONGGUO_ALIYUN_EDU_OCR_TIMEOUT_SECONDS=12
SONGGUO_ALIYUN_EDU_OCR_RETRY_ATTEMPTS=2
SONGGUO_ALIYUN_EDU_OCR_RETRY_BACKOFF_SECONDS=0.4
SONGGUO_ALIYUN_EDU_OCR_FALLBACK_PROVIDER=none
SONGGUO_ALIYUN_EDU_OCR_HYBRID_TEXT_FALLBACK=false
SONGGUO_ALIYUN_EDU_OCR_HYBRID_TEXT_FALLBACK_MIN_ANSWER_RATE=0.6
```

`SONGGUO_ALIYUN_EDU_OCR_SCENE` 可选 `auto`、`paper_cut`、`paper_ocr`、`question_ocr`、`oral_calculation`、`formula`、`paper_structed`。默认用 `auto`，由 `EducationOcrRouter` 先走 `PaperStructed` 做整页结构化识别；结构为空时回落到 `PaperCut`，答案覆盖率低或置信度偏低时最多追加 `SONGGUO_ALIYUN_EDU_OCR_MAX_SECONDARY_ACTIONS` 次 `PaperOcr` 文本补强。口算专项可显式切到 `oral_calculation`。

需要“每题框上打勾/叉”时，保持 `SONGGUO_ALIYUN_EDU_OCR_SCENE=auto` 即可。当前策略尽量整图一次结构化 OCR；只有质量信号不足时才追加一次 `paper_cut` 或 `paper_ocr`。`SONGGUO_ALIYUN_EDU_OCR_FALLBACK_PROVIDER=none` 用来避免教育 OCR 失败时自动切到视觉模型增加延迟和费用；要临时兜底时再显式改成 `vision`。

教育 OCR 失败时可临时回退到阿里云 MaaS 视觉模型。默认先用 `qwen3.6-flash`，它支持视觉理解且适合私测阶段控制延迟和成本：

```bash
SONGGUO_PHOTO_OCR_PROVIDER=vision
SONGGUO_VISION_BINDING=aliyun
SONGGUO_VISION_MODEL=qwen3.6-flash
SONGGUO_VISION_HOST=https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
SONGGUO_VISION_API_KEY=your_api_key_here
SONGGUO_VISION_TIMEOUT_SECONDS=20
```

如果照片识别质量不够，优先把 `SONGGUO_VISION_MODEL` 切到 `qwen3.6-plus`，后端接口不需要改。

OCR 样本评测入口：

```bash
PYTHONPATH=/Users/chen/code python scripts/songguo_run_ocr_eval.py --provider deterministic
PYTHONPATH=/Users/chen/code python scripts/songguo_run_ocr_eval.py --provider vision --manifest data/evaluation/ocr_samples/manifest.jsonl --json
PYTHONPATH=/Users/chen/code python scripts/songguo_run_ocr_eval.py --provider aliyun_edu --manifest data/evaluation/ocr_samples/manifest.jsonl --json
```

评测报告会输出 `answer_coverage_rate`、`ocr_action_counts`、题框质量、题干/答案匹配率和平均耗时。端到端评测还会输出 `fallback_running_rate`、`pending_rate` 和 `final_judgement_rate`，用于判断“先出部分结果、少量题异步复核”的产品体验是否可接受。

样本清单在 `data/evaluation/ocr_samples/manifest.jsonl`，真实手机照片样本可按 `data/evaluation/ocr_samples/README.md` 追加。

没有真实照片时可先生成合成压力样本：

```bash
PYTHONPATH=/Users/chen/code python scripts/songguo_generate_ocr_stress_samples.py --output /private/tmp/songguo_ocr_stress_full
PYTHONPATH=/Users/chen/code python scripts/songguo_run_ocr_eval.py --provider vision --manifest /private/tmp/songguo_ocr_stress_full/manifest.jsonl --json
```

图片到提交闭环端到端评测：

```bash
PYTHONPATH=/Users/chen/code python scripts/songguo_run_photo_submission_e2e_eval.py --provider deterministic --agents deterministic
PYTHONPATH=/Users/chen/code python scripts/songguo_run_photo_submission_e2e_eval.py --provider vision --agents env --manifest /private/tmp/songguo_ocr_stress_full/manifest.jsonl --json
PYTHONPATH=/Users/chen/code python scripts/songguo_run_photo_submission_e2e_eval.py --provider aliyun_edu --agents env --manifest /private/tmp/songguo_ocr_stress_full/manifest.jsonl --json
```

小程序设置页测试的后端地址为：

```text
http://127.0.0.1:8001
```

系统状态接口：

```text
GET /api/v1/system/status
```

## PostgreSQL 试点存储

本地开发默认仍使用 SQLite。真实小程序试点可切 PostgreSQL：

```bash
pip install -r requirements-postgres.txt
export SONGGUO_DATABASE_URL=postgresql://songguo:password@127.0.0.1:5432/songguo
PYTHONPATH=/Users/chen/code uvicorn songguo.backend.api.app:app --host 127.0.0.1 --port 8001
```

`SONGGUO_DATABASE_URL` 未设置时，后端继续使用 `data/learning.db`。

## 真实环境鉴权

本地开发默认允许无 `X-Session-Token` 调接口，方便微信开发者工具和后端测试。真实试点必须开启严格模式：

```bash
export SONGGUO_AUTH_MODE=strict
```

严格模式下：

- child 维度学习、错题、家长报告和复习计划 API 必须携带 `X-Session-Token`。
- token 只能访问当前 openid 绑定的 child。
- 无 token 返回 401，跨 child 返回 403。

## 当前状态

松果AI业务源码、文档、脚本和本地配置均已放在 `songguo/` 下。开源 DeepTutor 源码已经迁入 `others/deeptutor/`，不再作为 v0.1 主链路依赖。
