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
