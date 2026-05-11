# 松果AI Mini-Program V1

This directory contains a native WeChat mini-program scaffold for the child-facing learning flow.

The production child path uses the product-owned `learning` APIs. Raw engine debug helpers are not part of the child-facing path.

Current Songguo planning index: `/Users/chen/code/songguo/docs/README.md`.

## Included Pages

- `pages/chat-list`: list existing product learning sessions and start a new guided learning session
- `pages/chat-detail`: create a controlled math learning session and submit child attempts through `/api/v1/learning/*`
- `pages/parent-report`: show weekly report, learning memory, and wrong-question evidence through `/api/v1/parent/*`
- `pages/settings`: configure backend endpoints and test `/api/v1/system/status`

## Import into WeChat DevTools

1. Open WeChat DevTools.
2. Import `/Users/chen/code/songguo/miniprogram`.
3. Use your own AppID or the DevTools tourist AppID.
4. Keep the default local backend during laptop testing:
   - HTTP: `http://127.0.0.1:8001`
   - Settings now tests only `/api/v1/system/status`; the child-facing path does not use a WebSocket endpoint.

## Real Device Notes

- Real devices generally require whitelisted `https://` request domains.
- Debug WebSocket connections should use `wss://`.
- External teaching engines are optional provider layers and are not the v0.1 main runtime.

## Local Mock vs Real WeChat

Local development can run without a real WeChat AppID:

```bash
export WECHAT_MOCK_OPENID=openid_mock
export SONGGUO_SESSION_SECRET=local-dev-session-secret
```

In mock mode, `/api/v1/wechat/login` returns a deterministic `openid`, `session_token`, and default `child_id`. The mini-program can show this as Mock 微信登录 in Settings.

For a real device or small private trial, configure:

```bash
export WECHAT_APPID=your_wechat_appid
export WECHAT_SECRET=your_wechat_secret
export SONGGUO_SESSION_SECRET=replace_with_strong_random_secret
export WECHAT_CONTENT_SAFETY_PROVIDER=wechat
```

The backend accepts `X-Session-Token` on learning and parent APIs. When this header is present, the backend checks that the requested `child_id` is bound to the current `openid`; without the header, local M1 mock behavior remains available for desktop testing.

## OCR / Vision Configuration

The photo-review flow defaults to deterministic OCR fallback. It is enough for local fixture testing and will enter manual confirmation when recognition is uncertain.

To enable a real vision model seam:

```bash
export SONGGUO_PHOTO_OCR_PROVIDER=vision
export SONGGUO_VISION_MODEL=your_vision_model_name
```

The vision model must return JSON with:

```json
{
  "question_text": "36 x 5 = ?",
  "child_answer": "360",
  "work_steps": "optional",
  "confidence": 0.88
}
```

When OCR fails or confidence is too low, the mini-program asks the user to manually confirm the question and the child's answer instead of blocking the learning flow.

## Current Scope

- The child-facing page does not call a raw WebSocket endpoint or send engine-specific debug events.
- The first production flow is text-input grade-3 math wrong-question guidance.
- The backend controls `hint_level`, `answer_unlocked`, and resume state.
- Small trial mode has a minimal `openid/session_token/child_id` binding check when `X-Session-Token` is supplied.
- AI draft calls are logged with basic operation, token estimate, status, provider, and model fields for cost inspection.
- Knowledge-base upload is not wired into the mini-program yet.
- The project has been statically verified, but it has not been packaged or run inside WeChat DevTools in this environment.
