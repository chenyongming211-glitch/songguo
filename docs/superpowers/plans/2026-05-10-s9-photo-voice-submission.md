# S9 Photo And Voice Submission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect mini-program photo and voice entries to the existing Learning Submission confirmation flow without creating a separate teaching path.

**Architecture:** Photo upload reuses the existing backend OCR seam, then pre-fills `submission-review` with recognized text and waits for user confirmation. Voice is a non-blocking fallback in this round: it explains that ASR is not configured and routes users to text confirmation instead of breaking the unified entry.

**Tech Stack:** WeChat mini-program JavaScript, existing `miniprogram/lib/api.js`, existing `/api/v1/learning/photo-review`, existing `submission-review` page.

---

### Task 1: API Client

**Files:**
- Modify: `miniprogram/lib/api.js`
- Test: `miniprogram/tests/submission-api.test.js`

- [ ] Add a failing test that calls a new photo recognition API helper and expects `POST /api/v1/learning/photo-review`.
- [ ] Run `node miniprogram/tests/submission-api.test.js` and verify it fails because the helper does not exist.
- [ ] Add the helper by aliasing the existing upload path into a submission-facing function.
- [ ] Re-run `node miniprogram/tests/submission-api.test.js`.

### Task 2: Home Entry Behavior

**Files:**
- Modify: `miniprogram/pages/chat-list/index.js`
- Modify: `miniprogram/pages/chat-list/index.wxml`
- Test: `miniprogram/tests/learning-copy.test.js`

- [ ] Add a failing copy/static test that the home page contains photo and voice handlers instead of routing both directly to `submission-review`.
- [ ] Run `node miniprogram/tests/learning-copy.test.js` and verify it fails.
- [ ] Implement photo selection with `wx.chooseMedia`/`wx.chooseImage`, upload OCR, then navigate to `submission-review` with recognized text.
- [ ] Implement voice fallback with a child-facing toast and route to text confirmation.
- [ ] Re-run `node miniprogram/tests/learning-copy.test.js`.

### Task 3: Review Page Prefill

**Files:**
- Modify: `miniprogram/pages/submission-review/index.js`
- Test: `miniprogram/tests/learning-copy.test.js`

- [ ] Add a failing test that `submission-review` decodes an `initialText` option.
- [ ] Run `node miniprogram/tests/learning-copy.test.js` and verify it fails.
- [ ] Initialize `inputValue` and `canSubmit` from decoded `initialText`.
- [ ] Re-run all mini-program tests.

### Task 4: Manual Verification

**Files:**
- No production files.

- [ ] Run all mini-program tests with `for f in miniprogram/tests/*.test.js; do node "$f" || exit 1; done`.
- [ ] Start the backend locally and verify the photo OCR seam can produce a draft or fallback text.
- [ ] In WeChat DevTools, click photo and verify the flow lands on `submission-review` instead of old chat detail.
