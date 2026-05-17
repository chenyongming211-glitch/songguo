from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
MINIPROGRAM = "songguo/miniprogram"


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_miniprogram_has_photo_review_upload_contract() -> None:
    api = _read(f"{MINIPROGRAM}/lib/api.js")
    chat_list_js = _read(f"{MINIPROGRAM}/pages/chat-list/index.js")
    chat_list_wxml = _read(f"{MINIPROGRAM}/pages/chat-list/index.wxml")
    chat_js = _read(f"{MINIPROGRAM}/pages/chat-detail/index.js")
    chat_wxml = _read(f"{MINIPROGRAM}/pages/chat-detail/index.wxml")

    assert "function createPhotoReview(" in api
    assert "function confirmPhotoReview(" in api
    assert "function getSessionChildId(" in api
    assert "wx.uploadFile" in api
    assert "function getSessionToken()" in api
    assert '"X-Session-Token"' in api
    assert "/api/v1/learning/photo-review" in api
    assert "createPhotoReview" in chat_js
    assert "confirmPhotoReview" in chat_js
    assert "listChildren" in chat_list_js
    assert "upsertChild" in chat_list_js
    assert "childId: getSessionChildId()" in chat_list_js
    assert "handleChildChange" in chat_list_js
    assert 'bindtap="handleChildChange"' in chat_list_wxml
    assert "learner-pill-row" in chat_list_wxml
    assert "learner-strip card" not in chat_list_wxml
    assert "学习对象" not in chat_list_wxml
    assert "当前后端" not in chat_list_wxml
    assert "学习场景" not in chat_list_wxml
    assert "数学 错题引导" not in chat_list_wxml
    assert "英语" not in chat_list_wxml
    assert "语文" not in chat_list_wxml
    assert "拍照做题" in chat_list_wxml
    assert "动态关键点" not in chat_list_wxml
    assert "scope-row" not in chat_list_wxml
    assert "orbit-number" not in chat_list_wxml
    assert "级提示" not in chat_list_wxml
    assert "listChildren" in chat_js
    assert "upsertChild" in chat_js
    assert "childId: getSessionChildId()" in chat_js
    assert "handleChildChange" in chat_js
    assert "selectedSubject" in chat_js
    assert "subject: this.data.selectedSubject" in chat_js
    assert "createPhotoReview(filePath, childId, subject)" in api
    assert "songguo_selected_child_id" in chat_js
    assert "pendingPhotoReviewId" in chat_js
    assert "handleChoosePhoto" in chat_js
    assert 'bindtap="handleChoosePhoto"' in chat_wxml
    assert 'bindtap="handleSubjectChange"' not in chat_wxml
    assert 'bindtap="handleChildChange"' not in chat_wxml
    assert "学习学科" not in chat_wxml
    assert "动态关键点，不是固定" not in chat_wxml
    assert "题目" in chat_wxml
    assert "{{questionText}}" in chat_wxml
    assert "当前关键点" not in chat_wxml
    assert "progress-count" not in chat_wxml
    assert "answer_policy_label" in chat_wxml
    assert "hint-ladder" not in chat_wxml
    assert "/ 5" not in chat_wxml
    assert "等级 {{hintLevel" not in chat_wxml
    assert "根据你之前的错因" in chat_js
    assert "36 x 5" not in chat_js
    assert '"child_001"' not in chat_list_js
    assert '"child_001"' not in chat_js


def test_miniprogram_parent_page_shows_review_plan() -> None:
    api = _read(f"{MINIPROGRAM}/lib/api.js")
    app_js = _read(f"{MINIPROGRAM}/app.js")
    parent_js = _read(f"{MINIPROGRAM}/pages/parent-report/index.js")
    parent_wxml = _read(f"{MINIPROGRAM}/pages/parent-report/index.wxml")

    assert "function getParentReviewPlan(" in api
    assert "function getParentSafetyEvents(" in api
    assert "function submitPracticeResult(" in api
    assert "function loginWithWechat(" in api
    assert "function checkWechatTextSafety(" in api
    assert "function saveReminderSubscription(" in api
    assert "function getDueReminders(" in api
    assert "function generateLearningDiagram(" in api
    assert "function requestLearningAnimation(" in api
    assert "loginWithWechat" in app_js
    assert "function listChildren(" in api
    assert "function upsertChild(" in api
    assert "childId: getSessionChildId()" in parent_js
    assert "/review-plan?scope=" in api
    assert "function getParentLearningMemory(childId, scope)" in api
    assert "function getTargetedPractice(childId, limit, scope)" in api
    assert "const practiceLimit = Math.max(1, Math.min(Number(limit || 3), 3));" in api
    assert "targeted-practice?limit=${practiceLimit}" in api
    assert "function getParentSessionFeedback(childId, sessionId)" in api
    assert "/session-feedback?session_id=" in api
    assert "function getParentLearningDeposit(childId, sessionId, scope)" in api
    assert "/learning-deposit?session_id=" in api
    assert "function getParentSummaryDraft(childId, scope)" in api
    assert "/summary-draft?scope=" in api
    assert "function rollupParentSummary(childId, scope)" in api
    assert "/summary-rollup?scope=" in api
    assert "getParentSessionFeedback" in parent_js
    assert "getParentSummaryDraft" in parent_js
    assert "rollupParentSummary" in parent_js
    assert "sessionFeedback" in parent_js
    assert "summaryDraft" in parent_js
    assert "getParentReviewPlan" in parent_js
    assert "listChildren" in parent_js
    assert "upsertChild" in parent_js
    assert "handleChildChange" in parent_js
    assert "getParentSafetyEvents" in parent_js
    assert "generateLearningDiagram" in parent_js
    assert "requestLearningAnimation" in parent_js
    assert "submitPracticeResult" in parent_js
    assert "handleMarkResolved" in parent_js
    assert "handleGenerateDiagram" in parent_js
    assert "handleRequestAnimation" in parent_js
    assert "handleEnableReminder" in parent_js
    assert "safetyEvents" in parent_js
    assert "reviewPlan" in parent_js
    assert "reviewScope" in parent_js
    assert "handleScopeChange" in parent_js
    assert '"yearly"' in parent_js
    assert "报告周期" in parent_wxml
    assert "行动建议" in parent_wxml
    assert "薄弱点" in parent_wxml
    assert "最近错题" in parent_wxml
    assert "学习资产沉淀" not in parent_wxml
    assert "报告草稿" not in parent_wxml
    assert 'range="{{reviewScopeOptions}}"' in parent_wxml
    assert 'bindchange="handleScopePickerChange"' in parent_wxml
    assert "安全拦截记录" in parent_wxml
    assert 'bindtap="handleMarkResolved"' in parent_wxml
    assert 'bindtap="handleGenerateDiagram"' in parent_wxml
    assert 'bindtap="handleRequestAnimation"' in parent_wxml


def test_miniprogram_settings_page_shows_backend_status_and_wechat_mode() -> None:
    settings_js = _read(f"{MINIPROGRAM}/pages/settings/index.js")
    settings_wxml = _read(f"{MINIPROGRAM}/pages/settings/index.wxml")

    assert "httpBaseUrl" in settings_js
    assert "getSystemStatus" in settings_js
    assert "wechatMode" in settings_js
    assert "refreshWechatMode" in settings_js
    assert "songguo_wechat_session" in settings_js
    assert "微信模式" in settings_wxml
    assert "后端地址" in settings_wxml
    assert "当前服务" in settings_wxml
    assert "高级诊断" in settings_wxml
    assert "测试连接" in settings_wxml
