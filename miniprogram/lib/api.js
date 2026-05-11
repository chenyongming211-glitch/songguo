const { getBackendConfig } = require("./config");

const SESSION_STORAGE_KEY = "songguo_wechat_session";
const LEGACY_SESSION_STORAGE_KEY = ["deep", "tutor_wechat_session"].join("");

function buildUrl(path) {
  const { httpBaseUrl } = getBackendConfig();
  return `${httpBaseUrl}${path}`;
}

function getSessionToken() {
  try {
    const session =
      wx.getStorageSync(SESSION_STORAGE_KEY) ||
      wx.getStorageSync(LEGACY_SESSION_STORAGE_KEY) ||
      {};
    return session.session_token || "";
  } catch (_error) {
    return "";
  }
}

function clearStoredSession() {
  try {
    wx.removeStorageSync(SESSION_STORAGE_KEY);
    wx.removeStorageSync(LEGACY_SESSION_STORAGE_KEY);
  } catch (_error) {
    // 本地存储清理失败不影响本次匿名降级重试。
  }
}

function getSessionChildId() {
  try {
    const session =
      wx.getStorageSync(SESSION_STORAGE_KEY) ||
      wx.getStorageSync(LEGACY_SESSION_STORAGE_KEY) ||
      {};
    return session.child_id || "child_openid_local_dev";
  } catch (_error) {
    return "child_openid_local_dev";
  }
}

function buildHeaders(extraHeaders, options) {
  const headers = {
    ...(extraHeaders || {}),
  };
  if (!(options && options.skipContentType) && !headers["content-type"]) {
    headers["content-type"] = "application/json";
  }
  const token = options && options.skipSessionToken ? "" : getSessionToken();
  if (token) {
    headers["X-Session-Token"] = token;
  }
  return headers;
}

function request(options) {
  const method = options && options.method ? options.method : "GET";
  return new Promise((resolve, reject) => {
    const runRequest = (skipSessionToken) => {
      wx.request({
        url: buildUrl(options.path),
        method,
        data: options && options.data ? options.data : undefined,
        header: buildHeaders({}, { skipSessionToken }),
        success: (response) => {
          const { statusCode, data } = response;
          if (statusCode >= 200 && statusCode < 300) {
            resolve(data);
            return;
          }
          if (statusCode === 401 && !skipSessionToken && getSessionToken()) {
            clearStoredSession();
            runRequest(true);
            return;
          }
          const message =
            (data && (data.detail || data.message || data.error)) ||
            `Request failed with status ${statusCode}`;
          reject(new Error(String(message)));
        },
        fail: (error) => {
          reject(new Error((error && error.errMsg) || "Network request failed"));
        },
      });
    };
    runRequest(false);
  });
}

function upload(options) {
  return new Promise((resolve, reject) => {
    wx.uploadFile({
      url: buildUrl(options.path),
      filePath: options.filePath,
      name: options.name || "file",
      formData: options.formData || {},
      header: buildHeaders({}, { skipContentType: true }),
      success: (response) => {
        const statusCode = response.statusCode;
        let data = response.data;
        try {
          data = typeof data === "string" ? JSON.parse(data) : data;
        } catch (error) {
          reject(new Error("Upload response is not valid JSON"));
          return;
        }
        if (statusCode >= 200 && statusCode < 300) {
          resolve(data);
          return;
        }
        const message =
          (data && (data.detail || data.message || data.error)) ||
          `Upload failed with status ${statusCode}`;
        reject(new Error(String(message)));
      },
      fail: (error) => {
        reject(new Error((error && error.errMsg) || "Upload failed"));
      },
    });
  });
}

function listLearningSessions(childId) {
  const suffix = childId ? `?child_id=${encodeURIComponent(childId)}` : "";
  return request({
    path: `/api/v1/learning/sessions${suffix}`,
  });
}

function createLearningSession(payload) {
  return request({
    path: "/api/v1/learning/session",
    method: "POST",
    data: {
      child_id: payload.childId || getSessionChildId(),
      subject: payload.subject || "math",
      grade: Number(payload.grade || 3),
      input_type: "text",
      question_text: payload.questionText,
    },
  });
}

function createLearningSubmission(payload) {
  return request({
    path: "/api/v1/learning/submissions",
    method: "POST",
    data: {
      child_id: payload.childId || getSessionChildId(),
      subject: payload.subject || "math",
      grade: Number(payload.grade || 3),
      source_type: payload.sourceType || "text",
      raw_text: payload.rawText || "",
    },
  });
}

function getLearningSubmission(submissionId, childId) {
  const suffix = childId ? `?child_id=${encodeURIComponent(childId)}` : "";
  return request({
    path: `/api/v1/learning/submissions/${encodeURIComponent(submissionId)}${suffix}`,
  });
}

function confirmLearningSubmission(submissionId, payload) {
  return request({
    path: `/api/v1/learning/submissions/${encodeURIComponent(submissionId)}/confirm`,
    method: "POST",
    data: {
      raw_text: payload && payload.rawText !== undefined ? payload.rawText : undefined,
    },
  });
}

function startNextSubmissionTutorItem(submissionId) {
  return request({
    path: `/api/v1/learning/submissions/${encodeURIComponent(submissionId)}/tutor/next`,
    method: "POST",
  });
}

function submitSubmissionTutorAttempt(submissionId, childAnswer) {
  return request({
    path: `/api/v1/learning/submissions/${encodeURIComponent(submissionId)}/tutor/attempt`,
    method: "POST",
    data: {
      child_answer: childAnswer,
    },
  });
}

function submitLearningAttempt(sessionId, childAnswer) {
  return request({
    path: `/api/v1/learning/session/${encodeURIComponent(sessionId)}/attempt`,
    method: "POST",
    data: {
      child_answer: childAnswer,
    },
  });
}

function submitPracticeResult(questionId, payload) {
  return request({
    path: `/api/v1/learning/wrong-questions/${encodeURIComponent(
      questionId
    )}/practice-result`,
    method: "POST",
    data: {
      child_id: payload.childId || getSessionChildId(),
      correct: Boolean(payload.correct),
    },
  });
}

function resumeLearningSession(sessionId) {
  return request({
    path: `/api/v1/learning/session/${encodeURIComponent(sessionId)}/resume`,
  });
}

function createPhotoReview(filePath, childId, subject) {
  return upload({
    path: "/api/v1/learning/photo-review",
    filePath,
    formData: {
      child_id: childId || getSessionChildId(),
      subject: subject || "math",
      grade: 3,
    },
  });
}

function recognizeSubmissionPhoto(filePath, childId, subject) {
  return upload({
    path: "/api/v1/learning/submissions/photo-draft",
    filePath,
    formData: {
      child_id: childId || getSessionChildId(),
      subject: subject || "math",
      grade: 3,
    },
  });
}

function confirmPhotoReview(reviewId, payload) {
  return request({
    path: `/api/v1/learning/photo-review/${encodeURIComponent(reviewId)}/confirm`,
    method: "POST",
    data: {
      question_text: payload.questionText,
      child_answer: payload.childAnswer,
    },
  });
}

function getTargetedPractice(childId, limit, scope) {
  const practiceLimit = Math.max(1, Math.min(Number(limit || 3), 3));
  return request({
    path: `/api/v1/learning/children/${encodeURIComponent(
      childId || getSessionChildId()
    )}/targeted-practice?limit=${practiceLimit}&scope=${encodeURIComponent(
      scope || "weekly"
    )}`,
  });
}

function getParentWeeklyReport(childId) {
  return request({
    path: `/api/v1/parent/children/${encodeURIComponent(childId || getSessionChildId())}/weekly-report`,
  });
}

function getParentSessionFeedback(childId, sessionId) {
  return request({
    path: `/api/v1/parent/children/${encodeURIComponent(
      childId || getSessionChildId()
    )}/session-feedback?session_id=${encodeURIComponent(sessionId || "")}`,
  });
}

function getParentLearningDeposit(childId, sessionId, scope) {
  return request({
    path: `/api/v1/parent/children/${encodeURIComponent(
      childId || getSessionChildId()
    )}/learning-deposit?session_id=${encodeURIComponent(
      sessionId || ""
    )}&scope=${encodeURIComponent(scope || "weekly")}`,
  });
}

function getParentSummaryDraft(childId, scope) {
  return request({
    path: `/api/v1/parent/children/${encodeURIComponent(
      childId || getSessionChildId()
    )}/summary-draft?scope=${encodeURIComponent(scope || "weekly")}`,
  });
}

function rollupParentSummary(childId, scope) {
  return request({
    path: `/api/v1/parent/children/${encodeURIComponent(
      childId || getSessionChildId()
    )}/summary-rollup?scope=${encodeURIComponent(scope || "weekly")}`,
    method: "POST",
  });
}

function getParentWrongQuestions(childId) {
  return request({
    path: `/api/v1/parent/children/${encodeURIComponent(childId || getSessionChildId())}/wrong-questions`,
  });
}

function getParentLearningMemory(childId, scope) {
  return request({
    path: `/api/v1/parent/children/${encodeURIComponent(
      childId || getSessionChildId()
    )}/learning-memory?scope=${encodeURIComponent(scope || "weekly")}`,
  });
}

function getParentReviewPlan(childId, scope) {
  return request({
    path: `/api/v1/parent/children/${encodeURIComponent(
      childId || getSessionChildId()
    )}/review-plan?scope=${encodeURIComponent(scope || "weekly")}`,
  });
}

function getParentSafetyEvents(childId) {
  return request({
    path: `/api/v1/parent/children/${encodeURIComponent(
      childId || getSessionChildId()
    )}/safety-events`,
  });
}

function listChildren() {
  return request({
    path: "/api/v1/parent/children",
  });
}

function upsertChild(payload) {
  return request({
    path: "/api/v1/parent/children",
    method: "POST",
    data: {
      child_id: payload.childId,
      name: payload.name,
      grade: Number(payload.grade || 3),
      term_label: payload.termLabel || "",
    },
  });
}

function getSystemStatus() {
  return request({
    path: "/api/v1/system/status",
  });
}

function loginWithWechat(code) {
  return request({
    path: "/api/v1/wechat/login",
    method: "POST",
    data: { code },
  });
}

function checkWechatTextSafety(text, openid) {
  return request({
    path: "/api/v1/wechat/content-safety/text",
    method: "POST",
    data: {
      openid: openid || "",
      text,
    },
  });
}

function saveReminderSubscription(payload) {
  return request({
    path: "/api/v1/reminders/subscriptions",
    method: "POST",
    data: {
      openid: payload.openid,
      child_id: payload.childId || getSessionChildId(),
      template_id: payload.templateId,
      enabled: payload.enabled !== false,
      scope: payload.scope || "weekly",
    },
  });
}

function getDueReminders(childId) {
  const suffix = childId ? `?child_id=${encodeURIComponent(childId)}` : "";
  return request({
    path: `/api/v1/reminders/due${suffix}`,
  });
}

function generateLearningDiagram(payload) {
  return request({
    path: "/api/v1/learning-artifacts/diagram",
    method: "POST",
    data: {
      child_id: payload.childId || getSessionChildId(),
      question_text: payload.questionText,
      knowledge_point: payload.knowledgePoint || "grade_math_unknown",
    },
  });
}

function requestLearningAnimation(payload) {
  return request({
    path: "/api/v1/learning-artifacts/animation",
    method: "POST",
    data: {
      child_id: payload.childId || getSessionChildId(),
      question_text: payload.questionText,
      knowledge_point: payload.knowledgePoint || "grade_math_unknown",
    },
  });
}

module.exports = {
  checkWechatTextSafety,
  confirmLearningSubmission,
  confirmPhotoReview,
  createPhotoReview,
  createLearningSession,
  createLearningSubmission,
  getSystemStatus,
  getLearningSubmission,
  getTargetedPractice,
  getDueReminders,
  generateLearningDiagram,
  requestLearningAnimation,
  loginWithWechat,
  recognizeSubmissionPhoto,
  getParentLearningDeposit,
  getParentLearningMemory,
  getParentReviewPlan,
  getParentSafetyEvents,
  getParentSessionFeedback,
  getParentSummaryDraft,
  getParentWeeklyReport,
  getParentWrongQuestions,
  getSessionChildId,
  listChildren,
  listLearningSessions,
  resumeLearningSession,
  startNextSubmissionTutorItem,
  submitPracticeResult,
  submitSubmissionTutorAttempt,
  submitLearningAttempt,
  rollupParentSummary,
  saveReminderSubscription,
  upsertChild,
};
