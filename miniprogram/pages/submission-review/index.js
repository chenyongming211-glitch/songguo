const {
  confirmLearningSubmission,
  createLearningSubmission,
  getSessionChildId,
  startNextSubmissionTutorItem,
} = require("../../lib/api");
const { formatUserFacingError } = require("../../lib/errors");

function trimInput(value) {
  return String(value || "").trim();
}

function buildSourceLabel(sourceType) {
  if (sourceType === "photo") {
    return "拍照提交";
  }
  if (sourceType === "voice") {
    return "语音提交";
  }
  return "输入提交";
}

const QUESTION_TYPE_LABELS = {
  application: "应用题",
  arithmetic: "计算题",
  comparison: "比较问题",
  comparison_problem: "比较问题",
  grade_math_unknown: "数学题",
  unit_conversion_arithmetic: "单位换算",
  word_problem: "应用题",
};

function buildTypeText(item) {
  const raw = item.knowledge_point || item.question_type_id || "";
  if (!raw) {
    return "题型待确认";
  }
  const label = QUESTION_TYPE_LABELS[raw] || raw;
  return /^[a-z0-9_:-]+$/i.test(label) ? "数学题" : label;
}

function buildItemViewModel(item) {
  const isCorrect = item.judge_result === "correct";
  const isWrong = item.judge_result === "wrong";
  return {
    ...item,
    resultLabel: isCorrect ? "正确" : isWrong ? "需要陪练" : "待确认",
    resultClass: isCorrect ? "result-correct" : isWrong ? "result-wrong" : "result-pending",
    answerText: item.child_answer || "未识别到孩子答案",
    typeText: buildTypeText(item),
  };
}

Page({
  data: {
    childId: getSessionChildId(),
    sourceType: "text",
    sourceLabel: "输入提交",
    rawText: "",
    inputValue: "",
    submissionId: "",
    status: "intake",
    items: [],
    tutorQueue: [],
    activeTutorSession: null,
    itemCount: 0,
    correctCount: 0,
    wrongCount: 0,
    needsManualConfirmCount: 0,
    loading: false,
    error: "",
    canSubmit: false,
    confirmed: false,
  },

  onLoad(options) {
    const sourceType = decodeURIComponent((options && options.sourceType) || "text");
    const childId = decodeURIComponent((options && options.childId) || "") || getSessionChildId();
    const initialText = decodeURIComponent((options && options.initialText) || "");
    this.setData({
      sourceType,
      childId,
      sourceLabel: buildSourceLabel(sourceType),
      inputValue: initialText,
      canSubmit: Boolean(trimInput(initialText)),
    });
  },

  handleInput(event) {
    const inputValue = event.detail.value || "";
    this.setData({
      inputValue,
      canSubmit: Boolean(trimInput(inputValue)),
    });
  },

  async handleCreateSubmission() {
    const rawText = trimInput(this.data.inputValue);
    if (!rawText || this.data.loading) {
      return;
    }
    this.setData({ loading: true, error: "" });
    try {
      const payload = await createLearningSubmission({
        childId: this.data.childId,
        subject: "math",
        grade: 3,
        sourceType: this.data.sourceType,
        rawText,
      });
      this.applySubmission(payload, {
        rawText,
        status: "review",
        confirmed: false,
      });
    } catch (error) {
      this.setData({ error: formatUserFacingError(error) });
    } finally {
      this.setData({ loading: false });
    }
  },

  async handleConfirmSubmission() {
    if (!this.data.submissionId || this.data.loading) {
      return;
    }
    this.setData({ loading: true, error: "" });
    try {
      const payload = await confirmLearningSubmission(this.data.submissionId, {
        rawText: this.data.rawText,
      });
      this.applySubmission(payload, {
        status: payload.wrong_count ? "judged_summary" : "completed",
        confirmed: true,
      });
    } catch (error) {
      this.setData({ error: formatUserFacingError(error) });
    } finally {
      this.setData({ loading: false });
    }
  },

  async handleStartWrongTutor() {
    if (!this.data.submissionId || !this.data.wrongCount) {
      return;
    }
    this.setData({ loading: true, error: "" });
    try {
      const payload = await startNextSubmissionTutorItem(this.data.submissionId);
      this.applySubmission(payload, {
        status: "tutoring",
        confirmed: true,
      });
      wx.navigateTo({
        url: `/pages/chat-detail/index?submissionId=${encodeURIComponent(
          this.data.submissionId
        )}&childId=${encodeURIComponent(this.data.childId)}`,
      });
    } catch (error) {
      this.setData({ error: formatUserFacingError(error) });
    } finally {
      this.setData({ loading: false });
    }
  },

  applySubmission(payload, options) {
    const status = (options && options.status) || this.data.status;
    const rawText = (options && options.rawText) || this.data.rawText;
    this.setData({
      submissionId: payload.submission_id || this.data.submissionId,
      rawText,
      status,
      confirmed: Boolean(options && options.confirmed),
      items: (payload.items || []).map(buildItemViewModel),
      tutorQueue: payload.tutor_queue || [],
      activeTutorSession: payload.active_tutor_session || null,
      itemCount: payload.item_count || 0,
      correctCount: payload.correct_count || 0,
      wrongCount: payload.wrong_count || 0,
      needsManualConfirmCount: payload.needs_manual_confirm_count || 0,
    });
  },
});
