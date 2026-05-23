const {
  confirmLearningSubmission,
  createLearningSubmission,
  getSessionChildId,
  runLearningSubmissionVisualFallbackStep,
  startNextSubmissionTutorItem,
} = require("../../lib/api");
const { formatUserFacingError } = require("../../lib/errors");

const PENDING_SUBMISSION_DRAFT_KEY = "songguo_pending_submission_draft";

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
  chinese_reading_summary: "语文阅读题",
  english_sentence_pattern: "英语语法题",
  general_learning_strategy: "学习问题",
  grade_math_unknown: "数学题",
  grammar_fix: "英语语法题",
  reading_comprehension: "语文阅读题",
  unit_conversion_arithmetic: "单位换算",
  word_problem: "应用题",
};

const SUBJECT_LABELS = {
  math: "数学",
  chinese: "语文",
  english: "英语",
  unknown: "待确认",
};

function buildTypeText(item) {
  const raw = item.knowledge_point || item.question_type_id || "";
  if (!raw) {
    return "题型待确认";
  }
  const label = QUESTION_TYPE_LABELS[raw] || raw;
  return /^[a-z0-9_:-]+$/i.test(label) ? "数学题" : label;
}

function compactTextList(values) {
  return (Array.isArray(values) ? values : [])
    .map((value) => trimInput(value))
    .filter(Boolean)
    .slice(0, 4);
}

function buildItemViewModel(item) {
  const displayStatus = item.display_status || markerStatusFromItem(item);
  const isCorrect = displayStatus === "correct";
  const isWrong = displayStatus === "wrong";
  const fallbackStatus = item.visual_fallback_status || "";
  const isFallbackActive = displayStatus === "fallback_running" || fallbackStatus === "pending" || fallbackStatus === "running";
  const isFallbackManual = fallbackStatus === "failed" || fallbackStatus === "needs_manual_confirm";
  const correctAnswer = trimInput(item.correct_answer || "");
  return {
    ...item,
    display_status: displayStatus,
    resultLabel: isFallbackActive ? "复核中" : isCorrect ? "正确" : isWrong ? "需要陪练" : isFallbackManual ? "需确认" : "待确认",
    resultClass: isCorrect ? "result-correct" : isWrong ? "result-wrong" : isFallbackActive ? "result-running" : "result-pending",
    answerText: item.child_answer || "未识别到孩子答案",
    correctAnswerText: isWrong && correctAnswer ? `参考答案：${correctAnswer}` : "",
    rubricFeedback: item.rubric_feedback || "",
    visualFallbackMessage: item.visual_fallback_message || "",
    typeText: buildTypeText(item),
    evidencePoints: compactTextList(item.evidence_points),
  };
}

function readPendingSubmissionDraft() {
  try {
    return wx.getStorageSync(PENDING_SUBMISSION_DRAFT_KEY) || null;
  } catch (_error) {
    return null;
  }
}

function clearPendingSubmissionDraft() {
  try {
    wx.removeStorageSync(PENDING_SUBMISSION_DRAFT_KEY);
  } catch (_error) {
    // 清理临时识别草稿失败不影响提交。
  }
}

function confidenceText(value) {
  const confidence = Number(value || 0);
  if (!confidence) {
    return "";
  }
  return `${Math.round(confidence * 100)}%`;
}

function recognitionSourceText(sourceType) {
  if (sourceType === "photo") {
    return "照片识别";
  }
  if (sourceType === "voice") {
    return "语音识别";
  }
  return "";
}

function routeSubjectText(payload) {
  const subject = (payload && (payload.detected_subject || payload.subject)) || "";
  const label = SUBJECT_LABELS[subject] || subject || "待确认";
  const confidence = confidenceText(payload && payload.subject_confidence);
  return confidence ? `${label} · ${confidence}` : label;
}

function routeEvidencePoints(payload) {
  const evidence = compactTextList(payload && payload.routing_evidence);
  if (payload && payload.guard_reason) {
    evidence.push(`需要人工确认：${payload.guard_reason}`);
  }
  return evidence.slice(0, 4);
}

function qualityWarningText(warnings, qualityMessage) {
  const serverMessage = trimInput(qualityMessage);
  if (serverMessage) {
    return serverMessage;
  }
  const values = Array.isArray(warnings) ? warnings : [];
  if (!values.length) {
    return "";
  }
  const labels = values.map((warning) => {
    if (warning === "photo_decode_failed") {
      return "照片没有读取成功";
    }
    if (warning === "retake_required") {
      return "建议重拍";
    }
    if (warning === "no_homework_content") {
      return "没有清楚识别到题目内容";
    }
    if (warning === "no_structured_items") {
      return "没有拆出清晰题目和答案";
    }
    if (warning === "small_image") {
      return "照片分辨率偏低";
    }
    if (warning === "low_contrast") {
      return "字迹和背景对比度偏低";
    }
    if (warning === "blurry_image") {
      return "照片可能没有对焦";
    }
    if (warning === "glare_or_overexposed_area") {
      return "照片有反光或过曝";
    }
    if (warning === "photo_not_level") {
      return "照片角度不够平，系统已自动扶正";
    }
    if (warning === "ocr_provider_error") {
      return "OCR 服务暂时不可用";
    }
    if (warning === "ocr_service_expired") {
      return "OCR 套餐或服务状态异常";
    }
    return "照片质量需要复核";
  });
  return `${Array.from(new Set(labels)).join("，")}。请先核对识别内容；如果题目或答案缺失，建议重拍。`;
}

function clampPercent(value) {
  const number = Number(value || 0);
  return Math.max(0, Math.min(100, number));
}

function normalizeBbox(bbox) {
  if (!bbox) {
    return null;
  }
  const x = clampPercent(Number(bbox.x || 0) / 10);
  const y = clampPercent(Number(bbox.y || 0) / 10);
  const width = clampPercent(Number(bbox.width || 0) / 10);
  const height = clampPercent(Number(bbox.height || 0) / 10);
  if (!width || !height) {
    return null;
  }
  return {
    x,
    y,
    width: Math.min(width, 100 - x),
    height: Math.min(height, 100 - y),
  };
}

function bboxStyle(bbox) {
  const normalized = normalizeBbox(bbox);
  if (!normalized) {
    return "";
  }
  return [
    `left:${normalized.x}%`,
    `top:${normalized.y}%`,
    `width:${normalized.width}%`,
    `height:${normalized.height}%`,
  ].join(";");
}

function markerStatusFromItem(item) {
  const fallbackStatus = item && item.visual_fallback_status;
  if (fallbackStatus === "pending" || fallbackStatus === "running") {
    return "fallback_running";
  }
  if (item && item.judge_result === "correct") {
    return "correct";
  }
  if (item && item.judge_result === "wrong") {
    return "wrong";
  }
  return "pending";
}

function markerForStatus(status) {
  if (status === "correct") {
    return { label: "✓", className: "overlay-correct", text: "正确" };
  }
  if (status === "wrong") {
    return { label: "×", className: "overlay-wrong", text: "需陪练" };
  }
  if (status === "fallback_running") {
    return { label: "…", className: "overlay-running", text: "复核中" };
  }
  return { label: "?", className: "overlay-pending", text: "待确认" };
}

function overlayState(item) {
  return markerForStatus((item && item.display_status) || markerStatusFromItem(item));
}

function buildOverlayItems(resultItems, draftItems) {
  const draftsByIndex = {};
  (draftItems || []).forEach((item) => {
    draftsByIndex[Number(item.item_index || 0)] = item;
  });
  const hasResultItems = Boolean(resultItems && resultItems.length);
  const sourceItems = hasResultItems ? resultItems : draftItems || [];
  return sourceItems
    .map((item, index) => {
      const itemIndex = Number(item.item_index || index + 1);
      const draftItem = draftsByIndex[itemIndex] || {};
      const style = bboxStyle(hasResultItems ? item.bbox : draftItem.bbox);
      if (!style) {
        return null;
      }
      const state = overlayState(item);
      return {
        itemIndex,
        bboxStyle: style,
        markerLabel: state.label,
        markerClass: state.className,
        markerText: state.text,
        questionText: item.question_text || draftItem.question_text || `第 ${itemIndex} 题`,
      };
    })
    .filter(Boolean);
}

function buildSubmissionPageStatus(payload) {
  if (
    payload &&
    (payload.visual_fallback_active || Number(payload.pending_visual_fallback_count || 0) > 0)
  ) {
    return "judged_summary";
  }
  return payload && payload.wrong_count ? "judged_summary" : "completed";
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
    pendingVisualFallbackCount: 0,
    visualFallbackActive: false,
    visualFallbackPolling: false,
    fallbackStatusText: "",
    loading: false,
    autoJudging: false,
    error: "",
    canSubmit: false,
    confirmed: false,
    recognitionSummaryVisible: false,
    recognitionSourceText: "",
    recognitionConfidenceText: "",
    qualityWarningText: "",
    recognizedItemCount: 0,
    recognizedItemCountText: "1",
    photoImagePath: "",
    pendingDraftItems: [],
    pendingImageRefs: [],
    overlayItems: [],
    routeSummaryVisible: false,
    routeSubjectText: "",
    routeEvidencePoints: [],
  },

  onLoad(options) {
    const sourceType = decodeURIComponent((options && options.sourceType) || "text");
    const pendingDraft = options && options.pendingDraft === "1" ? readPendingSubmissionDraft() : null;
    const childId =
      (pendingDraft && pendingDraft.childId) ||
      decodeURIComponent((options && options.childId) || "") ||
      getSessionChildId();
    const initialText =
      (pendingDraft && pendingDraft.initialText) ||
      decodeURIComponent((options && options.initialText) || "");
    const recognizedItemCount = Number((pendingDraft && pendingDraft.itemCount) || 0);
    const recognitionConfidenceText = confidenceText(pendingDraft && pendingDraft.confidence);
    const qualityMessage = (pendingDraft && pendingDraft.qualityMessage) || "";
    const photoQualityWarningText = qualityWarningText(
      pendingDraft && pendingDraft.qualityWarnings,
      qualityMessage
    );
    const pendingDraftItems = (pendingDraft && pendingDraft.items) || [];
    const pendingImageRefs = (pendingDraft && pendingDraft.imageRefs) || [];
    const photoImagePath =
      (pendingDraft && (pendingDraft.previewImageUrl || pendingDraft.localImagePath)) || "";
    const shouldAutoJudge =
      sourceType === "photo" &&
      Boolean(pendingDraft) &&
      Boolean(trimInput(initialText)) &&
      Number((pendingDraft && pendingDraft.confidence) || 0) >= 0.8 &&
      pendingDraftItems.length > 0;
    this.setData({
      sourceType,
      childId,
      sourceLabel: buildSourceLabel(sourceType),
      inputValue: initialText,
      canSubmit: Boolean(trimInput(initialText)),
      recognitionSummaryVisible: Boolean(
        pendingDraft && sourceType !== "text" && (recognizedItemCount || recognitionConfidenceText)
      ),
      recognitionSourceText: recognitionSourceText(sourceType),
      recognitionConfidenceText,
      qualityWarningText: photoQualityWarningText,
      recognizedItemCount,
      recognizedItemCountText: String(recognizedItemCount || 1),
      photoImagePath,
      pendingDraftItems,
      pendingImageRefs,
      overlayItems: buildOverlayItems([], pendingDraftItems),
    });
    if (shouldAutoJudge) {
      this.handleAutoJudgePhotoDraft();
    }
  },

  onUnload() {
    if (this._fallbackPollTimer) {
      clearTimeout(this._fallbackPollTimer);
      this._fallbackPollTimer = null;
    }
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
        subject: "auto",
        grade: 3,
        sourceType: this.data.sourceType,
        rawText,
        imageRefs: this.data.pendingImageRefs,
        draftItems: this.data.pendingDraftItems,
      });
      clearPendingSubmissionDraft();
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

  scheduleVisualFallbackPolling(force) {
    if (
      !this.data.submissionId ||
      this.data.visualFallbackPolling ||
      (!force && !this.data.visualFallbackActive)
    ) {
      return;
    }
    this.setData({ visualFallbackPolling: true });
    this.runVisualFallbackPolling(0);
  },

  async runVisualFallbackPolling(attempt) {
    if (!this.data.submissionId) {
      this.setData({ visualFallbackPolling: false });
      return;
    }
    if (attempt >= 8) {
      this.setData({
        visualFallbackPolling: false,
        fallbackStatusText: "还有题目需要手动确认，可以先查看已出的结果。",
      });
      return;
    }
    try {
      const payload = await runLearningSubmissionVisualFallbackStep(this.data.submissionId, 1);
      this.applySubmission(payload, {
        status: buildSubmissionPageStatus(payload),
        confirmed: true,
        skipVisualFallbackPolling: true,
      });
      if (payload.visual_fallback_active) {
        this._fallbackPollTimer = setTimeout(() => {
          this.runVisualFallbackPolling(attempt + 1);
        }, 900);
        return;
      }
      this.setData({
        visualFallbackPolling: false,
        fallbackStatusText: "",
      });
    } catch (error) {
      this.setData({
        visualFallbackPolling: false,
        fallbackStatusText: "有题目暂时没有复核成功，可以先看已出的结果。",
        error: formatUserFacingError(error),
      });
    }
  },

  async handleAutoJudgePhotoDraft() {
    const rawText = trimInput(this.data.inputValue);
    if (!rawText || this.data.autoJudging || this.data.loading) {
      return;
    }
    this.setData({ loading: true, autoJudging: true, error: "" });
    try {
      const created = await createLearningSubmission({
        childId: this.data.childId,
        subject: "auto",
        grade: 3,
        sourceType: this.data.sourceType,
        rawText,
        imageRefs: this.data.pendingImageRefs,
        draftItems: this.data.pendingDraftItems,
      });
      this.applySubmission(created, {
        rawText,
        status: "review",
        confirmed: false,
      });
      const confirmed = await confirmLearningSubmission(created.submission_id, {
        rawText,
        startTutor: false,
      });
      clearPendingSubmissionDraft();
      this.applySubmission(confirmed, {
        status: buildSubmissionPageStatus(confirmed),
        confirmed: true,
      });
    } catch (error) {
      this.setData({ error: formatUserFacingError(error) });
    } finally {
      this.setData({ loading: false, autoJudging: false });
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
        status: buildSubmissionPageStatus(payload),
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
    const items = (payload.items || []).map(buildItemViewModel);
    const pendingVisualFallbackCount = Number(payload.pending_visual_fallback_count || 0);
    const visualFallbackActive = Boolean(payload.visual_fallback_active);
    const routePoints = routeEvidencePoints(payload);
    this.setData({
      submissionId: payload.submission_id || this.data.submissionId,
      rawText,
      status,
      confirmed: Boolean(options && options.confirmed),
      items,
      overlayItems: buildOverlayItems(items, this.data.pendingDraftItems),
      tutorQueue: payload.tutor_queue || [],
      activeTutorSession: payload.active_tutor_session || null,
      itemCount: payload.item_count || 0,
      correctCount: payload.correct_count || 0,
      wrongCount: payload.wrong_count || 0,
      needsManualConfirmCount: payload.needs_manual_confirm_count || 0,
      pendingVisualFallbackCount,
      visualFallbackActive,
      fallbackStatusText: visualFallbackActive
        ? `正在复核 ${pendingVisualFallbackCount || 1} 题，其他结果先显示。`
        : "",
      routeSummaryVisible: Boolean(payload.detected_subject || payload.route_to || routePoints.length),
      routeSubjectText: routeSubjectText(payload),
      routeEvidencePoints: routePoints,
    });
    if (
      Boolean(options && options.confirmed) &&
      visualFallbackActive &&
      !(options && options.skipVisualFallbackPolling)
    ) {
      this.scheduleVisualFallbackPolling(true);
    }
  },
});
