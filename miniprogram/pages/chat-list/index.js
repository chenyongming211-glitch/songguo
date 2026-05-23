const {
  getSessionChildId,
  listChildren,
  listLearningSessions,
  recognizeSubmissionPhoto,
  recognizeSubmissionVoice,
  upsertChild,
} = require("../../lib/api");
const { getBackendConfig } = require("../../lib/config");
const { formatUserFacingError } = require("../../lib/errors");
const { getSubjectModeLabel } = require("../../lib/learning-copy");
const { formatTimestamp, shortenText } = require("../../lib/utils");

const PENDING_SUBMISSION_DRAFT_KEY = "songguo_pending_submission_draft";

function getChildGradeText(child) {
  if (!child) {
    return "3 年级";
  }
  if (child.grade) {
    return `${child.grade} 年级`;
  }
  return child.term_label || "当前学期";
}

function toSessionViewModel(item) {
  const progress = item.teaching_progress || {};
  return {
    ...item,
    capabilityLabel: getSubjectModeLabel(item.subject),
    message_count: item.attempt_count || 0,
    progressLabel: progress.current_label || "动态引导",
    preview: shortenText(item.current_prompt || "还没有提示", 92),
    updatedLabel: formatTimestamp(item.updated_at) || "刚刚",
  };
}

function chooseHomeworkImage() {
  if (wx.navigateTo && wx.createCameraContext) {
    return chooseHomeworkImageWithCameraGuide();
  }
  return chooseHomeworkImageNative();
}

function chooseHomeworkImageWithCameraGuide() {
  return new Promise((resolve, reject) => {
    wx.navigateTo({
      url: "/pages/homework-camera/index",
      success: (response) => {
        let settled = false;
        const settle = (callback, value) => {
          if (settled) {
            return;
          }
          settled = true;
          callback(value);
        };
        const channel = response.eventChannel;
        channel.on("capturedHomeworkImage", (payload) => {
          const filePath = payload && payload.tempFilePath;
          if (filePath) {
            settle(resolve, filePath);
            return;
          }
          settle(reject, new Error("没有选择图片"));
        });
        channel.on("homeworkCameraClosed", () => {
          settle(reject, new Error("没有选择图片"));
        });
      },
      fail: () => {
        chooseHomeworkImageNative().then(resolve).catch(reject);
      },
    });
  });
}

function chooseHomeworkImageNative() {
  return new Promise((resolve, reject) => {
    if (wx.chooseMedia) {
      wx.chooseMedia({
        count: 1,
        mediaType: ["image"],
        sourceType: ["camera", "album"],
        success: (response) => {
          const file = response.tempFiles && response.tempFiles[0];
          if (file && file.tempFilePath) {
            resolve(file.tempFilePath);
            return;
          }
          reject(new Error("没有选择图片"));
        },
        fail: (error) => reject(new Error((error && error.errMsg) || "选择图片失败")),
      });
      return;
    }

    wx.chooseImage({
      count: 1,
      sizeType: ["compressed"],
      sourceType: ["camera", "album"],
      success: (response) => {
        const filePath = response.tempFilePaths && response.tempFilePaths[0];
        if (filePath) {
          resolve(filePath);
          return;
        }
        reject(new Error("没有选择图片"));
      },
      fail: (error) => reject(new Error((error && error.errMsg) || "选择图片失败")),
    });
  });
}

function compressHomeworkImage(filePath) {
  return new Promise((resolve) => {
    if (!wx.compressImage || !filePath) {
      resolve(filePath);
      return;
    }
    wx.compressImage({
      src: filePath,
      quality: 68,
      success: (response) => {
        resolve((response && response.tempFilePath) || filePath);
      },
      fail: () => resolve(filePath),
    });
  });
}

function recordHomeworkVoice() {
  return new Promise((resolve, reject) => {
    if (!wx.getRecorderManager) {
      reject(new Error("当前微信环境不支持录音"));
      return;
    }
    const recorder = wx.getRecorderManager();
    let settled = false;
    recorder.onStop((response) => {
      if (settled) {
        return;
      }
      settled = true;
      if (response && response.tempFilePath) {
        resolve(response.tempFilePath);
        return;
      }
      reject(new Error("没有录到语音"));
    });
    recorder.onError((error) => {
      if (settled) {
        return;
      }
      settled = true;
      reject(new Error((error && error.errMsg) || "录音失败"));
    });
    wx.showToast({
      title: "开始录音，请说题目和答案",
      icon: "none",
    });
    recorder.start({
      duration: 10000,
      sampleRate: 16000,
      numberOfChannels: 1,
      encodeBitRate: 48000,
      format: "mp3",
    });
  });
}

function buildRecognizedSubmissionText(review) {
  const rawText = String((review && review.raw_text) || "").trim();
  if (rawText) {
    return rawText;
  }
  const questionText = String((review && review.question_text) || "").trim();
  const childAnswer = String((review && review.child_answer) || "").trim();
  const workSteps = String((review && review.work_steps) || "").trim();
  return [
    questionText,
    childAnswer ? `孩子答案：${childAnswer}` : "",
    workSteps ? `解题过程：${workSteps}` : "",
  ]
    .filter(Boolean)
    .join("\n");
}

function buildPreviewImageUrl(draft) {
  const previewUrl = String((draft && draft.preview_image_url) || "").trim();
  if (!previewUrl) {
    return "";
  }
  if (/^https?:\/\//i.test(previewUrl)) {
    return previewUrl;
  }
  const { httpBaseUrl } = getBackendConfig();
  const separator = previewUrl.startsWith("/") ? "" : "/";
  return `${httpBaseUrl}${separator}${previewUrl}`;
}

function savePendingSubmissionDraft(sourceType, childId, draft, initialText, options) {
  const payload = {
    sourceType,
    childId,
    initialText,
    localImagePath: (options && options.localImagePath) || "",
    previewImageUrl: buildPreviewImageUrl(draft),
    confidence: Number((draft && draft.confidence) || 0),
    itemCount: Array.isArray(draft && draft.items) ? draft.items.length : 0,
    items: (draft && draft.items) || [],
    imageRefs: (draft && draft.image_refs) || [],
    detectedRegions: (draft && draft.detected_regions) || [],
    qualityWarnings: (draft && draft.quality_warnings) || [],
    qualityMessage: (draft && draft.quality_message) || "",
    preprocessSource: (draft && draft.preprocess_source) || "",
    ocrProvider: (draft && draft.ocr_provider) || "",
    ocrModel: (draft && draft.ocr_model) || "",
    createdAt: Date.now(),
  };
  try {
    wx.setStorageSync(PENDING_SUBMISSION_DRAFT_KEY, payload);
    return true;
  } catch (_error) {
    return false;
  }
}

Page({
  data: {
    backendBaseUrl: "",
    sessions: [],
    loading: false,
    error: "",
    childId: getSessionChildId(),
    selectedChildName: "默认孩子",
    selectedChildGradeText: "3 年级",
    children: [],
    recognizingPhoto: false,
    recognizingVoice: false,
  },

  onShow() {
    const { httpBaseUrl } = getBackendConfig();
    this.setData({ backendBaseUrl: httpBaseUrl });
    this.ensureChildrenAndLoad();
  },

  async ensureChildrenAndLoad() {
    this.setData({ loading: true, error: "" });
    try {
      let payload = await listChildren();
      let children = payload.children || [];
      if (!children.length) {
        const created = await upsertChild({
          childId: getSessionChildId(),
          name: "默认孩子",
          grade: 3,
          termLabel: "当前学期",
        });
        children = [created];
      }
      const selected = children.find((item) => item.child_id === this.data.childId) || children[0];
      this.setData({
        children,
        childId: selected.child_id,
        selectedChildName: selected.name || "默认孩子",
        selectedChildGradeText: getChildGradeText(selected),
      });
      wx.setStorageSync("songguo_selected_child_id", selected.child_id);
      await this.loadSessions();
    } catch (error) {
      this.setData({
        error: formatUserFacingError(error),
      });
    } finally {
      this.setData({ loading: false });
    }
  },

  async loadSessions() {
    this.setData({ loading: true, error: "" });
    try {
      const payload = await listLearningSessions(this.data.childId);
      const sessions = (payload.sessions || []).map(toSessionViewModel);
      this.setData({ sessions });
    } catch (error) {
      this.setData({
        error: formatUserFacingError(error),
      });
    } finally {
      this.setData({ loading: false });
      wx.stopPullDownRefresh();
    }
  },

  onPullDownRefresh() {
    this.loadSessions();
  },

  handleRefresh() {
    this.ensureChildrenAndLoad();
  },

  handleRetry() {
    this.ensureChildrenAndLoad();
  },

  handleOpenSettings() {
    wx.switchTab({
      url: "/pages/settings/index",
    });
  },

  handleCreateSession() {
    this.handleStartSubmission({ currentTarget: { dataset: { sourceType: "text" } } });
  },

  handleStartSubmission(event) {
    const sourceType = (event.currentTarget.dataset && event.currentTarget.dataset.sourceType) || "text";
    this.navigateToSubmissionReview(sourceType, "");
  },

  navigateToSubmissionReview(sourceType, initialText, options) {
    const hasDraft = options && options.usePendingDraft;
    wx.navigateTo({
      url: `/pages/submission-review/index?sourceType=${encodeURIComponent(
        sourceType
      )}&childId=${encodeURIComponent(this.data.childId)}&initialText=${encodeURIComponent(
        hasDraft ? "" : initialText || ""
      )}${hasDraft ? "&pendingDraft=1" : ""}`,
    });
  },

  async handleStartPhotoSubmission() {
    if (this.data.recognizingPhoto) {
      return;
    }
    this.setData({ recognizingPhoto: true, error: "" });
    try {
      const localImagePath = await chooseHomeworkImage();
      const uploadImagePath = await compressHomeworkImage(localImagePath);
      const review = await recognizeSubmissionPhoto(uploadImagePath, this.data.childId);
      const initialText = buildRecognizedSubmissionText(review);
      const usePendingDraft = savePendingSubmissionDraft(
        "photo",
        this.data.childId,
        review,
        initialText,
        { localImagePath }
      );
      if (!initialText) {
        wx.showToast({
          title: "没识别清楚，请手动确认",
          icon: "none",
        });
      }
      this.navigateToSubmissionReview("photo", initialText, { usePendingDraft });
    } catch (error) {
      this.setData({ error: formatUserFacingError(error) });
    } finally {
      this.setData({ recognizingPhoto: false });
    }
  },

  async handleStartVoiceSubmission() {
    if (this.data.recognizingVoice) {
      return;
    }
    this.setData({ recognizingVoice: true, error: "" });
    try {
      const filePath = await recordHomeworkVoice();
      const draft = await recognizeSubmissionVoice(filePath, this.data.childId);
      const initialText = String((draft && (draft.raw_text || draft.transcript)) || "").trim();
      const usePendingDraft = savePendingSubmissionDraft(
        "voice",
        this.data.childId,
        draft,
        initialText
      );
      if (!initialText) {
        wx.showToast({
          title: "没听清楚，请手动确认",
          icon: "none",
        });
      }
      this.navigateToSubmissionReview("voice", initialText, { usePendingDraft });
    } catch (error) {
      this.setData({ error: formatUserFacingError(error) });
    } finally {
      this.setData({ recognizingVoice: false });
    }
  },

  handleChildChange(event) {
    const childId = event.currentTarget.dataset.childId || getSessionChildId();
    if (childId === this.data.childId) {
      return;
    }
    const selected = this.data.children.find((item) => item.child_id === childId);
    this.setData({
      childId,
      selectedChildName: (selected && selected.name) || "默认孩子",
      selectedChildGradeText: getChildGradeText(selected),
    });
    wx.setStorageSync("songguo_selected_child_id", childId);
    this.loadSessions();
  },

  handleOpenSession(event) {
    const { sessionId } = event.currentTarget.dataset;
    if (!sessionId) {
      return;
    }
    wx.navigateTo({
      url: `/pages/chat-detail/index?sessionId=${encodeURIComponent(
        sessionId
      )}&questionText=${encodeURIComponent(event.currentTarget.dataset.questionText || "")}`,
    });
  },

  handleSessionActions() {
    wx.showToast({
      title: "学习会话暂不支持删除",
      icon: "none",
    });
  },
});
