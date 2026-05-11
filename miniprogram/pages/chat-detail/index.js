const {
  confirmPhotoReview,
  createPhotoReview,
  createLearningSession,
  getLearningSubmission,
  getSessionChildId,
  getTargetedPractice,
  listChildren,
  resumeLearningSession,
  startNextSubmissionTutorItem,
  submitLearningAttempt,
  submitSubmissionTutorAttempt,
  upsertChild,
} = require("../../lib/api");
const { formatUserFacingError } = require("../../lib/errors");
const { buildSubjectCopy } = require("../../lib/learning-copy");
const { buildStreamingFrames } = require("../../lib/streaming-text");
const { makeLocalMessage } = require("../../lib/utils");

const DEFAULT_SUBJECT_COPY = buildSubjectCopy({ subject: "math" });
const DEFAULT_TEACHING_PROGRESS = {
  mode: "not_started",
  phase_label: "准备开始",
  current_label: "等待输入题目",
  current_index: null,
  total_count: null,
  hint_level: 0,
  answer_policy_label: "答案锁定中",
  hint_policy_label: "提交题目后根据孩子回答继续追问",
};
const STREAM_CHUNK_SIZE = 8;
const STREAM_INTERVAL_MS = 28;

function trimInput(value) {
  return String(value || "").trim();
}

function parsePhotoConfirmation(value) {
  const text = trimInput(value);
  const questionMatch = text.match(/(?:题目|question)\s*[:：]\s*([^\n]+)/i);
  const answerMatch = text.match(/(?:答案|answer)\s*[:：]\s*([^\n]+)/i);
  if (questionMatch && answerMatch) {
    return {
      questionText: trimInput(questionMatch[1]),
      childAnswer: trimInput(answerMatch[1]),
    };
  }

  const parts = text.split("|").map((item) => trimInput(item));
  if (parts.length >= 2 && parts[0] && parts[1]) {
    return {
      questionText: parts[0],
      childAnswer: parts[1],
    };
  }

  return {
    questionText: "",
    childAnswer: "",
  };
}

function buildAssistantMessage(content) {
  return makeLocalMessage("assistant", content, Date.now());
}

function replaceMessageContent(messages, clientId, content) {
  return messages.map((message) =>
    message.clientId === clientId ? { ...message, content } : message
  );
}

function buildMessagesFromResume(payload) {
  const turns = (payload && payload.last_messages) || [];
  const messages = turns
    .filter((item) => item && item.content)
    .map((item, index) => makeLocalMessage(item.role || "assistant", item.content, Date.now() + index));
  if (messages.length) {
    return messages;
  }
  if (payload && payload.current_prompt) {
    return [buildAssistantMessage(payload.current_prompt)];
  }
  return [];
}

function normalizeTeachingProgress(payload) {
  const progress = (payload && payload.teaching_progress) || {};
  return {
    ...DEFAULT_TEACHING_PROGRESS,
    ...progress,
    phase_label: progress.phase_label || DEFAULT_TEACHING_PROGRESS.phase_label,
    current_label: progress.current_label || DEFAULT_TEACHING_PROGRESS.current_label,
    answer_policy_label:
      progress.answer_policy_label || DEFAULT_TEACHING_PROGRESS.answer_policy_label,
    hint_policy_label:
      progress.hint_policy_label || DEFAULT_TEACHING_PROGRESS.hint_policy_label,
  };
}

function buildChildGradeText(child) {
  if (!child) {
    return "未选择";
  }
  const gradeText = child.grade ? `${child.grade} 年级` : "年级未设";
  return child.term_label ? `${gradeText} · ${child.term_label}` : gradeText;
}

function chooseHomeworkImage() {
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

Page({
  data: {
    sessionId: "",
    questionText: "",
    sessionTitle: DEFAULT_SUBJECT_COPY.sessionTitle,
    introSubtitle: DEFAULT_SUBJECT_COPY.introSubtitle,
    messages: [],
    inputValue: "",
    inputPlaceholder: DEFAULT_SUBJECT_COPY.inputPlaceholder,
    composerLabel: DEFAULT_SUBJECT_COPY.composerLabel,
    emptyTitle: DEFAULT_SUBJECT_COPY.emptyTitle,
    emptyText: DEFAULT_SUBJECT_COPY.emptyText,
    startButtonLabel: DEFAULT_SUBJECT_COPY.startButtonLabel,
    canSend: false,
    loadingHistory: false,
    sending: false,
    uploadingPhoto: false,
    error: "",
    statusNote: DEFAULT_SUBJECT_COPY.statusNote,
    scrollTarget: "",
    hintLevel: 0,
    attemptCount: 0,
    answerUnlocked: false,
    teachingProgress: DEFAULT_TEACHING_PROGRESS,
    phase: "",
    submissionId: "",
    submissionStatus: "",
    activeQueueItemId: "",
    tutorComposerDisabled: false,
    childId: getSessionChildId(),
    selectedChildName: "默认孩子",
    selectedChildGradeText: "3 年级",
    children: [],
    pendingPhotoReviewId: "",
    selectedSubject: "math",
  },

  onLoad(options) {
    const sessionId = decodeURIComponent((options && options.sessionId) || "");
    const submissionId = decodeURIComponent((options && options.submissionId) || "");
    const selectedSubject = decodeURIComponent((options && options.subject) || "math");
    const questionText = decodeURIComponent((options && options.questionText) || "");
    const subjectCopy = buildSubjectCopy({ subject: selectedSubject, sessionId });
    this.setData({
      sessionId,
      submissionId,
      questionText,
      selectedSubject: subjectCopy.subject,
      childId:
        decodeURIComponent((options && options.childId) || "") ||
        wx.getStorageSync("songguo_selected_child_id") ||
        getSessionChildId(),
      ...subjectCopy,
    });
  },

  onShow() {
    this.ensureChildren();
    if (this.data.submissionId) {
      this.loadSubmissionTutor(this.data.submissionId);
      return;
    }
    if (this.data.sessionId) {
      this.resumeSession(this.data.sessionId);
    }
  },

  async ensureChildren() {
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
        selectedChildGradeText: buildChildGradeText(selected),
      });
      wx.setStorageSync("songguo_selected_child_id", selected.child_id);
    } catch (error) {
      this.setData({
        error: error.message || "加载孩子档案失败",
      });
    }
  },

  async resumeSession(sessionId) {
    this.setData({
      loadingHistory: true,
      error: "",
      statusNote: "正在恢复学习状态...",
    });
    try {
      const payload = await resumeLearningSession(sessionId);
      const restoredMessages = buildMessagesFromResume(payload);
      const sessionSubject = payload.subject || this.data.selectedSubject;
      const sessionCompleted = payload.phase === "SESSION_SUMMARY";
      const sessionLearningPaused = payload.phase === "LEARNING_PAUSED";
      const sessionPracticePaused = payload.phase === "PRACTICE_PAUSED";
      const subjectCopy = buildSubjectCopy({
        subject: sessionSubject,
        sessionId: payload.session_id || sessionId,
      });
      this.setData({
        sessionId: payload.session_id || sessionId,
        questionText: payload.question_text || this.data.questionText,
        selectedSubject: subjectCopy.subject,
        messages: restoredMessages,
        phase: payload.phase || "",
        hintLevel: payload.hint_level || 0,
        attemptCount: payload.attempt_count || 0,
        answerUnlocked: Boolean(payload.answer_unlocked),
        teachingProgress: normalizeTeachingProgress(payload),
        scrollTarget: restoredMessages.length ? restoredMessages[restoredMessages.length - 1].clientId : "",
        ...subjectCopy,
        tutorComposerDisabled: sessionCompleted,
        inputValue: sessionCompleted ? "" : this.data.inputValue,
        canSend: sessionCompleted ? false : this.data.canSend,
        inputPlaceholder: sessionCompleted
          ? "本次学习已结束，可以返回学习列表。"
          : subjectCopy.inputPlaceholder,
        statusNote: sessionCompleted
          ? "本次总结"
          : sessionLearningPaused
            ? "学习已暂停"
          : sessionPracticePaused
            ? "同类练习已暂停"
            : "学习状态已恢复",
      });
    } catch (error) {
      const message = formatUserFacingError(error, { action: "resume" });
      this.setData({
        sessionId: /不存在|失效/.test(message) ? "" : this.data.sessionId,
        error: message,
        statusNote: /不存在|失效/.test(message) ? "请返回学习列表重新开始" : "恢复失败",
      });
    } finally {
      this.setData({ loadingHistory: false });
    }
  },

  handleInput(event) {
    const inputValue = event.detail.value || "";
    this.setData({
      inputValue,
      canSend: !this.data.tutorComposerDisabled && Boolean(trimInput(inputValue)),
    });
  },

  handleChildChange(event) {
    if (this.data.sessionId) {
      this.setData({ statusNote: "当前会话已绑定孩子，结束后可切换" });
      return;
    }
    const childId = event.currentTarget.dataset.childId || getSessionChildId();
    if (childId === this.data.childId) {
      return;
    }
    const selected = this.data.children.find((item) => item.child_id === childId);
    this.setData({
      childId,
      selectedChildName: (selected && selected.name) || "默认孩子",
      selectedChildGradeText: buildChildGradeText(selected),
    });
    wx.setStorageSync("songguo_selected_child_id", childId);
  },

  async handleSend() {
    const content = trimInput(this.data.inputValue);
    if (!content || this.data.sending || this.data.tutorComposerDisabled) {
      return;
    }

    const userMessage = makeLocalMessage("user", content, Date.now());
    this.setData({
      messages: this.data.messages.concat([userMessage]),
      inputValue: "",
      canSend: false,
      sending: true,
      error: "",
      statusNote: this.data.sessionId ? "正在判断孩子尝试..." : "正在创建学习会话...",
      scrollTarget: userMessage.clientId,
    });

    try {
      if (this.data.pendingPhotoReviewId) {
        await this.confirmPhotoReviewFromInput(content);
        return;
      }
      if (this.data.submissionId) {
        await this.submitSubmissionTutorAnswer(content);
        return;
      }
      if (!this.data.sessionId) {
        await this.createSession(content);
        return;
      }
      await this.submitAttempt(content);
    } catch (error) {
      await this.handleFailure(error);
    } finally {
      this.setData({ sending: false });
    }
  },

  async handleChoosePhoto() {
    if (this.data.sending || this.data.uploadingPhoto) {
      return;
    }

    try {
      const filePath = await chooseHomeworkImage();
      const userMessage = makeLocalMessage("user", "[图片] 已上传一张作业照片", Date.now());
      this.setData({
        messages: this.data.messages.concat([userMessage]),
        uploadingPhoto: true,
        error: "",
        statusNote: "正在识别并批改作业照片...",
        scrollTarget: userMessage.clientId,
      });
      const review = await createPhotoReview(
        filePath,
        this.data.childId,
        this.data.selectedSubject
      );
      await this.applyPhotoReviewResponse(review);
    } catch (error) {
      await this.handleFailure(error);
    }
  },

  async createSession(questionText) {
    const payload = await createLearningSession({
      childId: this.data.childId,
      subject: this.data.selectedSubject,
      grade: 3,
      questionText,
    });
    await this.applyLearningResponse(payload, "学习会话已创建", { questionText });
  },

  async loadSubmissionTutor(submissionId) {
    this.setData({
      loadingHistory: true,
      error: "",
      statusNote: "正在进入错题队列...",
    });
    try {
      let payload = await getLearningSubmission(submissionId, this.data.childId);
      if ((payload.tutor_queue || []).length && !payload.active_tutor_session) {
        payload = await startNextSubmissionTutorItem(submissionId);
      }
      await this.applySubmissionTutorSnapshot(payload, {
        streamPrompt: true,
        statusNote: "正在讲错题",
      });
    } catch (error) {
      const message = formatUserFacingError(error);
      this.setData({
        error: message,
        statusNote: "错题队列加载失败",
      });
    } finally {
      this.setData({ loadingHistory: false });
    }
  },

  async submitSubmissionTutorAnswer(childAnswer) {
    const payload = await submitSubmissionTutorAttempt(this.data.submissionId, childAnswer);
    const attempt = payload.attempt || {};
    const submission = payload.submission || {};
    if (attempt.message) {
      await this.streamAssistantMessage(attempt.message, {
        statusNote: attempt.correct ? "这道错题已完成" : "已生成下一步提示",
      });
    }
    const nextStatusNote =
      submission.status === "completed"
        ? "本次错题已讲完"
        : attempt.correct
          ? "继续下一道错题"
          : "继续这道错题";
    await this.applySubmissionTutorSnapshot(submission, {
      streamPrompt: attempt.correct,
      statusNote: nextStatusNote,
    });
  },

  async submitAttempt(childAnswer) {
    const wasInSimilarPractice =
      this.data.phase === "SIMILAR_PRACTICE" && this.data.answerUnlocked;
    const payload = await submitLearningAttempt(this.data.sessionId, childAnswer);
    const statusNote = payload.phase === "SESSION_SUMMARY"
      ? "本次总结"
      : payload.phase === "LEARNING_PAUSED"
        ? "学习已暂停"
      : payload.phase === "PRACTICE_PAUSED"
        ? "同类练习已暂停"
      : wasInSimilarPractice
      ? "同类题反馈已更新"
      : payload.correct
        ? "做对了，进入巩固练习"
        : "已生成下一步提示";
    await this.applyLearningResponse(payload, statusNote);
    if (payload.correct && !wasInSimilarPractice && !(payload.practice_items || []).length) {
      await this.appendTargetedPractice();
    }
  },

  async appendTargetedPractice() {
    const payload = await getTargetedPractice(this.data.childId, 3);
    const items = payload.items || [];
    if (!items.length) {
      return;
    }
    const text = [
      "根据你之前的错因，先练这几道同类题：",
      ...items.map((item, index) => `${index + 1}. ${item.question}`),
    ].join("\n");
    await this.streamAssistantMessage(text, {
      statusNote: "已生成针对性练习",
    });
  },

  async confirmPhotoReviewFromInput(content) {
    const parsed = parsePhotoConfirmation(content);
    if (!parsed.questionText || !parsed.childAnswer) {
      const assistantText =
        "请按格式补充照片识别结果：\n题目：题目内容\n答案：孩子答案\n也可以写成：题目内容 | 孩子答案";
      this.setData({
        statusNote: "等待确认题目和孩子答案",
      });
      await this.streamAssistantMessage(assistantText);
      return;
    }

    const review = await confirmPhotoReview(this.data.pendingPhotoReviewId, parsed);
    await this.applyPhotoReviewResponse(review);
  },

  async applyLearningResponse(payload, statusNote, options) {
    const assistantText = payload.message || payload.current_prompt || "";
    const sessionId = payload.session_id || this.data.sessionId;
    const sessionCompleted = payload.phase === "SESSION_SUMMARY";
    const questionText =
      (payload && payload.question_text) ||
      (options && options.questionText) ||
      this.data.questionText;
    const subjectCopy = buildSubjectCopy({
      subject: payload.subject || this.data.selectedSubject,
      sessionId,
      pendingPhotoReviewId: this.data.pendingPhotoReviewId,
    });
    this.setData({
      sessionId,
      questionText,
      selectedSubject: subjectCopy.subject,
      phase: payload.phase || this.data.phase,
      hintLevel: payload.hint_level || this.data.hintLevel,
      answerUnlocked: Boolean(payload.answer_unlocked),
      teachingProgress: normalizeTeachingProgress(payload),
      ...subjectCopy,
      tutorComposerDisabled: sessionCompleted,
      inputValue: sessionCompleted ? "" : this.data.inputValue,
      canSend: sessionCompleted ? false : this.data.canSend,
      inputPlaceholder: sessionCompleted
        ? "本次学习已结束，可以返回学习列表。"
        : subjectCopy.inputPlaceholder,
      statusNote,
    });
    await this.streamAssistantMessage(assistantText);
  },

  async applySubmissionTutorSnapshot(payload, options) {
    const active = payload.active_tutor_session || null;
    const completedTutor = payload.status === "completed" && !active;
    const subjectCopy = buildSubjectCopy({
      subject: payload.subject || this.data.selectedSubject,
      sessionId: active ? active.session_id : this.data.sessionId,
    });
    const statusNote =
      (options && options.statusNote) ||
      (payload.status === "completed" ? "本次错题已讲完" : "正在讲错题");
    this.setData({
      submissionId: payload.submission_id || this.data.submissionId,
      submissionStatus: payload.status || this.data.submissionStatus,
      activeQueueItemId: payload.active_queue_item_id || "",
      sessionId: active ? active.session_id : this.data.sessionId,
      questionText: active ? active.question_text : this.data.questionText,
      selectedSubject: subjectCopy.subject,
      hintLevel: active ? active.hint_level : this.data.hintLevel,
      attemptCount: active ? active.attempt_count : this.data.attemptCount,
      answerUnlocked: active ? Boolean(active.answer_unlocked) : this.data.answerUnlocked,
      ...subjectCopy,
      tutorComposerDisabled: completedTutor,
      inputValue: completedTutor ? "" : this.data.inputValue,
      canSend: completedTutor ? false : this.data.canSend,
      inputPlaceholder: completedTutor
        ? "本次错题陪练已结束，可以返回本次总结。"
        : subjectCopy.inputPlaceholder,
      statusNote,
    });
    if (payload.status === "completed" && !active) {
      await this.streamAssistantMessage("本次总结：这次提交里的错题已经讲完，系统会把这些题型放进近期复习。", {
        statusNote: "本次总结",
      });
      return;
    }
    if (active && options && options.streamPrompt) {
      await this.streamAssistantMessage(active.current_prompt || "我们来看这道错题。");
    }
  },

  async applyPhotoReviewResponse(review) {
    const assistantText = this.buildPhotoReviewText(review);
    const sessionId = review.linked_session_id || this.data.sessionId;
    const questionText = review.question_text || this.data.questionText;
    const pendingPhotoReviewId = review.status === "needs_confirmation" ? review.review_id : "";
    const subjectCopy = buildSubjectCopy({
      subject: review.subject || this.data.selectedSubject,
      sessionId,
      questionText,
      pendingPhotoReviewId,
    });
    this.setData({
      sessionId,
      selectedSubject: subjectCopy.subject,
      pendingPhotoReviewId,
      ...subjectCopy,
      statusNote: this.buildPhotoReviewStatus(review),
    });
    await this.streamAssistantMessage(assistantText);
    this.setData({
      sending: false,
      uploadingPhoto: false,
    });
  },

  streamAssistantMessage(content, options) {
    const text = String(content || "");
    const assistantMessage = buildAssistantMessage("");
    const frames = buildStreamingFrames(text, STREAM_CHUNK_SIZE);
    this.setData({
      messages: this.data.messages.concat([assistantMessage]),
      scrollTarget: assistantMessage.clientId,
      ...(options && options.statusNote ? { statusNote: options.statusNote } : {}),
    });
    if (!frames.length) {
      return Promise.resolve();
    }
    return new Promise((resolve) => {
      let index = 0;
      const tick = () => {
        const frame = frames[index];
        this.setData({
          messages: replaceMessageContent(this.data.messages, assistantMessage.clientId, frame),
          scrollTarget: assistantMessage.clientId,
        });
        index += 1;
        if (index >= frames.length) {
          resolve();
          return;
        }
        setTimeout(tick, STREAM_INTERVAL_MS);
      };
      tick();
    });
  },

  buildPhotoReviewText(review) {
    if (review.status === "graded_correct") {
      return [review.feedback, review.better_method].filter(Boolean).join("\n\n");
    }
    if (review.status === "remediation_ready") {
      return [
        review.feedback,
        review.prerequisite_question,
        review.next_prompt,
      ]
        .filter(Boolean)
        .join("\n\n");
    }
    return [
      review.feedback || "图片识别不够确定。",
      review.question_text ? `识别到的题目：${review.question_text}` : "",
      "请在输入框里确认题目和孩子答案，格式：\n题目：题目内容\n答案：孩子答案",
    ]
      .filter(Boolean)
      .join("\n\n");
  },

  buildPhotoReviewStatus(review) {
    if (review.status === "graded_correct") {
      return "照片批改完成：答案正确";
    }
    if (review.status === "remediation_ready") {
      return "照片批改完成：已进入错题引导";
    }
    return "需要确认照片识别结果";
  },

  async handleFailure(error) {
    const message = formatUserFacingError(error);
    this.setData({
      error: message,
      statusNote: "请求失败",
    });
    await this.streamAssistantMessage(`[错误] ${message}`);
    this.setData({
      sending: false,
      uploadingPhoto: false,
    });
  },
});
