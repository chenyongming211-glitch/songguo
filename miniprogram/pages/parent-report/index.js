const {
  generateLearningDiagram,
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
  requestLearningAnimation,
  rollupParentSummary,
  saveReminderSubscription,
  submitPracticeResult,
  upsertChild,
} = require("../../lib/api");

const SELECTED_CHILD_STORAGE_KEY = "songguo_selected_child_id";
const REVIEW_SCOPES = [
  { value: "weekly", label: "本周" },
  { value: "monthly", label: "本月" },
  { value: "quarterly", label: "本季度" },
  { value: "term", label: "本学期" },
  { value: "yearly", label: "本年度" },
];

function getSelectedChildId() {
  try {
    return wx.getStorageSync(SELECTED_CHILD_STORAGE_KEY) || getSessionChildId();
  } catch (_error) {
    return getSessionChildId();
  }
}

function getReviewScopeLabel(scope) {
  const selected = REVIEW_SCOPES.find((item) => item.value === scope);
  return selected ? selected.label : "本周";
}

function toParentText(value) {
  return String(value || "")
    .replace(/misconception_1/g, "比较关系理解不清")
    .replace(/\bhigh\b/g, "高")
    .replace(/\bmedium\b/g, "中")
    .replace(/\blow\b/g, "低")
    .replace(/\bweekly\b/g, "本周")
    .replace(/\bmonthly\b/g, "本月")
    .replace(/\bquarterly\b/g, "本季度")
    .replace(/\bterm\b/g, "本学期")
    .replace(/\byearly\b/g, "本年度");
}

function normalizeReport(report) {
  return report ? { ...report, parent_summary: toParentText(report.parent_summary) } : null;
}

function normalizeSessionFeedback(feedback) {
  return feedback
    ? {
        ...feedback,
        summary: toParentText(feedback.summary),
        evidence: toParentText(feedback.evidence),
      }
    : null;
}

function normalizeLearningDeposit(deposit) {
  if (!deposit) {
    return null;
  }
  return {
    ...deposit,
    parent_summary: toParentText(deposit.parent_summary),
    mistake_record: deposit.mistake_record
      ? {
          ...deposit.mistake_record,
          main_error_reason_label: toParentText(deposit.mistake_record.main_error_reason_label),
        }
      : null,
  };
}

function normalizeSummaryDraft(draft, scopeLabel) {
  return draft
    ? {
        ...draft,
        scope_label: scopeLabel,
        parent_summary: toParentText(draft.parent_summary),
        next_actions: (draft.next_actions || []).map(toParentText),
      }
    : null;
}

function normalizeMemory(memory) {
  return memory
    ? {
        ...memory,
        summary: toParentText(memory.summary),
        top_weaknesses: (memory.top_weaknesses || []).map((item) => ({
          ...item,
          risk_level: toParentText(item.risk_level),
        })),
        next_actions: (memory.next_actions || []).map(toParentText),
      }
    : null;
}

function normalizeReviewPlan(plan) {
  return plan
    ? {
        ...plan,
        summary: toParentText(plan.summary),
        items: (plan.items || []).map((item) => ({
          ...item,
          reason: toParentText(item.reason),
        })),
      }
    : null;
}

function normalizeWrongQuestions(items) {
  return (items || []).map((item) => ({
    ...item,
    misconception_label: toParentText(item.misconception_label || ""),
  }));
}

Page({
  data: {
    childId: getSelectedChildId(),
    children: [],
    loading: false,
    error: "",
    report: null,
    summaryDraft: null,
    sessionFeedback: null,
    learningDeposit: null,
    memory: null,
    reviewPlan: null,
    safetyEvents: [],
    reviewScope: "weekly",
    reviewScopeLabel: "本周",
    artifactMessage: "",
    experimentalFeaturesEnabled: wx.getStorageSync("songguo_experimental_features") === true,
    reviewScopes: REVIEW_SCOPES,
    wrongQuestions: [],
  },

  onShow() {
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
      const selectedChildId = getSelectedChildId();
      const selected =
        children.find((item) => item.child_id === this.data.childId) ||
        children.find((item) => item.child_id === selectedChildId) ||
        children[0];
      this.setData({
        children,
        childId: selected.child_id,
      });
      wx.setStorageSync(SELECTED_CHILD_STORAGE_KEY, selected.child_id);
      await this.loadReport();
    } catch (error) {
      this.setData({ error: error.message || "加载孩子档案失败" });
    } finally {
      this.setData({ loading: false });
    }
  },

  async loadReport() {
    this.setData({ loading: true, error: "" });
    try {
      const [report, summaryDraft, memory, reviewPlan, safetyEvents, wrongQuestions] = await Promise.all([
        getParentWeeklyReport(this.data.childId),
        getParentSummaryDraft(this.data.childId, this.data.reviewScope),
        getParentLearningMemory(this.data.childId, this.data.reviewScope),
        getParentReviewPlan(this.data.childId, this.data.reviewScope),
        getParentSafetyEvents(this.data.childId),
        getParentWrongQuestions(this.data.childId),
      ]);
      const wrongItems = normalizeWrongQuestions(wrongQuestions.items || []);
      const latestSessionId = wrongItems.length ? wrongItems[0].session_id : "";
      let sessionFeedback = null;
      let learningDeposit = null;
      if (latestSessionId) {
        try {
          sessionFeedback = await getParentSessionFeedback(this.data.childId, latestSessionId);
        } catch (error) {
          sessionFeedback = null;
        }
        try {
          learningDeposit = await getParentLearningDeposit(
            this.data.childId,
            latestSessionId,
            this.data.reviewScope
          );
        } catch (error) {
          learningDeposit = null;
        }
      }
      this.setData({
        report: normalizeReport(report),
        summaryDraft: normalizeSummaryDraft(summaryDraft, this.data.reviewScopeLabel),
        sessionFeedback: normalizeSessionFeedback(sessionFeedback),
        learningDeposit: normalizeLearningDeposit(learningDeposit),
        memory: normalizeMemory(memory),
        reviewPlan: normalizeReviewPlan(reviewPlan),
        safetyEvents: safetyEvents.items || [],
        wrongQuestions: wrongItems,
      });
    } catch (error) {
      this.setData({
        error: error.message || "加载家长报告失败",
      });
    } finally {
      this.setData({ loading: false });
      wx.stopPullDownRefresh();
    }
  },

  onPullDownRefresh() {
    this.loadReport();
  },

  handleRefresh() {
    this.rollupAndLoadReport();
  },

  async rollupAndLoadReport() {
    this.setData({ loading: true, error: "" });
    try {
      await rollupParentSummary(this.data.childId, this.data.reviewScope);
    } catch (_error) {
      // 摘要沉淀失败不阻塞实时报告展示。
    } finally {
      await this.loadReport();
    }
  },

  handleScopeChange(event) {
    const scope = event.currentTarget.dataset.scope || "weekly";
    if (scope === this.data.reviewScope) {
      return;
    }
    this.setData({ reviewScope: scope, reviewScopeLabel: getReviewScopeLabel(scope) });
    this.loadReport();
  },

  handleChildChange(event) {
    const childId = event.currentTarget.dataset.childId || getSessionChildId();
    if (childId === this.data.childId) {
      return;
    }
    this.setData({ childId });
    wx.setStorageSync("songguo_selected_child_id", childId);
    this.loadReport();
  },

  async handleMarkResolved(event) {
    const questionId = event.currentTarget.dataset.questionId;
    if (!questionId || this.data.loading) {
      return;
    }
    this.setData({ loading: true, error: "" });
    try {
      await submitPracticeResult(questionId, {
        childId: this.data.childId,
        correct: true,
      });
      await this.loadReport();
    } catch (error) {
      this.setData({
        error: error.message || "标记掌握失败",
      });
    } finally {
      this.setData({ loading: false });
    }
  },

  async handleEnableReminder() {
    const app = getApp();
    const session =
      (app.globalData && app.globalData.wechatSession) ||
      wx.getStorageSync("songguo_wechat_session") ||
      wx.getStorageSync(["deep", "tutor_wechat_session"].join("")) ||
      {};
    if (!session.openid) {
      this.setData({ error: "请先完成微信登录后再开启提醒" });
      return;
    }
    this.setData({ loading: true, error: "" });
    try {
      await saveReminderSubscription({
        openid: session.openid,
        childId: this.data.childId,
        templateId: "review_plan_reminder",
        scope: this.data.reviewScope,
      });
      wx.showToast({
        title: "提醒已开启",
        icon: "success",
      });
    } catch (error) {
      this.setData({
        error: error.message || "开启提醒失败",
      });
    } finally {
      this.setData({ loading: false });
    }
  },

  async handleGenerateDiagram(event) {
    const payload = this.buildArtifactPayload(event);
    if (!payload.questionText || this.data.loading) {
      return;
    }
    this.setData({ loading: true, error: "", artifactMessage: "" });
    try {
      const artifact = await generateLearningDiagram(payload);
      this.setData({
        artifactMessage: `图解卡已生成：${artifact.title || artifact.artifact_id}`,
      });
    } catch (error) {
      this.setData({ error: error.message || "生成图解失败" });
    } finally {
      this.setData({ loading: false });
    }
  },

  async handleRequestAnimation(event) {
    const payload = this.buildArtifactPayload(event);
    if (!payload.questionText || this.data.loading) {
      return;
    }
    this.setData({ loading: true, error: "", artifactMessage: "" });
    try {
      const artifact = await requestLearningAnimation(payload);
      this.setData({
        artifactMessage: `动态图解任务已提交：${artifact.status || "queued"}`,
      });
    } catch (error) {
      this.setData({ error: error.message || "提交动态图解任务失败" });
    } finally {
      this.setData({ loading: false });
    }
  },

  buildArtifactPayload(event) {
    return {
      childId: this.data.childId,
      questionText: event.currentTarget.dataset.question || "",
      knowledgePoint: event.currentTarget.dataset.knowledgePoint || "grade_math_unknown",
    };
  },
});
