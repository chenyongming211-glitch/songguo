const {
  generateLearningDiagram,
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
const { formatUserFacingError } = require("../../lib/errors");

const SELECTED_CHILD_STORAGE_KEY = "songguo_selected_child_id";
const REVIEW_SCOPES = [
  { value: "weekly", label: "本周" },
  { value: "monthly", label: "本月" },
  { value: "quarterly", label: "本季度" },
  { value: "term", label: "本学期" },
  { value: "yearly", label: "本年度" },
];
const REVIEW_SCOPE_OPTIONS = REVIEW_SCOPES.map((item) => item.label);

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

function getReviewScopeIndex(scope) {
  const index = REVIEW_SCOPES.findIndex((item) => item.value === scope);
  return index >= 0 ? index : 0;
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

function firstNonEmpty(values, fallback) {
  const found = values.find((value) => String(value || "").trim());
  return found ? toParentText(found).trim() : fallback;
}

function buildSummaryStats(report) {
  if (!report) {
    return [];
  }
  return [
    { label: "学习会话", value: report.session_count || 0 },
    { label: "错题记录", value: report.wrong_question_count || 0 },
    { label: "平均提示", value: report.average_hint_level || 0 },
  ];
}

function buildParentConclusionItems({ sessionFeedback, summaryDraft, report, reviewScopeLabel }) {
  const items = [];
  if (sessionFeedback) {
    items.push({
      label: "本次卡点",
      title: firstNonEmpty([sessionFeedback.question_text], "最近一道题"),
      text: firstNonEmpty([sessionFeedback.summary], "本次学习反馈正在整理。"),
    });
  }
  if (summaryDraft || report) {
    items.push({
      label: reviewScopeLabel || "本期",
      title: "阶段表现",
      text: firstNonEmpty(
        [summaryDraft && summaryDraft.parent_summary, report && report.parent_summary],
        "孩子完成更多练习后，这里会显示阶段表现。"
      ),
    });
  }
  if (sessionFeedback && sessionFeedback.evidence) {
    items.push({
      label: "判断依据",
      title: "为什么这么判断",
      text: firstNonEmpty([sessionFeedback.evidence], "系统会根据提示轨迹和孩子回答给出依据。"),
    });
  }
  if (!items.length) {
    items.push({
      label: reviewScopeLabel || "本期",
      title: "还没有足够记录",
      text: "孩子完成一次错题引导后，这里会生成本期结论。",
    });
  }
  return items.slice(0, 3);
}

function buildParentActionItems(reviewPlan, summaryDraft) {
  const actions = [];
  if (reviewPlan && reviewPlan.summary) {
    actions.push({
      label: "复习方向",
      title: "先抓最容易反复错的地方",
      text: firstNonEmpty([reviewPlan.summary], "系统正在整理复习方向。"),
    });
  }
  (reviewPlan && reviewPlan.items ? reviewPlan.items : []).slice(0, 3).forEach((item) => {
    actions.push({
      label: "建议练习",
      title: firstNonEmpty([item.normalized_question], "同类题练习"),
      text: firstNonEmpty([item.reason], "按最近错因安排同类题。"),
    });
  });
  if (!actions.length && summaryDraft) {
    (summaryDraft.next_actions || []).slice(0, 3).forEach((item, index) => {
      actions.push({
        label: `建议 ${index + 1}`,
        title: "下一步可以这样做",
        text: firstNonEmpty([item], "完成一次做题后会生成建议。"),
      });
    });
  }
  return actions.slice(0, 4);
}

function buildWeaknessCards(memory) {
  const topWeaknesses = (memory && memory.top_weaknesses) || [];
  if (topWeaknesses.length) {
    return topWeaknesses.slice(0, 3).map((item) => ({
      title: firstNonEmpty([item.knowledge_point_label], "薄弱知识点"),
      text: `掌握度 ${item.mastery_score || 0} · 风险 ${toParentText(item.risk_level || "待观察")} · 错 ${
        item.wrong_count || 0
      } 次`,
    }));
  }
  if (memory && memory.summary) {
    return [
      {
        title: "整体观察",
        text: firstNonEmpty([memory.summary], "暂时还没有稳定薄弱点。"),
      },
    ];
  }
  return [];
}

function buildRecentWrongCards(items) {
  return (items || []).slice(0, 5).map((item) => ({
    questionId: item.question_id,
    questionText: firstNonEmpty([item.normalized_question], "最近错题"),
    knowledgePoint: item.knowledge_point || "grade_math_unknown",
    meta: `${firstNonEmpty([item.knowledge_point_label], "知识点待确认")} · 提示等级 ${
      item.highest_hint_level || 0
    } · ${firstNonEmpty([item.misconception_label], "错因待确认")}`,
  }));
}

Page({
  data: {
    childId: getSelectedChildId(),
    children: [],
    loading: false,
    error: "",
    summaryStats: [],
    parentConclusionItems: [],
    parentActionItems: [],
    weaknessCards: [],
    recentWrongCards: [],
    safetyEvents: [],
    reviewScope: "weekly",
    reviewScopeLabel: "本周",
    reviewScopeIndex: 0,
    reviewScopeOptions: REVIEW_SCOPE_OPTIONS,
    artifactMessage: "",
    experimentalFeaturesEnabled: wx.getStorageSync("songguo_experimental_features") === true,
    reviewScopes: REVIEW_SCOPES,
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
      this.setData({ error: formatUserFacingError(error) });
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
      if (latestSessionId) {
        try {
          sessionFeedback = await getParentSessionFeedback(this.data.childId, latestSessionId);
        } catch (error) {
          sessionFeedback = null;
        }
      }
      const normalizedReport = normalizeReport(report);
      const normalizedSummaryDraft = normalizeSummaryDraft(summaryDraft, this.data.reviewScopeLabel);
      const normalizedMemory = normalizeMemory(memory);
      const normalizedReviewPlan = normalizeReviewPlan(reviewPlan);
      const normalizedSessionFeedback = normalizeSessionFeedback(sessionFeedback);
      this.setData({
        summaryStats: buildSummaryStats(normalizedReport),
        parentConclusionItems: buildParentConclusionItems({
          sessionFeedback: normalizedSessionFeedback,
          summaryDraft: normalizedSummaryDraft,
          report: normalizedReport,
          reviewScopeLabel: this.data.reviewScopeLabel,
        }),
        parentActionItems: buildParentActionItems(normalizedReviewPlan, normalizedSummaryDraft),
        weaknessCards: buildWeaknessCards(normalizedMemory),
        recentWrongCards: buildRecentWrongCards(wrongItems),
        safetyEvents: safetyEvents.items || [],
      });
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
    this.loadReport();
  },

  handleRefresh() {
    this.rollupAndLoadReport();
  },

  handleRetry() {
    this.ensureChildrenAndLoad();
  },

  handleOpenSettings() {
    wx.switchTab({
      url: "/pages/settings/index",
    });
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
    this.setData({
      reviewScope: scope,
      reviewScopeLabel: getReviewScopeLabel(scope),
      reviewScopeIndex: getReviewScopeIndex(scope),
    });
    this.loadReport();
  },

  handleScopePickerChange(event) {
    const index = Number(event.detail.value || 0);
    const selected = REVIEW_SCOPES[index] || REVIEW_SCOPES[0];
    if (selected.value === this.data.reviewScope) {
      return;
    }
    this.setData({
      reviewScope: selected.value,
      reviewScopeLabel: selected.label,
      reviewScopeIndex: getReviewScopeIndex(selected.value),
    });
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
