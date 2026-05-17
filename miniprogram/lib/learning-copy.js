const SUBJECT_COPY = {
  math: {
    sessionTitle: "松鼠博士陪练",
  },
  english: {
    sessionTitle: "松鼠博士陪练",
  },
  chinese: {
    sessionTitle: "松鼠博士陪练",
  },
};

function normalizeSubject(subject) {
  return SUBJECT_COPY[subject] ? subject : "math";
}

function buildSubjectCopy(state) {
  const options = state || {};
  const subject = normalizeSubject(options.subject || "math");
  const base = SUBJECT_COPY[subject];
  const hasSession = Boolean(options.sessionId);
  const hasPhotoConfirmation = Boolean(options.pendingPhotoReviewId);

  return {
    subject,
    sessionTitle: base.sessionTitle,
    introSubtitle: hasSession ? "松鼠博士会根据孩子的回答继续追问。" : "先提交题目和孩子答案，系统记录后再进入陪练。",
    statusNote: hasSession ? "学习中" : "开始做题",
    emptyTitle: "开始做题后，松鼠博士会先记录再陪练。",
    emptyText: "本次做题过程会沉淀到学习档案；需要陪练的题会进入松鼠博士陪练。",
    composerLabel: hasSession ? "写下孩子这一步的想法" : "提交题目和答案",
    inputPlaceholder: hasPhotoConfirmation
      ? "题目：...\n答案：..."
      : hasSession
        ? "输入孩子这一步的答案或想法"
        : "输入题目和孩子答案，或拍照/语音识别后提交",
    startButtonLabel: hasSession ? "提交尝试" : "开始做题",
  };
}

function getSubjectModeLabel(subject) {
  normalizeSubject(subject);
  return "做题引导";
}

module.exports = {
  SUBJECT_COPY,
  buildSubjectCopy,
  getSubjectModeLabel,
};
