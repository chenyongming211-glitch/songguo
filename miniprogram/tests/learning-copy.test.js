const assert = require("assert");
const fs = require("fs");
const path = require("path");

const { buildSubjectCopy, getSubjectModeLabel } = require("../lib/learning-copy");

const appConfig = JSON.parse(
  fs.readFileSync(path.join(__dirname, "..", "app.json"), "utf8")
);
const chatListMarkup = fs.readFileSync(
  path.join(__dirname, "..", "pages", "chat-list", "index.wxml"),
  "utf8"
);
const chatListScript = fs.readFileSync(
  path.join(__dirname, "..", "pages", "chat-list", "index.js"),
  "utf8"
);
const chatDetailMarkup = fs.readFileSync(
  path.join(__dirname, "..", "pages", "chat-detail", "index.wxml"),
  "utf8"
);
const chatDetailStyle = fs.readFileSync(
  path.join(__dirname, "..", "pages", "chat-detail", "index.wxss"),
  "utf8"
);
const chatDetailScript = fs.readFileSync(
  path.join(__dirname, "..", "pages", "chat-detail", "index.js"),
  "utf8"
);
const submissionReviewMarkup = fs.existsSync(
  path.join(__dirname, "..", "pages", "submission-review", "index.wxml")
)
  ? fs.readFileSync(
      path.join(__dirname, "..", "pages", "submission-review", "index.wxml"),
      "utf8"
    )
  : "";
const submissionReviewScript = fs.existsSync(
  path.join(__dirname, "..", "pages", "submission-review", "index.js")
)
  ? fs.readFileSync(
      path.join(__dirname, "..", "pages", "submission-review", "index.js"),
      "utf8"
    )
  : "";
const parentMarkup = fs.readFileSync(
  path.join(__dirname, "..", "pages", "parent-report", "index.wxml"),
  "utf8"
);
const settingsMarkup = fs.readFileSync(
  path.join(__dirname, "..", "pages", "settings", "index.wxml"),
  "utf8"
);
const chatListConfig = JSON.parse(
  fs.readFileSync(path.join(__dirname, "..", "pages", "chat-list", "index.json"), "utf8")
);

function assertNoMathOnlyCopy(subject) {
  const copy = buildSubjectCopy({ subject });
  const joined = [
    copy.sessionTitle,
    copy.introSubtitle,
    copy.statusNote,
    copy.emptyTitle,
    copy.emptyText,
    copy.composerLabel,
    copy.inputPlaceholder,
    copy.startButtonLabel,
  ].join("\n");

  assert.ok(!joined.includes("数学错题"), `${subject} copy must not mention 数学错题`);
}

const math = buildSubjectCopy({ subject: "math" });
assert.equal(math.sessionTitle, "松鼠博士陪练");
assert.equal(math.emptyTitle, "开始做题后，松鼠博士会先记录再陪练。");
assert.equal(math.inputPlaceholder, "输入题目和孩子答案，或拍照/语音识别后提交");

const english = buildSubjectCopy({ subject: "english" });
assert.equal(english.sessionTitle, "松鼠博士陪练");
assert.equal(english.emptyTitle, "开始做题后，松鼠博士会先记录再陪练。");
assert.equal(english.inputPlaceholder, "输入题目和孩子答案，或拍照/语音识别后提交");
assertNoMathOnlyCopy("english");

const chinese = buildSubjectCopy({ subject: "chinese" });
assert.equal(chinese.sessionTitle, "松鼠博士陪练");
assert.equal(chinese.emptyTitle, "开始做题后，松鼠博士会先记录再陪练。");
assert.equal(chinese.inputPlaceholder, "输入题目和孩子答案，或拍照/语音识别后提交");
assertNoMathOnlyCopy("chinese");

const active = buildSubjectCopy({ subject: "english", sessionId: "s_1" });
assert.equal(active.inputPlaceholder, "输入孩子这一步的答案或想法");
assert.equal(active.composerLabel, "写下孩子这一步的想法");

const photo = buildSubjectCopy({ subject: "chinese", pendingPhotoReviewId: "r_1" });
assert.equal(photo.inputPlaceholder, "题目：...\n答案：...");

assert.equal(getSubjectModeLabel("math"), "做题引导");
assert.equal(getSubjectModeLabel("english"), "做题引导");
assert.equal(getSubjectModeLabel("chinese"), "做题引导");

assert.equal(appConfig.window.navigationBarTitleText, "松果AI");
assert.equal(chatListConfig.navigationBarTitleText, "松果AI");
assert.ok(chatListMarkup.includes("松果AI"), "home hero should use Songguo AI brand");
assert.ok(chatListMarkup.includes("松鼠博士"), "home hero should mention the tutor persona");
assert.ok(chatListMarkup.includes("开始做题"), "home primary action should be a child-facing task entry");
assert.ok(chatListMarkup.includes("我的练习记录"), "home history should be framed as practice records");
assert.ok(chatListMarkup.includes('data-source-type="photo"'), "home should expose photo entry under one submission flow");
assert.ok(chatListMarkup.includes('data-source-type="voice"'), "home should expose voice entry under one submission flow");
assert.ok(chatListMarkup.includes('data-source-type="text"'), "home should expose text entry under one submission flow");
assert.ok(chatListMarkup.includes("handleStartSubmission"), "home entries should route through unified submission start");
assert.ok(chatListMarkup.includes("handleStartPhotoSubmission"), "photo entry should run recognition before review");
assert.ok(chatListMarkup.includes("handleStartVoiceSubmission"), "voice entry should use ASR fallback before review");
assert.ok(
  chatListScriptIncludesSubmissionReview(),
  "home entries should navigate to submission-review instead of old chat-detail intake"
);
assert.ok(
  chatListScript.includes("recognizeSubmissionPhoto") &&
    chatListScript.includes("chooseHomeworkImage") &&
    chatListScript.includes("initialText"),
  "photo entry should recognize homework and prefill submission review"
);
assert.ok(
  chatListScript.includes("语音识别还在接入中") &&
    chatListScript.includes('navigateToSubmissionReview("voice"') &&
    chatListScript.includes("initialText"),
  "voice entry should fail softly into the unified review page"
);
assert.ok(chatListMarkup.includes("learner-pill-row"), "home learner status should be compact");
assert.ok(!chatListMarkup.includes("learner-strip card"), "home learner selector must not be a large card");
assert.ok(!chatListMarkup.includes("当前后端"), "home page must not expose backend debug URL");
assert.ok(!chatListMarkup.includes("DeepTutor"), "mini-program UI must not expose DeepTutor brand");
assert.ok(!parentMarkup.includes("DeepTutor"), "parent page must not expose DeepTutor brand");
assert.ok(!settingsMarkup.includes("DeepTutor"), "settings page must not expose DeepTutor brand");
assert.ok(parentMarkup.includes("experimentalFeaturesEnabled"), "experimental parent tools must be gated");
assert.ok(!chatDetailMarkup.includes("当前关键点"), "chat page must not expose internal key-point state");
assert.ok(!chatDetailMarkup.includes("progress-count"), "chat page must not expose fixed-step counters");
assert.ok(!chatDetailStyle.includes("progress-count"), "chat page styles must not keep fixed-step counters");
assert.ok(!chatListMarkup.includes("关键点"), "home page copy must not expose internal teaching mechanics");
assert.ok(chatDetailScript.includes("streamAssistantMessage"), "assistant replies should stream after backend safety checks");
assert.ok(chatDetailScript.includes("buildStreamingFrames"), "chat page should use safe final-message streaming frames");
assert.ok(
  chatDetailScript.includes("} finally {\n      this.setData({ sending: false });\n    }"),
  "send flow should keep input locked until all safe streaming work finishes"
);
assert.ok(
  chatDetailScript.includes("await this.handleFailure(error);"),
  "failure replies should be awaited so fallback streaming cannot overlap with the next send"
);
assert.ok(
  chatDetailScript.includes("await this.appendTargetedPractice();"),
  "targeted practice streaming should remain inside the awaited send flow"
);
assert.ok(
  chatDetailScript.includes("wasInSimilarPractice") &&
    chatDetailScript.includes("同类题反馈已更新") &&
    chatDetailScript.includes("payload.correct && !wasInSimilarPractice"),
  "similar-practice answers must not be treated as original-question completion"
);
assert.ok(
  appConfig.pages.includes("pages/submission-review/index"),
  "app.json should register the submission review page"
);
assert.ok(submissionReviewMarkup.includes("确认题目"), "submission review should show a confirmation step");
assert.ok(submissionReviewMarkup.includes("判题概览"), "submission review should show judged summary");
assert.ok(submissionReviewMarkup.includes("开始讲错题"), "submission review should expose wrong-question tutoring entry");
assert.ok(submissionReviewMarkup.includes("本次总结"), "submission review should show completion summary");
assert.ok(
  submissionReviewScript.includes("createLearningSubmission") &&
    submissionReviewScript.includes("confirmLearningSubmission") &&
    submissionReviewScript.includes("startNextSubmissionTutorItem"),
  "submission review page should use submissions API as the main flow"
);
assert.ok(
  submissionReviewScript.includes("initialText") &&
    submissionReviewScript.includes("canSubmit: Boolean(trimInput(initialText))"),
  "submission review should prefill recognized text from photo or voice"
);
assert.ok(
  chatDetailScript.includes("submissionId") &&
    chatDetailScript.includes("submitSubmissionTutorAttempt") &&
    chatDetailScript.includes("startNextSubmissionTutorItem"),
  "chat detail should support submission queue tutoring"
);
assert.ok(
  chatDetailScript.includes("tutorComposerDisabled") &&
    chatDetailMarkup.includes("disabled=\"{{tutorComposerDisabled || sending || !canSend}}\""),
  "completed submission tutoring must disable answer submission instead of calling tutor attempt again"
);
assert.ok(
  chatDetailScript.includes('payload.phase === "SESSION_SUMMARY"') &&
    chatDetailScript.includes('statusNote: "本次总结"'),
  "session summary should disable continued attempts and show summary status"
);
assert.ok(
  chatDetailScript.includes('payload.phase === "PRACTICE_PAUSED"') &&
    chatDetailScript.includes('"同类练习已暂停"'),
  "paused similar-practice sessions should stay resumable with a paused status"
);
assert.ok(
  chatDetailScript.includes('payload.phase === "LEARNING_PAUSED"') &&
    chatDetailScript.includes('"学习已暂停"'),
  "paused normal tutoring sessions should stay resumable with a learning-paused status"
);

function chatListScriptIncludesSubmissionReview() {
  const script = fs.readFileSync(
    path.join(__dirname, "..", "pages", "chat-list", "index.js"),
    "utf8"
  );
  return script.includes("/pages/submission-review/index");
}

console.log("learning-copy-ok");
