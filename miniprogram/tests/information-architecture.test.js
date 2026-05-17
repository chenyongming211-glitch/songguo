const assert = require("assert");
const fs = require("fs");
const path = require("path");

function readPage(page, file) {
  return fs.readFileSync(path.join(__dirname, "..", "pages", page, file), "utf8");
}

const homeMarkup = readPage("chat-list", "index.wxml");
const homeScript = readPage("chat-list", "index.js");
const parentMarkup = readPage("parent-report", "index.wxml");
const parentScript = readPage("parent-report", "index.js");
const settingsMarkup = readPage("settings", "index.wxml");
const settingsScript = readPage("settings", "index.js");
const submissionMarkup = readPage("submission-review", "index.wxml");

assert.ok(homeMarkup.includes("拍照做题"), "home primary action should be photo-first");
assert.ok(homeMarkup.includes("语音说题"), "home should keep voice as a secondary entry");
assert.ok(homeMarkup.includes("手动输入"), "home should keep text as a secondary entry");
assert.ok(!homeMarkup.includes("刷新列表"), "home must not expose refresh as a persistent action");
assert.ok(!homeMarkup.includes("promise-grid"), "home should not keep product-explainer cards above history");
assert.ok(homeMarkup.includes("服务暂时连不上"), "home error state should use product copy");
assert.ok(homeMarkup.includes("handleRetry"), "home error state should expose retry");
assert.ok(homeMarkup.includes("handleOpenSettings"), "home error state should link to settings");
assert.ok(homeScript.includes("handleRetry"), "home retry handler should reload automatically");
assert.ok(homeScript.includes("/pages/settings/index"), "home settings handler should navigate to settings");

assert.ok(
  submissionMarkup.includes("确认并保存") || submissionMarkup.includes("确认并开始讲错题"),
  "submission confirmation should use business action labels"
);

assert.ok(!parentMarkup.includes("刷新报告"), "parent page must not expose refresh as a persistent action");
assert.ok(parentMarkup.includes("本期结论"), "parent page should lead with a conclusion section");
assert.ok(parentMarkup.includes("行动建议"), "parent page should show action suggestions before raw evidence");
assert.ok(parentMarkup.includes("薄弱点"), "parent page should show a parent-facing weakness section");
assert.ok(parentMarkup.includes("parentConclusionItems"), "parent page should render conclusion view models");
assert.ok(parentMarkup.includes("parentActionItems"), "parent page should render action view models");
assert.ok(parentMarkup.includes("weaknessCards"), "parent page should render weakness view models");
assert.ok(parentMarkup.includes("recentWrongCards"), "parent page should render recent wrong-question view models");
assert.ok(!parentMarkup.includes("summaryDraft."), "parent page should not bind raw summary draft fields");
assert.ok(!parentMarkup.includes("reviewPlan.items"), "parent page should not bind raw review plan items");
assert.ok(!parentMarkup.includes("memory.top_weaknesses"), "parent page should not bind raw memory weaknesses");
assert.ok(!parentMarkup.includes("wrongQuestions"), "parent page should not bind raw wrong-question records");
assert.ok(!parentMarkup.includes("学习资产沉淀"), "parent page should hide internal learning-deposit wording");
assert.ok(!parentMarkup.includes("报告草稿"), "parent page should hide internal draft wording");
assert.ok(parentMarkup.includes('wx:if="{{safetyEvents.length}}"'), "empty safety records should not occupy space");
assert.ok(parentScript.includes("handleRetry"), "parent error state should expose retry");
assert.ok(parentScript.includes("buildParentConclusionItems"), "parent script should build conclusion view models");
assert.ok(parentScript.includes("buildParentActionItems"), "parent script should build action view models");
assert.ok(parentScript.includes("buildWeaknessCards"), "parent script should build weakness view models");
assert.ok(parentScript.includes("buildRecentWrongCards"), "parent script should build wrong-question view models");
assert.ok(!parentScript.includes("getParentLearningDeposit"), "parent page should not fetch internal learning deposit in the main report");

assert.ok(settingsMarkup.includes("高级诊断"), "settings should place technical diagnostics behind an advanced section");
assert.ok(
  settingsMarkup.includes('wx:if="{{diagnosticsAvailable}}"'),
  "settings should only expose advanced diagnostics when diagnostics are available"
);
assert.ok(
  !settingsMarkup.includes('<text class="section-text">{{httpBaseUrl}}</text>'),
  "settings default view must not expose the raw backend URL"
);
assert.ok(
  settingsScript.includes("生产学习服务"),
  "settings default view should describe the service without raw technical URL"
);
assert.ok(settingsScript.includes("advancedDiagnosticsVisible"), "settings should support collapsed advanced diagnostics");
assert.ok(settingsScript.includes("diagnosticsAvailable"), "settings should know whether diagnostics are available");
assert.ok(settingsScript.includes("isDevtoolsRuntime"), "settings should gate diagnostics by runtime");
assert.ok(settingsScript.includes("handleToggleAdvancedDiagnostics"), "settings should expose an advanced toggle handler");

console.log("information-architecture-ok");
