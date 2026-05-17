const assert = require("assert");
const fs = require("fs");
const path = require("path");

function readPage(file) {
  return fs.readFileSync(path.join(__dirname, "..", "pages", "submission-review", file), "utf8");
}

const markup = readPage("index.wxml");
const script = readPage("index.js");
const styles = readPage("index.wxss");
const homeScript = fs.readFileSync(
  path.join(__dirname, "..", "pages", "chat-list", "index.js"),
  "utf8"
);

assert.ok(homeScript.includes("localImagePath"), "photo draft should preserve the local image path");
assert.ok(homeScript.includes("items: (draft && draft.items) || []"), "photo draft should preserve item bbox data");
assert.ok(homeScript.includes("compressHomeworkImage"), "photo entry should compress images before upload");
assert.ok(homeScript.includes("wx.compressImage"), "photo compression should use the WeChat native compressor when available");
assert.ok(script.includes("buildOverlayItems"), "submission page should build overlay view models");
assert.ok(script.includes("bboxStyle"), "submission page should convert normalized bbox to style");
assert.ok(script.includes("photoImagePath"), "submission page should keep the photo image path");
assert.ok(script.includes("handleAutoJudgePhotoDraft"), "photo draft should auto-create and confirm for fast marking");
assert.ok(script.includes("autoJudging"), "submission page should expose an auto-judging loading state");
assert.ok(script.includes("buildSubmissionPageStatus"), "submission page should keep pending fallback submissions out of completed state");
assert.ok(script.includes("correctAnswerText"), "submission page should expose correct answers for judged wrong items");
assert.ok(script.includes("markerForStatus"), "submission page should map backend display status to markers");
assert.ok(script.includes("fallback_running"), "submission page should keep async fallback items visible as running");
assert.ok(script.includes("qualityWarningText"), "submission page should expose photo quality warnings");
assert.ok(script.includes("qualityMessage"), "submission page should prefer backend photo quality guidance");
assert.ok(script.includes('warning === "retake_required"'), "submission page should name retake-required quality warnings");
assert.ok(script.includes('warning === "glare_or_overexposed_area"'), "submission page should name glare quality warnings");
assert.ok(script.includes('warning === "no_structured_items"'), "submission page should name unstructured OCR warnings");
assert.ok(script.includes("请先核对识别内容"), "submission page should ask users to confirm recognized content before retaking");
assert.ok(homeScript.includes("qualityWarnings"), "photo draft should preserve quality warnings");
assert.ok(homeScript.includes("qualityMessage"), "photo draft should preserve backend quality guidance");
assert.ok(markup.includes("photo-overlay-panel"), "submission page should render a photo overlay panel");
assert.ok(markup.includes('src="{{photoImagePath}}"'), "submission page should show the original photo");
assert.ok(markup.includes("overlay-item"), "submission page should render per-item overlays");
assert.ok(markup.includes("overlay-marker"), "submission page should render per-item result markers");
assert.ok(markup.includes("item-correct-answer"), "submission page should render correct answer text");
assert.ok(markup.includes("quality-warning"), "submission page should render quality warning text");
assert.ok(styles.includes(".overlay-correct"), "styles should include correct marker state");
assert.ok(styles.includes(".overlay-wrong"), "styles should include wrong marker state");
assert.ok(styles.includes(".overlay-running"), "styles should include async fallback marker state");
assert.ok(styles.includes(".item-correct-answer"), "styles should include correct answer state");

console.log("photo-overlay-ok");
