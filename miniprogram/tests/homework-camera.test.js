const assert = require("assert");
const fs = require("fs");
const path = require("path");

const root = path.join(__dirname, "..");
const appJson = fs.readFileSync(path.join(root, "app.json"), "utf8");
const chatListScript = fs.readFileSync(path.join(root, "pages", "chat-list", "index.js"), "utf8");
const cameraScript = fs.readFileSync(path.join(root, "pages", "homework-camera", "index.js"), "utf8");
const cameraMarkup = fs.readFileSync(path.join(root, "pages", "homework-camera", "index.wxml"), "utf8");
const cameraStyles = fs.readFileSync(path.join(root, "pages", "homework-camera", "index.wxss"), "utf8");

assert.ok(appJson.includes("pages/homework-camera/index"), "app.json should register the guided camera page");
assert.ok(chatListScript.includes("chooseHomeworkImageWithCameraGuide"), "photo entry should open the guided camera first");
assert.ok(chatListScript.includes("capturedHomeworkImage"), "guided camera should return the captured image by event channel");
assert.ok(cameraMarkup.includes("<camera"), "guided camera page should use the WeChat camera component");
assert.ok(cameraMarkup.includes("请将手机放平拍摄"), "guided camera should show a level hint");
assert.ok(cameraScript.includes("wx.startDeviceMotionListening"), "guided camera should listen for device motion");
assert.ok(cameraScript.includes("wx.onDeviceMotionChange"), "guided camera should update the level hint from motion");
assert.ok(cameraScript.includes("takePhoto"), "guided camera should capture the homework image");
assert.ok(cameraScript.includes("homeworkCameraClosed"), "guided camera should notify cancellation");
assert.ok(cameraStyles.includes(".camera-grid"), "guided camera should draw framing grid");

console.log("homework-camera-ok");
