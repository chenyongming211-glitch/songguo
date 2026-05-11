const assert = require("assert");

const { buildStreamingFrames } = require("../lib/streaming-text");

const text = "这题先抓住位置方向。小红从右往左第5个，我们先把它换成从左边数的位置。";
const frames = buildStreamingFrames(text, 8);

assert.ok(frames.length > 1, "long text should be split into multiple frames");
assert.equal(frames[frames.length - 1], text);
assert.ok(frames.every((frame, index) => index === 0 || frame.startsWith(frames[index - 1])));
assert.equal(buildStreamingFrames("", 8).length, 0);
assert.deepEqual(buildStreamingFrames("短句", 8), ["短句"]);

console.log("streaming-text-ok");
