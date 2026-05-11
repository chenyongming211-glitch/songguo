const assert = require("assert");

const { formatUserFacingError } = require("../lib/errors");

assert.equal(
  formatUserFacingError(new Error("Network request failed")),
  "后端暂时连接不上，请到设置页检查服务地址或本地服务是否启动。"
);

assert.equal(
  formatUserFacingError(new Error("Learning session not found"), { action: "resume" }),
  "这个学习会话不存在或已失效，请返回学习列表重新开始。"
);

assert.equal(
  formatUserFacingError(new Error("child_id is required")),
  "请先创建或选择孩子档案，再开始学习。"
);

assert.equal(
  formatUserFacingError(new Error("Active tutor item not found")),
  "本次错题陪练已经结束，请返回本次总结。"
);

assert.equal(formatUserFacingError(new Error("受控教学输入被拦截")), "受控教学输入被拦截");

console.log("error-copy-ok");
