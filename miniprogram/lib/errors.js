function normalizeErrorMessage(error) {
  if (!error) {
    return "";
  }
  if (typeof error === "string") {
    return error;
  }
  return String(error.message || error.errMsg || error.detail || error);
}

function formatUserFacingError(error, options) {
  const message = normalizeErrorMessage(error);
  const action = options && options.action ? options.action : "";

  if (/Network request failed|request:fail|timeout|ECONNREFUSED|Failed to fetch/i.test(message)) {
    return "请稍后重试或到设置页测试连接。";
  }
  if (action === "resume" && /Learning session not found|session.*not found|404/i.test(message)) {
    return "这个学习会话不存在或已失效，请返回学习列表重新开始。";
  }
  if (/Active tutor item not found|错题陪练已经结束|no active tutor/i.test(message)) {
    return "本次陪练已经结束，请返回本次总结。";
  }
  if (/child_id|child id|孩子档案|child.*required/i.test(message)) {
    return "请先创建或选择孩子档案，再开始学习。";
  }
  return message || "请求失败，请稍后重试。";
}

module.exports = {
  formatUserFacingError,
};
