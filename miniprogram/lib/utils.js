function formatTimestamp(value) {
  const numeric = Number(value || 0);
  if (!numeric) {
    return "";
  }
  const millis = numeric < 100000000000 ? numeric * 1000 : numeric;
  const date = new Date(millis);
  if (Number.isNaN(date.getTime())) {
    return "";
  }

  const month = `${date.getMonth() + 1}`.padStart(2, "0");
  const day = `${date.getDate()}`.padStart(2, "0");
  const hour = `${date.getHours()}`.padStart(2, "0");
  const minute = `${date.getMinutes()}`.padStart(2, "0");
  return `${month}-${day} ${hour}:${minute}`;
}

function shortenText(value, maxLength) {
  const text = String(value || "").trim().replace(/\s+/g, " ");
  if (!text) {
    return "";
  }
  const limit = Number(maxLength || 0) || 72;
  if (text.length <= limit) {
    return text;
  }
  return `${text.slice(0, limit)}...`;
}

function formatCapability(value) {
  const capability = String(value || "").trim();
  if (capability === "deep_solve") {
    return "Deep Solve";
  }
  if (capability === "deep_question") {
    return "Deep Question";
  }
  return "Chat";
}

let localMessageCounter = 0;

function buildMessageId(prefix) {
  localMessageCounter += 1;
  return `${prefix}-${Date.now()}-${localMessageCounter}`;
}

function makeLocalMessage(role, content, createdAt, fixedId) {
  return {
    clientId: fixedId || buildMessageId(role || "message"),
    role: role || "assistant",
    roleLabel: role === "user" ? "你" : "松鼠博士",
    content: String(content || ""),
    createdAt: Number(createdAt || Date.now()),
    createdLabel: formatTimestamp(createdAt || Date.now()),
  };
}

module.exports = {
  formatCapability,
  formatTimestamp,
  makeLocalMessage,
  shortenText,
};
