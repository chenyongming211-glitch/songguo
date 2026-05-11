function buildStreamingFrames(value, chunkSize) {
  const text = String(value || "");
  if (!text) {
    return [];
  }
  const size = Math.max(1, Number(chunkSize || 8));
  const frames = [];
  for (let end = size; end < text.length; end += size) {
    frames.push(text.slice(0, end));
  }
  frames.push(text);
  return frames;
}

module.exports = {
  buildStreamingFrames,
};
