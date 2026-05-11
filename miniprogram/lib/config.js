const STORAGE_KEY = "songguo_backend_config";
const LEGACY_STORAGE_KEY = ["deep", "tutor_backend_config"].join("");
const DEFAULT_HTTP_BASE_URL = "http://127.0.0.1:8001";

function trimTrailingSlash(value) {
  return String(value || "").trim().replace(/\/+$/, "");
}

function normalizeHttpBaseUrl(value) {
  const normalized = trimTrailingSlash(value);
  return normalized || DEFAULT_HTTP_BASE_URL;
}

function normalizeConfig(config) {
  const httpBaseUrl = normalizeHttpBaseUrl(config && config.httpBaseUrl);
  return {
    httpBaseUrl,
  };
}

function getBackendConfig() {
  try {
    const stored =
      wx.getStorageSync(STORAGE_KEY) ||
      wx.getStorageSync(LEGACY_STORAGE_KEY) ||
      {};
    return normalizeConfig(stored);
  } catch (_error) {
    return normalizeConfig({});
  }
}

function saveBackendConfig(nextConfig) {
  const merged = normalizeConfig({
    ...getBackendConfig(),
    ...(nextConfig || {}),
  });
  wx.setStorageSync(STORAGE_KEY, merged);
  return merged;
}

function resetBackendConfig() {
  const defaults = {
    httpBaseUrl: DEFAULT_HTTP_BASE_URL,
  };
  wx.setStorageSync(STORAGE_KEY, defaults);
  return defaults;
}

module.exports = {
  DEFAULT_HTTP_BASE_URL,
  getBackendConfig,
  normalizeHttpBaseUrl,
  resetBackendConfig,
  saveBackendConfig,
};
