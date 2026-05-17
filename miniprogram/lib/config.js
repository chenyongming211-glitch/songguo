const STORAGE_KEY = "songguo_backend_config";
const LEGACY_STORAGE_KEY = ["deep", "tutor_backend_config"].join("");
const DEFAULT_HTTP_BASE_URL = "https://api.songguoxue.com";
const LOCAL_DEBUG_HOST_PATTERN =
  /^https?:\/\/(?:localhost|127(?:\.\d{1,3}){0,3}|0\.0\.0\.0|\[::1\]|::1)(?::|\/|$)/i;

function trimTrailingSlash(value) {
  return String(value || "").trim().replace(/\/+$/, "");
}

function isDevtoolsRuntime() {
  try {
    const info = wx.getSystemInfoSync ? wx.getSystemInfoSync() : {};
    return info && info.platform === "devtools";
  } catch (_error) {
    return false;
  }
}

function shouldUseProductionDefault(value) {
  if (!value || isDevtoolsRuntime()) {
    return false;
  }
  return !/^https:\/\//i.test(value) || LOCAL_DEBUG_HOST_PATTERN.test(value);
}

function normalizeHttpBaseUrl(value) {
  const normalized = trimTrailingSlash(value);
  if (shouldUseProductionDefault(normalized)) {
    return DEFAULT_HTTP_BASE_URL;
  }
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
  isDevtoolsRuntime,
  normalizeHttpBaseUrl,
  resetBackendConfig,
  saveBackendConfig,
};
