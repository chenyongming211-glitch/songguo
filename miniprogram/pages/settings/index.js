const { getSystemStatus } = require("../../lib/api");
const {
  getBackendConfig,
  isDevtoolsRuntime,
  resetBackendConfig,
  saveBackendConfig,
} = require("../../lib/config");
const { formatUserFacingError } = require("../../lib/errors");
const { formatTimestamp } = require("../../lib/utils");

Page({
  data: {
    httpBaseUrl: "",
    status: null,
    lastChecked: "",
    testing: false,
    error: "",
    wechatMode: "本地未登录 / 可使用 Mock",
    advancedDiagnosticsVisible: false,
    diagnosticsAvailable: false,
    serviceLabel: "生产学习服务",
  },

  onShow() {
    this.loadConfig();
    this.refreshWechatMode();
  },

  loadConfig() {
    const config = getBackendConfig();
    const diagnosticsAvailable = isDevtoolsRuntime();
    this.setData({
      httpBaseUrl: config.httpBaseUrl,
      diagnosticsAvailable,
      advancedDiagnosticsVisible: diagnosticsAvailable ? this.data.advancedDiagnosticsVisible : false,
      serviceLabel: diagnosticsAvailable ? "开发调试服务" : "生产学习服务",
    });
  },

  handleHttpInput(event) {
    this.setData({ httpBaseUrl: event.detail.value || "" });
  },

  handleToggleAdvancedDiagnostics() {
    if (!this.data.diagnosticsAvailable) {
      return;
    }
    this.setData({
      advancedDiagnosticsVisible: !this.data.advancedDiagnosticsVisible,
    });
  },

  refreshWechatMode() {
    const session =
      wx.getStorageSync("songguo_wechat_session") ||
      wx.getStorageSync(["deep", "tutor_wechat_session"].join("")) ||
      {};
    const openid = session.openid || "";
    let wechatMode = "本地未登录 / 可使用 Mock";
    if (openid && /^mock/i.test(openid)) {
      wechatMode = "Mock 微信登录";
    } else if (openid) {
      wechatMode = "真实微信登录";
    }
    this.setData({ wechatMode });
  },

  handleSave() {
    const config = saveBackendConfig({
      httpBaseUrl: this.data.httpBaseUrl,
    });
    this.setData({
      httpBaseUrl: config.httpBaseUrl,
      error: "",
    });
    wx.showToast({
      title: "配置已保存",
      icon: "success",
    });
  },

  handleReset() {
    const config = resetBackendConfig();
    this.setData({
      httpBaseUrl: config.httpBaseUrl,
      status: null,
      lastChecked: "",
      error: "",
    });
    wx.showToast({
      title: "已恢复默认值",
      icon: "success",
    });
  },

  async handleTest() {
    this.setData({
      testing: true,
      error: "",
    });
    try {
      const status = await getSystemStatus();
      this.setData({
        status,
        lastChecked: formatTimestamp(Date.now()),
      });
      wx.showToast({
        title: "连接成功",
        icon: "success",
      });
    } catch (error) {
      this.setData({
        error: formatUserFacingError(error),
      });
    } finally {
      this.setData({ testing: false });
    }
  },
});
