const { getSystemStatus } = require("../../lib/api");
const {
  getBackendConfig,
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
  },

  onShow() {
    this.loadConfig();
    this.refreshWechatMode();
  },

  loadConfig() {
    const config = getBackendConfig();
    this.setData({
      httpBaseUrl: config.httpBaseUrl,
    });
  },

  handleHttpInput(event) {
    this.setData({ httpBaseUrl: event.detail.value || "" });
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
