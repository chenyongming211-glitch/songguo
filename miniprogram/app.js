const { getBackendConfig } = require("./lib/config");
const { loginWithWechat } = require("./lib/api");

const SESSION_STORAGE_KEY = "songguo_wechat_session";
const LEGACY_SESSION_STORAGE_KEY = ["deep", "tutor_wechat_session"].join("");

App({
  globalData: {
    wechatSession: null,
  },

  onLaunch() {
    getBackendConfig();
    this.login();
  },

  login() {
    if (!wx.login) {
      return;
    }
    wx.login({
      success: async (response) => {
        if (!response.code) {
          return;
        }
        try {
          this.globalData.wechatSession = await loginWithWechat(response.code);
          wx.setStorageSync(SESSION_STORAGE_KEY, this.globalData.wechatSession);
        } catch (_error) {
          this.globalData.wechatSession =
            wx.getStorageSync(SESSION_STORAGE_KEY) ||
            wx.getStorageSync(LEGACY_SESSION_STORAGE_KEY) ||
            null;
        }
      },
    });
  },
});
