const assert = require("assert");

function loadConfigWithWx(wxMock) {
  delete require.cache[require.resolve("../lib/config")];
  global.wx = wxMock;
  return require("../lib/config");
}

function run() {
  const phoneConfig = loadConfigWithWx({
    getSystemInfoSync() {
      return { platform: "ios" };
    },
    getStorageSync(key) {
      if (key === "songguo_backend_config") {
        return { httpBaseUrl: "http://127.0.0.1:8001" };
      }
      return {};
    },
  });
  assert.equal(
    phoneConfig.getBackendConfig().httpBaseUrl,
    phoneConfig.DEFAULT_HTTP_BASE_URL
  );
  assert.equal(phoneConfig.isDevtoolsRuntime(), false);

  const devtoolsConfig = loadConfigWithWx({
    getSystemInfoSync() {
      return { platform: "devtools" };
    },
    getStorageSync(key) {
      if (key === "songguo_backend_config") {
        return { httpBaseUrl: "http://127.0.0.1:8001" };
      }
      return {};
    },
  });
  assert.equal(devtoolsConfig.getBackendConfig().httpBaseUrl, "http://127.0.0.1:8001");
  assert.equal(devtoolsConfig.isDevtoolsRuntime(), true);
}

try {
  run();
  console.log("backend-config-ok");
} catch (error) {
  console.error(error);
  process.exit(1);
}
