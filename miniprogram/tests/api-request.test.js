const assert = require("assert");

async function run() {
  const requests = [];
  const removedStorageKeys = [];

  global.wx = {
    getStorageSync(key) {
      if (key === "songguo_wechat_session") {
        return {
          child_id: "child_openid_local_dev",
          session_token: "expired-token",
        };
      }
      return {};
    },
    removeStorageSync(key) {
      removedStorageKeys.push(key);
    },
    request(options) {
      requests.push(options);
      if (requests.length === 1) {
        options.success({
          statusCode: 401,
          data: { detail: "Session token expired" },
        });
        return;
      }
      options.success({
        statusCode: 200,
        data: [{ child_id: "child_openid_local_dev", name: "默认孩子" }],
      });
    },
  };

  const { listChildren } = require("../lib/api");
  const children = await listChildren();

  assert.deepEqual(children, [
    { child_id: "child_openid_local_dev", name: "默认孩子" },
  ]);
  assert.equal(requests.length, 2);
  assert.equal(requests[0].header["X-Session-Token"], "expired-token");
  assert.equal(requests[1].header["X-Session-Token"], undefined);
  assert.deepEqual(removedStorageKeys, [
    "songguo_wechat_session",
    ["deep", "tutor_wechat_session"].join(""),
  ]);
}

run()
  .then(() => {
    console.log("api-request-ok");
  })
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
