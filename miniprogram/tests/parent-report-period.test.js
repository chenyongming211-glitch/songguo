const assert = require("assert");

async function run() {
  let pageDefinition = null;
  const storage = {};

  global.wx = {
    getStorageSync(key) {
      return storage[key] || "";
    },
    setStorageSync(key, value) {
      storage[key] = value;
    },
    stopPullDownRefresh() {},
  };

  global.Page = (definition) => {
    pageDefinition = definition;
  };

  delete require.cache[require.resolve("../pages/parent-report/index.js")];
  require("../pages/parent-report/index.js");

  assert.ok(pageDefinition, "parent report page should register itself");
  assert.deepEqual(pageDefinition.data.reviewScopeOptions, [
    "本周",
    "本月",
    "本季度",
    "本学期",
    "本年度",
  ]);
  assert.equal(pageDefinition.data.reviewScopeIndex, 0);
  assert.equal(typeof pageDefinition.handleScopePickerChange, "function");

  const page = {
    data: { ...pageDefinition.data },
    loadCount: 0,
    setData(patch) {
      this.data = { ...this.data, ...patch };
    },
    loadReport() {
      this.loadCount += 1;
    },
  };

  pageDefinition.handleScopePickerChange.call(page, { detail: { value: 2 } });

  assert.equal(page.data.reviewScope, "quarterly");
  assert.equal(page.data.reviewScopeLabel, "本季度");
  assert.equal(page.data.reviewScopeIndex, 2);
  assert.equal(page.loadCount, 1);
}

run()
  .then(() => {
    console.log("parent-report-period-ok");
  })
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
