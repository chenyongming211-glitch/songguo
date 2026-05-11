const assert = require("assert");

async function run() {
  const requests = [];

  global.wx = {
    getStorageSync() {
      return {};
    },
    request(options) {
      requests.push(options);
      options.success({
        statusCode: 200,
        data: { ok: true },
      });
    },
    uploadFile(options) {
      requests.push(options);
      options.success({
        statusCode: 200,
        data: JSON.stringify({
          question_text: "一根彩带2米35厘米，剪去80厘米，还剩多少厘米？",
          child_answer: "155厘米",
        }),
      });
    },
  };

  const {
    confirmLearningSubmission,
    createLearningSubmission,
    getLearningSubmission,
    recognizeSubmissionPhoto,
    startNextSubmissionTutorItem,
    submitSubmissionTutorAttempt,
  } = require("../lib/api");

  await createLearningSubmission({
    childId: "child_001",
    subject: "math",
    grade: 4,
    sourceType: "text",
    rawText: "48 ÷ 6 = ?\n孩子答案：8",
  });
  await getLearningSubmission("sub_001", "child_001");
  await confirmLearningSubmission("sub_001", { rawText: "48 ÷ 6 = ?\n孩子答案：8" });
  await startNextSubmissionTutorItem("sub_001");
  await submitSubmissionTutorAttempt("sub_001", "8");
  const photoDraft = await recognizeSubmissionPhoto("/tmp/homework.txt", "child_001", "math");

  assert.equal(requests[0].url.endsWith("/api/v1/learning/submissions"), true);
  assert.equal(requests[0].method, "POST");
  assert.deepEqual(requests[0].data, {
    child_id: "child_001",
    subject: "math",
    grade: 4,
    source_type: "text",
    raw_text: "48 ÷ 6 = ?\n孩子答案：8",
  });
  assert.equal(
    requests[1].url.endsWith("/api/v1/learning/submissions/sub_001?child_id=child_001"),
    true
  );
  assert.equal(
    requests[2].url.endsWith("/api/v1/learning/submissions/sub_001/confirm"),
    true
  );
  assert.equal(requests[2].method, "POST");
  assert.deepEqual(requests[2].data, { raw_text: "48 ÷ 6 = ?\n孩子答案：8" });
  assert.equal(
    requests[3].url.endsWith("/api/v1/learning/submissions/sub_001/tutor/next"),
    true
  );
  assert.equal(
    requests[4].url.endsWith("/api/v1/learning/submissions/sub_001/tutor/attempt"),
    true
  );
  assert.deepEqual(requests[4].data, { child_answer: "8" });
  assert.equal(requests[5].url.endsWith("/api/v1/learning/submissions/photo-draft"), true);
  assert.equal(requests[5].filePath, "/tmp/homework.txt");
  assert.deepEqual(requests[5].formData, {
    child_id: "child_001",
    subject: "math",
    grade: 3,
  });
  assert.equal(photoDraft.question_text, "一根彩带2米35厘米，剪去80厘米，还剩多少厘米？");
  assert.equal(photoDraft.child_answer, "155厘米");
}

run()
  .then(() => {
    console.log("submission-api-ok");
  })
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
