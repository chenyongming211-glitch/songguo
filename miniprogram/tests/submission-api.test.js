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
          raw_text: "一根彩带2米35厘米，剪去80厘米，还剩多少厘米？\n孩子答案：155厘米",
          question_text: "一根彩带2米35厘米，剪去80厘米，还剩多少厘米？",
          child_answer: "155厘米",
          confidence: 0.91,
          quality_warnings: [],
          quality_message: "",
          detected_regions: [{ x: 60, y: 80, width: 880, height: 720 }],
          preprocess_source: "opencv_document_projection_v0.2",
          items: [
            {
              item_index: 1,
              question_text: "一根彩带2米35厘米，剪去80厘米，还剩多少厘米？",
              child_answer: "155厘米",
              confidence: 0.91,
              bbox: { x: 60, y: 80, width: 880, height: 720 },
              ocr_action: "RecognizeEduPaperCut",
              ocr_source: "aliyun_edu_paper_cut",
              ocr_judgement: "",
              marking_source: "",
              correct_answer: "",
              evidence_points: [],
              display_status: "pending",
              quality_warnings: [],
            },
          ],
        }),
      });
    },
  };

  const {
    confirmLearningSubmission,
    createLearningSubmission,
    getLearningSubmission,
    recognizeSubmissionPhoto,
    recognizeSubmissionVoice,
    runLearningSubmissionVisualFallbackStep,
    startNextSubmissionTutorItem,
    submitSubmissionTutorAttempt,
  } = require("../lib/api");

  await createLearningSubmission({
    childId: "child_001",
    grade: 4,
    sourceType: "text",
    rawText: "48 ÷ 6 = ?\n孩子答案：8",
  });
  await getLearningSubmission("sub_001", "child_001");
  await confirmLearningSubmission("sub_001", { rawText: "48 ÷ 6 = ?\n孩子答案：8" });
  await startNextSubmissionTutorItem("sub_001");
  await submitSubmissionTutorAttempt("sub_001", "8");
  const photoDraft = await recognizeSubmissionPhoto("/tmp/homework.txt", "child_001");
  const voiceDraft = await recognizeSubmissionVoice("/tmp/homework.mp3", "child_001");
  await createLearningSubmission({
    childId: "child_001",
    grade: 4,
    sourceType: "photo",
    rawText: photoDraft.raw_text,
    imageRefs: ["artifact://photo_001"],
    draftItems: photoDraft.items,
  });
  await confirmLearningSubmission("sub_002", {
    rawText: photoDraft.raw_text,
    startTutor: false,
  });
  await runLearningSubmissionVisualFallbackStep("sub_002", 1);

  assert.equal(requests[0].url.endsWith("/api/v1/learning/submissions"), true);
  assert.equal(requests[0].method, "POST");
  assert.deepEqual(requests[0].data, {
    child_id: "child_001",
    subject: "auto",
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
    subject: "auto",
    grade: 3,
  });
  assert.equal(photoDraft.question_text, "一根彩带2米35厘米，剪去80厘米，还剩多少厘米？");
  assert.equal(photoDraft.child_answer, "155厘米");
  assert.equal(photoDraft.confidence, 0.91);
  assert.deepEqual(photoDraft.quality_warnings, []);
  assert.equal(photoDraft.quality_message, "");
  assert.equal(photoDraft.preprocess_source, "opencv_document_projection_v0.2");
  assert.equal(photoDraft.items[0].item_index, 1);
  assert.deepEqual(photoDraft.items[0].bbox, { x: 60, y: 80, width: 880, height: 720 });
  assert.equal(photoDraft.items[0].ocr_action, "RecognizeEduPaperCut");
  assert.equal(photoDraft.items[0].ocr_source, "aliyun_edu_paper_cut");
  assert.equal(photoDraft.items[0].display_status, "pending");
  assert.deepEqual(photoDraft.items[0].quality_warnings, []);
  assert.equal(requests[6].url.endsWith("/api/v1/learning/submissions/voice-draft"), true);
  assert.equal(requests[6].filePath, "/tmp/homework.mp3");
  assert.deepEqual(requests[6].formData, {
    child_id: "child_001",
    subject: "auto",
    grade: 3,
  });
  assert.equal(voiceDraft.question_text, "一根彩带2米35厘米，剪去80厘米，还剩多少厘米？");
  assert.deepEqual(requests[7].data.draft_items, [
    {
      item_index: 1,
      bbox: { x: 60, y: 80, width: 880, height: 720 },
      ocr_action: "RecognizeEduPaperCut",
      ocr_source: "aliyun_edu_paper_cut",
      ocr_judgement: "",
      marking_source: "",
      correct_answer: "",
      evidence_points: [],
      display_status: "pending",
      quality_warnings: [],
    },
  ]);
  assert.deepEqual(requests[7].data.image_refs, ["artifact://photo_001"]);
  assert.equal(
    requests[8].url.endsWith("/api/v1/learning/submissions/sub_002/confirm"),
    true
  );
  assert.deepEqual(requests[8].data, {
    raw_text: photoDraft.raw_text,
    start_tutor: false,
  });
  assert.equal(
    requests[9].url.endsWith("/api/v1/learning/submissions/sub_002/visual-fallback/step?max_items=1"),
    true
  );
  assert.equal(requests[9].method, "POST");
}

run()
  .then(() => {
    console.log("submission-api-ok");
  })
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
