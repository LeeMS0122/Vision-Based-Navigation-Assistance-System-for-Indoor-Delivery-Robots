const imageForm = document.getElementById("uploadForm");
const imageFileInput = document.getElementById("imageFile");
const imageStatusText = document.getElementById("statusText");

const videoForm = document.getElementById("videoUploadForm");
const videoFileInput = document.getElementById("videoFile");
const sampleFpsInput = document.getElementById("sampleFps");
const videoStatusText = document.getElementById("videoStatusText");

const resultVideo = document.getElementById("resultVideo");
const frameSlider = document.getElementById("frameSlider");
const frameMetaText = document.getElementById("frameMetaText");
const modeText = document.getElementById("modeText");

const imageTargets = {
  original_image: document.getElementById("imgOriginal"),
  detection_overlay: document.getElementById("imgDetection"),
  risk_overlay: document.getElementById("imgRisk"),
  astar_on_rgb: document.getElementById("imgAstar"),
  drive_mask: document.getElementById("imgDrive"),
  obstacle_mask: document.getElementById("imgObstacle"),
  depth_visualization: document.getElementById("imgDepth"),
  cost_map: document.getElementById("imgCost"),
};

const objList = document.getElementById("objList");
const warningText = document.getElementById("warningText");
const riskText = document.getElementById("riskText");
const directionText = document.getElementById("directionText");
const recalcText = document.getElementById("recalcText");
const pathMeta = document.getElementById("pathMeta");
const jsonDump = document.getElementById("jsonDump");

let currentVideoFrames = [];
let currentVideoInfo = null;
let currentShownVideoFrameIndex = -1;

function setImageSource(target, url) {
  const slot = target.closest(".media-slot");
  if (!slot) return;

  if (!url) {
    target.removeAttribute("src");
    target.classList.remove("has-media");
    slot.classList.remove("has-media");
    target.hidden = true;
    return;
  }

  target.onload = () => {
    target.classList.add("has-media");
    slot.classList.add("has-media");
    target.hidden = false;
  };
  target.onerror = () => {
    target.classList.remove("has-media");
    slot.classList.remove("has-media");
    target.hidden = true;
  };
  target.src = url;
}

function setImages(images, options = {}) {
  const useCacheBust = options.cacheBust !== false;
  const cacheBuster = useCacheBust ? `t=${Date.now()}` : "";
  Object.entries(imageTargets).forEach(([key, target]) => {
    const url = images[key];
    if (!url) {
      setImageSource(target, "");
      return;
    }
    const finalUrl = useCacheBust ? `${url}?${cacheBuster}` : url;
    setImageSource(target, finalUrl);
  });
}

function setObjectList(objCounts) {
  objList.innerHTML = "";
  const entries = Object.entries(objCounts || {});
  if (!entries.length) {
    const li = document.createElement("li");
    li.textContent = "감지된 객체 없음";
    objList.appendChild(li);
    return;
  }
  entries.forEach(([name, count]) => {
    const li = document.createElement("li");
    li.textContent = `${name}: ${count}`;
    objList.appendChild(li);
  });
}

function setPanelFromImageSummary(summary) {
  modeText.textContent = "IMAGE";
  setObjectList(summary.detected_objects);
  warningText.textContent = summary.nearest_obstacle_warning;
  riskText.textContent = summary.current_risk;
  directionText.textContent = summary.recommended_direction;
  recalcText.textContent = summary.path_recalculation_required ? "YES" : "NO";
  pathMeta.textContent = `path_found=${summary.path_found}, path_length=${summary.path_length}, start=${JSON.stringify(summary.start)}, goal=${JSON.stringify(summary.goal)}`;
}

function setPanelFromVideoFrame(frame) {
  modeText.textContent = "VIDEO";
  if (frame.images) {
    setImages(frame.images, { cacheBust: false });
  }
  setObjectList(frame.summary.detected_objects || frame.status.detected_objects);
  warningText.textContent = frame.status.nearest_obstacle_warning;
  riskText.textContent = frame.status.current_risk;
  directionText.textContent = frame.status.recommended_direction;
  recalcText.textContent = frame.status.path_recalculation_required ? "YES" : "NO";
  pathMeta.textContent = `frame=${frame.frame_index}, t=${frame.timestamp_sec.toFixed(2)}s, path_found=${frame.status.path_found}, path_length=${frame.summary.path_length}`;
  frameMetaText.textContent = `프레임 정보: index=${frame.frame_index}, source=${frame.source_frame_index}, timestamp=${frame.timestamp_sec.toFixed(2)}s`;
}

function getFrameByVideoTime(currentTimeSec) {
  if (!currentVideoFrames.length) return null;

  let best = currentVideoFrames[0];
  let bestDiff = Math.abs(best.timestamp_sec - currentTimeSec);
  for (const frame of currentVideoFrames) {
    const diff = Math.abs(frame.timestamp_sec - currentTimeSec);
    if (diff < bestDiff) {
      best = frame;
      bestDiff = diff;
    }
  }
  return best;
}

function setFrameByIndex(idx) {
  if (!currentVideoFrames.length) return;
  const clamped = Math.max(0, Math.min(idx, currentVideoFrames.length - 1));
  const frame = currentVideoFrames[clamped];
  if (!frame) return;
  if (currentShownVideoFrameIndex === frame.frame_index) return;
  currentShownVideoFrameIndex = frame.frame_index;
  setPanelFromVideoFrame(frame);
  frameSlider.value = String(clamped);
}

imageForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!imageFileInput.files.length) {
    alert("이미지를 선택하세요.");
    return;
  }

  const formData = new FormData();
  formData.append("file", imageFileInput.files[0]);
  imageStatusText.textContent = "이미지 파이프라인 실행 중...";

  try {
    const res = await fetch("/api/pipeline/run", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Unknown error");

    setImages(data.images);
    setPanelFromImageSummary(data.summary);
    jsonDump.textContent = JSON.stringify(data, null, 2);
    imageStatusText.textContent = `완료 (run_id=${data.run_id})`;
  } catch (err) {
    imageStatusText.textContent = "실패";
    alert(`실행 실패: ${err.message}`);
  }
});

videoForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  if (!videoFileInput.files.length) {
    alert("영상을 선택하세요.");
    return;
  }

  const sampleFps = Number(sampleFpsInput.value || 5);
  const formData = new FormData();
  formData.append("file", videoFileInput.files[0]);
  formData.append("sample_fps", String(sampleFps));

  videoStatusText.textContent = "영상 프레임 분석 중...";

  try {
    const res = await fetch("/api/video/run", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Unknown error");

    currentVideoFrames = data.frames_metadata.frames || [];
    currentVideoInfo = data.frames_metadata.video_info || null;

    resultVideo.src = `${data.result_video_path}?t=${Date.now()}`;
    resultVideo.load();

    if (currentVideoFrames.length > 0) {
      frameSlider.min = "0";
      frameSlider.max = String(currentVideoFrames.length - 1);
      frameSlider.value = "0";
      currentShownVideoFrameIndex = -1;
      setFrameByIndex(0);
    }

    jsonDump.textContent = JSON.stringify(data, null, 2);
    videoStatusText.textContent = `완료 (run_id=${data.run_id}, frames=${currentVideoFrames.length})`;
  } catch (err) {
    videoStatusText.textContent = "실패";
    alert(`영상 실행 실패: ${err.message}`);
  }
});

resultVideo.addEventListener("timeupdate", () => {
  if (!currentVideoFrames.length) return;
  const frame = getFrameByVideoTime(resultVideo.currentTime);
  if (!frame) return;
  if (currentShownVideoFrameIndex === frame.frame_index) return;
  setPanelFromVideoFrame(frame);
  currentShownVideoFrameIndex = frame.frame_index;
  frameSlider.value = String(frame.frame_index);
});

frameSlider.addEventListener("input", () => {
  const idx = Number(frameSlider.value || 0);
  setFrameByIndex(idx);

  if (currentVideoInfo && currentVideoInfo.sample_fps_actual > 0) {
    resultVideo.currentTime = idx / currentVideoInfo.sample_fps_actual;
  }
});
