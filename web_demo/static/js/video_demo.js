const videoFile = document.getElementById("videoFile");
const sampleFps = document.getElementById("sampleFps");
const statusText = document.getElementById("statusText");
const form = document.getElementById("videoForm");
const resultVideo = document.getElementById("resultVideo");
const resultImage = document.getElementById("resultImage");
const goalCanvas = document.getElementById("goalCanvas");
const goalMarker = document.getElementById("goalMarker");
const jsonDump = document.getElementById("jsonDump");

let goal = null;

videoFile.addEventListener("change", async () => {
  const file = videoFile.files && videoFile.files[0];
  if (!file) return;
  statusText.textContent = "첫 프레임 로딩 중...";

  const url = URL.createObjectURL(file);
  const v = document.createElement("video");
  v.preload = "metadata";
  v.src = url;
  await new Promise((resolve, reject) => {
    v.onloadedmetadata = () => resolve();
    v.onerror = () => reject(new Error("영상 메타데이터 로딩 실패"));
  });
  v.currentTime = 0;
  await new Promise((resolve) => {
    v.onseeked = () => resolve();
  });

  const c = document.createElement("canvas");
  c.width = v.videoWidth;
  c.height = v.videoHeight;
  c.getContext("2d").drawImage(v, 0, 0, c.width, c.height);
  goalCanvas.src = c.toDataURL("image/jpeg", 0.9);
  URL.revokeObjectURL(url);
  goal = null;
  goalMarker.style.display = "none";
  statusText.textContent = "첫 프레임 표시됨. Goal을 클릭하세요.";
});

goalCanvas.addEventListener("click", (e) => {
  if (!goalCanvas.src) {
    alert("먼저 영상을 선택해 첫 프레임을 표시하세요.");
    return;
  }
  const r = goalCanvas.getBoundingClientRect();
  const cx = e.clientX - r.left;
  const cy = e.clientY - r.top;
  goal = { click_x: cx, click_y: cy, display_w: r.width, display_h: r.height };
  goalMarker.style.left = `${cx}px`;
  goalMarker.style.top = `${cy}px`;
  goalMarker.style.display = "block";
  statusText.textContent = `Goal set (${Math.round(goal.click_x)}, ${Math.round(goal.click_y)})`;
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!videoFile.files.length) return alert("영상을 선택하세요.");
  if (!goal) return alert("먼저 Goal을 클릭하세요.");

  const fd = new FormData();
  fd.append("file", videoFile.files[0]);
  fd.append("sample_fps", String(Number(sampleFps.value || 5)));
  fd.append("goal_click_x", String(goal.click_x));
  fd.append("goal_click_y", String(goal.click_y));
  fd.append("display_w", String(Math.round(goal.display_w)));
  fd.append("display_h", String(Math.round(goal.display_h)));

  statusText.textContent = "분석 중...";
  const res = await fetch("/video-demo/api/run", { method: "POST", body: fd });
  const data = await res.json();
  if (!res.ok) return alert(data.detail || "실패");

  resultVideo.src = data.result_video_path + `?t=${Date.now()}`;
  const frames = data.frame_statuses.frames || [];
  if (frames.length) resultImage.src = frames[0].images.astar_on_rgb + `?t=${Date.now()}`;
  jsonDump.textContent = JSON.stringify(data, null, 2);
  statusText.textContent = "완료";
});
