const startBtn = document.getElementById("startBtn");
const camBtn = document.getElementById("camBtn");
const runBtn = document.getElementById("runBtn");
const stopBtn = document.getElementById("stopBtn");
const statusText = document.getElementById("statusText");
const cam = document.getElementById("cam");
const cap = document.getElementById("cap");
const jsonDump = document.getElementById("jsonDump");
const objList = document.getElementById("objList");
const warningText = document.getElementById("warningText");
const riskText = document.getElementById("riskText");
const directionText = document.getElementById("directionText");
const recalcText = document.getElementById("recalcText");
const pathMeta = document.getElementById("pathMeta");

const imageTargets = {
  astar_on_rgb: document.getElementById("imgAstar"),
  detection_overlay: document.getElementById("imgDetection"),
  risk_overlay: document.getElementById("imgRisk"),
  drive_mask: document.getElementById("imgDrive"),
  obstacle_mask: document.getElementById("imgObstacle"),
  depth_visualization: document.getElementById("imgDepth"),
  cost_map: document.getElementById("imgCost"),
};

let sessionId = null;
let timer = null;
let stream = null;
let frameCounter = 0;
let lastAuxUpdateAt = 0;

const SEND_INTERVAL_MS = 400;
const AUX_UPDATE_INTERVAL_MS = 1000;

function setImageSource(target, url) {
  const slot = target.closest(".media-slot");
  if (!slot) return;
  if (!url) {
    target.removeAttribute("src");
    target.classList.remove("has-media");
    target.hidden = true;
    slot.classList.remove("has-media");
    return;
  }
  target.onload = () => {
    target.classList.add("has-media");
    target.hidden = false;
    slot.classList.add("has-media");
  };
  target.onerror = () => {
    target.classList.remove("has-media");
    target.hidden = true;
    slot.classList.remove("has-media");
  };
  target.src = `${url}?t=${Date.now()}`;
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

function updateStatusPanel(statusJson, frameIndex) {
  setObjectList(statusJson.detected_objects || {});
  warningText.textContent = statusJson.near_obstacle_warning || statusJson.nearest_obstacle_warning || "-";
  riskText.textContent = statusJson.current_risk || "-";
  directionText.textContent = statusJson.recommended_direction || "-";
  recalcText.textContent = statusJson.replanning_required ? "YES" : "NO";
  pathMeta.textContent = `frame=${frameIndex}, path_found=${statusJson.path_found}, goal_active=${statusJson.goal_active}`;
  jsonDump.textContent = JSON.stringify(statusJson, null, 2);
}

startBtn.onclick = async () => {
  const res = await fetch("/live-demo/api/session", { method: "POST" });
  const data = await res.json();
  sessionId = data.session_id;
  frameCounter = 0;
  lastAuxUpdateAt = 0;
  statusText.textContent = `session=${sessionId}`;
};

camBtn.onclick = async () => {
  stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  cam.srcObject = stream;
};

cam.addEventListener("click", async (e) => {
  if (!sessionId) return alert("세션부터 시작하세요.");
  const r = cam.getBoundingClientRect();
  const body = {
    session_id: sessionId,
    click_x: e.clientX - r.left,
    click_y: e.clientY - r.top,
    display_w: Math.round(r.width),
    display_h: Math.round(r.height),
  };
  const res = await fetch("/live-demo/api/goal", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!res.ok) alert("goal 설정 실패");
});

async function sendFrame() {
  if (!sessionId || !cam.videoWidth) return;
  const ctx = cap.getContext("2d");
  cap.width = cam.videoWidth;
  cap.height = cam.videoHeight;
  ctx.drawImage(cam, 0, 0, cap.width, cap.height);
  const b64 = cap.toDataURL("image/jpeg", 0.7);

  const res = await fetch("/live-demo/api/frame", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, image_base64: b64 }),
  });
  const data = await res.json();
  if (!res.ok) return;

  // Always refresh main path view every tick for live feedback.
  setImageSource(imageTargets.astar_on_rgb, data.analysis_image_path || (data.images && data.images.astar_on_rgb) || "");
  updateStatusPanel(data.status_json || {}, data.frame_index);

  // Refresh auxiliary panels at lower cadence to keep UI responsive.
  const now = Date.now();
  if (now - lastAuxUpdateAt >= AUX_UPDATE_INTERVAL_MS && data.images) {
    setImageSource(imageTargets.detection_overlay, data.images.detection_overlay || "");
    setImageSource(imageTargets.risk_overlay, data.images.risk_overlay || "");
    setImageSource(imageTargets.drive_mask, data.images.drive_mask || "");
    setImageSource(imageTargets.obstacle_mask, data.images.obstacle_mask || "");
    setImageSource(imageTargets.depth_visualization, data.images.depth_visualization || "");
    setImageSource(imageTargets.cost_map, data.images.cost_map || "");
    lastAuxUpdateAt = now;
  }

  frameCounter += 1;
  statusText.textContent = `frame=${data.frame_index}, risk=${data.status_json.current_risk}`;
}

runBtn.onclick = () => {
  if (timer) clearInterval(timer);
  timer = setInterval(sendFrame, SEND_INTERVAL_MS);
};

stopBtn.onclick = () => {
  if (timer) clearInterval(timer);
  timer = null;
};
