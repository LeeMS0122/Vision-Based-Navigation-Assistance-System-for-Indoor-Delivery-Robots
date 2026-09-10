from pathlib import Path
import heapq
import numpy as np
import cv2
from PIL import Image
import torch
from ultralytics import YOLO
import sys
sys.path.append("/home/minsu/disk_a/sc_depth_pl")

from SC_DepthV3 import SC_DepthV3
from config import get_opts

# =========================
# 경로 설정
# =========================
YOLO_MODEL_PATH = "/home/minsu/disk_a/miniconda3/graduation_work_robot_vision/runs/v8n_seg_b16_e50/weights/best.pt"
SCDEPTH_CKPT_PATH = "/home/minsu/disk_a/sc_depth_pl/ckpts/robot_scv3_full/version_0/epoch=16-val_loss=0.3801.ckpt"

DATA_ROOT = Path("/home/minsu/disk_a/robot_dataset_scdepth_v3_official/training")
TRAIN_TXT = DATA_ROOT / "train.txt"
VAL_TXT = DATA_ROOT / "val.txt"

OUTPUT_ROOT = Path("/home/minsu/disk_a/miniconda3/graduation_work_robot_vision/planner_dataset")

# =========================
# 설정
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCDEPTH_INFER_SIZE = (640, 384)
TARGET_SIZE = (160, 96)

DRIVE_CLASS_ID = 0
OBS_CLASS_ID = 1

DRIVABLE_BASE_COST = 1
NON_DRIVABLE_BASE_COST = 80
OBSTACLE_COST = 255
DEPTH_PENALTY_SCALE = 50

OBSTACLE_THRESHOLD = 255
START_MARGIN_BOTTOM = 5
GOAL_SEARCH_ROWS = 15
GOAL_CENTER_BIAS = 2.0

MAX_IMAGES_PER_SCENE = 3   # 처음엔 작게
FRAME_STRIDE = 100           # 너무 촘촘하지 않게


# =========================
# 공통 유틸
# =========================
def load_scene_list(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def resize_mask(mask: np.ndarray, size):
    resized = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    return (resized > 0).astype(np.uint8)


def resize_depth(depth: np.ndarray, size):
    return cv2.resize(depth.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)


def normalize_depth(depth: np.ndarray):
    d = depth.astype(np.float32).copy()
    p2 = np.percentile(d, 2)
    p98 = np.percentile(d, 98)
    if p98 - p2 < 1e-8:
        return np.zeros_like(d, dtype=np.float32)
    d = np.clip(d, p2, p98)
    d = (d - p2) / (p98 - p2)
    return d


# =========================
# YOLO
# =========================
def run_yolo_inference(model, image_path: str):
    results = model.predict(
        source=image_path,
        task="segment",
        save=False,
        verbose=False,
        conf=0.25,
        retina_masks=True,
    )

    r = results[0]
    orig_h, orig_w = r.orig_shape
    drive_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    obs_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

    if r.masks is None or r.boxes is None:
        return drive_mask, obs_mask

    masks = r.masks.data.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy().astype(int)

    for mask_i, cls_id in zip(masks, classes):
        binary_mask = (mask_i > 0.5).astype(np.uint8)
        if cls_id == DRIVE_CLASS_ID:
            drive_mask = np.maximum(drive_mask, binary_mask)
        elif cls_id == OBS_CLASS_ID:
            obs_mask = np.maximum(obs_mask, binary_mask)

    return drive_mask, obs_mask


# =========================
# SC-Depth
# =========================
def load_image_for_scdepth(img_path, resize=None):
    img = Image.open(img_path).convert("RGB")
    if resize is not None:
        img = img.resize(resize)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    return img_tensor


def run_scdepth_inference(model, image_path: str):
    img_tensor = load_image_for_scdepth(image_path, resize=SCDEPTH_INFER_SIZE).to(DEVICE)
    with torch.no_grad():
        outputs = model.depth_net(img_tensor)
        if isinstance(outputs, dict):
            if ("disp", 0) in outputs:
                pred = outputs[("disp", 0)]
            elif 0 in outputs:
                pred = outputs[0]
            else:
                pred = list(outputs.values())[0]
        else:
            pred = outputs
    return pred.squeeze().detach().cpu().numpy()


# =========================
# Cost Map
# =========================
def build_cost_map(drive_mask: np.ndarray, obs_mask: np.ndarray, depth_map: np.ndarray):
    drive_mask = resize_mask(drive_mask, TARGET_SIZE)
    obs_mask = resize_mask(obs_mask, TARGET_SIZE)
    depth_map = resize_depth(depth_map, TARGET_SIZE)

    depth_norm = normalize_depth(depth_map)
    depth_penalty = (1.0 - depth_norm) * DEPTH_PENALTY_SCALE

    cost_map = np.where(drive_mask == 1, DRIVABLE_BASE_COST, NON_DRIVABLE_BASE_COST).astype(np.float32)
    cost_map += depth_penalty
    cost_map[obs_mask == 1] = OBSTACLE_COST
    cost_map = np.clip(cost_map, 0, 255).astype(np.uint8)

    return cost_map


# =========================
# A*
# =========================
NEIGHBORS_8 = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)
]

def heuristic(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])

def get_neighbors(y, x, h, w):
    for dy, dx, mc in NEIGHBORS_8:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w:
            yield ny, nx, mc

def astar(cost_map, start, goal):
    h, w = cost_map.shape
    open_heap = []
    heapq.heappush(open_heap, (0.0, start))
    came_from = {}
    g_score = {start: 0.0}
    visited = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        cy, cx = current
        for ny, nx, move_cost in get_neighbors(cy, cx, h, w):
            if cost_map[ny, nx] >= OBSTACLE_THRESHOLD:
                continue

            neighbor = (ny, nx)
            tentative_g = g_score[current] + float(cost_map[ny, nx]) + move_cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f, neighbor))

    return None

def find_start(cost_map):
    h, w = cost_map.shape
    y = h - 1 - START_MARGIN_BOTTOM
    x_center = w // 2
    for offset in range(w // 2):
        for x in [x_center - offset, x_center + offset]:
            if 0 <= x < w and cost_map[y, x] < OBSTACLE_THRESHOLD:
                return (y, x)
    return None

def find_goal(cost_map):
    h, w = cost_map.shape
    x_min = int(w * 0.3)
    x_max = int(w * 0.7)
    x_center = w // 2
    max_row = min(GOAL_SEARCH_ROWS, h)

    candidates = []
    for y in range(max_row):
        for x in range(x_min, x_max):
            c = float(cost_map[y, x])
            if c >= OBSTACLE_THRESHOLD:
                continue
            score = c + GOAL_CENTER_BIAS * abs(x - x_center) + 0.5 * y
            candidates.append((score, (y, x)))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[0])
    return candidates[0][1]


# =========================
# 저장용 변환
# =========================
def point_to_map(pt, shape):
    h, w = shape
    m = np.zeros((h, w), dtype=np.uint8)
    if pt is not None:
        m[pt[0], pt[1]] = 1
    return m

def path_to_map(path, shape):
    h, w = shape
    m = np.zeros((h, w), dtype=np.uint8)
    if path is not None:
        for y, x in path:
            m[y, x] = 1
    return m


# =========================
# 메인
# =========================
def process_split(split_name: str, scenes, out_dir: Path, yolo_model, scdepth_model):
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_idx = 0

    for scene in scenes:
        scene_dir = DATA_ROOT / scene
        imgs = sorted(list(scene_dir.glob("*.png")) + list(scene_dir.glob("*.jpg")))

        picked = imgs[::FRAME_STRIDE][:MAX_IMAGES_PER_SCENE]

        for img_path in picked:
            drive_mask, obs_mask = run_yolo_inference(yolo_model, str(img_path))
            depth_map = run_scdepth_inference(scdepth_model, str(img_path))
            cost_map = build_cost_map(drive_mask, obs_mask, depth_map)

            start = find_start(cost_map)
            goal = find_goal(cost_map)
            if start is None or goal is None:
                continue

            path = astar(cost_map, start, goal)
            if path is None:
                continue

            start_map = point_to_map(start, cost_map.shape)
            goal_map = point_to_map(goal, cost_map.shape)
            path_map = path_to_map(path, cost_map.shape)

            prefix = f"sample_{sample_idx:05d}"
            np.save(out_dir / f"{prefix}_cost.npy", cost_map.astype(np.uint8))
            np.save(out_dir / f"{prefix}_start.npy", start_map.astype(np.uint8))
            np.save(out_dir / f"{prefix}_goal.npy", goal_map.astype(np.uint8))
            np.save(out_dir / f"{prefix}_path.npy", path_map.astype(np.uint8))

            sample_idx += 1
            print(f"[{split_name}] saved: {prefix}")

    print(f"[{split_name}] total samples: {sample_idx}")


def main():
    train_scenes = load_scene_list(TRAIN_TXT)
    val_scenes = load_scene_list(VAL_TXT)

    yolo_model = YOLO(YOLO_MODEL_PATH)

    hparams = get_opts()
    hparams.ckpt_path = SCDEPTH_CKPT_PATH
    scdepth_model = SC_DepthV3.load_from_checkpoint(SCDEPTH_CKPT_PATH, hparams=hparams)
    scdepth_model = scdepth_model.to(DEVICE)
    scdepth_model.eval()

    process_split(
        split_name="train",
        scenes=train_scenes,
        out_dir=OUTPUT_ROOT / "train",
        yolo_model=yolo_model,
        scdepth_model=scdepth_model
    )

    process_split(
        split_name="val",
        scenes=val_scenes,
        out_dir=OUTPUT_ROOT / "val",
        yolo_model=yolo_model,
        scdepth_model=scdepth_model
    )

    print("\nplanner dataset 생성 완료")
    print(OUTPUT_ROOT)


if __name__ == "__main__":
    main()