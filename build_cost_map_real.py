from pathlib import Path
import sys
sys.path.append("/home/minsu/disk_a/sc_depth_pl")

import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO

from SC_DepthV3 import SC_DepthV3
from config import get_opts

# =========================
# 경로 설정
# =========================
YOLO_MODEL_PATH = "/home/minsu/disk_a/miniconda3/graduation_work_robot_vision/runs/v8n_seg_b16_e50/weights/best.pt"
SCDEPTH_CKPT_PATH = "/home/minsu/disk_a/sc_depth_pl/ckpts/robot_scv3_full/version_0/epoch=16-val_loss=0.3801.ckpt"

IMAGE_PATH = "/home/minsu/disk_a/robot_dataset_scdepth_v3_official/training/Scene002/000000.png"

OUTPUT_DIR = Path("/home/minsu/disk_a/miniconda3/graduation_work_robot_vision/cost_map_real_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 설정
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SC-Depth 추론용 입력 크기
SCDEPTH_INFER_SIZE = (640, 384)  # (width, height)

# planning 해상도
TARGET_SIZE = (160, 96)  # (width, height)

# 클래스 ID
DRIVE_CLASS_ID = 0
OBS_CLASS_ID = 1

# cost parameters
DRIVABLE_BASE_COST = 1
NON_DRIVABLE_BASE_COST = 80
OBSTACLE_COST = 255
DEPTH_PENALTY_SCALE = 50


# =========================
# 유틸
# =========================
def save_mask(mask: np.ndarray, save_path: Path):
    img = (mask * 255).astype(np.uint8)
    cv2.imwrite(str(save_path), img)


def save_depth_vis(depth_tensor_or_array, save_path: Path):
    if isinstance(depth_tensor_or_array, torch.Tensor):
        depth = depth_tensor_or_array.squeeze().detach().cpu().numpy()
    else:
        depth = depth_tensor_or_array

    dmin, dmax = np.percentile(depth, 2), np.percentile(depth, 98)
    depth_vis = np.clip((depth - dmin) / (dmax - dmin + 1e-8), 0, 1)

    plt.figure(figsize=(10, 4))
    plt.imshow(depth_vis, cmap="plasma")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_costmap_visualization(cost_map: np.ndarray, save_path: Path):
    vis = cost_map.copy()
    obstacle_region = (vis == 255)

    color = cv2.applyColorMap(
        np.uint8(255 - vis),
        cv2.COLORMAP_JET
    )
    color[obstacle_region] = (0, 0, 0)
    cv2.imwrite(str(save_path), color)


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
# YOLO 추론
# =========================
def run_yolo_inference(image_path: str):
    model = YOLO(YOLO_MODEL_PATH)

    results = model.predict(
        source=image_path,
        task="segment",
        save=False,
        verbose=True,
        conf=0.25,
        retina_masks=True,
    )

    r = results[0]
    orig_h, orig_w = r.orig_shape

    drive_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    obs_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

    if r.masks is None or r.boxes is None:
        print("[경고] YOLO segmentation 결과가 없습니다.")
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
# SC-Depth 추론
# =========================
def load_image_for_scdepth(img_path, resize=None):
    img = Image.open(img_path).convert("RGB")
    if resize is not None:
        img = img.resize(resize)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    return img, img_tensor


def run_scdepth_inference(image_path: str):
    hparams = get_opts()
    hparams.ckpt_path = SCDEPTH_CKPT_PATH

    model = SC_DepthV3.load_from_checkpoint(SCDEPTH_CKPT_PATH, hparams=hparams)
    model = model.to(DEVICE)
    model.eval()

    pil_img, img_tensor = load_image_for_scdepth(image_path, resize=SCDEPTH_INFER_SIZE)
    img_tensor = img_tensor.to(DEVICE)

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

    depth_map = pred.squeeze().detach().cpu().numpy()
    return pil_img, depth_map


# =========================
# Cost map 생성
# =========================
def build_cost_map(drive_mask: np.ndarray, obs_mask: np.ndarray, depth_map: np.ndarray):
    drive_mask = resize_mask(drive_mask, TARGET_SIZE)
    obs_mask = resize_mask(obs_mask, TARGET_SIZE)
    depth_map = resize_depth(depth_map, TARGET_SIZE)

    depth_norm = normalize_depth(depth_map)

    # 현재는 depth_norm이 클수록 "멀다"고 가정
    depth_penalty = (1.0 - depth_norm) * DEPTH_PENALTY_SCALE

    cost_map = np.where(
        drive_mask == 1,
        DRIVABLE_BASE_COST,
        NON_DRIVABLE_BASE_COST
    ).astype(np.float32)

    cost_map += depth_penalty
    cost_map[obs_mask == 1] = OBSTACLE_COST
    cost_map = np.clip(cost_map, 0, 255).astype(np.uint8)

    return cost_map, drive_mask, obs_mask, depth_norm


# =========================
# 메인
# =========================
def main():
    print(f"device: {DEVICE}")
    print(f"image: {IMAGE_PATH}")

    # 1. YOLO
    drive_mask, obs_mask = run_yolo_inference(IMAGE_PATH)

    # 2. SC-Depth
    scdepth_rgb, depth_map = run_scdepth_inference(IMAGE_PATH)

    # 3. Cost map
    cost_map, drive_small, obs_small, depth_norm = build_cost_map(
        drive_mask=drive_mask,
        obs_mask=obs_mask,
        depth_map=depth_map
    )

    # 4. 저장
    rgb_img = Image.open(IMAGE_PATH).convert("RGB")
    rgb_img.save(OUTPUT_DIR / "input_rgb.png")
    scdepth_rgb.save(OUTPUT_DIR / "scdepth_input_rgb.png")

    save_mask(drive_mask, OUTPUT_DIR / "drive_mask_raw.png")
    save_mask(obs_mask, OUTPUT_DIR / "obs_mask_raw.png")

    save_mask(drive_small, OUTPUT_DIR / "drive_mask_small.png")
    save_mask(obs_small, OUTPUT_DIR / "obs_mask_small.png")

    save_depth_vis(depth_map, OUTPUT_DIR / "depth_raw_vis.png")
    save_mask((depth_norm > 0).astype(np.uint8), OUTPUT_DIR / "depth_exists.png")
    cv2.imwrite(str(OUTPUT_DIR / "depth_norm.png"), np.uint8(depth_norm * 255))

    save_costmap_visualization(cost_map, OUTPUT_DIR / "cost_map.png")
    np.save(OUTPUT_DIR / "cost_map.npy", cost_map)

    print("\n저장 완료:")
    for p in [
        "input_rgb.png",
        "scdepth_input_rgb.png",
        "drive_mask_raw.png",
        "obs_mask_raw.png",
        "drive_mask_small.png",
        "obs_mask_small.png",
        "depth_raw_vis.png",
        "depth_norm.png",
        "cost_map.png",
        "cost_map.npy",
    ]:
        print(OUTPUT_DIR / p)


if __name__ == "__main__":
    main()