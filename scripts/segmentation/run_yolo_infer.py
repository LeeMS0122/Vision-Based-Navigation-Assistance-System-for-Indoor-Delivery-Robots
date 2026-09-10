from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

# =========================
# 경로 설정
# =========================
MODEL_PATH = "/home/minsu/disk_a/miniconda3/graduation_work_robot_vision/runs/v8n_seg_b16_e50/weights/best.pt"
IMAGE_PATH = "/home/minsu/disk_a/robot_dataset_scdepth_v3_official/training/Scene002/000000.png"
OUTPUT_DIR = Path("/home/minsu/disk_a/miniconda3/graduation_work_robot_vision/yolo_infer_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 클래스 ID
DRIVE_CLASS_ID = 0
OBS_CLASS_ID = 1


def save_mask(mask: np.ndarray, save_path: Path):
    img = (mask * 255).astype(np.uint8)
    cv2.imwrite(str(save_path), img)


def main():
    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=IMAGE_PATH,
        task="segment",
        save=False,
        verbose=True,
        conf=0.25,
        retina_masks=True,
    )

    r = results[0]

    # 원본 이미지 크기
    orig_h, orig_w = r.orig_shape

    drive_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    obs_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

    if r.masks is None or r.boxes is None:
        print("[경고] segmentation 결과가 없습니다.")
    else:
        masks = r.masks.data.cpu().numpy()   # [N, H, W]
        classes = r.boxes.cls.cpu().numpy().astype(int)

        for mask_i, cls_id in zip(masks, classes):
            binary_mask = (mask_i > 0.5).astype(np.uint8)

            if cls_id == DRIVE_CLASS_ID:
                drive_mask = np.maximum(drive_mask, binary_mask)

            elif cls_id == OBS_CLASS_ID:
                obs_mask = np.maximum(obs_mask, binary_mask)

    # 저장
    save_mask(drive_mask, OUTPUT_DIR / "drive_mask.png")
    save_mask(obs_mask, OUTPUT_DIR / "obs_mask.png")

    # 겹침 확인용 컬러 시각화
    vis = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    vis[drive_mask == 1] = (0, 255, 0)   # 초록
    vis[obs_mask == 1] = (0, 0, 255)     # 빨강
    cv2.imwrite(str(OUTPUT_DIR / "mask_overlay.png"), vis)

    # numpy 저장도 같이
    np.save(OUTPUT_DIR / "drive_mask.npy", drive_mask)
    np.save(OUTPUT_DIR / "obs_mask.npy", obs_mask)

    print("저장 완료:")
    print(OUTPUT_DIR / "drive_mask.png")
    print(OUTPUT_DIR / "obs_mask.png")
    print(OUTPUT_DIR / "mask_overlay.png")
    print(OUTPUT_DIR / "drive_mask.npy")
    print(OUTPUT_DIR / "obs_mask.npy")


if __name__ == "__main__":
    main()