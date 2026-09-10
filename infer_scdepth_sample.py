from pathlib import Path
import sys
sys.path.append("/home/minsu/disk_a/sc_depth_pl")

import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt

from SC_DepthV3 import SC_DepthV3
from config import get_opts

# =========================
# 경로 설정
# =========================
CKPT_PATH = "/home/minsu/disk_a/sc_depth_pl/ckpts/robot_scv3_smoke/version_1/epoch=0-val_loss=0.3773.ckpt"
SCENE_DIR = Path("/home/minsu/disk_a/robot_dataset_scdepth_v3_official/training/Scene002")
OUTPUT_DIR = Path("/home/minsu/disk_a/miniconda3/graduation_work_robot_vision/scdepth_infer_outputs")

# 추론할 이미지 인덱스
TARGET_IMAGES = ["000000.png", "000100.png", "000200.png"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INFER_SIZE = (640, 384)  # (width, height), 32의 배수

def load_image(img_path, resize=None):
    img = Image.open(img_path).convert("RGB")
    if resize is not None:
        img = img.resize(resize)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    return img, img_tensor


def save_depth_vis(depth_tensor, save_path):
    depth = depth_tensor.squeeze().detach().cpu().numpy()

    # inverse depth처럼 보일 수 있으니 시각화용 정규화
    dmin, dmax = np.percentile(depth, 2), np.percentile(depth, 98)
    depth_vis = np.clip((depth - dmin) / (dmax - dmin + 1e-8), 0, 1)

    plt.figure(figsize=(10, 4))
    plt.imshow(depth_vis, cmap="plasma")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # hparams 로드
    hparams = get_opts()
    hparams.ckpt_path = CKPT_PATH

    model = SC_DepthV3.load_from_checkpoint(CKPT_PATH, hparams=hparams)
    model = model.to(DEVICE)
    model.eval()

    print(f"device: {DEVICE}")
    print(f"ckpt: {CKPT_PATH}")
    print(f"scene: {SCENE_DIR}")

    with torch.no_grad():
        for name in TARGET_IMAGES:
            img_path = SCENE_DIR / name
            if not img_path.exists():
                print(f"[경고] 파일 없음: {img_path}")
                continue

            pil_img, img_tensor = load_image(img_path, resize=INFER_SIZE)
            img_tensor = img_tensor.to(DEVICE)

            # 모델 forward
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

            rgb_save = OUTPUT_DIR / f"{img_path.stem}_rgb.png"
            depth_save = OUTPUT_DIR / f"{img_path.stem}_depth.png"

            pil_img.save(rgb_save)
            save_depth_vis(pred, depth_save)

            print(f"저장 완료: {rgb_save}")
            print(f"저장 완료: {depth_save}")

    print("\n완료")
    print(f"출력 폴더: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()