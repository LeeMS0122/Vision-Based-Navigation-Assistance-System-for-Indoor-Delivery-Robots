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
CKPT_PATH = "/home/minsu/disk_a/sc_depth_pl/ckpts/robot_scv3_full/version_0/epoch=16-val_loss=0.3801.ckpt"

# 대표 샘플 이미지들
IMAGE_PATHS = [
    "/home/minsu/disk_a/robot_dataset_scdepth_v3_official/training/Scene002/000244.png",
    "/home/minsu/disk_a/robot_dataset_scdepth_v3_official/training/Scene009/000086.png",
    "/home/minsu/disk_a/robot_dataset_scdepth_v3_official/training/Scene004/000003.png",
]

OUTPUT_DIR = Path("/home/minsu/disk_a/miniconda3/graduation_work_robot_vision/depth_eval_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INFER_SIZE = (640, 384)  # (width, height)


def load_image(img_path, resize=None):
    img = Image.open(img_path).convert("RGB")
    orig_img = img.copy()

    if resize is not None:
        img = img.resize(resize)

    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

    return orig_img, img, img_tensor


def normalize_depth(depth: np.ndarray):
    d = depth.astype(np.float32).copy()
    p2 = np.percentile(d, 2)
    p98 = np.percentile(d, 98)

    if p98 - p2 < 1e-8:
        return np.zeros_like(d, dtype=np.float32), p2, p98

    d = np.clip(d, p2, p98)
    d = (d - p2) / (p98 - p2)
    return d, p2, p98


def save_depth_vis(depth: np.ndarray, save_path: Path, cmap="plasma"):
    dmin, dmax = np.percentile(depth, 2), np.percentile(depth, 98)
    depth_vis = np.clip((depth - dmin) / (dmax - dmin + 1e-8), 0, 1)

    plt.figure(figsize=(10, 4))
    plt.imshow(depth_vis, cmap=cmap)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_gray(depth_norm: np.ndarray, save_path: Path):
    img = np.clip(depth_norm * 255, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(save_path)


def save_stats_text(depth: np.ndarray, depth_norm: np.ndarray, p2, p98, save_path: Path):
    text = []
    text.append(f"raw min: {depth.min():.6f}")
    text.append(f"raw max: {depth.max():.6f}")
    text.append(f"raw mean: {depth.mean():.6f}")
    text.append(f"raw std: {depth.std():.6f}")
    text.append(f"percentile 2: {p2:.6f}")
    text.append(f"percentile 98: {p98:.6f}")
    text.append(f"norm min: {depth_norm.min():.6f}")
    text.append(f"norm max: {depth_norm.max():.6f}")
    text.append(f"norm mean: {depth_norm.mean():.6f}")
    text.append(f"norm std: {depth_norm.std():.6f}")

    save_path.write_text("\n".join(text), encoding="utf-8")


def main():
    hparams = get_opts()
    hparams.ckpt_path = CKPT_PATH

    model = SC_DepthV3.load_from_checkpoint(CKPT_PATH, hparams=hparams)
    model = model.to(DEVICE)
    model.eval()

    print(f"device: {DEVICE}")
    print(f"ckpt: {CKPT_PATH}")

    with torch.no_grad():
        for idx, img_path_str in enumerate(IMAGE_PATHS):
            img_path = Path(img_path_str)
            if not img_path.exists():
                print(f"[경고] 파일 없음: {img_path}")
                continue

            orig_img, resized_img, img_tensor = load_image(img_path, resize=INFER_SIZE)
            img_tensor = img_tensor.to(DEVICE)

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

            depth = pred.squeeze().detach().cpu().numpy()
            depth_norm, p2, p98 = normalize_depth(depth)

            prefix = f"sample_{idx:02d}_{img_path.stem}"

            orig_img.save(OUTPUT_DIR / f"{prefix}_rgb_orig.png")
            resized_img.save(OUTPUT_DIR / f"{prefix}_rgb_resized.png")
            save_depth_vis(depth, OUTPUT_DIR / f"{prefix}_depth_raw_vis.png", cmap="plasma")
            save_gray(depth_norm, OUTPUT_DIR / f"{prefix}_depth_norm_gray.png")
            save_depth_vis(depth_norm, OUTPUT_DIR / f"{prefix}_depth_norm_vis.png", cmap="viridis")
            save_stats_text(depth, depth_norm, p2, p98, OUTPUT_DIR / f"{prefix}_stats.txt")

            print(f"[완료] {prefix}")

    print("\n모든 depth 샘플 저장 완료")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()