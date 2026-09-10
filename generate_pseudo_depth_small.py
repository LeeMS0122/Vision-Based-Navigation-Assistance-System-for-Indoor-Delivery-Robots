from pathlib import Path
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from transformers import pipeline

DATA_ROOT = Path("/home/minsu/disk_a/robot_dataset_scdepth_v3_official/training")
SCENE_LIST = DATA_ROOT / "train_small.txt"

# Transformers-compatible Depth Anything V2 checkpoint
MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

# 0이면 GPU 0번 사용, CPU면 -1
DEVICE = 0 if torch.cuda.is_available() else -1

def to_uint16(depth_img: Image.Image) -> Image.Image:
    arr = np.array(depth_img).astype(np.float32)

    # relative depth를 scene별 PNG로 저장하기 위해 0~65535 정규화
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        out = np.zeros_like(arr, dtype=np.uint16)
    else:
        out = ((arr - mn) / (mx - mn) * 65535.0).clip(0, 65535).astype(np.uint16)

    return Image.fromarray(out, mode="I;16")

def main():
    with open(SCENE_LIST, "r", encoding="utf-8") as f:
        scenes = [line.strip() for line in f if line.strip()]

    pipe = pipeline(
        task="depth-estimation",
        model=MODEL_ID,
        device=DEVICE,
    )

    print(f"num scenes: {len(scenes)}")
    print(f"device: {'cuda' if DEVICE >= 0 else 'cpu'}")

    for scene in scenes:
        scene_dir = DATA_ROOT / scene
        out_dir = scene_dir / "leres_depth"
        out_dir.mkdir(parents=True, exist_ok=True)

        imgs = sorted(list(scene_dir.glob("*.png")) + list(scene_dir.glob("*.jpg")))
        print(f"\n[{scene}] images: {len(imgs)}")

        for img_path in tqdm(imgs, desc=scene):
            out_path = out_dir / f"{img_path.stem}.png"
            if out_path.exists():
                continue

            image = Image.open(img_path).convert("RGB")
            pred = pipe(image)["depth"]   # PIL Image

            depth_u16 = to_uint16(pred)
            depth_u16.save(out_path)

    print("\n완료")

if __name__ == "__main__":
    main()