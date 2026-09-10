from pathlib import Path
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from transformers import pipeline

# =========================
# 경로 설정
# =========================
DATA_ROOT = Path("/home/minsu/disk_a/robot_dataset_scdepth_v3_official/training")
TRAIN_LIST = DATA_ROOT / "train.txt"
VAL_LIST = DATA_ROOT / "val.txt"

# Hugging Face depth model
MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

# GPU 사용
DEVICE = 0 if torch.cuda.is_available() else -1

# 이미 만들어진 depth는 건너뛸지 여부
SKIP_EXISTING = True

# 특정 scene부터 재개하고 싶을 때 사용
START_SCENE_INDEX = 77


def load_scene_list(txt_path: Path):
    if not txt_path.exists():
        return []
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def to_uint16(depth_img: Image.Image) -> Image.Image:
    arr = np.array(depth_img).astype(np.float32)

    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        out = np.zeros_like(arr, dtype=np.uint16)
    else:
        out = ((arr - mn) / (mx - mn) * 65535.0).clip(0, 65535).astype(np.uint16)

    return Image.fromarray(out, mode="I;16")


def main():
    train_scenes = load_scene_list(TRAIN_LIST)
    val_scenes = load_scene_list(VAL_LIST)

    # 중복 제거 + 순서 유지
    all_scenes = []
    seen = set()
    for s in train_scenes + val_scenes:
        if s not in seen:
            seen.add(s)
            all_scenes.append(s)

    print(f"총 scene 수: {len(all_scenes)}")
    print(f"device: {'cuda' if DEVICE >= 0 else 'cpu'}")
    print(f"model: {MODEL_ID}")

    pipe = pipeline(
        task="depth-estimation",
        model=MODEL_ID,
        device=DEVICE,
    )

    for scene_idx, scene in enumerate(all_scenes):
        if scene_idx < START_SCENE_INDEX:
            continue

        scene_dir = DATA_ROOT / scene
        if not scene_dir.exists():
            print(f"[경고] scene 폴더 없음: {scene_dir}")
            continue

        out_dir = scene_dir / "leres_depth"
        out_dir.mkdir(parents=True, exist_ok=True)

        imgs = sorted(list(scene_dir.glob("*.png")) + list(scene_dir.glob("*.jpg")))

        print(f"\n[{scene_idx+1}/{len(all_scenes)}] {scene} | images={len(imgs)}")

        for img_path in tqdm(imgs, desc=scene):
            out_path = out_dir / f"{img_path.stem}.png"

            if SKIP_EXISTING and out_path.exists():
                continue

            try:
                image = Image.open(img_path).convert("RGB")
                pred = pipe(image)["depth"]  # PIL Image
                depth_u16 = to_uint16(pred)
                depth_u16.save(out_path)

            except Exception as e:
                print(f"\n[오류] {scene} / {img_path.name}: {e}")

    print("\n전체 leres_depth 생성 완료")


if __name__ == "__main__":
    main()